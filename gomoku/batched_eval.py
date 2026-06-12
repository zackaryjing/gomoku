from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gomoku.board import Board
from gomoku.encoding import encode_board
from gomoku.model import PolicyValueNet


STOP_REQUEST = "__gomoku_stop_batched_evaluator__"


@dataclass
class BatchedEvalStats:
    requests: int = 0
    batches: int = 0
    max_batch: int = 0
    total_batch: int = 0
    forward_sec: float = 0.0

    @property
    def avg_batch(self) -> float:
        return 0.0 if self.batches == 0 else self.total_batch / self.batches


class RemoteBatchedEvaluator:
    """Synchronous MCTS evaluator client backed by a shared batch server."""

    def __init__(
        self,
        request_queue: Queue,
        response_queue: Queue,
        worker_slot: int,
        response_timeout: float = 120.0,
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_slot = worker_slot
        self.response_timeout = response_timeout
        self.pending: dict[str, tuple[np.ndarray, float]] = {}
        self.counter = 0
        self.pid = os.getpid()

    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        request_id = f"{self.pid}:{self.counter}"
        self.counter += 1
        self.request_queue.put((self.worker_slot, request_id, encode_board(board)))

        while True:
            cached = self.pending.pop(request_id, None)
            if cached is not None:
                return cached

            response_id, policy, value = self.response_queue.get(timeout=self.response_timeout)
            if response_id == request_id:
                return policy, value
            self.pending[response_id] = (policy, value)


def run_batched_evaluator_server(
    request_queue: Queue,
    response_queues: list[Queue],
    checkpoint_path: str | Path,
    device: str,
    batch_size: int,
    timeout_ms: float,
    torch_threads: int = 1,
) -> None:
    """Own the model in one process and answer many workers with batched forwards."""

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model = PolicyValueNet()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    torch_device = torch.device(device)
    timeout_sec = max(0.0, timeout_ms / 1000.0)
    stats = BatchedEvalStats()
    print(
        "[gomoku] batched evaluator ready "
        f"pid={os.getpid()} device={torch_device} batch_size={batch_size} timeout_ms={timeout_ms}",
        flush=True,
    )

    while True:
        item = request_queue.get()
        if item == STOP_REQUEST:
            break

        batch: list[tuple[int, str, np.ndarray]] = [item]
        deadline = time.perf_counter() + timeout_sec
        while len(batch) < batch_size:
            wait = deadline - time.perf_counter()
            if wait <= 0:
                break
            try:
                next_item = request_queue.get(timeout=wait)
            except queue.Empty:
                break
            if next_item == STOP_REQUEST:
                request_queue.put(STOP_REQUEST)
                break
            batch.append(next_item)

        _serve_batch(model, torch_device, response_queues, batch, stats)

    print(
        "[gomoku] batched evaluator stopped "
        f"requests={stats.requests} batches={stats.batches} "
        f"avg_batch={stats.avg_batch:.2f} max_batch={stats.max_batch} "
        f"forward_sec={stats.forward_sec:.2f}",
        flush=True,
    )


@torch.no_grad()
def _serve_batch(
    model: PolicyValueNet,
    device: torch.device,
    response_queues: list[Queue],
    batch: list[tuple[int, str, Any]],
    stats: BatchedEvalStats,
) -> None:
    worker_slots = [worker_slot for worker_slot, _request_id, _state in batch]
    request_ids = [request_id for _worker_slot, request_id, _state in batch]
    states = np.stack([state for _worker_slot, _request_id, state in batch], axis=0)
    tensor = torch.from_numpy(states).to(device)
    start = time.perf_counter()
    logits, values = model(tensor)
    policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
    values_np = values.squeeze(1).cpu().numpy().astype(np.float64)
    stats.forward_sec += time.perf_counter() - start
    stats.requests += len(batch)
    stats.batches += 1
    stats.total_batch += len(batch)
    stats.max_batch = max(stats.max_batch, len(batch))

    for worker_slot, request_id, policy, value in zip(worker_slots, request_ids, policies, values_np):
        response_queues[worker_slot].put((request_id, policy, float(value)))

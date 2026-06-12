from __future__ import annotations

import numpy as np
import torch

from gomoku.board import Board
from gomoku.encoding import encode_board
from gomoku.model import PolicyValueNet


class NeuralEvaluator:
    """MCTS evaluator wrapper for trained policy/value checkpoints."""

    def __init__(self, model: PolicyValueNet, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        policies, values = self.evaluate_batch([board])
        return policies[0], values[0]

    @torch.no_grad()
    def evaluate_batch(self, boards: list[Board]) -> tuple[list[np.ndarray], list[float]]:
        state = torch.from_numpy(np.stack([encode_board(board) for board in boards], axis=0)).to(self.device)
        logits, value = self.model(state)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
        values = value.squeeze(1).cpu().numpy().astype(np.float64)
        return [policy for policy in policies], [float(item) for item in values]

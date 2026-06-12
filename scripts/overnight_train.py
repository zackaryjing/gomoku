from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm.auto import tqdm

from gomoku.arena import evaluate_against_baseline
from gomoku.mcts import MCTS
from gomoku.model import PolicyValueNet
from gomoku.neural_eval import NeuralEvaluator
from gomoku.self_play import play_self_play_game
from gomoku.training import TrainConfig, TrainState, load_checkpoint, save_checkpoint, train_model

WORKER_EVALUATOR = None
WORKER_SIMULATIONS = 40


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated self-play and resume training cycles.")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--games-per-cycle", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--self-play-evaluator", choices=["heuristic", "checkpoint-mcts"], default="heuristic")
    parser.add_argument("--self-play-workers", type=int, default=1)
    parser.add_argument("--self-play-device", default="cpu")
    parser.add_argument("--self-play-torch-threads", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gomoku_resnet_latest.pt"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gomoku_resnet_latest.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/overnight"))
    parser.add_argument("--log", type=Path, default=Path("runs/overnight_train.jsonl"))
    parser.add_argument("--arena-games", type=int, default=0)
    parser.add_argument("--arena-simulations", type=int, default=40)
    parser.add_argument("--target-win-rate", type=float, default=0.55)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[gomoku] stop requested by signal {signum}; finishing current step then saving.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    deadline = time.time() + args.hours * 3600
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    state = TrainState()
    if args.checkpoint.exists():
        state = load_checkpoint(args.checkpoint, model, optimizer, map_location=args.device)
        print(f"[gomoku] resumed {args.checkpoint} epoch={state.epoch} step={state.global_step}")
    else:
        print(f"[gomoku] checkpoint not found; starting fresh: {args.checkpoint}")

    cycle = 0
    print(
        "[gomoku] overnight loop "
        f"hours={args.hours} games_per_cycle={args.games_per_cycle} "
        f"simulations={args.simulations} self_play_evaluator={args.self_play_evaluator} "
        f"self_play_workers={args.self_play_workers} self_play_device={args.self_play_device} "
        f"train_epochs={args.train_epochs} device={args.device}"
    )

    while time.time() < deadline and not stop_requested:
        cycle += 1
        cycle_started = time.time()
        run_id = time.strftime("%Y%m%d_%H%M%S")
        data_path = args.data_dir / f"self_play_cycle_{cycle:04d}_{run_id}.npz"

        print(f"[gomoku] cycle={cycle} self-play start -> {data_path}")
        self_play_checkpoint = args.out if args.out.exists() else args.checkpoint
        states, policies, values, moves = generate_cycle_data(
            games=args.games_per_cycle,
            simulations=args.simulations,
            evaluator_mode=args.self_play_evaluator,
            checkpoint_path=self_play_checkpoint,
            self_play_device=args.self_play_device,
            workers=args.self_play_workers,
            torch_threads=args.self_play_torch_threads,
            stop_check=lambda: stop_requested or time.time() >= deadline,
        )
        if len(values) == 0:
            print("[gomoku] no new games generated; stopping.")
            break

        np.savez_compressed(data_path, states=states, policies=policies, values=values)
        print(
            f"[gomoku] cycle={cycle} self-play done games={len(moves)} "
            f"samples={len(values)} avg_moves={float(np.mean(moves)):.2f}"
        )

        if stop_requested:
            break

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(states).float(),
            torch.from_numpy(policies).float(),
            torch.from_numpy(values).float().unsqueeze(1),
        )
        config = TrainConfig(
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            save_every=1,
        )
        state = train_model(
            model,
            dataset,
            config,
            device=args.device,
            optimizer=optimizer,
            state=state,
            checkpoint_path=args.out,
        )
        save_checkpoint(args.out, model, optimizer, state, config)

        arena_result = None
        if args.arena_games > 0:
            arena_result = run_arena(
                model=model,
                device=args.device,
                games=args.arena_games,
                ai_simulations=args.arena_simulations,
                baseline_simulations=args.arena_simulations,
            )
            print(f"[gomoku] cycle={cycle} arena {arena_result}")

        event = {
            "cycle": cycle,
            "data": str(data_path),
            "games": len(moves),
            "samples": int(len(values)),
            "avg_moves": float(np.mean(moves)),
            "epoch": state.epoch,
            "global_step": state.global_step,
            "metrics": state.metrics or {},
            "arena": arena_result,
            "duration_sec": time.time() - cycle_started,
        }
        append_jsonl(args.log, event)
        print(f"[gomoku] cycle={cycle} done {event}")
        if arena_result and arena_result["ai_win_rate"] >= args.target_win_rate:
            print(
                f"[gomoku] target reached: ai_win_rate={arena_result['ai_win_rate']:.3f} "
                f">= {args.target_win_rate:.3f}"
            )
            break

    save_checkpoint(args.out, model, optimizer, state, TrainConfig(lr=args.lr))
    print(f"[gomoku] overnight loop finished cycle={cycle} checkpoint={args.out}")


def generate_cycle_data(
    games: int,
    simulations: int,
    evaluator_mode: str,
    checkpoint_path: Path,
    self_play_device: str,
    workers: int,
    torch_threads: int,
    stop_check,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if workers > 1:
        return generate_cycle_data_parallel(
            games=games,
            simulations=simulations,
            evaluator_mode=evaluator_mode,
            checkpoint_path=checkpoint_path,
            self_play_device=self_play_device,
            workers=workers,
            torch_threads=torch_threads,
            stop_check=stop_check,
        )

    evaluator = build_self_play_evaluator(checkpoint_path, self_play_device, evaluator_mode)
    all_states = []
    all_policies = []
    all_values = []
    moves = []
    for _ in tqdm(range(games), desc="cycle self-play", unit="game"):
        if stop_check():
            break
        game = play_self_play_game(simulations=simulations, evaluator=evaluator)
        all_states.append(game.states)
        all_policies.append(game.policies)
        all_values.append(game.values)
        moves.append(int(len(game.values)))

    if not all_values:
        return (
            np.empty((0, 3, 15, 15), dtype=np.float32),
            np.empty((0, 225), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            moves,
        )

    return (
        np.concatenate(all_states, axis=0).astype(np.float32),
        np.concatenate(all_policies, axis=0).astype(np.float32),
        np.concatenate(all_values, axis=0).astype(np.float32),
        moves,
    )


def generate_cycle_data_parallel(
    games: int,
    simulations: int,
    evaluator_mode: str,
    checkpoint_path: Path,
    self_play_device: str,
    workers: int,
    torch_threads: int,
    stop_check,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    all_states = []
    all_policies = []
    all_values = []
    moves = []
    max_workers = max(1, workers)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_self_play_worker,
        initargs=(evaluator_mode, str(checkpoint_path), self_play_device, simulations, torch_threads),
    ) as executor:
        futures = []
        for game_idx in range(games):
            if stop_check():
                break
            futures.append(executor.submit(play_one_self_play_worker, game_idx))

        for future in tqdm(as_completed(futures), total=len(futures), desc="cycle self-play", unit="game"):
            game = future.result()
            all_states.append(game.states)
            all_policies.append(game.policies)
            all_values.append(game.values)
            moves.append(int(len(game.values)))

    if not all_values:
        return (
            np.empty((0, 3, 15, 15), dtype=np.float32),
            np.empty((0, 225), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            moves,
        )

    return (
        np.concatenate(all_states, axis=0).astype(np.float32),
        np.concatenate(all_policies, axis=0).astype(np.float32),
        np.concatenate(all_values, axis=0).astype(np.float32),
        moves,
    )


def init_self_play_worker(
    evaluator_mode: str,
    checkpoint_path: str,
    self_play_device: str,
    simulations: int,
    torch_threads: int,
) -> None:
    global WORKER_EVALUATOR, WORKER_SIMULATIONS
    WORKER_SIMULATIONS = simulations
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    WORKER_EVALUATOR = build_self_play_evaluator(Path(checkpoint_path), self_play_device, evaluator_mode)


def play_one_self_play_worker(game_idx: int):
    seed = (os.getpid() * 1_000_003 + game_idx + int(time.time() * 1000)) % (2**32 - 1)
    np.random.seed(seed)
    return play_self_play_game(simulations=WORKER_SIMULATIONS, evaluator=WORKER_EVALUATOR)


def build_self_play_evaluator(checkpoint_path: Path, device: str, mode: str):
    if mode == "checkpoint-mcts":
        model = PolicyValueNet()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        return NeuralEvaluator(model, device=device)
    return None


def run_arena(
    model: PolicyValueNet,
    device: str,
    games: int,
    ai_simulations: int,
    baseline_simulations: int,
) -> dict[str, float | int]:
    ai = MCTS(evaluator=NeuralEvaluator(model, device=device), simulations=ai_simulations)
    baseline = MCTS(simulations=baseline_simulations)
    return evaluate_against_baseline(
        ai=ai,
        baseline=baseline,
        games=games,
        show_progress=True,
    ).to_dict()


def append_jsonl(path: Path, payload: dict) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

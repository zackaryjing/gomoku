from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gomoku.arena import evaluate_against_baseline
from gomoku.mcts import MCTS
from gomoku.model import PolicyValueNet
from gomoku.neural_eval import NeuralEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint MCTS against heuristic MCTS.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gomoku_resnet_latest.pt"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--ai-simulations", type=int, default=40)
    parser.add_argument("--baseline-simulations", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = PolicyValueNet()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    ai = MCTS(evaluator=NeuralEvaluator(model, device=args.device), simulations=args.ai_simulations)
    baseline = MCTS(simulations=args.baseline_simulations)

    result = evaluate_against_baseline(ai, baseline, games=args.games)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

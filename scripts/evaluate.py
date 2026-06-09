from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gomoku.board import Board
from gomoku.mcts import MCTS
from gomoku.model import PolicyValueNet
from gomoku.neural_eval import NeuralEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint:
        model = PolicyValueNet()
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        evaluator = NeuralEvaluator(model, device=args.device)
        mcts = MCTS(evaluator=evaluator, simulations=args.simulations)
    else:
        mcts = MCTS(simulations=args.simulations)

    board = Board()
    action = mcts.select_action(board, temperature=0.0)
    print({"best_action": action, "coord": board.action_to_coord(action)})


if __name__ == "__main__":
    main()

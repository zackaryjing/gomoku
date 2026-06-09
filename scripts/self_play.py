from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gomoku.self_play import play_self_play_game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/self_play_demo.npz"))
    args = parser.parse_args()

    all_states = []
    all_policies = []
    all_values = []

    for game_idx in range(args.games):
        game = play_self_play_game(simulations=args.simulations)
        all_states.append(game.states)
        all_policies.append(game.policies)
        all_values.append(game.values)
        print(f"game={game_idx + 1} moves={len(game.values)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=np.concatenate(all_states, axis=0),
        policies=np.concatenate(all_policies, axis=0),
        values=np.concatenate(all_values, axis=0),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

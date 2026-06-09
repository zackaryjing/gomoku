from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gomoku.model import PolicyValueNet
from gomoku.training import TrainConfig, load_npz_dataset, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gomoku_resnet.pt"))
    args = parser.parse_args()

    dataset = load_npz_dataset(args.data)
    model = PolicyValueNet()
    metrics = train_model(
        model,
        dataset,
        TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
        ),
        device=args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": metrics}, args.out)
    print(metrics)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

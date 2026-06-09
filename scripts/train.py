from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gomoku.model import PolicyValueNet
from gomoku.training import TrainConfig, load_checkpoint, load_npz_dataset, save_checkpoint, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gomoku_resnet.pt"))
    parser.add_argument("--resume", type=Path, help="Resume model, optimizer, epoch, and step state.")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    dataset = load_npz_dataset(args.data)
    model = PolicyValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    state = None
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, map_location=args.device)
        print(f"resumed {args.resume} at epoch={state.epoch} step={state.global_step}")

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )
    metrics = train_model(
        model,
        dataset,
        config,
        device=args.device,
        optimizer=optimizer,
        state=state,
        checkpoint_path=args.out,
    )

    save_checkpoint(args.out, model, optimizer, metrics, config)
    print(metrics.metrics)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

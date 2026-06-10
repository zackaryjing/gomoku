# Gomoku AI

Free-style 15x15 Gomoku with a pygame UI and an AlphaZero-style AI training
foundation. The project is intentionally small and readable for learning.

## Environment

Use the existing `dmcad` environment:

```bash
/root/miniconda3/envs/dmcad/bin/python -m pytest
/root/miniconda3/envs/dmcad/bin/python apps/play_pygame.py
```

## Features

- 15x15 free-style Gomoku, black moves first.
- No forbidden moves; five or more connected stones wins.
- Pygame human-vs-human and human-vs-AI modes.
- Heuristic MCTS placeholder AI before neural checkpoints exist.
- Policy/value ResNet and self-play/training entrypoints for future scaling.

## Controls

- Left click: place a stone.
- `A`: toggle human-vs-AI.
- `S`: switch human side and restart. Human is black by default.
- `U`: undo.
- `R`: restart.
- `Esc` or window close: quit.

## Training Skeleton

The first training code is a runnable foundation, not a tuned final trainer.

```bash
/root/miniconda3/envs/dmcad/bin/python scripts/self_play.py --games 2 --out data/self_play_demo.npz
/root/miniconda3/envs/dmcad/bin/python scripts/train.py --data data/self_play_demo.npz --epochs 1
```

`scripts/train.py` defaults to `--num-workers 0` so it also works in restricted
shells. On the training server, increase it after confirming the local process
limits.

Training checkpoints include model weights, optimizer state, completed epoch,
global step, metrics, and the training config used by that run:

```bash
/root/miniconda3/envs/dmcad/bin/python scripts/train.py \
  --data data/self_play_demo.npz \
  --resume checkpoints/gomoku_resnet.pt \
  --out checkpoints/gomoku_resnet.pt \
  --epochs 5
```

To pull trained parameters from the SSH host named `school` into a local clone:

```bash
./scripts/sync_params_from_school.sh
```

Override `REMOTE_ROOT` if the server-side project path changes.

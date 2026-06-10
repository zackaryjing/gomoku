# Gomoku AI

Free-style 15x15 Gomoku with a pygame UI and an AlphaZero-style AI training
foundation. The project is intentionally small and readable for learning.

## Environment

Use your local Python environment with `numpy`, `pygame`, `torch`, and `tqdm`
installed:

```bash
python -m pytest
python apps/play_pygame.py
```

The pygame app loads the checkpoint at startup when it exists. Before the first
move, press `N` to cycle through checkpoint policy, heuristic MCTS, and
checkpoint MCTS; after the first move the selection is locked until restart. UI
checkpoint inference uses CPU by default to avoid slowdowns on busy GPUs.

```bash
python apps/play_pygame.py
```

To force a different initial selection or device:

```bash
python apps/play_pygame.py --initial-ai heuristic-mcts
python apps/play_pygame.py --initial-ai checkpoint-policy --device cpu
python apps/play_pygame.py --initial-ai checkpoint-mcts --mcts-simulations 20
python apps/play_pygame.py --initial-ai heuristic-mcts --mcts-simulations 40
```

The default `checkpoint-policy` mode uses the neural network directly and does
not run MCTS search. The current checkpoint is based on a very small first run,
so it can look weak. `heuristic-mcts` does not use the checkpoint; it searches
with a hand-written tactical evaluator. `checkpoint-mcts` uses both the
checkpoint and MCTS: the network evaluates positions and suggests priors, while
MCTS searches candidate continuations. `--mcts-simulations` controls how many
MCTS simulations are run per AI move for both MCTS modes. Larger values usually
play better but respond more slowly.

For terminal diagnostics while playing:

```bash
python apps/play_pygame.py --debug
```

Debug mode prints click handling, ASCII board states, AI timing, backend choice,
CUDA visibility, and PyTorch CPU thread settings.

The UI tracks AI wins, losses, and draws for the current program session. It
writes JSONL stats on game end and writes a final summary on window close, Esc,
or Ctrl+C:

```bash
python apps/play_pygame.py --stats-path runs/ui_stats.jsonl
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
- `N`: cycle AI mode before the first move.
- `S`: switch human side and restart. Human is black by default.
- `U`: undo.
- `R`: restart.
- `Esc` or window close: quit.

## Training Skeleton

The first training code is a runnable foundation, not a tuned final trainer.

```bash
python scripts/self_play.py --games 2 --out data/self_play_demo.npz
python scripts/train.py --data data/self_play_demo.npz --epochs 1
```

`scripts/train.py` defaults to `--num-workers 0` so it also works in restricted
shells. On the training server, increase it after confirming the local process
limits.

Training checkpoints include model weights, optimizer state, completed epoch,
global step, metrics, and the training config used by that run:

```bash
python scripts/train.py \
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

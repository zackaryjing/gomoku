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
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy.astype(np.float64), float(value.item())

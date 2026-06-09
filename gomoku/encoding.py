from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, Board


def encode_board(board: Board) -> np.ndarray:
    """Encode board state from the current player's perspective.

    Plane 0 stores current-player stones, plane 1 stores opponent stones, and
    plane 2 stores whose turn it is. This keeps model inputs stable for self-play.
    """

    player = board.current_player
    current = (board.grid == player).astype(np.float32)
    opponent = (board.grid == -player).astype(np.float32)
    turn = np.full_like(current, 1.0 if player == BLACK else 0.0, dtype=np.float32)
    return np.stack([current, opponent, turn], axis=0)


def mask_illegal_logits(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    masked = np.array(logits, dtype=np.float64, copy=True)
    masked[~legal_mask] = -np.inf
    return masked

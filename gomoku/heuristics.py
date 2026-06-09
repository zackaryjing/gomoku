from __future__ import annotations

import numpy as np

from gomoku.board import Board

PATTERN_SCORES = {
    5: 1_000_000,
    4: 20_000,
    3: 1_000,
    2: 100,
}


def heuristic_policy(board: Board) -> np.ndarray:
    """Return a legal prior distribution based on local tactical patterns."""

    actions = board.legal_actions()
    priors = np.zeros(board.size * board.size, dtype=np.float64)
    if not actions:
        return priors

    center = (board.size - 1) / 2.0
    for action in actions:
        row, col = board.action_to_coord(action)
        own = move_score(board, row, col, board.current_player)
        block = move_score(board, row, col, -board.current_player) * 0.9
        center_bonus = max(0.0, 8.0 - abs(row - center) - abs(col - center))
        priors[action] = 1.0 + own + block + center_bonus

    total = priors.sum()
    if total <= 0:
        priors[actions] = 1.0 / len(actions)
    else:
        priors /= total
    return priors


def evaluate_board(board: Board, player: int) -> float:
    """Estimate value for player using pattern balance."""

    if board.is_over:
        if board.winner == player:
            return 1.0
        if board.winner == -player:
            return -1.0
        return 0.0

    own = _aggregate_patterns(board, player)
    opp = _aggregate_patterns(board, -player)
    return float(np.tanh((own - opp) / 30_000.0))


def move_score(board: Board, row: int, col: int, player: int) -> float:
    score = 0.0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        length = 1
        open_ends = 0
        count, open_end = _count_line(board, row, col, dr, dc, player)
        length += count
        open_ends += int(open_end)
        count, open_end = _count_line(board, row, col, -dr, -dc, player)
        length += count
        open_ends += int(open_end)

        if length >= 5:
            score += PATTERN_SCORES[5]
        elif length in PATTERN_SCORES:
            score += PATTERN_SCORES[length] * max(1, open_ends)
    return score


def _count_line(
    board: Board, row: int, col: int, dr: int, dc: int, player: int
) -> tuple[int, bool]:
    count = 0
    row += dr
    col += dc
    while board.in_bounds(row, col) and board.grid[row, col] == player:
        count += 1
        row += dr
        col += dc
    return count, board.in_bounds(row, col) and board.grid[row, col] == 0


def _aggregate_patterns(board: Board, player: int) -> float:
    score = 0.0
    rows, cols = np.where(board.grid == 0)
    for row, col in zip(rows, cols):
        score += move_score(board, int(row), int(col), player)
    return score

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1


class GameResult(Enum):
    ONGOING = "ongoing"
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"


@dataclass(frozen=True)
class Move:
    row: int
    col: int
    player: int


class Board:
    """Mutable 15x15 free-style Gomoku board.

    The board stores black stones as 1 and white stones as -1. Public methods use
    row/column coordinates so the same core rules can drive pygame, MCTS, and tests.
    """

    def __init__(self, size: int = 15):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.result = GameResult.ONGOING
        self.winner = EMPTY
        self.history: list[Move] = []

    def copy(self) -> Board:
        board = Board(self.size)
        board.grid = self.grid.copy()
        board.current_player = self.current_player
        board.result = self.result
        board.winner = self.winner
        board.history = list(self.history)
        return board

    @property
    def is_over(self) -> bool:
        return self.result is not GameResult.ONGOING

    def action_to_coord(self, action: int) -> tuple[int, int]:
        return divmod(action, self.size)

    def coord_to_action(self, row: int, col: int) -> int:
        return row * self.size + col

    def legal_actions(self) -> list[int]:
        if self.is_over:
            return []
        rows, cols = np.where(self.grid == EMPTY)
        return [self.coord_to_action(int(row), int(col)) for row, col in zip(rows, cols)]

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.size * self.size, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def play_action(self, action: int) -> None:
        row, col = self.action_to_coord(action)
        self.play(row, col)

    def play(self, row: int, col: int) -> None:
        if self.is_over:
            raise ValueError("Cannot play after the game is over.")
        if not self.in_bounds(row, col):
            raise ValueError(f"Move out of bounds: {(row, col)}")
        if self.grid[row, col] != EMPTY:
            raise ValueError(f"Cell is already occupied: {(row, col)}")

        player = self.current_player
        self.grid[row, col] = player
        self.history.append(Move(row, col, player))

        if self._has_five_or_more(row, col, player):
            self.winner = player
            self.result = GameResult.BLACK_WIN if player == BLACK else GameResult.WHITE_WIN
            return

        if not np.any(self.grid == EMPTY):
            self.result = GameResult.DRAW
            return

        self.current_player = -self.current_player

    def undo(self) -> bool:
        if not self.history:
            return False
        move = self.history.pop()
        self.grid[move.row, move.col] = EMPTY
        self.current_player = move.player
        self.winner = EMPTY
        self.result = GameResult.ONGOING
        return True

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def _has_five_or_more(self, row: int, col: int, player: int) -> bool:
        # Check only four axes through the latest move; all other lines are unchanged.
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= 5:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        row += dr
        col += dc
        while self.in_bounds(row, col) and self.grid[row, col] == player:
            count += 1
            row += dr
            col += dc
        return count


def result_value_for_player(result: GameResult, player: int) -> float:
    if result is GameResult.DRAW:
        return 0.0
    if result is GameResult.BLACK_WIN:
        return 1.0 if player == BLACK else -1.0
    if result is GameResult.WHITE_WIN:
        return 1.0 if player == WHITE else -1.0
    raise ValueError("Ongoing games do not have a final value.")

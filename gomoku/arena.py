from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tqdm.auto import tqdm

from gomoku.board import BLACK, WHITE, Board


class Agent(Protocol):
    def select_action(self, board: Board, temperature: float = 0.0) -> int:
        """Select an action for board.current_player."""


@dataclass
class ArenaResult:
    games: int
    ai_wins: int
    baseline_wins: int
    draws: int
    ai_black_games: int
    ai_white_games: int
    total_moves: int

    @property
    def ai_win_rate(self) -> float:
        return 0.0 if self.games == 0 else self.ai_wins / self.games

    @property
    def non_loss_rate(self) -> float:
        return 0.0 if self.games == 0 else (self.ai_wins + self.draws) / self.games

    @property
    def avg_moves(self) -> float:
        return 0.0 if self.games == 0 else self.total_moves / self.games

    def to_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "ai_wins": self.ai_wins,
            "baseline_wins": self.baseline_wins,
            "draws": self.draws,
            "ai_black_games": self.ai_black_games,
            "ai_white_games": self.ai_white_games,
            "total_moves": self.total_moves,
            "ai_win_rate": self.ai_win_rate,
            "non_loss_rate": self.non_loss_rate,
            "avg_moves": self.avg_moves,
        }


def play_game(black: Agent, white: Agent, board_size: int = 15) -> tuple[int, int]:
    board = Board(size=board_size)
    while not board.is_over:
        agent = black if board.current_player == BLACK else white
        action = agent.select_action(board, temperature=0.0)
        board.play_action(action)
    return board.winner, len(board.history)


def evaluate_against_baseline(
    ai: Agent,
    baseline: Agent,
    games: int,
    board_size: int = 15,
    show_progress: bool = True,
) -> ArenaResult:
    ai_wins = 0
    baseline_wins = 0
    draws = 0
    total_moves = 0
    ai_black_games = 0
    ai_white_games = 0

    iterator = range(games)
    if show_progress:
        iterator = tqdm(iterator, desc="arena", unit="game")

    for game_idx in iterator:
        ai_is_black = game_idx % 2 == 0
        if ai_is_black:
            winner, moves = play_game(ai, baseline, board_size=board_size)
            ai_black_games += 1
            ai_player = BLACK
        else:
            winner, moves = play_game(baseline, ai, board_size=board_size)
            ai_white_games += 1
            ai_player = WHITE

        total_moves += moves
        if winner == ai_player:
            ai_wins += 1
        elif winner == 0:
            draws += 1
        else:
            baseline_wins += 1

    return ArenaResult(
        games=games,
        ai_wins=ai_wins,
        baseline_wins=baseline_wins,
        draws=draws,
        ai_black_games=ai_black_games,
        ai_white_games=ai_white_games,
        total_moves=total_moves,
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku.board import Board, result_value_for_player
from gomoku.encoding import encode_board
from gomoku.mcts import Evaluator, MCTS


@dataclass
class SelfPlayGame:
    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray


def play_self_play_game(
    board_size: int = 15,
    simulations: int = 80,
    temperature_moves: int = 20,
    evaluator: Evaluator | None = None,
) -> SelfPlayGame:
    board = Board(size=board_size)
    mcts = MCTS(evaluator=evaluator, simulations=simulations)
    examples: list[tuple[np.ndarray, np.ndarray, int]] = []

    while not board.is_over:
        temperature = 1.0 if len(board.history) < temperature_moves else 0.1
        policy = mcts.run(board, temperature=temperature)
        examples.append((encode_board(board), policy, board.current_player))
        action = int(np.random.choice(np.arange(board.size * board.size), p=policy))
        board.play_action(action)

    states = np.stack([item[0] for item in examples]).astype(np.float32)
    policies = np.stack([item[1] for item in examples]).astype(np.float32)
    values = np.array(
        [result_value_for_player(board.result, player) for _, _, player in examples],
        dtype=np.float32,
    )
    return SelfPlayGame(states=states, policies=policies, values=values)

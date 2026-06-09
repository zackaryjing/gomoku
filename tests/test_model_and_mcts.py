import numpy as np
import torch

from gomoku.board import Board
from gomoku.encoding import encode_board
from gomoku.mcts import MCTS
from gomoku.model import PolicyValueNet


def test_encoding_shape():
    board = Board()
    encoded = encode_board(board)
    assert encoded.shape == (3, 15, 15)
    assert encoded.dtype == np.float32


def test_model_output_shapes():
    model = PolicyValueNet(channels=16, blocks=1)
    policy, value = model(torch.zeros(2, 3, 15, 15))
    assert policy.shape == (2, 225)
    assert value.shape == (2, 1)


def test_mcts_only_legal_actions_and_non_mutating():
    board = Board(size=3)
    for action in range(8):
        board.play_action(action)
    before = board.grid.copy()
    policy = MCTS(simulations=4).run(board)
    assert np.array_equal(board.grid, before)
    assert policy[8] == 1.0


def test_mcts_takes_immediate_win():
    board = Board()
    for col in range(4):
        board.play(7, col)
        board.play(8, col)
    action = MCTS(simulations=20).select_action(board, temperature=0.0)
    assert board.action_to_coord(action) == (7, 4)

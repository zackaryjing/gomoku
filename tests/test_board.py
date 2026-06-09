import pytest

from gomoku.board import BLACK, WHITE, Board, GameResult


def test_horizontal_five_wins():
    board = Board()
    for col in range(4):
        board.play(7, col)
        board.play(8, col)
    board.play(7, 4)
    assert board.result is GameResult.BLACK_WIN
    assert board.winner == BLACK


def test_vertical_long_line_wins():
    board = Board()
    moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)]
    for row, col in moves:
        board.play(row, col)
    assert board.result is GameResult.BLACK_WIN


def test_diagonal_win():
    board = Board()
    moves = [
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (2, 2),
        (0, 3),
        (3, 3),
        (0, 4),
        (4, 4),
    ]
    for row, col in moves:
        board.play(row, col)
    assert board.result is GameResult.BLACK_WIN


def test_repeated_move_is_illegal():
    board = Board()
    board.play(7, 7)
    with pytest.raises(ValueError):
        board.play(7, 7)


def test_undo_restores_turn_and_cell():
    board = Board()
    board.play(7, 7)
    board.play(7, 8)
    assert board.undo()
    assert board.grid[7, 8] == 0
    assert board.current_player == WHITE

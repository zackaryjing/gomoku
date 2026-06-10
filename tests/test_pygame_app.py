import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from apps.play_pygame import GomokuApp
from gomoku.board import BLACK, Board


class ImmediateAI:
    def __init__(self):
        self.calls = 0

    def select_action(self, board: Board, temperature: float = 0.0) -> int:
        self.calls += 1
        return board.legal_actions()[0]


def test_fast_ai_result_is_not_overwritten():
    app = GomokuApp(initial_ai="heuristic-mcts", device="cpu")
    ai = ImmediateAI()
    app.ai_backends = {"test": (ai, "test ai")}
    app.selected_ai_key = "test"

    app.board.play(7, 7)
    app._start_ai_if_needed()
    if app.ai_thread:
        app.ai_thread.join(timeout=1)

    # This used to start another thread before applying the pending result.
    app._start_ai_if_needed()
    app._apply_ai_result()

    assert ai.calls == 1
    assert len(app.board.history) == 2


def test_stats_written_on_finished_game(tmp_path):
    stats_path = tmp_path / "ui_stats.jsonl"
    app = GomokuApp(initial_ai="heuristic-mcts", device="cpu", stats_path=stats_path)
    app.selected_ai_key = "heuristic-mcts"
    app.ai_player = BLACK

    for col in range(4):
        app.board.play(7, col)
        app.board.play(8, col)
    app.board.play(7, 4)

    app._record_finished_game_if_needed()
    app.write_stats_summary()

    text = stats_path.read_text(encoding="utf-8")
    assert '"event": "game"' in text
    assert '"event": "summary"' in text
    assert '"ai_win": 1' in text

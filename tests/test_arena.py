from gomoku.arena import evaluate_against_baseline
from gomoku.mcts import MCTS


def test_arena_runs_balanced_colors():
    result = evaluate_against_baseline(
        ai=MCTS(simulations=1),
        baseline=MCTS(simulations=1),
        games=2,
        show_progress=False,
    )

    assert result.games == 2
    assert result.ai_black_games == 1
    assert result.ai_white_games == 1
    assert result.ai_wins + result.baseline_wins + result.draws == 2

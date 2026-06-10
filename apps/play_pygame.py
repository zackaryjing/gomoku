from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pygame
import torch

from gomoku.board import BLACK, WHITE, Board, GameResult
from gomoku.encoding import encode_board
from gomoku.mcts import MCTS
from gomoku.model import PolicyValueNet
from gomoku.neural_eval import NeuralEvaluator

CELL = 42
MARGIN = 48
BOARD_PIXELS = CELL * 14
SIDE_PANEL = 250
WIDTH = MARGIN * 2 + BOARD_PIXELS + SIDE_PANEL
HEIGHT = MARGIN * 2 + BOARD_PIXELS
STONE_RADIUS = 17

BG = (235, 190, 120)
GRID = (55, 40, 25)
PANEL = (32, 36, 40)
TEXT = (236, 238, 240)
MUTED = (170, 176, 184)
ACCENT = (84, 160, 255)
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "gomoku_resnet_latest.pt"


class PolicyCheckpointAI:
    """Fast UI opponent that plays the highest-logit legal move from a checkpoint."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(device)
        self.model = PolicyValueNet().to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def select_action(self, board: Board, temperature: float = 0.0) -> int:
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, _ = self.model(state)
        scores = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
        scores[~board.legal_mask()] = -np.inf
        return int(np.argmax(scores))


class GomokuApp:
    """Small pygame shell around the reusable board and MCTS logic."""

    def __init__(
        self,
        ai_backend: str = "auto",
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Gomoku AI")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 17)
        self.board = Board(size=15)
        self.ai_enabled = True
        self.ai_player = WHITE
        self.ai, self.ai_name = self._build_ai(ai_backend, checkpoint, device)
        self.ai_thread: threading.Thread | None = None
        self.ai_result: tuple[int, int, int] | None = None
        self.ai_error: str | None = None
        self.ai_generation = 0

    def run(self) -> None:
        while True:
            self._handle_events()
            self._start_ai_if_needed()
            self._apply_ai_result()
            self._draw()
            self.clock.tick(60)

    def _build_ai(self, backend: str, checkpoint: Path, device: str):
        if backend == "auto":
            backend = "checkpoint-policy" if checkpoint.exists() else "heuristic-mcts"
        if backend == "checkpoint-policy":
            return PolicyCheckpointAI(checkpoint, device), f"checkpoint policy ({device})"
        if backend == "checkpoint-mcts":
            model = PolicyValueNet()
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state["model"])
            return MCTS(evaluator=NeuralEvaluator(model, device=device), simulations=80), f"checkpoint MCTS ({device})"
        return MCTS(simulations=80), "heuristic MCTS"

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                if event.key == pygame.K_r:
                    self._restart()
                if event.key == pygame.K_u:
                    self._undo()
                if event.key == pygame.K_a:
                    self.ai_enabled = not self.ai_enabled
                if event.key == pygame.K_s:
                    self.ai_player = BLACK if self.ai_player == WHITE else WHITE
                    self._restart()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.board.is_over:
            return
        if self.ai_enabled and self.board.current_player == self.ai_player:
            return

        coord = self._pixel_to_coord(*pos)
        if coord is None:
            return
        row, col = coord
        try:
            self.board.play(row, col)
            self._clear_pending_ai()
        except ValueError:
            return

    def _start_ai_if_needed(self) -> None:
        if not self.ai_enabled or self.board.is_over or self.board.current_player != self.ai_player:
            return
        if self.ai_thread and self.ai_thread.is_alive():
            return

        board_snapshot = self.board.copy()
        expected_history_len = len(self.board.history)
        generation = self.ai_generation
        self.ai_result = None
        self.ai_error = None

        # Search runs off the pygame thread, so the human stone is drawn immediately.
        self.ai_thread = threading.Thread(
            target=self._compute_ai_move,
            args=(board_snapshot, expected_history_len, generation),
            daemon=True,
        )
        self.ai_thread.start()

    def _compute_ai_move(self, board: Board, expected_history_len: int, generation: int) -> None:
        try:
            action = self.ai.select_action(board, temperature=0.0)
            self.ai_result = (generation, expected_history_len, action)
        except Exception as exc:  # pragma: no cover - surfaced in the UI.
            self.ai_error = str(exc)

    def _apply_ai_result(self) -> None:
        if self.ai_result is None:
            return
        generation, expected_history_len, action = self.ai_result
        self.ai_result = None
        if generation != self.ai_generation:
            return
        if len(self.board.history) != expected_history_len:
            return
        if self.board.is_over or self.board.current_player != self.ai_player:
            return
        if action not in self.board.legal_actions():
            return
        self.board.play_action(action)

    def _undo(self) -> None:
        self._clear_pending_ai()
        if self.ai_enabled and len(self.board.history) >= 2:
            self.board.undo()
            self.board.undo()
        else:
            self.board.undo()

    def _restart(self) -> None:
        self._clear_pending_ai()
        self.board = Board(size=15)

    def _clear_pending_ai(self) -> None:
        self.ai_generation += 1
        self.ai_result = None
        self.ai_error = None

    def _pixel_to_coord(self, x: int, y: int) -> tuple[int, int] | None:
        col = round((x - MARGIN) / CELL)
        row = round((y - MARGIN) / CELL)
        if not self.board.in_bounds(row, col):
            return None
        cx = MARGIN + col * CELL
        cy = MARGIN + row * CELL
        if abs(x - cx) > CELL * 0.42 or abs(y - cy) > CELL * 0.42:
            return None
        return row, col

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_board()
        self._draw_panel()
        pygame.display.flip()

    def _draw_board(self) -> None:
        for i in range(self.board.size):
            start = (MARGIN, MARGIN + i * CELL)
            end = (MARGIN + BOARD_PIXELS, MARGIN + i * CELL)
            pygame.draw.line(self.screen, GRID, start, end, 2)
            start = (MARGIN + i * CELL, MARGIN)
            end = (MARGIN + i * CELL, MARGIN + BOARD_PIXELS)
            pygame.draw.line(self.screen, GRID, start, end, 2)

        for row in range(self.board.size):
            for col in range(self.board.size):
                stone = self.board.grid[row, col]
                if stone == 0:
                    continue
                center = (MARGIN + col * CELL, MARGIN + row * CELL)
                color = (20, 22, 24) if stone == BLACK else (242, 242, 238)
                pygame.draw.circle(self.screen, color, center, STONE_RADIUS)
                pygame.draw.circle(self.screen, GRID, center, STONE_RADIUS, 1)

    def _draw_panel(self) -> None:
        x = MARGIN * 2 + BOARD_PIXELS
        pygame.draw.rect(self.screen, PANEL, (x, 0, SIDE_PANEL, HEIGHT))
        self._text("Gomoku AI", x + 24, 32, self.font, TEXT)
        self._text("15x15 free-style", x + 24, 65, self.small_font, MUTED)

        mode = "Human vs AI" if self.ai_enabled else "Human vs Human"
        self._text(f"Mode: {mode}", x + 24, 120, self.small_font, TEXT)
        self._text(f"Turn: {self._player_name(self.board.current_player)}", x + 24, 150, self.small_font, TEXT)
        self._text(f"Human: {self._player_name(-self.ai_player)}", x + 24, 180, self.small_font, TEXT)
        self._text(f"Moves: {len(self.board.history)}", x + 24, 210, self.small_font, TEXT)

        status = self._status_text()
        self._text(status, x + 24, 260, self.font, ACCENT)
        self._text(f"AI: {self.ai_name}", x + 24, 305, self.small_font, MUTED)

        self._text("S  Switch side", x + 24, HEIGHT - 180, self.small_font, MUTED)
        self._text("A  Toggle AI", x + 24, HEIGHT - 150, self.small_font, MUTED)
        self._text("U  Undo", x + 24, HEIGHT - 120, self.small_font, MUTED)
        self._text("R  Restart", x + 24, HEIGHT - 90, self.small_font, MUTED)
        self._text("Esc  Quit", x + 24, HEIGHT - 60, self.small_font, MUTED)

    def _status_text(self) -> str:
        if self.board.result is GameResult.BLACK_WIN:
            return "Black wins"
        if self.board.result is GameResult.WHITE_WIN:
            return "White wins"
        if self.board.result is GameResult.DRAW:
            return "Draw"
        if self.ai_error:
            return "AI error"
        if self.ai_enabled and self.board.current_player == self.ai_player:
            return "AI thinking"
        return "Playing"

    def _player_name(self, player: int) -> str:
        return "Black" if player == BLACK else "White"

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple[int, int, int]) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ai-backend",
        choices=["auto", "heuristic-mcts", "checkpoint-policy", "checkpoint-mcts"],
        default="auto",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    GomokuApp(ai_backend=args.ai_backend, checkpoint=args.checkpoint, device=args.device).run()

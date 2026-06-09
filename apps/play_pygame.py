from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame

from gomoku.board import BLACK, WHITE, Board, GameResult
from gomoku.mcts import MCTS

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


class GomokuApp:
    """Small pygame shell around the reusable board and MCTS logic."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Gomoku AI")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 17)
        self.board = Board(size=15)
        self.ai_enabled = True
        self.ai_player = WHITE
        self.mcts = MCTS(simulations=80)

    def run(self) -> None:
        while True:
            self._handle_events()
            self._maybe_ai_move()
            self._draw()
            self.clock.tick(60)

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
                    self.board = Board(size=15)
                if event.key == pygame.K_u:
                    self._undo()
                if event.key == pygame.K_a:
                    self.ai_enabled = not self.ai_enabled
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
        except ValueError:
            return

    def _maybe_ai_move(self) -> None:
        if not self.ai_enabled or self.board.is_over or self.board.current_player != self.ai_player:
            return
        # A modest simulation count keeps the UI responsive while exercising the real MCTS path.
        action = self.mcts.select_action(self.board, temperature=0.0)
        self.board.play_action(action)

    def _undo(self) -> None:
        if self.ai_enabled and len(self.board.history) >= 2:
            self.board.undo()
            self.board.undo()
        else:
            self.board.undo()

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
        self._text(f"Moves: {len(self.board.history)}", x + 24, 180, self.small_font, TEXT)

        status = self._status_text()
        self._text(status, x + 24, 235, self.font, ACCENT)

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
        if self.ai_enabled and self.board.current_player == self.ai_player:
            return "AI thinking"
        return "Playing"

    def _player_name(self, player: int) -> str:
        return "Black" if player == BLACK else "White"

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple[int, int, int]) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))


if __name__ == "__main__":
    GomokuApp().run()

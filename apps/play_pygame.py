from __future__ import annotations

import argparse
import sys
import threading
import time
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


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cpu"
    return device


def configure_torch_threads(torch_threads: int) -> None:
    if torch_threads <= 0:
        return
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def print_startup_info(initial_ai: str, checkpoint: Path, device: str) -> None:
    print("[gomoku] startup")
    print(f"[gomoku] torch={torch.__version__}")
    print(f"[gomoku] cuda_available={torch.cuda.is_available()}")
    print(f"[gomoku] cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"[gomoku] cuda_current_device={current}")
        for idx in range(torch.cuda.device_count()):
            print(f"[gomoku] cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
    print(f"[gomoku] initial_ai={initial_ai}")
    print(f"[gomoku] checkpoint={checkpoint}")
    print(f"[gomoku] checkpoint_exists={checkpoint.exists()}")
    print(f"[gomoku] resolved_device={device}")
    print(f"[gomoku] torch_num_threads={torch.get_num_threads()}")
    print(f"[gomoku] torch_num_interop_threads={torch.get_num_interop_threads()}")


def board_to_text(board: Board) -> str:
    symbols = {BLACK: "X", WHITE: "O", 0: "."}
    header = "    " + " ".join(f"{col:02d}" for col in range(board.size))
    rows = [
        f"{row:02d}  " + "  ".join(symbols[int(board.grid[row, col])] for col in range(board.size))
        for row in range(board.size)
    ]
    return "\n".join([header, *rows])


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
        initial_ai: str = "checkpoint-policy",
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
        mcts_simulations: int = 20,
        debug: bool = False,
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
        self.debug = debug
        device = resolve_device(device)
        print_startup_info(initial_ai, checkpoint, device)
        self.ai_backends = self._load_ai_backends(checkpoint, device, mcts_simulations)
        if initial_ai not in self.ai_backends:
            initial_ai = "heuristic-mcts"
        self.selected_ai_key = initial_ai
        print(f"[gomoku] loaded_ai={', '.join(self.ai_backends)}")
        print(f"[gomoku] selected_ai={self.ai_name}")
        if self.debug:
            print("[gomoku][debug] enabled")
            self._debug_board("initial board")
        self.ai_thread: threading.Thread | None = None
        self.ai_result: tuple[int, int, int, float] | None = None
        self.ai_error: str | None = None
        self.ai_generation = 0

    def run(self) -> None:
        while True:
            self._handle_events()
            self._apply_ai_result()
            self._start_ai_if_needed()
            self._draw()
            self.clock.tick(60)

    @property
    def ai(self):
        return self.ai_backends[self.selected_ai_key][0]

    @property
    def ai_name(self) -> str:
        return self.ai_backends[self.selected_ai_key][1]

    def _load_ai_backends(self, checkpoint: Path, device: str, mcts_simulations: int):
        backends = {
            "heuristic-mcts": (
                MCTS(simulations=mcts_simulations),
                f"heuristic MCTS ({mcts_simulations} sims)",
            )
        }
        if checkpoint.exists():
            backends["checkpoint-policy"] = (
                PolicyCheckpointAI(checkpoint, device),
                f"checkpoint policy ({device})",
            )
            model = PolicyValueNet()
            state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state["model"])
            backends["checkpoint-mcts"] = (
                MCTS(evaluator=NeuralEvaluator(model, device=device), simulations=mcts_simulations),
                f"checkpoint MCTS ({mcts_simulations} sims, {device})",
            )
        else:
            print(f"[gomoku] checkpoint_missing={checkpoint}")
        return backends

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
                if event.key == pygame.K_n:
                    self._toggle_ai_backend()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _toggle_ai_backend(self) -> None:
        if self.board.history:
            return
        keys = list(self.ai_backends)
        current_idx = keys.index(self.selected_ai_key)
        self.selected_ai_key = keys[(current_idx + 1) % len(keys)]
        print(f"[gomoku] selected_ai={self.ai_name}")
        if self.debug:
            self._debug(f"ai backend changed before opening move: {self.ai_name}")

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.board.is_over:
            return
        if self.ai_enabled and self.board.current_player == self.ai_player:
            return

        coord = self._pixel_to_coord(*pos)
        if coord is None:
            if self.debug:
                self._debug(f"ignored click outside board/snap area: pixel={pos}")
            return
        row, col = coord
        try:
            self.board.play(row, col)
            self._clear_pending_ai()
            if self.debug:
                self._debug(f"human move: player={self._player_name(-self.ai_player)} coord=({row}, {col})")
                self._debug_board("after human move")
        except ValueError:
            if self.debug:
                self._debug(f"ignored illegal click: pixel={pos} coord=({row}, {col})")
            return

    def _start_ai_if_needed(self) -> None:
        if not self.ai_enabled or self.board.is_over or self.board.current_player != self.ai_player:
            return
        if self.ai_result is not None:
            return
        if self.ai_thread and self.ai_thread.is_alive():
            return

        board_snapshot = self.board.copy()
        expected_history_len = len(self.board.history)
        generation = self.ai_generation
        self.ai_result = None
        self.ai_error = None
        if self.debug:
            self._debug(
                f"ai start: backend={self.ai_name} player={self._player_name(self.ai_player)} "
                f"move_count={expected_history_len}"
            )

        # Search runs off the pygame thread, so the human stone is drawn immediately.
        self.ai_thread = threading.Thread(
            target=self._compute_ai_move,
            args=(board_snapshot, expected_history_len, generation),
            daemon=True,
        )
        self.ai_thread.start()

    def _compute_ai_move(self, board: Board, expected_history_len: int, generation: int) -> None:
        try:
            started = time.perf_counter()
            action = self.ai.select_action(board, temperature=0.0)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.ai_result = (generation, expected_history_len, action, elapsed_ms)
            if self.debug:
                row, col = board.action_to_coord(action)
                self._debug(f"ai done: coord=({row}, {col}) elapsed_ms={elapsed_ms:.2f}")
        except Exception as exc:  # pragma: no cover - surfaced in the UI.
            self.ai_error = str(exc)
            if self.debug:
                self._debug(f"ai error: {exc}")

    def _apply_ai_result(self) -> None:
        if self.ai_result is None:
            return
        generation, expected_history_len, action, elapsed_ms = self.ai_result
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
        if self.debug:
            row, col = self.board.action_to_coord(action)
            self._debug(f"ai move applied: coord=({row}, {col}) elapsed_ms={elapsed_ms:.2f}")
            self._debug_board("after ai move")

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
        if self.debug:
            self._debug("restart")
            self._debug_board("after restart")

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
        lock = "locked" if self.board.history else "changeable"
        self._text(f"AI select: {lock}", x + 24, 330, self.small_font, MUTED)

        self._text("N  Cycle AI", x + 24, HEIGHT - 210, self.small_font, MUTED)
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

    def _debug(self, message: str) -> None:
        print(f"[gomoku][debug] {message}", flush=True)

    def _debug_board(self, title: str) -> None:
        self._debug(title)
        print(board_to_text(self.board), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-ai",
        choices=["heuristic-mcts", "checkpoint-policy", "checkpoint-mcts"],
        default="checkpoint-policy",
        help="Initial UI selection. Press N before the first move to switch.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto defaults to cpu for responsive UI; also accepts cpu, cuda, or cuda:N")
    parser.add_argument("--mcts-simulations", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    configure_torch_threads(args.torch_threads)
    GomokuApp(
        initial_ai=args.initial_ai,
        checkpoint=args.checkpoint,
        device=args.device,
        mcts_simulations=args.mcts_simulations,
        debug=args.debug,
    ).run()

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol

import numpy as np

from gomoku.board import Board, result_value_for_player
from gomoku.heuristics import evaluate_board, heuristic_policy


class Evaluator(Protocol):
    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        """Return policy priors and value from board.current_player perspective."""


class HeuristicEvaluator:
    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        return heuristic_policy(board), evaluate_board(board, board.current_player)


@dataclass
class Node:
    prior: float
    player: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    """PUCT search used by both heuristic and future neural evaluators."""

    def __init__(self, evaluator: Evaluator | None = None, simulations: int = 160, c_puct: float = 1.5):
        self.evaluator = evaluator or HeuristicEvaluator()
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        root = Node(prior=1.0, player=board.current_player)
        self._expand(root, board)

        for _ in range(self.simulations):
            sim_board = board.copy()
            node = root
            search_path = [node]

            while node.children and not sim_board.is_over:
                action, node = self._select_child(node)
                sim_board.play_action(action)
                search_path.append(node)

            value = self._evaluate_leaf(node, sim_board)
            self._backpropagate(search_path, value, sim_board.current_player)

        visits = np.zeros(board.size * board.size, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if visits.sum() == 0:
            legal = board.legal_actions()
            visits[legal] = 1.0
        return _visits_to_policy(visits, temperature)

    def select_action(self, board: Board, temperature: float = 0.0) -> int:
        policy = self.run(board, temperature=temperature)
        legal = board.legal_actions()
        if temperature == 0.0:
            return int(max(legal, key=lambda action: policy[action]))
        return int(np.random.choice(np.arange(board.size * board.size), p=policy))

    def _expand(self, node: Node, board: Board) -> None:
        priors, _ = self.evaluator.evaluate(board)
        legal = board.legal_actions()
        if not legal:
            return
        prior_sum = float(np.sum(priors[legal]))
        if prior_sum <= 0:
            prior = 1.0 / len(legal)
            for action in legal:
                node.children[action] = Node(prior=prior, player=-board.current_player)
        else:
            for action in legal:
                node.children[action] = Node(
                    prior=float(priors[action] / prior_sum),
                    player=-board.current_player,
                )

    def _select_child(self, node: Node) -> tuple[int, Node]:
        total_visits = max(1, node.visit_count)

        def score(item: tuple[int, Node]) -> float:
            _, child = item
            exploration = self.c_puct * child.prior * sqrt(total_visits) / (1 + child.visit_count)
            return -child.value + exploration

        return max(node.children.items(), key=score)

    def _evaluate_leaf(self, node: Node, board: Board) -> float:
        if board.is_over:
            return result_value_for_player(board.result, board.current_player)
        self._expand(node, board)
        _, value = self.evaluator.evaluate(board)
        return value

    def _backpropagate(self, search_path: list[Node], value: float, leaf_player: int) -> None:
        for node in reversed(search_path):
            signed_value = value if node.player == leaf_player else -value
            node.value_sum += signed_value
            node.visit_count += 1


def _visits_to_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 0.0:
        policy = np.zeros_like(visits, dtype=np.float64)
        policy[int(np.argmax(visits))] = 1.0
        return policy

    adjusted = np.power(visits, 1.0 / temperature)
    total = adjusted.sum()
    if total <= 0:
        return adjusted
    return adjusted / total

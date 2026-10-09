"""
AlphaZero Monte Carlo Tree Search (MCTS) implementation with PUCT and Tactics Filtering.
Features:
- Exact 1-step tactical filtering from `tactics.py`:
  * Root analysis: forces instant winning or single-reply blocking moves
  * Interior analysis: terminal proven nodes bypass neural network evaluation (+1/-1)
- PUCT with sqrt(parent_visits + 1) initialization
- Exact alternating sign backup in zero-sum tree
- Dirichlet noise injection at root for self-play exploration
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from board import Board
from tactics import interior_analysis, root_analysis


class TreeNode:
    """
    A node in the MCTS tree.
    Tracks visit count, prior probability, action value Q, and exploration bonus U.
    """

    def __init__(self, parent: Optional['TreeNode'], prior_p: float):
        self._parent = parent
        self._children: Dict[int, 'TreeNode'] = {}
        self._n_visits: int = 0
        self._q_value: float = 0.0
        self._u: float = 0.0
        self._p: float = prior_p
        self._is_proven: bool = False
        self._proven_value: Optional[float] = None

    def expand(self, action_priors: List[Tuple[int, float]]):
        """Expand tree by creating new children nodes with prior probabilities."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct: float) -> Tuple[int, 'TreeNode']:
        """Select action among children that maximizes action value Q + U."""
        return max(
            self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def get_value(self, c_puct: float) -> float:
        """
        PUCT formula:
        Q + c_puct * P * sqrt(N_parent + 1) / (1 + N_child)
        The `+ 1` ensures that unvisited children are ranked strictly by prior P.
        """
        parent_visits = self._parent._n_visits if self._parent else 1
        self._u = (
            c_puct * self._p * math.sqrt(parent_visits + 1) / (1.0 + self._n_visits)
        )
        return self._q_value + self._u

    def update(self, leaf_value: float):
        """Update visit count and running mean Q."""
        self._n_visits += 1
        self._q_value += (leaf_value - self._q_value) / self._n_visits

    def update_recursive(self, leaf_value: float):
        """
        Recursively update all ancestors up to the root.
        Alternates the sign of leaf_value at each step for two-player zero-sum game.
        """
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children expanded yet)."""
        return len(self._children) == 0


class MCTS:
    """
    Monte Carlo Tree Search with PUCT algorithm and Exact Tactical Pruning.
    """

    def __init__(
        self,
        policy_value_fn: Callable,
        c_puct: float = 2.5,
        n_playout: int = 400
    ):
        self._root = TreeNode(None, 1.0)
        self._policy_value_fn = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, board: Board):
        """Run a single MCTS simulation from root to a leaf, evaluate, and backpropagate."""
        node = self._root
        current_board = board.copy()

        # 1. Selection
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            current_board.do_move(action)

        # 2. Evaluation & 3. Expansion
        end, winner = current_board.game_end()
        if end:
            leaf_value = 0.0 if winner == -1 else (1.0 if winner == current_board.current_player else -1.0)
        else:
            # Tactical check: fast 1-step 5-in-a-row detection
            mask, proven = interior_analysis(current_board)

            if proven is not None:
                # Mathematically proven outcome (+1 immediate win, -1 double threat loss)
                # Skips neural network evaluation entirely!
                leaf_value = proven
            else:
                # Evaluate using neural network or heuristic evaluator
                if hasattr(self._policy_value_fn, 'evaluate'):
                    action_probs, leaf_value = self._policy_value_fn.evaluate(current_board, candidate_mask=mask)
                else:
                    action_probs, leaf_value = self._policy_value_fn(current_board)

                node.expand(action_probs)

        # 4. Backpropagation
        node.update_recursive(-leaf_value)

    def get_move_probs(
        self,
        board: Board,
        temp: float = 1e-3,
        add_dirichlet_noise: bool = False,
        noise_eps: float = 0.25,
        noise_alpha: float = 0.15
    ) -> Tuple[List[int], List[float]]:
        """Run all simulations and return candidate moves and their visit probabilities."""
        # Tactical root analysis
        root_mask = root_analysis(board)
        forced_indices = np.flatnonzero(root_mask)

        # If tactical root analysis forced only 1 move (e.g. immediate 5 or forced block)
        if len(forced_indices) == 1 and not add_dirichlet_noise:
            best_move = int(forced_indices[0])
            return [best_move], [1.0]

        # Expand root if leaf
        if self._root.is_leaf():
            if hasattr(self._policy_value_fn, 'evaluate'):
                action_probs, _ = self._policy_value_fn.evaluate(board, candidate_mask=root_mask)
            else:
                action_probs, _ = self._policy_value_fn(board)

            if add_dirichlet_noise and action_probs:
                # Sample Dirichlet noise
                gammas = [random.gammavariate(noise_alpha, 1.0) for _ in action_probs]
                s = sum(gammas) if sum(gammas) > 0 else 1.0
                noise = [g / s for g in gammas]
                action_probs = [
                    (act, (1 - noise_eps) * p + noise_eps * n)
                    for (act, p), n in zip(action_probs, noise)
                ]

            self._root.expand(action_probs)

        # Run simulations
        for _ in range(self._n_playout):
            self._playout(board)

        # Gather visit counts
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        if not act_visits:
            return [], []

        actions, visits = zip(*act_visits)
        actions = list(actions)
        visits = list(visits)

        if temp <= 1e-2:
            probs = [0.0] * len(visits)
            best_idx = max(
                range(len(actions)),
                key=lambda i: (
                    visits[i],
                    -self._root._children[actions[i]]._q_value,
                    self._root._children[actions[i]]._p
                )
            )
            probs[best_idx] = 1.0
        else:
            log_visits = [math.log(max(v, 1e-10)) / temp for v in visits]
            max_log = max(log_visits)
            exp_visits = [math.exp(v - max_log) for v in log_visits]
            sum_exp = sum(exp_visits)
            probs = [v / sum_exp for v in exp_visits]

        return actions, probs

    def update_with_move(self, last_move: int):
        """Advance the tree root along the chosen action branch."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def reset_root(self):
        """Reset root node for a new game."""
        self._root = TreeNode(None, 1.0)

    def get_root_eval(self) -> float:
        """Get the root node's estimated value in [-1.0, 1.0]."""
        return -self._root._q_value if self._root else 0.0

    def get_top_moves(self, board: Board, top_k: int = 3) -> List[Dict]:
        """Return the top K candidate moves with visit counts, Q-values, and win rates."""
        results = []
        for act, node in self._root._children.items():
            if node._n_visits > 0:
                q = -node._q_value
                win_rate = (q + 1.0) / 2.0 * 100.0
                results.append({
                    "action": act,
                    "coord": board.coord_to_str(act),
                    "visits": node._n_visits,
                    "q_value": q,
                    "win_rate": win_rate,
                    "prior": node._p
                })
        results.sort(key=lambda x: (x["visits"], x["q_value"]), reverse=True)
        return results[:top_k]

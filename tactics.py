"""
Exact tactical detector for Gomoku (一步成五检测与战术剪枝).
Precomputes all 572 length-5 straight line windows on the 15x15 board.
Uses vectorized window reduction to check if either player can form five in one move.
"""

from __future__ import annotations
import numpy as np

BOARD_SIZE = 15
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE


def _build_windows() -> np.ndarray:
    """Precompute every length-5 straight line on the board as flat cell indices."""
    windows = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                end_r, end_c = r + 4 * dr, c + 4 * dc
                if 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE:
                    windows.append([(r + i * dr) * BOARD_SIZE + (c + i * dc)
                                    for i in range(5)])
    return np.asarray(windows, dtype=np.int32)


# Precomputed 572 windows for 15x15 board
WINDOWS = _build_windows()


def five_in_one(state: np.ndarray):
    """
    Find cells that complete five for the mover (+1) and for the opponent (-1).
    `state` is the flat 225-element view of the board (+1 mine, -1 opponent, 0 empty).
    Returns (my_cells, opp_cells) as numpy arrays.
    """
    values = state[WINDOWS]                      # (572, 5)
    sums = values.sum(axis=1)                    # (572,)

    my_cells = _empty_cells(values, sums, 4) if sums.max() == 4 else _NO_CELLS
    opp_cells = _empty_cells(values, sums, -4) if sums.min() == -4 else _NO_CELLS
    return my_cells, opp_cells


_NO_CELLS = np.empty(0, dtype=np.int32)


def _empty_cells(values: np.ndarray, sums: np.ndarray, target: int) -> np.ndarray:
    """Extract empty cells from windows whose values sum to target (+4 or -4)."""
    selected = sums == target
    return WINDOWS[selected][values[selected] == 0]


def interior_analysis(board):
    """
    Tactical verdict for a leaf node during MCTS expansion.
    Returns (mask, proven_value):
      - proven_value is not None when the outcome is mathematically decided:
          +1.0: current player completes five immediately (skip neural net!)
          -1.0: opponent has two separate winning cells (unblockable defeat)
      - mask: restricted legal moves. If opponent has exactly 1 winning cell,
        that single blocking cell is the ONLY move allowed to be searched.
    """
    state = (board.get_stones_array() * board.to_play).reshape(-1)
    my_cells, opp_cells = five_in_one(state)

    if my_cells.size:
        return None, 1.0

    if opp_cells.size:
        cells = np.unique(opp_cells)
        if cells.size > 1:
            return None, -1.0
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        mask[cells[0]] = True
        return mask, None

    return board.candidate_mask(), None


def root_analysis(board) -> np.ndarray:
    """
    Tactical candidate moves at the root node.
    Forces immediate win if available, or immediate block if opponent threatens win.
    """
    state = (board.get_stones_array() * board.to_play).reshape(-1)
    my_cells, opp_cells = five_in_one(state)

    if my_cells.size:
        forced = np.unique(my_cells)
    elif opp_cells.size:
        forced = np.unique(opp_cells)
    else:
        return board.candidate_mask()

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[forced] = True
    return mask

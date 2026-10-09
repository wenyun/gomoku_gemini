"""
Gomoku (五子棋) Board & Game State Implementation.
Supports standard 15x15 board and arbitrary sizes.
Features:
- Fast numpy array representation for stones and incremental neighborhood mask
- 5-plane feature encoding (my stones, opp stones, opp last move, my prev move, color plane)
- Exact win checking and copy operations
- Seamless compatibility with Tkinter GUI and MCTS engines
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
import numpy as np


class Board:
    """
    Board state for Gomoku.
    Black is player 1 (+1).
    White is player 2 (-1).
    Empty cell is 0.
    """

    def __init__(self, width: int = 15, height: int = 15, n_in_row: int = 5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]

        # Numpy state representations: +1 (Black), -1 (White), 0 (Empty)
        self.stones = np.zeros((height, width), dtype=np.int8)
        self.near = np.zeros((height, width), dtype=np.int16)
        self.to_play = 1  # +1: Black, -1: White

        # Compatibility dictionaries and lists
        self.states: Dict[int, int] = {}  # move -> player (1 or 2)
        self.current_player: int = 1      # 1 or 2
        self.last_move: Optional[int] = None
        self.prev_move: Optional[int] = None
        self.move_history: List[Tuple[int, int]] = []  # [(move, player), ...]

    def init_board(self, start_player: int = 1):
        """Reset the board to initial state."""
        self.stones.fill(0)
        self.near.fill(0)
        self.to_play = 1 if start_player == 1 else -1
        self.current_player = start_player
        self.states = {}
        self.last_move = None
        self.prev_move = None
        self.move_history = []

    def move_to_location(self, move: int) -> Tuple[int, int]:
        """Convert 1D move index to (row, col)."""
        row = move // self.width
        col = move % self.width
        return row, col

    def location_to_move(self, row: int, col: int) -> int:
        """Convert (row, col) to 1D move index."""
        return row * self.width + col

    def coord_to_str(self, move: int) -> str:
        """Convert move index to algebraic coordinate string like H8."""
        row, col = self.move_to_location(move)
        col_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        col_char = col_letters[col] if col < len(col_letters) else chr(ord('A') + col)
        row_num = self.height - row
        return f"{col_char}{row_num}"

    def get_stones_array(self) -> np.ndarray:
        """Return the flat (H*W) or 2D stones array."""
        return self.stones

    def candidate_mask(self) -> np.ndarray:
        """
        Boolean mask of candidate moves within distance 2 of existing stones.
        If board is empty, only the center cell is true.
        """
        empty = (self.stones == 0)
        if len(self.states) == 0:
            mask = np.zeros((self.height, self.width), dtype=bool)
            mask[self.height // 2, self.width // 2] = True
            return mask.reshape(-1)

        mask = empty & (self.near > 0)
        if not np.any(mask):
            return empty.reshape(-1)
        return mask.reshape(-1)

    def get_candidate_moves(self, distance: int = 2) -> List[int]:
        """Return list of candidate moves within Chebyshev distance 2 of existing stones."""
        mask = self.candidate_mask()
        return list(np.flatnonzero(mask))

    def get_legal_moves(self) -> List[int]:
        """Return all empty cells on the board."""
        return list(np.flatnonzero(self.stones.reshape(-1) == 0))

    def _update_near(self, r: int, c: int, delta: int):
        """Update neighbor counts in a 5x5 window around (r, c)."""
        r_min = max(0, r - 2)
        r_max = min(self.height, r + 3)
        c_min = max(0, c - 2)
        c_max = min(self.width, c + 3)
        self.near[r_min:r_max, c_min:c_max] += delta

    def do_move(self, move: int):
        """Execute a move for the current player."""
        if move in self.states:
            raise ValueError(f"Move {move} is already occupied!")

        r, c = self.move_to_location(move)
        sign = 1 if self.current_player == 1 else -1

        self.stones[r, c] = sign
        self._update_near(r, c, +1)
        self.states[move] = self.current_player
        self.move_history.append((move, self.current_player))

        self.prev_move = self.last_move
        self.last_move = move

        # Switch player
        self.current_player = 3 - self.current_player
        self.to_play = -self.to_play

    def undo_move(self) -> Optional[Tuple[int, int]]:
        """Undo the last move. Returns (move, player) or None if board empty."""
        if not self.move_history:
            return None

        last_move, last_player = self.move_history.pop()
        del self.states[last_move]
        r, c = self.move_to_location(last_move)
        self.stones[r, c] = 0
        self._update_near(r, c, -1)

        self.current_player = last_player
        self.to_play = 1 if last_player == 1 else -1
        self.last_move = self.move_history[-1][0] if self.move_history else None
        self.prev_move = self.move_history[-2][0] if len(self.move_history) >= 2 else None
        return last_move, last_player

    def has_a_winner(self) -> Tuple[bool, int, List[Tuple[int, int]]]:
        """
        Check if the game has a winner based on last_move.
        Returns: (has_winner: bool, winner: int, winning_line: List[(r, c)])
        """
        if self.last_move is None or not self.move_history:
            return False, -1, []

        last_move, last_player = self.move_history[-1]
        r, c = self.move_to_location(last_move)
        sign = 1 if last_player == 1 else -1
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            line = [(r, c)]
            # Forward
            step = 1
            while True:
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < self.height and 0 <= nc < self.width and self.stones[nr, nc] == sign:
                    line.append((nr, nc))
                    step += 1
                else:
                    break

            # Backward
            step = 1
            while True:
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < self.height and 0 <= nc < self.width and self.stones[nr, nc] == sign:
                    line.append((nr, nc))
                    step += 1
                else:
                    break

            if len(line) >= self.n_in_row:
                return True, last_player, line

        return False, -1, []

    def game_end(self) -> Tuple[bool, int]:
        """Check if game has ended. winner: 1 (Black), 2 (White), or -1 (Draw)."""
        has_win, winner, _ = self.has_a_winner()
        if has_win:
            return True, winner
        if len(self.states) == self.width * self.height:
            return True, -1  # Draw
        return False, -1

    def copy(self) -> 'Board':
        """Create a fast deep copy of the board state."""
        b = Board(self.width, self.height, self.n_in_row)
        b.stones = self.stones.copy()
        b.near = self.near.copy()
        b.to_play = self.to_play
        b.states = self.states.copy()
        b.current_player = self.current_player
        b.last_move = self.last_move
        b.prev_move = self.prev_move
        b.move_history = self.move_history.copy()
        return b

    def encode_planes(self) -> np.ndarray:
        """
        AlphaZero 5-plane feature representation:
        Shape: (5, H, W)
        - Plane 0: Current player's stones (+1.0 where present)
        - Plane 1: Opponent's stones (+1.0 where present)
        - Plane 2: Opponent's last move (1.0 at last_move)
        - Plane 3: Current player's previous move (1.0 at prev_move)
        - Plane 4: Color to move (all 1.0 if Black, all 0.0 if White)
        """
        planes = np.zeros((5, self.height, self.width), dtype=np.float32)

        # View from current mover perspective
        planes[0] = (self.stones == self.to_play).astype(np.float32)
        planes[1] = (self.stones == -self.to_play).astype(np.float32)

        if self.last_move is not None:
            lr, lc = self.move_to_location(self.last_move)
            planes[2, lr, lc] = 1.0

        if self.prev_move is not None:
            pr, pc = self.move_to_location(self.prev_move)
            planes[3, pr, pc] = 1.0

        if self.to_play == 1:
            planes[4].fill(1.0)

        return planes

    def current_state_planes(self) -> List[List[List[float]]]:
        """Backward-compatibility interface."""
        return self.encode_planes().tolist()

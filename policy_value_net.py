"""
AlphaZero Policy-Value Neural Network & Heuristic Evaluator.
Features:
- AlphaZero Deep Residual Network with:
  * 5 input feature planes (my stones, opp stones, opp last move, my prev move, color plane)
  * 6 Residual Blocks (128 channels)
  * Translation-equivariant 1x1 Conv Policy Head (avoids 7200->225 dense flattening)
  * Non-saturating 1-channel bottleneck Value Head (avoids Tanh saturation dead-end)
- HeuristicPolicyValueNet fallback for instant play
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from board import Board


# ============================================================================
# 1. Pure-Python Fast Pattern-Guided Policy-Value Evaluator
# ============================================================================

class HeuristicPolicyValueNet:
    """Expert heuristic pattern evaluator for zero-shot instant play."""

    SCORE_FIVE = 100000
    SCORE_LIVE_FOUR = 10000
    SCORE_RUSH_FOUR = 2500
    SCORE_LIVE_THREE = 1800
    SCORE_SLEEP_THREE = 400
    SCORE_LIVE_TWO = 120
    SCORE_SLEEP_TWO = 25

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def _eval_line_segment(self, segment: str) -> int:
        if '11111' in segment:
            return self.SCORE_FIVE
        score = 0
        if '_1111_' in segment:
            score += self.SCORE_LIVE_FOUR
        elif any(p in segment for p in ('21111_', '_11112', '111_1', '11_11', '1_111')):
            score += self.SCORE_RUSH_FOUR

        if any(p in segment for p in ('_111_', '_11_1_', '_1_11_')):
            score += self.SCORE_LIVE_THREE
        elif any(p in segment for p in ('2111__', '__1112', '211_1_', '_1_112', '21_11_', '_11_12')):
            score += self.SCORE_SLEEP_THREE

        if any(p in segment for p in ('__11__', '___11_', '_11___', '_1_1_', '_1__1_')):
            score += self.SCORE_LIVE_TWO
        elif any(p in segment for p in ('211___', '___112', '21_1__', '__1_12')):
            score += self.SCORE_SLEEP_TWO

        return score

    def _score_point(self, board: Board, r: int, c: int, player: int) -> int:
        opp = 3 - player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total = 0
        for dr, dc in directions:
            chars = []
            for step in range(-4, 5):
                if step == 0:
                    chars.append('1')
                    continue
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < board.height and 0 <= nc < board.width:
                    m = board.location_to_move(nr, nc)
                    st = board.states.get(m, 0)
                    if st == player:
                        chars.append('1')
                    elif st == opp:
                        chars.append('2')
                    else:
                        chars.append('_')
                else:
                    chars.append('2')
            total += self._eval_line_segment("".join(chars))
        return total

    def evaluate(self, board: Board, candidate_mask: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, float]], float]:
        if candidate_mask is not None:
            candidates = [int(x) for x in np.flatnonzero(candidate_mask)]
        else:
            candidates = board.get_candidate_moves(distance=2)
        if not candidates:
            return [], 0.0

        current_player = board.current_player
        opp_player = 3 - current_player
        scores = []
        max_score = 0
        center_r, center_c = board.height // 2, board.width // 2

        critical_wins, critical_blocks = [], []

        for move in candidates:
            r, c = board.move_to_location(move)
            offense = self._score_point(board, r, c, current_player)
            defense = self._score_point(board, r, c, opp_player)

            if offense >= self.SCORE_FIVE:
                critical_wins.append(move)
            if defense >= self.SCORE_FIVE:
                critical_blocks.append(move)

            dist_to_center = abs(r - center_r) + abs(c - center_c)
            center_bonus = max(0, 15 - dist_to_center)
            move_score = offense + int(1.3 * defense) + center_bonus
            scores.append((move, move_score))
            if move_score > max_score:
                max_score = move_score

        if critical_wins:
            p_win = 0.95 / len(critical_wins)
            p_other = 0.05 / max(1, len(candidates) - len(critical_wins))
            return [(m, p_win if m in critical_wins else p_other) for m in candidates], 1.0

        if critical_blocks:
            p_block = 0.90 / len(critical_blocks)
            p_other = 0.10 / max(1, len(candidates) - len(critical_blocks))
            return [(m, p_block if m in critical_blocks else p_other) for m in candidates], -0.3

        temp = max(self.temperature, 0.2)
        mean_score = sum(sc for _, sc in scores) / len(scores)
        scaled = [(sc - mean_score) / (400.0 * temp) for _, sc in scores]
        exp_s = [math.exp(max(-20.0, min(sc, 20.0))) for sc in scaled]
        sum_exp = sum(exp_s) if sum(exp_s) > 0 else 1.0
        action_probs = [(m, s / sum_exp) for (m, _), s in zip(scores, exp_s)]

        curr_max = max((self._score_point(board, *board.move_to_location(m), current_player) for m in candidates), default=0)
        opp_max = max((self._score_point(board, *board.move_to_location(m), opp_player) for m in candidates), default=0)
        value = math.tanh((curr_max - opp_max) / 2000.0)

        return action_probs, value

    def __call__(self, board: Board) -> Tuple[List[Tuple[int, float]], float]:
        return self.evaluate(board)


# ============================================================================
# 2. Advanced AlphaZero Deep Residual Network
# ============================================================================

if HAS_TORCH:
    class ResidualBlock(nn.Module):
        """AlphaZero Residual Block with Pre/Post BN and skip connection."""
        def __init__(self, channels: int = 128):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out, inplace=True)

    class AlphaZeroResNet(nn.Module):
        """
        Full AlphaZero ResNet:
        - 5 feature planes input
        - 6 residual blocks with 128 channels
        - 1x1 Conv translation-equivariant policy head
        - 1-channel bottleneck non-saturating value head
        """
        def __init__(self, board_size: int = 15, in_channels: int = 5,
                     num_channels: int = 128, num_res_blocks: int = 6):
            super().__init__()
            self.board_size = board_size

            # Stem
            self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
            self.bn_in = nn.BatchNorm2d(num_channels)

            # Residual Backbone
            self.res_blocks = nn.ModuleList([
                ResidualBlock(num_channels) for _ in range(num_res_blocks)
            ])

            # Policy Head: 1x1 Conv (translation-equivariant)
            self.p_conv1 = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
            self.p_bn1 = nn.BatchNorm2d(32)
            self.p_conv2 = nn.Conv2d(32, 1, kernel_size=1, bias=True)

            # Value Head: 1x1 Conv to 1 channel bottleneck -> Linear(225, 64) -> Tanh
            self.v_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
            self.v_bn = nn.BatchNorm2d(1)
            self.v_fc1 = nn.Linear(board_size * board_size, 64)
            self.v_fc2 = nn.Linear(64, 1)

        def forward(self, x):
            # Backbone
            h = F.relu(self.bn_in(self.conv_in(x)), inplace=True)
            for block in self.res_blocks:
                h = block(h)

            # Policy Head: (B, 1, H, W) -> (B, H*W)
            p = F.relu(self.p_bn1(self.p_conv1(h)), inplace=True)
            logits = self.p_conv2(p).view(x.size(0), -1)

            # Value Head: (B, 1, H, W) -> (B, H*W) -> (B, 64) -> (B, 1)
            v = F.relu(self.v_bn(self.v_conv(h)), inplace=True)
            v = v.view(x.size(0), -1)
            v = F.relu(self.v_fc1(v), inplace=True)
            value = torch.tanh(self.v_fc2(v))

            return logits, value


    class PyTorchPolicyValueNet:
        """PyTorch AlphaZero wrapper supporting MPS (Apple Silicon GPU), CUDA, and CPU."""

        def __init__(self, board_size: int = 15, model_file: Optional[str] = None, use_gpu: bool = True):
            self.board_size = board_size
            if use_gpu and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.policy_value_net = AlphaZeroResNet(
                board_size=board_size, in_channels=5, num_channels=128, num_res_blocks=6
            ).to(self.device)

            self.optimizer = optim.AdamW(
                self.policy_value_net.parameters(), lr=1e-3, weight_decay=1e-4
            )

            if model_file:
                self.load_model(model_file)

        def evaluate(self, board: Board, candidate_mask: Optional[np.ndarray] = None) -> Tuple[List[Tuple[int, float]], float]:
            """Evaluate a single board position."""
            self.policy_value_net.eval()
            planes = board.encode_planes()
            tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value = self.policy_value_net(tensor)
                logits_np = logits.squeeze(0).cpu().numpy()
                val = float(value.item())

            # Apply candidate mask if provided, else board.candidate_mask()
            if candidate_mask is None:
                mask = board.candidate_mask()
            else:
                mask = candidate_mask

            legal_indices = np.flatnonzero(mask)
            if legal_indices.size == 0:
                legal_indices = np.array(board.get_legal_moves())

            # Softmax over masked candidate logits
            masked_logits = logits_np[legal_indices]
            max_logit = np.max(masked_logits)
            exp_logits = np.exp(masked_logits - max_logit)
            sum_exp = np.sum(exp_logits)
            probs = exp_logits / (sum_exp if sum_exp > 0 else 1.0)

            action_probs = [(int(idx), float(p)) for idx, p in zip(legal_indices, probs)]
            return action_probs, val

        def __call__(self, board: Board) -> Tuple[List[Tuple[int, float]], float]:
            return self.evaluate(board)

        def save_model(self, file_path: str):
            torch.save(self.policy_value_net.state_dict(), file_path)

        def load_model(self, file_path: str):
            state_dict = torch.load(file_path, map_location=self.device)
            self.policy_value_net.load_state_dict(state_dict)

else:
    class PyTorchPolicyValueNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed in the current Python environment.")

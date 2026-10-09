"""
Upgraded AlphaZero Training Pipeline for Gomoku.
Features:
- Exact tactical pruning via tactics.py (prevents blunders during self-play)
- Sliding-window ReplayBuffer (discards stale games from early weak play)
- 8-fold symmetry data augmentation on 5 feature planes and 15x15 policy distributions
- AdamW optimizer with 500-step LR warmup to prevent Tanh saturation
- Optimized for Apple Silicon MPS (Metal Performance Shaders) GPU acceleration
"""

from __future__ import annotations
import os
import random
import time
from collections import deque
from typing import List, Tuple, Dict
import numpy as np

from board import Board
from mcts import MCTS
from policy_value_net import HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from policy_value_net import PyTorchPolicyValueNet


class ReplayBuffer:
    """Sliding-window replay buffer for AlphaZero training positions."""

    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_added = 0

    def extend(self, items: List[Tuple[np.ndarray, np.ndarray, float]]):
        self.buffer.extend(items)
        self.total_added += len(items)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)

        # Convert to torch tensors
        state_tensor = torch.from_numpy(np.stack(states)).float()
        policy_tensor = torch.from_numpy(np.stack(policies)).float()
        value_tensor = torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(1)
        return state_tensor, policy_tensor, value_tensor

    def __len__(self):
        return len(self.buffer)


class TrainPipelineV2:
    """Upgraded AlphaZero training pipeline."""

    def __init__(
        self,
        board_size: int = 15,
        model_file: str = None,
        save_name: str = "alphazero_15x15_v2.pth"
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to run the training pipeline.")

        self.board_size = board_size
        self.board = Board(width=board_size, height=board_size)

        # Hyperparameters
        self.c_puct = 2.5
        self.n_playout = 200
        self.batch_size = 128
        self.epochs_per_iter = 3
        self.lr = 1e-3
        self.warmup_steps = 500
        self.step_count = 0

        # Buffer
        self.replay_buffer = ReplayBuffer(capacity=200000)

        # Paths
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_save_path = os.path.join(self.models_dir, save_name)

        # Initialize network
        if model_file is None and os.path.exists(self.model_save_path):
            model_file = self.model_save_path
            print(f"🔄 发现已有权重，自动加载增量训练: {self.model_save_path}", flush=True)

        self.policy_value_net = PyTorchPolicyValueNet(board_size=board_size, model_file=model_file)
        self.mcts = MCTS(self.policy_value_net, c_puct=self.c_puct, n_playout=self.n_playout)

        # Optimizer with warmup
        self.optimizer = optim.AdamW(
            self.policy_value_net.policy_value_net.parameters(),
            lr=self.lr, weight_decay=1e-4
        )

    def _get_augmented_data(
        self, game_data: List[Tuple[np.ndarray, np.ndarray, float]]
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Apply 8 symmetries (4 rotations x horizontal flip) to (state, policy, value).
        state: (5, H, W)
        policy: (H*W,)
        """
        augmented = []
        for state, pi, z in game_data:
            pi_2d = pi.reshape(self.board_size, self.board_size)

            for rot in range(4):
                # Rotate state and policy
                rot_state = np.rot90(state, rot, axes=(1, 2)).copy()
                rot_pi = np.rot90(pi_2d, rot).copy().reshape(-1)
                augmented.append((rot_state, rot_pi, z))

                # Flip horizontally
                flip_state = np.flip(rot_state, axis=2).copy()
                flip_pi = np.fliplr(rot_pi.reshape(self.board_size, self.board_size)).copy().reshape(-1)
                augmented.append((flip_state, flip_pi, z))

        return augmented

    def self_play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Run one full game of self-play and return augmented trajectory."""
        self.board.init_board(start_player=1)
        self.mcts.reset_root()

        states, policies, players = [], [], []
        step = 0

        while True:
            step += 1
            # Temperature schedule: 1.0 for first 10 moves, then 0.25
            temp = 1.0 if step <= 10 else 0.25

            acts, probs = self.mcts.get_move_probs(
                self.board, temp=temp, add_dirichlet_noise=True, noise_eps=0.25, noise_alpha=0.15
            )

            # Record state and full 225-dim policy vector
            full_pi = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            for act, p in zip(acts, probs):
                full_pi[act] = p

            states.append(self.board.encode_planes())
            policies.append(full_pi)
            players.append(self.board.current_player)

            # Sample move
            if temp <= 1e-2:
                move = acts[probs.index(max(probs))]
            else:
                move = random.choices(acts, weights=probs)[0]

            self.board.do_move(move)
            self.mcts.update_with_move(move)

            end, winner = self.board.game_end()
            if end:
                # Value z from perspective of current mover at that step
                values = []
                for p in players:
                    if winner == -1:
                        values.append(0.0)
                    elif winner == p:
                        values.append(1.0)
                    else:
                        values.append(-1.0)

                raw_data = list(zip(states, policies, values))
                return self._get_augmented_data(raw_data), step, winner

    def train_step(self) -> Tuple[float, float]:
        """Perform mini-batch gradient descent."""
        self.policy_value_net.policy_value_net.train()
        total_p_loss, total_v_loss = 0.0, 0.0

        for _ in range(self.epochs_per_iter):
            state_b, policy_b, value_b = self.replay_buffer.sample(self.batch_size)
            state_b = state_b.to(self.policy_value_net.device)
            policy_b = policy_b.to(self.policy_value_net.device)
            value_b = value_b.to(self.policy_value_net.device)

            self.step_count += 1
            # LR warmup
            if self.step_count <= self.warmup_steps:
                cur_lr = self.lr * (self.step_count / self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = cur_lr

            self.optimizer.zero_grad()
            logits, value_pred = self.policy_value_net.policy_value_net(state_b)

            # Policy Loss: Cross Entropy with MCTS target distribution
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -torch.mean(torch.sum(policy_b * log_probs, dim=1))

            # Value Loss: MSE with game winner
            value_loss = F.mse_loss(value_pred, value_b)

            loss = policy_loss + 1.0 * value_loss
            loss.backward()
            self.optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()

        return total_p_loss / self.epochs_per_iter, total_v_loss / self.epochs_per_iter

    def run(self, max_iterations: int = 800, games_per_iter: int = 2):
        """Main training loop."""
        print(f"🚀 启动 AlphaZero V2 强化学习流水线 (棋盘 {self.board_size}x{self.board_size})...", flush=True)
        dev = self.policy_value_net.device
        dev_str = "Apple Silicon GPU (MPS 加速已激活)" if dev.type == "mps" else str(dev)
        print(f"训练硬件: {dev_str}", flush=True)
        print(f"模型保存位置: {self.model_save_path}", flush=True)

        start_time = time.time()

        for it in range(1, max_iterations + 1):
            iter_start = time.time()
            step_lens = []

            for g in range(games_per_iter):
                aug_data, moves, winner = self.self_play_game()
                self.replay_buffer.extend(aug_data)
                step_lens.append(moves)

            p_loss, v_loss = self.train_step()
            iter_time = time.time() - iter_start
            elapsed_total = (time.time() - start_time) / 3600.0

            avg_moves = sum(step_lens) / len(step_lens)
            print(
                f"[{it:04d}/{max_iterations}] "
                f"对局手数: {avg_moves:.1f} | "
                f"策略Loss: {p_loss:.4f} | 价值Loss: {v_loss:.4f} | "
                f"样本池: {len(self.replay_buffer):6d} | "
                f"单轮耗时: {iter_time:.1f}s | "
                f"累计用时: {elapsed_total:.2f}h",
                flush=True
            )

            # Save checkpoint every iteration
            self.policy_value_net.save_model(self.model_save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero Gomoku V2 Training")
    parser.add_argument("--board_size", type=int, default=15, help="Board size")
    parser.add_argument("--iterations", type=int, default=800, help="Number of training iterations")
    parser.add_argument("--playouts", type=int, default=200, help="MCTS playouts per move")
    parser.add_argument("--model_name", type=str, default="alphazero_15x15_v2.pth", help="Checkpoint filename")
    args = parser.parse_args()

    trainer = TrainPipelineV2(
        board_size=args.board_size,
        save_name=args.model_name
    )
    trainer.n_playout = args.playouts
    trainer.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()

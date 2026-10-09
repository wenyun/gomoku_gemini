"""
Cross-Repository Benchmark:
Our AlphaZero V2 (Trained on Apple M5 GPU)
vs
Target Repo AlphaZero (Trained on 8x NVIDIA B200 GPUs, 1.25M games).

Pits both models against each other with identical MCTS simulation budgets,
alternating Black (first) and White (second).
"""

from __future__ import annotations
import sys
import os
import time
from typing import Tuple

import torch
import numpy as np

# Add both repo paths to sys.path
GEMINI_DIR = os.path.dirname(os.path.abspath(__file__))
B200_DIR = "/Users/wenyun/dev/gomoku-alphazero"

sys.path.insert(0, B200_DIR)
sys.path.insert(0, GEMINI_DIR)

from gomoku_zero.player import AlphaZeroPlayer as B200Player
from gomoku_zero.board import Board as B200Board, action_to_coord
from board import Board as GeminiBoard
from mcts import MCTS as GeminiMCTS
from policy_value_net import PyTorchPolicyValueNet as GeminiNet


def play_one_game(
    black_agent_type: str,
    gemini_mcts: GeminiMCTS,
    b200_player: B200Player,
    max_steps: int = 120
) -> Tuple[str, int, str]:
    """
    Play one game.
    black_agent_type: 'gemini_v2' or 'b200'
    Returns: (winner_name, steps, last_coord)
    """
    gb = GeminiBoard(15, 15)
    bb = B200Board()
    gemini_mcts.reset_root()

    step = 0
    last_coord = ""

    while step < max_steps:
        step += 1
        curr_player = gb.current_player  # 1: Black, 2: White

        is_gemini_turn = (curr_player == 1 and black_agent_type == 'gemini_v2') or \
                         (curr_player == 2 and black_agent_type == 'b200')

        if is_gemini_turn:
            acts, probs = gemini_mcts.get_move_probs(gb, temp=1e-3)
            move = acts[probs.index(max(probs))]
        else:
            moves = b200_player.select_moves([bb])
            move = moves[0]

        last_coord = action_to_coord(move)

        # Apply to both board states
        gb.do_move(move)
        gemini_mcts.update_with_move(move)
        bb.play(move)

        if bb.is_over:
            # Winner is either 1 (Black) or -1 (White) in bb.winner
            if bb.winner == 1:
                winner_name = "Gemini V2" if black_agent_type == 'gemini_v2' else "B200 模型"
            elif bb.winner == -1:
                winner_name = "B200 模型" if black_agent_type == 'gemini_v2' else "Gemini V2"
            else:
                winner_name = "和棋"
            return winner_name, step, last_coord

    return "和棋 (达到最大步数)", step, last_coord


def run_cross_battle(n_games: int = 4, sims: int = 200):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    v2_weights = os.path.join(GEMINI_DIR, "models", "alphazero_15x15_v2.pth")
    b200_weights = os.path.join(B200_DIR, "runs", "az", "latest.pt")

    if not os.path.exists(v2_weights):
        print(f"Error: Gemini V2 weights not found at {v2_weights}")
        return
    if not os.path.exists(b200_weights):
        print(f"Error: B200 weights not found at {b200_weights}")
        return

    print("=" * 72)
    print("  ⚔️  AlphaZero 世纪对决：Mac 本地 V2 模型  VS  8x B200 工业级模型")
    print("=" * 72)
    print(f"  ● Gemini V2 模型 : {v2_weights} (Apple M5 GPU, 2,122 轮)")
    print(f"  ○ B200 模型      : {b200_weights} (8x B200, 125 万局)")
    print(f"  推演硬件: {device} (Apple M5 GPU MPS 加速)")
    print(f"  每步 MCTS 模拟: {sims} 次 | 对局数: {n_games} 局 (黑白轮换)")
    print("=" * 72 + "\n")

    # Load agents
    print("正在加载双边模型权重至 Mac GPU...", end="", flush=True)
    g_net = GeminiNet(board_size=15, model_file=v2_weights)
    gemini_mcts = GeminiMCTS(g_net, c_puct=2.5, n_playout=sims)

    b200_player = B200Player(b200_weights, device=device, simulations=sims)
    print(" 加载完成！\n")

    scores = {"Gemini V2": 0, "B200 模型": 0, "和棋": 0}
    lengths = []
    t0 = time.time()

    for i in range(1, n_games + 1):
        g_start = time.time()
        if i % 2 == 1:
            black_agent = 'gemini_v2'
            print(f"[{i:02d}/{n_games:02d}] 执黑先手: Gemini V2  vs  执白后手: B200 模型 ... ", end="", flush=True)
        else:
            black_agent = 'b200'
            print(f"[{i:02d}/{n_games:02d}] 执黑先手: B200 模型  vs  执白后手: Gemini V2 ... ", end="", flush=True)

        winner, steps, last_c = play_one_game(black_agent, gemini_mcts, b200_player)
        g_time = time.time() - g_start
        lengths.append(steps)

        if "Gemini" in winner:
            scores["Gemini V2"] += 1
            print(f"🎉 Gemini V2 获胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)", flush=True)
        elif "B200" in winner:
            scores["B200 模型"] += 1
            print(f"🏆 B200 模型 获胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)", flush=True)
        else:
            scores["和棋"] += 1
            print(f"🤝 和棋! (手数: {steps:2d}, 耗时: {g_time:.1f}s)", flush=True)

    total_time = time.time() - t0
    print("\n" + "=" * 72)
    print("                       🏁 巅峰对决终局战报")
    print("=" * 72)
    print(f"总耗时: {total_time:.1f} 秒 | 平均每局手数: {sum(lengths)/len(lengths):.1f} 手")
    print(f"● Gemini V2 总得分 : {scores['Gemini V2']} 胜")
    print(f"○ B200 模型 总得分 : {scores['B200 模型']} 胜")
    if scores['和棋'] > 0:
        print(f"🤝 和棋局数        : {scores['和棋']} 局")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4, help="Number of games")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations")
    args = parser.parse_args()

    run_cross_battle(n_games=args.games, sims=args.sims)

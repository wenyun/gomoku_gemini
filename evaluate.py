"""
Benchmark and Evaluation script for AlphaZero Gomoku.
Pits the trained PyTorch AlphaZero neural network against the Heuristic expert engine.
Supports Black/White alternating matches, tracking win rates, game lengths, and stats.
"""

from __future__ import annotations
import os
import time
from typing import Tuple, Dict

from board import Board
from mcts import MCTS
from policy_value_net import HeuristicPolicyValueNet, PyTorchPolicyValueNet, HAS_TORCH


def play_eval_game(black_mcts, white_mcts, board_size=15, max_steps=120, opening_random=2) -> Tuple[int, int, str]:
    """
    Play one game between two players.
    Returns: (winner, num_moves, last_move_coord)
    winner: 1 (Black), 2 (White), -1 (Draw)
    """
    board = Board(width=board_size, height=board_size)
    black_mcts.reset_root()
    white_mcts.reset_root()

    step = 0
    last_coord = ""
    while step < max_steps:
        step += 1
        curr_p = board.current_player
        active_mcts = black_mcts if curr_p == 1 else white_mcts

        temp = 1.0 if step <= opening_random else 1e-3
        acts, probs = active_mcts.get_move_probs(board, temp=temp)
        if acts and probs:
            if temp <= 1e-2:
                move = acts[probs.index(max(probs))]
            else:
                import random
                move = random.choices(acts, weights=probs)[0]
        else:
            candidates = board.get_candidate_moves()
            move = candidates[0] if candidates else 0

        last_coord = board.coord_to_str(move)
        board.do_move(move)
        black_mcts.update_with_move(move)
        white_mcts.update_with_move(move)

        end, winner = board.game_end()
        if end:
            return winner, step, last_coord

    return -1, step, last_coord


def run_tournament(n_games=8, board_size=15, sims=150, model_name=None):
    if not HAS_TORCH:
        print("PyTorch is required for benchmark.")
        return

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if model_name:
        model_path = os.path.join(models_dir, model_name)
    else:
        v2_path = os.path.join(models_dir, f"alphazero_{board_size}x{board_size}_v2.pth")
        v1_path = os.path.join(models_dir, f"alphazero_{board_size}x{board_size}.pth")
        model_path = v2_path if os.path.exists(v2_path) else v1_path

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print("=" * 70)
    print(f"  🏆 AlphaZero vs 启发式引擎 巅峰多局锦标赛")
    print(f"  神经网络模型: {os.path.basename(model_path)}")
    print(f"  棋盘规格: {board_size}x{board_size} | 每步 MCTS 模拟: {sims} 次 | 总局数: {n_games} 局")
    print("=" * 70)

    # Initialize agents
    net = PyTorchPolicyValueNet(board_size=board_size, model_file=model_path)
    nn_mcts = MCTS(net, c_puct=2.5, n_playout=sims)

    heuristic = HeuristicPolicyValueNet()
    heu_mcts = MCTS(heuristic, c_puct=4.0, n_playout=sims)

    print(f"计算硬件: {net.device} (Metal/MPS 硬件加速激活)")
    print("\n对抗开始（黑白交替轮换先手）：\n")

    stats = {
        "nn_black_wins": 0, "nn_black_games": 0,
        "nn_white_wins": 0, "nn_white_games": 0,
        "heu_black_wins": 0, "heu_black_games": 0,
        "heu_white_wins": 0, "heu_white_games": 0,
        "draws": 0,
        "game_lengths": []
    }

    t0 = time.time()

    for i in range(1, n_games + 1):
        g_start = time.time()
        if i % 2 == 1:
            # Odd games: AlphaZero is Black (First), Heuristic is White
            print(f"[{i:02d}/{n_games:02d}] 执黑先手: AlphaZero  vs  执白后手: 启发式 ... ", end="", flush=True)
            winner, steps, last_c = play_eval_game(nn_mcts, heu_mcts, board_size)
            stats["nn_black_games"] += 1
            stats["heu_white_games"] += 1
            g_time = time.time() - g_start

            if winner == 1:
                stats["nn_black_wins"] += 1
                res = f"🎉 AlphaZero 胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)"
            elif winner == 2:
                stats["heu_white_wins"] += 1
                res = f"🛡️ 启发式 胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)"
            else:
                stats["draws"] += 1
                res = f"🤝 和棋 (手数: {steps:2d}, 耗时: {g_time:.1f}s)"
        else:
            # Even games: Heuristic is Black (First), AlphaZero is White
            print(f"[{i:02d}/{n_games:02d}] 执黑先手: 启发式  vs  执白后手: AlphaZero ... ", end="", flush=True)
            winner, steps, last_c = play_eval_game(heu_mcts, nn_mcts, board_size)
            stats["heu_black_games"] += 1
            stats["nn_white_games"] += 1
            g_time = time.time() - g_start

            if winner == 2:
                stats["nn_white_wins"] += 1
                res = f"🎉 AlphaZero 胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)"
            elif winner == 1:
                stats["heu_black_wins"] += 1
                res = f"🛡️ 启发式 胜! (手数: {steps:2d}, 绝杀点: {last_c}, 耗时: {g_time:.1f}s)"
            else:
                stats["draws"] += 1
                res = f"🤝 和棋 (手数: {steps:2d}, 耗时: {g_time:.1f}s)"

        print(res, flush=True)
        stats["game_lengths"].append(steps)

    total_time = time.time() - t0
    total_nn_wins = stats["nn_black_wins"] + stats["nn_white_wins"]
    total_heu_wins = stats["heu_black_wins"] + stats["heu_white_wins"]
    nn_winrate = (total_nn_wins / n_games) * 100.0

    print("\n" + "=" * 70)
    print(f"                       🏁 比赛结果统计报告")
    print("=" * 70)
    print(f"总耗时: {total_time:.1f} 秒 (平均每局 {(total_time/n_games):.1f} 秒)")
    print(f"平均每局手数: {sum(stats['game_lengths'])/len(stats['game_lengths']):.1f} 手")
    print("-" * 70)
    print(f"● AlphaZero 执黑 (先手): {stats['nn_black_wins']} 胜 / {stats['nn_black_games']} 局 (胜率: {(stats['nn_black_wins']/max(1, stats['nn_black_games']))*100:.1f}%)")
    print(f"○ AlphaZero 执白 (后手): {stats['nn_white_wins']} 胜 / {stats['nn_white_games']} 局 (胜率: {(stats['nn_white_wins']/max(1, stats['nn_white_games']))*100:.1f}%)")
    print("-" * 70)
    print(f"🏆 AlphaZero 总胜场: {total_nn_wins} | 启发式 总胜场: {total_heu_wins} | 和棋: {stats['draws']}")
    print(f"🔥 AlphaZero 整体综合胜率: {nn_winrate:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8, help="Number of tournament games")
    parser.add_argument("--board_size", type=int, default=15, help="Board size")
    parser.add_argument("--sims", type=int, default=150, help="MCTS simulations per move")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint model filename")
    args = parser.parse_args()

    run_tournament(n_games=args.games, board_size=args.board_size, sims=args.sims, model_name=args.model)

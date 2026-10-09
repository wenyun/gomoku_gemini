"""
Command-Line Interface (CLI) for playing Gomoku against AlphaZero.
Allows text-based terminal play as an alternative to the GUI.
"""

from board import Board
from mcts import MCTS
from policy_value_net import HeuristicPolicyValueNet


def print_board(board: Board):
    """Print the board in ASCII."""
    col_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    header = "   " + " ".join(col_letters[c] for c in range(board.width))
    print(header)
    for r in range(board.height):
        row_num = board.height - r
        row_str = f"{row_num:2d} "
        for c in range(board.width):
            m = board.location_to_move(r, c)
            st = board.states.get(m, 0)
            if st == 1:
                row_str += " ●"
            elif st == 2:
                row_str += " ○"
            else:
                row_str += " ┼"
        print(row_str)
    print()


def play_cli():
    print("=" * 50)
    print("      AlphaZero 五子棋 (终端模式 / CLI)")
    print("=" * 50)
    print("规则: 黑方先手(●), 白方后手(○), 5子连线获胜。")
    print("输入坐标例如: H8, G7 等。输入 'q' 退出。\n")

    board_size = 15
    board = Board(width=board_size, height=board_size)
    evaluator = HeuristicPolicyValueNet()
    mcts = MCTS(evaluator, c_puct=4.0, n_playout=400)

    # Choose color
    choice = input("请选择你的棋子颜色 [1: 黑方先手, 2: 白方后手] (默认1): ").strip()
    human_player = 2 if choice == "2" else 1

    print(f"\n对局开始！你执{'黑(●)' if human_player == 1 else '白(○)'}。")
    print_board(board)

    while True:
        curr_p = board.current_player
        if curr_p == human_player:
            # Human turn
            while True:
                user_input = input("请输入你的落子坐标 (如 H8): ").strip().upper()
                if user_input in ('Q', 'QUIT', 'EXIT'):
                    print("对局退出。")
                    return
                try:
                    col_char = user_input[0]
                    row_num = int(user_input[1:])
                    col_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
                    col = col_letters.index(col_char)
                    row = board.height - row_num
                    move = board.location_to_move(row, col)
                    if move in board.states:
                        print("该位置已有棋子，请重新输入！")
                        continue
                    break
                except Exception:
                    print("无效坐标格式！请输入如 H8 的格式。")

            board.do_move(move)
            mcts.update_with_move(move)
        else:
            # AI turn
            print("AlphaZero AI 正在思考中 (MCTS 搜索)...")
            acts, probs = mcts.get_move_probs(board, temp=1e-3)
            move = acts[probs.index(max(probs))]
            coord_str = board.coord_to_str(move)
            print(f"AI 落子: {coord_str}")
            board.do_move(move)
            mcts.update_with_move(move)

        print_board(board)

        # Check win/draw
        has_win, winner, _ = board.has_a_winner()
        if has_win:
            winner_name = "黑方(●)" if winner == 1 else "白方(○)"
            identity = "玩家" if winner == human_player else "AlphaZero AI"
            print(f"\n🎉 恭喜！{winner_name} {identity} 获胜！五子连珠！")
            break

        is_end, _ = board.game_end()
        if is_end:
            print("\n🤝 和棋！棋盘已满。")
            break


if __name__ == "__main__":
    play_cli()

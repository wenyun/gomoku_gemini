"""
AlphaZero Gomoku Graphical User Interface (Tkinter).
Features:
- High-resolution board with Kaya wood tone, coordinates, star points, and 3D stones
- Real-time KataGo/AlphaGo style win rate meter
- MCTS background threading with thinking animations
- Move number toggles, undo, AI hints, and AI-vs-AI demo
- Board size configuration (15x15, 11x11, 9x9)
- Engine selection (Heuristic Zero-Shot vs PyTorch Deep Net)
"""

import math
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple, Dict

from board import Board
from mcts import MCTS
from policy_value_net import HeuristicPolicyValueNet, HAS_TORCH

if HAS_TORCH:
    from policy_value_net import PyTorchPolicyValueNet


class GomokuGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AlphaZero 五子棋 (Gomoku AI)")
        self.root.geometry("1060x820")
        self.root.minsize(920, 720)
        self.root.configure(bg="#2B2D42")

        # Game State
        self.board_size = 15
        self.human_player = 1  # 1: Black (first), 2: White (second)
        self.ai_playouts = 500
        self.engine_type = "torch" if HAS_TORCH else "heuristic"
        self.show_move_numbers = False
        self.ai_hints: List[Dict] = []
        self.is_ai_thinking = False
        self.is_ai_vs_ai = False
        self.hover_pos: Optional[Tuple[int, int]] = None
        self.winning_line: List[Tuple[int, int]] = []
        self.game_over = False

        # Initialize Board and Evaluator
        self.board = Board(width=self.board_size, height=self.board_size, n_in_row=5)
        self.init_ai_engine()

        # Canvas Dimensions
        self.canvas_size = 680
        self.margin = 38
        self._calc_cell_size()

        # Build UI layout
        self._setup_styles()
        self._build_widgets()
        self._update_device_label()

        # Canvas event bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Draw initial board
        self._redraw_board()
        self._update_status()

    def _calc_cell_size(self):
        """Calculate grid cell size based on board size and canvas size."""
        self.cell_size = (self.canvas_size - 2 * self.margin) / (self.board_size - 1)

    def init_ai_engine(self):
        """Initialize the selected AI evaluator and MCTS engine."""
        if self.engine_type == "torch" and HAS_TORCH:
            try:
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                v2_path = os.path.join(models_dir, f"alphazero_{self.board_size}x{self.board_size}_v2.pth")
                v1_path = os.path.join(models_dir, f"alphazero_{self.board_size}x{self.board_size}.pth")
                model_path = v2_path if os.path.exists(v2_path) else v1_path

                if os.path.exists(model_path):
                    self.evaluator = PyTorchPolicyValueNet(board_size=self.board_size, model_file=model_path)
                else:
                    self.evaluator = PyTorchPolicyValueNet(board_size=self.board_size)
                dev_name = "Apple M5 GPU (MPS加速)" if getattr(self.evaluator, 'device', None) and self.evaluator.device.type == "mps" else "PyTorch"
                if hasattr(self, "device_info_label"):
                    self.device_info_label.config(text=f"计算硬件：{dev_name}", fg="#48CAE4")
            except Exception as e:
                print(f"Failed to load PyTorch engine ({e}), falling back to Heuristic")
                self.evaluator = HeuristicPolicyValueNet()
                if hasattr(self, "device_info_label"):
                    self.device_info_label.config(text="计算引擎：启发式战术引擎 (CPU)", fg="#06D6A0")
        else:
            self.evaluator = HeuristicPolicyValueNet()
            if hasattr(self, "device_info_label"):
                self.device_info_label.config(text="计算引擎：启发式战术引擎 (CPU)", fg="#06D6A0")

        self.mcts = MCTS(self.evaluator, c_puct=4.0, n_playout=self.ai_playouts)
        self._update_device_label()

    def _update_device_label(self):
        """Update hardware acceleration label based on active engine."""
        if hasattr(self, "device_info_label"):
            if self.engine_type == "torch" and getattr(self, "evaluator", None):
                dev_name = "Apple M5 GPU (MPS加速)" if getattr(self.evaluator, 'device', None) and self.evaluator.device.type == "mps" else "PyTorch"
                self.device_info_label.config(text=f"计算硬件：{dev_name}", fg="#48CAE4")
            else:
                self.device_info_label.config(text="计算引擎：启发式战术引擎 (CPU)", fg="#06D6A0")

    def _setup_styles(self):
        """Configure modern dark ttk styles."""
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#2B2D42", foreground="#EDF2F4")
        self.style.configure("TLabel", background="#2B2D42", foreground="#EDF2F4", font=("Arial", 11))
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), foreground="#FFD166")
        self.style.configure("SubHeader.TLabel", font=("Arial", 12, "bold"), foreground="#8D99AE")
        self.style.configure("TButton", font=("Arial", 11, "bold"), padding=6)
        self.style.configure("Accent.TButton", background="#EF233C", foreground="#FFFFFF")
        self.style.configure("TCombobox", fieldbackground="#1E1F29", background="#2B2D42")

    def _build_widgets(self):
        """Construct the GUI layout."""
        main_container = tk.Frame(self.root, bg="#1E1F29")
        main_container.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Left Column: Go Board Canvas & Win Rate Bar
        left_frame = tk.Frame(main_container, bg="#1E1F29")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        # Win Rate Bar at top of board
        self.winrate_frame = tk.Frame(left_frame, bg="#1E1F29", height=32)
        self.winrate_frame.pack(fill=tk.X, pady=(0, 8))

        self.winrate_label_black = tk.Label(
            self.winrate_frame, text="黑方胜率 50.0%", font=("Arial", 10, "bold"),
            bg="#1E1F29", fg="#EDF2F4"
        )
        self.winrate_label_black.pack(side=tk.LEFT)

        self.winrate_label_white = tk.Label(
            self.winrate_frame, text="白方胜率 50.0%", font=("Arial", 10, "bold"),
            bg="#1E1F29", fg="#EDF2F4"
        )
        self.winrate_label_white.pack(side=tk.RIGHT)

        self.winrate_canvas = tk.Canvas(left_frame, height=14, bg="#333", highlightthickness=0)
        self.winrate_canvas.pack(fill=tk.X, pady=(0, 10))

        # Main Board Canvas
        canvas_border = tk.Frame(left_frame, bg="#5c4018", padx=4, pady=4, relief=tk.RAISED, bd=2)
        canvas_border.pack()
        self.canvas = tk.Canvas(
            canvas_border, width=self.canvas_size, height=self.canvas_size,
            bg="#DEB887", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack()

        # Right Column: Control & Status Panel
        right_panel = tk.Frame(main_container, bg="#2B2D42", width=340, padx=18, pady=16)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)

        # Title
        title_label = tk.Label(
            right_panel, text="AlphaZero 五子棋", font=("Arial", 18, "bold"),
            bg="#2B2D42", fg="#FFD166"
        )
        title_label.pack(anchor="w", pady=(0, 4))

        subtitle_label = tk.Label(
            right_panel, text="蒙特卡洛树搜索 + 深度/启发策略价值网络", font=("Arial", 9),
            bg="#2B2D42", fg="#8D99AE"
        )
        subtitle_label.pack(anchor="w", pady=(0, 16))

        # Status Card
        status_box = tk.LabelFrame(
            right_panel, text=" 对局状态 ", font=("Arial", 10, "bold"),
            bg="#1E1F29", fg="#FFD166", padx=12, pady=10, bd=1, relief=tk.SOLID
        )
        status_box.pack(fill=tk.X, pady=(0, 14))

        self.turn_indicator_label = tk.Label(
            status_box, text="轮到：黑方 (玩家)", font=("Arial", 12, "bold"),
            bg="#1E1F29", fg="#48CAE4"
        )
        self.turn_indicator_label.pack(anchor="w", pady=2)

        self.step_info_label = tk.Label(
            status_box, text="手数：0 手   |   最近落子：无", font=("Arial", 10),
            bg="#1E1F29", fg="#EDF2F4"
        )
        self.step_info_label.pack(anchor="w", pady=2)

        self.thinking_label = tk.Label(
            status_box, text="AI状态：待命中", font=("Arial", 10, "italic"),
            bg="#1E1F29", fg="#8D99AE"
        )
        self.thinking_label.pack(anchor="w", pady=2)

        self.device_info_label = tk.Label(
            status_box, text="计算引擎：启发式战术引擎 (CPU)", font=("Arial", 9),
            bg="#1E1F29", fg="#06D6A0"
        )
        self.device_info_label.pack(anchor="w", pady=2)

        # Settings Box
        settings_box = tk.LabelFrame(
            right_panel, text=" 游戏设置 ", font=("Arial", 10, "bold"),
            bg="#1E1F29", fg="#FFD166", padx=12, pady=10, bd=1, relief=tk.SOLID
        )
        settings_box.pack(fill=tk.X, pady=(0, 14))

        # Board Size Selection
        size_frame = tk.Frame(settings_box, bg="#1E1F29")
        size_frame.pack(fill=tk.X, pady=4)
        tk.Label(size_frame, text="棋盘规格：", bg="#1E1F29", fg="#EDF2F4", font=("Arial", 10)).pack(side=tk.LEFT)
        self.size_var = tk.StringVar(value="15x15 (标准)")
        self.size_combo = ttk.Combobox(
            size_frame, textvariable=self.size_var,
            values=["15x15 (标准)", "11x11 (快棋)", "9x9 (微棋盘)"],
            state="readonly", width=14
        )
        self.size_combo.pack(side=tk.RIGHT)
        self.size_combo.bind("<<ComboboxSelected>>", self._on_size_change)

        # Player Color Selection
        color_frame = tk.Frame(settings_box, bg="#1E1F29")
        color_frame.pack(fill=tk.X, pady=4)
        tk.Label(color_frame, text="执子选择：", bg="#1E1F29", fg="#EDF2F4", font=("Arial", 10)).pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="执黑 (先手)")
        self.color_combo = ttk.Combobox(
            color_frame, textvariable=self.color_var,
            values=["执黑 (先手)", "执白 (后手)"],
            state="readonly", width=14
        )
        self.color_combo.pack(side=tk.RIGHT)
        self.color_combo.bind("<<ComboboxSelected>>", self._on_color_change)

        # AI Engine Selection
        engine_frame = tk.Frame(settings_box, bg="#1E1F29")
        engine_frame.pack(fill=tk.X, pady=4)
        engine_options = []
        if HAS_TORCH:
            engine_options.append("AlphaZero V2 (本地训练)")
        engine_options.append("AlphaZero启发式策略")
        self.engine_var = tk.StringVar(value="AlphaZero V2 (本地训练)" if HAS_TORCH else "AlphaZero启发式策略")
        self.engine_combo = ttk.Combobox(
            engine_frame, textvariable=self.engine_var,
            values=engine_options, state="readonly", width=16
        )
        self.engine_combo.pack(side=tk.RIGHT)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)

        # Search Depth / Playouts
        playout_frame = tk.Frame(settings_box, bg="#1E1F29")
        playout_frame.pack(fill=tk.X, pady=4)
        tk.Label(playout_frame, text="AI思考深度：", bg="#1E1F29", fg="#EDF2F4", font=("Arial", 10)).pack(side=tk.LEFT)
        self.playout_var = tk.StringVar(value="均衡 (500次推演)")
        self.playout_combo = ttk.Combobox(
            playout_frame, textvariable=self.playout_var,
            values=["快速 (200次推演)", "均衡 (500次推演)", "大师 (1200次推演)"],
            state="readonly", width=14
        )
        self.playout_combo.pack(side=tk.RIGHT)
        self.playout_combo.bind("<<ComboboxSelected>>", self._on_playout_change)

        # Action Buttons Box
        btn_box = tk.Frame(right_panel, bg="#2B2D42")
        btn_box.pack(fill=tk.X, pady=(0, 10))

        new_game_btn = tk.Button(
            btn_box, text="🔄 新对局", font=("Arial", 11, "bold"),
            bg="#06D6A0", fg="#000000", activebackground="#05B386",
            command=self.start_new_game, relief=tk.FLAT, pady=6, cursor="hand2"
        )
        new_game_btn.pack(fill=tk.X, pady=3)

        undo_btn = tk.Button(
            btn_box, text="↩️ 悔棋", font=("Arial", 11, "bold"),
            bg="#118AB2", fg="#FFFFFF", activebackground="#0D6B8A",
            command=self.undo_move, relief=tk.FLAT, pady=6, cursor="hand2"
        )
        undo_btn.pack(fill=tk.X, pady=3)

        hint_btn = tk.Button(
            btn_box, text="💡 AI 支招 (显示候选落点)", font=("Arial", 11, "bold"),
            bg="#FFD166", fg="#000000", activebackground="#E0B750",
            command=self.request_ai_hint, relief=tk.FLAT, pady=6, cursor="hand2"
        )
        hint_btn.pack(fill=tk.X, pady=3)

        self.ai_vs_ai_btn = tk.Button(
            btn_box, text="🤖 AI 自对弈演示", font=("Arial", 11, "bold"),
            bg="#8338EC", fg="#FFFFFF", activebackground="#6A2CBF",
            command=self.toggle_ai_vs_ai, relief=tk.FLAT, pady=6, cursor="hand2"
        )
        self.ai_vs_ai_btn.pack(fill=tk.X, pady=3)

        # Checkbuttons
        check_frame = tk.Frame(right_panel, bg="#2B2D42")
        check_frame.pack(fill=tk.X, pady=6)

        self.move_num_var = tk.BooleanVar(value=False)
        move_num_cb = tk.Checkbutton(
            check_frame, text="显示落子手数序号", variable=self.move_num_var,
            bg="#2B2D42", fg="#EDF2F4", selectcolor="#1E1F29", activebackground="#2B2D42",
            font=("Arial", 10), command=self._toggle_move_numbers
        )
        move_num_cb.pack(anchor="w")

        # Hint Box at bottom
        self.hint_text_label = tk.Label(
            right_panel, text="提示：点击棋盘交叉点落子。\nAlphaZero 通过 MCTS 蒙特卡洛树\n自主推演最佳应对方案。",
            font=("Arial", 9), bg="#2B2D42", fg="#8D99AE", justify=tk.LEFT
        )
        self.hint_text_label.pack(anchor="w", pady=(10, 0))

    # =========================================================================
    # Board Rendering
    # =========================================================================

    def _redraw_board(self):
        """Draw the entire Go board: wood background, grid lines, star points, stones."""
        self.canvas.delete("all")
        self._calc_cell_size()

        # Draw Wood Board Background Gradient / Border
        self.canvas.create_rectangle(
            0, 0, self.canvas_size, self.canvas_size,
            fill="#D2A054", outline="#B38838", width=3
        )
        # Subtle inner wood glow
        self.canvas.create_rectangle(
            self.margin - 10, self.margin - 10,
            self.canvas_size - self.margin + 10, self.canvas_size - self.margin + 10,
            outline="#E8C37A", width=1
        )

        col_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

        # Draw Grid Lines and Coordinate Labels
        for i in range(self.board_size):
            pos = self.margin + i * self.cell_size

            # Horizontal Line
            self.canvas.create_line(
                self.margin, pos, self.canvas_size - self.margin, pos,
                fill="#4A3525", width=1.4
            )
            # Vertical Line
            self.canvas.create_line(
                pos, self.margin, pos, self.canvas_size - self.margin,
                fill="#4A3525", width=1.4
            )

            # Coordinate Labels
            # Left & Right Row Numbers (1-15 from bottom to top)
            row_label = str(self.board_size - i)
            self.canvas.create_text(
                self.margin - 18, pos, text=row_label,
                fill="#5C4018", font=("Arial", 9, "bold")
            )
            self.canvas.create_text(
                self.canvas_size - self.margin + 18, pos, text=row_label,
                fill="#5C4018", font=("Arial", 9, "bold")
            )

            # Top & Bottom Column Letters (A-O)
            col_label = col_letters[i] if i < len(col_letters) else chr(ord('A') + i)
            self.canvas.create_text(
                pos, self.margin - 18, text=col_label,
                fill="#5C4018", font=("Arial", 9, "bold")
            )
            self.canvas.create_text(
                pos, self.canvas_size - self.margin + 18, text=col_label,
                fill="#5C4018", font=("Arial", 9, "bold")
            )

        # Draw Star Points (星位与天元)
        star_points = self._get_star_points()
        for r, c in star_points:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            self.canvas.create_oval(
                x - 3.5, y - 3.5, x + 3.5, y + 3.5,
                fill="#4A3525", outline="#4A3525"
            )

        # Draw Stones
        radius = self.cell_size * 0.44
        for idx, (move, player) in enumerate(self.board.move_history, 1):
            r, c = self.board.move_to_location(move)
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            self._draw_stone(x, y, radius, player, move_number=idx if self.show_move_numbers else None)

        # Draw Last Move Indicator (Red / Gold Dot)
        if self.board.last_move is not None:
            lr, lc = self.board.move_to_location(self.board.last_move)
            lx = self.margin + lc * self.cell_size
            ly = self.margin + lr * self.cell_size
            last_player = self.board.states.get(self.board.last_move, 1)
            dot_color = "#FF3366" if last_player == 1 else "#CC0033"
            self.canvas.create_oval(
                lx - 4, ly - 4, lx + 4, ly + 4,
                fill=dot_color, outline="#FFFFFF", width=1.2
            )

        # Draw Winning Line Highlight
        if self.winning_line and len(self.winning_line) >= 2:
            self._draw_winning_line()

        # Draw AI Hint Markers
        if self.ai_hints:
            self._draw_ai_hints()

        # Draw Hover Preview Stone
        if self.hover_pos and not self.game_over and not self.is_ai_thinking:
            hr, hc = self.hover_pos
            hmove = self.board.location_to_move(hr, hc)
            if hmove not in self.board.states:
                hx = self.margin + hc * self.cell_size
                hy = self.margin + hr * self.cell_size
                ghost_color = "#000000" if self.board.current_player == 1 else "#FFFFFF"
                self.canvas.create_oval(
                    hx - radius * 0.8, hy - radius * 0.8,
                    hx + radius * 0.8, hy + radius * 0.8,
                    outline=ghost_color, fill="", width=2, dash=(4, 4)
                )

        # Update Win Rate Bar
        self._draw_winrate_bar()

    def _get_star_points(self) -> List[Tuple[int, int]]:
        """Return star point (r, c) coordinates for current board size."""
        if self.board_size == 15:
            return [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        elif self.board_size == 11:
            return [(2, 2), (2, 8), (5, 5), (8, 2), (8, 8)]
        elif self.board_size == 9:
            return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        return [(self.board_size // 2, self.board_size // 2)]

    def _draw_stone(self, x: float, y: float, r: float, player: int, move_number: Optional[int] = None):
        """Draw realistic 3D Go stone with shadow and reflection highlight."""
        # Drop Shadow
        self.canvas.create_oval(
            x - r + 2.5, y - r + 3.5, x + r + 2.5, y + r + 3.5,
            fill="#7A5826", outline="", stipple="gray50"
        )

        if player == 1:
            # Black Stone: Dark base with subtle gloss highlight
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="#1C1D21", outline="#000000", width=1
            )
            # Top-left reflection highlight
            self.canvas.create_oval(
                x - r * 0.65, y - r * 0.65, x - r * 0.15, y - r * 0.15,
                fill="#50535E", outline=""
            )
            num_color = "#E0E0E0"
        else:
            # White Stone: Crisp white base with subtle light rim
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="#FAFAFA", outline="#C8C8C8", width=1.2
            )
            # Specular glow
            self.canvas.create_oval(
                x - r * 0.65, y - r * 0.65, x - r * 0.2, y - r * 0.2,
                fill="#FFFFFF", outline=""
            )
            num_color = "#333333"

        # Draw Move Number if enabled
        if move_number is not None:
            font_size = max(7, int(r * 0.75))
            self.canvas.create_text(
                x, y, text=str(move_number),
                fill=num_color, font=("Arial", font_size, "bold")
            )

    def _draw_winning_line(self):
        """Draw thick glowing line connecting the 5 winning stones."""
        coords = []
        for r, c in self.winning_line:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            coords.extend([x, y])

        if len(coords) >= 4:
            # Glowing border
            self.canvas.create_line(*coords, fill="#FFD166", width=7, capstyle=tk.ROUND)
            self.canvas.create_line(*coords, fill="#EF233C", width=3.5, capstyle=tk.ROUND)

    def _draw_ai_hints(self):
        """Draw top candidate moves recommended by AI with rank and win rate."""
        colors = ["#06D6A0", "#118AB2", "#FFD166"]
        radius = self.cell_size * 0.42

        for idx, hint in enumerate(self.ai_hints):
            r, c = self.board.move_to_location(hint["action"])
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            col = colors[idx % len(colors)]

            # Candidate ring
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline=col, width=3, dash=(3, 2)
            )
            # Label
            label = f"#{idx + 1}\n{hint['win_rate']:.0f}%"
            self.canvas.create_text(
                x, y, text=label, fill=col, font=("Arial", 8, "bold")
            )

    def _draw_winrate_bar(self):
        """Draw the KataGo-style Win Rate bar (Black vs White)."""
        w = self.winrate_canvas.winfo_width()
        if w <= 1:
            w = self.canvas_size

        # Retrieve root evaluation from MCTS
        eval_v = self.mcts.get_root_eval()
        # eval_v is from perspective of current player
        # Convert to Black win rate
        if self.board.current_player == 1:
            black_win_rate = (eval_v + 1.0) / 2.0 * 100.0
        else:
            black_win_rate = (1.0 - (eval_v + 1.0) / 2.0) * 100.0

        # Clamp between 1% and 99% for visual aesthetics
        black_win_rate = max(1.0, min(99.0, black_win_rate))
        white_win_rate = 100.0 - black_win_rate

        self.winrate_label_black.config(text=f"黑方胜率 {black_win_rate:.1f}%")
        self.winrate_label_white.config(text=f"白方胜率 {white_win_rate:.1f}%")

        self.winrate_canvas.delete("all")
        split_x = (black_win_rate / 100.0) * w

        # Black portion (dark)
        self.winrate_canvas.create_rectangle(0, 0, split_x, 14, fill="#1A1A1A", outline="")
        # White portion (light)
        self.winrate_canvas.create_rectangle(split_x, 0, w, 14, fill="#E0E0E0", outline="")
        # Divider line
        self.winrate_canvas.create_line(split_x, 0, split_x, 14, fill="#EF233C", width=2)

    # =========================================================================
    # Game Logic & Event Handlers
    # =========================================================================

    def _on_canvas_click(self, event):
        """Handle human player click on board."""
        if self.game_over or self.is_ai_thinking or self.is_ai_vs_ai:
            return

        # Check if it is human's turn
        if self.board.current_player != self.human_player:
            return

        # Snap to nearest intersection
        c = round((event.x - self.margin) / self.cell_size)
        r = round((event.y - self.margin) / self.cell_size)

        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return

        # Check distance threshold
        target_x = self.margin + c * self.cell_size
        target_y = self.margin + r * self.cell_size
        dist = math.hypot(event.x - target_x, event.y - target_y)
        if dist > self.cell_size * 0.45:
            return

        move = self.board.location_to_move(r, c)
        if move in self.board.states:
            return

        # Clear hints on move
        self.ai_hints = []

        # Execute Human Move
        self.execute_move(move)

        # Trigger AI move if game continues
        if not self.game_over and self.board.current_player != self.human_player:
            self.trigger_ai_move()

    def _on_canvas_motion(self, event):
        """Track mouse for ghost hover preview stone."""
        c = round((event.x - self.margin) / self.cell_size)
        r = round((event.y - self.margin) / self.cell_size)
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            target_x = self.margin + c * self.cell_size
            target_y = self.margin + r * self.cell_size
            if math.hypot(event.x - target_x, event.y - target_y) <= self.cell_size * 0.45:
                if self.hover_pos != (r, c):
                    self.hover_pos = (r, c)
                    self._redraw_board()
                return

        if self.hover_pos is not None:
            self.hover_pos = None
            self._redraw_board()

    def _on_canvas_leave(self, event):
        """Clear ghost stone on mouse leave."""
        if self.hover_pos is not None:
            self.hover_pos = None
            self._redraw_board()

    def execute_move(self, move: int):
        """Execute a move on the board, update MCTS, and check for win/draw."""
        player = self.board.current_player
        self.board.do_move(move)
        self.mcts.update_with_move(move)

        # Check game end
        has_win, winner, line = self.board.has_a_winner()
        if has_win:
            self.game_over = True
            self.winning_line = line
            winner_str = "黑方 (执黑)" if winner == 1 else "白方 (执白)"
            self._redraw_board()
            self._update_status()
            messagebox.showinfo("对局结束", f"🎉 {winner_str} 获胜！五子连珠！")
            return

        is_end, winner = self.board.game_end()
        if is_end:
            self.game_over = True
            self._redraw_board()
            self._update_status()
            messagebox.showinfo("对局结束", "🤝 和棋！棋盘已满。")
            return

        self._redraw_board()
        self._update_status()

    def trigger_ai_move(self):
        """Start MCTS search for AI move in a background thread."""
        if self.game_over or self.is_ai_thinking:
            return

        self.is_ai_thinking = True
        self.thinking_label.config(text="AI状态：思考中 (MCTS推演)...", fg="#FFD166")
        self.root.config(cursor="watch")

        thread = threading.Thread(target=self._ai_worker, daemon=True)
        thread.start()

    def _ai_worker(self):
        """Worker thread for running MCTS simulations."""
        try:
            time_start = time.time()
            # Run MCTS with deterministic temperature
            acts, probs = self.mcts.get_move_probs(self.board, temp=1e-3)
            elapsed = time.time() - time_start

            if acts and probs:
                best_move = acts[probs.index(max(probs))]
            else:
                candidates = self.board.get_candidate_moves()
                best_move = candidates[0] if candidates else 0

            # Schedule UI update in main thread
            self.root.after(0, self._ai_move_completed, best_move, elapsed)
        except Exception as e:
            print(f"Error in AI search thread: {e}")
            self.root.after(0, self._ai_move_failed)

    def _ai_move_completed(self, move: int, elapsed: float):
        """Callback when AI finishes thinking."""
        self.is_ai_thinking = False
        self.root.config(cursor="")
        self.thinking_label.config(text=f"AI状态：待命中 (上步用时 {elapsed:.2f}s)", fg="#8D99AE")

        self.execute_move(move)

        # If in AI vs AI demo mode, continue next move
        if self.is_ai_vs_ai and not self.game_over:
            self.root.after(400, self.trigger_ai_move)

    def _ai_move_failed(self):
        """Handle AI search failure gracefully."""
        self.is_ai_thinking = False
        self.root.config(cursor="")
        self.thinking_label.config(text="AI状态：搜索异常", fg="#EF233C")

    def _update_status(self):
        """Update status card text."""
        curr_p = self.board.current_player
        curr_name = "黑方" if curr_p == 1 else "白方"
        identity = " (玩家)" if curr_p == self.human_player else " (AI)"
        if self.is_ai_vs_ai:
            identity = " (AI)"

        color_hex = "#48CAE4" if curr_p == 1 else "#FFD166"
        self.turn_indicator_label.config(
            text=f"轮到：{curr_name}{identity}",
            fg=color_hex
        )

        n_moves = len(self.board.move_history)
        if self.board.last_move is not None:
            last_p = self.board.move_history[-1][1]
            last_p_str = "黑方" if last_p == 1 else "白方"
            coord_str = self.board.coord_to_str(self.board.last_move)
            last_move_text = f"{last_p_str}落子: {coord_str}"
        else:
            last_move_text = "无"

        self.step_info_label.config(
            text=f"手数：{n_moves} 手   |   最近落子：{last_move_text}"
        )

    # =========================================================================
    # Controls & Toolbar Actions
    # =========================================================================

    def start_new_game(self):
        """Start a brand new game."""
        self.game_over = False
        self.winning_line = []
        self.ai_hints = []
        self.is_ai_thinking = False
        self.is_ai_vs_ai = False
        self.ai_vs_ai_btn.config(text="🤖 AI 自对弈演示", bg="#8338EC")

        self.board.init_board(start_player=1)
        self.mcts.reset_root()

        self._redraw_board()
        self._update_status()

        # If human is White (second), AI makes first move as Black
        if self.human_player == 2:
            self.trigger_ai_move()

    def undo_move(self):
        """Undo last move (or pair of moves so it's always player's turn)."""
        if self.is_ai_thinking or not self.board.move_history:
            return

        self.game_over = False
        self.winning_line = []
        self.ai_hints = []

        # If human's turn: undo AI's previous move and human's previous move
        if self.board.current_player == self.human_player:
            self.board.undo_move()
            self.board.undo_move()
        else:
            # AI's turn: just undo human's last move
            self.board.undo_move()

        # Reset MCTS root to match current board state
        self.mcts.reset_root()
        self._redraw_board()
        self._update_status()

    def request_ai_hint(self):
        """Request top 3 recommended moves from AlphaZero MCTS."""
        if self.game_over or self.is_ai_thinking:
            return

        self.thinking_label.config(text="AI状态：正在分析局面推荐着法...", fg="#FFD166")
        self.root.config(cursor="watch")

        def _hint_worker():
            # Run quick MCTS playouts for analysis
            self.mcts.get_move_probs(self.board, temp=1.0)
            top_moves = self.mcts.get_top_moves(self.board, top_k=3)
            self.root.after(0, self._apply_hints, top_moves)

        threading.Thread(target=_hint_worker, daemon=True).start()

    def _apply_hints(self, top_moves: List[Dict]):
        """Render AI hint markers and show details in status."""
        self.root.config(cursor="")
        self.thinking_label.config(text="AI状态：待命中", fg="#8D99AE")
        self.ai_hints = top_moves
        self._redraw_board()

        if top_moves:
            summary = " | ".join([f"#{i+1} {m['coord']} ({m['win_rate']:.0f}%)" for i, m in enumerate(top_moves)])
            self.hint_text_label.config(text=f"AI推荐着法:\n{summary}")

    def toggle_ai_vs_ai(self):
        """Toggle AI vs AI self-play demonstration mode."""
        if self.is_ai_vs_ai:
            self.is_ai_vs_ai = False
            self.ai_vs_ai_btn.config(text="🤖 AI 自对弈演示", bg="#8338EC")
        else:
            self.is_ai_vs_ai = True
            self.ai_vs_ai_btn.config(text="⏹️ 停止自对弈", bg="#EF233C")
            if not self.game_over and not self.is_ai_thinking:
                self.trigger_ai_move()

    def _toggle_move_numbers(self):
        """Toggle displaying move sequence numbers on stones."""
        self.show_move_numbers = self.move_num_var.get()
        self._redraw_board()

    def _on_size_change(self, event=None):
        """Handle board size change from dropdown."""
        val = self.size_var.get()
        if "15x15" in val:
            new_size = 15
        elif "11x11" in val:
            new_size = 11
        elif "9x9" in val:
            new_size = 9
        else:
            new_size = 15

        if new_size != self.board_size:
            self.board_size = new_size
            self.board = Board(width=new_size, height=new_size, n_in_row=5)
            self.init_ai_engine()
            self.start_new_game()

    def _on_color_change(self, event=None):
        """Handle player piece color selection."""
        val = self.color_var.get()
        self.human_player = 1 if "执黑" in val else 2
        self.start_new_game()

    def _on_engine_change(self, event=None):
        """Handle engine switch."""
        val = self.engine_var.get()
        if ("V2" in val or "PyTorch" in val) and HAS_TORCH:
            self.engine_type = "torch"
        else:
            self.engine_type = "heuristic"
        self.init_ai_engine()

    def _on_playout_change(self, event=None):
        """Handle playout simulation count selection."""
        val = self.playout_var.get()
        if "快速" in val:
            self.ai_playouts = 200
        elif "均衡" in val:
            self.ai_playouts = 500
        elif "大师" in val:
            self.ai_playouts = 1200
        self.mcts._n_playout = self.ai_playouts


def launch_gui():
    """Main entry point to launch the Gomoku GUI."""
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

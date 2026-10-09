"""
AlphaZero Gomoku Main Entry Point.
Launches the Desktop Graphical User Interface (GUI) by default,
or interactive CLI mode with `--cli`.
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Gomoku (五子棋)")
    parser.add_argument("--cli", action="store_true", help="Launch in command-line terminal mode instead of GUI")
    args = parser.parse_args()

    if args.cli:
        from play_cli import play_cli
        play_cli()
    else:
        try:
            from gui import launch_gui
            launch_gui()
        except Exception as e:
            print(f"Failed to start GUI ({e}). Falling back to CLI mode...")
            from play_cli import play_cli
            play_cli()


if __name__ == "__main__":
    main()

"""
Plot AlphaZero training loss curve and game statistics.
Generates loss_curve.png in ~/dev/gomoku_gemini.
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_and_plot(log_file='train_15x15.log', output_file='loss_curve.png'):
    with open(log_file, 'r') as f:
        text = f.read()

    # Extract policy and value losses (support both v1 and v2 formats)
    loss_matches = re.findall(r'(?:Policy|策略)Loss:?\s*([\d\.]+)[,\s\|]+(?:Value|价值)Loss:?\s*([\d\.]+)', text)
    if not loss_matches:
        print("No loss data found in log.")
        return

    policy_losses = [float(p) for p, _ in loss_matches]
    value_losses = [float(v) for _, v in loss_matches]
    iterations = list(range(1, len(policy_losses) + 1))

    # Extract game lengths
    game_matches = re.findall(r'finished in (\d+) moves', text)
    game_lengths = [int(m) for m in game_matches]

    # Calculate rolling averages (window = 20)
    window = 20
    def moving_avg(data, w):
        return [sum(data[max(0, i-w):i+1])/len(data[max(0, i-w):i+1]) for i in range(len(data))]

    smooth_policy = moving_avg(policy_losses, window)
    smooth_value = moving_avg(value_losses, window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: Policy Loss
    axes[0].plot(iterations, policy_losses, color='#888888', alpha=0.35, label='Raw Policy Loss')
    axes[0].plot(iterations, smooth_policy, color='#E63946', linewidth=2.0, label='Smoothed Policy Loss (Moving Avg)')
    axes[0].set_title('AlphaZero 15x15 Policy Loss Progression (MCTS Distillation)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # Subplot 2: Value Loss
    axes[1].plot(iterations, value_losses, color='#888888', alpha=0.35, label='Raw Value Loss')
    axes[1].plot(iterations, smooth_value, color='#1D3557', linewidth=2.0, label='Smoothed Value Loss (MSE)')
    axes[1].set_title('AlphaZero 15x15 Value Loss Progression (Game Outcome Prediction)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Mean Squared Error', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Loss plot successfully saved to {output_file}")


if __name__ == "__main__":
    parse_and_plot()

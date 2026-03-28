"""
으뜸 딥러닝 — 13장 05절
보상 곡선 시각화
"""

def moving_average(data, window=20):
    """Compute moving average with given window size."""
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window

plt.figure(figsize=(8, 4))
plt.plot(rewards_history, alpha=0.3, color="steelblue",
         label="Episode reward")
ma = moving_average(rewards_history, window=20)
plt.plot(range(19, len(rewards_history)), ma,
         color="darkorange", linewidth=2,
         label="Moving avg (20)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.show()

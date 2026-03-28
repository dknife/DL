"""
으뜸 딥러닝 — 13장 05절
CartPole DQN 학습 루프
"""

num_episodes = 300
batch_size = 64
gamma = 0.99
eps_start, eps_end, eps_decay = 1.0, 0.01, 0.995
target_update = 10         # sync target net every N episodes
lr = 1e-3

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
epsilon = eps_start
rewards_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = \
            env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state,
                     float(done))
        state = next_state
        total_reward += reward
        update(batch_size, gamma)
        if done:
            break

    epsilon = max(eps_end, epsilon * eps_decay)
    rewards_history.append(total_reward)

    # Sync target network
    if (episode + 1) % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 50 == 0:
        avg = np.mean(rewards_history[-50:])
        print(f"Episode {episode+1:3d}, "
              f"Avg Reward: {avg:.1f}, Eps: {epsilon:.3f}")
# Episode  50, Avg Reward:  22.4, Eps: 0.780
# Episode 100, Avg Reward:  45.7, Eps: 0.608
# Episode 150, Avg Reward: 132.5, Eps: 0.474
# Episode 200, Avg Reward: 312.8, Eps: 0.369
# Episode 250, Avg Reward: 467.2, Eps: 0.288
# Episode 300, Avg Reward: 500.0, Eps: 0.224

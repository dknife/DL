"""
으뜸 딥러닝 — 13장 05절
행동 선택과 DQN 갱신
"""

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(s)
        return q_values.argmax(dim=1).item()

def update(batch_size=64, gamma=0.99):
    if len(buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = \
        buffer.sample(batch_size)

    # Q(s, a) from policy network
    q_values = policy_net(states).gather(
        1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a') from target network
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

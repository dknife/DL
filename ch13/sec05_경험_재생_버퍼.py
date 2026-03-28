"""
으뜸 딥러닝 — 13장 05절
경험 재생 버퍼
"""

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward,
                            next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = \
            zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(
                    np.array(next_states)).to(device),
                torch.FloatTensor(dones).to(device))

    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer(capacity=10000)

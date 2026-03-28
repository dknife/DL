"""
으뜸 딥러닝 — 04장 06절
2층 MLP 정의
"""

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()             # 28x28 -> 784
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)                  # output layer: no activation
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

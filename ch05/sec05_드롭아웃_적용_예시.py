"""
으뜸 딥러닝 — 05장 05절
드롭아웃 적용 예시
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),     # hidden layer: 30% drop
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10)     # no dropout before output layer
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
model.train()    # dropout enabled
output = model(x_train)

model.eval()     # dropout disabled (inference mode)
output = model(x_test)

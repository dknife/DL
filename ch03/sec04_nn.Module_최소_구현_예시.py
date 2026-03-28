"""
으뜸 딥러닝 — 03장 04절
nn.Module 최소 구현 예시
"""

import torch
import torch.nn as nn

class SingleLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()                        # mandatory: initialize base class
        self.linear = nn.Linear(in_features, out_features)
        self.relu   = nn.ReLU()

    def forward(self, x):
        z = self.linear(x)   # linear transform: y = Wx + b
        a = self.relu(z)     # non-linear activation
        return a

model = SingleLayer(4, 8)
x = torch.randn(16, 4)      # batch of 16 samples, 4 features each
out = model(x)              # calls forward() internally
print(out.shape)            # (16, 8)

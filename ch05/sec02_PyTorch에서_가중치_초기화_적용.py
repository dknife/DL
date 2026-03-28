"""
으뜸 딥러닝 — 05장 02절
PyTorch에서 가중치 초기화 적용
"""

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# He (Kaiming) initialization for ReLU layers
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# Check the first layer's weight statistics
w = model[0].weight
print(f"mean: {w.mean():.4f}, std: {w.std():.4f}")
# mean: -0.0003, std: 0.0505  (close to sqrt(2/784)=0.0505)

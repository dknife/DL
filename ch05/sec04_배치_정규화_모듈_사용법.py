"""
으뜸 딥러닝 — 05장 04절
배치 정규화 모듈 사용법
"""

import torch.nn as nn

# 1D: after fully-connected layers (input: [B, C])
bn1d = nn.BatchNorm1d(num_features=256)

# 2D: after conv layers (input: [B, C, H, W])
bn2d = nn.BatchNorm2d(num_features=64)

# Typical CNN block
block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

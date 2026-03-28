"""
으뜸 딥러닝 — 06장 01절
Conv2d 기본 사용법
"""

import torch
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(in_channels=3, out_channels=32,
                 kernel_size=3, stride=1, padding=1)

x = torch.randn(1, 3, 32, 32)   # [batch, channels, height, width]
y = conv(x)
print(y.shape)                   # torch.Size([1, 32, 32, 32])

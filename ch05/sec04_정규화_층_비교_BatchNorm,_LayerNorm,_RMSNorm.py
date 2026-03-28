"""
으뜸 딥러닝 — 05장 04절
정규화 층 비교: BatchNorm, LayerNorm, RMSNorm
"""

import torch
import torch.nn as nn

# BatchNorm: used in CNN blocks (specify num channels)
bn = nn.BatchNorm2d(num_features=64)

# LayerNorm: used in Transformer blocks (specify feature dim)
ln = nn.LayerNorm(normalized_shape=512)

# RMSNorm: available in PyTorch 2.4+
rms = nn.RMSNorm(normalized_shape=512)

# Switch train/eval mode (affects BatchNorm only)
model.train()   # use mini-batch statistics
model.eval()    # use running mean/variance

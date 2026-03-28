"""
으뜸 딥러닝 — 12장 06절
노이즈 스케줄과 순방향 과정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Noise schedule
T = 1000
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_sample(x0, t, noise=None):
    """Forward process: sample x_t from x_0 directly."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1, 1, 1).to(x0.device)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

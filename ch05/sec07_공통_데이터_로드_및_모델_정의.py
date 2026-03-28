"""
으뜸 딥러닝 — 05장 07절
공통 데이터 로드 및 모델 정의
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy, math

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))   # FashionMNIST mean/std
])
train_set = datasets.FashionMNIST('.', train=True,  download=True,
                                  transform=transform)
test_set  = datasets.FashionMNIST('.', train=False, download=True,
                                  transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=512)

# --- Model ---
def make_mlp(act=nn.ReLU, use_bn=False, drop_p=0.0):
    """Build a 4-layer MLP. use_bn adds BatchNorm, drop_p adds Dropout."""
    def block(in_f, out_f):
        layers = [nn.Linear(in_f, out_f)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_f))
        layers.append(act())
        if drop_p > 0:
            layers.append(nn.Dropout(drop_p))
        return layers

    return nn.Sequential(
        nn.Flatten(),
        *block(784, 512),
        *block(512, 256),
        *block(256, 128),
        nn.Linear(128, 10)
    )

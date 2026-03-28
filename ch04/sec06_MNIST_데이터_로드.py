"""
으뜸 딥러닝 — 04장 06절
MNIST 데이터 로드
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),                  # [0,255] -> [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_set = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_set  = datasets.MNIST('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1000)

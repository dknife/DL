"""
으뜸 딥러닝 — 11장 06절
MNIST 데이터셋 로드
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),  # [0, 255] -> [0, 1]
])

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
loader = DataLoader(train_data, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

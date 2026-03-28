"""
으뜸 딥러닝 — 05장 01절
문제 유형별 손실 함수 사용
"""

import torch
import torch.nn as nn

# Regression: MSE loss (no output activation)
loss_mse = nn.MSELoss()
pred   = torch.tensor([2.5, 0.3, 1.8])
target = torch.tensor([3.0, 0.0, 2.0])
print(loss_mse(pred, target))       # tensor(0.1133)

# Binary classification: BCE (sigmoid built-in)
loss_bce = nn.BCEWithLogitsLoss()
logit = torch.tensor([1.2, -0.5, 0.8])
label = torch.tensor([1.0,  0.0, 1.0])
print(loss_bce(logit, label))       # tensor(0.3711)

# Multi-class: CrossEntropy (softmax built-in)
loss_ce = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1]])   # raw logits
target = torch.tensor([0])                  # class index
print(loss_ce(logits, target))              # tensor(0.4170)

"""
으뜸 딥러닝 — 03장 04절
주요 내장 층 소개
"""

import torch.nn as nn

# Fully connected layer: y = xW^T + b
#   in_features: input dimension, out_features: output dimension
fc = nn.Linear(in_features=128, out_features=64)

# Activation functions
relu    = nn.ReLU()          # max(0, x)
sigmoid = nn.Sigmoid()       # 1 / (1 + exp(-x))
tanh    = nn.Tanh()          # (exp(x)-exp(-x)) / (exp(x)+exp(-x))

# Regularization
dropout = nn.Dropout(p=0.3)  # randomly zero out 30% of neurons during training
bn      = nn.BatchNorm1d(64) # batch normalization for 1-D feature vectors

# Loss functions
mse_loss = nn.MSELoss()           # regression
ce_loss  = nn.CrossEntropyLoss()  # classification (includes softmax internally)

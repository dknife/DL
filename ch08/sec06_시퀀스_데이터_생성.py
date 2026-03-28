"""
으뜸 딥러닝 — 08장 06절
시퀀스 데이터 생성
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Sequence generation parameters
size, seq_len = 300, 200
np.random.seed(42)

# Generate sequences: add random change to previous value
seq_X = np.empty(shape=(size, seq_len, 1))
for i in range(size):
    seq_X[i, 0, 0] = np.random.uniform(low=0, high=10)
    for j in range(1, seq_len):
        seq_X[i, j, 0] = seq_X[i, j-1, 0] + np.random.uniform(-3, 3)

# Weighted-sum label: earlier elements get larger weights
n = seq_len
weights = 2 * (n - np.arange(seq_len)) / (n * (n + 1))
y = np.sum(seq_X.squeeze() * weights, axis=1)

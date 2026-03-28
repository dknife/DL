"""
으뜸 딥러닝 — 03장 03절
커스텀 Dataset 구현
"""

import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    """Dataset for (feature matrix X, label vector y) pairs."""

    def __init__(self, X, y):
        # convert to float32 tensors if not already
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)          # total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # one (feature, label) pair

# --- usage ---
import numpy as np
X = np.random.randn(200, 4)    # 200 samples, 4 features each
y = np.random.randint(0, 2, 200).astype(float)

dataset = TabularDataset(X, y)
print(len(dataset))            # 200
print(dataset[0])              # (tensor([...]), tensor(0.))

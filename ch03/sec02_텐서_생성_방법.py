"""
으뜸 딥러닝 — 03장 02절
텐서 생성 방법
"""

import torch
import numpy as np

# 1) Create directly from data
a = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])       # 2x2 matrix tensor

# 2) Create by specifying shape
zeros  = torch.zeros(3, 4)          # all zeros, shape (3,4)
ones   = torch.ones(3, 4)           # all ones
rand   = torch.rand(3, 4)           # uniform U(0,1) random
randn  = torch.randn(3, 4)          # standard normal N(0,1) random
eye    = torch.eye(3)               # 3x3 identity matrix

# 3) Convert from NumPy array (shares memory)
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)         # NumPy -> Tensor
arr_back = t.numpy()                # Tensor -> NumPy

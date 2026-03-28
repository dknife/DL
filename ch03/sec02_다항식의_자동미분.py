"""
으뜸 딥러닝 — 03장 02절
다항식의 자동미분
"""

import torch

x = torch.tensor(2.0, requires_grad=True)  # start gradient tracking

y = x**2 + 3*x + 1                         # forward pass: build computation graph

y.backward()                               # backward pass: compute dy/dx

print(f"y    = {y.item():.1f}")            # y    = 11.0
print(f"dy/dx = {x.grad.item():.1f}")     # dy/dx = 7.0  [ok]

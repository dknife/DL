"""
으뜸 딥러닝 — 03장 02절
단일 뉴런의 자동미분
"""

import torch

# Parameters (requires_grad=True: tracked for gradients)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Data (requires_grad not needed)
x       = torch.tensor(3.0)
y_true  = torch.tensor(2.0)

# Forward pass
y_hat = w * x + b              # prediction: 0.5*3 + 0 = 1.5
loss  = (y_hat - y_true)**2    # MSE: (1.5 - 2.0)^2 = 0.25

# Backward pass
loss.backward()

# Compare with analytical result
# dL/dw = 2*(y_hat - y_true)*x = 2*(-0.5)*3 = -3.0
# dL/db = 2*(y_hat - y_true)   = 2*(-0.5)   = -1.0
print(f"dL/dw = {w.grad.item():.1f}")    # -3.0 [ok]
print(f"dL/db = {b.grad.item():.1f}")    # -1.0 [ok]

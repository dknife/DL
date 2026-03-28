"""
으뜸 딥러닝 — 02장 02절
PyTorch 선형회귀: 경사하강법으로 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Model: linear layer with input 1 -> output 1
model = nn.Linear(1, 1)
criterion = nn.MSELoss()

# SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    optimizer.zero_grad()          # reset gradients
    y_pred = model(X_train)        # forward pass
    loss = criterion(y_pred, y_train)
    loss.backward()                # backward pass: auto-compute partial derivatives
    optimizer.step()               # update params: w <- w - eta * grad_L

print(f"weight: {model.weight.item():.4f}")
print(f"bias:   {model.bias.item():.4f}")

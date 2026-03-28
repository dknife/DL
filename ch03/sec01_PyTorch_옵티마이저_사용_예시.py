"""
으뜸 딥러닝 — 03장 01절
PyTorch 옵티마이저 사용 예시
"""

import torch.optim as optim

# SGD (with momentum): simple, highly tunable baseline optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam: adaptive learning rate, converges fast on most tasks
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW: Adam + decoupled weight decay -- standard for transformer training
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Typical usage pattern inside training loop
for x, y in dataloader:
    optimizer.zero_grad()           # clear previous gradients
    loss = criterion(model(x), y)
    loss.backward()                 # backward pass: compute gradients
    optimizer.step()                # update parameters

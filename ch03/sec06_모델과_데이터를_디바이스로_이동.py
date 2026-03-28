"""
으뜸 딥러닝 — 03장 06절
모델과 데이터를 디바이스로 이동
"""

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)

# Move model parameters to device (done once before the loop)
model = model.to(device)

# Verify all parameters are on the correct device
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
# 0.weight: cuda:0
# 0.bias  : cuda:0  ...

# Inside training loop: move each batch to device
for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)      # move input
    y_batch = y_batch.to(device)      # move label

    # all operations stay on GPU
    pred = model(X_batch)
    loss = criterion(pred, y_batch)
    ...

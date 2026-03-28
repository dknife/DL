"""
으뜸 딥러닝 — 03장 05절
표준 PyTorch 학습 루프
"""

import torch
import torch.nn as nn
import torch.optim as optim

# --- model / loss / optimizer setup ---
model     = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    # ===== training phase =====
    model.train()                        # enable dropout / batch norm train mode
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()            # 1) clear gradients from previous step
        y_pred = model(X_batch)          # 2) forward pass
        loss   = criterion(y_pred, y_batch.unsqueeze(1))
        loss.backward()                  # 3) backward pass: compute gradients
        optimizer.step()                 # 4) update parameters

        total_loss += loss.item()        # accumulate scalar loss (detached)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:3d}/{num_epochs}  train loss: {avg_loss:.4f}")

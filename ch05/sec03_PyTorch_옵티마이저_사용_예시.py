"""
으뜸 딥러닝 — 05장 03절
PyTorch 옵티마이저 사용 예시
"""

import torch.optim as optim

model = ...  # nn.Module

# 1. SGD + Momentum + Nesterov
opt_sgd = optim.SGD(model.parameters(),
                    lr=0.01, momentum=0.9, nesterov=True)

# 2. Adam (default betas=(0.9, 0.999))
opt_adam = optim.Adam(model.parameters(), lr=1e-3)

# 3. AdamW (decoupled weight decay)
opt_adamw = optim.AdamW(model.parameters(),
                        lr=1e-3, weight_decay=0.01)

# Training loop (common pattern)
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        loss = criterion(model(x_batch), y_batch)
        opt_adamw.zero_grad()
        loss.backward()
        opt_adamw.step()

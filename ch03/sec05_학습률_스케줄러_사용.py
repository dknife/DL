"""
으뜸 딥러닝 — 03장 05절
학습률 스케줄러 사용
"""

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# StepLR: multiply lr by gamma every step_size epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# CosineAnnealingLR: decay lr following a cosine curve
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ReduceLROnPlateau: reduce lr when val_loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

for epoch in range(num_epochs):
    # ... training and validation ...

    # call after each epoch (pass val_loss for ReduceLROnPlateau)
    scheduler.step()                       # or: scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"  lr = {current_lr:.2e}")

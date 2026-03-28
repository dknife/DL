"""
으뜸 딥러닝 — 05장 06절
StepLR 스케줄러 사용
"""

from torch.optim.lr_scheduler import StepLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(90):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()   # update LR at end of epoch

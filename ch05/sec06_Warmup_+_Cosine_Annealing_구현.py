"""
으뜸 딥러닝 — 05장 06절
Warmup + Cosine Annealing 구현
"""

import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_schedule(warmup_steps, total_steps):
    """Return a Warmup + Cosine Annealing LR lambda."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps          # linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay
    return lr_lambda

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = LambdaLR(
    optimizer,
    lr_lambda=warmup_cosine_schedule(
        warmup_steps=500, total_steps=10000
    )
)

for step, (x_batch, y_batch) in enumerate(dataloader):
    loss = criterion(model(x_batch), y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()     # update LR every step

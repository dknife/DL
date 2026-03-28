"""
으뜸 딥러닝 — 03장 06절
GPU 메모리 관리 패턴
"""

# 1) Check memory usage
print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
print(torch.cuda.memory_reserved()  / 1e6, "MB reserved")

# 2) Release cache after heavy operations
torch.cuda.empty_cache()

# 3) Use half precision (float16) to halve memory usage
model = model.half()                     # model weights -> float16
X_batch = X_batch.half()                 # input -> float16

# 4) Automatic Mixed Precision (AMP): best of both worlds
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X_b, y_b in train_loader:
    X_b, y_b = X_b.to(device), y_b.to(device)
    optimizer.zero_grad()

    with autocast():                     # float16 forward for speed
        pred = model(X_b)
        loss = criterion(pred, y_b)

    scaler.scale(loss).backward()        # scaled backward for stability
    scaler.step(optimizer)
    scaler.update()

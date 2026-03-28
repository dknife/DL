"""
으뜸 딥러닝 — 03장 03절
DataLoader 기본 사용
"""

from torch.utils.data import random_split

# Split: 80% train / 20% validation
train_set, val_set = random_split(dataset, [160, 40])

train_loader = DataLoader(
    train_set,
    batch_size=32,      # samples per mini-batch
    shuffle=True,       # randomize order each epoch
    num_workers=2,      # parallel data loading processes
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False,      # no shuffle for evaluation
    num_workers=2,
)

# Each iteration yields a batch
for X_batch, y_batch in train_loader:
    print(X_batch.shape)   # (32, 4)  -- batch_size x features
    print(y_batch.shape)   # (32,)
    break

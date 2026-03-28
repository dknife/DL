"""
으뜸 딥러닝 — 03장 03절
torchvision CIFAR-10 로딩과 transform
"""

import torchvision
import torchvision.transforms as T

# Define preprocessing pipeline
transform = T.Compose([
    T.ToTensor(),               # PIL image -> float32 tensor [0,1]
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
        std =(0.2023, 0.1994, 0.2010),   # CIFAR-10 channel stds
    ),
])

# Download and wrap with transform
train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False, num_workers=4)

# Verify shape
images, labels = next(iter(train_loader))
print(images.shape)   # (64, 3, 32, 32)  -- batch x C x H x W
print(labels.shape)   # (64,)

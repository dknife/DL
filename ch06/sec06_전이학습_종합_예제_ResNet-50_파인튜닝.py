"""
으뜸 딥러닝 — 06장 06절
전이학습 종합 예제: ResNet-50 파인튜닝
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Data preparation with augmentation ---
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

train_set = datasets.ImageFolder('data/train', train_tf)
val_set   = datasets.ImageFolder('data/val',   val_tf)
train_loader = DataLoader(train_set, batch_size=32,
                          shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32)

# --- 2. Model setup ---
model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_set.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# --- 3. Phase 1: Train head only ---
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
model.to(device)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), \
                         labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- 4. Phase 2: Fine-tune upper layers ---
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), \
                         labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

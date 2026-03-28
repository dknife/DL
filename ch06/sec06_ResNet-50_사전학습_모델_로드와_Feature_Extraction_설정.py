"""
으뜸 딥러닝 — 06장 06절
ResNet-50 사전학습 모델 로드와 Feature Extraction 설정
"""

import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights='IMAGENET1K_V2')

# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
# New layer parameters are trainable by default

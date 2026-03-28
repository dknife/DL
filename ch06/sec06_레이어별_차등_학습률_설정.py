"""
으뜸 딥러닝 — 06장 06절
레이어별 차등 학습률 설정
"""

import torch.optim as optim

# Group parameters by depth
param_groups = [
    # Lower layers: very small learning rate
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    # Upper layers: moderate learning rate
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    # Classification head: highest learning rate
    {'params': model.fc.parameters(),     'lr': 1e-3},
]

optimizer = optim.Adam(param_groups)

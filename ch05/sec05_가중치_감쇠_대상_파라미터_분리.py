"""
으뜸 딥러닝 — 05장 05절
가중치 감쇠 대상 파라미터 분리
"""

import torch.optim as optim

# Separate params with/without weight decay
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 1e-2},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-3)

"""
으뜸 딥러닝 — 11장 05절
PyTorch 스펙트럼 정규화
"""

import torch.nn.utils as utils

# Apply spectral normalization to a layer
layer = utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1))

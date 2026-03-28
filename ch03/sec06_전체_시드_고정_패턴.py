"""
으뜸 딥러닝 — 03장 06절
전체 시드 고정 패턴
"""

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)                        # Python built-in random
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)         # PyTorch GPU (all devices)
    # make CUDA operations deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)

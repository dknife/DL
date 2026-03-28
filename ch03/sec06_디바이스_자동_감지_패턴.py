"""
으뜸 딥러닝 — 03장 06절
디바이스 자동 감지 패턴
"""

import torch

# Auto-select: CUDA > MPS (Apple Silicon) > CPU
device = (
    "cuda"  if torch.cuda.is_available()               else
    "mps"   if torch.backends.mps.is_available()       else
    "cpu"
)
print(f"Using device: {device}")

# CUDA details (when available)
if device == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    total  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {total:.1f} GB")

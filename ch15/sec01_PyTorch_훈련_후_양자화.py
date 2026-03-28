"""
으뜸 딥러닝 — 15장 01절
PyTorch 훈련 후 양자화
"""

import torch

# Load a trained model
model = torch.load("model.pth")
model.eval()

# Dynamic quantization (CPU inference)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # quantize Linear layers
    dtype=torch.qint8    # to INT8
)

# Compare model sizes
def get_size_mb(m):
    torch.save(m.state_dict(), "/tmp/tmp.pt")
    import os
    return os.path.getsize("/tmp/tmp.pt") / 1e6

print(f"Original:  {get_size_mb(model):.1f} MB")
print(f"Quantized: {get_size_mb(quantized_model):.1f} MB")
# Original:  120.4 MB
# Quantized:  30.2 MB  (about 4x smaller)

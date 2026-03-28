"""
으뜸 딥러닝 — 07장 06절
사전학습 Faster R-CNN으로 객체 탐지
"""

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

# Load pretrained Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights='DEFAULT')
model.eval()

# Read and normalize image
img = read_image('test.jpg') / 255.0  # [C, H, W]

with torch.no_grad():
    preds = model([img])[0]

# preds keys: 'boxes', 'labels', 'scores'
for box, label, score in zip(
        preds['boxes'], preds['labels'], preds['scores']):
    if score > 0.7:
        print(f"class {label.item()}: "
              f"score {score:.2f}, box {box.tolist()}")

"""
으뜸 딥러닝 — 07장 06절
Faster R-CNN 분류 헤드 교체와 학습 루프
"""

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor)

# Load pretrained model, replace head
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor\
                    .cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, num_classes=3)  # bg + 2 classes

# Training loop
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005,
    momentum=0.9, weight_decay=5e-4)

model.train()
for epoch in range(10):
    for images, targets in train_loader:
        # targets: list of dict with 'boxes', 'labels'
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""
으뜸 딥러닝 — 06장 02절
FashionMNIST용 간단한 CNN
"""

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 28x28 -> 14x14

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # 7x7 -> 1x1 (GAP)
            nn.Flatten(),                # [B, 64, 1, 1] -> [B, 64]
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

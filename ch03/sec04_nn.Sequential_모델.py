"""
으뜸 딥러닝 — 03장 04절
nn.Sequential 모델
"""

# Multi-layer perceptron: 784 -> 256 -> 128 -> 10
mlp = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),   # output logits (no softmax: use CrossEntropyLoss)
)

x = torch.randn(32, 784)
logits = mlp(x)           # data flows through layers in order
print(logits.shape)       # (32, 10)

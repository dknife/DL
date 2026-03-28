"""
으뜸 딥러닝 — 04장 06절
활성화 함수 비교 실험
"""

def make_deep_mlp(activation):
    """Build a 5-layer MLP with the given activation function."""
    layers = [nn.Flatten(), nn.Linear(784, 256), activation]
    for _ in range(3):                          # add 3 hidden layers
        layers += [nn.Linear(256, 256), activation]
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)

# Sigmoid vs ReLU
for name, act in [("Sigmoid", nn.Sigmoid()), ("ReLU", nn.ReLU())]:
    model = make_deep_mlp(act)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for epoch in range(1, 6):
        loss = train(nn.Module(), train_loader, crit, opt)  # reuse train() above
        acc  = evaluate(model, test_loader)
    print(f"{name:8s} -> test_acc={acc:.4f}")
# Sigmoid  -> test_acc=0.9621
# ReLU     -> test_acc=0.9780

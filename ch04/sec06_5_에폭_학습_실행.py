"""
으뜸 딥러닝 — 04장 06절
5 에폭 학습 실행
"""

model     = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 6):
    loss = train(model, train_loader, criterion, optimizer)
    acc  = evaluate(model, test_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={acc:.4f}")
# Epoch 1: loss=0.2438, test_acc=0.9567
# ...
# Epoch 5: loss=0.0354, test_acc=0.9761

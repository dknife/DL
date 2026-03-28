"""
으뜸 딥러닝 — 04장 06절
학습 및 평가 함수
"""

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()                         # backpropagation (Sec 4.3)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():                       # disable gradient computation
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

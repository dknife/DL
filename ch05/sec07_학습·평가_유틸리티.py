"""
으뜸 딥러닝 — 05장 07절
학습·평가 유틸리티
"""

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    for X, y in loader:
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()          # for step-level schedulers
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

def run_experiment(model, optimizer, n_epochs=20, scheduler=None):
    """Train for n_epochs and return lists of (train loss, test acc)."""
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'acc': []}
    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion,
                               optimizer, scheduler)
        acc  = evaluate(model, test_loader)
        history['loss'].append(loss)
        history['acc'].append(acc)
    return history

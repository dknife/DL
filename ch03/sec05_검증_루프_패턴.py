"""
으뜸 딥러닝 — 03장 05절
검증 루프 패턴
"""

def evaluate(model, loader, criterion, device):
    model.eval()                          # disable dropout / use running stats in BN
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():                 # no gradient tracking needed
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()

            # accuracy for classification
            preds    = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# integrate into training loop
for epoch in range(num_epochs):
    # ... training phase (as above) ...

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1:3d}  val loss: {val_loss:.4f}  val acc: {val_acc:.3f}")

"""
으뜸 딥러닝 — 03장 05절
학습 이력 추적과 조기 종료
"""

history = {"train_loss": [], "val_loss": [], "val_acc": []}

best_val_loss = float("inf")
patience      = 5        # stop if no improvement for 5 consecutive epochs
no_improve    = 0

for epoch in range(num_epochs):
    # --- train ---
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- validate ---
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:3d}  "
          f"train: {train_loss:.4f}  val: {val_loss:.4f}  acc: {val_acc:.3f}")

    # --- early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")  # save best weights
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# reload best weights after training
model.load_state_dict(torch.load("best_model.pth"))

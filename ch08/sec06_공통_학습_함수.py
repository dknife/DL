"""
으뜸 딥러닝 — 08장 06절
공통 학습 함수
"""

def train_model(model, train_X, train_y, epochs=25, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        pred = model(train_X)
        loss = criterion(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

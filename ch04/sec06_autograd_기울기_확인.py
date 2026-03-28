"""
으뜸 딥러닝 — 04장 06절
autograd 기울기 확인
"""

# Forward pass with a single input
x = torch.randn(1, 784)
model.zero_grad()
out = model.net(x)                     # direct input without flatten
loss = nn.CrossEntropyLoss()(out, torch.tensor([3]))
loss.backward()

# Check gradients of the first Linear layer
W1_grad = model.net[0].weight.grad     # shape: (256, 784)
print(W1_grad.shape)                   # torch.Size([256, 784])
print(W1_grad.abs().mean().item())     # mean absolute gradient

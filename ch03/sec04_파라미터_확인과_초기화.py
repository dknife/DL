"""
으뜸 딥러닝 — 03장 04절
파라미터 확인과 초기화
"""

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10),
)

# Count total trainable parameters
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total:,}")   # 233,738

# Inspect each layer's name and shape
for name, param in model.named_parameters():
    print(f"{name:30s}  {str(param.shape):20s}  {param.numel():,}")
# 0.weight    torch.Size([256, 784])    200,704
# 0.bias      torch.Size([256])             256
# 2.weight    torch.Size([10, 256])      2,560
# 2.bias      torch.Size([10])              10

# Custom weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)   # apply recursively to all submodules

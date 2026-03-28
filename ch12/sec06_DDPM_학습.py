"""
으뜸 딥러닝 — 12장 06절
DDPM 학습
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
])
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
loader = DataLoader(train_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Move schedule tensors to device
alpha_bar_dev = alpha_bar.to(device)

for epoch in range(10):
    total_loss = 0
    for images, _ in loader:
        images = images.to(device)
        # Sample random timesteps
        t = torch.randint(0, T, (images.size(0),), device=device)
        # Sample noise and create noisy images
        noise = torch.randn_like(images)
        x_t = q_sample(images, t, noise)
        # Predict noise
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1:2d}, Loss: {avg:.4f}")
# Epoch  2, Loss: 0.0512
# Epoch  4, Loss: 0.0387
# Epoch  6, Loss: 0.0341
# Epoch  8, Loss: 0.0312
# Epoch 10, Loss: 0.0295

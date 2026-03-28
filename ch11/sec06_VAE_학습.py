"""
으뜸 딥러닝 — 11장 06절
VAE 학습
"""

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    bce = F.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

vae = VAE(latent_dim=20).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for images, _ in loader:
        images = images.to(device)
        recon, mu, logvar = vae(images)
        loss = vae_loss(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader.dataset)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}, Loss: {avg:.1f}")
# Epoch  5, Loss: 136.2
# Epoch 10, Loss: 120.8
# Epoch 15, Loss: 115.3
# Epoch 20, Loss: 112.1

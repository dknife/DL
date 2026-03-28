"""
으뜸 딥러닝 — 11장 06절
DCGAN 학습 루프
"""

latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Fixed noise for monitoring generation progress
fixed_z = torch.randn(16, latent_dim).to(device)

for epoch in range(20):
    for images, _ in loader:
        batch_size = images.size(0)
        real = (images.to(device) * 2 - 1)  # [0,1] -> [-1,1]
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, latent_dim).to(device)
        fake = G(z).detach()

        loss_D = criterion(D(real), real_label) \
               + criterion(D(fake), fake_label)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train Generator (non-saturating loss) ---
        z = torch.randn(batch_size, latent_dim).to(device)
        fake = G(z)
        loss_G = criterion(D(fake), real_label)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}, D_loss: {loss_D:.3f}, "
              f"G_loss: {loss_G:.3f}")
# Epoch  5, D_loss: 0.812, G_loss: 1.543
# Epoch 10, D_loss: 0.693, G_loss: 1.245
# Epoch 15, D_loss: 0.701, G_loss: 0.987
# Epoch 20, D_loss: 0.694, G_loss: 0.856

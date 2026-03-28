"""
으뜸 딥러닝 — 11장 04절
GAN 학습 루프 (1배치)
"""

# --- Discriminator step ---
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# Forward pass on real data
out_real = D(real_images)
loss_real = F.binary_cross_entropy(out_real, real_labels)

# Forward pass on fake data
z = torch.randn(batch_size, latent_dim, device=device)
fake_images = G(z).detach()  # detach to avoid updating G
out_fake = D(fake_images)
loss_fake = F.binary_cross_entropy(out_fake, fake_labels)

loss_D = loss_real + loss_fake
optimizer_D.zero_grad()
loss_D.backward()
optimizer_D.step()

# --- Generator step ---
z = torch.randn(batch_size, latent_dim, device=device)
fake_images = G(z)
out_fake = D(fake_images)
# Non-saturating loss: maximize log D(G(z))
loss_G = F.binary_cross_entropy(out_fake, real_labels)

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()

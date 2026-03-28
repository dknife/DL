"""
으뜸 딥러닝 — 11장 06절
GAN 이미지 생성과 비교
"""

# Generate DCGAN samples
G.eval()
with torch.no_grad():
    gan_samples = G(fixed_z).cpu()
    gan_samples = (gan_samples + 1) / 2  # [-1,1] -> [0,1]

# Generate VAE samples (same number)
vae.eval()
with torch.no_grad():
    z_vae = torch.randn(16, 20).to(device)
    vae_samples = vae.decode(z_vae).view(-1, 1, 28, 28).cpu()

# Side-by-side comparison
fig, axes = plt.subplots(2, 8, figsize=(10, 2.5))
for i in range(8):
    axes[0, i].imshow(vae_samples[i, 0], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(gan_samples[i, 0], cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("VAE", fontsize=10)
axes[1, 0].set_ylabel("GAN", fontsize=10)
plt.suptitle("VAE vs GAN Generated Samples")
plt.tight_layout()
plt.show()

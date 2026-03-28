"""
으뜸 딥러닝 — 11장 06절
VAE 이미지 생성과 잠재 공간 보간
"""

import matplotlib.pyplot as plt

# Generate new images from random z
vae.eval()
with torch.no_grad():
    z = torch.randn(16, 20).to(device)
    samples = vae.decode(z).view(-1, 1, 28, 28).cpu()

fig, axes = plt.subplots(2, 8, figsize=(10, 2.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")
plt.suptitle("VAE Generated Samples")
plt.show()

# Latent space interpolation between two digits
img_a, _ = train_data[0]   # e.g. digit 5
img_b, _ = train_data[1]   # e.g. digit 0
with torch.no_grad():
    mu_a, _ = vae.encode(img_a.view(1, -1).to(device))
    mu_b, _ = vae.encode(img_b.view(1, -1).to(device))

    interp = []
    for t in torch.linspace(0, 1, 10):
        z_t = (1 - t) * mu_a + t * mu_b
        interp.append(vae.decode(z_t).view(28, 28).cpu())

fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(interp[i], cmap="gray")
    ax.axis("off")
plt.suptitle("VAE Latent Interpolation")
plt.show()

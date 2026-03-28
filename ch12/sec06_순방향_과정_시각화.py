"""
으뜸 딥러닝 — 12장 06절
순방향 과정 시각화
"""

# Take one clean image
img, _ = train_data[0]
img = img.unsqueeze(0).to(device)

# Show forward process at selected timesteps
steps = [0, 50, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(steps), figsize=(12, 2))
for i, t_val in enumerate(steps):
    t = torch.tensor([t_val], device=device)
    noisy = q_sample(img, t)
    noisy_img = (noisy[0, 0].cpu().clamp(-1, 1) + 1) / 2
    axes[i].imshow(noisy_img, cmap="gray")
    axes[i].set_title(f"t={t_val}", fontsize=9)
    axes[i].axis("off")
plt.suptitle("Forward Diffusion Process")
plt.tight_layout()
plt.show()

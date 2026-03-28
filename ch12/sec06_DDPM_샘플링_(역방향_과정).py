"""
으뜸 딥러닝 — 12장 06절
DDPM 샘플링 (역방향 과정)
"""

@torch.no_grad()
def p_sample(model, x_t, t_idx):
    """One step of reverse process."""
    t = torch.full((x_t.size(0),), t_idx, device=x_t.device,
                   dtype=torch.long)
    pred = model(x_t, t)
    a = alpha[t_idx].to(x_t.device)
    ab = alpha_bar[t_idx].to(x_t.device)
    # Predicted mean
    mean = (1 / torch.sqrt(a)) * (
        x_t - (1 - a) / torch.sqrt(1 - ab) * pred
    )
    if t_idx > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta[t_idx].to(x_t.device))
        return mean + sigma * noise
    return mean

@torch.no_grad()
def generate(model, n_samples=16):
    """Generate images via full reverse process."""
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    for t_idx in reversed(range(T)):
        x = p_sample(model, x, t_idx)
    return x.clamp(-1, 1)

samples = generate(model)
samples = (samples + 1) / 2  # [-1,1] -> [0,1]

fig, axes = plt.subplots(2, 8, figsize=(10, 2.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu(), cmap="gray")
    ax.axis("off")
plt.suptitle("DDPM Generated Samples (10 epochs)")
plt.show()

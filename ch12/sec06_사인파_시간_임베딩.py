"""
으뜸 딥러닝 — 12장 06절
사인파 시간 임베딩
"""

def sinusoidal_embedding(t, dim=128):
    """Sinusoidal positional embedding for timestep t."""
    half = dim // 2
    freq = torch.exp(
        -torch.arange(half, device=t.device).float()
        * (torch.log(torch.tensor(10000.0)) / half)
    )
    emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=1)

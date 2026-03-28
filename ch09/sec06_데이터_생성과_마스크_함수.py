"""
으뜸 딥러닝 — 09장 06절
데이터 생성과 마스크 함수
"""

def make_data(n_samples=3000, seq_len=8, vocab=10):
    """Generate sequence reversal dataset."""
    # Token 0 = <pad>, 1 = <bos>, 2 = <eos>, 3..vocab+2 = digits
    src = torch.randint(3, vocab + 3, (n_samples, seq_len))
    bos = torch.full((n_samples, 1), 1)  # <bos>
    eos = torch.full((n_samples, 1), 2)  # <eos>
    tgt_in = torch.cat([bos, src.flip(1)], dim=1)   # teacher input
    tgt_out = torch.cat([src.flip(1), eos], dim=1)   # target label
    return src, tgt_in, tgt_out

def causal_mask(size):
    """Upper-triangular mask for masked self-attention."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.float().masked_fill(mask, float('-inf'))

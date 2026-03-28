"""
으뜸 딥러닝 — 09장 06절
그리디 디코딩 추론
"""

@torch.no_grad()
def greedy_decode(model, src, max_len=10, bos=1, eos=2):
    """Generate output sequence using greedy decoding."""
    enc_out = model.encode(src.unsqueeze(0))
    tgt = torch.tensor([[bos]])
    for _ in range(max_len):
        mask = causal_mask(tgt.size(1))
        logits = model.decode(tgt, enc_out, mask)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_tok], dim=1)
        if next_tok.item() == eos:
            break
    return tgt.squeeze(0)

model.eval()
test_src = src[0]
result = greedy_decode(model, test_src)
print(f"Input:    {test_src.tolist()}")
print(f"Expected: {test_src.flip(0).tolist()}")
print(f"Predicted:{result[1:].tolist()}")  # skip <bos>

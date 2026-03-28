"""
으뜸 딥러닝 — 09장 06절
인코더 블록
"""

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sublayer 1: self-attention + Add & Norm
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.drop1(attn_out))
        # Sublayer 2: FFN + Add & Norm
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

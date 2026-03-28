"""
으뜸 딥러닝 — 09장 06절
디코더 블록
"""

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, y, enc_out, tgt_mask=None):
        # Sublayer 1: masked self-attention
        attn1, _ = self.masked_attn(
            y, y, y, attn_mask=tgt_mask
        )
        y = self.norm1(y + self.drop1(attn1))
        # Sublayer 2: cross-attention (Q=decoder, K,V=encoder)
        attn2, _ = self.cross_attn(
            y, enc_out, enc_out
        )
        y = self.norm2(y + self.drop2(attn2))
        # Sublayer 3: FFN
        y = self.norm3(y + self.drop3(self.ffn(y)))
        return y

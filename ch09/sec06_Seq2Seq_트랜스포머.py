"""
으뜸 딥러닝 — 09장 06절
Seq2Seq 트랜스포머
"""

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128,
                 nhead=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, tgt_vocab)

    def encode(self, src):
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, tgt, enc_out, tgt_mask):
        y = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            y = layer(y, enc_out, tgt_mask)
        return self.output_proj(y)

    def forward(self, src, tgt, tgt_mask):
        enc_out = self.encode(src)
        return self.decode(tgt, enc_out, tgt_mask)

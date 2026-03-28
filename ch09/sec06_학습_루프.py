"""
으뜸 딥러닝 — 09장 06절
학습 루프
"""

vocab_size = 13  # 0=pad, 1=bos, 2=eos, 3..12=digits
model = Seq2SeqTransformer(
    src_vocab=vocab_size, tgt_vocab=vocab_size,
    d_model=128, nhead=4, d_ff=256, n_layers=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

src, tgt_in, tgt_out = make_data()

for epoch in range(30):
    tgt_mask = causal_mask(tgt_in.size(1))
    logits = model(src, tgt_in, tgt_mask)
    # logits: (batch, seq_len, vocab) -> flatten for loss
    loss = criterion(
        logits.reshape(-1, vocab_size), tgt_out.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

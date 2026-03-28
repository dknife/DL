"""
으뜸 딥러닝 — 08장 06절
어텐션 가중치 시각화
"""

attn_model.eval()
with torch.no_grad():
    _ = attn_model(test_X)
    w = attn_model.attn_weights[0, 0].numpy()  # first test sample

plt.figure(figsize=(10, 2))
plt.bar(range(seq_len), w, color='steelblue', alpha=0.7)
plt.xlabel('sequence position')
plt.ylabel('attention weight')
plt.title('Attention weights (test sample 0)')
plt.tight_layout(); plt.show()

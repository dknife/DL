"""
으뜸 딥러닝 — 06장 02절
CNN vs MLP 파라미터 수 비교
"""

cnn = SimpleCNN()
mlp_params = 784*512 + 512 + 512*256 + 256 + 256*128 + 128 + 128*10 + 10
cnn_params = sum(p.numel() for p in cnn.parameters())
print(f"MLP: {mlp_params:,}")    # MLP: 567,434
print(f"CNN: {cnn_params:,}")    # CNN: 19,146

"""
으뜸 딥러닝 — 08장 06절
데이터 분할 및 텐서 변환
"""

train_size = int(size * 0.8)
train_X = torch.FloatTensor(seq_X[:train_size])
train_y = torch.FloatTensor(y[:train_size]).unsqueeze(1)
test_X = torch.FloatTensor(seq_X[train_size:])
test_y = torch.FloatTensor(y[train_size:]).unsqueeze(1)

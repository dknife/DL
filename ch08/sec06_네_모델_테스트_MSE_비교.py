"""
으뜸 딥러닝 — 08장 06절
네 모델 테스트 MSE 비교
"""

criterion = nn.MSELoss()
for name, model in [('RNN', rnn_model), ('LSTM', lstm_model),
                    ('GRU', gru_model), ('Attention', attn_model)]:
    model.eval()
    with torch.no_grad():
        pred = model(test_X)
        mse = criterion(pred, test_y).item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{name:10s} | params: {n_params:,} | test MSE: {mse:.4f}')

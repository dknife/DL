"""
으뜸 딥러닝 — 08장 06절
손실 곡선 및 예측 결과 시각화
"""

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
models = [rnn_model, lstm_model, gru_model]
losses_list = [rnn_losses, lstm_losses, gru_losses]
names = ['RNN', 'LSTM', 'GRU']

for i, (model, losses, name) in enumerate(zip(models, losses_list, names)):
    # Loss curve
    axes[0, i].plot(losses)
    axes[0, i].set_title(f'{name} learning')
    axes[0, i].set_xlabel('epoch'); axes[0, i].set_ylabel('loss')
    # Test prediction
    model.eval()
    with torch.no_grad():
        y_hat = model(test_X).numpy()
    axes[1, i].scatter(test_y, y_hat, s=10, alpha=0.7)
    axes[1, i].plot([-30,30], [-30,30], 'r--')
    axes[1, i].set_title(f'{name} test')
    axes[1, i].set_xlabel('y label'); axes[1, i].set_ylabel('y prediction')
plt.tight_layout(); plt.show()

"""
으뜸 딥러닝 — 08장 06절
순환 모델 정의
"""

class SeqModel(nn.Module):
    def __init__(self, rnn_type='RNN', input_size=1, hidden_size=128):
        super().__init__()
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)        # (batch, seq_len, hidden)
        out = out[:, -1, :]          # last time step output
        return self.fc(out)

"""
으뜸 딥러닝 — 08장 06절
어텐션 모델 정의
"""

class AttentionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128):
        super().__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        # Learnable global query
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.d_k = hidden_size ** 0.5

    def forward(self, x):
        h = torch.relu(self.embed(x))       # (batch, seq_len, hidden)
        K = self.key_layer(h)                # (batch, seq_len, hidden)
        V = self.value_layer(h)              # (batch, seq_len, hidden)
        Q = self.query.expand(x.size(0), -1, -1)  # (batch, 1, hidden)
        # Scaled Dot-Product Attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.d_k
        self.attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(self.attn_weights, V)  # (batch, 1, hidden)
        return self.fc(context.squeeze(1))

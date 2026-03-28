"""
으뜸 딥러닝 — 03장 04절
잔차 블록: nn.Module 상속 예시
"""

class ResidualBlock(nn.Module):
    """Block with skip connection: out = F(x) + x"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x                     # save input for skip connection
        out = self.block(x)              # main path: two linear layers
        out = out + residual             # add skip connection
        return self.relu(out)            # activation after addition

# Stack multiple residual blocks
model = nn.Sequential(
    nn.Linear(128, 64),
    ResidualBlock(64),
    ResidualBlock(64),
    nn.Linear(64, 10),
)

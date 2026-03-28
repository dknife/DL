"""
으뜸 딥러닝 — 12장 06절
MNIST용 간단한 U-Net
"""

class SimpleUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 64), nn.ReLU()
        )
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        # Bottleneck
        self.bot = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        # Decoder (with skip connections)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t))
        # Inject time embedding into input
        t_emb = t_emb.view(-1, 64, 1, 1)

        e1 = self.enc1(x)                     # 32 x 28 x 28
        e2 = self.enc2(e1)                     # 64 x 14 x 14
        b = self.bot(e2)                       # 128 x 7 x 7
        b = b + t_emb.expand_as(b[:, :64])     # time injection
        d2 = self.up2(b)                       # 64 x 14 x 14
        d2 = self.dec2(torch.cat([d2, e2], 1)) # skip connection
        d1 = self.up1(d2)                      # 32 x 28 x 28
        d1 = self.dec1(torch.cat([d1, e1], 1)) # skip connection
        return self.out(d1)

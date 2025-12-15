import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2


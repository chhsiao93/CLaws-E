import torch.nn as nn

class MLPStressPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):  # 2D stress assumed (2x2=4)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


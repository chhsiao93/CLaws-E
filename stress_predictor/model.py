import torch.nn as nn
import torch

class MLPStressPredictor(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 2x2 flattened output
        )

    def forward(self, F: torch.Tensor):
        I = torch.eye(2, dtype=F.dtype, device=F.device, requires_grad=False)
        F = F.view(-1, 2, 2)
        U, sigma, Vh = torch.linalg.svd(F)
        R = torch.matmul(U, Vh)
        # transpose
        Ft = F.transpose(1, 2)
        # F^T * F
        FtF = torch.matmul(Ft, F) # (N, 2, 2)

        I1 = sigma - 1.0
        I2 = (FtF - I).view(-1, 4)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        invariants = torch.cat([I1, I2, I3], dim=1)
        x = invariants
        
        x = self.net(x)  # (N, 4)
        x = x.view(-1, 2, 2)  # reshape to (N, 2, 2)
        # symmetrize the F tensor
        x = (x + x.transpose(1, 2)) / 2
        # Rotate: R σ̂ Rᵀ
        delta_Fp = torch.matmul(R, x)  # (N, 2, 2)
        Fp = delta_Fp + F
        return Fp


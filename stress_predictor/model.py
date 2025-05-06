import torch.nn as nn
import torch

class DeformationCorrector(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 2x2 flattened output
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
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

class StressPredictorFromFp(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 2x2 flattened output
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        U, sigma, Vh = torch.linalg.svd(F)
        R = torch.matmul(U, Vh)

        Ft = F.transpose(1, 2)
        I1 = sigma.sum(dim=1) - 2.0
        I2 = torch.diagonal(torch.matmul(Ft, F), dim1=1, dim2=2).sum(dim=1) - 1.0
        I3 = torch.linalg.det(F) - 1.0

        x = torch.stack([I1, I2, I3], dim=1)  # (N, 3)

        x = self.net(x)  # (N, 3)
        x = x.view(-1, 2, 2)
        x = (x + x.transpose(1, 2)) / 2  # symmetrize

        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, F.transpose(1, 2))  

        return cauchy


class LearnedConstitutiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Fp_model = DeformationCorrector()
        self.stress_model = StressPredictorFromFp()

    def forward(self, F: torch.Tensor):
        Fp = self.Fp_model(F) # Fp shape: (N, 2, 2)
        sigma = self.stress_model(Fp)
        return sigma.view(-1, 4)  # Flatten to (N, 4)

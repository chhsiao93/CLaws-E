import torch
import numpy as np
from model import MLPStressPredictor
from trajectory_dataset import StressPredictionDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# Config
npz_path = "data/train.npz"
metadata_path = "data/metadata.json"
output_path = "results/"
model_path = "models/"
denormalize = False

def reconstruct_tensor(sym):
    # sym: (N, 3) => [xx, xy, yy]
    σ_xx = sym[:, 0]
    σ_xy = sym[:, 1]
    σ_yy = sym[:, 2]
    σ = torch.stack([
        torch.stack([σ_xx, σ_xy], dim=-1),
        torch.stack([σ_xy, σ_yy], dim=-1)
    ], dim=-2)  # (N, 2, 2)
    return σ

# === Load metadata for denormalization ===
with open(metadata_path, 'r') as f:
    stats = json.load(f)['train']
stress_mean = torch.tensor(np.array(stats['mean_stress'])).reshape(1, 4)
stress_std = torch.tensor(np.array(stats['std_stress'])).reshape(1, 4)
# stress_mean = torch.tensor(np.array(stats['mean_stress'])[ [0, 1, 3] ]).reshape(1, 3)
# stress_std = torch.tensor(np.array(stats['std_stress'])[ [0, 1, 3] ]).reshape(1, 3)
# === Load dataset ===
dataset = StressPredictionDataset(npz_path=npz_path, metadata_path=metadata_path, downsample=False)
loader = DataLoader(dataset, batch_size=8192, shuffle=False)

# === Load model ===
checkpoint = torch.load(f'{model_path}/trained_model.pth')
loss_history = checkpoint['loss_history']
model = MLPStressPredictor()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Collect all predictions and targets ===
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in loader:
        pred = model(x).view(-1, 4)
        all_preds.append(pred)
        all_targets.append(y)

preds = torch.cat(all_preds, dim=0)      # (N, 4)
targets = torch.cat(all_targets, dim=0)  # (N, 4)

# === Denormalize ===
if denormalize:
    preds = preds * stress_std + stress_mean
    targets = targets * stress_std + stress_mean


# === Reconstruct full tensors ===
# preds_tensor = reconstruct_tensor(preds)      # (N, 2, 2)
# targets_tensor = reconstruct_tensor(targets)  # (N, 2, 2)
preds_tensor = preds.view(-1, 2, 2)      # (N, 2, 2)
targets_tensor = targets.view(-1, 2, 2)  # (N, 2, 2)

# === Compute absolute error ===
errors = torch.abs(preds_tensor - targets_tensor)  # (N, 2, 2)

# === Plot histograms ===
labels = ['xx', 'xy', 'yx', 'yy']
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(errors[:, i, j].numpy(), bins=100, alpha=0.7, log=True)
        axs[i, j].set_title(f"Error Histogram: σ_{labels[i*2 + j]}")
        axs[i, j].set_xlabel('Absolute Error')
        axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{output_path}/stress_error_histograms.png')

fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.yscale('log')
plt.title('Loss Curve')
plt.grid(True)
plt.savefig(f'{output_path}/loss_curve.png')

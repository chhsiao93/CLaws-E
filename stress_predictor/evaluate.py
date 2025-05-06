import torch
import numpy as np
from model import LearnedConstitutiveModel
from trajectory_dataset import StressPredictionDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# Config
npz_path = "data/train.npz"
metadata_path = "data/metadata.json"
output_path = "results/"
model_path = "models/"


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
model = LearnedConstitutiveModel()
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

preds = torch.cat(all_preds, dim=0) # (N, 4)
targets = torch.cat(all_targets, dim=0) # (N, 4)

# === Denormalize ===
denorm_preds = preds * stress_std + stress_mean
denorm_targets = targets * stress_std + stress_mean

# === Compute absolute error ===
errors = torch.abs(preds - targets).view(-1, 2, 2)  # (N, 2, 2)
denorm_errors = torch.abs(denorm_preds - denorm_targets).view(-1, 2, 2)  # (N, 2, 2)

# === Plot histograms ===
labels = ['xx', 'xy', 'yx', 'yy']
# Plot normalized stress histograms
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(errors[:, i, j].numpy(), bins=100, alpha=0.7, log=True)
        axs[i, j].set_title(f"Error Histogram: σ_{labels[i*2 + j]}")
        axs[i, j].set_xlabel('Absolute Error (Normalized)')
        axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{output_path}/stress_error_histograms_normalized.png')
# Plot denormalized stress histograms
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
for i in range(2):
    for j in range(2):
        axs[i, j].hist(denorm_errors[:, i, j].numpy(), bins=100, alpha=0.7, log=True)
        axs[i, j].set_title(f"Error Histogram: σ_{labels[i*2 + j]}")
        axs[i, j].set_xlabel('Absolute Error (Denormalized)')
        axs[i, j].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{output_path}/stress_error_histograms_denormalized.png')

# === Plot loss curve ===
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.yscale('log')
plt.title('Loss Curve')
plt.grid(True)
plt.savefig(f'{output_path}/loss_curve.png')

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from trajectory_dataset import StressPredictionDataset
import json

# Config
npz_path = "data/train.npz"
metadata_path = "data/metadata.json"
output_path = "results/"

# === Load metadata for denormalization ===
with open(metadata_path, 'r') as f:
    stats = json.load(f)['train']
stress_mean = np.array(stats['mean_stress'])[ [0, 1, 3] ]
stress_std = np.array(stats['std_stress'])[ [0, 1, 3] ]

# === Load dataset & dataloader ===
dataset = StressPredictionDataset(npz_path=npz_path, metadata_path=metadata_path, downsample=True)
loader = DataLoader(dataset, batch_size=8192, shuffle=False)

# === Collect all y (stress targets) ===
all_stresses = []
for _, y, _ in loader:
    all_stresses.append(y)
all_stresses = torch.cat(all_stresses, dim=0).numpy()  # shape: (N, )
# get the first 3 components of stress
all_stresses = all_stresses[:, [0, 1, 3]]  # shape: (N, 3)
print(all_stresses.shape)

# === Denormalize ===
all_stresses = all_stresses * stress_std + stress_mean

# === Plot histograms ===
labels = ['σ_xx', 'σ_xy', 'σ_yy']
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(all_stresses[:, i], bins=100, alpha=0.75, color='skyblue', log=True, density=True)
    plt.title(f"{labels[i]} Distribution")
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(f"{output_path}/stress_distribution_downsample.png")

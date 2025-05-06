import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from trajectory_dataset import StressPredictionDataset
from model import MLPStressPredictor
from tqdm import tqdm
import json

# Config
npz_path = "/Users/clawsy/Documents/GitHub/CLaws-E/taichi_mpm/data/sand45/train/train.npz"
metadata_path = "/Users/clawsy/Documents/GitHub/CLaws-E/taichi_mpm/data/sand45/train/metadata.json"
model_path = "models/"
batch_size = 1024
epochs = 1000
lr = 1e-3

# Load dataset and prepare dataloader
dataset = StressPredictionDataset(npz_path=npz_path, metadata_path=metadata_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Infer input/output size
input_dim = dataset[0][0].shape[0]
output_dim = dataset[0][1].shape[0]

# Initialize model
model = MLPStressPredictor(input_dim=input_dim, output_dim=output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print summary
print("🚀 Training Setup")
print(f"  Total samples           : {len(dataset)}")
print(f"  Batch size              : {batch_size}")
print(f"  Batches per epoch       : {len(dataloader)}")
print(f"  Input feature dimension : {input_dim}")
print(f"  Output dimension        : {output_dim}")
print(f"  Using device            : {device}")
print(f"  Number of epochs        : {epochs}")
print()

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

loss_history = []
for epoch in range(epochs):
    model.train()
    total_loss = 0

    # Wrap your dataloader with tqdm for progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")

# Save model weights and metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'output_dim': output_dim,
    'loss_history': loss_history
}, f'{model_path}/trained_model.pth')

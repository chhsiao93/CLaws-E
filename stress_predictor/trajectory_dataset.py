import numpy as np
import torch
from torch.utils.data import Dataset
import json

class StressPredictionDataset(Dataset):
    def __init__(self, npz_path, metadata_path, downsample=True, max_samples_per_bin=10000, n_bins=10):
        def extract_symmetric_components(stress):
            sigma_xx = stress[..., 0]
            sigma_xy = stress[..., 1]
            sigma_yx = stress[..., 2]
            sigma_yy = stress[..., 3]
            return np.stack([sigma_xx, 0.5 * (sigma_xy + sigma_yx), sigma_yy], axis=-1)

        self.inputs = []
        self.targets = []
        # self.rotations = []

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        stats = metadata['train']

        # Load normalization stats
        self.vel_mean = np.array(stats['mean_velocity'])
        self.vel_std = np.array(stats['std_velocity'])
        self.stress_mean = np.array(stats['mean_stress'])#[ [0, 1, 3] ]
        self.stress_std = np.array(stats['std_stress'])#[ [0, 1, 3] ]

        # Temp storage
        all_inputs = []
        all_targets = []
        # all_rotations = []
        # Load npz data
        data = np.load(npz_path, allow_pickle=True)
        for key in data.files:
            traj = data[key].item()
            positions = traj['positions']
            stress = traj['stress']
            F = traj['deformation_gradient']

            velocity = np.zeros_like(positions)
            velocity[:-1] = positions[1:] - positions[:-1]

            velocity = (velocity - self.vel_mean) / self.vel_std
            # stress = extract_symmetric_components(stress)
            stress = (stress - self.stress_mean) / self.stress_std
            next_stress = stress[1:]

        for t in range(positions.shape[0] - 1):
            # v = velocity[t]
            # s = stress[t]
            # f = F[t]
            # x = np.concatenate([v, s, f], axis=-1)
            # y = next_stress[t]
            f = F[t]  # shape (N, 4)
            # f_matrix = f.reshape(-1, 2, 2)

            # # SVD
            # u, svals, vh = np.linalg.svd(f_matrix)
            # r = np.matmul(u, vh)  # N x 2 x 2

            # # Compute C
            # c = np.einsum('nij,njk->nik', f_matrix.transpose(0, 2, 1), f_matrix)  # N x 2 x 2
            # c_flat = c.reshape(-1, 4)

            # # det(F)
            # det_f = np.linalg.det(f_matrix)

            # # Input x: [C11, C12, C21, C22, σ1, σ2, detF]
            # x = np.concatenate([c_flat, svals, det_f[:, None]], axis=-1)  # N x 7
            x = np.concatenate([f])

            all_inputs.append(x)
            all_targets.append(next_stress[t])
            # all_rotations.append(r)

        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # all_rotations = np.concatenate(all_rotations, axis=0)

        if downsample:
            target_magnitudes = np.linalg.norm(all_targets, axis=-1)
            bins = np.linspace(target_magnitudes.min(), target_magnitudes.max(), n_bins + 1)
            bin_indices = np.digitize(target_magnitudes, bins)

            selected_indices = []
            for b in range(1, n_bins + 1):
                idx = np.where(bin_indices == b)[0]
                if len(idx) > max_samples_per_bin:
                    idx = np.random.choice(idx, max_samples_per_bin, replace=False)
                selected_indices.extend(idx)

            selected_indices = np.array(selected_indices)
            all_inputs = all_inputs[selected_indices]
            all_targets = all_targets[selected_indices]

        self.inputs = torch.tensor(all_inputs, dtype=torch.float32)
        self.targets = torch.tensor(all_targets, dtype=torch.float32)
        # self.rotations = torch.tensor(all_rotations, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

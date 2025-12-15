import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ProteinDataset(Dataset):
    def __init__(self, npz_path, max_points=128):
        data = np.load(npz_path, allow_pickle=True)
        self.structures = data["structures"]
        self.labels = data["labels"]
        self.max_points = max_points

        self.label_to_indices = {}
        for i, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(i)

    def _sample_points(self, coords):
        if len(coords) >= self.max_points:
            idx = np.random.choice(len(coords), self.max_points, replace=False)
            return coords[idx]
        else:
            pad = self.max_points - len(coords)
            return np.pad(coords, ((0, pad), (0, 0)), mode="constant")

    def __getitem__(self, idx):
        x1 = self._sample_points(self.structures[idx])
        y = self.labels[idx]

        if random.random() < 0.5:
            idx2 = random.choice(self.label_to_indices[y])
            label = 1
        else:
            idx2 = random.choice([
                i for l, inds in self.label_to_indices.items()
                if l != y for i in inds
            ])
            label = 0

        x2 = self._sample_points(self.structures[idx2])

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.structures)

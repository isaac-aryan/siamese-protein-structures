import numpy as np
import torch
from torch.utils.data import Dataset
import random


class ProteinDataset(Dataset):
    """
    Generates positive and negative pairs from protein embeddings.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)

        self.X = data["embeddings"].astype(np.float32)
        self.y = data["labels"]

        self.label_to_indices = {}
        for idx, label in enumerate(self.y):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.indices = list(range(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]
        label = self.y[idx]

        # 50% positive, 50% negative
        if random.random() < 0.5:
            # Positive pair
            idx2 = random.choice(self.label_to_indices[label])
            target = 1
        else:
            # Negative pair
            neg_label = random.choice(
                [l for l in self.label_to_indices if l != label]
            )
            idx2 = random.choice(self.label_to_indices[neg_label])
            target = 0

        x2 = self.X[idx2]

        return (
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.tensor(target, dtype=torch.float32),
        )

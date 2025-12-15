import torch
import numpy as np
from dataset import ProteinDataset
from model import SiameseNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ProteinDataset("data/processed/structures.npz")
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pt"))
model.eval()

embeddings = []
labels = []

with torch.no_grad():
    for x1, x2, y in dataset:
        x1 = x1.unsqueeze(0).to(DEVICE)
        z = model.forward_once(x1)
        embeddings.append(z.cpu().numpy()[0])
        labels.append(y.item())

np.savez(
    "data/processed/embeddings.npz",
    embeddings=np.array(embeddings),
    labels=np.array(labels),
)

print("Saved embeddings")

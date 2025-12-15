import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
PDB_DIR = "data/pdb"
META = "data/metadata.csv"
OUT = "data/processed/structures.npz"

os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Load metadata
# -----------------------------
df = pd.read_csv(META)
parser = PDBParser(QUIET=True)

embeddings = []
labels = []


# -----------------------------
# Feature utilities
# -----------------------------
def compute_features(coords):
    """
    coords: (L, 3)

    returns: (L, 6)
    """
    centroid = coords.mean(axis=0)
    dist_centroid = np.linalg.norm(coords - centroid, axis=1, keepdims=True)
    coord_norm = np.linalg.norm(coords, axis=1, keepdims=True)

    displacement = np.zeros_like(coords)
    displacement[1:] = coords[1:] - coords[:-1]
    disp_norm = np.linalg.norm(displacement, axis=1, keepdims=True)

    return np.concatenate(
        [coords, dist_centroid, coord_norm, disp_norm],
        axis=1,
    )


def pool_embedding(emb):
    """
    Global mean pooling
    emb: (L, D)
    returns: (D,)
    """
    return emb.mean(axis=0)


# -----------------------------
# Main loop
# -----------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):
    pdb_file = os.path.join(PDB_DIR, f"pdb{row.pdb_id}.ent")

    if not os.path.exists(pdb_file):
        continue

    structure = parser.get_structure(row.pdb_id, pdb_file)
    coords = []

    for model in structure:
        for chain in model:
            if chain.id != row.chain_id:
                continue
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord())

    if len(coords) < 30:
        continue

    coords = np.array(coords, dtype=np.float32)
    coords -= coords.mean(axis=0)

    residue_features = compute_features(coords)   # (L, 6)
    protein_embedding = pool_embedding(residue_features)  # (6,)

    embeddings.append(protein_embedding)
    labels.append(row.fold_id)


# -----------------------------
# Save
# -----------------------------
X = np.stack(embeddings)
y = np.array(labels)

np.savez(
    OUT,
    embeddings=X,
    labels=y
)

print(f"Saved {X.shape[0]} proteins")
print(f"Embedding shape: {X.shape}")
print(f"Saved to {OUT}")

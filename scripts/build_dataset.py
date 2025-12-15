import os
import re
import pandas as pd
from Bio.PDB import PDBList
from tqdm import tqdm


# CONFIG (do not change)
N_FOLDS = 40
DOMAINS_PER_FOLD = 25

SCOPE_FILE = "../data/raw/dir.cla.scope.txt"
PDB_DIR = "../data/pdb"
OUTPUT_CSV = "../data/metadata.csv"

os.makedirs(PDB_DIR, exist_ok=True)


# Parse SCOPe
records = []

with open(SCOPE_FILE) as f:
    for line in f:
        if line.startswith("#"):
            continue

        domain = line.split()[0]
        pdb_id = domain[1:5]
        chain = domain[5].upper()

        match = re.search(r"cf=(\d+)", line)
        if not match:
            continue

        fold_id = match.group(1)

        records.append({
            "domain_id": domain,
            "pdb_id": pdb_id,
            "chain_id": chain,
            "fold_id": fold_id
        })

df = pd.DataFrame(records)


# Subsample folds-
df = (
    df.groupby("fold_id")
      .head(DOMAINS_PER_FOLD)
      .reset_index(drop=True)
)

selected_folds = df["fold_id"].unique()[:N_FOLDS]
df = df[df["fold_id"].isin(selected_folds)]

df.to_csv(OUTPUT_CSV, index=False)

print(f"Final dataset:")
print(f"  Folds: {df['fold_id'].nunique()}")
print(f"  Structures: {len(df)}")

# Download PDBs
pdbl = PDBList()
for pdb in tqdm(df["pdb_id"].unique(), desc="Downloading PDBs"):
    pdbl.retrieve_pdb_file(
        pdb,
        pdir=PDB_DIR,
        file_format="pdb",
        overwrite=False
    )

# Siamese Networks for Protein Structure Similarity

## Project Aim

This project learns a **continuous, alignment-free embedding space for protein 3D structures**, such that proteins with similar folds are close together and structurally dissimilar proteins are far apart.

Instead of relying on classical pairwise structural alignment algorithms, the system trains a **Siamese neural network** directly on protein atomic coordinates. Once trained, structural similarity can be computed efficiently using simple distance metrics in embedding space.

The focus of the project is representation learning, geometric intuition, and biological interpretability rather than raw benchmark performance.

---

## Biological Motivation

Protein function is determined primarily by **three-dimensional structure**, which is often more conserved than amino-acid sequence across evolution. Proteins with low sequence identity can still share highly similar folds and perform related biological roles.

To study this, the project uses **SCOPe (Structural Classification of Proteins – extended)**, which hierarchically classifies protein domains into classes, folds, superfamilies, and families. The **fold level** is used as the notion of structural similarity during training.

Learning a smooth structural embedding aligns well with biological reality: protein fold space is continuous, with gradual geometric transitions rather than hard categorical boundaries.

---

## Why Learn Structure Embeddings?

Traditional protein structure comparison relies on explicit 3D alignment methods such as TM-align or DALI. While accurate, these methods are computationally expensive and scale poorly for large databases.

This project instead learns a mapping

[
f(\text{protein structure}) \rightarrow \mathbb{R}^d
]

such that:

* structurally similar proteins map close together
* structurally dissimilar proteins are far apart

After training, similarity queries reduce to fast vector comparisons, enabling scalable structural search without alignment.

---

## Machine Learning Approach

### Siamese Network

A **Siamese neural network** is trained on pairs of protein structures. Each branch shares weights and encodes one protein into a fixed-dimensional embedding.

The network is trained using **CosineEmbeddingLoss**:

* positive pairs: proteins from the same SCOPe fold
* negative pairs: proteins from different folds

This is a form of **metric learning**, where the model learns a meaningful distance function rather than performing explicit classification.

---

### Input Representation

Each protein structure is represented as a **point cloud of Cα atoms**:

* one 3D coordinate per residue
* structures are variable length
* coordinates are mean-centred to remove translation effects

This representation preserves global geometry while remaining simple and efficient.

---

## Intuition Linking ML and Biology

Protein fold space is continuous rather than discrete. By training with a contrastive objective instead of classification, the model:

* preserves smooth geometric transitions between related structures
* avoids hard decision boundaries
* learns embeddings that reflect real structural variation

This mirrors how proteins evolve and how fold similarity is understood biologically.

---

## Visualising the Embedding Space

After training, embeddings are projected into 2D using **t-SNE**.

### Interpretation of Results

The t-SNE plot shows:

* smooth, curved manifolds rather than tight clusters
* separation between major fold groups
* local neighbourhoods of structurally similar proteins

The sinusoidal or arc-like shapes arise because:

* the learned embedding space is continuous and low-dimensional
* t-SNE preserves local neighbourhoods but distorts global geometry
* dominant latent factors (such as size and compactness) are unfolded into curves

This behaviour indicates the model has learned a meaningful structural manifold rather than memorising labels.

---

## Repository Structure

```
.
├── configs/                 # Configuration files (future extensions)
├── data/
│   ├── raw/
│   │   └── dir.cla.scope.txt   # Raw SCOPe classification file
│   ├── pdb/                   # Downloaded PDB structure files
│   ├── processed/
│   │   ├── structures.npz     # Preprocessed protein structures
│   │   └── embeddings.npz     # Learned protein embeddings
│   └── metadata.csv           # Metadata with fold labels
├── notebooks/
│   └── embeddings_viz.ipynb   # t-SNE visualisation notebook
├── scripts/
│   ├── build_dataset.py       # Dataset and metadata construction
│   ├── process_structures.py  # PDB parsing and coordinate extraction
│   ├── dataset.py             # PyTorch Dataset for Siamese training
│   ├── model.py               # Siamese network architecture
│   └── extract_embeddings.py  # Embedding extraction script
├── model.pt                   # Trained model weights
├── training.ipynb             # Training notebook
├── main.py                    # Scripted training entry point
├── README.md
└── pyproject.toml
```

---

## Current Status

At this stage, the project includes:

* a fully trained Siamese network on protein structures
* biologically meaningful structure embeddings
* visual evidence of a learned structural manifold

The system is suitable for protein similarity search, embedding analysis, and further research-oriented extensions.

---

## Possible Next Steps

* Quantitative evaluation using retrieval metrics (Precision@K)
* Approximate nearest-neighbour search with FAISS
* Graph-based or equivariant encoders
* Interactive protein similarity search interface

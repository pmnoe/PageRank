import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.pagerank import build_transition_matrix

DATA = "data"

edges = pd.read_csv(f"{DATA}/edges.csv")
pages = pd.read_csv(f"{DATA}/pages.csv", index_col="id")
n = len(pages)

M = build_transition_matrix(edges, n)

# ── Sparsity ───────────────────────────────────────────────────────────────────
nnz = M.nnz
density = nnz / (n * n)
print(f"Nodes: {n},  Edges: {nnz}")
print(f"Density: {density:.6f}  ({density * 100:.4f}%)")

# ── Strong connectivity / irreducibility ───────────────────────────────────────
n_scc, labels = connected_components(M, directed=True, connection="strong")
comp_sizes = np.bincount(labels)
singletons = int((comp_sizes == 1).sum())

print(f"\nStrongly connected components: {n_scc}")
print(f"Largest SCC: {comp_sizes.max()} nodes  ({comp_sizes.max() / n * 100:.1f}%)")
print(f"Singleton SCCs: {singletons}")
print(f"Irreducible: {n_scc == 1}")

# ── Dangling nodes ─────────────────────────────────────────────────────────────
out_deg = np.asarray(M.sum(axis=0)).ravel()
dangling = int((out_deg == 0).sum())
print(f"\nDangling nodes (zero out-degree): {dangling}")

# ── Link counts ────────────────────────────────────────────────────────────────
src_counts = edges["src"].value_counts()
median_links = src_counts.median()
max_page_id = src_counts.idxmax()
max_links = src_counts.max()
max_page_title = pages.loc[max_page_id, "title"]
print(f"\nMedian outgoing links per page: {median_links:.1f}")
print(f"Page with most links: '{max_page_title}' ({max_links} links)")

import numpy as np
import pandas as pd

from src.pagerank import build_transition_matrix
from src.experiments import convergence_experiment, personalization_experiment
from src.plotting import plot_sparsity, plot_convergence

DATA = "data"
FIGURES = "results/figures"
TABLES = "results/tables"

# ── Load data ──────────────────────────────────────────────────────────────────
edges = pd.read_csv(f"{DATA}/edges.csv")
pages = pd.read_csv(f"{DATA}/pages.csv", index_col="id")
n = len(pages)
M = build_transition_matrix(edges, n)
v_uniform = np.ones(n) / n

# ── Sparsity plot ──────────────────────────────────────────────────────────────
plot_sparsity(M, pages, save_path=f"{FIGURES}/sparsity_pattern.pdf")
print("Saved sparsity_pattern.pdf")

# ── Experiment 1: convergence under different α ────────────────────────────────
alphas = [0.75, 0.85, 0.95]
conv = convergence_experiment(M, alphas, v_uniform, max_iter=100)
plot_convergence(conv, save_path=f"{FIGURES}/convergence.pdf")
print("Saved convergence.pdf")

# ── Experiment 2: personalization vector ──────────────────────────────────────
pers = personalization_experiment(M, pages.reset_index(), alpha=0.85, top_n=10)
pers.index += 1
pers.to_csv(f"{TABLES}/personalization.csv")
print("Saved personalization.csv")

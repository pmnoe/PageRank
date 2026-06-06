import numpy as np
import pandas as pd

from src.pagerank import build_transition_matrix
from src.experiments import convergence_experiment, personalization_experiment, timing_experiment
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

# ── Experiment 3: timing — direct vs. iterative ────────────────────────────────
timing = timing_experiment(M, n)
print(f"\nTiming (α=0.85, tol=1e-8, n={n}):")
print(f"  Direct solve:    {timing['t_direct_s']*1000:.1f} ms")
print(f"  Iterative solve: {timing['t_iterative_s']*1000:.1f} ms  ({timing['n_iter']} iterations)")
print(f"  Speedup (direct/iterative): {timing['speedup']:.2f}x")
pd.DataFrame([{
    "n": n,
    "alpha": 0.85,
    "tol": 1e-8,
    "t_direct_ms": round(timing["t_direct_s"] * 1000, 2),
    "t_iterative_ms": round(timing["t_iterative_s"] * 1000, 2),
    "speedup": round(timing["speedup"], 2),
    "n_iter": timing["n_iter"],
}]).to_csv(f"{TABLES}/timing.csv", index=False)
print("Saved timing.csv")

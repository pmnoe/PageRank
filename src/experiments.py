import time

import time

import numpy as np
import pandas as pd
from src.pagerank import solve_direct, solve_iterative


def timing_experiment(M, n, alpha=0.85, tol=1e-8, n_trials=5):
    """
    Compare wall-clock time of direct solve vs. iterative solve to ||p-p*||_1 < tol
    with uniform teleportation.  Returns mean times, speedup, and iteration count.
    """
    v = np.ones(n) / n

    t0 = time.perf_counter()
    for _ in range(n_trials):
        solve_direct(M, alpha, v)
    t_direct = (time.perf_counter() - t0) / n_trials

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _, history = solve_iterative(M, alpha, v, tol=tol)
    t_iterative = (time.perf_counter() - t0) / n_trials

    return {
        "t_direct_s": t_direct,
        "t_iterative_s": t_iterative,
        "speedup": t_direct / t_iterative,
        "n_iter": len(history) - 1,
    }


def convergence_experiment(M, alphas, v, max_iter=300):
    """
    For each alpha, solve for p* directly then track ||p^(k) - p*||_1 per iteration.
    Returns dict: alpha -> array of errors.
    """
    results = {}
    for alpha in alphas:
        p_star = solve_direct(M, alpha, v)
        _, history = solve_iterative(M, alpha, v, tol=0, max_iter=max_iter)
        results[alpha] = np.array([np.linalg.norm(p - p_star, 1) for p in history])
    return results


def ranking_experiment(pages, p, top_n=10):
    """Top-N pages by PageRank score for alpha=0.85, uniform v."""
    return (
        pages.assign(pagerank=p)
        .sort_values("pagerank", ascending=False)
        .head(top_n)[["title", "category", "pagerank"]]
        .reset_index(drop=True)
    )


def personalization_experiment(M, pages, alpha=0.85, top_n=20):
    """
    Compare uniform PageRank vs. topic-biased PageRank for each category.
    Returns a DataFrame with columns: uniform, <category>, ...
    """
    n = len(pages)

    def top_titles(v):
        p, _ = solve_iterative(M, alpha, v)
        return (
            pages.assign(pagerank=p)
            .sort_values("pagerank", ascending=False)
            .head(top_n)["title"]
            .reset_index(drop=True)
        )

    FEATURED = [
        "Category:Probability_theory",
        "Category:Geometry",
        "Category:Functional_analysis",
        "Category:Number_theory",
    ]

    result = pd.DataFrame({"Uniform": top_titles(np.ones(n) / n)})

    for cat in FEATURED:
        mask = (pages["category"] == cat).astype(float).values
        label = cat.replace("Category:", "").replace("_", " ")
        result[label] = top_titles(mask / mask.sum())

    return result

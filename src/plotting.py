import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PAPER_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "savefig.dpi": 300,
}

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def plot_sparsity(M, pages, save_path=None):
    mpl.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spy(M, markersize=1.5, marker=".", color="black",
           markeredgewidth=0, alpha=1, rasterized=True)
    ax.set_xlabel("Source page")
    ax.set_ylabel("Destination page")
    ax.set_title("Nonzero pattern of $M$")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_convergence(convergence_results, save_path=None):
    mpl.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    actual_handles, bound_handles = [], []
    for i, (alpha, errors) in enumerate(sorted(convergence_results.items())):
        solid, = ax.semilogy(
            errors, color=COLORS[i], linewidth=1.6,
            label=f"$\\alpha = {alpha:.2f}$",
        )
        k = np.arange(len(errors))
        dashed, = ax.semilogy(
            k, (alpha ** k) * errors[0],
            color=COLORS[i], linewidth=1.0, linestyle="--", alpha=0.7,
            label=f"Theor. rate, $\\alpha = {alpha:.2f}$",
        )
        actual_handles.append(solid)
        bound_handles.append(dashed)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|p^{(k)} - p^*\|_1$")
    ax.set_title("PageRank convergence")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.grid(True, linewidth=0.4, linestyle="--", color="0.82")
    ax.tick_params(direction="in", which="both", top=True, right=True, length=4)
    ax.legend(
        handles=actual_handles + bound_handles,
        labels=[h.get_label() for h in actual_handles] + ["Theoretical rate"] * 3,
        ncol=2, loc="lower left", fontsize=9,
        columnspacing=1.0, handlelength=1.6,
    )
    ax.set_xlim(left=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

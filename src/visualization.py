import os
from typing import Dict, List

import matplotlib.pyplot as plt


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def plot_loss_curve(loss_history: List[float], save_path: str) -> None:
    """
    Plot training loss versus iterations for the Adaptive LASSO optimizer.

    Parameters
    ----------
    loss_history : list of float
        Values of the objective (MSE + L1 penalty) at each iteration.
    save_path : str
        Path where the generated plot should be saved.
    """
    if not loss_history:
        raise ValueError("loss_history is empty; nothing to plot.")

    _ensure_parent_dir(save_path)

    iters = list(range(1, len(loss_history) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(iters, loss_history, marker="o", linewidth=1.5, markersize=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value (MSE + L1)")
    plt.title("Adaptive LASSO: Loss vs. Iterations")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sparsity_curve(sparsity_history: List[int], total_features: int, save_path: str) -> None:
    """
    Plot number of non-zero coefficients versus iterations.

    Parameters
    ----------
    sparsity_history : list of int
        Number of non-zero coefficients at each iteration.
    total_features : int
        Total number of features in the design matrix.
    save_path : str
        Path where the generated plot should be saved.
    """
    if not sparsity_history:
        raise ValueError("sparsity_history is empty; nothing to plot.")

    _ensure_parent_dir(save_path)

    iters = list(range(1, len(sparsity_history) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(iters, sparsity_history, marker="o", linewidth=1.5, markersize=2, color="tab:orange")
    plt.xlabel("Iteration")
    plt.ylabel("Non-zero coefficients")
    plt.title("Adaptive LASSO: Sparsity vs. Iterations")
    plt.ylim(bottom=0, top=max(total_features, max(sparsity_history)) * 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    total_features: int,
    save_path: str,
) -> None:
    """
    Create a bar chart comparing baseline models and Adaptive LASSO.

    Parameters
    ----------
    metrics : dict
        Mapping from model name to a dictionary with keys:
        - 'MSE'
        - 'Non-zero Features'
    total_features : int
        Total number of features after preprocessing.
    save_path : str
        Path where the generated plot should be saved.
    """
    if not metrics:
        raise ValueError("metrics is empty; nothing to plot.")

    _ensure_parent_dir(save_path)

    model_names = list(metrics.keys())
    mses = [metrics[m]["MSE"] for m in model_names]
    non_zero = [metrics[m]["Non-zero Features"] for m in model_names]

    x_positions = range(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Test MSE
    axes[0].bar(x_positions, mses, color="tab:blue", alpha=0.8)
    axes[0].set_xticks(list(x_positions))
    axes[0].set_xticklabels(model_names, rotation=30, ha="right")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Model Comparison: Test MSE")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Right: Number of non-zero coefficients
    axes[1].bar(x_positions, non_zero, color="tab:green", alpha=0.8)
    axes[1].set_xticks(list(x_positions))
    axes[1].set_xticklabels(model_names, rotation=30, ha="right")
    axes[1].set_ylabel("Non-zero coefficients")
    axes[1].set_title("Model Comparison: Sparsity")
    axes[1].set_ylim(bottom=0, top=max(total_features, max(non_zero)) * 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


__all__ = ["plot_loss_curve", "plot_sparsity_curve", "plot_model_comparison"]


import json
import os
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error

from adaptive_lasso import AdaptiveLasso
from baselines import compute_baseline_metrics
from data_preprocessing import get_processed_data
from visualization import (
    plot_loss_curve,
    plot_sparsity_curve,
    plot_model_comparison,
)


def get_project_root() -> str:
    """Return the absolute path to the project root (one level above src)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_adaptive_lasso() -> Dict[str, Dict[str, float]]:
    """
    Train the Adaptive LASSO model, compare against baselines, and save results.

    This function:
    - Loads the preprocessed House Prices dataset
    - Trains AdaptiveLasso with proximal gradient descent
    - Computes test MSE and sparsity
    - Trains baseline models (Linear, Ridge, LASSO)
    - Saves metrics to results/metrics.json
    - Saves training curves and comparison plots to the results/ folder

    Returns
    -------
    all_results : dict
        Dictionary mapping model names to metric dictionaries.
    """
    print("Loading processed dataset via preprocessing pipeline...")
    X_train, X_test, y_train, y_test, _ = get_processed_data(save_to_disk=False)

    # Train Adaptive LASSO
    print("\n" + "=" * 50)
    print("Training Adaptive LASSO (custom proximal gradient optimizer)")
    print("=" * 50)

    adaptive = AdaptiveLasso(
        learning_rate=0.01,
        lambda0=0.5,
        n_iters=3000,
        schedule="inverse_sqrt",  # or 'exp_decay', 'constant'
        decay_k=0.01,
        tol=1e-6,
        fit_intercept=True,
        random_state=42,
        verbose=True,
    )
    adaptive.fit(X_train, y_train)

    y_pred_adaptive = adaptive.predict(X_test)
    adaptive_mse = mean_squared_error(y_test, y_pred_adaptive)
    # Use the same threshold as baselines for counting non-zero coefficients
    adaptive_nonzero = int(np.sum(np.abs(adaptive.coef_) > 1e-5))

    diagnostics = adaptive.get_training_diagnostics()

    # Train baseline models on the same processed data
    print("\n" + "=" * 50)
    print("Training baseline models (Linear, Ridge, Standard LASSO)")
    print("=" * 50)
    baseline_results, total_features = compute_baseline_metrics(
        X_train, X_test, y_train, y_test
    )

    # Combine baseline metrics with Adaptive LASSO metrics
    all_results: Dict[str, Dict[str, float]] = dict(baseline_results)
    all_results["Adaptive LASSO"] = {
        "MSE": adaptive_mse,
        "Non-zero Features": adaptive_nonzero,
    }

    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, "metrics.json")
    metrics_payload = {
        "total_features": int(total_features),
        "adaptive_lasso_config": {
            "learning_rate": adaptive.learning_rate,
            "lambda0": adaptive.lambda0,
            "n_iters": adaptive.n_iters,
            "schedule": adaptive.schedule,
            "decay_k": adaptive.decay_k,
            "tol": adaptive.tol,
            "fit_intercept": adaptive.fit_intercept,
        },
        "models": {
            name: {
                "MSE": float(metrics["MSE"]),
                "Non-zero Features": int(metrics["Non-zero Features"]),
            }
            for name, metrics in all_results.items()
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate and save plots
    loss_curve_path = os.path.join(results_dir, "loss_curve.png")
    sparsity_curve_path = os.path.join(results_dir, "sparsity_curve.png")
    model_comparison_path = os.path.join(results_dir, "model_comparison.png")

    print(f"Saving loss curve to {loss_curve_path}")
    plot_loss_curve(diagnostics["loss_history"], save_path=loss_curve_path)

    print(f"Saving sparsity curve to {sparsity_curve_path}")
    plot_sparsity_curve(
        diagnostics["sparsity_history"],
        total_features=total_features,
        save_path=sparsity_curve_path,
    )

    print(f"Saving model comparison plot to {model_comparison_path}")
    plot_model_comparison(
        metrics=all_results,
        total_features=total_features,
        save_path=model_comparison_path,
    )

    # Print summary to console
    print("\n" + "=" * 50)
    print("MODEL COMPARISON (including Adaptive LASSO)")
    print("=" * 50)
    print(f"{'Model':<20} | {'Test MSE':<12} | {'Non-zero Features'}")
    print("-" * 50)
    for model_name, metrics in all_results.items():
        print(
            f"{model_name:<20} | {metrics['MSE']:<12.4f} | "
            f"{metrics['Non-zero Features']} / {total_features}"
        )
    print("=" * 50)

    return all_results


if __name__ == "__main__":
    train_adaptive_lasso()


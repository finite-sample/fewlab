"""
Comprehensive evaluation: Random sampling vs. FewLab methods

This script compares the efficiency of random sampling against various
FewLab sampling strategies for survey estimation tasks.

Methods compared:
- Random sampling (baseline)
- Deterministic A-optimal (items_to_label)
- Balanced sampling (balanced_fixed_size)
- Hybrid core+tail (core_plus_tail)
- Adaptive hybrid (adaptive_core_tail)

Metrics:
- Bias, Variance, RMSE of coefficient estimates
- Relative efficiency (variance ratio: random/method)
- Computational time
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add repo root to path before importing fewlab
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import FewLab methods after path setup
import fewlab  # noqa: E402
from fewlab.utils import (  # noqa: E402
    compute_g_matrix,
    compute_horvitz_thompson_weights,
)


@dataclass
class SimConfig:
    """Configuration for simulation experiments."""

    n_units: int = 500  # Number of units (e.g., users)
    n_items: int = 300  # Number of items (e.g., products)
    p_features: int = 6  # Number of features/covariates
    K_budget: int = 30  # Budget: number of items to label
    n_sims: int = 150  # Number of simulation runs
    lambda_counts: float = 15.0  # Poisson parameter for count generation
    signal_to_noise: float = 1.5  # Signal-to-noise ratio
    seed: int = 42
    count_heterogeneity: float = 5.0  # Variance in count generation


def generate_synthetic_data(
    cfg: SimConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate synthetic count and covariate data with realistic structure.

    The key insight: items with higher influence on regression should have
    heterogeneous counts across units, making them more informative.

    Returns
    -------
    counts : pd.DataFrame (n_units, n_items)
        Count matrix
    X : pd.DataFrame (n_units, p_features)
        Covariate matrix
    true_labels : np.ndarray (n_items,)
        True binary labels for items
    beta_true : np.ndarray (p_features,)
        True regression coefficients
    """
    n, m, p = cfg.n_units, cfg.n_items, cfg.p_features

    # Generate covariates (add intercept automatically)
    X_vals = rng.normal(size=(n, p - 1))
    X_vals = np.c_[np.ones(n), X_vals]
    X = pd.DataFrame(
        X_vals,
        index=[f"unit_{i}" for i in range(n)],
        columns=[f"x{i}" for i in range(p)],
    )

    # Generate true regression coefficients
    beta_true = rng.normal(scale=cfg.signal_to_noise, size=p)

    # Generate item labels with some correlation to item popularity
    item_base_quality = rng.normal(0, 1, size=m)
    true_labels = (item_base_quality > 0).astype(float)

    # Generate heterogeneous count matrix
    # Key: Create diversity in count patterns to benefit from optimal selection
    unit_activity = np.exp(rng.normal(0, 0.5, size=n))  # Some units more active
    item_popularity = np.exp(
        rng.normal(0, cfg.count_heterogeneity, size=m)
    )  # High variance in popularity

    # Counts depend on unit activity and item popularity
    expected_counts = unit_activity[:, None] * item_popularity[None, :]
    counts_vals = rng.poisson(
        expected_counts * cfg.lambda_counts / expected_counts.mean()
    )
    counts_vals = np.maximum(counts_vals, 1)  # Ensure at least 1 count per cell

    counts = pd.DataFrame(
        counts_vals, index=X.index, columns=[f"item_{j}" for j in range(m)]
    )

    return counts, X, true_labels, beta_true


def fit_weighted_ols(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Fit weighted OLS regression.

    Parameters
    ----------
    X : np.ndarray (n, p)
        Design matrix
    y : np.ndarray (n,)
        Response vector
    weights : np.ndarray (n,), optional
        Sample weights. If None, uses uniform weights.

    Returns
    -------
    beta_hat : np.ndarray (p,)
        Estimated coefficients
    """
    if weights is None:
        weights = np.ones(len(X))

    W = np.diag(weights)
    XtWX = X.T @ W @ X
    ridge = 1e-8 * np.eye(X.shape[1])
    XtWy = X.T @ W @ y

    return np.linalg.solve(XtWX + ridge, XtWy)


def sample_random(
    counts: pd.DataFrame, X: pd.DataFrame, K: int, rng: np.random.Generator
) -> tuple[pd.Index, pd.Series, float]:
    """Random sampling baseline."""
    start_time = time.time()

    # Simple random sample without replacement
    selected = pd.Index(rng.choice(counts.columns, size=K, replace=False))

    # Uniform inclusion probabilities
    pi = pd.Series(K / len(counts.columns), index=counts.columns)

    elapsed = time.time() - start_time
    return selected, pi, elapsed


def sample_deterministic_aopt(
    counts: pd.DataFrame, X: pd.DataFrame, K: int, rng: np.random.Generator
) -> tuple[pd.Index, pd.Series, float]:
    """Deterministic A-optimal selection."""
    start_time = time.time()

    selected = fewlab.items_to_label(counts, X, K)

    # Deterministic selection -> inclusion probabilities are 0 or 1
    pi = pd.Series(0.0, index=counts.columns)
    pi[selected] = 1.0

    elapsed = time.time() - start_time
    return pd.Index(selected), pi, elapsed


def sample_balanced(
    counts: pd.DataFrame, X: pd.DataFrame, K: int, rng: np.random.Generator
) -> tuple[pd.Index, pd.Series, float]:
    """Balanced fixed-size sampling."""
    start_time = time.time()

    # First compute A-optimal inclusion probabilities
    pi = fewlab.pi_aopt_for_budget(counts, X, K)

    # Compute g matrix
    g = compute_g_matrix(counts, X)

    # Balanced sampling
    seed = rng.integers(0, 2**31)
    selected = fewlab.balanced_fixed_size(pi, g, K, seed=seed)

    elapsed = time.time() - start_time
    return selected, pi, elapsed


def sample_hybrid_core_tail(
    counts: pd.DataFrame, X: pd.DataFrame, K: int, rng: np.random.Generator
) -> tuple[pd.Index, pd.Series, float]:
    """Hybrid core+tail sampling."""
    start_time = time.time()

    seed = rng.integers(0, 2**31)
    selected, pi, info = fewlab.core_plus_tail(counts, X, K, tail_frac=0.3, seed=seed)

    elapsed = time.time() - start_time
    return selected, pi, elapsed


def sample_adaptive_hybrid(
    counts: pd.DataFrame, X: pd.DataFrame, K: int, rng: np.random.Generator
) -> tuple[pd.Index, pd.Series, float]:
    """Adaptive hybrid sampling."""
    start_time = time.time()

    seed = rng.integers(0, 2**31)
    selected, pi, info = fewlab.adaptive_core_tail(counts, X, K, seed=seed)

    elapsed = time.time() - start_time
    return selected, pi, elapsed


def estimate_coefficients(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    true_labels: np.ndarray,
    selected: pd.Index,
    pi: pd.Series,
    item_names: pd.Index,
) -> np.ndarray:
    """
    Estimate regression coefficients using Horvitz-Thompson estimator.

    The goal is to estimate shares y_i for each unit, then regress X on y.
    """
    # Create labels series for selected items
    labels_series = pd.Series(true_labels, index=item_names)

    # Compute HT weights for selected items
    ht_weights = compute_horvitz_thompson_weights(pi, selected)

    # Compute weighted labels
    weighted_labels = labels_series[selected] * ht_weights

    # Estimate shares for each unit using HT estimator
    # y_i = (1/T_i) * sum_j (w_j * a_j * C_ij) for j in selected
    counts_selected = counts[selected].to_numpy()
    row_totals = counts.sum(axis=1).to_numpy()

    y_hat = (counts_selected @ weighted_labels.to_numpy()) / (row_totals + 1e-10)

    # Fit OLS: X -> y_hat
    X_vals = X.to_numpy()
    beta_hat = fit_weighted_ols(X_vals, y_hat)

    return beta_hat


def run_single_simulation(
    cfg: SimConfig,
    rng: np.random.Generator,
    methods: dict,
) -> dict[str, dict]:
    """Run a single simulation comparing all methods."""
    # Generate data
    counts, X, true_labels, beta_true = generate_synthetic_data(cfg, rng)

    results = {}

    for method_name, method_fn in methods.items():
        # Sample items
        selected, pi, elapsed = method_fn(counts, X, cfg.K_budget, rng)

        # Estimate coefficients
        beta_hat = estimate_coefficients(
            counts, X, true_labels, selected, pi, counts.columns
        )

        results[method_name] = {
            "beta_hat": beta_hat,
            "beta_true": beta_true,
            "time": elapsed,
            "n_selected": len(selected),
        }

    return results


def run_simulations(cfg: SimConfig) -> pd.DataFrame:
    """Run multiple simulations and aggregate results."""
    rng = np.random.default_rng(cfg.seed)

    methods = {
        "Random": sample_random,
        "Deterministic A-opt": sample_deterministic_aopt,
        "Balanced": sample_balanced,
        "Hybrid Core+Tail": sample_hybrid_core_tail,
        "Adaptive Hybrid": sample_adaptive_hybrid,
    }

    all_results = []

    print(f"Running {cfg.n_sims} simulations...")
    for sim_idx in range(cfg.n_sims):
        if (sim_idx + 1) % 20 == 0:
            print(f"  Simulation {sim_idx + 1}/{cfg.n_sims}")

        sim_results = run_single_simulation(cfg, rng, methods)

        for method_name, result in sim_results.items():
            beta_hat = result["beta_hat"]
            beta_true = result["beta_true"]
            error = beta_hat - beta_true

            for coef_idx in range(cfg.p_features):
                all_results.append(
                    {
                        "simulation": sim_idx,
                        "method": method_name,
                        "coefficient": coef_idx,
                        "beta_hat": beta_hat[coef_idx],
                        "beta_true": beta_true[coef_idx],
                        "error": error[coef_idx],
                        "squared_error": error[coef_idx] ** 2,
                        "time": result["time"],
                    }
                )

    return pd.DataFrame(all_results)


def compute_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute bias, variance, RMSE, and relative efficiency."""
    summary = (
        results_df.groupby(["method", "coefficient"])
        .agg(
            {
                "error": ["mean", "var"],
                "squared_error": "mean",
                "time": "mean",
            }
        )
        .reset_index()
    )

    summary.columns = ["method", "coefficient", "bias", "variance", "mse", "time"]
    summary["rmse"] = np.sqrt(summary["mse"])

    # Compute relative efficiency (variance ratio: random / method)
    random_variance = summary[summary["method"] == "Random"][
        ["coefficient", "variance"]
    ].rename(columns={"variance": "random_variance"})

    summary = summary.merge(random_variance, on="coefficient", how="left")
    summary["rel_efficiency"] = summary["random_variance"] / summary["variance"]
    summary["efficiency_gain_pct"] = (summary["rel_efficiency"] - 1) * 100

    return summary


def plot_results(summary: pd.DataFrame, cfg: SimConfig):
    """Create visualizations comparing methods."""
    methods = summary["method"].unique()

    # Set style
    sns.set_style("whitegrid")

    # 1. Variance comparison by coefficient
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Variance
    for method in methods:
        data = summary[summary["method"] == method]
        axes[0].plot(
            data["coefficient"], data["variance"], marker="o", label=method, linewidth=2
        )
    axes[0].set_xlabel("Coefficient Index")
    axes[0].set_ylabel("Variance")
    axes[0].set_title("Variance of Coefficient Estimates")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE
    for method in methods:
        data = summary[summary["method"] == method]
        axes[1].plot(
            data["coefficient"], data["rmse"], marker="o", label=method, linewidth=2
        )
    axes[1].set_xlabel("Coefficient Index")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE of Coefficient Estimates")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Relative efficiency
    for method in methods:
        if method != "Random":
            data = summary[summary["method"] == method]
            axes[2].plot(
                data["coefficient"],
                data["rel_efficiency"],
                marker="o",
                label=method,
                linewidth=2,
            )
    axes[2].axhline(y=1.0, color="black", linestyle="--", label="Random baseline")
    axes[2].set_xlabel("Coefficient Index")
    axes[2].set_ylabel("Relative Efficiency (Var_random / Var_method)")
    axes[2].set_title("Efficiency Gain over Random Sampling")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = BASE_DIR / "efficiency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.relative_to(REPO_ROOT)}")

    # 2. Average metrics across all coefficients
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    avg_summary = (
        summary.groupby("method")
        .agg(
            {
                "variance": "mean",
                "rmse": "mean",
                "rel_efficiency": "mean",
                "time": "mean",
            }
        )
        .reset_index()
    )

    # Average variance
    x_pos = np.arange(len(methods))
    axes[0].bar(x_pos, avg_summary["variance"])
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(avg_summary["method"], rotation=45, ha="right")
    axes[0].set_ylabel("Average Variance")
    axes[0].set_title("Average Variance Across All Coefficients")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Computation time
    axes[1].bar(x_pos, avg_summary["time"] * 1000)  # Convert to ms
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(avg_summary["method"], rotation=45, ha="right")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title("Average Computation Time")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = BASE_DIR / "average_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path.relative_to(REPO_ROOT)}")


def print_summary_table(summary: pd.DataFrame):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: Random vs. FewLab Methods")
    print("=" * 80)

    # Average across all coefficients
    avg_summary = (
        summary.groupby("method")
        .agg(
            {
                "bias": "mean",
                "variance": "mean",
                "rmse": "mean",
                "rel_efficiency": "mean",
                "efficiency_gain_pct": "mean",
                "time": "mean",
            }
        )
        .round(4)
    )

    print("\nAverage metrics across all coefficients:")
    print(avg_summary.to_string())

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)

    # Compute efficiency gains
    random_variance = avg_summary.loc["Random", "variance"]

    print(f"\nRandom sampling baseline variance: {random_variance:.4f}")
    print("\nEfficiency gains compared to random sampling:")

    for method in avg_summary.index:
        if method != "Random":
            variance = avg_summary.loc[method, "variance"]
            rel_eff = avg_summary.loc[method, "rel_efficiency"]
            gain_pct = (rel_eff - 1) * 100
            time_ms = avg_summary.loc[method, "time"] * 1000

            print(f"\n{method}:")
            print(f"  - Variance: {variance:.4f} ({gain_pct:+.1f}% efficiency gain)")
            print(f"  - Relative efficiency: {rel_eff:.2f}x")
            print(f"  - Computation time: {time_ms:.2f} ms")

            if rel_eff > 1:
                print(f"  --> {rel_eff:.2f}x MORE EFFICIENT than random sampling")
            else:
                print(f"  --> {1/rel_eff:.2f}x LESS EFFICIENT than random sampling")


def main():
    """Main evaluation script."""
    # Configuration
    cfg = SimConfig(
        n_units=1000,
        n_items=200,
        p_features=5,
        K_budget=40,
        n_sims=100,
        seed=42,
    )

    print("Evaluation Configuration:")
    print(f"  n_units: {cfg.n_units}")
    print(f"  n_items: {cfg.n_items}")
    print(f"  p_features: {cfg.p_features}")
    print(f"  K_budget: {cfg.K_budget}")
    print(f"  n_sims: {cfg.n_sims}")

    # Run simulations
    results_df = run_simulations(cfg)

    # Compute summary statistics
    summary = compute_summary_statistics(results_df)

    # Print results
    print_summary_table(summary)

    # Create visualizations
    plot_results(summary, cfg)

    # Save detailed results
    output_path = BASE_DIR / "simulation_results.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

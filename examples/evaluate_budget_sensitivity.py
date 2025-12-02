"""
Budget sensitivity analysis: How efficiency gains vary with budget size

This script evaluates how the efficiency of FewLab methods compared to random
sampling changes as the budget (number of items to sample) increases.

Key insight: Optimal methods should show larger gains with tighter budgets.
"""

from __future__ import annotations

import sys
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

# Local imports after path setup
import fewlab  # noqa: E402
from fewlab.utils import (  # noqa: E402
    compute_g_matrix,
    compute_horvitz_thompson_weights,
)


@dataclass
class BudgetConfig:
    """Configuration for budget sensitivity experiments."""

    n_units: int = 500
    n_items: int = 300
    p_features: int = 6
    budgets: list[int] | None = None  # Will test multiple budgets
    n_sims: int = 100
    lambda_counts: float = 15.0
    signal_to_noise: float = 1.5
    seed: int = 42
    count_heterogeneity: float = 3.0

    def __post_init__(self):
        if self.budgets is None:
            # Test budgets from 5% to 40% of items
            self.budgets = [15, 30, 45, 60, 90, 120]


def generate_synthetic_data(cfg: BudgetConfig, rng: np.random.Generator):
    """Generate synthetic data with heterogeneous count structure."""
    n, m, p = cfg.n_units, cfg.n_items, cfg.p_features

    # Generate covariates
    X_vals = rng.normal(size=(n, p - 1))
    X_vals = np.c_[np.ones(n), X_vals]
    X = pd.DataFrame(
        X_vals,
        index=[f"unit_{i}" for i in range(n)],
        columns=[f"x{i}" for i in range(p)],
    )

    # True coefficients
    beta_true = rng.normal(scale=cfg.signal_to_noise, size=p)

    # Item labels
    item_quality = rng.normal(0, 1, size=m)
    true_labels = (item_quality > 0).astype(float)

    # Heterogeneous counts (key for showing efficiency gains)
    unit_activity = np.exp(rng.normal(0, 0.5, size=n))
    item_popularity = np.exp(rng.normal(0, cfg.count_heterogeneity, size=m))

    expected_counts = unit_activity[:, None] * item_popularity[None, :]
    counts_vals = rng.poisson(
        expected_counts * cfg.lambda_counts / expected_counts.mean()
    )
    counts_vals = np.maximum(counts_vals, 1)

    counts = pd.DataFrame(
        counts_vals, index=X.index, columns=[f"item_{j}" for j in range(m)]
    )

    return counts, X, true_labels, beta_true


def sample_and_estimate(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    true_labels: np.ndarray,
    K: int,
    method_name: str,
    rng: np.random.Generator,
) -> dict:
    """Sample K items using specified method and estimate coefficients."""

    # Select items based on method
    if method_name == "Random":
        selected = pd.Index(rng.choice(counts.columns, size=K, replace=False))
        pi = pd.Series(K / len(counts.columns), index=counts.columns)

    elif method_name == "Deterministic A-opt":
        selected = pd.Index(fewlab.items_to_label(counts, X, K))
        pi = pd.Series(0.0, index=counts.columns)
        pi[selected] = 1.0

    elif method_name == "Balanced":
        pi = fewlab.pi_aopt_for_budget(counts, X, K)
        g = compute_g_matrix(counts, X)
        seed = rng.integers(0, 2**31)
        selected = fewlab.balanced_fixed_size(pi, g, K, seed=seed)

    elif method_name == "Adaptive Hybrid":
        seed = rng.integers(0, 2**31)
        selected, pi, _ = fewlab.adaptive_core_tail(counts, X, K, seed=seed)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Estimate using Horvitz-Thompson
    labels_series = pd.Series(true_labels, index=counts.columns)
    ht_weights = compute_horvitz_thompson_weights(pi, selected)
    weighted_labels = labels_series[selected] * ht_weights

    counts_selected = counts[selected].to_numpy()
    row_totals = counts.sum(axis=1).to_numpy()
    y_hat = (counts_selected @ weighted_labels.to_numpy()) / (row_totals + 1e-10)

    # Fit OLS
    X_vals = X.to_numpy()
    XtX = X_vals.T @ X_vals
    ridge = 1e-8 * np.eye(X_vals.shape[1])
    beta_hat = np.linalg.solve(XtX + ridge, X_vals.T @ y_hat)

    return {"beta_hat": beta_hat, "n_selected": len(selected)}


def run_budget_experiment(cfg: BudgetConfig) -> pd.DataFrame:
    """Run experiments across multiple budget sizes."""
    rng = np.random.default_rng(cfg.seed)

    methods = ["Random", "Deterministic A-opt", "Balanced", "Adaptive Hybrid"]

    results = []

    assert cfg.budgets is not None, "budgets must be set after __post_init__"
    print(f"Testing budgets: {cfg.budgets}")
    print(f"Running {cfg.n_sims} simulations per budget...\n")

    for K in cfg.budgets:
        print(f"Budget K={K} ({100 * K / cfg.n_items:.1f}% of items)")

        for sim_idx in range(cfg.n_sims):
            # Generate fresh data for each simulation
            counts, X, true_labels, beta_true = generate_synthetic_data(cfg, rng)

            for method_name in methods:
                result = sample_and_estimate(
                    counts, X, true_labels, K, method_name, rng
                )

                beta_hat = result["beta_hat"]
                error = beta_hat - beta_true

                for coef_idx in range(cfg.p_features):
                    results.append(
                        {
                            "K_budget": K,
                            "K_pct": 100 * K / cfg.n_items,
                            "simulation": sim_idx,
                            "method": method_name,
                            "coefficient": coef_idx,
                            "error": error[coef_idx],
                            "squared_error": error[coef_idx] ** 2,
                        }
                    )

    return pd.DataFrame(results)


def compute_efficiency_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute variance and relative efficiency for each budget and method."""
    # Compute variance for each (K, method, coefficient)
    summary = (
        results_df.groupby(["K_budget", "K_pct", "method", "coefficient"])
        .agg(
            {
                "error": "var",
                "squared_error": "mean",
            }
        )
        .reset_index()
    )

    summary.columns = ["K_budget", "K_pct", "method", "coefficient", "variance", "mse"]
    summary["rmse"] = np.sqrt(summary["mse"])

    # Get random baseline variance for each (K, coefficient)
    random_variance = summary[summary["method"] == "Random"][
        ["K_budget", "coefficient", "variance"]
    ].rename(columns={"variance": "random_variance"})

    summary = summary.merge(random_variance, on=["K_budget", "coefficient"], how="left")
    summary["rel_efficiency"] = summary["random_variance"] / summary["variance"]
    summary["efficiency_gain_pct"] = (summary["rel_efficiency"] - 1) * 100

    return summary


def plot_budget_sensitivity(summary: pd.DataFrame, cfg: BudgetConfig):
    """Create visualizations showing efficiency gains vs budget."""
    sns.set_style("whitegrid")

    # Average across all coefficients
    avg_by_budget = (
        summary.groupby(["K_budget", "K_pct", "method"])
        .agg(
            {
                "variance": "mean",
                "rel_efficiency": "mean",
                "efficiency_gain_pct": "mean",
            }
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = [m for m in avg_by_budget["method"].unique() if m != "Random"]

    # Plot 1: Variance vs Budget
    for method in methods + ["Random"]:
        data = avg_by_budget[avg_by_budget["method"] == method]
        axes[0].plot(
            data["K_pct"],
            data["variance"],
            marker="o",
            label=method,
            linewidth=2,
            markersize=8,
        )
    axes[0].set_xlabel("Budget (% of items)", fontsize=12)
    axes[0].set_ylabel("Average Variance", fontsize=12)
    axes[0].set_title("Variance vs. Budget Size", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Relative Efficiency vs Budget
    for method in methods:
        data = avg_by_budget[avg_by_budget["method"] == method]
        axes[1].plot(
            data["K_pct"],
            data["rel_efficiency"],
            marker="o",
            label=method,
            linewidth=2,
            markersize=8,
        )
    axes[1].axhline(
        y=1.0, color="black", linestyle="--", label="Random baseline", linewidth=1.5
    )
    axes[1].set_xlabel("Budget (% of items)", fontsize=12)
    axes[1].set_ylabel("Relative Efficiency (vs. Random)", fontsize=12)
    axes[1].set_title(
        "Relative Efficiency vs. Budget Size", fontsize=14, fontweight="bold"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Efficiency Gain % vs Budget
    for method in methods:
        data = avg_by_budget[avg_by_budget["method"] == method]
        axes[2].plot(
            data["K_pct"],
            data["efficiency_gain_pct"],
            marker="o",
            label=method,
            linewidth=2,
            markersize=8,
        )
    axes[2].axhline(y=0.0, color="black", linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("Budget (% of items)", fontsize=12)
    axes[2].set_ylabel("Efficiency Gain (%)", fontsize=12)
    axes[2].set_title("Efficiency Gain over Random", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = BASE_DIR / "budget_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {output_path.relative_to(REPO_ROOT)}")


def print_summary_table(summary: pd.DataFrame, cfg: BudgetConfig):
    """Print summary table of efficiency gains by budget."""
    print("\n" + "=" * 100)
    print("BUDGET SENSITIVITY ANALYSIS: Efficiency Gains over Random Sampling")
    print("=" * 100)

    # Average across coefficients
    avg_by_budget = (
        summary.groupby(["K_budget", "K_pct", "method"])
        .agg(
            {
                "variance": "mean",
                "rel_efficiency": "mean",
                "efficiency_gain_pct": "mean",
            }
        )
        .reset_index()
    )

    print("\nAverage metrics across all coefficients:")
    print("-" * 100)

    assert cfg.budgets is not None, "budgets must be set after __post_init__"
    for K in cfg.budgets:
        K_pct = 100 * K / cfg.n_items
        print(f"\nBudget K={K} ({K_pct:.1f}% of items):")
        print("-" * 60)

        data = avg_by_budget[avg_by_budget["K_budget"] == K]

        for _, row in data.iterrows():
            if row["method"] == "Random":
                print("  Random (baseline):")
                print(f"    Variance: {row['variance']:.4f}")
            else:
                print(f"  {row['method']}:")
                print(f"    Variance: {row['variance']:.4f}")
                print(f"    Relative efficiency: {row['rel_efficiency']:.3f}x")
                print(f"    Efficiency gain: {row['efficiency_gain_pct']:+.2f}%")

    print("\n" + "=" * 100)
    print("KEY INSIGHT:")
    print("=" * 100)

    # Find budget with largest efficiency gain
    optimal_methods = avg_by_budget[avg_by_budget["method"] != "Random"]
    best_gain = optimal_methods.groupby("method")["efficiency_gain_pct"].max()

    print("\nMaximum efficiency gains by method:")
    for method in best_gain.index:
        gain = best_gain[method]
        best_budget_row = optimal_methods[
            (optimal_methods["method"] == method)
            & (optimal_methods["efficiency_gain_pct"] == gain)
        ].iloc[0]

        print(f"  {method}:")
        print(
            f"    Best gain: {gain:+.2f}% at K={int(best_budget_row['K_budget'])} ({best_budget_row['K_pct']:.1f}% of items)"
        )


def main():
    """Main budget sensitivity analysis."""
    cfg = BudgetConfig(
        n_units=500,
        n_items=300,
        p_features=6,
        budgets=[15, 30, 45, 60, 90, 120],  # 5% to 40%
        n_sims=100,
        count_heterogeneity=3.0,
        seed=42,
    )

    print("Budget Sensitivity Analysis Configuration:")
    print(f"  n_units: {cfg.n_units}")
    print(f"  n_items: {cfg.n_items}")
    print(f"  p_features: {cfg.p_features}")
    print(f"  budgets: {cfg.budgets}")
    print(f"  n_sims: {cfg.n_sims}")
    print(f"  count_heterogeneity: {cfg.count_heterogeneity}")
    print()

    # Run experiments
    results_df = run_budget_experiment(cfg)

    # Compute efficiency metrics
    summary = compute_efficiency_metrics(results_df)

    # Print summary
    print_summary_table(summary, cfg)

    # Create visualizations
    plot_budget_sensitivity(summary, cfg)

    # Save results
    output_path = BASE_DIR / "budget_sensitivity_results.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 100)
    print("Budget sensitivity analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

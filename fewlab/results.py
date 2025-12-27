"""
Structured result classes for fewlab functions.

This module provides typed, structured return values that replace loose tuples
and dict-based returns with direct attribute access for better API consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CoreTailResult:
    """
    Structured result for hybrid core+tail selection methods.

    Attributes:
        selected: All selected item identifiers (core + tail).
        probabilities: A-optimal inclusion probabilities for all items.
        core: Deterministic core items (highest influence).
        tail: Probabilistic tail items (balanced sampling).
        ht_weights: Standard Horvitz-Thompson weights for the selected items.
        mixed_weights: Mixed weights (1/pi for core, 1.0 for tail) for variance reduction.
        diagnostics: Additional metadata such as budget splits and tail fraction.

    Properties:
        budget_used: Total number of items selected.
        budget_core: Number of items in the deterministic core.
        budget_tail: Number of items in the probabilistic tail.
        tail_frac: Fraction of the budget allocated to the tail.

    Examples:
        >>> result = design.sample(budget=50, method="core_plus_tail", tail_frac=0.2)
        >>> len(result.selected), len(result.core), len(result.tail)
        (50, 40, 10)
    """

    selected: pd.Index
    probabilities: pd.Series  # pi for all items
    core: pd.Index
    tail: pd.Index
    ht_weights: pd.Series  # Standard 1/pi weights
    mixed_weights: pd.Series  # Biased weights (1/pi for core, 1.0 for tail)
    diagnostics: dict[str, Any]

    @property
    def budget_used(self) -> int:
        """Total number of items selected."""
        return len(self.selected)

    @property
    def budget_core(self) -> int:
        """Number of items in deterministic core."""
        return len(self.core)

    @property
    def budget_tail(self) -> int:
        """Number of items in probabilistic tail."""
        return len(self.tail)

    @property
    def tail_frac(self) -> float:
        """Fraction of budget allocated to tail."""
        return self.diagnostics.get("tail_frac", 0.0)

    @property
    def probability_sum(self) -> float:
        """Sum of inclusion probabilities."""
        return float(self.probabilities.sum())

    def __len__(self) -> int:
        """Number of selected items."""
        return len(self.selected)

    def __iter__(self):
        """Iterate over selected items."""
        return iter(self.selected)

    def __getitem__(self, key):
        """Access selected items by index."""
        return self.selected[key]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CoreTailResult(selected={len(self.selected)} items, "
            f"core={len(self.core)}, tail={len(self.tail)}, "
            f"tail_frac={self.tail_frac:.2f})"
        )


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """
    Structured result for deterministic selection methods.

    Attributes:
        selected: Selected item identifiers ordered by influence.
        influence_weights: A-optimal influence weights used for selection.
        diagnostics: Selection diagnostics and metadata.

    Properties:
        budget_used: Number of items selected.

    Examples:
        >>> result = design.select(budget=30, method="deterministic")
        >>> len(result.selected)
        30
    """

    selected: pd.Index
    influence_weights: pd.Series
    diagnostics: dict[str, Any]

    @property
    def budget_used(self) -> int:
        """Number of items selected."""
        return len(self.selected)

    def __len__(self) -> int:
        """Number of selected items."""
        return len(self.selected)

    def __iter__(self):
        """Iterate over selected items."""
        return iter(self.selected)

    def __getitem__(self, key):
        """Access selected items by index."""
        return self.selected[key]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SelectionResult(selected={len(self.selected)} items, "
            f"max_weight={self.influence_weights.loc[self.selected].max():.3f})"
        )


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """
    Structured result for probabilistic sampling methods.

    Attributes:
        sample: Sampled item identifiers.
        probabilities: Inclusion probabilities used for sampling.
        weights: Suggested sampling weights for the sampled items.
        diagnostics: Sampling diagnostics and metadata.

    Properties:
        sample_size: Number of sampled items.

    Examples:
        >>> result = design.sample(budget=30, method="balanced")
        >>> result.sample_size
        30
    """

    sample: pd.Index
    probabilities: pd.Series  # pi for all items
    weights: pd.Series  # suggested weights for sampled items
    diagnostics: dict[str, Any]

    @property
    def sample_size(self) -> int:
        """Number of sampled items."""
        return len(self.sample)

    @property
    def probability_sum(self) -> float:
        """Sum of inclusion probabilities."""
        return float(self.probabilities.sum())

    def __len__(self) -> int:
        """Number of sampled items."""
        return len(self.sample)

    def __iter__(self):
        """Iterate over sampled items."""
        return iter(self.sample)

    def __getitem__(self, key):
        """Access sampled items by index."""
        return self.sample[key]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SamplingResult(sample_size={self.sample_size}, "
            f"probability_sum={self.probability_sum:.1f})"
        )


@dataclass(frozen=True, slots=True)
class EstimationResult:
    """
    Structured result for estimation methods.

    Attributes
    ----------
    estimates : pd.Series
        Row-wise estimates.
    weights : pd.Series
        Calibrated weights used for estimation.
    selected : pd.Index
        Items used for estimation.
    diagnostics : dict[str, Any]
        Estimation diagnostics.

    Examples
    --------
    >>> result = design.estimate(selected, labels)
    >>> print(f"Mean estimate: {result.estimates.mean():.3f}")
    """

    estimates: pd.Series
    weights: pd.Series
    selected: pd.Index
    diagnostics: dict[str, Any]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EstimationResult(n_estimates={len(self.estimates)}, "
            f"mean={self.estimates.mean():.3f})"
        )


@dataclass(frozen=True, slots=True)
class ProbabilityResult:
    """
    Structured result for probability computation methods.

    Provides access to computed probabilities, influence projections, and computation diagnostics.

    Attributes
    ----------
    probabilities : pd.Series
        Inclusion probabilities indexed by item identifiers.
    influence_projections : np.ndarray
        Regression projections g_j = X^T v_j for all items (shape (p, m)).
        Used for balanced sampling and weight calibration.
    diagnostics : dict[str, Any]
        Computation diagnostics and metadata.

    Properties
    ----------
    budget_used : float
        Sum of inclusion probabilities.

    Examples
    --------
    >>> result = design.inclusion_probabilities(budget=50, method="aopt")
    >>> print(f"Budget used: {result.budget_used:.1f}")
    >>> # Now you can use influence_projections for balanced sampling
    >>> from fewlab import balanced_fixed_size
    >>> selected = balanced_fixed_size(result.probabilities, result.influence_projections, 50)
    """

    probabilities: pd.Series
    influence_projections: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def budget_used(self) -> float:
        """Sum of inclusion probabilities."""
        return float(self.probabilities.sum())

    def __len__(self) -> int:
        """Number of items with probabilities."""
        return len(self.probabilities)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProbabilityResult(n_items={len(self.probabilities)}, "
            f"budget_used={self.budget_used:.1f})"
        )


@dataclass(frozen=True, slots=True)
class RowSEResult:
    """
    Result container for `row_se_min_labels`.

    Attributes:
        probabilities: Inclusion probabilities indexed by item identifiers.
        max_violation: Maximum constraint violation encountered.
        tolerance: Target violation tolerance.
        iterations: Number of iterations executed.
        best_iteration: Iteration index where the best solution was recorded.
        feasible: Whether the best solution satisfies the tolerance.
    """

    probabilities: pd.Series
    max_violation: float
    tolerance: float
    iterations: int
    best_iteration: int
    feasible: bool

    def to_series(self) -> pd.Series:
        """Return a copy of the probabilities as a Series."""
        return self.probabilities.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostic information as a dict."""
        return {
            "max_violation": self.max_violation,
            "tolerance": self.tolerance,
            "iterations": self.iterations,
            "best_iteration": self.best_iteration,
            "feasible": self.feasible,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "feasible" if self.feasible else "infeasible"
        return (
            f"RowSEResult(n_items={len(self.probabilities)}, "
            f"status={status}, max_violation={self.max_violation:.3e})"
        )

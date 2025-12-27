"""
Weight calibration methods for optimal survey sampling.

This module implements GREG (Generalized Regression) calibration and related
techniques for adjusting sampling weights to match known population totals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import DIVISION_EPS, SMALL_RIDGE
from .utils import get_item_positions

# Type alias for item selection types (Python 3.12+)
type ItemSelection = Sequence[str] | pd.Index


def calibrate_weights(
    pi: pd.Series,
    g: np.ndarray,
    selected: ItemSelection,
    pop_totals: np.ndarray | None = None,
    *,
    distance: str = "chi2",
    ridge: float = SMALL_RIDGE,
    nonneg: bool = True,
) -> pd.Series:
    """
    Compute calibrated weights for selected items using GREG/Deville-Särndal calibration.

    Args:
        pi: Inclusion probabilities for all items (index = item names).
        g: Regression projections `g_j = X^T v_j` for all items (shape `(p, m)`).
        selected: Item identifiers drawn in the sample.
        pop_totals: Known population totals (shape `(p,)`); defaults to `g.sum(axis=1)`.
        distance: Calibration distance measure; currently only `"chi2"` is supported.
        ridge: Ridge regularization parameter for numerical stability.
        nonneg: Whether to enforce non-negative calibrated weights.

    Returns:
        Calibrated weights indexed by the selected items.

    Raises:
        NotImplementedError: If `distance` is not `"chi2"`.
        ValueError: If `pop_totals` has the wrong shape.

    Notes:
        The closed-form solution for chi-square distance is
        `w* = d_S + G_S^T (G_S G_S^T + ridge I)^{-1} (t - G_S d_S)` where `d_S` are base weights.

    References:
        Deville, J.-C., & Särndal, C.-E. (1992). Calibration estimators in survey sampling.
        Journal of the American Statistical Association, 87(418), 376-382.
    """
    if distance != "chi2":
        raise NotImplementedError(f"Distance '{distance}' not implemented yet")

    # Map selected items to positions in pi
    sel_pos = get_item_positions(selected, pi.index)

    # Base HT weights for selected items
    pi_array = pi.to_numpy(dtype=float)
    d_full = 1.0 / (pi_array + DIVISION_EPS)
    d = d_full[sel_pos]  # (K,)

    # G matrix for selected items
    G_s = g[:, sel_pos]  # (p, K)

    # Population totals
    if pop_totals is None:
        t = g.sum(axis=1)  # (p,)
    else:
        t = np.asarray(pop_totals, dtype=float)
        if t.shape != (g.shape[0],):
            raise ValueError(f"pop_totals must have shape ({g.shape[0]},)")

    # Solve calibration equation: G_S w = t
    # w* = d + G_S^T (G_S G_S^T)^{-1} (t - G_S d)
    A = G_s @ G_s.T + ridge * np.eye(G_s.shape[0])  # (p, p)
    rhs = t - (G_s @ d)  # (p,)

    try:
        lam = np.linalg.solve(A, rhs)  # (p,)
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse if singular
        lam = np.linalg.lstsq(A, rhs, rcond=None)[0]

    w = d + G_s.T @ lam  # (K,)

    if nonneg:
        w = np.maximum(w, DIVISION_EPS)

    if isinstance(selected, pd.Index):
        index = selected
    else:
        index = pd.Index(list(selected))
    result_series: pd.Series = pd.Series(w, index=index, name="calibrated_weights")
    return result_series


def calibrated_ht_estimator(
    counts: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series,
    *,
    normalize_by_total: bool = True,
) -> pd.Series:
    """
    Compute calibrated Horvitz-Thompson estimator for row shares.

    Args:
        counts: Count matrix with rows as units and columns as items.
        labels: Item labels for the selected items.
        weights: Calibrated weights for the selected items.
        normalize_by_total: Whether to divide by row totals to obtain shares.

    Returns:
        Estimated row shares (or totals if `normalize_by_total` is False).
    """
    # Align weights and labels with counts columns
    w = weights.reindex(counts.columns).fillna(0.0).to_numpy(dtype=float)
    a = labels.reindex(counts.columns).fillna(0.0).to_numpy(dtype=float)

    # Weighted sum
    numerator = counts.to_numpy(dtype=float) @ (w * a)  # (n,)

    if normalize_by_total:
        T = counts.sum(axis=1).to_numpy(float)
        result = numerator / (T + DIVISION_EPS)
    else:
        result = numerator

    return pd.Series(
        result, index=counts.index, name="calibrated_ht_estimate", dtype=float
    )

"""Weight calibration methods for optimal survey sampling.

This module implements GREG (Generalized Regression) calibration and related
techniques for adjusting sampling weights to match known population totals.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import DIVISION_EPS, SMALL_RIDGE
from .utils import get_item_positions

# Type alias for item selection types (Python 3.12+)
ItemSelection = Sequence[str] | pd.Index


def _solve_calibration(
    G_s: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    ridge: float,
    free: np.ndarray,
) -> np.ndarray:
    """Chi-square calibration over the free weights, holding the rest at the floor.

    Solves ``min ||w_free - d_free||^2`` subject to ``G_free w_free = t - G_held
    d_held``, whose closed form is ``d + G^T (G G^T + ridge I)^{-1} (t - G d)``
    restricted to the free columns.

    Args:
        G_s: Projections for the selected items, shape ``(p, K)``.
        d: Base Horvitz-Thompson weights for the selected items.
        t: Population totals to reproduce.
        ridge: Ridge added to the normal equations for stability.
        free: Boolean mask of the weights allowed to move.

    Returns:
        np.ndarray: Weights for all ``K`` selected items; held ones sit at the
        floor.
    """
    w = np.where(free, d, DIVISION_EPS)
    G_free = G_s[:, free]
    residual = t - (G_s @ w)

    matrix = G_free @ G_free.T + ridge * np.eye(G_s.shape[0])
    try:
        lam = np.linalg.solve(matrix, residual)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(matrix, residual, rcond=None)[0]

    w[free] += G_free.T @ lam
    return w


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
    """Compute calibrated weights via GREG/Deville-Särndal calibration.

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
        `w* = d_S + G_S^T (G_S G_S^T + ridge I)^{-1} (t - G_S d_S)`, where `d_S`
        are the base weights.

    References:
        Deville, J.-C., & Särndal, C.-E. (1992). Calibration estimators in survey
        sampling.
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

    w = _solve_calibration(G_s, d, t, ridge, np.ones(d.size, dtype=bool))

    if nonneg and (w < DIVISION_EPS).any():
        # Clipping the negatives would break the calibration equation this
        # function exists to satisfy, silently and sometimes badly: on a
        # 100-item fixture the raw solution went negative in 41 draws out of 200,
        # and in 10 of those the clipped weights reproduced the population totals
        # *worse* than the unadjusted Horvitz-Thompson weights did.
        #
        # So pin the offenders at the floor and re-solve over the rest, which
        # restores the constraint exactly whenever the remaining items can carry
        # it. Each pass pins at least one weight, so it terminates.
        free = np.ones(d.size, dtype=bool)
        for _ in range(d.size):
            offenders = free & (w < DIVISION_EPS)
            if not offenders.any():
                break
            free &= ~offenders
            if free.sum() < G_s.shape[0]:
                # Fewer free weights than constraints: no exact solution remains.
                w = np.maximum(w, DIVISION_EPS)
                warnings.warn(
                    "Calibration could not be satisfied with non-negative "
                    f"weights: only {int(free.sum())} of {d.size} weights remain "
                    f"free against {G_s.shape[0]} constraints. The returned "
                    "weights are non-negative but do not reproduce the "
                    "population totals. Pass nonneg=False to get the exact "
                    "calibration, which will contain negative weights.",
                    UserWarning,
                    stacklevel=2,
                )
                break
            w = _solve_calibration(G_s, d, t, ridge, free)
        w = np.maximum(w, DIVISION_EPS)

    index = selected if isinstance(selected, pd.Index) else pd.Index(list(selected))
    result_series: pd.Series = pd.Series(w, index=index, name="calibrated_weights")
    return result_series


def calibrated_ht_estimator(
    counts: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series,
    *,
    normalize_by_total: bool = True,
) -> pd.Series:
    """Compute calibrated Horvitz-Thompson estimator for row shares.

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

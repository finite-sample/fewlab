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

    Solves the optimization problem:
        min ||w - d||^2  s.t. G_S w = t
    where d = 1/pi are base HT weights, G_S is the matrix of g-vectors for selected items,
    and t are population totals.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fewlab import calibrate_weights, _influence
    >>>
    >>> # Sample data
    >>> counts = pd.DataFrame(np.random.poisson(5, (100, 50)))
    >>> X = pd.DataFrame(np.random.randn(100, 3))
    >>> selected = counts.columns[:20]
    >>>
    >>> # Compute influence and calibrate
    >>> inf = _influence(counts, X)
    >>> pi = pd.Series(0.4, index=counts.columns)
    >>> weights = calibrate_weights(pi, inf.g, selected)
    >>>
    >>> # Weights will satisfy calibration constraint:
    >>> # sum(weights * g_selected) ≈ sum(g_all)

    Parameters
    ----------
    pi : pd.Series
        Inclusion probabilities for all items (index = item names).
    g : np.ndarray, shape (p, m)
        Regression projections g_j = X^T v_j for all items.
    selected : sequence of str or pd.Index
        Item identifiers that were actually selected.
    pop_totals : np.ndarray, shape (p,), optional
        Known population totals. If None, uses g.sum(axis=1).

    distance : {'chi2', 'euclidean'}
        Distance measure for calibration. Currently only 'chi2' is implemented.

    ridge : float
        Ridge regularization parameter for numerical stability.

    nonneg : bool
        If True, enforce non-negative weights (may slightly violate calibration).

    Returns
    -------
    pd.Series
        Calibrated weights indexed by selected items.

    Notes
    -----
    The closed-form solution for chi-square distance is:
        w* = d_S + G_S^T (G_S G_S^T + ridge I)^{-1} (t - G_S d_S)
    where d_S are the base weights for selected items.

    References
    ----------
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

    For each row i, estimates:
        y_i = (1/T_i) * sum_{j in S} w_j * a_j * C_ij
    where w_j are calibrated weights, a_j are labels, and C_ij are counts.

    Parameters
    ----------
    counts : pd.DataFrame, shape (n, m)
        Count matrix with rows=units, columns=items.
    labels : pd.Series
        Item labels (only for selected items).
    weights : pd.Series
        Calibrated weights for selected items.
    normalize_by_total : bool
        If True, divide by row totals T_i to get shares.

    Returns
    -------
    pd.Series
        Estimated row shares (or totals if normalize_by_total=False).

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

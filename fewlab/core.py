from __future__ import annotations
import numpy as np
import pandas as pd

def items_to_label(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    K: int,
    *,
    item_axis: int = 1,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> list:
    """
    Return a deterministic list of item identifiers to label (length K),
    using the A-opt square-root rule on w_j = g_j^T (X^T X)^{-1} g_j.

    Parameters
    ----------
    counts : DataFrame (n x m)
        Nonnegative counts C with rows = units and columns = items.
        Index must align with X.index.
    X : DataFrame (n x p)
        Covariate matrix used in the regression y ~ X.
        Index must align with counts.index.
    K : int
        Desired number of items to label (K <= m).
    item_axis : {1}, ignored (for API future-proofing).
    ensure_full_rank : bool
        If True, and X^T X is rank-deficient, add a small ridge.
    ridge : float or None
        If not None, use (X^T X + ridge I)^{-1} explicitly.

    Returns
    -------
    list
        A list of item identifiers (counts.columns) to label, deterministic.

    Notes
    -----
    - We compute T_i = sum_j c_ij, v_j = c_{·j}/T, g_j = X^T v_j, and
      w_j = g_j^T (X^T X)^{-1} g_j. We then pick the top-K items by w_j.
    - This deterministically approximates the fixed-budget A-opt solution.
    - If ridge is None but X is ill-conditioned, a tiny ridge is applied if
      ensure_full_rank is True.
    """
    if not isinstance(counts, pd.DataFrame) or not isinstance(X, pd.DataFrame):
        raise TypeError("counts and X must be pandas DataFrames")

    if not counts.index.equals(X.index):
        raise ValueError("counts.index must align with X.index")

    if K <= 0:
        return []

    n, m = counts.shape
    if K > m:
        raise ValueError(f"K={K} exceeds number of items m={m}")

    # Compute T_i and guard against zero totals
    T = counts.sum(axis=1).to_numpy(dtype=float)
    if np.any(T <= 0):
        # Drop zero-total rows from both counts and X
        keep = T > 0
        counts = counts.loc[keep]
        X = X.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValueError("All rows have zero totals; nothing to compute.")

    # v_j = c_{·j} / T
    V = counts.to_numpy(dtype=float) / T[:, None]  # (n x m)

    # g_j = X^T v_j  => shape (p x m)
    Xn = X.to_numpy(dtype=float)
    XtX = Xn.T @ Xn

    # Regularize if needed
    if ridge is None and ensure_full_rank:
        # Tiny ridge if singular or poorly conditioned
        small = 1e-8
        cond = np.linalg.cond(XtX)
        if not np.isfinite(cond) or cond > 1e12:
            ridge = small
    if ridge is not None and ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0])

    XtX_inv = np.linalg.inv(XtX)
    G = Xn.T @ V  # (p x m)

    # w_j = g_j^T XtX_inv g_j  (vectorized)
    # w = diag( G^T XtX_inv G )
    # -> einsum over j: (j p) (p p) (p j) -> j
    w = np.einsum("jp,pk,kj->j", G.T, XtX_inv, G)

    # Pick top-K items by w
    order = np.argsort(-w)
    chosen = order[:K]
    item_ids = list(counts.columns[chosen])
    return item_ids

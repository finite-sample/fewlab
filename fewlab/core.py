from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    BINARY_SEARCH_HI,
    BINARY_SEARCH_LO,
    CONDITION_THRESHOLD,
    MAX_ITER_BINARY_SEARCH,
    PI_MIN_DEFAULT,
    SMALL_RIDGE,
)
from .selection import topk


@dataclass(slots=True)
class Influence:
    """Influence data structure with memory-optimized slots."""

    w: np.ndarray  # (m,)   A-opt weights w_j
    g: np.ndarray  # (p, m) regression projections g_j = X^T v_j
    cols: list[str]  # item column names in the same order


def _influence(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> Influence:
    """Compute (w_j, g_j) given counts and X."""
    if not counts.index.equals(X.index):
        raise ValueError("counts.index must align with X.index")

    # Handle sparse counts efficiently
    T: np.ndarray = counts.sum(axis=1).to_numpy(float)

    keep: np.ndarray = T > 0
    if not np.all(keep):
        counts = counts.loc[keep]
        X = X.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValueError("All rows have zero totals; nothing to compute.")

    # V = counts / T[:, None]
    # If counts is sparse, we want to avoid densifying (n x m) if possible.
    # However, the current implementation computes G = X.T @ V
    # G = X.T @ (counts / T) = (X.T / T) @ counts ? No.
    # G_jp = sum_i X_ip * (C_ij / T_i)
    #      = sum_i (X_ip / T_i) * C_ij
    # Let X_scaled = X / T[:, None]. Then G = X_scaled.T @ counts.

    # This is much more efficient if counts is sparse!

    Xn: np.ndarray = X.to_numpy(float)
    X_scaled = Xn / T[:, None]

    # Check if we have sparse data and can use efficient sparse operations
    try:
        # Check if any columns are sparse using pandas API
        has_sparse = any(
            hasattr(dtype, "subtype") and dtype.subtype is not None
            for dtype in counts.dtypes
        )
        if has_sparse:
            # Use pandas sparse accessor if available
            try:
                # For sparse DataFrames, convert to scipy sparse for efficient computation
                import scipy.sparse  # type: ignore[import-untyped]  # noqa: F401

                C_sparse = counts.sparse.to_coo()  # type: ignore[attr-defined]
                G = X_scaled.T @ C_sparse
            except (ImportError, AttributeError):
                G = X_scaled.T @ counts.to_numpy(float)
        else:
            G = X_scaled.T @ counts.to_numpy(float)
    except (ImportError, AttributeError):
        # Fallback to dense computation
        G = X_scaled.T @ counts.to_numpy(float)

    XtX: np.ndarray = Xn.T @ Xn
    if ridge is None and ensure_full_rank:
        cond: float = np.linalg.cond(XtX)
        if not np.isfinite(cond) or cond > CONDITION_THRESHOLD:
            ridge = SMALL_RIDGE
    if ridge is not None and ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0])

    # Use solve instead of inv for stability: w_j = g_j^T (X^T X)^{-1} g_j
    # We want diag(G^T (XtX)^{-1} G).
    # Let H = (XtX)^{-1} G. We can find H by solving (XtX) H = G.
    # Then w_j is dot product of j-th column of G and H.
    try:
        H = np.linalg.solve(XtX, G)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix if ridge didn't help enough
        H = np.linalg.lstsq(XtX, G, rcond=None)[0]

    w: np.ndarray = np.einsum("jp,jp->j", G.T, H.T)  # (m,)
    return Influence(w=w, g=G, cols=list(counts.columns))


def pi_aopt_for_budget(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    K: int,
    *,
    pi_min: float = PI_MIN_DEFAULT,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> pd.Series:
    """
    Return A-opt first-order inclusion probabilities pi_j for expected budget K:
        pi_j = clip(c * sqrt(w_j), [pi_min, 1]), with c chosen so sum pi = K.
    """
    inf: Influence = _influence(
        counts, X, ensure_full_rank=ensure_full_rank, ridge=ridge
    )
    sqrtw: np.ndarray = np.sqrt(np.maximum(inf.w, 0.0))
    if K <= 0:
        return pd.Series(np.full_like(sqrtw, pi_min), index=inf.cols, name="pi")

    m: int = sqrtw.size
    K = min(K, m)

    def sum_pi(c: float) -> tuple[float, np.ndarray]:
        pi: np.ndarray = np.clip(c * sqrtw, pi_min, 1.0)
        return pi.sum(), pi

    lo: float = BINARY_SEARCH_LO
    hi: float = BINARY_SEARCH_HI
    for _ in range(MAX_ITER_BINARY_SEARCH):
        c: float = (lo * hi) ** 0.5
        s: float
        s, _ = sum_pi(c)
        if s > K:
            hi = c
        else:
            lo = c
    _, pi = sum_pi(hi)
    return pd.Series(pi, index=inf.cols, name="pi")


def items_to_label(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    K: int,
    *,
    item_axis: int = 1,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> list[str]:
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
    item_axis : {1}
        Currently only axis=1 (columns=items) is supported. Must be 1.
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

    if item_axis != 1:
        raise ValueError(f"item_axis must be 1, got {item_axis}")

    if K <= 0:
        return []

    _, m = counts.shape
    if K > m:
        raise ValueError(f"K={K} exceeds number of items m={m}")

    # Pick top-K items by w
    inf: Influence = _influence(
        counts, X, ensure_full_rank=ensure_full_rank, ridge=ridge
    )
    chosen: np.ndarray = topk(inf.w, K)
    return list(pd.Index(inf.cols)[chosen])

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import ProbabilityResult, SelectionResult

from .constants import (
    CONDITION_THRESHOLD,
    PI_MIN_DEFAULT,
    SMALL_RIDGE,
)


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
    budget: int,
    *,
    pi_min: float = PI_MIN_DEFAULT,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> ProbabilityResult:
    """
    Compute A-optimal first-order inclusion probabilities for a target budget.

    The probabilities follow the square-root rule `pi_j = clip(c * sqrt(w_j), [pi_min, 1])` with
    `c` chosen so that `sum(pi) = budget`.

    Args:
        counts: Count matrix with non-negative values.
        X: Feature matrix aligned with `counts.index`.
        budget: Expected total budget (sum of inclusion probabilities).
        pi_min: Minimum allowed inclusion probability.
        ensure_full_rank: Whether to add a small ridge term when `X^T X` is ill-conditioned.
        ridge: Explicit ridge parameter overriding the automatic heuristic.

    Returns:
        Probability result with inclusion probabilities and computation diagnostics.

    Note:
        If `budget < m * pi_min` (where m is the number of items), the budget constraint
        cannot be satisfied. In this case, the function returns all probabilities as `pi_min`,
        resulting in `sum(pi) = m * pi_min > budget`, and issues a warning. The violation
        details are included in the result's diagnostics under `budget_violation`.

    See Also:
        items_to_label: Deterministic selection using the same influence weights.
        balanced_fixed_size: Fixed-size balanced sampling using these probabilities.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import pi_aopt_for_budget
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(5, (1000, 200)))
        >>> X = pd.DataFrame(np.random.randn(1000, 3))
        >>> result = pi_aopt_for_budget(counts, X, budget=50)
        >>> round(result.budget_used, 1)
        50.0
    """
    from .design import Design

    # Create Design instance (caches influence computation)
    design = Design(
        counts,
        X,
        ridge="auto" if ridge is None else ridge,
        ensure_full_rank=ensure_full_rank,
    )

    return design.inclusion_probabilities(budget, pi_min=pi_min)


def items_to_label(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    budget: int,
    *,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> SelectionResult:
    """
    Select items to label using deterministic A-optimal design.

    Influence weights are computed as `w_j = g_j^T (X^T X)^{-1} g_j`, and the top entries are
    returned.

    Args:
        counts: Count matrix with units as rows and items as columns.
        X: Feature matrix aligned with `counts.index`.
        budget: Number of items to select.
        ensure_full_rank: Whether to add a ridge term when `X^T X` is ill-conditioned.
        ridge: Optional ridge parameter overriding the automatic heuristic.

    Returns:
        Selection result with items, influence weights, and diagnostics.

    See Also:
        pi_aopt_for_budget: Compute inclusion probabilities for the same design.
        greedy_aopt_selection: Greedy sequential variant.
        core_plus_tail: Hybrid deterministic and probabilistic selection.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import items_to_label
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(5, (1000, 200)))
        >>> X = pd.DataFrame(np.random.randn(1000, 3))
        >>> result = items_to_label(counts, X, budget=50)
        >>> len(result.selected)
        50
    """
    from .design import Design

    # Create Design instance (caches influence computation)
    design = Design(
        counts,
        X,
        ridge="auto" if ridge is None else ridge,
        ensure_full_rank=ensure_full_rank,
    )

    return design.select(budget, method="deterministic")

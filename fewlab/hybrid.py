"""
Hybrid sampling strategies combining deterministic and probabilistic selection.

This module implements advanced sampling designs that combine the benefits of
deterministic high-influence selection with balanced probabilistic sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .balanced import balanced_fixed_size
from .constants import DIVISION_EPS
from .core import _influence, items_to_label, pi_aopt_for_budget
from .utils import (
    compute_horvitz_thompson_weights,
    get_item_positions,
    validate_fraction,
)

# Type aliases for commonly used types (Python 3.12+)
type SelectionResult = tuple[pd.Index, pd.Series, dict[str, Any]]


def core_plus_tail(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    K: int,
    *,
    tail_frac: float = 0.2,
    seed: int | None = None,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> SelectionResult:
    """
    Hybrid sampler combining deterministic core with balanced probabilistic tail.

    Strategy:
    1. Select K_core = (1-tail_frac)*K items deterministically (highest w_j)
    2. Compute A-optimal π for full budget K
    3. Select K_tail = K - K_core items from remainder using balanced sampling

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fewlab import core_plus_tail
    >>>
    >>> # Survey data with 1000 units and 200 items
    >>> counts = pd.DataFrame(np.random.poisson(10, (1000, 200)))
    >>> X = pd.DataFrame(np.random.randn(1000, 5))  # 5 covariates
    >>>
    >>> # Select 50 items: 80% deterministic, 20% probabilistic
    >>> selected, pi, info = core_plus_tail(counts, X, K=50, tail_frac=0.2)
    >>>
    >>> # Use calibrated weights for estimation
    >>> from fewlab import calibrate_weights, calibrated_ht_estimator
    >>> weights = calibrate_weights(pi, info['g'], selected)
    >>>
    >>> # info contains:
    >>> # - info['core']: 40 deterministic items (highest influence)
    >>> # - info['tail']: 10 probabilistic items (balanced sampling)
    >>> # - info['weights']: Standard HT weights
    >>> # - info['tail_only_weights']: Mixed weights for variance reduction

    Parameters
    ----------
    counts : pd.DataFrame, shape (n, m)
        Count matrix with rows=units, columns=items.
    X : pd.DataFrame, shape (n, p)
        Covariate matrix, index must align with counts.index.
    K : int
        Total budget (number of items to select).
    tail_frac : float, default=0.2
        Fraction of budget allocated to probabilistic tail (0 < tail_frac < 1).
    seed : int, optional
        Random seed for balanced tail selection.
    ensure_full_rank : bool
        If True, add ridge to X^T X if rank-deficient.
    ridge : float, optional
        Explicit ridge parameter for (X^T X + ridge I)^{-1}.

    Returns
    -------
    selected : pd.Index
        Selected item identifiers (length K).
    pi : pd.Series
        Inclusion probabilities for all items (computed for full budget K).
    info : dict
        Additional information including:
        - 'core': Items in deterministic core
        - 'tail': Items in probabilistic tail
        - 'weights': Suggested weights (1/pi for selected items)
        - 'tail_only_weights': Alternative weights (1/pi for core, 1.0 for tail)
    """
    validate_fraction(tail_frac, "tail_frac")

    _, m = counts.shape
    K = min(K, m)
    K_core = int((1 - tail_frac) * K)
    K_tail = K - K_core

    if K_core <= 0 or K_tail <= 0:
        raise ValueError(f"Invalid split: K_core={K_core}, K_tail={K_tail}")

    # Step 1: Deterministic core selection (top-K_core by w_j)
    core_items = items_to_label(
        counts=counts, X=X, K=K_core, ensure_full_rank=ensure_full_rank, ridge=ridge
    )
    core = pd.Index(core_items)

    # Step 2: Compute A-optimal π for full budget
    pi = pi_aopt_for_budget(
        counts=counts, X=X, K=K, ensure_full_rank=ensure_full_rank, ridge=ridge
    )

    # Step 3: Balanced selection from remainder
    remainder = pi.index.difference(core)
    if len(remainder) < K_tail:
        # Edge case: not enough items left, take all remainder
        tail = remainder
        selected = core.union(tail)
    else:
        # Get g matrix for influence calculations
        inf = _influence(counts, X, ensure_full_rank=ensure_full_rank, ridge=ridge)
        g = inf.g  # (p, m)

        # Map remainder items to their positions
        remainder_idx = get_item_positions(remainder, counts.columns)

        # Extract g and π for remainder
        g_remainder = g[:, remainder_idx]
        pi_remainder = pi.loc[remainder]

        # Balanced sampling for tail
        tail = balanced_fixed_size(pi=pi_remainder, g=g_remainder, K=K_tail, seed=seed)
        selected = core.union(tail)

    # Compute suggested weights
    weights_ht = compute_horvitz_thompson_weights(pi, selected)

    # Alternative "tiny-bias" weights: 1/pi for core, 1.0 for tail
    weights_mixed = pd.Series(index=selected, dtype=float)
    weights_mixed.loc[core] = (1.0 / pi).reindex(core)
    weights_mixed.loc[tail] = 1.0  # Intentional bias for variance reduction

    info = {
        "core": core,
        "tail": tail,
        "weights": weights_ht,
        "tail_only_weights": weights_mixed,
        "K_core": K_core,
        "K_tail": K_tail,
        "tail_frac": tail_frac,
    }

    return selected, pi, info


def adaptive_core_tail(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    K: int,
    *,
    min_tail_frac: float = 0.1,
    max_tail_frac: float = 0.4,
    condition_threshold: float = 1e6,
    seed: int | None = None,
) -> SelectionResult:
    """
    Adaptive core+tail selection with data-driven tail fraction.

    Automatically determines optimal tail_frac based on:
    - Condition number of X^T X (higher -> more tail)
    - Distribution of influence weights w_j (more skewed -> less tail)

    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix.
    X : pd.DataFrame
        Covariate matrix.
    K : int
        Total budget.
    min_tail_frac : float, default=0.1
        Minimum fraction for tail.
    max_tail_frac : float, default=0.4
        Maximum fraction for tail.
    condition_threshold : float
        Threshold for considering X^T X ill-conditioned.
    seed : int, optional
        Random seed.

    Returns
    -------
    Same as core_plus_tail, with adaptive tail_frac in info dict.
    """
    # Compute condition number
    Xn = X.to_numpy()
    XtX = Xn.T @ Xn
    cond = np.linalg.cond(XtX)

    # Compute influence weights
    inf = _influence(counts, X, ensure_full_rank=True)
    w = inf.w

    # Adaptive logic
    # 1. Higher condition number -> more tail (for stability)
    cond_score = np.clip(np.log10(cond / condition_threshold + 1), 0, 1)

    # 2. More skewed w distribution -> less tail (core captures most influence)
    w_sorted = np.sort(w)[::-1]
    if len(w_sorted) > K:
        # Ratio of top-K influence to total
        concentration = w_sorted[:K].sum() / (w.sum() + DIVISION_EPS)
    else:
        concentration = 0.5
    skew_score = 1 - concentration

    # Combine scores (equal weighting)
    combined_score = 0.5 * cond_score + 0.5 * skew_score
    tail_frac = min_tail_frac + combined_score * (max_tail_frac - min_tail_frac)

    # Use computed tail_frac
    selected, pi, info = core_plus_tail(
        counts=counts, X=X, K=K, tail_frac=tail_frac, seed=seed
    )

    # Add adaptive info
    info["adaptive"] = True
    info["condition_number"] = cond
    info["concentration_ratio"] = concentration
    info["adaptive_tail_frac"] = tail_frac

    return selected, pi, info

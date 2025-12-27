"""
Hybrid sampling strategies combining deterministic and probabilistic selection.

This module implements advanced sampling designs that combine the benefits of
deterministic high-influence selection with balanced probabilistic sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .balanced import balanced_fixed_size
from .constants import DIVISION_EPS
from .core import _influence, items_to_label, pi_aopt_for_budget
from .results import CoreTailResult
from .utils import (
    compute_horvitz_thompson_weights,
    get_item_positions,
    validate_fraction,
)
from .validation import (
    ValidationError,
    validate_budget,
    validate_counts_matrix,
    validate_data_alignment,
    validate_features_matrix,
)


def core_plus_tail(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    budget: int,
    *,
    tail_frac: float = 0.2,
    random_state: None | int | np.random.Generator = None,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> CoreTailResult:
    """
    Hybrid sampler combining a deterministic core with a balanced probabilistic tail.

    Strategy:
        1. Select `budget_core = (1 - tail_frac) * budget` items deterministically (largest `w_j`).
        2. Compute A-optimal inclusion probabilities for the full budget.
        3. Draw the remaining `budget_tail` items using balanced sampling.

    Args:
        counts: Count matrix with units as rows and candidate items as columns.
        X: Feature matrix aligned with `counts.index`.
        budget: Total number of items to select.
        tail_frac: Fraction of the budget allocated to the probabilistic tail.
        random_state: Random state for balanced tail selection. Can be None, int, or Generator.
        ensure_full_rank: Whether to regularize `X^T X` if it is rank-deficient.
        ridge: Optional ridge penalty added to `X^T X`.

    Returns:
        Selection result containing the chosen items, inclusion probabilities, and metadata.

    Raises:
        ValidationError: If inputs fail validation or the core/tail split is infeasible.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import core_plus_tail
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(10, (1000, 200)))
        >>> X = pd.DataFrame(np.random.randn(1000, 5))
        >>> result = core_plus_tail(counts, X, budget=50, tail_frac=0.2)
        >>> result.selected.shape
        (50,)
    """
    # Validate inputs
    counts = validate_counts_matrix(counts)
    X = validate_features_matrix(X)
    counts, X = validate_data_alignment(counts, X)
    budget = validate_budget(budget, counts.shape[1])
    validate_fraction(tail_frac, "tail_frac")

    budget_core = int((1 - tail_frac) * budget)
    budget_tail = budget - budget_core

    if budget_core <= 0 or budget_tail <= 0:
        raise ValidationError(
            f"Invalid split: budget_core={budget_core}, budget_tail={budget_tail}",
            f"Adjust tail_frac for budget={budget}",
        )

    # Step 1: Deterministic core selection
    core_result = items_to_label(
        counts=counts,
        X=X,
        budget=budget_core,
        ensure_full_rank=ensure_full_rank,
        ridge=ridge,
    )
    core = core_result.selected

    # Step 2: Compute A-optimal π for full budget
    pi_result = pi_aopt_for_budget(
        counts=counts,
        X=X,
        budget=budget,
        ensure_full_rank=ensure_full_rank,
        ridge=ridge,
    )
    pi = pi_result.probabilities

    # Step 3: Balanced selection from remainder
    remainder = pi.index.difference(core)
    if len(remainder) < budget_tail:
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
        tail = balanced_fixed_size(
            pi=pi_remainder,
            g=g_remainder,
            budget=budget_tail,
            random_state=random_state,
        )
        selected = core.union(tail)

    selected.name = "selected_items"

    # Compute suggested weights
    weights_ht = compute_horvitz_thompson_weights(pi, selected)

    # Alternative "tiny-bias" weights: 1/pi for core, 1.0 for tail
    weights_mixed = pd.Series(index=selected, dtype=float)
    weights_mixed.loc[core] = (1.0 / pi).reindex(core)
    weights_mixed.loc[tail] = 1.0  # Intentional bias for variance reduction

    diagnostics = {
        "budget_core": budget_core,
        "budget_tail": budget_tail,
        "tail_frac": tail_frac,
    }

    return CoreTailResult(
        selected=selected,
        probabilities=pi,
        core=core,
        tail=tail,
        ht_weights=weights_ht,
        mixed_weights=weights_mixed,
        diagnostics=diagnostics,
    )


def adaptive_core_tail(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    budget: int,
    *,
    min_tail_frac: float = 0.1,
    max_tail_frac: float = 0.4,
    condition_threshold: float = 1e6,
    random_state: None | int | np.random.Generator = None,
) -> CoreTailResult:
    """
    Adaptive core+tail selection with a data-driven tail fraction.

    The routine increases the tail fraction when `X^T X` is poorly conditioned and decreases it
    when influence weights are highly concentrated.

    Args:
        counts: Count matrix.
        X: Feature matrix.
        budget: Total number of items to select.
        min_tail_frac: Minimum allowable tail fraction.
        max_tail_frac: Maximum allowable tail fraction.
        condition_threshold: Baseline condition number scale.
        random_state: Random state for the balanced sampling step. Can be None, int, or Generator.

    Returns:
        Selection result identical to `core_plus_tail`, with adaptive metadata in `info`.
    """
    # Validate inputs
    counts = validate_counts_matrix(counts)
    X = validate_features_matrix(X)
    counts, X = validate_data_alignment(counts, X)
    budget = validate_budget(budget, counts.shape[1])

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
    if len(w_sorted) > budget:
        # Ratio of top-budget influence to total
        concentration = w_sorted[:budget].sum() / (w.sum() + DIVISION_EPS)
    else:
        concentration = 0.5
    skew_score = 1 - concentration

    # Combine scores (equal weighting)
    combined_score = 0.5 * cond_score + 0.5 * skew_score
    tail_frac = min_tail_frac + combined_score * (max_tail_frac - min_tail_frac)

    # Use computed tail_frac
    result = core_plus_tail(
        counts=counts,
        X=X,
        budget=budget,
        tail_frac=tail_frac,
        random_state=random_state,
    )

    # Add adaptive info to existing diagnostics
    adaptive_diagnostics = result.diagnostics.copy()
    adaptive_diagnostics.update(
        {
            "adaptive": True,
            "condition_number": cond,
            "concentration_ratio": concentration,
            "adaptive_tail_frac": tail_frac,
        }
    )

    return CoreTailResult(
        selected=result.selected,
        probabilities=result.probabilities,
        core=result.core,
        tail=result.tail,
        ht_weights=result.ht_weights,
        mixed_weights=result.mixed_weights,
        diagnostics=adaptive_diagnostics,
    )

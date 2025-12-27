from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import SelectionResult

from .constants import SMALL_RIDGE
from .core import _influence
from .validation import (
    validate_budget,
    validate_counts_matrix,
    validate_data_alignment,
    validate_features_matrix,
)


def greedy_aopt_selection(
    counts: pd.DataFrame,
    X: pd.DataFrame,
    budget: int,
    *,
    ensure_full_rank: bool = True,
    ridge: float | None = None,
) -> SelectionResult:
    """
    Select items using greedy A-optimal sequential selection.

    The algorithm iteratively chooses the item that maximally reduces the trace of the covariance
    matrix using Sherman-Morrison updates.

    Args:
        counts: Count matrix with non-negative entries.
        X: Feature matrix aligned with `counts.index`.
        budget: Number of items to select sequentially.
        ensure_full_rank: Whether to add a ridge if the information matrix becomes singular.
        ridge: Optional explicit ridge parameter.

    Returns:
        Selection result with items, influence weights, and diagnostics.

    See Also:
        items_to_label: Batch A-optimal selection (faster, different results).
        pi_aopt_for_budget: Compute inclusion probabilities for A-optimal design.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import greedy_aopt_selection
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(5, (1000, 100)))
        >>> X = pd.DataFrame(np.random.randn(1000, 3))
        >>> result = greedy_aopt_selection(counts, X, budget=20)
        >>> len(result.selected)
        20
    """

    # Validate inputs
    counts = validate_counts_matrix(counts)
    X = validate_features_matrix(X)
    counts, X = validate_data_alignment(counts, X)
    budget = validate_budget(budget, counts.shape[1])

    if budget == 0:
        from .results import SelectionResult

        empty_selected = pd.Index([], name="selected_items")
        empty_weights = pd.Series([], dtype=float, name="influence_weights")
        return SelectionResult(
            selected=empty_selected,
            influence_weights=empty_weights,
            diagnostics={"method": "greedy", "budget": budget},
        )

    _, m = counts.shape

    # 1. Precompute candidate vectors g_j
    # _influence computes g_j = X^T v_j where v_j is the normalized count column
    # We don't need 'w' from _influence for the greedy step, just 'g'
    inf = _influence(counts, X, ensure_full_rank=ensure_full_rank, ridge=ridge)
    G = inf.g  # (p, m)

    # 2. Initialize covariance matrix inverse M_inv = (X'X)^-1
    # We start with the "prior" covariance. If we haven't selected anything yet,
    # our "current" information matrix is just the prior (ridge).
    # However, the standard A-opt formulation usually assumes we are *adding*
    # to an existing design or starting from scratch.
    # Here we assume we are selecting a subset to *label*.
    # The objective is to minimize Trace((X'X + sum_S g_j g_j')^-1).
    # Wait, the formulation in `core.py` implies w_j = g_j' (X'X)^-1 g_j.
    # That `w_j` is the leverage of item j *assuming* we have the full population X.
    # The goal of `items_to_label` is to pick items that give us the best estimate
    # of the regression coefficients *if we only had those items*.
    # Actually, the problem description says: "prioritizes items that... would most change your conclusions".
    # The standard "sensor selection" or "experimental design" greedy approach:
    # Start with M = epsilon * I (or prior).
    # At each step, pick j to maximize: g_j' M^{-1} g_j / (1 + g_j' M^{-1} g_j) ?
    # Or if we are minimizing Trace(M^{-1}), the reduction is proportional to that quantity.

    # Let's stick to the standard greedy A-opt update:
    # M_{k+1} = M_k + g_j g_j^T
    # We want to pick j to minimize Trace(M_{k+1}^{-1}).
    # Using Sherman-Morrison:
    # Trace(M_{k+1}^{-1}) = Trace(M_k^{-1}) - (g_j^T M_k^{-2} g_j) / (1 + g_j^T M_k^{-1} g_j)
    # So we want to MAXIMIZE: (g_j^T M_k^{-2} g_j) / (1 + g_j^T M_k^{-1} g_j)

    # Initial M = X^T X (from the full population? No, that's not right.)
    # If we label items, we get observations.
    # The "information" from item j is roughly proportional to g_j g_j^T.
    # So we start with M_0 = small_ridge * I (representing prior or regularization).
    # And we build up M.

    p = X.shape[1]
    # Start with a small ridge for stability/prior
    current_ridge = ridge if ridge is not None else SMALL_RIDGE
    M_inv = np.eye(p) / current_ridge

    selected_indices = []
    candidates = set(range(m))

    # Pre-calculate M_inv * g for all candidates to speed up first iteration?
    # No, M_inv changes every step.

    for _ in range(budget):
        best_j = -1

        # This loop is O(m * p^2) which might be slow for large m.
        # We can optimize:
        # score = (g_j^T M_inv^2 g_j) / (1 + g_j^T M_inv g_j)
        # Let u_j = M_inv g_j.
        # score = (u_j^T u_j) / (1 + g_j^T u_j)

        # We can compute U = M_inv @ G_candidates  (p x m_rem)
        # Then score_j = ||u_j||^2 / (1 + g_j^T u_j)

        # Convert candidates to list for indexing
        cand_list = list(candidates)
        if not cand_list:
            break

        G_cand = G[:, cand_list]  # (p, m_rem)
        U = M_inv @ G_cand  # (p, m_rem)

        # Numerator: sum(U**2, axis=0) -> (m_rem,)
        numer = np.sum(U**2, axis=0)

        # Denominator: 1 + sum(G_cand * U, axis=0)
        denom = 1.0 + np.sum(G_cand * U, axis=0)

        scores = numer / denom

        best_idx_in_cand = np.argmax(scores)
        best_j = cand_list[best_idx_in_cand]

        selected_indices.append(best_j)
        candidates.remove(best_j)

        # Update M_inv using Sherman-Morrison
        # M_{k+1}^{-1} = M_k^{-1} - (M_k^{-1} g g^T M_k^{-1}) / (1 + g^T M_k^{-1} g)
        #              = M_k^{-1} - (u u^T) / (1 + g^T u)

        u_best = U[:, best_idx_in_cand]  # (p,)
        denom_best = denom[best_idx_in_cand]

        # Outer product update
        M_inv -= np.outer(u_best, u_best) / denom_best

    from .results import SelectionResult

    selected_items = pd.Index([inf.cols[i] for i in selected_indices])
    selected_items.name = "selected_items"

    # Create influence weights series for the result
    influence_weights = pd.Series(inf.w, index=inf.cols, name="influence_weights")

    diagnostics = {
        "method": "greedy",
        "budget": budget,
        "ensure_full_rank": ensure_full_rank,
        "ridge": ridge,
    }

    return SelectionResult(
        selected=selected_items,
        influence_weights=influence_weights,
        diagnostics=diagnostics,
    )

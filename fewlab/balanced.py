from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .constants import (
    DIVISION_EPS,
    MAX_SWAPS_BALANCED,
    SEARCH_LIMIT_BALANCED,
    TOLERANCE_DEFAULT,
    TOLERANCE_STRICT,
)
from .utils import _get_random_generator
from .validation import (
    ValidationError,
    validate_budget,
    validate_probability_series,
)


def balanced_fixed_size(
    pi: pd.Series,
    g: np.ndarray,
    budget: int,
    *,
    random_state: None | int | np.random.Generator = None,
    max_swaps: int = MAX_SWAPS_BALANCED,
    tol: float = TOLERANCE_DEFAULT,
) -> pd.Index:
    """
    Fixed-size balanced sampling with variance reduction.

    Implements a two-step heuristic:

    1. Initial selection proportional to inclusion probabilities pi
    2. Greedy local search to minimize calibration residual ||sum((I/pi)-1) g||_2

    This balancing procedure aims to reduce the variance of Horvitz-Thompson estimators by
    making the sample more representative.

    Args:
        pi: Inclusion probabilities for items. Index contains item identifiers.
        g: Regression projections g_j = X^T v_j for each item j (shape (p, m)).
        budget: Fixed sample size (number of items to select).
        random_state: Random state for reproducible sampling. Can be None, int, or Generator.
        max_swaps: Maximum number of swap iterations for balancing.
        tol: Tolerance for stopping criterion (residual norm).

    Returns:
        Index of selected items. Length equals `budget`.

    Raises:
        ValidationError: If `pi`, `g`, or `budget` fail validation checks.

    See Also:
        pi_aopt_for_budget: Compute optimal inclusion probabilities.
        core_plus_tail: Hybrid deterministic + balanced sampling.
        calibrate_weights: Post-stratification weight adjustment.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import pi_aopt_for_budget, balanced_fixed_size, _influence
        >>>
        >>> # Setup data
        >>> counts = pd.DataFrame(np.random.poisson(5, (1000, 100)))
        >>> X = pd.DataFrame(np.random.randn(1000, 3))
        >>>
        >>> # Compute probabilities and influence matrix
        >>> pi = pi_aopt_for_budget(counts, X, budget=30)
        >>> inf = _influence(counts, X)
        >>>
        >>> # Balanced sampling
        >>> selected = balanced_fixed_size(pi, inf.g, budget=30, random_state=42)
        >>> print(f"Selected {len(selected)} items with balanced design")

    Notes:
        The balancing algorithm aims to make sum_S (I_j/pi_j - 1) * g_j ≈ 0, where S is the
        selected sample and I_j are selection indicators. This reduces variance in calibrated
        estimators.
    """

    # Validate inputs
    pi = validate_probability_series(pi)
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        raise ValidationError(
            "g must be a 2D numpy array", "Ensure g has shape (n_features, n_items)"
        )

    m: int = len(pi)
    if g.shape[1] != m:
        raise ValidationError(
            f"g has {g.shape[1]} columns but pi has {m} items",
            "Ensure g and pi represent the same items in the same order",
        )

    budget = validate_budget(budget, m)

    rng: np.random.Generator = _get_random_generator(random_state)
    cols: pd.Index = pi.index

    # 1) Initial budget-draw proportional to pi
    probs: np.ndarray = pi.to_numpy(float)
    probs = probs / probs.sum()
    init: np.ndarray = rng.choice(m, size=budget, replace=False, p=probs)
    selected: np.ndarray = np.zeros(m, dtype=bool)
    selected[init] = True

    inv_pi: np.ndarray = 1.0 / (pi.to_numpy(float) + DIVISION_EPS)
    # current residual R = sum((I/pi)-1) g
    coeff: np.ndarray = selected * inv_pi - 1.0
    R: np.ndarray = g @ coeff  # (p,)

    # 2) Greedy local search: try swaps that reduce ||R||_2
    # Precompute convenience arrays
    in_idx: np.ndarray = np.flatnonzero(selected)
    out_idx: np.ndarray = np.flatnonzero(~selected)

    def norm2(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    improved: bool = True
    nswaps: int = 0
    best_norm: float = norm2(R)

    while improved and nswaps < max_swaps:
        improved = False
        # try a random subset of candidates to keep O(max_swaps) bounded
        rng.shuffle(in_idx)
        rng.shuffle(out_idx)
        tried: int = 0
        for j_in in in_idx[: min(len(in_idx), SEARCH_LIMIT_BALANCED)]:
            for j_out in out_idx[: min(len(out_idx), SEARCH_LIMIT_BALANCED)]:
                tried += 1
                # delta R = g(:,j_out)*(1/pi_out) - g(:,j_in)*(1/pi_in)
                dR: np.ndarray = g[:, j_out] * inv_pi[j_out] - g[:, j_in] * inv_pi[j_in]
                new_norm: float = norm2(R + dR)
                if new_norm + TOLERANCE_STRICT < best_norm:
                    # commit swap
                    selected[j_in] = False
                    selected[j_out] = True
                    R = R + dR
                    best_norm = new_norm
                    # update candidate lists
                    in_idx = np.flatnonzero(selected)
                    out_idx = np.flatnonzero(~selected)
                    improved = True
                    nswaps += 1
                    break
            if improved:
                break
        if best_norm < tol:
            break

    return cast(pd.Index, cols[selected])

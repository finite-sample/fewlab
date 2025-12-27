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


def balanced_fixed_size(
    pi: pd.Series,
    g: np.ndarray,
    K: int,
    *,
    seed: int | None = None,
    max_swaps: int = MAX_SWAPS_BALANCED,
    tol: float = TOLERANCE_DEFAULT,
) -> pd.Index:
    """
    Heuristic fixed-size sampler that:
      1) starts with a K-sized draw from pi (normalized),
      2) greedily swaps in/out items to reduce ||sum((I/pi)-1) g||_2.

    Parameters
    ----------
    pi : Series (length m), index=items
    g  : ndarray (p x m) regression projections g_j
    K  : int fixed sample size

    Returns
    -------
    Pandas Index of selected item ids (length K).
    """
    m: int = len(pi)
    if K <= 0 or K > m:
        raise ValueError("K must be in [1, m]")

    rng: np.random.Generator = np.random.default_rng(seed)
    cols: pd.Index = pi.index

    # 1) Initial K-draw proportional to pi
    probs: np.ndarray = pi.to_numpy(float)
    probs = probs / probs.sum()
    init: np.ndarray = rng.choice(m, size=K, replace=False, p=probs)
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

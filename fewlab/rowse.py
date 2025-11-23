from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    DIVISION_EPS,
    DUAL_START_VALUE,
    MAX_ITER_ROWSE,
    NOISE_SCALE,
    PI_MIN_DEFAULT,
    TOLERANCE_DEFAULT,
)


def row_se_min_labels(
    counts: pd.DataFrame,
    eps2: np.ndarray | pd.Series,
    *,
    pi_min: float = PI_MIN_DEFAULT,
    max_iter: int = MAX_ITER_ROWSE,
    tol: float = TOLERANCE_DEFAULT,
    seed: int | None = None,
) -> pd.Series:
    """
    Fewest-labels design subject to per-row SE caps:
       sum_j q_ij / pi_j <= eps2_i + sum_j q_ij  for all rows i,
    where q_ij = (c_ij / T_i)^2.

    Returns
    -------
    Series of pi_j (index = item ids).
    """
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a DataFrame")

    T: np.ndarray = counts.sum(axis=1).to_numpy(float)
    keep: np.ndarray = T > 0
    if not np.all(keep):
        counts = counts.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValueError("All rows have zero totals")

    q: np.ndarray = (counts.to_numpy(float) / T[:, None]) ** 2  # (n x m)
    n: int
    m: int
    n, m = q.shape
    cols: list[str] = list(counts.columns)

    eps2 = np.asarray(eps2, dtype=float)
    if eps2.ndim == 0:
        eps2 = np.full(n, float(eps2))
    if eps2.size != n:
        raise ValueError("eps2 must be a scalar or length n")

    b: np.ndarray = eps2 + q.sum(axis=1)  # (n,)
    mu: np.ndarray = np.full(n, DUAL_START_VALUE)  # dual start

    def primal_from_mu(mu_vec: np.ndarray) -> np.ndarray:
        s: np.ndarray = q.T @ mu_vec  # (m,)
        pi: np.ndarray = np.sqrt(s + DIVISION_EPS)
        return np.clip(pi, pi_min, 1.0)

    pi: np.ndarray = primal_from_mu(mu)

    def lhs(pi_vec: np.ndarray) -> np.ndarray:
        return (q / (pi_vec[None, :] + DIVISION_EPS)).sum(axis=1)

    L = lhs(pi)
    viol = L - b
    best_pi, best_max_viol = pi.copy(), float(np.max(viol))

    rng = np.random.default_rng(seed)
    for _t in range(1, max_iter + 1):
        if best_max_viol <= tol:
            break
        # subgradient step on mu: mu <- [mu + eta*(L - b)]_+
        g = np.maximum(viol, 0.0)
        gnorm = float(np.linalg.norm(g))
        eta = 0.5 / (1.0 + gnorm)
        # small random jitter helps avoid cycling
        mu = np.maximum(0.0, mu + eta * g + rng.normal(scale=NOISE_SCALE, size=n))
        pi = primal_from_mu(mu)
        L = lhs(pi)
        viol = L - b
        mv = float(np.max(viol))
        if mv < best_max_viol:
            best_max_viol = mv
            best_pi = pi.copy()

    return pd.Series(best_pi, index=cols, name="pi")

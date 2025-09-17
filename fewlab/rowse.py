from __future__ import annotations
import numpy as np
import pandas as pd


def row_se_min_labels(
    counts: pd.DataFrame,
    eps2: np.ndarray | pd.Series,
    *,
    pi_min: float = 1e-4,
    max_iter: int = 8000,
    tol: float = 1e-6,
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

    T = counts.sum(axis=1).to_numpy(float)
    keep = T > 0
    if not np.all(keep):
        counts = counts.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValueError("All rows have zero totals")

    q = (counts.to_numpy(float) / T[:, None]) ** 2  # (n x m)
    n, m = q.shape
    cols = list(counts.columns)

    eps2 = np.asarray(eps2, dtype=float)
    if eps2.ndim == 0:
        eps2 = np.full(n, float(eps2))
    if eps2.size != n:
        raise ValueError("eps2 must be a scalar or length n")

    b = eps2 + q.sum(axis=1)  # (n,)
    mu = np.full(n, 1e-4)  # dual start

    def primal_from_mu(mu_vec: np.ndarray) -> np.ndarray:
        s = q.T @ mu_vec  # (m,)
        pi = np.sqrt(s + 1e-18)
        return np.clip(pi, pi_min, 1.0)

    pi = primal_from_mu(mu)

    def lhs(pi_vec: np.ndarray) -> np.ndarray:
        return (q / (pi_vec[None, :] + 1e-18)).sum(axis=1)

    L = lhs(pi)
    viol = L - b
    best_pi, best_max_viol = pi.copy(), float(np.max(viol))

    rng = np.random.default_rng(seed)
    for t in range(1, max_iter + 1):
        if best_max_viol <= tol:
            break
        # subgradient step on mu: mu <- [mu + eta*(L - b)]_+
        g = np.maximum(viol, 0.0)
        gnorm = float(np.linalg.norm(g))
        eta = 0.5 / (1.0 + gnorm)
        # small random jitter helps avoid cycling
        mu = np.maximum(0.0, mu + eta * g + rng.normal(scale=1e-6, size=n))
        pi = primal_from_mu(mu)
        L = lhs(pi)
        viol = L - b
        mv = float(np.max(viol))
        if mv < best_max_viol:
            best_max_viol = mv
            best_pi = pi.copy()

    return pd.Series(best_pi, index=cols, name="pi")

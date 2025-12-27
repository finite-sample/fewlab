from __future__ import annotations

import warnings

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
from .results import RowSEResult
from .utils import _get_random_generator
from .validation import (
    ValidationError,
    validate_counts_matrix,
)


def row_se_min_labels(
    counts: pd.DataFrame,
    eps2: np.ndarray | pd.Series,
    *,
    pi_min: float = PI_MIN_DEFAULT,
    max_iter: int = MAX_ITER_ROWSE,
    tol: float = TOLERANCE_DEFAULT,
    random_state: None | int | np.random.Generator = None,
    return_result: bool = False,
    raise_on_failure: bool = False,
) -> RowSEResult | pd.Series:
    """
    Compute inclusion probabilities that minimize total expected labels under row-wise SE limits.

    The routine solves:

    ```
    minimize   sum_j pi_j
    subject to sum_j q_ij / pi_j <= eps2_i + sum_j q_ij,  q_ij = (c_ij / T_i)^2
    ```

    Args:
        counts: Non-negative count matrix with units as rows and items as columns.
        eps2: Row-wise squared standard-error tolerance; scalar applies to every row.
        pi_min: Minimum allowable inclusion probability.
        max_iter: Maximum optimization iterations.
        tol: Convergence tolerance for constraint violations.
        random_state: Random state for the stochastic subgradient steps. Can be None, int, or Generator.
        return_result: If True, return a `RowSEResult` with diagnostics.
        raise_on_failure: If True, raise a `ValidationError` when constraints remain violated.

    Returns:
        Probability series if `return_result` is False (default) or a `RowSEResult` with
        diagnostics when `return_result` is True.

    Raises:
        ValidationError: If inputs are invalid or `raise_on_failure` is True and the constraints
            remain violated after optimization.

    See Also:
        pi_aopt_for_budget: A-optimal probabilities for a fixed budget.
        items_to_label: Deterministic selection without SE constraints.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import row_se_min_labels
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(10, (500, 50)))
        >>> pi = row_se_min_labels(counts, eps2=0.05**2)
        >>> float(pi.sum())
    """
    # Validate inputs
    counts = validate_counts_matrix(counts)

    T: np.ndarray = counts.sum(axis=1).to_numpy(float)
    keep: np.ndarray = T > 0
    if not np.all(keep):
        counts = counts.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValidationError(
                "All rows have zero totals",
                "Remove rows with no usage data or check your count matrix",
            )

    q: np.ndarray = (counts.to_numpy(float) / T[:, None]) ** 2  # (n x m)
    n, _ = q.shape
    cols: list[str] = list(counts.columns)

    eps2 = np.asarray(eps2, dtype=float)
    if eps2.ndim == 0:
        eps2 = np.full(n, float(eps2))
    if eps2.size != n:
        raise ValidationError(
            f"eps2 has {eps2.size} values but counts has {n} rows",
            "Provide either a scalar eps2 or an array with one value per row",
        )

    if (eps2 <= 0).any():
        raise ValidationError(
            "eps2 values must be positive (SE^2 > 0)",
            "Use positive values for SE^2 tolerances, e.g., eps2=0.05**2 for 5% SE",
        )

    b: np.ndarray = eps2 + q.sum(axis=1)  # (n,)
    mu: np.ndarray = np.full(n, DUAL_START_VALUE)  # dual start

    def primal_from_mu(mu_vec: np.ndarray) -> np.ndarray:
        s: np.ndarray = q.T @ mu_vec  # (m,)
        pi: np.ndarray = np.sqrt(s + DIVISION_EPS)
        return np.clip(pi, pi_min, 1.0)

    pi: np.ndarray = primal_from_mu(mu)

    def lhs(pi_vec: np.ndarray) -> np.ndarray:
        result: np.ndarray = (q / (pi_vec[None, :] + DIVISION_EPS)).sum(axis=1)
        return result

    L = lhs(pi)
    viol = L - b
    best_pi, best_max_viol = pi.copy(), float(np.max(viol))

    rng = _get_random_generator(random_state)
    best_iteration = 0
    iterations_run = 0
    for iteration in range(1, max_iter + 1):
        iterations_run = iteration
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
            best_iteration = iteration

    feasible = best_max_viol <= tol
    diagnostics = {
        "max_violation": max(0.0, best_max_viol),
        "tolerance": tol,
        "iterations": iterations_run,
        "best_iteration": best_iteration,
        "feasible": feasible,
    }

    pi_series = pd.Series(best_pi, index=cols, name="pi")
    pi_series.attrs["row_se_diagnostics"] = diagnostics

    if not feasible:
        message = (
            f"row_se_min_labels returned infeasible probabilities: "
            f"max violation {best_max_viol:.3e} exceeds tolerance {tol:.3e}"
        )
        suggestion = "Relax eps2, increase pi_min, or allow more iterations/tolerance."
        if raise_on_failure:
            raise ValidationError(message, suggestion)
        warnings.warn(message, UserWarning, stacklevel=2)

    result = RowSEResult(
        probabilities=pi_series,
        max_violation=diagnostics["max_violation"],
        tolerance=tol,
        iterations=iterations_run,
        best_iteration=best_iteration,
        feasible=feasible,
    )

    return result if return_result else pi_series

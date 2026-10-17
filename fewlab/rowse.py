"""Inclusion probabilities under row-wise standard-error constraints.

Instead of fixing a budget and minimizing variance, this module fixes a
per-row variance ceiling and minimizes the expected number of labels.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    DIVISION_EPS,
    DUAL_START_VALUE,
    MAX_ITER_ROWSE,
    PI_MIN_DEFAULT,
    STALL_PATIENCE,
    TOLERANCE_DEFAULT,
)
from .validation import (
    ValidationError,
    validate_counts_matrix,
)


@dataclass(frozen=True, slots=True)
class RowSEResult:
    """Probabilities and convergence detail from the row-SE optimizer.

    Attributes:
        probabilities: Inclusion probabilities, indexed by entity.
        expected_labels: The objective, `sum_j pi_j`.
        max_violation: Largest remaining constraint violation.
        tolerance: Feasibility tolerance used.
        iterations: Dual steps taken.
        best_iteration: Step at which the returned design was found.
        feasible: Whether every row-SE ceiling is met.
    """

    probabilities: pd.Series
    expected_labels: float
    max_violation: float
    tolerance: float
    iterations: int
    best_iteration: int
    feasible: bool

    def to_series(self) -> pd.Series:
        """Return just the probabilities.

        Returns:
            The inclusion probabilities.
        """
        return self.probabilities


def row_se_min_labels(
    counts: pd.DataFrame,
    eps2: np.ndarray | pd.Series,
    *,
    pi_min: float = PI_MIN_DEFAULT,
    max_iter: int = MAX_ITER_ROWSE,
    tol: float = TOLERANCE_DEFAULT,
    return_result: bool = False,
    raise_on_failure: bool = False,
) -> RowSEResult | pd.Series:
    """Minimize expected labels subject to row-wise SE limits.

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
        return_result: If True, return a `RowSEResult` with diagnostics.
        raise_on_failure: If True, raise a `ValidationError` when constraints
            remain violated.

    Returns:
        Probability series if `return_result` is False (default) or a `RowSEResult` with
        diagnostics when `return_result` is True.

    Raises:
        ValidationError: If inputs are invalid or `raise_on_failure` is True and
            the constraints
            remain violated after optimization.

    See Also:
        pi_aopt_for_budget: A-optimal probabilities for a fixed budget.
        items_to_label: Deterministic selection without SE constraints.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import row_se_min_labels
        >>> rng = np.random.default_rng(0)
        >>> counts = pd.DataFrame(rng.poisson(10, (200, 20)))
        >>> pi = row_se_min_labels(counts, eps2=0.3**2)
        >>> bool(0 < pi.sum() <= counts.shape[1])
        True
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
        # Minimizing pi_j + (sum_i mu_i q_ij)/pi_j over pi_j gives the exact
        # stationary point sqrt(sum_i mu_i q_ij).
        s: np.ndarray = q.T @ mu_vec  # (m,)
        pi: np.ndarray = np.sqrt(s + DIVISION_EPS)
        return np.clip(pi, pi_min, 1.0)

    def lhs(pi_vec: np.ndarray) -> np.ndarray:
        result: np.ndarray = (q / (pi_vec[None, :] + DIVISION_EPS)).sum(axis=1)
        return result

    pi: np.ndarray = primal_from_mu(mu)
    L = lhs(pi)
    viol = L - b

    # The objective is sum_j pi_j, so track the cheapest *feasible* iterate.
    # Tracking the smallest constraint violation instead -- as this did -- stops
    # at whatever design first became feasible, which is a feasibility check
    # rather than a minimization.
    best_pi = pi.copy()
    best_obj = float("inf")
    best_max_viol = float(np.max(viol))
    fallback_pi = pi.copy()

    best_iteration = 0
    iterations_run = 0
    stale = 0
    for iteration in range(1, max_iter + 1):
        iterations_run = iteration
        # Dual ascent on mu with the *signed* violation. Clipping the gradient
        # at zero -- as this did -- means a multiplier can never fall when its
        # constraint goes slack, so probabilities ratchet upward and the search
        # halts at first feasibility instead of minimizing the objective.
        # Projection onto mu >= 0 is the np.maximum below, not the gradient.
        gnorm = float(np.linalg.norm(viol))
        eta = 0.5 / (1.0 + gnorm)
        mu = np.maximum(0.0, mu + eta * viol)
        pi = primal_from_mu(mu)
        L = lhs(pi)
        viol = L - b
        mv = float(np.max(viol))
        obj = float(pi.sum())

        if mv < best_max_viol:
            best_max_viol = mv
            fallback_pi = pi.copy()
        if mv <= tol and obj < best_obj - tol:
            best_obj = obj
            best_pi = pi.copy()
            best_iteration = iteration
            stale = 0
        elif best_obj < float("inf"):
            stale += 1
            if stale >= STALL_PATIENCE:
                break

    if best_obj == float("inf"):
        best_pi = fallback_pi

    # Dual ascent converges at O(1/sqrt(k)), which is fine for finding the shape
    # of the solution and hopeless for driving the last 1e-4 of violation to
    # 1e-6. Repair it directly instead: scaling every probability up by a common
    # factor is monotone in every constraint, so a bisection on that factor
    # finds the smallest feasible rescaling. It always succeeds, because pi = 1
    # everywhere gives variance zero and satisfies every ceiling by construction
    # -- the row-SE problem cannot be infeasible.
    best_pi, scale = _repair(best_pi, lhs, b, tol, pi_min)
    best_max_viol = float(np.max(lhs(best_pi) - b))

    feasible = best_max_viol <= tol
    diagnostics = {
        "expected_labels": float(best_pi.sum()),
        "repair_scale": scale,
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
        expected_labels=float(best_pi.sum()),
        max_violation=diagnostics["max_violation"],
        tolerance=tol,
        iterations=iterations_run,
        best_iteration=best_iteration,
        feasible=feasible,
    )

    return result if return_result else pi_series


def _repair(
    pi: np.ndarray,
    lhs: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    tol: float,
    pi_min: float,
) -> tuple[np.ndarray, float]:
    """Rescale probabilities upward by the smallest factor that is feasible.

    Every constraint left-hand side `sum_j q_ij / pi_j` is decreasing in a common
    positive scaling of `pi`, so feasibility is monotone in the factor and a
    bisection finds the smallest one. The upper end is always feasible, since
    `pi = 1` everywhere makes the design-variance term vanish.

    Args:
        pi: Probabilities from the dual ascent.
        lhs: Constraint left-hand side as a function of probabilities.
        b: Constraint right-hand sides.
        tol: Feasibility tolerance.
        pi_min: Lower clip for probabilities.

    Returns:
        The repaired probabilities and the scaling factor applied.
    """
    if float(np.max(lhs(pi) - b)) <= tol:
        return pi, 1.0
    lo, hi = 1.0, max(2.0, float(1.0 / max(pi.min(), pi_min)))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.max(lhs(np.clip(mid * pi, pi_min, 1.0)) - b)) <= tol:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return np.clip(hi * pi, pi_min, 1.0), float(hi)

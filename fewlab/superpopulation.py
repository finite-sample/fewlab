"""Estimating a superpopulation coefficient rather than a finite-population one.

The rest of this package targets `B`, the coefficient obtained by labeling every
entity in the frame. That is a fixed number, and the only linear estimator
unbiased for it is ordinary least squares applied to `yhat`, because any
estimator `L yhat` has expectation `L y`, and `L y = (X'X)^-1 X' y` for an
arbitrary unknown `y` forces `L = (X'X)^-1 X'`. Generalized least squares uses a
different `L` and is therefore design-*biased* for `B`: not less efficient, but
aimed elsewhere.

It is aimed somewhere real. If the units are a sample from a population in which
`y = X beta + eps`, then `beta` rather than `B` is the target, `yhat = y + u` has
error variance `sigma^2 I + Var(u)`, and that variance is markedly
heteroskedastic across units. A unit whose activity concentrates on the
certainty core carries almost no design noise; a unit spread across obscure
entities carries a great deal. GLS exploits exactly that.

What makes it unusually safe here is that `Var(u)` is not estimated from
residuals. The design chose `pi`, so up to the labels the covariance is known,
which removes the usual objection that GLS collapses when the weight matrix is
poorly estimated.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .counts import CountMatrix, as_counts
from .estimator import LabelSample
from .influence import Influence


@dataclass(frozen=True, slots=True)
class SuperpopulationEstimate:
    """A GLS coefficient and its standard error, for a superpopulation target.

    Attributes:
        coefficients: `(p, q)` estimate of `beta`.
        standard_errors: Standard errors from the GLS sandwich.
        sigma2: Estimated residual variance per label class.
        ols: The ordinary least squares estimate on the same data, for
            comparison. It is unbiased for the finite-population `B` and is what
            the rest of the package reports.
        diagnostics: Includes the spread of per-unit design variance, which is
            what GLS is exploiting; with no spread there is nothing to gain.
    """

    coefficients: pd.DataFrame
    standard_errors: pd.DataFrame
    sigma2: np.ndarray
    ols: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


def _qr_factor(matrix: np.ndarray) -> np.ndarray:
    """Return the triangular factor without allocating orthogonal columns."""
    return np.asarray(np.linalg.qr(matrix, mode="r"))


def _gls(
    X: np.ndarray, y: np.ndarray, Z: np.ndarray, residual_sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit GLS with covariance `residual_sd**2 I + Z Z.T` using square roots.

    Args:
        X: Unit features.
        y: Estimated outcome for one label class.
        Z: Share columns scaled by their design standard deviations.
        residual_sd: Standard deviation of the model error.

    Returns:
        Coefficients and their model-based standard errors.
    """
    n, k = Z.shape
    p = X.shape[1]
    responses = np.column_stack((X, y))
    if k < n:
        # Eliminate u from min ||y - X beta - Z u||² + residual_sd² ||u||².
        # The remaining block is residual_sd times the whitened regression.
        augmented = np.zeros((n + k, k + p + 1))
        augmented[:n, :k] = Z
        augmented[:n, k:] = responses
        np.fill_diagonal(augmented[n:, :k], residual_sd)
        factor = _qr_factor(augmented)[k : k + p, k:]
        scale = residual_sd
    else:
        # The covariance has the smaller dimension. R.T R equals the covariance.
        stacked = np.vstack((Z.T, residual_sd * np.eye(n)))
        root = _qr_factor(stacked)
        whitened = np.linalg.solve(root.T, responses)
        factor = _qr_factor(whitened)[:p]
        scale = 1.0

    design = factor[:, :p]
    solution = np.linalg.solve(design, np.column_stack((factor[:, p], np.eye(p))))
    return solution[:, 0], scale * np.linalg.norm(solution[:, 1:], axis=1)


def estimate_superpopulation(
    infl: Influence,
    counts: pd.DataFrame | CountMatrix,
    X: pd.DataFrame,
    sample: LabelSample,
    labels: pd.Series | pd.DataFrame,
) -> SuperpopulationEstimate:
    """Estimate a superpopulation coefficient by design-weighted GLS.

    This does not estimate the same quantity as `fewlab.estimate`. Read the
    module docstring before using it: if the target is "the coefficient I would
    obtain by labeling everything", this function is biased for it and
    `fewlab.estimate` is not.

    The weight matrix is `Sigma = sigma^2 I + Var(u)`, where the design
    contributes `Var(u) = sum_j (1-pi_j)/pi_j a_j^2 v_j v_j'`. That is diagonal
    plus a term of rank at most the sample size. Square-root GLS works with
    the smaller of the unit count and the number of entities with positive
    design variance. It remains stable when the residual variance hits its
    floor and the design accounts for nearly all the uncertainty.

    Args:
        infl: Influence decomposition for the frame.
        counts: The count matrix.
        X: Unit feature matrix.
        sample: The realized draw, carrying delivered probabilities.
        labels: Observed labels for the sampled entities.

    Returns:
        The GLS estimate, the OLS estimate beside it, and diagnostics.

    Raises:
        ValueError: If the sample references entities outside the frame, or a
            sampled entity has no label.
        np.linalg.LinAlgError: If design variance is positive and the feature
            coefficients are not separately identifiable.
    """
    C = as_counts(counts)
    T = C.row_totals()
    keep = T > 0
    C = C.select_rows(keep)
    T = T[keep]

    items = pd.Index(infl.items)
    pos = items.get_indexer(pd.Index(sample.items))
    if (pos < 0).any():
        raise ValueError("sample contains entities absent from the influence frame")

    frame = labels.to_frame() if isinstance(labels, pd.Series) else labels
    A_s = frame.reindex(pd.Index(sample.items))
    if A_s.isna().to_numpy().any():
        raise ValueError("labels are missing for entities that were sampled")
    classes = pd.Index(A_s.columns)
    A_arr = A_s.to_numpy(float)

    pi = sample.probabilities.reindex(sample.items).to_numpy(float)
    V_s = C.dense_columns(pos) / T[:, None]
    Xn = X.reindex(C.index).to_numpy(float)
    n, p = Xn.shape

    y_hat = V_s @ (A_arr / pi[:, None])
    ols, _, rank, _ = np.linalg.lstsq(Xn, y_hat, rcond=None)

    coefs, ses, sigmas = [], [], []
    for q in range(A_arr.shape[1]):
        # Estimated design weights for this label class, one per sampled entity.
        d = (1.0 - pi) / pi**2 * A_arr[:, q] ** 2
        resid = y_hat[:, q] - Xn @ ols[:, q]
        design_var = np.einsum("nk,k,nk->n", V_s, d, V_s)
        sigma2 = max(float((resid @ resid - design_var.sum()) / max(n - p, 1)), 1e-12)
        sigmas.append(sigma2)

        active = d > 0
        if not active.any():
            coefs.append(ols[:, q])
            ses.append(np.sqrt(sigma2 * np.diag(np.linalg.pinv(Xn.T @ Xn))))
            continue

        if rank < p:
            raise np.linalg.LinAlgError("GLS requires linearly independent features")
        Z = V_s[:, active] * np.sqrt(d[active])
        beta, se = _gls(Xn, y_hat[:, q], Z, np.sqrt(sigma2))
        coefs.append(beta)
        ses.append(se)

    features = pd.Index(X.columns)
    spread = np.einsum(
        "nk,k,nk->n", V_s, (1.0 - pi) / pi**2 * (A_arr**2).sum(axis=1), V_s
    )
    positive = spread[spread > 0]
    return SuperpopulationEstimate(
        coefficients=pd.DataFrame(np.array(coefs).T, index=features, columns=classes),
        standard_errors=pd.DataFrame(np.array(ses).T, index=features, columns=classes),
        sigma2=np.array(sigmas),
        ols=pd.DataFrame(ols, index=features, columns=classes),
        diagnostics={
            "design_variance_spread": (
                float(np.percentile(positive, 90) / np.percentile(positive, 10))
                if positive.size
                else 1.0
            ),
            "n_units": n,
            "n_labeled": len(sample.items),
        },
    )

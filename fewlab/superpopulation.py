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


def _gls(
    X: np.ndarray, y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit GLS with covariance `residual_sd**2 I + Z Z.T`.

    For fixed entity errors u, the least-squares coefficients fit y - Z u.
    Projecting out the features leaves the ridge problem
    min ||residual_y - residual_Z u||² + residual_sd² ||u||².

    Args:
        X: Full-rank unit features.
        y: Estimated outcome for one label class.
        Z: Share columns scaled by their design standard deviations.

    Returns:
        Coefficients, their model-based standard errors, and residual variance.
    """
    Q, R = np.linalg.qr(X, mode="reduced")
    feature_shares = Q.T @ Z
    feature_y = Q.T @ y
    residual_y = y - Q @ feature_y
    residual_shares = Z - Q @ feature_shares
    U, singular, Vt = np.linalg.svd(residual_shares, full_matrices=False)
    # The OLS residual sum of squares includes only design noise orthogonal
    # to X. Its expected contribution is the squared norm of the projected Z.
    rss = residual_y @ residual_y
    design_trace = singular @ singular
    sigma2 = max(float((rss - design_trace) / max(X.shape[0] - X.shape[1], 1)), 1e-12)
    residual_sd = np.sqrt(sigma2)
    # Removing the features can leave roundoff when a share lies in their span.
    # Use the scale before projection to recognize these zero singular values.
    cutoff = np.finfo(float).eps * max(Z.shape) * np.linalg.norm(Z)
    singular[singular <= cutoff] = 0.0
    denominator = singular**2 + sigma2
    rotated = feature_shares @ Vt.T
    ridge_scores = (singular / denominator) * (U.T @ residual_y)
    beta = np.linalg.solve(R, feature_y - rotated @ ridge_scores)

    # The conditional entity covariance shrinks each right singular direction
    # by residual_sd² / denominator and keeps its orthogonal complement.
    factors = [
        residual_sd * np.eye(R.shape[0]),
        rotated * (residual_sd / np.sqrt(denominator)),
    ]
    if Vt.shape[0] < Vt.shape[1]:
        factors.append(feature_shares - rotated @ Vt)
    covariance_root = np.linalg.solve(R, np.column_stack(factors))
    return beta, np.linalg.norm(covariance_root, axis=1), sigma2


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
    plus a term of rank at most the sample size. Fitting the equivalent
    penalized least-squares problem by a thin singular-value decomposition
    avoids a solve over every sampled entity when there are few units.

    To estimate the residual variance, subtract the design variance remaining
    after fitting the unit features from the OLS residual sum of squares, then
    divide by the residual degrees of freedom. Design noise lying in the
    feature span can change coefficient uncertainty while leaving the residuals
    unchanged.

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
        active = d > 0
        if active.any():
            if rank < p:
                raise np.linalg.LinAlgError(
                    "GLS requires linearly independent features"
                )
            Z = V_s[:, active] * np.sqrt(d[active])
            beta, se, sigma2 = _gls(Xn, y_hat[:, q], Z)
        else:
            beta = ols[:, q]
            resid = y_hat[:, q] - Xn @ beta
            sigma2 = max(float((resid @ resid) / max(n - rank, 1)), 1e-12)
            se = np.sqrt(sigma2 * np.diag(np.linalg.pinv(Xn.T @ Xn)))
        sigmas.append(sigma2)
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

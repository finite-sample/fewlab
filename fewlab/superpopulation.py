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


def _whiten(
    mat: np.ndarray, Vk: np.ndarray, inner: np.ndarray, sigma2: float
) -> np.ndarray:
    """Apply `Sigma^-1` to a matrix through the Woodbury identity.

    `Sigma = sigma^2 I + Vk D Vk'` is diagonal plus a term of rank at most the
    sample size, so its inverse costs a solve of the sample's size rather than
    of the unit count.

    Args:
        mat: Matrix to premultiply.
        Vk: Share columns for entities carrying positive design weight.
        inner: `sigma^2 D^-1 + Vk' Vk`, the inner matrix of the Woodbury form.
        sigma2: Residual variance.

    Returns:
        `Sigma^-1 @ mat`.
    """
    return (mat - Vk @ np.linalg.solve(inner, Vk.T @ mat)) / sigma2


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
    plus a low-rank term of rank at most the sample size, so the Woodbury
    identity turns an `n by n` inverse into a solve of the size of the sample:

        Sigma^-1 = (1/s)[ I - V_S (s D^-1 + V_S' V_S)^-1 V_S' ]

    with `s = sigma^2` and `D` the diagonal of estimated design weights.

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
    ols = np.linalg.lstsq(Xn, y_hat, rcond=None)[0]

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

        Vk = V_s[:, active]
        inner = sigma2 * np.diag(1.0 / d[active]) + Vk.T @ Vk

        XtSX = Xn.T @ _whiten(Xn, Vk, inner, sigma2)
        rhs = Xn.T @ _whiten(y_hat[:, q : q + 1], Vk, inner, sigma2)
        beta = np.linalg.solve(XtSX, rhs)[:, 0]
        coefs.append(beta)
        ses.append(np.sqrt(np.clip(np.diag(np.linalg.inv(XtSX)), 0.0, None)))

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

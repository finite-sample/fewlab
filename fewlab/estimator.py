"""Design-based estimation from a realized label sample.

Nothing here accepts a set of entities on its own. An estimator needs the
probability with which each entity *could* have been labeled, not merely which
ones were, so every function takes a `LabelSample` carrying the probabilities the
sampler actually delivered. That is the whole reason the type exists: a ranking
of entities by importance is not a probability sample, and no probability
assigned to it after the fact makes it one.

Two variances are reported, and they answer different questions:

    Var(beta_hat) = sigma^2 (X'X)^-1     +     sum_j (1-pi_j)/pi_j b_j
                    fixed by your n units      fixed by your labeling budget

Only the second moves when you spend more on labels. Their ratio is a stopping
rule: once the design term is small next to the regression term, further labels
buy almost nothing.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .counts import CountMatrix, as_counts
from .influence import Influence


@dataclass(frozen=True, slots=True)
class LabelSample:
    """A realized draw from a probability design.

    Attributes:
        items: The entities that were labeled.
        probabilities: The probabilities the sampler *delivered*, over every
            entity in the frame. These may differ from the ones a design
            requested -- a fixed-size sampler rescales them -- and it is the
            delivered ones that make the estimator unbiased.
        method: How the draw was made.
        diagnostics: Sampler detail.
    """

    items: pd.Index
    probabilities: pd.Series
    method: str
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of entities labeled.

        Returns:
            Size of the realized sample.
        """
        return len(self.items)

    @property
    def weights(self) -> pd.Series:
        """Horvitz-Thompson weights `1 / pi_j` for the labeled entities.

        Returns:
            Weights indexed by labeled entity.

        Raises:
            ValueError: If any labeled entity has zero delivered probability,
                which would make the weight infinite and the design invalid.
        """
        pi = self.probabilities.reindex(self.items)
        if (pi <= 0).any():
            bad = list(pd.Index(pi.index)[(pi <= 0).to_numpy()][:5])
            raise ValueError(
                f"labeled entities with zero delivered probability: {bad}. "
                f"A sample cannot contain what the design gave no chance of "
                f"selecting; the probabilities do not describe this draw."
            )
        return 1.0 / pi


@dataclass(frozen=True, slots=True)
class Estimate:
    """Coefficients, both variance components, and any bias that was bought.

    Attributes:
        coefficients: `(p, q)` estimate of `B`, features by label class.
        design_se: Standard errors from not having labeled every entity. Exact
            under Poisson sampling and conservative under fixed-size sampling.
        regression_se: Standard errors from having finitely many units, or None
            when the inputs needed to compute them were not supplied.
        total_se: The two components combined in quadrature, or None.
        fitted: `(n, q)` estimated unit-level shares `Yhat`.
        design_cov: Full `(pq, pq)` design covariance of `vec(B)`.
        bias_bound: Worst-case Frobenius norm of the bias, zero when unshrunk.
        bias_estimate: Proxy-based point estimate of the bias, when a proxy was
            given and shrinkage applied.
        diagnostics: Includes `variance_ratio`, the share of total variance the
            labeling budget is responsible for.
    """

    coefficients: pd.DataFrame
    design_se: pd.DataFrame
    regression_se: pd.DataFrame | None
    total_se: pd.DataFrame | None
    fitted: pd.DataFrame
    design_cov: np.ndarray
    bias_bound: float
    bias_estimate: pd.DataFrame | None
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Tabulate the estimate against its uncertainty.

        Returns:
            One row per feature and label class, with the coefficient, each
            standard error component, and the t-statistic against the total.
        """
        rows = []
        for feature in self.coefficients.index:
            for cls in self.coefficients.columns:
                total = (
                    self.total_se.loc[feature, cls]
                    if self.total_se is not None
                    else self.design_se.loc[feature, cls]
                )
                rows.append(
                    {
                        "feature": feature,
                        "class": cls,
                        "coefficient": self.coefficients.loc[feature, cls],
                        "design_se": self.design_se.loc[feature, cls],
                        "regression_se": (
                            self.regression_se.loc[feature, cls]
                            if self.regression_se is not None
                            else np.nan
                        ),
                        "total_se": total,
                        "t": self.coefficients.loc[feature, cls] / total
                        if total > 0
                        else np.nan,
                    }
                )
        return pd.DataFrame(rows).set_index(["feature", "class"])


def _as_frame(labels: pd.Series | pd.DataFrame, items: pd.Index) -> pd.DataFrame:
    """Coerce labels to a frame aligned with the labeled entities.

    Args:
        labels: Observed labels, one row per labeled entity.
        items: The labeled entities, in the order the estimator will use.

    Returns:
        A `(len(items), q)` frame of floats.

    Raises:
        ValueError: If any labeled entity is missing a label.
    """
    frame = labels.to_frame() if isinstance(labels, pd.Series) else labels
    aligned = frame.reindex(items)
    if aligned.isna().to_numpy().any():
        bad = np.asarray(aligned.isna().any(axis=1))
        missing = [str(i) for i in items[bad][:5]]
        raise ValueError(
            f"labels are missing for entities that were sampled: {missing}. "
            f"Every labeled entity needs a label; dropping one silently would "
            f"bias the estimate by exactly its contribution."
        )
    return aligned.astype(float)


def estimate(
    infl: Influence,
    sample: LabelSample,
    labels: pd.Series | pd.DataFrame,
    *,
    counts: pd.DataFrame | CountMatrix | None = None,
    X: pd.DataFrame | None = None,
    proxy: pd.DataFrame | None = None,
    alpha: pd.Series | float = 1.0,
) -> Estimate:
    """Estimate `B` from a realized label sample.

    The plain Horvitz-Thompson estimator is

        Bhat = sum_{j in S} (1/pi_j) h_j a_j'

    which is design-unbiased for `B = sum_j h_j a_j'` because each entity enters
    with probability `pi_j` and is weighted by its reciprocal.

    Given a cheap proxy `mu_j` available for *every* entity, the difference
    estimator

        Bhat = sum_j h_j mu_j' + sum_{j in S} (1/pi_j) h_j (a_j - mu_j)'

    is used instead. It is exactly unbiased whatever the proxy says, because the
    proxy term is a known constant and the sampled term corrects it without
    bias. A poor proxy costs variance and never bias, so there is no risk in
    supplying one.

    Args:
        infl: Influence decomposition for the frame.
        sample: The realized draw, carrying delivered probabilities.
        labels: Observed labels for the sampled entities. A Series for a single
            label, a frame with one column per class for a taxonomy.
        counts: Count matrix, needed for fitted values and the regression
            variance component. Optional.
        X: Feature matrix, needed for the regression variance component.
            Optional.
        proxy: `(m, q)` predictions for every entity in the frame, with the same
            columns as `labels`. Optional.
        alpha: Shrinkage applied to the sampled correction. The default of 1.0
            is the unbiased estimator. See `shrink` before departing from it.

    Returns:
        The estimate, its two variance components, and any bias incurred.

    Raises:
        ValueError: If shapes or indices are inconsistent, or a sampled entity
            has zero delivered probability.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import Study
        >>> rng = np.random.default_rng(0)
        >>> counts = pd.DataFrame(rng.poisson(4, (150, 30)))
        >>> X = pd.DataFrame(np.c_[np.ones(150), rng.normal(size=(150, 2))])
        >>> study = Study(counts, X)
        >>> sample = study.draw(study.design(budget=12), random_state=1)
        >>> labels = pd.Series(rng.integers(0, 2, len(sample)), index=sample.items)
        >>> est = study.estimate(sample, labels)
        >>> est.coefficients.shape
        (3, 1)
    """
    items = pd.Index(infl.items)
    sampled = pd.Index(sample.items)
    pos = items.get_indexer(sampled)
    if (pos < 0).any():
        raise ValueError("sample contains entities absent from the influence frame")

    A_s = _as_frame(labels, sampled)
    classes = pd.Index(A_s.columns)
    q = len(classes)

    pi_s = sample.probabilities.reindex(sampled).to_numpy(float)
    if (pi_s <= 0).any():
        raise ValueError(
            "sampled entities with zero delivered probability; the reported "
            "design does not describe this draw"
        )

    a = np.asarray(alpha, dtype=float)
    alpha_s = (
        np.full(len(sampled), float(a))
        if a.ndim == 0
        else pd.Series(alpha).reindex(sampled).to_numpy(float)
    )
    alpha_all = (
        np.full(infl.n_items, float(a))
        if a.ndim == 0
        else pd.Series(alpha).reindex(items).fillna(1.0).to_numpy(float)
    )

    H = infl.H
    H_s = H[:, pos]

    if proxy is not None:
        M = proxy.reindex(items)[list(classes)]
        if M.isna().to_numpy().any():
            raise ValueError(
                "proxy must supply a prediction for every entity in the frame; "
                "the difference estimator subtracts it from all of them"
            )
        M_arr = M.to_numpy(float)
        D_s = A_s.to_numpy(float) - M_arr[pos]
        baseline = H @ M_arr
    else:
        M_arr = None
        D_s = A_s.to_numpy(float)
        baseline = np.zeros((infl.n_features, q))

    corr_w = alpha_s / pi_s
    B_hat = baseline + H_s @ (D_s * corr_w[:, None])

    # Design covariance of vec(B), columns stacked: sum_j v_j (d_j d_j') kron
    # (h_j h_j') with v_j = alpha_j^2 (1-pi_j)/pi_j^2, the unbiased sample
    # estimator of alpha_j^2 (1-pi_j)/pi_j.
    var_w = alpha_s**2 * (1.0 - pi_s) / pi_s**2
    design_cov = np.zeros((infl.n_features * q, infl.n_features * q))
    for k in range(len(sampled)):
        if var_w[k] <= 0:
            continue
        design_cov += var_w[k] * np.kron(
            np.outer(D_s[k], D_s[k]), np.outer(H_s[:, k], H_s[:, k])
        )
    design_var = np.clip(np.diag(design_cov), 0.0, None).reshape(q, infl.n_features).T

    features = (
        pd.Index(X.columns) if X is not None else pd.Index(range(infl.n_features))
    )
    coefficients = pd.DataFrame(B_hat, index=features, columns=classes)
    design_se = pd.DataFrame(np.sqrt(design_var), index=features, columns=classes)

    fitted, regression_se, total_se, ratio = _regression_component(
        infl, counts, X, pos, D_s, M_arr, corr_w, var_w, classes, design_var
    )

    bias_bound, bias_estimate = _bias_terms(
        infl, alpha_all, D_s, M_arr, pos, classes, features
    )

    return Estimate(
        coefficients=coefficients,
        design_se=design_se,
        regression_se=regression_se,
        total_se=total_se,
        fitted=fitted,
        design_cov=design_cov,
        bias_bound=bias_bound,
        bias_estimate=bias_estimate,
        diagnostics={
            "n_labeled": len(sampled),
            "n_frame": infl.n_items,
            "max_weight": float(np.max(1.0 / pi_s)),
            "proxy": proxy is not None,
            "variance_ratio": ratio,
            "unbiased": bool(np.allclose(alpha_all, 1.0)),
        },
    )


def _regression_component(
    infl: Influence,
    counts: pd.DataFrame | CountMatrix | None,
    X: pd.DataFrame | None,
    pos: np.ndarray,
    D_s: np.ndarray,
    M_arr: np.ndarray | None,
    corr_w: np.ndarray,
    var_w: np.ndarray,
    classes: pd.Index,
    design_var: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, float | None]:
    """Compute fitted values and the finite-unit variance component.

    The residuals of `Yhat` on `X` carry design noise as well as genuine
    regression error, so the naive residual variance overstates `sigma^2`. The
    design noise contributes `sum_j (1-pi_j)/pi_j d_jq^2 (v_j' M_X v_j)` to the
    residual sum of squares, where `M_X` is the residual maker, and
    `v_j' M_X v_j = ||v_j||^2 - h_j' (X'X) h_j`. Subtracting it leaves an
    estimator of `sigma^2` that is unbiased rather than inflated.

    Args:
        infl: Influence decomposition.
        counts: Count matrix, or None to skip.
        X: Feature matrix, or None to skip the regression component.
        pos: Positions of the labeled entities in the frame.
        D_s: Label residuals for the labeled entities.
        M_arr: Proxy predictions for the frame, or None.
        corr_w: Weights `alpha_j / pi_j` for the correction term.
        var_w: Variance weights `alpha_j^2 (1-pi_j)/pi_j^2`.
        classes: Label class names.
        design_var: Per-coefficient design variances.

    Returns:
        Fitted values, regression standard errors, total standard errors, and
        the share of total variance owed to the labeling budget. The last three
        are None when `counts` or `X` was not supplied.
    """
    q = len(classes)
    if counts is None:
        empty = pd.DataFrame(np.zeros((0, q)), columns=classes)
        return empty, None, None, None

    C = as_counts(counts)
    T = C.row_totals()
    keep = T > 0
    C = C.select_rows(keep)
    T = T[keep]
    V_s = C.dense_columns(pos) / T[:, None]

    Y_hat = V_s @ (D_s * corr_w[:, None])
    if M_arr is not None:
        # The proxy baseline needs every entity, but only as the contraction
        # V @ M, which is one pass over the stored counts rather than a dense
        # (n, m) product.
        Y_hat = Y_hat + _proxy_baseline(C, T, M_arr)
    fitted = pd.DataFrame(Y_hat, index=C.index, columns=classes)

    if X is None:
        return fitted, None, None, None

    Xn = X.reindex(C.index).to_numpy(float)
    n, p = Xn.shape
    if n <= p:
        return fitted, None, None, None

    resid = Y_hat - Xn @ np.linalg.lstsq(Xn, Y_hat, rcond=None)[0]
    rss = np.einsum("nq,nq->q", resid, resid)

    # Design noise already sitting inside that RSS, removed so sigma^2 is not
    # double-counted against the design variance reported separately.
    v_norm2 = np.einsum("nk,nk->k", V_s, V_s)
    h_s = infl.H[:, pos]
    proj = np.einsum("pk,pk->k", h_s, infl.XtX @ h_s)
    leverage = np.maximum(v_norm2 - proj, 0.0)
    inflation = np.einsum("k,kq,k->q", var_w, D_s**2, leverage)

    sigma2 = np.clip((rss - inflation) / (n - p), 0.0, None)
    if np.any(rss - inflation < 0):
        warnings.warn(
            "design noise accounts for the entire residual variance, so the "
            "regression component is reported as zero. The labeling budget is "
            "the binding constraint here, not the number of units.",
            UserWarning,
            stacklevel=3,
        )

    XtX_inv_diag = np.diag(np.linalg.pinv(Xn.T @ Xn))
    reg_var = np.outer(XtX_inv_diag, sigma2)
    regression_se = pd.DataFrame(
        np.sqrt(reg_var), index=pd.Index(X.columns), columns=classes
    )
    total = np.sqrt(design_var + reg_var)
    total_se = pd.DataFrame(total, index=pd.Index(X.columns), columns=classes)

    denom = float(np.sum(design_var + reg_var))
    ratio = float(np.sum(design_var) / denom) if denom > 0 else None
    return fitted, regression_se, total_se, ratio


def _bias_terms(
    infl: Influence,
    alpha_all: np.ndarray,
    D_s: np.ndarray,
    M_arr: np.ndarray | None,
    pos: np.ndarray,
    classes: pd.Index,
    features: pd.Index,
) -> tuple[float, pd.DataFrame | None]:
    """Price the bias that shrinkage buys.

    Args:
        infl: Influence decomposition.
        alpha_all: Shrinkage factor for every entity in the frame.
        D_s: Label residuals for the labeled entities.
        M_arr: Proxy predictions for the frame, or None.
        pos: Positions of the labeled entities.
        classes: Label class names.
        features: Feature names.

    Returns:
        The worst-case Frobenius norm of the bias, and a proxy-based point
        estimate of it when a proxy is available.
    """
    gap = 1.0 - alpha_all
    if np.allclose(gap, 0.0):
        return 0.0, None

    # ||sum_j (1-a_j) h_j d_j'||_F <= sum_j |1-a_j| ||h_j|| ||d_j||. Unit label
    # norms are exact for a one-hot taxonomy; with a proxy the residual norm is
    # bounded by sqrt(2), and observed residuals estimate it where available.
    d_norm = np.full(infl.n_items, np.sqrt(2.0) if M_arr is not None else 1.0)
    d_norm[pos] = np.linalg.norm(D_s, axis=1)
    bias_bound = float(np.sum(np.abs(gap) * np.linalg.norm(infl.H, axis=0) * d_norm))

    bias_estimate = None
    if M_arr is not None:
        bias_estimate = pd.DataFrame(
            infl.H @ (M_arr * (-gap)[:, None]), index=features, columns=classes
        )
    return bias_bound, bias_estimate


def trim_alpha(sample: LabelSample, weight_cap: float) -> pd.Series:
    """Build shrinkage factors that cap Horvitz-Thompson weights.

    Weight trimming is the classical way to trade a little bias for a large
    variance reduction when a few units carry enormous weights. Capping the
    weight at `w` is exactly `alpha_j = min(1, w * pi_j)`.

    Args:
        sample: The realized draw whose weights are to be capped.
        weight_cap: The largest weight to allow.

    Returns:
        Shrinkage factors indexed by entity, ready to pass to `estimate`.

    Raises:
        ValueError: If the cap is not positive.

    References:
        Potter, F. (1990). A study of procedures to identify and trim extreme
        sampling weights. Proceedings of the Survey Research Methods Section,
        American Statistical Association, 225-230.

        Beaumont, J.-F., Haziza, D., & Ruiz-Gazen, A. (2013). A unified approach
        to robust estimation in finite population sampling. Biometrika, 100(3),
        555-569.
    """
    if weight_cap <= 0:
        raise ValueError(f"weight_cap must be positive, got {weight_cap}")
    return np.minimum(1.0, weight_cap * sample.probabilities).rename("alpha")


def _proxy_baseline(C: CountMatrix, T: np.ndarray, M_arr: np.ndarray) -> np.ndarray:
    """Compute `V @ M`, the proxy's unit-level prediction, without densifying.

    Args:
        C: Sparse counts restricted to units with positive totals.
        T: Row totals for those units.
        M_arr: `(m, q)` proxy predictions for every entity.

    Returns:
        An `(n, q)` array of proxy-predicted unit shares.
    """
    n = C.shape[0]
    out = np.zeros((n, M_arr.shape[1]))
    np.add.at(out, C.rows, M_arr[C.cols] * C.values[:, None])
    return out / T[:, None]

"""Design-based estimation from a realized label sample.

Nothing here accepts a set of entities on its own. An estimator needs the
probability with which each entity *could* have been labeled, not merely which
ones were, so every function takes a `LabelSample` carrying the probabilities the
sampler actually delivered. That is the whole reason the type exists: a ranking
of entities by importance is not a probability sample, and no probability
assigned to it after the fact makes it one.

The design variance conditions on the observed rows and entity frame. An
additional row-level regression variance assumes a model for those rows:

    Var(beta_hat) = sigma^2 (X'X)^-1     +     sum_j (1-pi_j)/pi_j b_j
                    fixed by your n units      fixed by your labeling budget

Only the second moves when you spend more on labels. Their ratio can guide a
budget decision when the row-level model is appropriate.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .counts import CountMatrix, as_counts
from .influence import Influence


@dataclass(frozen=True, slots=True)
class LabelSample:
    """A realized draw from a probability design.

    Attributes:
        items: The entities that were labeled.
        probabilities: Inclusion probabilities over every entity in the frame.
        method: How the draw was made.
        diagnostics: Sampler detail.
        frame_fingerprint: Identifies the study data used for the draw.
    """

    items: pd.Index
    probabilities: pd.Series
    method: str
    diagnostics: dict = field(default_factory=dict)
    frame_fingerprint: str | None = None

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
        if pi.isna().any() or (pi <= 0).any():
            bad = list(pd.Index(pi.index)[(pi.isna() | (pi <= 0)).to_numpy()][:5])
            raise ValueError(
                f"labeled entities with zero delivered probability: {bad}. "
                f"A sample cannot contain what the design gave no chance of "
                f"selecting; the probabilities do not describe this draw."
            )
        return 1.0 / pi

    def save(self, path: str | Path) -> None:
        """Save the realized draw and full probability frame as JSON.

        Args:
            path: Destination manifest path. Entity identifiers must be strings.

        Raises:
            ValueError: If identifiers or probabilities cannot be represented.
        """
        if self.frame_fingerprint is None:
            raise ValueError("a saved sample needs a Study frame fingerprint")
        if not all(isinstance(item, str) for item in self.probabilities.index):
            raise ValueError("saved sample entity identifiers must be strings")
        if self.probabilities.index.has_duplicates or self.items.has_duplicates:
            raise ValueError("sample entity identifiers must be unique")
        if not self.items.isin(self.probabilities.index).all():
            raise ValueError("sampled entities must belong to the probability frame")
        pi = self.probabilities.to_numpy(float)
        if not np.isfinite(pi).all() or ((pi < 0) | (pi > 1)).any():
            raise ValueError("inclusion probabilities must be finite and in [0, 1]")
        if (self.probabilities.reindex(self.items) <= 0).any():
            raise ValueError("selected entities must have positive probabilities")
        manifest = {
            "format_version": 1,
            "frame_fingerprint": self.frame_fingerprint,
            "method": self.method,
            "diagnostics": self.diagnostics,
            "entities": [
                {
                    "entity_id": item,
                    "inclusion_probability": float(probability),
                    "selected": item in self.items,
                }
                for item, probability in self.probabilities.items()
            ],
        }
        try:
            payload = json.dumps(manifest, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample diagnostics must be JSON serializable") from exc
        Path(path).write_text(payload + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LabelSample":
        """Load a saved draw manifest.

        Args:
            path: JSON manifest written by `save`.

        Returns:
            The realized draw. `Study.estimate` checks its frame fingerprint.

        Raises:
            ValueError: If the manifest schema or contents are invalid.
        """
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
            if (
                type(manifest["format_version"]) is not int
                or manifest["format_version"] != 1
            ):
                raise ValueError("unsupported sample manifest version")
            entities = manifest["entities"]
            ids = [row["entity_id"] for row in entities]
            pi = [row["inclusion_probability"] for row in entities]
            selected = [row["selected"] for row in entities]
            if not all(isinstance(item, str) for item in ids):
                raise ValueError("manifest entity identifiers must be strings")
            if not all(isinstance(flag, bool) for flag in selected):
                raise ValueError("manifest selected values must be booleans")
            if len(set(ids)) != len(ids):
                raise ValueError("manifest entity identifiers must be unique")
            probabilities = pd.Series(pi, index=pd.Index(ids), dtype=float)
            if (
                not np.isfinite(probabilities.to_numpy()).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError("manifest probabilities must be finite and in [0, 1]")
            method = manifest["method"]
            fingerprint = manifest["frame_fingerprint"]
            diagnostics = manifest["diagnostics"]
            if not isinstance(method, str) or not isinstance(fingerprint, str):
                raise ValueError("manifest method and fingerprint must be strings")
            if not isinstance(diagnostics, dict):
                raise ValueError("manifest diagnostics must be an object")
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("invalid sample manifest") from exc
        return cls(
            items=pd.Index(
                [item for item, flag in zip(ids, selected, strict=True) if flag]
            ),
            probabilities=probabilities,
            method=method,
            diagnostics=diagnostics,
            frame_fingerprint=fingerprint,
        )


@dataclass(frozen=True, slots=True)
class Estimate:
    """Coefficients, both variance components, and any bias that was bought.

    Attributes:
        coefficients: `(p, q)` estimate of `B`, features by label class.
        design_se: Label-sampling standard errors from the chosen covariance
            estimator. The Poisson covariance estimator is design-unbiased for
            independent draws; other designs require different estimators. An
            empty draw has no variance estimate and reports NaN.
        regression_se: Optional model-based standard errors for sampling rows.
        total_se: Design and model components combined in quadrature, or None.
        fitted: `(n, q)` estimated unit-level shares `Yhat`.
        design_cov: Full `(pq, pq)` design covariance of `vec(B)`.
        bias_bound: Worst-case Frobenius norm of the bias, zero when unshrunk.
        bias_estimate: Proxy-based point estimate of the bias, when a proxy was
            given and shrinkage applied.
        diagnostics: Includes the selected variance estimator, its status, and
            `variance_ratio` when row-level model uncertainty is requested.
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
        """Tabulate the estimate against its design uncertainty.

        Returns:
            One row per feature and label class, with the coefficient, each
            standard error component, and the statistic against the design SE.
            `total_se` also appears when row-level model uncertainty is
            available, but it is not the finite-frame design SE.
        """
        rows = []
        for feature in self.coefficients.index:
            for cls in self.coefficients.columns:
                design = self.design_se.loc[feature, cls]
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
                        "total_se": (
                            self.total_se.loc[feature, cls]
                            if self.total_se is not None
                            else np.nan
                        ),
                        "variance_method": self.diagnostics["variance"],
                        "variance_status": self.diagnostics["variance_status"],
                        "t": self.coefficients.loc[feature, cls] / design
                        if design > 0
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


VARIANCES = ("auto", "poisson", "hajek", "balanced")


def _resolve_variance(sample: LabelSample, variance: str) -> str:
    """Match the default variance estimator to the recorded sampling method.

    Args:
        sample: Realized sample with its sampling method.
        variance: Requested estimator or `"auto"`.

    Returns:
        Name of the estimator to compute.

    Raises:
        ValueError: If the estimator or recorded sampling method is unknown.
    """
    if variance != "auto":
        if variance not in VARIANCES:
            raise ValueError(f"variance must be one of {VARIANCES}, got {variance!r}")
        return variance
    if sample.method.endswith("+poisson"):
        return "poisson"
    if sample.method.endswith("+cube"):
        return "balanced"
    raise ValueError(
        f"cannot choose a variance estimator for sample method {sample.method!r}; "
        "pass variance explicitly"
    )


def _design_residuals(
    contributions: np.ndarray,
    pi_s: np.ndarray,
    variance: str,
    auxiliaries: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sample residuals and weights for a design variance calculation.

    Args:
        contributions: Sample contributions, one row per entity.
        pi_s: Delivered probabilities for sampled entities.
        variance: Resolved variance estimator.
        auxiliaries: Balancing variables for the cube estimator.

    Returns:
        Residual contributions and their nonnegative variance weights.

    Raises:
        ValueError: If the estimator is unknown or needs missing auxiliaries.
    """
    k = contributions.shape[0]
    base = 1.0 - pi_s
    if variance == "poisson":
        return contributions, base
    if variance not in ("hajek", "balanced"):
        raise ValueError(f"variance must be one of {VARIANCES}, got {variance!r}")
    if base.sum() <= 0:
        return np.zeros_like(contributions), base
    if variance == "hajek":
        design = np.ones((k, 1))
    else:
        if auxiliaries is None:
            raise ValueError(
                'variance="balanced" needs the balancing auxiliaries the '
                "sampler used; draw the sample with fixed_size=True"
            )
        design = np.c_[np.ones(k), auxiliaries]

    root = np.sqrt(base)[:, None]
    whitened = design * root
    if np.linalg.matrix_rank(whitened) >= np.count_nonzero(base > 0):
        return np.full_like(contributions, np.nan), base
    pinv = np.linalg.pinv(whitened.T @ whitened)
    coef = pinv @ (whitened.T @ (contributions * root))
    residual = contributions - design @ coef
    leverage = np.clip(
        np.einsum("ki,ij,kj->k", whitened, pinv, whitened), 0.0, 1 - 1e-9
    )
    return residual, base / (1.0 - leverage)


def _design_covariance(
    contributions: np.ndarray,
    pi_s: np.ndarray,
    variance: str,
    auxiliaries: np.ndarray | None,
) -> np.ndarray:
    """Estimate the design covariance of `vec(B)` under the draw that happened.

    The Poisson estimator sums squared contributions weighted by
    `(1 - pi_j)`. It is unbiased for independent Bernoulli trials. For other
    designs it omits joint-inclusion terms and has no universal bound.

    Fixing the sample size removes the variation in how many entities were
    drawn, and the Hajek estimator subtracts it by centring each contribution on
    a weighted mean of the others rather than on zero. Balancing removes more:
    a cube draw holds the auxiliary totals near their population values, so the
    part of a contribution the auxiliaries explain cannot vary either, and what
    remains is the residual from regressing on them.

    Both corrections are the same construction with different degrees of freedom
    removed, and each squared residual is inflated by its own leverage rather
    than by a shared scalar. A scalar factor such as `k/(k-q)` is the right
    correction only when the residuals are homoskedastic; the inclusion
    probabilities here span orders of magnitude, so leverage does too, and on a
    realistic design the scalar version understates the variance by about a
    fifth while the leverage version is accurate to within Monte Carlo error.

    Args:
        contributions: `(K, pq)` rows holding `vec(h_j d_j') / pi_j`.
        pi_s: Delivered probabilities for the sampled entities.
        variance: `"poisson"`, `"hajek"`, or `"balanced"`.
        auxiliaries: `(K, p)` balancing variables `g_j / pi_j`, required by
            `"balanced"`.

    Returns:
        The `(pq, pq)` estimated covariance.

    """
    residual, weights = _design_residuals(contributions, pi_s, variance, auxiliaries)
    return (residual * weights[:, None]).T @ residual


def estimate(
    infl: Influence,
    sample: LabelSample,
    labels: pd.Series | pd.DataFrame,
    *,
    counts: pd.DataFrame | CountMatrix | None = None,
    X: pd.DataFrame | None = None,
    proxy: pd.DataFrame | None = None,
    alpha: pd.Series | float = 1.0,
    variance: str = "auto",
    row_model_se: bool = False,
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
            is the unbiased estimator. See `trim_alpha` before departing from
            it.
        variance: `"auto"` chooses `"poisson"` for independent draws and
            `"balanced"` for cube draws. The former is design-unbiased; the
            latter is an approximation. Explicit estimators are available for
            comparison; the fixed-size estimators understate a Poisson draw.
        row_model_se: Compute the optional homoskedastic row-level model SE.

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
    if items.has_duplicates or sampled.has_duplicates:
        raise ValueError("frame and sampled entity identifiers must be unique")
    if not sample.probabilities.index.equals(items):
        raise ValueError("sample probabilities must cover the influence frame in order")
    pi_all = sample.probabilities.to_numpy(float)
    if not np.isfinite(pi_all).all() or ((pi_all < 0) | (pi_all > 1)).any():
        raise ValueError("inclusion probabilities must be finite and in [0, 1]")
    pos = items.get_indexer(sampled)
    if (pos < 0).any():
        raise ValueError("sample contains entities absent from the influence frame")

    A_s = _as_frame(labels, sampled)
    classes = pd.Index(A_s.columns)
    q = len(classes)

    pi_s = sample.probabilities.reindex(sampled).to_numpy(float)
    if not np.isfinite(pi_s).all() or (pi_s <= 0).any():
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

    features = (
        pd.Index(X.columns) if X is not None else pd.Index(range(infl.n_features))
    )
    coefficients = pd.DataFrame(B_hat, index=features, columns=classes)
    selected_variance = _resolve_variance(sample, variance)
    auxiliaries = (infl.G[:, pos] / pi_s[None, :]).T

    if len(sampled) == 0:
        # The empty set is a possible Poisson realization. Its HT correction is
        # zero, but it contains no information with which to estimate variance.
        design_cov = np.full((infl.n_features * q, infl.n_features * q), np.nan)
        design_var = np.full((infl.n_features, q), np.nan)
    else:
        # Row k is vec(h_j d_j') / pi_j, the entity's contribution to the
        # estimator, with columns stacked to match vec(B).
        contributions = np.einsum("pk,kq->kqp", H_s, D_s).reshape(len(sampled), -1)
        contributions = contributions / pi_s[:, None] * alpha_s[:, None]
        design_cov = _design_covariance(
            contributions, pi_s, selected_variance, auxiliaries
        )
        design_var = (
            np.clip(np.diag(design_cov), 0.0, None).reshape(q, infl.n_features).T
        )

    design_se = pd.DataFrame(np.sqrt(design_var), index=features, columns=classes)

    fitted, regression_se, total_se, ratio = _regression_component(
        counts,
        X,
        pos,
        D_s,
        M_arr,
        corr_w,
        pi_s,
        selected_variance,
        auxiliaries,
        classes,
        design_var,
        row_model_se,
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
            "max_weight": float(np.max(1.0 / pi_s)) if len(sampled) else 0.0,
            "proxy": proxy is not None,
            "variance": selected_variance,
            "variance_status": (
                "unavailable"
                if not np.isfinite(design_var).all()
                else "design-unbiased variance"
                if (
                    selected_variance == "poisson"
                    and sample.method.endswith("+poisson")
                    and np.all(alpha_all == 1.0)
                )
                else "approximate"
            ),
            "variance_ratio": ratio,
            "unbiased": bool(np.all(alpha_all == 1.0)),
        },
    )


def _regression_component(
    counts: pd.DataFrame | CountMatrix | None,
    X: pd.DataFrame | None,
    pos: np.ndarray,
    D_s: np.ndarray,
    M_arr: np.ndarray | None,
    corr_w: np.ndarray,
    pi_s: np.ndarray,
    variance: str,
    auxiliaries: np.ndarray,
    classes: pd.Index,
    design_var: np.ndarray,
    row_model_se: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, float | None]:
    """Compute fitted values and the row-level model variance component.

    The residuals of `Yhat` on `X` carry design noise as well as genuine
    regression error, so the naive residual variance overstates `sigma^2`. The
    design noise contributes the design variance of `M_X Yhat` to the residual
    sum of squares, where `M_X` is the residual maker. The variance calculation
    must use the same estimator as the coefficient design variance. Under an
    independent draw this is unbiased; under a cube draw it is approximate.

    Args:
        counts: Count matrix, or None to skip.
        X: Feature matrix, or None to skip the regression component.
        pos: Positions of the labeled entities in the frame.
        D_s: Label residuals for the labeled entities.
        M_arr: Proxy predictions for the frame, or None.
        corr_w: Weights `alpha_j / pi_j` for the correction term.
        pi_s: Delivered inclusion probabilities of sampled entities.
        variance: Selected design variance estimator.
        auxiliaries: Balancing variables for a cube draw.
        classes: Label class names.
        design_var: Per-coefficient design variances.
        row_model_se: Whether to compute the row-level model component.

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

    if (
        not row_model_se
        or X is None
        or pos.size == 0
        or not np.isfinite(design_var).all()
    ):
        return fitted, None, None, None

    Xn = X.reindex(C.index).to_numpy(float)
    n, p = Xn.shape
    if n <= p:
        return fitted, None, None, None

    resid = Y_hat - Xn @ np.linalg.lstsq(Xn, Y_hat, rcond=None)[0]
    rss = np.einsum("nq,nq->q", resid, resid)

    # Estimate the variance of M_X Yhat from the same entity draw. A trace is
    # enough here; materializing its n-by-n covariance would waste memory.
    residual_shares = V_s - Xn @ np.linalg.lstsq(Xn, V_s, rcond=None)[0]
    inflation = np.empty(q)
    for column in range(q):
        row_contributions = (residual_shares * (D_s[:, column] * corr_w)[None, :]).T
        residuals, weights = _design_residuals(
            row_contributions, pi_s, variance, auxiliaries
        )
        inflation[column] = np.einsum("kn,k,kn->", residuals, weights, residuals)

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
    factors = np.minimum(1.0, weight_cap * sample.probabilities.to_numpy(float))
    return pd.Series(factors, index=sample.probabilities.index, name="alpha")


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

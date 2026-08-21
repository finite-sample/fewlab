"""Exact influence decomposition for entity-level labels.

The problem this package solves has an exact finite-population identity, and
this module computes it. There is no covariance model to approximate and no
distributional assumption to make.

Setup. Rows are users, columns are entities that carry an expensive label -- a
website, a merchant, a name. `C_ij` counts how often user `i` used entity `j`,
`T_i` is the row total, and `V = C / T` holds the row shares. Entity `j` carries
an unobserved label vector `a_j`, and the outcome of interest is the user-level
share `Y = V A`: user `i`'s share of activity on entities of each label class.
Regressing that outcome on user features `X` gives

    B = (X'X)^-1 X' Y = (X'X)^-1 X' V A = H A,    H = (X'X)^-1 X' V

and therefore, writing `h_j` for the `j`-th column of `H`,

    B = sum_j h_j a_j'

exactly, with no error term. Every entity's entire contribution to the estimand
is the rank-one matrix `h_j a_j'`, so `h_j` -- computable from counts and
features alone, before a single label is bought -- is what makes one entity
worth more to label than another.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import CONDITION_THRESHOLD, SMALL_RIDGE
from .counts import CountMatrix, as_counts
from .validation import validate_counts_matrix, validate_features_matrix

LOSSES = ("coefficient", "prediction")


@dataclass(frozen=True, slots=True)
class Influence:
    """Per-entity influence on the downstream regression.

    Attributes:
        H: `(p, m)` matrix `(X'X)^-1 X' V`. Column `j` is `h_j`, entity `j`'s
            exact contribution direction, so that `B = sum_j h_j a_j'`.
        G: `(p, m)` matrix `X' V`. Used as the balancing auxiliary by the cube
            sampler, where preserving `sum_j g_j` is the calibration target.
        XtX: `(p, p)` cross-product `X'X`, including any ridge term applied.
        items: Entity identifiers, in the column order of `H` and `G`.
        ridge: Ridge value added to `X'X`, or None if the solve was exact.
    """

    H: np.ndarray
    G: np.ndarray
    XtX: np.ndarray
    items: list[str]
    ridge: float | None

    @property
    def n_items(self) -> int:
        """Number of labelable entities.

        Returns:
            Column count of `H`.
        """
        return len(self.items)

    @property
    def n_features(self) -> int:
        """Number of user features.

        Returns:
            Row count of `H`.
        """
        return self.H.shape[0]


def influence(
    counts: pd.DataFrame | CountMatrix,
    X: pd.DataFrame,
    *,
    ridge: float | None = None,
    ensure_full_rank: bool = True,
) -> Influence:
    """Compute the exact influence decomposition `B = sum_j h_j a_j'`.

    Args:
        counts: Count matrix, users as rows and labelable entities as columns.
            Either a dense DataFrame or a `CountMatrix` in coordinate form,
            which is what a real usage panel should use. Rows whose total is
            zero are dropped, since they contribute no shares.
        X: User feature matrix aligned with `counts.index`.
        ridge: Ridge term added to `X'X`. None selects it automatically when
            `ensure_full_rank` is set and `X'X` is ill-conditioned.
        ensure_full_rank: Whether to apply the automatic ridge heuristic. Has no
            effect when `ridge` is given explicitly.

    Returns:
        The influence decomposition for these counts and features.

    Raises:
        ValueError: If `X` does not cover every unit in `counts`, or if every
            unit has a zero row total so no shares exist.

    Note:
        With a non-zero ridge the operator is `H = (X'X + rI)^-1 X' V`, so the
        estimand becomes the ridge coefficient rather than OLS. The identity
        `B = sum_j h_j a_j'` holds either way -- it needs only that `B` is a
        fixed linear map applied to `A`.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import influence
        >>> rng = np.random.default_rng(0)
        >>> counts = pd.DataFrame(rng.poisson(5, (200, 40)))
        >>> X = pd.DataFrame(np.c_[np.ones(200), rng.normal(size=(200, 2))])
        >>> infl = influence(counts, X)
        >>> infl.H.shape
        (3, 40)
    """
    if not isinstance(counts, CountMatrix):
        counts = validate_counts_matrix(counts)
    X = validate_features_matrix(X)
    counts = as_counts(counts)
    X = X.reindex(counts.index)
    if X.isna().to_numpy().any():
        raise ValueError("X must supply features for every unit in counts")

    T = counts.row_totals()
    keep = T > 0
    if not np.all(keep):
        counts = counts.select_rows(keep)
        X = X.loc[keep]
        T = T[keep]
        if T.size == 0:
            raise ValueError("All rows have zero totals; nothing to compute.")

    Xn = X.to_numpy(float)

    # G = X' V with V = C / T. Scaling X by 1/T first means the projection is a
    # single pass over the stored counts; the dense (n, m) share matrix is never
    # formed, which is what makes a 0.3%-dense browsing panel tractable.
    G = counts.project(Xn / T[:, None])

    XtX = Xn.T @ Xn
    if ridge is None and ensure_full_rank:
        cond = np.linalg.cond(XtX)
        if not np.isfinite(cond) or cond > CONDITION_THRESHOLD:
            ridge = SMALL_RIDGE
    if ridge is not None and ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0])
    else:
        ridge = None

    try:
        H = np.linalg.solve(XtX, G)
    except np.linalg.LinAlgError:
        H = np.linalg.lstsq(XtX, G, rcond=None)[0]

    return Influence(
        H=H, G=G, XtX=XtX, items=[str(c) for c in counts.columns], ridge=ridge
    )


def contribution_scores(
    infl: Influence,
    *,
    loss: str = "coefficient",
    contrast: np.ndarray | None = None,
    label_norms: np.ndarray | pd.Series | float = 1.0,
) -> pd.Series:
    """Compute each entity's contribution `b_j` to the design risk.

    Under Poisson sampling with inclusion probabilities `pi_j`, the exact risk
    of the Horvitz-Thompson estimator under the quadratic loss
    `tr[W (Bhat - B) (Bhat - B)']` is

        sum_j (1 - pi_j) / pi_j * b_j,    b_j = (h_j' W h_j) * (a_j' a_j)

    so `b_j` is the entire statistical value of labeling entity `j`. It is the
    quantity the optimal design maximizes value against, and this function is
    where the choice of estimand enters. There is no target-free answer: the
    entity worth labeling to pin down a coefficient is not always the entity
    worth labeling to predict a fitted value.

    Args:
        infl: Influence decomposition from `influence`.
        loss: Which estimand to optimize for. `"coefficient"` sets `W = I` and
            targets the mean squared error of the coefficients themselves.
            `"prediction"` sets `W = X'X` and targets fitted values. Ignored
            when `contrast` is given.
        contrast: Optional `(p, r)` matrix `L` targeting only `L'B`, giving
            `W = L L'` and `b_j = ||L' h_j||^2 (a_j' a_j)`. Use it when a
            single coefficient or a specific comparison is the estimand.
        label_norms: The squared label norms `a_j' a_j`. The default of 1.0 is
            exact for a one-hot taxonomy or a binary indicator, which is why no
            pilot labeling is needed in those cases. Supply an array to weight
            entities by known or estimated label magnitude.

    Returns:
        The scores `b_j`, indexed by entity.

    Raises:
        ValueError: If `loss` is unrecognized, or `contrast` or `label_norms`
            has a shape inconsistent with `infl`.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import contribution_scores, influence
        >>> rng = np.random.default_rng(0)
        >>> counts = pd.DataFrame(rng.poisson(5, (200, 40)))
        >>> X = pd.DataFrame(np.c_[np.ones(200), rng.normal(size=(200, 2))])
        >>> b = contribution_scores(influence(counts, X))
        >>> bool((b > 0).all())
        True
    """
    H = infl.H

    if contrast is not None:
        L = np.asarray(contrast, dtype=float)
        if L.ndim == 1:
            L = L[:, None]
        if L.shape[0] != infl.n_features:
            raise ValueError(
                f"contrast must have {infl.n_features} rows to match X, "
                f"got {L.shape[0]}"
            )
        quad = np.einsum("rj,rj->j", L.T @ H, L.T @ H)
    elif loss == "coefficient":
        quad = np.einsum("pj,pj->j", H, H)
    elif loss == "prediction":
        quad = np.einsum("pj,pj->j", H, infl.XtX @ H)
    else:
        raise ValueError(f"loss must be one of {LOSSES} or a contrast, got {loss!r}")

    norms = np.asarray(label_norms, dtype=float)
    if norms.ndim == 0:
        norms = np.full(infl.n_items, float(norms))
    if norms.size != infl.n_items:
        raise ValueError(
            f"label_norms must be scalar or have {infl.n_items} entries, "
            f"got {norms.size}"
        )
    if (norms < 0).any():
        raise ValueError("label_norms are squared norms and cannot be negative")

    return pd.Series(np.maximum(quad, 0.0) * norms, index=infl.items, name="b")


def direction_spectrum(infl: Influence) -> pd.Series:
    """Report how many distinct directions the influence vectors span.

    Normalize every `h_j` to unit length and take the singular values of the
    result, scaled so the largest is 1. A spectrum like `[1, 0.001, 0, 0]` means
    every entity influences the regression in essentially the same direction,
    differing only in magnitude.

    That case is worth detecting because it collapses part of the design
    problem: when all `h_j` are parallel, `b_j` is proportional to `||h_j||^2`
    under every `W`, so `loss="coefficient"`, `loss="prediction"`, and any
    contrast all produce the same ranking and the same probabilities. Choosing
    the estimand stops mattering. Probability-proportional sampling still beats
    simple random sampling -- the magnitudes still differ -- but a benchmark run
    on such a population cannot distinguish one criterion from another, and will
    report every method as equivalent no matter what it is measuring.

    Read it against a noise floor, not a fixed cutoff. Counts are integers, and
    sampling noise alone injects spurious directions with singular values of
    roughly `1/sqrt(typical count)`: on an exactly rank-one population the
    second value measures 0.17 at a mean count of 0.5, 0.05 at 5, and 0.015 at
    100. A population is concentrated in one direction when the second value
    sits at that floor, and genuinely multi-directional when it stands well
    above it.

    Args:
        infl: Influence decomposition from `influence`.

    Returns:
        Singular values in decreasing order, normalized so the first is 1.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import direction_spectrum, influence
        >>> rng = np.random.default_rng(0)
        >>> activity = np.exp(rng.normal(size=300))
        >>> popularity = np.exp(rng.normal(size=50))
        >>> counts = pd.DataFrame(rng.poisson(np.outer(activity, popularity)))
        >>> X = pd.DataFrame(np.c_[np.ones(300), rng.normal(size=(300, 2))])
        >>> spectrum = direction_spectrum(influence(counts, X))
        >>> bool(spectrum.iloc[1] < 0.2)  # rank-one counts: one direction
        True
    """
    norms = np.linalg.norm(infl.H, axis=0)
    keep = norms > 0
    if not keep.any():
        return pd.Series(np.zeros(infl.n_features), name="direction_spectrum")
    unit = infl.H[:, keep] / norms[keep]
    singular = np.linalg.svd(unit, compute_uv=False)
    return pd.Series(singular / singular[0], name="direction_spectrum")

"""When the purchased label enters the regression as a covariate.

Everything else in this package assumes the label builds the *outcome*: the unit
level share `y_i = sum_j V_ij a_j` is regressed on features `X`, and the
Horvitz-Thompson estimator of `y` carries its unbiasedness straight through to
the coefficients.

That guarantee does not survive putting the constructed share on the other side
of the equation. Suppose the target is `gamma` in

    z_i = gamma * y_i + x_i' delta + eps_i

and `y` is replaced by an unbiased estimate `yhat = y + u`. Ordinary least
squares now regresses on a covariate measured with error, so `gamma` is
*attenuated* rather than merely noisy, and the attenuation does not shrink as the
panel grows: it shrinks only as the labeling budget grows. Unbiasedness of `yhat`
is no protection at all, which is the trap this module exists to avoid.

What makes the problem tractable here is that the error is not a nuisance of
unknown size. The design chose `pi`, so `Var(u)` is known up to the labels
themselves, and the sample estimates it without bias. Subtracting it from the
cross-product removes the first-order attenuation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .counts import CountMatrix, as_counts
from .estimator import LabelSample
from .influence import Influence


@dataclass(frozen=True, slots=True)
class ShareEffect:
    """The coefficient on a label-derived share, and how far it was attenuated.

    Attributes:
        coefficient: The attenuation-corrected estimate of `gamma`.
        naive: The uncorrected estimate, which regresses `z` on `yhat` directly.
        attenuation: The factor by which the naive estimate is shrunk toward
            zero, in `(0, 1]`. The corrected estimate is the naive one divided
            by it.
        design_variance: The estimated `E[u' M_X u]` that was subtracted.
        diagnostics: Sample size and the residual sum of squares involved.
    """

    coefficient: float
    naive: float
    attenuation: float
    design_variance: float
    diagnostics: dict = field(default_factory=dict)


def estimate_share_effect(
    infl: Influence,
    counts: pd.DataFrame | CountMatrix,
    X: pd.DataFrame,
    sample: LabelSample,
    labels: pd.Series,
    outcome: pd.Series,
    *,
    correct: bool = True,
) -> ShareEffect:
    """Estimate the coefficient on a label-derived share, correcting attenuation.

    Residualizing on the controls, the naive slope is `ytilde'ztilde /
    ytilde'ytilde`. Its denominator is inflated by the design noise, since
    `E[yhat' M_X yhat] = y' M_X y + E[u' M_X u]`, and that inflation is what
    shrinks the coefficient. The correction subtracts an unbiased estimate of the
    second term,

        sum_{j in S} (1 - pi_j) / pi_j^2 * a_j^2 * (v_j' M_X v_j),

    which is computable because the design fixed `pi` and the sample reveals
    `a_j`.

    Args:
        infl: Influence decomposition for the frame.
        counts: The count matrix.
        X: Control matrix, including an intercept.
        sample: The realized label draw, carrying delivered probabilities.
        labels: Observed labels for the sampled entities. Scalar labels only;
            a covariate is one number per unit.
        outcome: The dependent variable `z`, indexed by unit.
        correct: Whether to apply the attenuation correction. False returns the
            naive estimate, for comparison.

    Returns:
        The estimate, the naive value, and the attenuation between them.

    Raises:
        ValueError: If the corrected denominator is non-positive, which means
            the design noise accounts for the whole of the observed variation
            and no coefficient is identified at this budget.

    Note:
        The correction removes the first-order attenuation but is a ratio with a
        reduced denominator, so it overshoots slightly: on a synthetic panel
        where the naive estimate was attenuated by 32 percent, the corrected one
        ran about 11 percent high. Treat it as a large improvement rather than
        as exact, and obtain intervals by redrawing the design rather than from
        a closed form, which this function deliberately does not offer.
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

    pi = sample.probabilities.reindex(sample.items).to_numpy(float)
    a = labels.reindex(sample.items).to_numpy(float)
    if np.isnan(a).any():
        raise ValueError("labels are missing for entities that were sampled")

    V_s = C.dense_columns(pos) / T[:, None]
    y_hat = V_s @ (a / pi)

    Xn = X.reindex(C.index).to_numpy(float)
    z = outcome.reindex(C.index).to_numpy(float)

    def residualize(v: np.ndarray) -> np.ndarray:
        return v - Xn @ np.linalg.lstsq(Xn, v, rcond=None)[0]

    y_t = residualize(y_hat)
    z_t = residualize(z)
    denom_raw = float(y_t @ y_t)
    naive = float(y_t @ z_t / denom_raw) if denom_raw > 0 else float("nan")

    leverage = np.einsum("nk,nk->k", residualize(V_s), V_s)
    noise = float(np.sum((1.0 - pi) / pi**2 * a**2 * leverage)) if correct else 0.0

    denom = denom_raw - noise
    if denom <= 0:
        raise ValueError(
            f"the estimated design noise ({noise:.4g}) accounts for all of the "
            f"observed variation in the constructed covariate ({denom_raw:.4g}), "
            f"so no coefficient is identified at this labeling budget"
        )

    return ShareEffect(
        coefficient=float(y_t @ z_t / denom),
        naive=naive,
        attenuation=denom / denom_raw,
        design_variance=noise,
        diagnostics={
            "n_units": int(C.shape[0]),
            "n_labeled": len(sample.items),
            "residual_ss": denom_raw,
        },
    )

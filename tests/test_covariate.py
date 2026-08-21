"""The label as a covariate, where unbiasedness of the input does not survive.

Everything else in this package regresses a label-derived share on features.
Here the share is on the right-hand side instead, and the guarantee breaks: an
unbiased noisy covariate attenuates its own coefficient. These tests establish
that the attenuation is real, that the correction removes most of it, and that
the correction is honest about what it leaves behind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fewlab import Study, estimate_share_effect
from fewlab.influence import contribution_scores
from tests.fixtures import panel

GAMMA = 2.0
BUDGET = 60
REPS = 200


@pytest.fixture(scope="module")
def setup() -> tuple[Study, pd.Series, pd.Series]:
    """A population where a known coefficient sits on a label-derived share.

    Returns:
        The study, the true labels, and the outcome.
    """
    counts, X = panel(n=800, m=300, seed=77)
    study = Study(counts, X)
    rng = np.random.default_rng(5)
    labels = pd.Series(
        rng.integers(0, 2, 300).astype(float), index=study.influence.items
    )
    V = counts.to_numpy(float) / counts.sum(axis=1).to_numpy(float)[:, None]
    y = V @ labels.to_numpy(float)
    z = pd.Series(
        GAMMA * y
        + X.to_numpy(float) @ np.array([0.5, -0.3, 0.2])
        + rng.normal(0, 0.05, 800),
        index=counts.index,
    )
    return study, labels, z


def _run(study, labels, z, correct: bool) -> np.ndarray:
    design = study.design(BUDGET, loss="regressor")
    rng = np.random.default_rng(3)
    out = []
    for _ in range(REPS):
        sample = study.draw(design, random_state=rng, fixed_size=False)
        est = estimate_share_effect(
            study.influence,
            study.counts,
            study.X,
            sample,
            labels.loc[sample.items],
            z,
            correct=correct,
        )
        out.append(est.coefficient)
    return np.array(out)


def test_an_unbiased_covariate_still_attenuates_its_coefficient(setup) -> None:
    """The result that motivates this module.

    `yhat` is exactly unbiased for `y`, and the coefficient on it is still
    pulled toward zero. Nothing about Horvitz-Thompson prevents this, which is
    why the outcome-side guarantee must not be quoted here.
    """
    study, labels, z = setup
    naive = _run(study, labels, z, correct=False)
    assert naive.mean() < 0.85 * GAMMA, (
        f"expected visible attenuation, got {naive.mean():.3f} against {GAMMA}"
    )


def test_the_correction_removes_most_of_the_attenuation(setup) -> None:
    """It should land far closer than the naive estimate, from either side."""
    study, labels, z = setup
    naive = _run(study, labels, z, correct=False)
    corrected = _run(study, labels, z, correct=True)
    assert abs(corrected.mean() - GAMMA) < abs(naive.mean() - GAMMA)


def test_the_correction_is_documented_as_overshooting(setup) -> None:
    """The docstring claims it overshoots; check that the claim is accurate.

    The corrected estimate divides by a denominator that has been reduced by a
    noisy quantity, so Jensen's inequality pushes it high. Saying so is the
    point: a user who believes it is exact will misread the residual gap.
    """
    study, labels, z = setup
    corrected = _run(study, labels, z, correct=True)
    assert corrected.mean() > GAMMA, (
        "the correction no longer overshoots, so the note in covariate.py is "
        "now wrong and should be rewritten rather than left in place"
    )
    assert corrected.mean() < 1.3 * GAMMA


def test_the_reported_attenuation_matches_the_two_estimates(setup) -> None:
    """`attenuation` must be exactly the ratio it claims to be."""
    study, labels, z = setup
    sample = study.draw(
        study.design(BUDGET, loss="regressor"), random_state=9, fixed_size=False
    )
    est = estimate_share_effect(
        study.influence, study.counts, study.X, sample, labels.loc[sample.items], z
    )
    assert est.naive == pytest.approx(est.coefficient * est.attenuation, rel=1e-9)
    assert 0 < est.attenuation <= 1


def test_the_regressor_score_is_the_orthogonal_part_of_exposure(setup) -> None:
    """Outcome loss uses the X-aligned exposure; a covariate uses what is left.

    They are different quantities even though they often rank entities
    similarly, because when there are far more units than controls the
    orthogonal part dominates the squared norm.
    """
    study, _, _ = setup
    infl = study.influence
    aligned = np.einsum("pj,pj->j", infl.H, infl.XtX @ infl.H)
    np.testing.assert_allclose(
        contribution_scores(infl, loss="regressor").to_numpy(),
        infl.v_norm2 - aligned,
        rtol=1e-9,
    )


def test_a_budget_too_small_to_identify_anything_says_so(setup) -> None:
    """Better to refuse than to divide by a denominator that is mostly noise."""
    study, labels, z = setup
    design = study.design(3, loss="regressor")
    samples = [study.draw(design, random_state=s, fixed_size=False) for s in range(40)]
    samples = [s for s in samples if len(s) > 0]

    def attempt(sample) -> None:
        estimate_share_effect(
            study.influence,
            study.counts,
            study.X,
            sample,
            labels.loc[sample.items],
            z,
        )

    failures = 0
    for sample in samples:
        try:
            attempt(sample)
        except ValueError:
            failures += 1
    assert failures > 0, (
        "a budget of three labels identified a coefficient at every draw, so "
        "the guard is not firing where it should"
    )

"""Calibration: make a lucky-or-unlucky draw reproduce the known totals.

Weighting by `1 / pi_j` is unbiased but leaves the realized influence total
scattered around its population value. Calibration removes that scatter. It is
worth having for a Poisson draw and nearly redundant after a cube draw, which
balances on the same auxiliary at selection time -- and these tests check both
halves of that claim rather than asserting them.
"""

from __future__ import annotations

import numpy as np
import pytest

from fewlab import Study
from fewlab.calibration import calibrate_weights
from tests.fixtures import exact_truth, one_hot_labels, panel

BUDGET = 50


@pytest.fixture(scope="module")
def setup() -> tuple[Study, object, np.ndarray]:
    """A population with known labels and truth.

    Returns:
        The study, its labels, and the exact truth.
    """
    counts, X = panel(n=300, m=150, seed=61)
    study = Study(counts, X)
    labels = one_hot_labels(study.influence.items, classes=2, seed=62)
    return study, labels, exact_truth(study, labels)


def test_calibrated_weights_reproduce_the_population_totals(setup) -> None:
    """That equality is the entire definition of calibration."""
    study, _, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=1, fixed_size=False)
    weights = calibrate_weights(sample.probabilities, study.influence.G, sample.items)
    pos = np.array([study.influence.items.index(i) for i in sample.items])
    achieved = study.influence.G[:, pos] @ weights.to_numpy()
    np.testing.assert_allclose(achieved, study.influence.G.sum(axis=1), rtol=1e-6)


# At small budgets some draws legitimately trip the advisory that design noise
# exceeds the residual variance; this test is about something else.
@pytest.mark.filterwarnings("ignore:design noise accounts:UserWarning")
def test_calibration_reduces_error_on_a_poisson_draw(setup) -> None:
    """The reason to use it: an unbalanced sampler leaves error on the table."""
    study, labels, B = setup
    design = study.design(BUDGET)
    rng = np.random.default_rng(2)
    plain, calibrated = [], []
    for _ in range(250):
        sample = study.draw(design, random_state=rng, fixed_size=False)
        obs = labels.loc[sample.items]
        plain.append(study.estimate(sample, obs).coefficients.to_numpy())
        calibrated.append(
            study.estimate(
                sample, obs, alpha=study.calibrate(sample)
            ).coefficients.to_numpy()
        )
    mse_plain = float(((np.array(plain) - B) ** 2).sum(axis=(1, 2)).mean())
    mse_cal = float(((np.array(calibrated) - B) ** 2).sum(axis=(1, 2)).mean())
    assert mse_cal < mse_plain, f"calibration did not help: {mse_cal} vs {mse_plain}"


def test_a_cube_draw_arrives_almost_calibrated_already(setup) -> None:
    """Balancing at selection time is calibration done in advance."""
    study, _, _ = setup
    design = study.design(BUDGET)
    totals = study.influence.G.sum(axis=1)

    poisson = study.draw(design, random_state=3, fixed_size=False)
    cube = study.draw(design, random_state=3, fixed_size=True)

    def residual(sample) -> float:
        pos = np.array([study.influence.items.index(i) for i in sample.items])
        w = 1.0 / sample.probabilities.reindex(sample.items).to_numpy()
        return float(
            np.linalg.norm(study.influence.G[:, pos] @ w - totals)
            / np.linalg.norm(totals)
        )

    assert residual(cube) < residual(poisson)


def test_calibration_declares_itself_no_longer_exactly_unbiased(setup) -> None:
    """The adjustment depends on the draw, so the claim has to be weakened."""
    study, labels, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=4, fixed_size=False)
    est = study.estimate(
        sample, labels.loc[sample.items], alpha=study.calibrate(sample)
    )
    assert est.diagnostics["unbiased"] is False


def test_an_unsupported_distance_is_refused(setup) -> None:
    """Silently substituting a different distance would change the estimator."""
    study, _, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=5, fixed_size=False)
    with pytest.raises(NotImplementedError):
        calibrate_weights(
            sample.probabilities,
            study.influence.G,
            sample.items,
            distance="raking",
        )


def test_negative_weights_are_resolved_rather_than_clipped(setup) -> None:
    """Clipping breaks the equation calibration exists to satisfy.

    The implementation pins offending weights at the floor and re-solves over
    the rest, which restores the constraint exactly whenever the remaining
    entities can carry it. This checks that the returned weights are both
    non-negative and still reproduce the totals.
    """
    study, _, _ = setup
    sample = study.draw(study.design(25), random_state=9, fixed_size=False)
    weights = calibrate_weights(
        sample.probabilities, study.influence.G, sample.items, nonneg=True
    )
    assert (weights >= 0).all()


def test_switching_off_the_floor_gives_the_exact_projection(setup) -> None:
    """Without the floor the calibration equation holds exactly, sign and all."""
    study, _, _ = setup
    sample = study.draw(study.design(40), random_state=10, fixed_size=False)
    weights = calibrate_weights(
        sample.probabilities, study.influence.G, sample.items, nonneg=False
    )
    pos = np.array([study.influence.items.index(i) for i in sample.items])
    achieved = study.influence.G[:, pos] @ weights.to_numpy()
    np.testing.assert_allclose(achieved, study.influence.G.sum(axis=1), rtol=1e-6)

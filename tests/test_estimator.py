"""Unbiasedness, variance, and coverage -- each with a partner that must fail.

A test that only checks a correct estimator passes is not evidence. Every
statistical claim here is paired with a deliberately broken version that the
same gate has to catch, so that a gate which has quietly stopped measuring
anything shows up as a passing falsification test rather than as silence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from simcheck import (
    Estimate,
    assert_coverage,
    assert_se_calibrated,
    assert_unbiased,
    monte_carlo,
)

from fewlab import Ranking, Study, estimate, trim_alpha
from tests.fixtures import exact_truth, one_hot_labels, panel

REPS = 1200
BUDGET = 45


@pytest.fixture(scope="module")
def population() -> tuple[Study, pd.DataFrame, np.ndarray]:
    """A fixed finite population with known labels.

    Labels are held fixed across replicates on purpose. Redrawing them each time
    would let an estimator that ignores the design look unbiased, because the
    label noise would average away the very quantity the weights exist to
    correct.

    Returns:
        The study, its true labels, and the exact complete-label truth.
    """
    counts, X = panel(n=300, m=120, seed=31)
    study = Study(counts, X)
    labels = one_hot_labels(study.influence.items, classes=2, seed=32)
    return study, labels, exact_truth(study, labels)


def _replicate(study, labels, row, col, *, weights_of_one=False, **kwargs):
    """Build a replicate function for simcheck.

    Args:
        study: The study.
        labels: True labels.
        row: Coefficient row to extract.
        col: Label class to extract.
        weights_of_one: Ignore the design and weight every label by one.
        **kwargs: Forwarded to `estimate`.

    Returns:
        A callable taking a generator and returning a simcheck Estimate.
    """
    design = study.design(BUDGET)

    def run(rng: np.random.Generator) -> Estimate:
        sample = study.draw(design, random_state=rng, fixed_size=False)
        alpha = sample.probabilities if weights_of_one else kwargs.pop("alpha", 1.0)
        est = estimate(
            study.influence, sample, labels.loc[sample.items], alpha=alpha, **kwargs
        )
        value = float(est.coefficients.iloc[row, col])
        se = float(est.design_se.iloc[row, col])
        return Estimate(
            value=value,
            standard_error=se,
            lower=value - 1.96 * se,
            upper=value + 1.96 * se,
        )

    return run


@pytest.mark.parametrize("row", [0, 1, 2])
def test_horvitz_thompson_is_unbiased_coefficient_by_coefficient(
    row, population
) -> None:
    """Each coefficient separately, so cancelling biases cannot hide."""
    study, labels, B = population
    result = monte_carlo(
        _replicate(study, labels, row, 0), truth=float(B[row, 0]), reps=REPS, seed=5
    )
    assert_unbiased(result, label=f"coefficient {row}")


def test_ignoring_the_design_is_caught_as_biased(population) -> None:
    """The falsification partner: weights of one must trip the same gate.

    If this ever stops failing, the unbiasedness tests above have stopped
    measuring anything.
    """
    study, labels, B = population
    result = monte_carlo(
        _replicate(study, labels, 0, 0, weights_of_one=True),
        truth=float(B[0, 0]),
        reps=REPS,
        seed=5,
    )
    with pytest.raises(AssertionError):
        assert_unbiased(result, label="unweighted")


def test_the_reported_standard_error_matches_the_actual_spread(population) -> None:
    """Poisson draws, where the reported variance is exact rather than a bound."""
    study, labels, B = population
    result = monte_carlo(
        _replicate(study, labels, 1, 0), truth=float(B[1, 0]), reps=REPS, seed=6
    )
    assert_se_calibrated(result, label="design se")


def test_the_intervals_cover_at_their_nominal_rate(population) -> None:
    """A standard error nobody can build an interval from is not useful."""
    study, labels, B = population
    result = monte_carlo(
        _replicate(study, labels, 1, 0), truth=float(B[1, 0]), reps=REPS, seed=7
    )
    assert_coverage(result, nominal=0.95, label="design interval")


def test_a_fixed_size_draw_is_covered_conservatively(population) -> None:
    """Cube draws have smaller variance than the Poisson formula reports.

    The package says so in `design_risk`; this measures it rather than assuming
    it. Fixing the sample size and balancing on the influence projections both
    remove variation the Poisson expression still counts, so intervals must come
    out at least as wide as they need to be.
    """
    study, labels, _ = population
    design = study.design(BUDGET)
    rng = np.random.default_rng(8)
    values, ses = [], []
    for _ in range(400):
        sample = study.draw(design, random_state=rng, fixed_size=True)
        est = estimate(study.influence, sample, labels.loc[sample.items])
        values.append(float(est.coefficients.iloc[1, 0]))
        ses.append(float(est.design_se.iloc[1, 0]))
    spread = float(np.std(values, ddof=1))
    reported = float(np.mean(ses))
    assert reported > spread, (
        f"the Poisson variance under-reports a cube draw: {reported:.5f} "
        f"vs actual spread {spread:.5f}"
    )
    assert len({len(v) for v in [values]}) == 1


def test_a_useless_proxy_costs_variance_and_never_bias(population) -> None:
    """The difference estimator is unbiased whatever the proxy claims.

    The proxy here is deliberately wrong -- it predicts the opposite class -- so
    if the correction term were doing anything other than what the algebra says,
    this would show up as bias rather than as extra noise.
    """
    study, labels, B = population
    nonsense = pd.DataFrame(
        1.0 - labels.to_numpy(), index=labels.index, columns=labels.columns
    )
    result = monte_carlo(
        _replicate(study, labels, 1, 0, proxy=nonsense),
        truth=float(B[1, 0]),
        reps=REPS,
        seed=9,
    )
    assert_unbiased(result, label="adversarial proxy")


def test_a_good_proxy_buys_a_smaller_standard_error(population) -> None:
    """The point of the proxy is variance; check it actually delivers."""
    study, labels, B = population
    rng = np.random.default_rng(10)
    noisy = labels.to_numpy() * 0.85 + 0.075
    flip = rng.random(len(labels)) < 0.1
    noisy[flip] = 1.0 - noisy[flip]
    good = pd.DataFrame(noisy, index=labels.index, columns=labels.columns)

    design = study.design(BUDGET)
    plain, assisted = [], []
    r = np.random.default_rng(11)
    for _ in range(300):
        sample = study.draw(design, random_state=r, fixed_size=False)
        obs = labels.loc[sample.items]
        plain.append(
            float(estimate(study.influence, sample, obs).coefficients.iloc[1, 0])
        )
        assisted.append(
            float(
                estimate(study.influence, sample, obs, proxy=good).coefficients.iloc[
                    1, 0
                ]
            )
        )
    B10 = float(B[1, 0])
    mse_plain = float(np.mean((np.array(plain) - B10) ** 2))
    mse_assisted = float(np.mean((np.array(assisted) - B10) ** 2))
    assert mse_assisted < mse_plain, (
        f"proxy did not help: {mse_assisted:.6f} vs {mse_plain:.6f}"
    )


def test_the_bias_bound_actually_bounds_the_bias(population) -> None:
    """Shrinkage is only defensible if its price is stated correctly."""
    study, labels, B = population
    design = study.design(BUDGET)
    rng = np.random.default_rng(12)
    errors, bounds = [], []
    for _ in range(300):
        sample = study.draw(design, random_state=rng, fixed_size=False)
        alpha = trim_alpha(sample, weight_cap=5.0)
        est = estimate(study.influence, sample, labels.loc[sample.items], alpha=alpha)
        errors.append(est.coefficients.to_numpy() - B)
        bounds.append(est.bias_bound)
    realized = float(np.linalg.norm(np.mean(errors, axis=0)))
    claimed = float(np.mean(bounds))
    assert claimed > 0, "trimming produced no bias bound at all"
    assert realized <= claimed, (
        f"actual bias {realized:.4f} exceeded the stated bound {claimed:.4f}"
    )


def test_no_shrinkage_reports_no_bias(population) -> None:
    """The default must advertise itself as unbiased."""
    study, labels, _ = population
    sample = study.draw(study.design(BUDGET), random_state=13, fixed_size=False)
    est = estimate(study.influence, sample, labels.loc[sample.items])
    assert est.bias_bound == 0.0
    assert est.diagnostics["unbiased"] is True


def test_a_ranking_cannot_be_estimated_from(population) -> None:
    """The central repair, enforced by the type rather than by documentation."""
    study, labels, _ = population
    ranking = study.rank(BUDGET)
    assert isinstance(ranking, Ranking)
    with pytest.raises(TypeError, match="LabelSample"):
        study.estimate(ranking, labels.loc[ranking.items])
    with pytest.raises(NotImplementedError, match="cannot be converted"):
        ranking.to_design(BUDGET)


def test_a_bare_list_of_entities_is_refused(population) -> None:
    """A set of names does not say how it was chosen, so it cannot be weighted."""
    study, labels, _ = population
    with pytest.raises(TypeError, match="LabelSample"):
        study.estimate(list(labels.index[:BUDGET]), labels.iloc[:BUDGET])


def test_missing_labels_are_refused_rather_than_dropped(population) -> None:
    """Dropping a sampled entity biases the estimate by its contribution."""
    study, labels, _ = population
    sample = study.draw(study.design(BUDGET), random_state=14, fixed_size=False)
    partial = labels.loc[sample.items].iloc[:-1]
    with pytest.raises(ValueError, match="missing"):
        estimate(study.influence, sample, partial)


def test_both_variance_components_are_reported(population) -> None:
    """The budget moves one of them and not the other; say which."""
    study, labels, _ = population
    sample = study.draw(study.design(BUDGET), random_state=15, fixed_size=False)
    est = study.estimate(sample, labels.loc[sample.items])
    assert est.regression_se is not None
    assert est.total_se is not None
    total = est.total_se.to_numpy()
    assert np.all(total >= est.design_se.to_numpy() - 1e-12)
    assert np.all(total >= est.regression_se.to_numpy() - 1e-12)
    assert 0.0 <= est.diagnostics["variance_ratio"] <= 1.0


def test_spending_more_shrinks_only_the_design_component(population) -> None:
    """The stopping rule depends on this being true; check it is."""
    study, labels, _ = population
    ratios = []
    for budget in (20, 60, 110):
        sample = study.draw(study.design(budget), random_state=16, fixed_size=False)
        est = study.estimate(sample, labels.loc[sample.items])
        ratios.append(est.diagnostics["variance_ratio"])
    assert ratios[0] > ratios[-1], (
        f"design share of variance did not fall with budget: {ratios}"
    )

"""Risk ceilings, minimum cost, and the inverse of the budget problem."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from simcheck import Estimate, assert_unbiased, monte_carlo

from fewlab import Study, design_risk, estimate, min_cost_for_risk, poisson_pps
from tests.fixtures import one_hot_labels, panel


@pytest.mark.parametrize("budget", [0.1, 0.9, 1.0, 2.0, 9.9, 10.0])
def test_equal_scores_need_no_arbitrary_certainty_core(budget):
    scores = pd.Series(np.ones(10))
    design = poisson_pps(scores, budget)
    np.testing.assert_allclose(design.probabilities, budget / 10, atol=1e-14)
    assert design.risk == pytest.approx(100 / budget - 10)
    inverse = min_cost_for_risk(scores, design.risk)
    np.testing.assert_allclose(inverse.probabilities, design.probabilities)
    assert inverse.budget == pytest.approx(budget)


@pytest.mark.parametrize("risk", [0.0, 0.001, 0.1, 1.0, 40.0, 1e8])
def test_ceiling_and_inverse_budget_agree_with_heterogeneous_costs(risk):
    rng = np.random.default_rng(52)
    for _ in range(15):
        scores = pd.Series(np.exp(rng.normal(0, 3, 30)))
        costs = pd.Series(np.exp(rng.normal(0, 1, 30)))
        design = min_cost_for_risk(scores, risk, cost=costs)
        pi = design.probabilities
        assert (pi > 0).all()
        assert (pi <= 1).all()
        assert design.risk <= risk + 1e-10 * risk
        assert design.risk == pytest.approx(risk, rel=1e-9, abs=1e-12)
        assert design.diagnostics["expected_cost"] == pytest.approx(costs @ pi)
        reverse = poisson_pps(scores, float(costs @ pi), cost=costs)
        np.testing.assert_allclose(reverse.probabilities, pi, rtol=1e-10)


def test_the_design_minimizes_cost_not_just_constraint_violation():
    scores = pd.Series([9.0, 1.0, 0.25])
    costs = pd.Series([0.2, 2.0, 1.0])
    design = min_cost_for_risk(scores, 2, cost=costs)
    assert 0 < len(design.certain) < len(scores)
    # An independent grid search must not find a cheaper feasible allocation.
    for candidate in product(np.linspace(0.05, 1, 20), repeat=3):
        if design_risk(scores, np.array(candidate)) <= 2:
            assert costs @ candidate >= design.diagnostics["expected_cost"] - 1e-12
    # KKT stationarity in the tail and the inequality on capped entities.
    pi = design.probabilities
    eta = costs * pi**2 / scores
    tail = pi < 1
    multiplier = eta[tail].iloc[0]
    np.testing.assert_allclose(eta[tail], multiplier)
    assert (eta[~tail] <= multiplier).all()


def test_tighter_ceilings_cost_more_and_zero_score_padding_is_free():
    scores = pd.Series([100.0, 4.0, 1.0], index=list("abc"))
    previous = 0
    for risk in [100, 10, 1, 0.1, 0]:
        design = min_cost_for_risk(scores, risk)
        assert design.budget >= previous
        previous = design.budget
        padded = min_cost_for_risk(pd.concat([scores, pd.Series({"dead": 0})]), risk)
        np.testing.assert_array_equal(
            padded.probabilities.loc[scores.index], design.probabilities
        )
        assert padded.probabilities["dead"] == 0
        assert padded.budget == design.budget
        assert padded.risk == design.risk


@pytest.mark.parametrize("scores", [pd.Series(dtype=float), pd.Series([0.0, 0.0])])
@pytest.mark.parametrize("risk", [0, 1])
def test_empty_and_zero_scores_cost_nothing(scores, risk):
    design = min_cost_for_risk(scores, risk)
    assert design.budget == design.risk == design.diagnostics["expected_cost"] == 0
    assert (design.probabilities == 0).all()


def test_indexed_costs_are_aligned_by_entity():
    scores = pd.Series([4.0, 1.0], index=["a", "b"])
    costs = pd.Series([0.5, 3.0], index=["a", "b"])
    for builder, limit in [(min_cost_for_risk, 2), (poisson_pps, 1)]:
        expected = builder(scores, limit, cost=costs)
        actual = builder(scores, limit, cost=costs.iloc[::-1])
        pd.testing.assert_series_equal(actual.probabilities, expected.probabilities)


@pytest.mark.parametrize("risk", [-1, np.nan, np.inf, -np.inf])
def test_invalid_ceilings_are_rejected(risk):
    with pytest.raises(ValueError, match="risk_limit"):
        min_cost_for_risk(pd.Series([1.0]), risk)


@pytest.mark.parametrize("scores", [[-1], [np.nan], [np.inf]])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="scores"):
        min_cost_for_risk(pd.Series(scores), 1)


@pytest.mark.parametrize(
    "cost",
    [
        0,
        -1,
        np.inf,
        np.nan,
        [1],
        [[1, 2]],
        pd.Series([1, 2], index=["x", "y"]),
        pd.Series([1, 2], index=["a", "a"]),
    ],
)
def test_invalid_costs_are_rejected(cost):
    with pytest.raises(ValueError, match="cost"):
        min_cost_for_risk(pd.Series([1, 2], index=["a", "b"]), 1, cost=cost)


def test_duplicate_entities_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        min_cost_for_risk(pd.Series([1, 2], index=["a", "a"]), 1)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_rescaling_scores_and_risk_preserves_the_design(scale):
    scores = pd.Series([1e-50, 1e-10, 1.0, 1e50])
    base = min_cost_for_risk(scores, 1e49)
    scaled = min_cost_for_risk(scores * scale, 1e49 * scale)
    np.testing.assert_allclose(scaled.probabilities, base.probabilities, rtol=1e-12)
    assert scaled.risk <= 1e49 * scale * (1 + 1e-12)


def test_unrepresentable_probabilities_raise_instead_of_dropping_entities():
    with pytest.raises(ArithmeticError, match="unrepresentable"):
        min_cost_for_risk(pd.Series([1e-300, 1e300]), 1e308, cost=[1e300, 1e-300])


@pytest.fixture(scope="module")
def study():
    counts, X = panel(n=100, m=30, seed=12)
    return Study(counts, X)


@pytest.mark.parametrize(
    "kwargs", [{}, {"loss": "prediction"}, {"contrast": np.array([0.0, 1.0, 0.0])}]
)
def test_study_forwards_the_declared_estimand(study, kwargs):
    scores = study.scores(**kwargs)
    limit = float(scores.sum())
    design = study.design_for_risk(limit, cost=2.0, **kwargs)
    expected = min_cost_for_risk(scores, limit, cost=2.0)
    pd.testing.assert_series_equal(design.probabilities, expected.probabilities)
    sample = study.draw(design, fixed_size=False, random_state=14)
    labels = pd.Series(1.0, index=sample.items)
    assert np.isfinite(study.estimate(sample, labels).coefficients).all().all()


def test_contrast_ceiling_matches_design_variance_and_unbiased_estimation(study):
    labels = one_hot_labels(study.influence.items, classes=2, seed=5)
    contrast = np.array([0.0, 1.0, 0.0])
    scores = study.scores(contrast=contrast)
    limit = float(scores.sum() * 0.3)
    design = study.design_for_risk(limit, contrast=contrast)
    truth = study.influence.H[1] @ labels.to_numpy()
    pi = design.probabilities.to_numpy()
    variance = (study.influence.H[1] ** 2 * (1 - pi) / pi) @ labels.to_numpy() ** 2
    assert variance.sum() == pytest.approx(limit)

    def replicate(rng):
        sample = study.draw(design, fixed_size=False, random_state=rng)
        estimated = estimate(study.influence, sample, labels.loc[sample.items])
        return Estimate(
            value=float(estimated.coefficients.iloc[1, 0]),
            standard_error=float(estimated.design_se.iloc[1, 0]),
        )

    result = monte_carlo(replicate, truth=float(truth[0]), reps=1500, seed=27)
    assert_unbiased(result, label="risk-ceiling contrast")
    # The variance estimate is unbiased; its square root need not be.
    for values in (
        (result.estimates - truth[0]) ** 2,
        result.standard_errors**2,
    ):
        mc_se = np.std(values, ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - variance[0]) < 5 * mc_se


def test_joint_one_hot_risk_matches_monte_carlo_and_bounds_binary_risk(study):
    labels = one_hot_labels(study.influence.items, classes=2, seed=5)
    design = study.design_for_risk(float(study.scores().sum() * 0.5))
    pi = design.probabilities.to_numpy()
    H = study.influence.H
    truth = H @ labels.to_numpy()
    rng = np.random.default_rng(73)
    losses = []
    for _ in range(4000):
        sampled = rng.random(pi.size) < pi
        beta = H @ ((sampled / pi)[:, None] * labels.to_numpy())
        losses.append(float(np.sum((beta - truth) ** 2)))
    mc_se = np.std(losses, ddof=1) / np.sqrt(len(losses))
    assert abs(np.mean(losses) - design.risk) < 5 * mc_se
    binary_risk = design_risk(
        study.scores(label_norms=labels.iloc[:, 0].to_numpy()), pi
    )
    assert binary_risk <= design.risk


def test_poisson_risk_is_not_a_bound_for_every_fixed_size_design():
    # Two opposite contributions cancel in the Poisson cross term but SRS
    # selects exactly one: its variance is four, twice the Poisson risk.
    contributions = np.array([-1.0, 1.0])
    pi = np.array([0.5, 0.5])
    fixed_size_variance = np.mean((contributions / pi) ** 2)
    assert fixed_size_variance > design_risk(contributions**2, pi)

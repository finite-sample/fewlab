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


VARIANCE_REPS = 900


def _variance_ratios(study, labels, fixed_size: bool, reps: int) -> dict:
    """Measure `E[v_hat] / Var(B_hat)` for each variance estimator.

    Args:
        study: The study.
        labels: True labels.
        fixed_size: Whether to draw with the cube method.
        reps: Replicates.

    Returns:
        Median ratio per variance estimator; 1.0 is exactly right.
    """
    design = study.design(BUDGET)
    rng = np.random.default_rng(21)
    values: list[np.ndarray] = []
    var: dict[str, list[np.ndarray]] = {"poisson": [], "hajek": [], "balanced": []}
    for _ in range(reps):
        sample = study.draw(design, random_state=rng, fixed_size=fixed_size)
        if len(sample) == 0:
            continue
        obs = labels.loc[sample.items]
        values.append(
            estimate(study.influence, sample, obs).coefficients.to_numpy().ravel()
        )
        for name in var:
            var[name].append(
                estimate(study.influence, sample, obs, variance=name)
                .design_se.to_numpy()
                .ravel()
                ** 2
            )
    truth = np.array(values).var(axis=0)
    return {
        name: float(np.median(np.array(v).mean(axis=0) / truth))
        for name, v in var.items()
    }


def test_the_poisson_variance_is_exact_for_a_poisson_draw(population) -> None:
    """The default has to be right on the design it is derived for."""
    study, labels, _ = population
    ratios = _variance_ratios(study, labels, fixed_size=False, reps=VARIANCE_REPS)
    assert 0.9 < ratios["poisson"] < 1.1, ratios


def test_the_poisson_variance_is_conservative_for_a_cube_draw(population) -> None:
    """The package calls it conservative; measure that rather than assert it."""
    study, labels, _ = population
    ratios = _variance_ratios(study, labels, fixed_size=True, reps=VARIANCE_REPS // 2)
    assert ratios["poisson"] > 1.3, (
        f"the Poisson variance is no longer conservative under a cube draw "
        f"({ratios['poisson']:.2f}); the documentation says it is"
    )


def test_the_balanced_variance_recovers_what_conservatism_costs(population) -> None:
    """The reason to have it: a cube interval should not be twice as wide as it needs.

    The gate is deliberately tight. A scalar degrees-of-freedom correction such
    as `k/(k-q)` passes a loose band while understating the variance by about a
    fifth, which is how the earlier implementation looked correct for longer
    than it was. Only the leverage correction lands inside this one.
    """
    study, labels, _ = population
    ratios = _variance_ratios(study, labels, fixed_size=True, reps=VARIANCE_REPS // 2)
    assert ratios["balanced"] < ratios["hajek"] < ratios["poisson"], ratios
    assert 0.85 < ratios["balanced"] < 1.15, ratios


def test_the_leverage_correction_beats_the_scalar_one_it_replaced(population) -> None:
    """The falsification partner for the correction itself.

    A shared factor such as `k/(k-q)` is the natural first implementation and it
    is the right correction only when the residuals are homoskedastic. The
    inclusion probabilities here span orders of magnitude, so leverage does too.
    The two agree when few entities sit at certainty and diverge as the core
    grows, which is why this runs at a budget large enough to build one, and
    compares them on identical draws rather than against fixed thresholds.
    """
    study, labels, _ = population
    budget = 80
    design = study.design(budget)
    rng = np.random.default_rng(31)
    items = list(study.influence.items)
    q = study.influence.n_features + 1
    values, scalar, leverage = [], [], []
    for _ in range(400):
        sample = study.draw(design, random_state=rng, fixed_size=True)
        if len(sample) == 0:
            continue
        pos = np.array([items.index(i) for i in sample.items])
        pi = sample.probabilities.reindex(sample.items).to_numpy()
        obs = labels.loc[sample.items].to_numpy()
        contrib = (
            np.einsum("pk,kq->kqp", study.influence.H[:, pos], obs).reshape(len(pi), -1)
            / pi[:, None]
        )
        values.append(contrib.sum(axis=0))

        k = len(pi)
        mat = np.c_[np.ones(k), (study.influence.G[:, pos] / pi).T]
        base = 1.0 - pi
        root = np.sqrt(base)[:, None]
        whitened = mat * root
        pinv = np.linalg.pinv(whitened.T @ whitened)
        resid = contrib - mat @ (pinv @ (whitened.T @ (contrib * root)))
        lev = np.clip(np.einsum("ki,ij,kj->k", whitened, pinv, whitened), 0.0, 1 - 1e-9)
        scalar.append(np.diag((resid * (base * k / max(k - q, 1))[:, None]).T @ resid))
        leverage.append(np.diag((resid * (base / (1 - lev))[:, None]).T @ resid))

    truth = np.array(values).var(axis=0)
    r_scalar = float(np.median(np.array(scalar).mean(axis=0) / truth))
    r_leverage = float(np.median(np.array(leverage).mean(axis=0) / truth))
    assert abs(r_leverage - 1) < abs(r_scalar - 1), (
        f"the leverage correction is no better than the scalar one it replaced: "
        f"{r_leverage:.3f} against {r_scalar:.3f}"
    )
    assert r_scalar < 0.9, (
        f"the scalar correction is no longer visibly biased ({r_scalar:.3f}), so "
        f"this test no longer measures the difference it was written for"
    )


def test_the_fixed_size_estimators_understate_a_poisson_draw(population) -> None:
    """The falsification partner: using the wrong one has to be visibly wrong.

    If `balanced` were quietly correct on a Bernoulli draw too, the distinction
    the API asks users to make would be pointless, and the warning in the
    docstring would be noise.
    """
    study, labels, _ = population
    ratios = _variance_ratios(study, labels, fixed_size=False, reps=VARIANCE_REPS)
    assert ratios["balanced"] < 0.7, (
        f"the balanced variance no longer understates a Poisson draw "
        f"({ratios['balanced']:.2f}), so the estimators are no longer distinct"
    )


def test_an_unknown_variance_estimator_is_refused(population) -> None:
    """Silently falling back would change the reported uncertainty."""
    study, labels, _ = population
    sample = study.draw(study.design(BUDGET), random_state=22, fixed_size=False)
    with pytest.raises(ValueError, match="variance must be one of"):
        estimate(study.influence, sample, labels.loc[sample.items], variance="syg")

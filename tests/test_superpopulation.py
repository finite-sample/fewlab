"""GLS: much more precise, and aimed at a different quantity.

These tests exist to stop anyone reading the variance reduction as a free
upgrade. The efficiency is real and the bias for the finite-population target is
real, and a user has to choose which they want before either number means
anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fewlab import LabelSample, Study, estimate_superpopulation
from tests.fixtures import exact_truth, one_hot_labels, panel

BUDGET = 50
REPS = 300


@pytest.fixture(scope="module")
def setup() -> tuple[Study, pd.DataFrame, np.ndarray]:
    """A population with heterogeneous per-unit design variance.

    Returns:
        The study, its labels, and the finite-population truth.
    """
    counts, X = panel(n=500, m=250, seed=44)
    study = Study(counts, X)
    labels = one_hot_labels(study.influence.items, classes=1, seed=9)
    return study, labels, exact_truth(study, labels)


def _draws(study, labels, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect GLS and OLS estimates over repeated draws.

    Args:
        study: The study.
        labels: True labels.
        reps: Replicates.

    Returns:
        Arrays of GLS and OLS estimates.
    """
    design = study.design(BUDGET)
    rng = np.random.default_rng(1)
    gls, ols = [], []
    for _ in range(reps):
        sample = study.draw(design, random_state=rng, fixed_size=False)
        if len(sample) == 0:
            continue
        est = estimate_superpopulation(
            study.influence, study.counts, study.X, sample, labels.loc[sample.items]
        )
        gls.append(est.coefficients.to_numpy().ravel())
        ols.append(est.ols.to_numpy().ravel())
    return np.array(gls), np.array(ols)


def test_gls_is_more_precise_than_ols(setup) -> None:
    """The reason anyone would want it."""
    study, labels, _ = setup
    gls, ols = _draws(study, labels, REPS)
    assert np.all(gls.var(axis=0) < ols.var(axis=0)), (
        f"GLS was not more precise: {gls.var(axis=0)} vs {ols.var(axis=0)}"
    )


def test_gls_is_biased_for_the_finite_population_target(setup) -> None:
    """The falsification partner, and the whole reason for a separate module.

    Any linear estimator `L yhat` has expectation `L y`, so hitting
    `(X'X)^-1 X' y` for an unknown `y` forces `L = (X'X)^-1 X'`. GLS uses a
    different `L`. If this test ever passes as unbiased, either the weighting
    has stopped doing anything or the estimand has been confused again.
    """
    study, labels, truth = setup
    gls, ols = _draws(study, labels, REPS)
    gls_bias = np.abs(gls.mean(axis=0) - truth.ravel())
    ols_bias = np.abs(ols.mean(axis=0) - truth.ravel())
    mc_error = gls.std(axis=0) / np.sqrt(len(gls))
    assert np.any(gls_bias > 5 * mc_error), (
        "GLS is no longer distinguishably biased for B, which contradicts the "
        "algebra in superpopulation.py"
    )
    assert np.all(ols_bias <= 5 * ols.std(axis=0) / np.sqrt(len(ols))), (
        "OLS on yhat should remain unbiased for the finite-population target"
    )


def test_it_reports_the_heterogeneity_it_exploits(setup) -> None:
    """With no spread in per-unit design variance there is nothing to gain, so
    the diagnostic is what tells a user whether to bother."""
    study, labels, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=2, fixed_size=False)
    est = estimate_superpopulation(
        study.influence, study.counts, study.X, sample, labels.loc[sample.items]
    )
    assert est.diagnostics["design_variance_spread"] > 1.0
    assert (est.standard_errors.to_numpy() > 0).all()
    assert est.sigma2.shape == (1,)


def test_the_ols_estimate_it_returns_matches_the_main_estimator(setup) -> None:
    """It reports OLS alongside so the two can be compared; they must agree
    with what the rest of the package computes."""
    import fewlab

    study, labels, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=3, fixed_size=False)
    est = estimate_superpopulation(
        study.influence, study.counts, study.X, sample, labels.loc[sample.items]
    )
    direct = fewlab.estimate(study.influence, sample, labels.loc[sample.items])
    np.testing.assert_allclose(
        est.ols.to_numpy(), direct.coefficients.to_numpy(), rtol=1e-8
    )


def test_missing_labels_are_refused(setup) -> None:
    """Same contract as everywhere else."""
    study, labels, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=4, fixed_size=False)
    with pytest.raises(ValueError, match="missing"):
        estimate_superpopulation(
            study.influence,
            study.counts,
            study.X,
            sample,
            labels.loc[sample.items].iloc[:-1],
        )


@pytest.mark.parametrize(("repeats", "copies"), [(4, 1), (1, 1), (1, 12)])
def test_gls_matches_four_independent_group_means(repeats, copies) -> None:
    """Repeated units share their design noise and average their residual noise."""
    counts = pd.DataFrame(np.kron(np.eye(4), np.ones((repeats, 2 * copies))))
    counts.columns = [f"entity{i}" for i in range(counts.shape[1])]
    group_x = np.column_stack((np.ones(4), [-1.0, -0.2, 0.6, 1.1]))
    X = pd.DataFrame(np.repeat(group_x, repeats, axis=0), columns=["const", "slope"])
    study = Study(counts, X)
    probabilities = np.array([0.42, 0.65, 0.48, 0.6])
    pi = pd.Series(np.repeat(probabilities, 2 * copies), index=counts.columns)
    sample = LabelSample(counts.columns[::2], pi, "poisson")
    labels = pd.Series(1.0, index=sample.items, name="class")
    result = estimate_superpopulation(study.influence, counts, X, sample, labels)

    np.testing.assert_array_equal(result.sigma2, [1e-12])
    group_mean = 0.5 / probabilities
    variance = (1 - probabilities) / (4 * copies * probabilities**2) + 1e-12 / repeats
    precision = group_x.T @ (group_x / variance[:, None])
    beta = np.linalg.solve(precision, group_x.T @ (group_mean / variance))
    se = np.sqrt(np.diag(np.linalg.inv(precision)))
    np.testing.assert_allclose(result.coefficients.iloc[:, 0], beta, atol=1e-9)
    np.testing.assert_allclose(result.standard_errors.iloc[:, 0], se, atol=1e-9)


def test_gls_rejects_an_aliased_feature(setup) -> None:
    study, labels, _ = setup
    sample = study.draw(study.design(BUDGET), random_state=4, fixed_size=False)
    X = study.X.assign(duplicate=study.X.iloc[:, -1])
    with pytest.raises(np.linalg.LinAlgError, match="independent"):
        estimate_superpopulation(study.influence, study.counts, X, sample, labels)

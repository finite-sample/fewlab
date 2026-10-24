"""The workflow: long-format data in, a design, a draw, an estimate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fewlab import LabelSample, Study
from fewlab.counts import CountMatrix
from tests.fixtures import one_hot_labels, panel


@pytest.fixture(scope="module")
def study() -> Study:
    """A study over a realistic panel.

    Returns:
        The study.
    """
    counts, X = panel(n=250, m=150, seed=41)
    return Study(counts, X)


def test_long_format_reproduces_the_dense_build() -> None:
    """A panel arrives as triples; that path must not change the answer."""
    counts, X = panel(n=120, m=60, seed=42)
    long = (
        counts.stack()
        .rename("visits")
        .reset_index()
        .rename(columns={"level_0": "user", "level_1": "site"})
    )
    long = long[long["visits"] > 0]
    dense = Study(counts, X)
    sparse = Study.from_long(
        long,
        X,
        unit="user",
        entity="site",
        count="visits",
        entities=pd.Index(counts.columns),
    )
    np.testing.assert_allclose(dense.influence.H, sparse.influence.H, atol=1e-12)
    assert dense.frame_fingerprint == sparse.frame_fingerprint
    sample = dense.draw(dense.design(20), random_state=12)
    labels = one_hot_labels(dense.influence.items, classes=2, seed=13)
    from_dense = dense.estimate(sample, labels.loc[sample.items])
    from_sparse = sparse.estimate(sample, labels.loc[sample.items])
    np.testing.assert_allclose(from_dense.coefficients, from_sparse.coefficients)


def test_the_sample_reports_the_probabilities_it_was_drawn_under(study) -> None:
    """Weights come from delivered probabilities, never recomputed from a size."""
    design = study.design(40)
    sample = study.draw(design, random_state=3, fixed_size=True)
    pd.testing.assert_series_equal(sample.probabilities, design.probabilities)
    weights = sample.weights
    np.testing.assert_allclose(
        weights.to_numpy(), 1.0 / design.probabilities.reindex(sample.items).to_numpy()
    )


def test_a_fixed_size_draw_delivers_exactly_the_budget(study) -> None:
    """A strict budget has to be strict."""
    design = study.design(40)
    for seed in range(6):
        assert len(study.draw(design, random_state=seed, fixed_size=True)) == 40


def test_the_default_draw_is_poisson_and_varies_around_the_budget(study) -> None:
    """The main workflow has an expected count and exact design variance."""
    design = study.design(40)
    rng = np.random.default_rng(4)
    samples = [study.draw(design, random_state=rng) for _ in range(60)]
    sizes = [len(sample) for sample in samples]
    assert all(sample.method.endswith("+poisson") for sample in samples)
    assert len(set(sizes)) > 1
    assert abs(np.mean(sizes) - 40) < 4

    hits = pd.Series(0, index=design.probabilities.index, dtype=float)
    reps = 1500
    for _ in range(reps):
        hits.loc[study.draw(design, random_state=rng).items] += 1
    pi = design.probabilities.to_numpy()
    tolerance = 5 * np.sqrt(pi * (1 - pi) / reps) + 1 / reps
    assert np.all(np.abs(hits.to_numpy() / reps - pi) < tolerance)


def test_a_fixed_size_draw_needs_an_integer_budget(study) -> None:
    """Half an entity cannot be labeled."""
    design = study.design(40.5)
    with pytest.raises(ValueError, match="integer budget"):
        study.draw(design, random_state=0, fixed_size=True)


def test_certainty_entities_appear_in_every_draw(study) -> None:
    """Probability one has to mean probability one."""
    design = study.design(60)
    certain = set(design.certain)
    assert certain
    for seed in range(8):
        drawn = set(study.draw(design, random_state=seed, fixed_size=True).items)
        assert certain <= drawn


def test_the_ranking_reports_how_much_mass_it_covers(study) -> None:
    """A priority list is only useful alongside what it leaves out."""
    ranking = study.rank(30)
    assert len(ranking) == 30
    assert 0.0 < ranking.coverage <= 1.0
    full = study.rank()
    assert full.coverage == pytest.approx(1.0)


def test_diagnostics_expose_the_degeneracy_check(study) -> None:
    """A user should be able to see when no design can help them."""
    d = study.diagnostics
    assert d["n_entities"] == 150
    assert 0 < d["density"] <= 1
    assert d["effective_directions"] >= 1
    assert len(d["direction_spectrum"]) == study.influence.n_features


def test_an_end_to_end_estimate_lands_near_the_truth(study) -> None:
    """One draw, one estimate, checked against the population it came from."""
    labels = one_hot_labels(study.influence.items, classes=2, seed=43)
    B = study.influence.H @ labels.loc[study.influence.items].to_numpy(float)
    sample = study.draw(study.design(70), random_state=5, fixed_size=True)
    est = study.estimate(sample, labels.loc[sample.items])
    err = np.abs(est.coefficients.to_numpy() - B)
    assert np.all(err <= 4 * est.design_se.to_numpy() + 1e-9)


def test_counts_can_be_handed_over_already_sparse(study) -> None:
    """The sparse container is a first-class input, not an internal detail."""
    counts, X = panel(n=100, m=50, seed=44)
    direct = Study(CountMatrix.from_dense(counts), X)
    assert direct.influence.n_items == 50


def test_the_variance_estimator_is_reachable_through_the_study(study) -> None:
    """A choice only available on the low-level function is not available.

    On a cube draw the Poisson variance is conservative, so the alternatives
    must be narrower; a user going through `Study` should be able to say so.
    """
    labels = one_hot_labels(study.influence.items, classes=2, seed=45)
    sample = study.draw(study.design(40), random_state=6, fixed_size=True)
    widths = {
        name: float(
            study.estimate(sample, labels.loc[sample.items], variance=name)
            .design_se.to_numpy()
            .mean()
        )
        for name in ("poisson", "hajek", "balanced")
    }
    assert widths["balanced"] < widths["hajek"] < widths["poisson"], widths


@pytest.mark.parametrize(
    ("fixed_size", "expected"), [(False, "poisson"), (True, "balanced")]
)
def test_default_variance_matches_the_draw(study, fixed_size, expected) -> None:
    """The public workflow must carry its draw method into inference."""
    labels = one_hot_labels(study.influence.items, classes=2, seed=46)
    design = study.design(60)
    assert len(design.certain) > 0
    sample = study.draw(design, random_state=7, fixed_size=fixed_size)
    observed = labels.loc[sample.items]
    proxy = labels * 0.5
    automatic = study.estimate(sample, observed, proxy=proxy, row_model_se=True)
    explicit = study.estimate(
        sample, observed, proxy=proxy, variance=expected, row_model_se=True
    )
    assert automatic.diagnostics["variance"] == expected
    np.testing.assert_allclose(automatic.design_cov, explicit.design_cov)
    np.testing.assert_allclose(automatic.regression_se, explicit.regression_se)
    np.testing.assert_allclose(automatic.total_se, explicit.total_se)


@pytest.mark.parametrize("fixed_size", [False, True])
def test_saved_draw_restores_the_same_estimate(study, tmp_path, fixed_size) -> None:
    """The handoff preserves selected IDs, every probability, and draw method."""
    labels = one_hot_labels(study.influence.items, classes=2, seed=47)
    sample = study.draw(study.design(40), random_state=8, fixed_size=fixed_size)
    path = tmp_path / "labels_to_acquire.json"
    sample.save(path)
    restored = LabelSample.load(path)
    assert restored.items.equals(sample.items)
    np.testing.assert_array_equal(restored.probabilities, sample.probabilities)
    assert restored.method == sample.method
    assert restored.frame_fingerprint == study.frame_fingerprint
    assert restored.diagnostics["random_seed"] == 8
    direct = study.estimate(sample, labels.loc[sample.items])
    resumed = study.estimate(restored, labels.loc[restored.items])
    np.testing.assert_allclose(resumed.coefficients, direct.coefficients)
    np.testing.assert_allclose(resumed.design_cov, direct.design_cov)


def test_sample_from_different_study_is_rejected_even_with_same_ids(study) -> None:
    """Reusing probabilities after changing the data is an invalid handoff."""
    sample = study.draw(study.design(40), random_state=9)
    changed = Study(study.counts, study.X.assign(extra=0.0))
    labels = one_hot_labels(study.influence.items, classes=2, seed=48)
    with pytest.raises(ValueError, match="not drawn from this Study frame"):
        changed.estimate(sample, labels.loc[sample.items])


def test_corrupt_manifest_probability_is_rejected(study, tmp_path) -> None:
    """An invalid probability must fail when the draw is restored."""
    sample = study.draw(study.design(40), random_state=10)
    path = tmp_path / "draw.json"
    sample.save(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["entities"][0]["inclusion_probability"] = -0.1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="probabilities must be finite"):
        LabelSample.load(path)


def test_draw_rejects_a_design_for_another_entity_frame(study) -> None:
    """The design must cover the actual frame in its original order."""
    from dataclasses import replace

    design = study.design(40)
    wrong = replace(design, probabilities=design.probabilities.iloc[::-1])
    with pytest.raises(ValueError, match="cover this Study frame"):
        study.draw(wrong, random_state=0)

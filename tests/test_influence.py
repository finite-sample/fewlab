"""The exact decomposition, and the diagnostic that says when it stops mattering.

Every claim this package makes rests on `B = sum_j h_j a_j'` being an identity
rather than an approximation. The first test here is therefore the load-bearing
one: if it fails, nothing downstream means anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fewlab import contribution_scores, direction_spectrum, influence
from fewlab.counts import CountMatrix
from tests.fixtures import one_hot_labels, panel, rank_one


def test_the_decomposition_is_an_identity_not_an_approximation() -> None:
    """`sum_j h_j a_j'` must equal the complete-label OLS fit exactly."""
    counts, X = panel(seed=1)
    infl = influence(counts, X)
    A = one_hot_labels(list(counts.columns), seed=2)

    V = counts.to_numpy(float) / counts.sum(axis=1).to_numpy(float)[:, None]
    Y = V @ A.to_numpy(float)
    B_ols = np.linalg.lstsq(X.to_numpy(float), Y, rcond=None)[0]

    np.testing.assert_allclose(infl.H @ A.to_numpy(float), B_ols, atol=1e-10)


def test_the_sparse_and_dense_paths_agree() -> None:
    """A panel held as triples must give the same answer as a dense matrix."""
    counts, X = panel(m=80, seed=3)
    dense = influence(counts, X)
    sparse = influence(CountMatrix.from_dense(counts), X)
    np.testing.assert_allclose(dense.H, sparse.H, atol=1e-12)
    np.testing.assert_allclose(dense.G, sparse.G, atol=1e-12)


def test_the_two_losses_score_the_documented_quantities() -> None:
    """Coefficient loss is `h'h`; prediction loss is `h'(X'X)h`."""
    counts, X = panel(seed=4)
    infl = influence(counts, X)
    H = infl.H
    np.testing.assert_allclose(
        contribution_scores(infl, loss="coefficient").to_numpy(),
        np.einsum("pj,pj->j", H, H),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        contribution_scores(infl, loss="prediction").to_numpy(),
        np.einsum("pj,pj->j", H, infl.XtX @ H),
        rtol=1e-12,
    )


def test_declaring_the_loss_changes_which_entities_are_chosen() -> None:
    """The two losses must disagree on a population with spread directions.

    This is the test the pre-3.0 evaluation could not have passed. Its generator
    produced parallel influence vectors, on which every loss ranks entities
    identically -- so it reported all methods as equivalent and concluded they
    were.
    """
    counts, X = panel(n=600, m=400, p=3, seed=5)
    # Correlated features push (X'X)^-1 away from a scaled identity, which is
    # what makes the two quadratic forms rank entities differently.
    X = X.assign(x2=0.97 * X["x1"] + 0.24 * X["x2"])
    infl = influence(counts, X)
    a = contribution_scores(infl, loss="coefficient").rank(ascending=False)
    b = contribution_scores(infl, loss="prediction").rank(ascending=False)
    top_a = set(a.nsmallest(40).index)
    top_b = set(b.nsmallest(40).index)
    assert len(top_a & top_b) < 40, "the losses picked identical sets"


def test_a_contrast_targets_only_what_it_names() -> None:
    """Targeting one coefficient must reproduce that row's squared influence."""
    counts, X = panel(seed=6)
    infl = influence(counts, X)
    L = np.zeros(infl.n_features)
    L[1] = 1.0
    np.testing.assert_allclose(
        contribution_scores(infl, contrast=L).to_numpy(),
        infl.H[1] ** 2,
        rtol=1e-12,
    )


def test_label_norms_scale_the_scores() -> None:
    """`b_j` is linear in `a_j' a_j`, so doubling norms doubles scores."""
    counts, X = panel(m=60, seed=7)
    infl = influence(counts, X)
    base = contribution_scores(infl)
    doubled = contribution_scores(infl, label_norms=2.0)
    np.testing.assert_allclose(doubled.to_numpy(), 2 * base.to_numpy(), rtol=1e-12)


def test_the_spectrum_separates_rank_one_usage_from_a_real_panel() -> None:
    """The diagnostic must discriminate, at a matched count scale.

    An absolute cutoff would be wrong: integer counts inject spurious directions
    at roughly `1/sqrt(mean count)`, so the same population reads as more
    degenerate simply for being observed more heavily. What has to hold is that
    a rank-one population sits far below a genuinely multi-directional one.
    """
    flat, X_flat = rank_one(mean_count=20.0, seed=8)
    real, X_real = panel(seed=9)
    degenerate = direction_spectrum(influence(flat, X_flat)).iloc[1]
    genuine = direction_spectrum(influence(real, X_real)).iloc[1]
    assert degenerate < 0.05, f"rank-one population read as {degenerate:.3f}"
    assert genuine > 4 * degenerate, (
        f"diagnostic cannot separate them: {degenerate:.3f} vs {genuine:.3f}"
    )


def test_entities_nobody_touched_score_zero() -> None:
    """An all-zero column contributes nothing and must be scored as such."""
    counts, X = panel(m=40, seed=10)
    counts = counts.assign(dead=0.0)
    scores = contribution_scores(influence(counts, X))
    assert scores["dead"] == pytest.approx(0.0, abs=1e-20)


def test_an_unknown_loss_is_refused() -> None:
    """A misspelled estimand must fail loudly, not fall back to a default."""
    counts, X = panel(m=30, seed=11)
    with pytest.raises(ValueError, match="loss must be one of"):
        contribution_scores(influence(counts, X), loss="a-optimal")


def test_features_must_cover_every_unit() -> None:
    """Silently dropping units would change the estimand."""
    counts, X = panel(m=30, seed=12)
    with pytest.raises(ValueError, match="every unit"):
        influence(counts, X.iloc[:-5])


def test_a_column_of_labels_can_be_soft() -> None:
    """Probability-vector labels need no special handling."""
    counts, X = panel(m=50, seed=13)
    infl = influence(counts, X)
    rng = np.random.default_rng(0)
    soft = rng.dirichlet(np.ones(3), size=infl.n_items)
    A = pd.DataFrame(soft, index=infl.items, columns=["c0", "c1", "c2"])
    V = counts.to_numpy(float) / counts.sum(axis=1).to_numpy(float)[:, None]
    B_ols = np.linalg.lstsq(X.to_numpy(float), V @ A.to_numpy(float), rcond=None)[0]
    np.testing.assert_allclose(infl.H @ A.to_numpy(float), B_ols, atol=1e-10)

"""Does the sampler deliver the design it reports?

Everything fewlab claims rests on one identity. ``Design.sample`` returns a vector
of inclusion probabilities ``pi``, and every downstream estimator weights the
labelled items by ``1 / pi``. That makes the Horvitz-Thompson estimator unbiased
-- but only if the sampler actually includes item ``j`` with probability
``pi[j]``. Report a ``pi`` you do not deliver and the weights are wrong, the
standard errors are wrong, and ``row_se_min_labels`` solves for a design that
never happens.

Nothing tested that identity. The suite checked that the sample had the right
*length*, which a sampler ignoring ``pi`` entirely would also pass.

The tests here check the identity itself, at three levels:

* **Certainty items.** When ``pi[j] == 1`` the item must appear in every draw.
  This needs no Monte Carlo tolerance at all -- the binomial band around a
  nominal 1.0 is a single point -- so it is the sharpest statement available and
  the first thing to look at when this file goes red.
* **The whole vector.** Every item's empirical inclusion rate against its stated
  ``pi``, at a per-item level corrected for testing ``m`` of them at once.
* **The estimator.** Unbiasedness of ``calibrated_ht_estimator`` row by row,
  which is the property the inclusion probabilities exist to provide. Row by row
  rather than pooled, because a bias in one row cancelling another must not pass.

The last of those has a falsification partner: an estimator that ignores ``pi``
must be *caught*. Without it, an unbiasedness test that passes proves only that
the gate is silent.
"""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd
import pytest
from simcheck import Estimate, assert_proportion, assert_unbiased, monte_carlo, reps_for

from fewlab import (
    balanced_fixed_size,
    calibrated_ht_estimator,
    core_plus_tail,
    scale_pi_to_budget,
)
from fewlab.core import _influence, pi_aopt_for_budget

from .data_synth import make_synth

# Small enough that a few hundred draws stay quick, large enough that the
# A-optimal probabilities span the full range from pi_min to certainty.
N_UNITS, N_ITEMS, N_FEATURES, BUDGET = 60, 120, 5, 40

# An item whose stated probability is this close to one is a certainty item: it
# is not sampled at all, it is simply always in.
CERTAINTY = 1.0 - 1e-9


def _design(
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build one fixed design: counts, delivered probabilities, and projections.

    The probabilities returned are the ones the sampler *delivers*, which is
    ``scale_pi_to_budget`` of the A-optimal vector rather than that vector
    itself. The two differ only by however far ``pi_aopt_for_budget``'s solver
    lands from the budget, but asserting against the requested vector would be
    checking a target the sampler never promised, and would leave the gates
    passing for a reason unrelated to whether the design is right.

    Args:
        seed: Seed for the synthetic data generator.

    Returns:
        tuple: The count matrix, the delivered inclusion probabilities indexed by
        item, and the influence projections ``g`` in the same item order.
    """
    counts, X = make_synth(n=N_UNITS, m=N_ITEMS, p=N_FEATURES, random_state=seed)
    influence = _influence(counts, X)
    requested = pi_aopt_for_budget(counts, X, BUDGET).probabilities.reindex(
        influence.cols
    )
    delivered = pd.Series(
        scale_pi_to_budget(requested.to_numpy(float), BUDGET), index=requested.index
    )
    return counts, delivered, influence.g


def _inclusion_rates(
    pi: pd.Series, g: np.ndarray, reps: int, seed: int = 0
) -> pd.Series:
    """Empirical inclusion rate of every item over repeated draws.

    Args:
        pi: Stated inclusion probabilities.
        g: Influence projections aligned with ``pi``.
        reps: Number of independent draws.
        seed: Base seed; draw ``r`` uses ``seed + r``.

    Returns:
        pd.Series: Fraction of draws containing each item, indexed like ``pi``.
    """
    hits = pd.Series(0.0, index=pi.index)
    for rep in range(reps):
        hits.loc[balanced_fixed_size(pi, g, BUDGET, random_state=seed + rep)] += 1.0
    return hits / reps


def _per_test_sigmas(n_tests: int, family_alpha: float = 0.01) -> float:
    """Sigma giving family-wise error ``family_alpha`` over ``n_tests`` gates.

    Checking every item's inclusion rate means running ``m`` gates at once, and
    at the usual three sigmas a design that is perfectly correct still trips one
    of them now and again. The correction comes from the number of tests rather
    than from picking a looser number until the suite goes quiet.

    Args:
        n_tests: How many simultaneous gates are being applied.
        family_alpha: Probability that a correct design fails any of them.

    Returns:
        float: Sigmas to allow per test.
    """
    return statistics.NormalDist().inv_cdf(1.0 - family_alpha / n_tests / 2.0)


# --------------------------------------------------------------------------
# The design the sampler reports
# --------------------------------------------------------------------------


def test_certainty_items_are_always_selected():
    """An item with ``pi == 1`` is not sampled. It is always in.

    The sharpest statement in this file, and the one that needs no tolerance:
    the binomial band around a nominal rate of 1.0 has zero width, so a single
    draw omitting a certainty item is a failure with no appeal to noise.
    """
    _, pi, g = _design()
    certain = [item for item, stated in pi.items() if stated >= CERTAINTY]
    assert len(certain) > 0, "the fixture must contain certainty items to test"

    reps = reps_for()
    rates = _inclusion_rates(pi, g, reps)
    for item in certain:
        assert_proportion(
            float(rates.loc[item]), reps, 1.0, label=f"certainty item {item}"
        )


def test_every_inclusion_rate_matches_its_stated_probability():
    """The identity the Horvitz-Thompson weights depend on, item by item."""
    _, pi, g = _design()
    reps = reps_for()
    rates = _inclusion_rates(pi, g, reps)
    sigmas = _per_test_sigmas(len(pi))

    for item, stated in pi.items():
        assert_proportion(
            float(rates.loc[item]),
            reps,
            float(stated),
            label=f"item {item}",
            sigmas=sigmas,
        )


def test_the_sample_is_exactly_the_requested_size():
    """A fixed-size design must not vary its size across draws."""
    _, pi, g = _design()
    for rep in range(20):
        assert len(balanced_fixed_size(pi, g, BUDGET, random_state=rep)) == BUDGET


def test_the_stated_probabilities_sum_to_the_budget():
    """Fixed size and exact inclusion probabilities are consistent only if they do.

    ``sum(pi)`` is the expected sample size. If it did not equal the budget, no
    sampler could achieve both the stated probabilities and the fixed size, and
    the two tests above would be asserting contradictory things. It holds to
    floating point rather than to a solver tolerance because these are the
    delivered probabilities, which are rescaled to the budget by construction.
    """
    _, pi, _ = _design()
    assert float(pi.sum()) == pytest.approx(BUDGET, abs=1e-9)


# --------------------------------------------------------------------------
# The estimator those probabilities exist to support
# --------------------------------------------------------------------------


def _labels(pi: pd.Series, seed: int = 3) -> pd.Series:
    """A fixed binary label per item, held constant across replicates.

    The truth has to stay put while the *sample* varies. Redrawing labels each
    replicate would average over them too, and an estimator that ignored the
    design entirely could come out looking unbiased.

    Args:
        pi: Probabilities, used only for their index.
        seed: Seed for the label draw.

    Returns:
        pd.Series: A 0/1 label for every item.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 2, len(pi)).astype(float), index=pi.index)


def _true_share(counts: pd.DataFrame, labels: pd.Series, row: int) -> float:
    """The population quantity the estimator targets for one row.

    Args:
        counts: The count matrix.
        labels: The item labels.
        row: Positional row index.

    Returns:
        float: Share of row ``row``'s mass carried by labelled items.
    """
    c = counts.iloc[row]
    return float((c * labels.reindex(c.index)).sum() / c.sum())


@pytest.mark.parametrize("row", [0, 1, 2])
def test_horvitz_thompson_is_unbiased_row_by_row(row: int):
    """The estimator's defining property, checked per row rather than pooled."""
    counts, pi, g = _design()
    labels = _labels(pi)
    truth = _true_share(counts, labels, row)

    def replicate(rng: np.random.Generator) -> Estimate:
        sample = balanced_fixed_size(pi, g, BUDGET, random_state=rng)
        weights = 1.0 / pi.loc[sample]
        estimate = calibrated_ht_estimator(counts, labels.loc[sample], weights)
        return Estimate(value=float(estimate.iloc[row]))

    result = monte_carlo(replicate, truth, reps_for(), seed=row)
    assert_unbiased(result, label=f"calibrated_ht_estimator row {row}")


def test_ignoring_the_design_is_caught_as_biased():
    """The falsification partner. Without it the test above proves nothing.

    Weighting every sampled item by one, rather than by ``1 / pi``, discards the
    design. That estimator is genuinely biased, so the gate must fire on it -- if
    it stayed silent here it would be silent on a real bias too.
    """
    counts, pi, g = _design()
    labels = _labels(pi)
    truth = _true_share(counts, labels, 0)

    def replicate(rng: np.random.Generator) -> Estimate:
        sample = balanced_fixed_size(pi, g, BUDGET, random_state=rng)
        weights = pd.Series(1.0, index=sample)
        estimate = calibrated_ht_estimator(counts, labels.loc[sample], weights)
        return Estimate(value=float(estimate.iloc[0]))

    result = monte_carlo(replicate, truth, reps_for(), seed=99)
    with pytest.raises(AssertionError, match="Monte Carlo standard errors"):
        assert_unbiased(result, label="unweighted mean over sampled items")


# --------------------------------------------------------------------------
# Balance, which is the reason for the sampler over a plain draw
# --------------------------------------------------------------------------


def _balance_residual(pi: pd.Series, g: np.ndarray, sample: pd.Index) -> float:
    """Norm of ``sum_j (I_j / pi_j - 1) g_j``, the quantity balancing targets.

    Args:
        pi: Inclusion probabilities.
        g: Influence projections aligned with ``pi``.
        sample: The selected items.

    Returns:
        float: Euclidean norm of the calibration residual.
    """
    indicator = pi.index.isin(sample).astype(float)
    return float(np.linalg.norm(g @ (indicator / pi.to_numpy(float) - 1.0)))


def test_balancing_reduces_the_calibration_residual():
    """The sampler's whole purpose, and a guard on any future rewrite of it.

    Exact inclusion probabilities alone are easy -- a plain systematic draw gives
    them. What this sampler adds is that the projections come out balanced, and a
    change that quietly dropped the balancing would leave every other test in
    this file green.
    """
    _, pi, g = _design()
    probs = pi.to_numpy(float) / float(pi.sum())

    balanced, unbalanced = [], []
    for rep in range(20):
        rng = np.random.default_rng(rep)
        sample = balanced_fixed_size(pi, g, BUDGET, random_state=rep)
        balanced.append(_balance_residual(pi, g, sample))
        naive = pi.index.take(rng.choice(len(pi), size=BUDGET, replace=False, p=probs))
        unbalanced.append(_balance_residual(pi, g, naive))

    assert np.median(balanced) < 0.5 * np.median(unbalanced), (
        f"balanced residual {np.median(balanced):.4f} is not materially below "
        f"the unbalanced draw's {np.median(unbalanced):.4f}"
    )


# --------------------------------------------------------------------------
# The hybrid sampler, whose design is not the pi it optimises
# --------------------------------------------------------------------------


def test_core_plus_tail_reports_the_design_it_runs():
    """The core is chosen deterministically, so its inclusion probability is one.

    ``core_plus_tail`` picks its core by largest influence and only its tail at
    random, so the A-optimal ``pi`` it optimises over is not the design it
    executes. It used to report that ``pi`` anyway, which gave every core item a
    Horvitz-Thompson weight of ``1 / pi_j > 1`` for a selection that involved no
    randomness at all -- inflating exactly the items chosen for having the most
    influence.
    """
    counts, X = make_synth(n=N_UNITS, m=N_ITEMS, p=N_FEATURES, random_state=7)
    reps = reps_for()

    first = core_plus_tail(counts, X, BUDGET, tail_frac=0.25, random_state=0)
    stated = first.probabilities

    hits = pd.Series(0.0, index=stated.index)
    for rep in range(reps):
        result = core_plus_tail(counts, X, BUDGET, tail_frac=0.25, random_state=rep)
        hits.loc[result.selected] += 1.0
        assert list(result.core) == list(first.core), "the core must be deterministic"
    rates = hits / reps

    for item in first.core:
        assert stated.loc[item] == 1.0, f"core item {item} is not reported as certain"
        assert_proportion(float(rates.loc[item]), reps, 1.0, label=f"core item {item}")

    sigmas = _per_test_sigmas(len(stated))
    for item, claimed in stated.items():
        assert_proportion(
            float(rates.loc[item]),
            reps,
            float(claimed),
            label=f"core_plus_tail item {item}",
            sigmas=sigmas,
        )

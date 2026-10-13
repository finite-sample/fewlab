"""Probability designs: budgets, risk ordering, and the closed-form core size.

The designs are where unbiasedness is won or lost. A design that gives an
entity zero probability while it still contributes to the estimand cannot be
weighted back to unbiasedness afterward by anything, so the tests that matter
here are about which entities get positive probability and how much.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from fewlab import (
    Study,
    core_plus_tail,
    design_risk,
    poisson_pps,
    robust_pps,
    simple_random,
)
from tests.fixtures import panel

BUDGETS = [10, 50, 120]


@pytest.fixture(scope="module")
def scores() -> pd.Series:
    """Contribution scores from a realistic panel.

    Returns:
        The scores.
    """
    counts, X = panel(seed=21)
    return Study(counts, X).scores()


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("builder", [poisson_pps, core_plus_tail, simple_random])
def test_the_design_spends_exactly_its_budget(builder, budget, scores) -> None:
    """Reported probabilities must sum to what the caller asked to spend."""
    design = builder(scores, budget)
    assert design.budget == pytest.approx(budget, abs=1e-9)


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("builder", [poisson_pps, core_plus_tail, simple_random])
def test_every_contributing_entity_keeps_a_positive_chance(
    builder, budget, scores
) -> None:
    """Unbiasedness is exactly this property; nothing downstream restores it."""
    pi = builder(scores, budget).probabilities
    contributing = scores > 0
    assert (pi[contributing] > 0).all()
    assert (pi <= 1.0 + 1e-12).all()


@pytest.mark.parametrize("budget", BUDGETS)
def test_the_optimal_design_is_never_beaten_by_the_simpler_ones(budget, scores) -> None:
    """PPS minimizes the risk, so nothing else may come in below it."""
    pps = poisson_pps(scores, budget).risk
    cpt = core_plus_tail(scores, budget).risk
    srs = simple_random(scores, budget).risk
    assert pps <= cpt + 1e-9, f"core+tail beat the optimum: {cpt} < {pps}"
    assert cpt <= srs + 1e-9, f"SRS beat core+tail: {srs} < {cpt}"


@pytest.mark.parametrize("budget", [8, 25, 60])
def test_the_closed_form_core_size_matches_an_exhaustive_search(budget, scores) -> None:
    """`R(H)` must pick the same core an exhaustive scan would."""
    design = core_plus_tail(scores, budget)
    b = np.sort(scores[scores > 0].to_numpy())[::-1]
    m = b.size
    best, best_risk = None, np.inf
    for H in range(budget):
        pi = np.concatenate([np.ones(H), np.full(m - H, (budget - H) / (m - H))])
        risk = design_risk(b, pi)
        if risk < best_risk:
            best, best_risk = H, risk
    assert design.diagnostics["core_size"] == best
    assert design.risk == pytest.approx(best_risk, rel=1e-9)


def test_an_entity_nobody_touched_is_left_unlabeled(scores) -> None:
    """Zero contribution means zero probability costs zero bias."""
    padded = pd.concat([scores, pd.Series({"dead": 0.0})])
    for builder in (poisson_pps, core_plus_tail, simple_random):
        design = builder(padded, 40)
        assert design.probabilities["dead"] == 0.0


def test_a_costlier_label_is_bought_less_often(scores) -> None:
    """The rule is value per dollar, so cost must move probabilities."""
    cheap = poisson_pps(scores, 60).probabilities
    cost = pd.Series(1.0, index=scores.index)
    expensive = scores.index[int(np.argmax(scores.to_numpy()))]
    cost[expensive] = 100.0
    dear = poisson_pps(scores, 60, cost=cost).probabilities
    assert dear[expensive] < cheap[expensive]


def test_asking_for_more_than_exists_is_refused(scores) -> None:
    """Overspending the frame is a mistake, not something to silently clamp."""
    contributing = int((scores > 0).sum())
    with pytest.raises(ValueError, match="exceeds"):
        poisson_pps(scores, contributing + 1)
    with pytest.raises(ValueError, match="exceeds"):
        core_plus_tail(scores, contributing + 1)


def test_a_design_that_abandons_an_entity_is_scored_as_infinite_risk(
    scores,
) -> None:
    """The risk function must refuse to price a biased design as merely costly."""
    pi = poisson_pps(scores, 40).probabilities.copy()
    victim = scores.index[int(np.argmax(scores.to_numpy()))]
    pi[victim] = 0.0
    assert design_risk(scores, pi) == np.inf


def test_risk_falls_as_the_budget_rises(scores) -> None:
    """More labels can never buy a worse design."""
    risks = [poisson_pps(scores, b).risk for b in (20, 40, 80, 160)]
    assert all(a > b for a, b in pairwise(risks))


def test_the_certainty_core_is_produced_by_the_cap_not_chosen(scores) -> None:
    """Entities hit probability one because the constraint binds on them."""
    design = poisson_pps(scores, 80)
    certain = design.certain
    assert len(certain) > 0
    # Everything in the core outscores everything outside it.
    assert scores[certain].min() >= scores.drop(certain).max()


def test_the_robust_design_reduces_to_the_plain_one_at_full_prevalence(
    scores,
) -> None:
    """A proved identity, checked to catch implementation error rather than to
    establish the result.

    When every entity may be positive, `M = m`, and the sum of the `m` largest
    terms is the whole sum. The robust program is then literally the plain one,
    so the two worst-case risks must agree to numerical precision, not merely
    be close.
    """
    m = int((scores > 0).sum())
    plain = poisson_pps(scores, 40, pi_min=1e-4)
    robust = robust_pps(scores, 40, prevalence=1.0)

    def worst(design) -> float:
        t = ((1 - design.probabilities) / design.probabilities * scores).to_numpy()
        return float(np.sort(t)[::-1][:m].sum())

    assert worst(robust) == pytest.approx(worst(plain), rel=1e-6)


@pytest.mark.parametrize("prevalence", [0.02, 0.1, 0.5])
def test_the_robust_design_wins_on_the_case_it_optimizes(prevalence, scores) -> None:
    """It minimizes worst-case risk, so nothing else may beat it there."""
    m = int((scores > 0).sum())
    n_pos = max(1, int(np.ceil(prevalence * m)))
    plain = poisson_pps(scores, 40, pi_min=1e-4)
    robust = robust_pps(scores, 40, prevalence=prevalence)

    def worst(design) -> float:
        t = ((1 - design.probabilities) / design.probabilities * scores).to_numpy()
        return float(np.sort(t)[::-1][:n_pos].sum())

    assert worst(robust) <= worst(plain) + 1e-6


def test_the_robust_design_loses_on_the_case_it_does_not_optimize(scores) -> None:
    """The falsification partner: insurance has to cost something.

    If the robust design were also better on the all-positive risk, it would be
    dominating the optimum of that problem, which is impossible. A test that
    only checked the worst case would not notice an implementation that had
    quietly stopped being robust at all.
    """
    plain = poisson_pps(scores, 40, pi_min=1e-4)
    robust = robust_pps(scores, 40, prevalence=0.02)
    assert robust.risk > plain.risk


@pytest.mark.parametrize("prevalence", [0.05, 0.3])
def test_the_robust_design_spends_its_budget(prevalence, scores) -> None:
    """Convexity buys nothing if the constraint is not met."""
    design = robust_pps(scores, 40, prevalence=prevalence)
    assert design.budget == pytest.approx(40, abs=1e-6)
    assert (design.probabilities[scores > 0] > 0).all()


def test_an_impossible_prevalence_is_refused(scores) -> None:
    """A fraction outside (0, 1] describes no population."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="prevalence"):
            robust_pps(scores, 40, prevalence=bad)

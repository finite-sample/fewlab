"""The cube sampler, tested directly rather than through the estimators.

``test_sampling_design.py`` checks the properties that matter to a user of the
package. This file checks the machinery underneath on designs small enough to
hammer with tens of thousands of draws, where the Monte Carlo band is tight
enough to catch an error of a couple of percent.

The last test is the important one. It runs the *previous* algorithm through the
same gate the new one passes, and requires that the gate fire. Without it, every
other test here is consistent with a gate that cannot fail -- and the bug this
module was written to fix survived for two releases precisely because no test in
the suite was capable of noticing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from simcheck import assert_proportion, binomial_band

from fewlab.cube import _systematic_pps, cube_sample, scale_pi_to_budget

# Two certainty items and ten at 0.3, summing to exactly the budget. The
# certainty items are what make the falsification test sharp: any sampler that
# does not deliver pi will miss them sometimes, and the band around a nominal
# 1.0 has zero width.
BUDGET = 5
PI = np.array([1.0, 1.0] + [0.3] * 10)
REPS = 4000

# Sized for the number of items under test, so a correct sampler trips the suite
# about once in a hundred runs rather than once in eight.
SIGMAS = 3.9


def _fixed_size_only(n_items: int) -> np.ndarray:
    """The constraint matrix for a design with no balancing, only a fixed size.

    Args:
        n_items: Number of items.

    Returns:
        np.ndarray: A single row of ones.
    """
    return np.ones((1, n_items))


def _inclusion_rates(
    pi: np.ndarray, constraints: np.ndarray, reps: int, seed: int = 0
) -> np.ndarray:
    """Empirical inclusion rate per item over repeated cube draws.

    Args:
        pi: Target inclusion probabilities.
        constraints: Rows the sampler must preserve.
        reps: Number of draws.
        seed: Seed for the generator.

    Returns:
        np.ndarray: Fraction of draws containing each item.
    """
    rng = np.random.default_rng(seed)
    hits = np.zeros(pi.size)
    for _ in range(reps):
        hits += cube_sample(pi, constraints, rng)
    return hits / reps


# --------------------------------------------------------------------------
# scale_pi_to_budget
# --------------------------------------------------------------------------


def test_scaling_hits_the_budget_exactly():
    """The sum is the expected sample size, so it has to be the budget."""
    rng = np.random.default_rng(0)
    for n_items in (5, 20, 100):
        for budget in (1, n_items // 3, n_items - 1):
            scaled = scale_pi_to_budget(rng.random(n_items), budget)
            assert scaled.sum() == pytest.approx(budget, abs=1e-9)
            assert (scaled >= 0.0).all() and (scaled <= 1.0).all()


def test_scaling_pins_entries_that_would_exceed_one():
    """Scaling by a constant is not enough: a probability cannot exceed one.

    The large entry here would go to 2.5 under a plain rescale, so it is pinned
    at one and its surplus redistributed over the rest.
    """
    scaled = scale_pi_to_budget(np.array([0.5, 0.1, 0.1, 0.1]), 3)
    assert scaled[0] == 1.0
    assert scaled.sum() == pytest.approx(3.0, abs=1e-9)
    assert (scaled[1:] < 1.0).all()


def test_scaling_preserves_the_ordering_of_unpinned_entries():
    """Rescaling must not reorder who is more likely to be sampled."""
    values = np.array([0.05, 0.4, 0.2, 0.1])
    scaled = scale_pi_to_budget(values, 1)
    assert list(np.argsort(scaled)) == list(np.argsort(values))


def test_scaling_handles_the_degenerate_budgets():
    """An empty sample selects nothing; a full one selects everything."""
    values = np.array([0.3, 0.7, 0.2])
    assert (scale_pi_to_budget(values, 0) == 0.0).all()
    assert (scale_pi_to_budget(values, 3) == 1.0).all()


def test_scaling_rejects_an_impossible_budget():
    """More items than exist cannot be drawn, and saying so beats guessing."""
    with pytest.raises(ValueError, match="budget must be in"):
        scale_pi_to_budget(np.array([0.5, 0.5]), 3)
    with pytest.raises(ValueError, match="budget must be in"):
        scale_pi_to_budget(np.array([0.5, 0.5]), -1)


def test_scaling_rejects_a_budget_the_positive_entries_cannot_carry():
    """Zero-probability items are never selected, so they cannot fill a budget.

    Scaling leaves a zero at zero, so a budget larger than the number of positive
    entries is unsatisfiable. This used to return a vector summing to less than
    the budget, which produced a sample quietly smaller than the one requested.
    """
    with pytest.raises(ValueError, match="positive probability"):
        scale_pi_to_budget(np.zeros(4), 2)
    with pytest.raises(ValueError, match="positive probability"):
        scale_pi_to_budget(np.array([0.0, 0.0, 0.5]), 2)


def test_scaling_a_budget_the_positive_entries_can_just_carry():
    """The boundary case must work rather than trip the guard above."""
    scaled = scale_pi_to_budget(np.array([0.0, 0.2, 0.5]), 2)
    assert scaled.sum() == pytest.approx(2.0, abs=1e-9)
    assert scaled[0] == 0.0


# --------------------------------------------------------------------------
# cube_sample
# --------------------------------------------------------------------------


def test_the_sample_size_never_varies():
    """Fixed size is a constraint the walk preserves, not a tendency."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert cube_sample(PI, _fixed_size_only(PI.size), rng).sum() == BUDGET


def test_inclusion_probabilities_come_out_exact():
    """The property the whole module exists for."""
    rates = _inclusion_rates(PI, _fixed_size_only(PI.size), REPS)
    for item, (observed, stated) in enumerate(zip(rates, PI, strict=True)):
        assert_proportion(
            float(observed), REPS, float(stated), label=f"item {item}", sigmas=SIGMAS
        )


def test_balancing_constraints_do_not_disturb_the_probabilities():
    """Adding balance must not be paid for in bias.

    A sampler could hit any balance target by choosing items to suit it. The
    point of the cube method is that it does so while leaving the inclusion
    probabilities alone, so this runs the same check with a real balancing row
    in place.
    """
    rng = np.random.default_rng(2)
    auxiliary = rng.normal(size=PI.size)
    constraints = np.vstack([auxiliary / PI, np.ones(PI.size)])

    rates = _inclusion_rates(PI, constraints, REPS, seed=3)
    for item, (observed, stated) in enumerate(zip(rates, PI, strict=True)):
        assert_proportion(
            float(observed), REPS, float(stated), label=f"item {item}", sigmas=SIGMAS
        )


def test_balancing_actually_balances():
    """Balancing must buy a materially smaller residual than not balancing.

    Not zero, though. The flight phase holds the balancing row exactly, but it
    stalls once fewer coordinates remain unsettled than there are constraints,
    and the landing phase then drops that row to finish. So the residual is
    whatever the last couple of coordinates contribute.

    That makes the achievable ratio a function of design size, and this design is
    deliberately tiny: 12 items, of which two are certainties, leaves ten free
    coordinates and hands two of them to the landing phase. Measured ratio of
    medians here is 0.28, against 0.20 for the 120-item design in
    ``test_sampling_design.py``. The gate is set at 0.5 in both, which is a
    characterisation with room rather than a derived tolerance -- unlike the
    inclusion-probability gates above, there is no nominal value for this to be
    checked against.
    """
    rng = np.random.default_rng(4)
    auxiliary = rng.normal(size=PI.size)
    constraints = np.vstack([auxiliary / PI, np.ones(PI.size)])

    def residual(mask: np.ndarray) -> float:
        return abs(float(np.dot(auxiliary, mask / PI - 1.0)))

    balanced = [residual(cube_sample(PI, constraints, rng)) for _ in range(200)]
    plain = [
        residual(cube_sample(PI, _fixed_size_only(PI.size), rng)) for _ in range(200)
    ]
    assert np.median(balanced) < 0.5 * np.median(plain), (
        f"balanced residual {np.median(balanced):.4f} is not materially below "
        f"the unbalanced {np.median(plain):.4f}"
    )


def test_the_straggler_path_keeps_both_marginals_and_the_count():
    """The landing fallback must not trade the sample size for exactness.

    A coordinate can survive the flight and landing phases: probabilities that do
    not sum to an integer leave one over, and so, in principle, does accumulated
    floating-point drift in the preserved rows. Settling such coordinates
    independently would keep the marginals but let the count wander by one per
    straggler. Systematic pi-ps keeps both, and this checks it directly since the
    path is hard to provoke through ``cube_sample``.
    """
    probabilities = np.array([0.2, 0.5, 0.3, 0.6, 0.4])
    rng = np.random.default_rng(7)

    hits = np.zeros(probabilities.size)
    for _ in range(REPS):
        drawn = _systematic_pps(probabilities, rng)
        assert drawn.sum() == 2, "the count must be the total, not a per-item flip"
        hits += drawn
    rates = hits / REPS

    for item, (observed, stated) in enumerate(zip(rates, probabilities, strict=True)):
        assert_proportion(
            float(observed), REPS, float(stated), label=f"item {item}", sigmas=SIGMAS
        )


def test_certainty_items_are_never_missed():
    """A probability of one leaves nothing to sample."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        selected = cube_sample(PI, _fixed_size_only(PI.size), rng)
        assert selected[0] and selected[1]


# --------------------------------------------------------------------------
# The falsification test
# --------------------------------------------------------------------------


def test_the_gate_catches_the_algorithm_this_module_replaced():
    """The previous sampler must fail the test the new one passes.

    fewlab drew ``budget`` items without replacement with probabilities
    proportional to ``pi`` and called the result a design with inclusion
    probabilities ``pi``. It is not: a sequential draw conditions on what it has
    already taken, so an item that should always be included is not.

    If this test ever stops failing on that draw, the gates above have stopped
    measuring anything and the rest of this file is decoration.
    """
    rng = np.random.default_rng(6)
    probabilities = PI / PI.sum()

    hits = np.zeros(PI.size)
    for _ in range(REPS):
        drawn = rng.choice(PI.size, size=BUDGET, replace=False, p=probabilities)
        hits[drawn] += 1.0
    rates = hits / REPS

    # The certainty items are the sharpest case: they must be in every draw, and
    # this sampler leaves them out often enough to see immediately.
    assert rates[0] < 1.0, "the old draw must miss certainty items to be worth testing"
    with pytest.raises(AssertionError, match="outside"):
        assert_proportion(float(rates[0]), REPS, 1.0, label="certainty item")

    # And the failure is not confined to certainties: most of the vector is off.
    low, high = binomial_band(0.3, REPS, SIGMAS)
    outside = sum(1 for rate in rates[2:] if not low <= rate <= high)
    assert outside >= 5, (
        f"only {outside} of 10 non-certainty items fell outside their band; "
        "the old sampler was expected to miss badly"
    )

"""Fewest labels for a fixed standard error -- the dual of the budget problem.

`fewlab`'s title names both directions: smallest standard error for a fixed
budget, and fewest labels for a fixed standard error. This is the second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fewlab.rowse import row_se_min_labels


@pytest.fixture(scope="module")
def counts() -> pd.DataFrame:
    """The input from issue #21.

    Returns:
        A plain 500 by 50 Poisson count matrix.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.poisson(10, (500, 50)))


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3])
def test_ordinary_input_converges_within_the_default_budget(eps, counts) -> None:
    """Regression test for issue #21.

    The old dual step clipped the gradient at zero, so multipliers could only
    rise and the search stopped at whatever design first became feasible. On
    this input it never reached tolerance at all within the default iteration
    budget.
    """
    result = row_se_min_labels(counts, eps2=eps**2, return_result=True)
    assert result.feasible, f"max violation {result.max_violation:.2e}"
    assert result.max_violation <= result.tolerance


def test_the_returned_design_actually_meets_every_ceiling(counts) -> None:
    """Check the constraint directly rather than trusting the flag."""
    eps2 = 0.15**2
    pi = row_se_min_labels(counts, eps2=eps2).to_numpy()
    T = counts.sum(axis=1).to_numpy(float)
    q = (counts.to_numpy(float) / T[:, None]) ** 2
    variance = (q * (1.0 / pi - 1.0)).sum(axis=1)
    assert np.all(variance <= eps2 + 1e-6)


def test_a_looser_ceiling_never_costs_more_labels(counts) -> None:
    """The objective is the label count, so it must respond to the constraint."""
    previous = np.inf
    for eps in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        result = row_se_min_labels(counts, eps2=eps**2, return_result=True)
        assert result.expected_labels <= previous + 1e-6, (
            f"eps={eps} needed more labels than a tighter ceiling"
        )
        previous = result.expected_labels


def test_it_beats_labeling_everything(counts) -> None:
    """Labeling every entity is always feasible; the point is to do better."""
    result = row_se_min_labels(counts, eps2=0.2**2, return_result=True)
    assert result.expected_labels < counts.shape[1]


def test_an_impossible_tolerance_is_rejected(counts) -> None:
    """A non-positive variance ceiling is a mistake, not a hard problem."""
    with pytest.raises(Exception, match="positive"):
        row_se_min_labels(counts, eps2=0.0)

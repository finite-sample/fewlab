import numpy as np
import pandas as pd
import pytest

from fewlab import balanced_fixed_size, pi_aopt_for_budget, row_se_min_labels
from fewlab.validation import ValidationError

from .data_synth import make_synth


def test_pi_aopt_budget_sums():
    counts, X = make_synth(n=80, m=120, p=5, random_state=11)
    K = 30
    pi_result = pi_aopt_for_budget(counts, X, K, pi_min=1e-4)
    pi = pi_result.probabilities
    assert abs(pi.sum() - K) < 1e-2
    assert (pi.values >= 1e-4 - 1e-12).all() and (pi.values <= 1.0 + 1e-12).all()


def test_balanced_fixed_size_returns_K():
    counts, X = make_synth(n=60, m=100, p=4, random_state=7)
    K = 25
    # get g via influence path (private) using the pi function as shortcut
    from fewlab.core import _influence

    inf = _influence(counts, X)
    pi = pd.Series(np.clip(np.sqrt(np.maximum(inf.w, 0)), 1e-4, 1.0), index=inf.cols)
    sel = balanced_fixed_size(pi, inf.g, K, random_state=1)
    assert len(sel) == K
    assert set(sel).issubset(set(inf.cols))


def test_row_se_min_labels_basic():
    counts, X = make_synth(n=50, m=90, p=5, random_state=21)
    T = counts.sum(axis=1).to_numpy(float)
    sumq = ((counts.to_numpy(float) / np.maximum(T[:, None], 1e-12)) ** 2).sum(axis=1)
    # lax caps => tiny budget
    eps2 = 0.2 * ((1.0 / 1e-4) - 1.0) * sumq
    result = row_se_min_labels(
        counts, eps2, pi_min=1e-4, max_iter=2000, return_result=True
    )
    pi = result.probabilities
    assert (pi.values >= 1e-4).all() and (pi.values <= 1.0).all()
    assert result.feasible
    assert result.max_violation <= result.tolerance + 1e-9


def test_pi_aopt_budget_violation_warning():
    """Test that budget violation is detected and warned about."""
    counts, X = make_synth(n=50, m=100, p=3, random_state=42)

    # Set pi_min high enough that budget cannot be satisfied
    pi_min = 0.1
    n_items = counts.shape[1]
    min_possible_budget = n_items * pi_min  # 100 * 0.1 = 10

    # Request a budget that's impossible to achieve
    impossible_budget = 5  # Less than min_possible_budget

    # Should issue a warning
    with pytest.warns(UserWarning, match=f"Budget {impossible_budget} is infeasible"):
        result = pi_aopt_for_budget(counts, X, impossible_budget, pi_min=pi_min)

    # Check that all probabilities are set to pi_min
    assert (result.probabilities == pi_min).all()

    # Check that actual budget exceeds requested budget
    assert abs(result.budget_used - min_possible_budget) < 1e-10
    assert result.budget_used > impossible_budget

    # Check that violation info is in diagnostics
    assert "budget_violation" in result.diagnostics
    violation = result.diagnostics["budget_violation"]
    assert violation["requested_budget"] == impossible_budget
    assert violation["actual_budget"] == min_possible_budget
    assert violation["pi_min"] == pi_min
    assert violation["n_items"] == n_items


def test_row_se_min_labels_reports_infeasible():
    counts, _ = make_synth(n=20, m=30, p=3, random_state=123)
    eps2 = np.full(counts.shape[0], 1e-8)
    with pytest.warns(UserWarning, match="infeasible"):
        result = row_se_min_labels(
            counts,
            eps2,
            pi_min=0.2,
            max_iter=0,
            tol=1e-10,
            return_result=True,
        )
    assert not result.feasible
    assert result.max_violation > result.tolerance

    with pytest.raises(ValidationError):
        row_se_min_labels(
            counts,
            eps2,
            pi_min=0.2,
            max_iter=0,
            tol=1e-10,
            raise_on_failure=True,
        )

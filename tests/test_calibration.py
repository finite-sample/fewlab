"""
Tests for fewlab calibration and hybrid sampling methods.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from fewlab.constants import TOLERANCE_LOOSE


class TestCalibration:
    """Test calibration methods."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n, self.m, self.p = 100, 50, 4

        # Simple test data
        self.counts = pd.DataFrame(
            np.random.poisson(5, (self.n, self.m)),
            columns=[f"item_{j}" for j in range(self.m)],
        )
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.p), columns=[f"x{i}" for i in range(self.p)]
        )
        self.X["x0"] = 1  # Add intercept

        # Compute g matrix
        T = self.counts.sum(axis=1).to_numpy()
        V = self.counts.to_numpy() / T[:, None]
        self.g = self.X.to_numpy().T @ V  # (p, m)

        # Simple pi
        self.pi = pd.Series(0.5, index=self.counts.columns)

        # Select subset
        self.K = 20
        self.selected = self.counts.columns[: self.K]

    def test_calibrate_weights_basic(self):
        """Test basic weight calibration."""
        from fewlab.calibration import calibrate_weights

        weights = calibrate_weights(pi=self.pi, g=self.g, selected=self.selected)

        # Check shape and type
        assert isinstance(weights, pd.Series)
        assert len(weights) == self.K
        assert all(weights.index == self.selected)

        # Check weights are positive
        assert all(weights > 0)

        # Check calibration constraint (approximately)
        g_selected = self.g[:, : self.K]
        g_total = self.g.sum(axis=1)
        weighted_total = g_selected @ weights.to_numpy()
        assert_allclose(weighted_total, g_total, rtol=TOLERANCE_LOOSE * 5)

    def test_calibrate_weights_custom_totals(self):
        """Test calibration with custom population totals."""
        from fewlab.calibration import calibrate_weights

        # Custom totals - use realistic values based on the data structure
        # Scale the natural totals by a reasonable factor
        natural_totals = self.g.sum(axis=1)
        custom_totals = natural_totals * 1.2  # 20% increase

        weights = calibrate_weights(
            pi=self.pi, g=self.g, selected=self.selected, pop_totals=custom_totals
        )

        # Check calibration to custom totals
        g_selected = self.g[:, : self.K]
        weighted_total = g_selected @ weights.to_numpy()
        assert_allclose(weighted_total, custom_totals, rtol=TOLERANCE_LOOSE * 5)

    def test_calibrated_ht_estimator(self):
        """Test calibrated HT estimator."""
        from fewlab.calibration import calibrate_weights, calibrated_ht_estimator

        # Get calibrated weights
        weights = calibrate_weights(pi=self.pi, g=self.g, selected=self.selected)

        # Mock labels
        labels = pd.Series(np.random.binomial(1, 0.3, self.K), index=self.selected)

        # Compute estimate
        estimate = calibrated_ht_estimator(
            counts=self.counts, labels=labels, weights=weights
        )

        # Check output
        assert isinstance(estimate, pd.Series)
        assert len(estimate) == self.n
        assert all(estimate >= 0) & all(estimate <= 1)  # Shares should be in [0,1]

    def test_calibration_edge_cases(self):
        """Test edge cases in calibration."""
        from fewlab.calibration import calibrate_weights

        # Single item selected
        weights_single = calibrate_weights(
            pi=self.pi, g=self.g, selected=[self.counts.columns[0]]
        )
        assert len(weights_single) == 1

        # All items selected
        weights_all = calibrate_weights(
            pi=self.pi, g=self.g, selected=self.counts.columns
        )
        assert len(weights_all) == self.m

        # Very small ridge
        weights_small_ridge = calibrate_weights(
            pi=self.pi, g=self.g, selected=self.selected, ridge=1e-12
        )
        assert all(weights_small_ridge > 0)


class TestHybridMethods:
    """Test hybrid sampling methods."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(123)
        self.n, self.m, self.p = 100, 50, 4

        self.counts = pd.DataFrame(
            np.random.poisson(5, (self.n, self.m)),
            columns=[f"item_{j}" for j in range(self.m)],
        )
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.p), columns=[f"x{i}" for i in range(self.p)]
        )
        self.X["x0"] = 1

    def test_core_plus_tail_basic(self):
        """Test basic core+tail functionality."""
        from fewlab.hybrid import core_plus_tail

        budget = 20
        result = core_plus_tail(
            counts=self.counts, X=self.X, budget=budget, tail_frac=0.2, random_state=456
        )

        # Check structured result
        from fewlab.results import CoreTailResult

        assert isinstance(result, CoreTailResult)

        selected = result.selected
        pi = result.probabilities

        # Check outputs
        assert isinstance(selected, pd.Index)
        assert len(selected) == budget
        assert isinstance(pi, pd.Series)
        assert len(pi) == self.m

        # Check core and tail
        assert isinstance(result.core, pd.Index)
        assert isinstance(result.tail, pd.Index)
        assert isinstance(result.ht_weights, pd.Series)
        assert isinstance(result.mixed_weights, pd.Series)

        # Check split
        assert len(result.core) == result.budget_core
        assert len(result.tail) == result.budget_tail
        assert result.budget_core + result.budget_tail == budget

        # Core and tail should be disjoint
        assert len(result.core.intersection(result.tail)) == 0

        # Union should equal selected
        assert result.core.union(result.tail).equals(selected)

    def test_core_plus_tail_weights(self):
        """Test weight computation in core+tail."""
        from fewlab.hybrid import core_plus_tail

        result = core_plus_tail(counts=self.counts, X=self.X, budget=20, tail_frac=0.3)
        selected = result.selected

        # Standard HT weights
        weights_ht = result.ht_weights
        assert all(weights_ht > 0)
        assert len(weights_ht) == len(selected)

        # Mixed weights (tiny-bias)
        weights_mixed = result.mixed_weights
        assert len(weights_mixed) == len(selected)

        # Check mixed weights structure
        for item in result.core:
            assert weights_mixed[item] == weights_ht[item]  # Same as HT for core
        for item in result.tail:
            assert abs(weights_mixed[item] - 1.0) < 1e-6  # Should be 1.0 for tail

    def test_core_plus_tail_fractions(self):
        """Test different tail fractions."""
        from fewlab.hybrid import core_plus_tail

        budget = 30
        for tail_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = core_plus_tail(
                counts=self.counts, X=self.X, budget=budget, tail_frac=tail_frac
            )
            selected = result.selected

            expected_core = int((1 - tail_frac) * budget)
            expected_tail = budget - expected_core

            assert result.budget_core == expected_core
            assert result.budget_tail == expected_tail
            assert len(selected) == budget

    def test_adaptive_core_tail(self):
        """Test adaptive core+tail selection."""
        from fewlab.hybrid import adaptive_core_tail

        result = adaptive_core_tail(
            counts=self.counts,
            X=self.X,
            budget=25,
            min_tail_frac=0.1,
            max_tail_frac=0.4,
            random_state=789,
        )
        selected = result.selected
        pi = result.probabilities

        # Check basic outputs
        assert len(selected) == 25
        assert isinstance(pi, pd.Series)

        # Check adaptive info
        assert result.diagnostics["adaptive"] is True
        assert "condition_number" in result.diagnostics
        assert "concentration_ratio" in result.diagnostics
        assert "adaptive_tail_frac" in result.diagnostics

        # Check tail_frac is within bounds
        assert 0.1 <= result.diagnostics["adaptive_tail_frac"] <= 0.4

        # Check that it actually used the adaptive fraction
        actual_tail_frac = result.budget_tail / 25
        assert abs(actual_tail_frac - result.diagnostics["adaptive_tail_frac"]) < 0.05

    def test_edge_cases(self):
        """Test edge cases for hybrid methods."""
        from fewlab.hybrid import core_plus_tail

        # Invalid tail_frac
        with pytest.raises(ValueError, match=r"tail_frac must be in \(0, 1\)"):
            core_plus_tail(self.counts, self.X, budget=10, tail_frac=0)

        with pytest.raises(ValueError, match=r"tail_frac must be in \(0, 1\)"):
            core_plus_tail(self.counts, self.X, budget=10, tail_frac=1.0)

        # budget larger than m should raise ValidationError
        from fewlab.validation import ValidationError

        with pytest.raises(ValidationError):
            core_plus_tail(self.counts, self.X, budget=100, tail_frac=0.2)

        # Very small budget
        result = core_plus_tail(self.counts, self.X, budget=2, tail_frac=0.5)
        selected = result.selected
        assert len(selected) == 2
        assert result.budget_core == 1
        assert result.budget_tail == 1


class TestIntegration:
    """Test integration of calibration with hybrid methods."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(999)
        self.n, self.m, self.p = 200, 100, 5

        self.counts = pd.DataFrame(
            np.random.poisson(10, (self.n, self.m)),
            columns=[f"item_{j}" for j in range(self.m)],
        )
        self.X = pd.DataFrame(
            np.random.randn(self.n, self.p), columns=[f"x{i}" for i in range(self.p)]
        )
        self.X["x0"] = 1

        # Mock labels
        self.labels = pd.Series(
            np.random.binomial(1, 0.4, self.m), index=self.counts.columns
        )

    def test_calibrated_core_tail(self):
        """Test using calibration with core+tail selection."""
        from fewlab.calibration import calibrate_weights, calibrated_ht_estimator
        from fewlab.core import _influence
        from fewlab.hybrid import core_plus_tail

        budget = 40

        # Core+tail selection
        result = core_plus_tail(
            counts=self.counts, X=self.X, budget=budget, tail_frac=0.25
        )
        selected = result.selected
        pi = result.probabilities

        # Get g matrix
        inf = _influence(self.counts, self.X)
        g = inf.g

        # Calibrate weights
        calibrated_weights = calibrate_weights(pi=pi, g=g, selected=selected)

        # Compute estimates
        selected_labels = self.labels.loc[selected]

        # Calibrated estimate
        estimate_cal = calibrated_ht_estimator(
            counts=self.counts, labels=selected_labels, weights=calibrated_weights
        )

        # Standard HT estimate
        estimate_ht = calibrated_ht_estimator(
            counts=self.counts, labels=selected_labels, weights=result.ht_weights
        )

        # Both should produce valid estimates
        assert len(estimate_cal) == self.n
        assert len(estimate_ht) == self.n
        assert all(estimate_cal >= 0) & all(estimate_cal <= 1)
        assert all(estimate_ht >= 0) & all(estimate_ht <= 1)

        # Calibrated should satisfy constraint better
        g_selected = g[:, [self.counts.columns.get_loc(c) for c in selected]]
        g_total = g.sum(axis=1)

        cal_total = g_selected @ calibrated_weights.to_numpy()
        ht_total = g_selected @ result.ht_weights.to_numpy()

        cal_error = np.linalg.norm(cal_total - g_total)
        ht_error = np.linalg.norm(ht_total - g_total)

        # Calibrated should have smaller error
        assert cal_error < ht_error or np.isclose(cal_error, ht_error, rtol=1e-3)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

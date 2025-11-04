"""
Edge case tests for fewlab calibration and hybrid sampling.

Tests extreme conditions, boundary values, and numerical stability.
"""

import numpy as np
import pandas as pd
import pytest


class TestCalibrationEdgeCases:
    """Test calibration edge cases."""

    def test_single_item_selection(self):
        """Test calibration with only one selected item."""
        from fewlab.calibration import calibrate_weights

        n, m, p = 50, 20, 3
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))

        # Compute g matrix
        T = counts.sum(axis=1).to_numpy()
        V = counts.to_numpy() / T[:, None]
        g = X.to_numpy().T @ V

        pi = pd.Series(0.5, index=counts.columns)
        selected = [counts.columns[0]]  # Single item

        weights = calibrate_weights(pi, g, selected)
        assert len(weights) == 1
        assert weights.iloc[0] > 0

    def test_all_items_selected(self):
        """Test calibration when all items are selected."""
        from fewlab.calibration import calibrate_weights

        n, m, p = 50, 10, 3
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))

        T = counts.sum(axis=1).to_numpy()
        V = counts.to_numpy() / T[:, None]
        g = X.to_numpy().T @ V

        pi = pd.Series(0.8, index=counts.columns)
        selected = counts.columns  # All items

        weights = calibrate_weights(pi, g, selected)
        assert len(weights) == m
        assert all(weights > 0)

    def test_extreme_inclusion_probabilities(self):
        """Test with very small and very large inclusion probabilities."""
        from fewlab.calibration import calibrate_weights

        n, m, p = 50, 20, 3
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))

        T = counts.sum(axis=1).to_numpy()
        V = counts.to_numpy() / T[:, None]
        g = X.to_numpy().T @ V

        # Mix of extreme probabilities
        pi_values = np.concatenate(
            [
                np.full(5, 1e-6),  # Very small
                np.full(10, 0.5),  # Medium
                np.full(5, 0.999),  # Very large
            ]
        )
        pi = pd.Series(pi_values, index=counts.columns)
        selected = counts.columns[:10]

        weights = calibrate_weights(pi, g, selected)
        assert all(weights > 0)
        assert np.isfinite(weights).all()

    def test_nearly_singular_matrix(self):
        """Test calibration with nearly singular g matrix."""
        from fewlab.calibration import calibrate_weights

        n, m, p = 50, 20, 3
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )

        # Create nearly collinear X
        X = pd.DataFrame(np.random.randn(n, p))
        X.iloc[:, 1] = X.iloc[:, 0] * 0.999 + np.random.randn(n) * 0.001

        T = counts.sum(axis=1).to_numpy()
        V = counts.to_numpy() / T[:, None]
        g = X.to_numpy().T @ V

        pi = pd.Series(0.5, index=counts.columns)
        selected = counts.columns[:10]

        # Should handle with ridge regularization
        weights = calibrate_weights(pi, g, selected, ridge=1e-6)
        assert all(weights > 0)
        assert np.isfinite(weights).all()


class TestHybridEdgeCases:
    """Test hybrid sampling edge cases."""

    def test_extreme_tail_fractions(self):
        """Test with very small and very large tail fractions."""
        from fewlab.hybrid import core_plus_tail

        n, m, p = 100, 50, 4
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))
        K = 20

        # Very small tail (almost all deterministic)
        selected, pi, info = core_plus_tail(counts, X, K=K, tail_frac=0.05)
        assert len(selected) == K
        assert info["K_tail"] == 1
        assert info["K_core"] == 19

        # Very large tail (mostly probabilistic)
        selected, pi, info = core_plus_tail(counts, X, K=K, tail_frac=0.95)
        assert len(selected) == K
        assert info["K_tail"] == 19
        assert info["K_core"] == 1

    def test_budget_larger_than_items(self):
        """Test when K > m (budget exceeds item count)."""
        from fewlab.hybrid import core_plus_tail

        n, m, p = 100, 20, 4
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))

        # K larger than m
        selected, pi, info = core_plus_tail(counts, X, K=50, tail_frac=0.2)
        assert len(selected) <= m

    def test_insufficient_remainder_for_tail(self):
        """Test when not enough items remain for tail sampling."""
        from fewlab.hybrid import core_plus_tail

        n, m, p = 100, 25, 4
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )
        X = pd.DataFrame(np.random.randn(n, p))

        # Large K with small tail_frac might not leave enough for tail
        selected, pi, info = core_plus_tail(counts, X, K=24, tail_frac=0.1)
        assert len(selected) == 24
        # Should handle edge case gracefully

    def test_adaptive_with_extreme_conditions(self):
        """Test adaptive core+tail with extreme matrix conditions."""
        from fewlab.hybrid import adaptive_core_tail

        n, m, p = 100, 50, 4
        counts = pd.DataFrame(
            np.random.poisson(5, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )

        # Create poorly conditioned X
        X = pd.DataFrame(np.random.randn(n, p))
        X.iloc[:, 1] = X.iloc[:, 0] * 0.99

        selected, pi, info = adaptive_core_tail(counts, X, K=25)

        assert len(selected) == 25
        assert "adaptive_tail_frac" in info
        assert 0.1 <= info["adaptive_tail_frac"] <= 0.4

    def test_zero_variance_items(self):
        """Test with items that have zero variance."""
        from fewlab.hybrid import core_plus_tail

        n, m, p = 100, 30, 4
        counts = pd.DataFrame(
            np.ones((n, m)) * 5,  # Constant counts
            columns=[f"item_{j}" for j in range(m)],
        )
        # Add some variance to a few items
        counts.iloc[:, :5] = np.random.poisson(10, (n, 5))

        X = pd.DataFrame(np.random.randn(n, p))

        selected, pi, info = core_plus_tail(counts, X, K=15, tail_frac=0.3)

        assert len(selected) == 15
        assert all(pi > 0)


class TestNumericalStability:
    """Test numerical stability in extreme conditions."""

    def test_very_large_counts(self):
        """Test with very large count values."""
        from fewlab.calibration import calibrated_ht_estimator

        n, m = 50, 20
        # Very large counts
        counts = pd.DataFrame(
            np.random.poisson(1000, (n, m)), columns=[f"item_{j}" for j in range(m)]
        )

        labels = pd.Series(np.random.binomial(1, 0.3, m), index=counts.columns)
        weights = pd.Series(np.random.uniform(0.5, 2, m), index=counts.columns)

        estimate = calibrated_ht_estimator(counts, labels, weights)
        assert np.isfinite(estimate).all()
        assert all(0 <= estimate) & all(estimate <= 1)

    def test_very_small_counts(self):
        """Test with very small (near-zero) count values."""
        from fewlab.calibration import calibrated_ht_estimator

        n, m = 50, 20
        # Very small counts
        counts = pd.DataFrame(
            np.random.uniform(1e-6, 0.01, (n, m)),
            columns=[f"item_{j}" for j in range(m)],
        )

        labels = pd.Series(np.random.binomial(1, 0.3, m), index=counts.columns)
        weights = pd.Series(np.random.uniform(0.5, 2, m), index=counts.columns)

        estimate = calibrated_ht_estimator(counts, labels, weights)
        assert np.isfinite(estimate).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

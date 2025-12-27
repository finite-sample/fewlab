"""
Comprehensive tests for the Design class API.

Tests the primary object-oriented interface to fewlab functionality,
including cached computations, diagnostics, and method consistency.
"""

import numpy as np
import pandas as pd
import pytest

from fewlab import (
    CoreTailResult,
    Design,
    EstimationResult,
    ProbabilityResult,
    SamplingResult,
    SelectionResult,
)
from fewlab.validation import ValidationError


class TestDesignBasics:
    """Test basic Design class functionality."""

    def test_design_creation(self, basic_data):
        """Test basic Design instantiation."""
        counts, X = basic_data
        design = Design(counts, X)

        assert design.n_units == counts.shape[0]
        assert design.n_items == counts.shape[1]
        assert isinstance(design.influence_weights, pd.Series)
        assert len(design.influence_weights) == design.n_items
        assert isinstance(design.diagnostics, dict)

    def test_design_repr(self, basic_data):
        """Test Design string representation."""
        counts, X = basic_data
        design = Design(counts, X)

        repr_str = repr(design)
        assert "Design" in repr_str
        assert f"n_units={design.n_units}" in repr_str
        assert f"n_items={design.n_items}" in repr_str

    def test_ridge_auto_well_conditioned(self):
        """Test auto ridge selection for well-conditioned problems."""
        np.random.seed(42)
        n, m, p = 100, 50, 3
        counts = pd.DataFrame(np.random.poisson(5, (n, m)))
        X = pd.DataFrame(np.random.randn(n, p))

        design = Design(counts, X, ridge="auto")

        # Well-conditioned problems should not need ridge
        assert design.diagnostics["ridge"] is None
        assert design.diagnostics["ridge_reason"] == "auto (well-conditioned)"

    def test_ridge_auto_ill_conditioned(self):
        """Test auto ridge selection for ill-conditioned problems."""
        np.random.seed(42)
        n, m = 50, 30
        counts = pd.DataFrame(np.random.poisson(5, (n, m)))

        # Create rank-deficient X by making columns dependent
        X_base = pd.DataFrame(np.random.randn(n, 2))
        # Add a perfectly dependent column
        dependent_col = X_base.iloc[:, 0] + X_base.iloc[:, 1]
        X = pd.concat([X_base, dependent_col.to_frame(name=2)], axis=1)

        design = Design(counts, X, ridge="auto")

        # Should automatically add ridge
        assert design.diagnostics["ridge"] is not None
        assert design.diagnostics["ridge_reason"] == "auto (ill-conditioned)"

    def test_ridge_explicit(self, basic_data):
        """Test explicit ridge specification."""
        counts, X = basic_data
        ridge_val = 0.1

        design = Design(counts, X, ridge=ridge_val)

        assert design.diagnostics["ridge"] == ridge_val
        assert design.diagnostics["ridge_reason"] == "user-specified"

    def test_diagnostics_content(self, basic_data):
        """Test comprehensive diagnostics."""
        counts, X = basic_data
        design = Design(counts, X)

        diag = design.diagnostics

        # Required keys
        assert "condition_number" in diag
        assert "ridge" in diag
        assert "ridge_reason" in diag
        assert "original_shape" in diag
        assert "processed_shape" in diag
        assert "n_dropped_rows" in diag
        assert "n_dropped_cols" in diag

        # Shape tracking
        assert diag["original_shape"]["counts"] == counts.shape
        assert diag["original_shape"]["X"] == X.shape

        # No rows/columns should be dropped from clean test data
        assert diag["n_dropped_rows"] == 0
        assert diag["n_dropped_cols"] == 0


class TestDesignMethods:
    """Test Design class methods."""

    def test_select_deterministic(self, basic_data):
        """Test deterministic selection method."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 10

        result = design.select(budget, method="deterministic")

        assert isinstance(result, SelectionResult)
        assert len(result.selected) == budget
        assert result.selected.name == "selected_items"
        assert all(item in counts.columns for item in result.selected)

    def test_select_greedy(self, basic_data):
        """Test greedy selection method."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 5  # Smaller budget for faster greedy

        result = design.select(budget, method="greedy")

        assert isinstance(result, SelectionResult)
        assert len(result.selected) == budget
        assert result.selected.name == "selected_items"
        assert all(item in counts.columns for item in result.selected)

    def test_select_zero_budget(self, basic_data):
        """Test selection with zero budget."""
        counts, X = basic_data
        design = Design(counts, X)

        # Zero budget should be rejected in validation
        with pytest.raises(ValidationError, match="budget must be positive"):
            design.select(budget=0)

    def test_select_invalid_method(self, basic_data):
        """Test selection with invalid method."""
        counts, X = basic_data
        design = Design(counts, X)

        with pytest.raises(ValidationError, match="Unknown selection method"):
            design.select(budget=5, method="invalid")

    def test_inclusion_probabilities_aopt(self, basic_data):
        """Test A-optimal inclusion probabilities."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 20

        result = design.inclusion_probabilities(budget, method="aopt")

        assert isinstance(result, ProbabilityResult)
        assert len(result.probabilities) == design.n_items
        assert result.probabilities.name == "pi"
        assert (
            np.abs(result.probabilities.sum() - budget) < 1e-6
        )  # Should sum to budget
        assert (result.probabilities >= 0).all()
        assert (result.probabilities <= 1).all()

    def test_inclusion_probabilities_row_se(self, basic_data):
        """Test row SE inclusion probabilities."""
        counts, X = basic_data
        design = Design(counts, X)

        eps2 = 0.01
        result = design.inclusion_probabilities(budget=20, method="row_se", eps2=eps2)

        assert isinstance(result, ProbabilityResult)
        assert len(result.probabilities) == design.n_items
        assert (result.probabilities >= 0).all()
        assert (result.probabilities <= 1).all()

    def test_inclusion_probabilities_row_se_missing_eps2(self, basic_data):
        """Test row SE without required eps2 parameter."""
        counts, X = basic_data
        design = Design(counts, X)

        with pytest.raises(ValidationError, match="Must provide 'eps2'"):
            design.inclusion_probabilities(budget=20, method="row_se")

    def test_sample_balanced(self, basic_data):
        """Test balanced sampling."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 15

        result = design.sample(budget, method="balanced", random_state=42)

        assert isinstance(result, SamplingResult)
        assert len(result.sample) == budget
        assert result.sample.name == "sampled_items"
        assert all(item in counts.columns for item in result.sample)

    def test_sample_core_plus_tail(self, basic_data):
        """Test core+tail sampling."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 20
        tail_frac = 0.3

        result = design.sample(
            budget, method="core_plus_tail", tail_frac=tail_frac, random_state=42
        )

        assert isinstance(result, CoreTailResult)
        assert len(result.selected) == budget
        assert result.selected.name == "selected_items"

    def test_sample_adaptive(self, basic_data):
        """Test adaptive sampling."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 20

        result = design.sample(budget, method="adaptive", random_state=42)

        assert isinstance(result, CoreTailResult)
        assert len(result.selected) == budget
        assert result.selected.name == "selected_items"

    def test_sample_zero_budget(self, basic_data):
        """Test sampling with zero budget."""
        counts, X = basic_data
        design = Design(counts, X)

        # Zero budget should be rejected in validation
        with pytest.raises(ValidationError, match="budget must be positive"):
            design.sample(budget=0)

    def test_calibrate_weights(self, basic_data):
        """Test weight calibration."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 10

        result = design.select(budget)
        weights = design.calibrate_weights(result.selected)

        assert isinstance(weights, pd.Series)
        assert len(weights) == len(result.selected)
        assert weights.index.equals(result.selected)
        assert (weights > 0).all()  # Weights should be positive

    def test_estimate(self, basic_data):
        """Test estimation method."""
        counts, X = basic_data
        design = Design(counts, X)
        budget = 10

        select_result = design.select(budget)
        # Create binary labels for selected items
        labels = pd.Series(
            np.random.binomial(1, 0.5, len(select_result.selected)),
            index=select_result.selected,
        )

        result = design.estimate(select_result.selected, labels)

        assert isinstance(result, EstimationResult)
        assert len(result.estimates) == design.n_units
        assert result.estimates.index.equals(counts.index)
        assert (result.estimates >= 0).all()
        assert (result.estimates <= 1).all()  # Should be shares


class TestDesignConsistency:
    """Test consistency between Design methods and functional API."""

    def test_select_vs_items_to_label(self, basic_data):
        """Test Design.select() matches items_to_label()."""
        from fewlab import items_to_label

        counts, X = basic_data
        design = Design(counts, X)
        budget = 10

        design_result = design.select(budget, method="deterministic")
        func_result = items_to_label(counts, X, budget)

        assert design_result.selected.equals(func_result.selected)

    def test_inclusion_probabilities_vs_pi_aopt_for_budget(self, basic_data):
        """Test Design.inclusion_probabilities() matches pi_aopt_for_budget()."""
        from fewlab import pi_aopt_for_budget

        counts, X = basic_data
        design = Design(counts, X)
        budget = 15

        design_pi = design.inclusion_probabilities(budget, method="aopt").probabilities
        func_pi = pi_aopt_for_budget(counts, X, budget).probabilities

        assert np.allclose(design_pi.values, func_pi.values)
        assert design_pi.index.equals(func_pi.index)

    def test_cached_computations_efficiency(self, basic_data):
        """Test that multiple method calls reuse cached computations."""
        counts, X = basic_data
        design = Design(counts, X)

        # Get reference to cached influence
        cached_influence = design._influence

        # Multiple method calls should use same cached influence
        design.select(10)
        design.inclusion_probabilities(15)
        design.sample(8, random_state=42)

        # Should still be the same object (not recomputed)
        assert design._influence is cached_influence


class TestDesignEdgeCases:
    """Test Design class edge cases and error handling."""

    def test_misaligned_indices(self):
        """Test error on misaligned count and feature indices."""
        counts = pd.DataFrame(np.random.poisson(5, (50, 20)))
        X = pd.DataFrame(np.random.randn(50, 3))

        # Create explicit index mismatch
        counts.index = range(50)
        X.index = range(100, 150)  # Different indices

        with pytest.raises(
            ValidationError, match="counts and X have no common index values"
        ):
            Design(counts, X)

    def test_zero_totals_filtering(self):
        """Test automatic filtering of zero-total rows."""
        np.random.seed(42)
        counts = pd.DataFrame(np.random.poisson(5, (50, 20)))
        # Set some rows to zero
        counts.iloc[5] = 0
        counts.iloc[15] = 0
        X = pd.DataFrame(np.random.randn(50, 3), index=counts.index)

        design = Design(counts, X)

        # Should have fewer units after filtering
        assert design.n_units == 48  # 50 - 2 zero rows
        assert design.diagnostics["n_dropped_rows"] == 2

    def test_high_condition_warning(self):
        """Test warning for high condition number matrices."""
        np.random.seed(42)
        n, m = 50, 30
        counts = pd.DataFrame(np.random.poisson(5, (n, m)))

        # Create nearly singular X matrix
        X_base = pd.DataFrame(np.random.randn(n, 2), columns=["f0", "f1"])
        # Add almost linearly dependent column
        dependent_series = X_base.sum(axis=1) + 1e-10 * np.random.randn(n)
        X = pd.concat([X_base, dependent_series.to_frame("f2")], axis=1)

        design = Design(counts, X, ensure_full_rank=False)

        # Should have warning about high condition number
        assert "warnings" in design.diagnostics
        assert len(design.diagnostics["warnings"]) > 0
        assert "High condition number" in design.diagnostics["warnings"][0]

    def test_budget_validation(self, basic_data):
        """Test budget validation in methods."""
        counts, X = basic_data
        design = Design(counts, X)

        # Budget too large
        with pytest.raises(ValidationError, match="budget .* exceeds"):
            design.select(budget=1000)  # More items than available

        # Negative budget
        with pytest.raises(ValidationError, match="budget must be positive"):
            design.select(budget=-5)


@pytest.fixture
def basic_data():
    """Create basic test data for Design class tests."""
    np.random.seed(42)
    n, m, p = 100, 50, 3
    counts = pd.DataFrame(np.random.poisson(5, (n, m)))
    X = pd.DataFrame(np.random.randn(n, p))
    return counts, X

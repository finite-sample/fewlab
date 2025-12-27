"""
Primary Design class for optimal experimental design with cached computations.

This module provides the main object-oriented interface to fewlab functionality,
replacing the functional API with a stateful design that caches expensive
influence computations and provides comprehensive diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .constants import (
    CONDITION_THRESHOLD,
    PI_MIN_DEFAULT,
    SMALL_RIDGE,
)
from .core import _influence
from .validation import (
    ValidationError,
    validate_budget,
    validate_counts_matrix,
    validate_data_alignment,
    validate_features_matrix,
)

# Import result classes at the end to avoid circular imports
if TYPE_CHECKING:
    from .results import (
        CoreTailResult,
        EstimationResult,
        ProbabilityResult,
        SamplingResult,
        SelectionResult,
    )


class Design:
    """
    Primary interface for optimal experimental design with cached computations.

    The class stores processed data, cached influence matrices, and diagnostics so that repeated
    operations such as selection, sampling, and calibration can reuse expensive intermediate
    results.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from fewlab import Design
        >>>
        >>> counts = pd.DataFrame(np.random.poisson(5, (1000, 100)))
        >>> X = pd.DataFrame(np.random.randn(1000, 3))
        >>> design = Design(counts, X)
        >>> design.select(budget=20).shape[0]
        20
    """

    def __init__(
        self,
        counts: pd.DataFrame,
        X: pd.DataFrame,
        *,
        ridge: float | Literal["auto"] = "auto",
        ensure_full_rank: bool = True,
    ) -> None:
        """
        Initialize the design with validated data and cached influence computation.

        Args:
            counts: Count matrix with non-negative entries.
            X: Feature matrix aligned with `counts.index`.
            ridge: Ridge value or `"auto"` to infer it from conditioning.
            ensure_full_rank: Whether to add a ridge when `X^T X` is ill-conditioned.
        """
        # Store original inputs for diagnostics
        self._original_counts_shape = counts.shape
        self._original_X_shape = X.shape

        # Validate and preprocess inputs
        counts = validate_counts_matrix(counts, "counts")
        X = validate_features_matrix(X, "X")
        counts, X = validate_data_alignment(counts, X)

        # Store processed data
        self._counts = counts
        self._X = X
        self._ridge_param = ridge
        self._ensure_full_rank = ensure_full_rank

        # Compute ridge value and diagnostics
        self._compute_diagnostics()

        # Compute and cache influence matrix
        self._influence = _influence(
            self._counts,
            self._X,
            ensure_full_rank=ensure_full_rank,
            ridge=self._ridge_value,
        )

        # Create convenience properties
        self._influence_weights = pd.Series(
            self._influence.w, index=self._influence.cols, name="influence_weights"
        )

    def _compute_diagnostics(self) -> None:
        """Compute diagnostic information about the design."""
        X_array = self._X.to_numpy(dtype=float)
        XtX = X_array.T @ X_array

        # Condition number
        condition_number = np.linalg.cond(XtX)

        # Determine ridge value
        if self._ridge_param == "auto":
            if self._ensure_full_rank and (
                not np.isfinite(condition_number)
                or condition_number > CONDITION_THRESHOLD
            ):
                ridge_value = SMALL_RIDGE
                ridge_reason = "auto (ill-conditioned)"
            else:
                ridge_value = None
                ridge_reason = "auto (well-conditioned)"
        else:
            ridge_value = float(self._ridge_param)
            ridge_reason = "user-specified"

        self._ridge_value = ridge_value

        # Build diagnostics dictionary
        self._diagnostics = {
            "condition_number": condition_number,
            "ridge": ridge_value,
            "ridge_reason": ridge_reason,
            "original_shape": {
                "counts": self._original_counts_shape,
                "X": self._original_X_shape,
            },
            "processed_shape": {
                "counts": self._counts.shape,
                "X": self._X.shape,
            },
            "n_dropped_rows": self._original_counts_shape[0] - self._counts.shape[0],
            "n_dropped_cols": self._original_counts_shape[1] - self._counts.shape[1],
        }

        # Add warnings for numerical issues
        if condition_number > CONDITION_THRESHOLD:
            self._diagnostics["warnings"] = self._diagnostics.get("warnings", [])
            self._diagnostics["warnings"].append(
                f"High condition number ({condition_number:.2e}) may indicate numerical issues"
            )

    @property
    def n_units(self) -> int:
        """Number of units (rows) after preprocessing."""
        return self._counts.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items (columns) after preprocessing."""
        return self._counts.shape[1]

    @property
    def influence_weights(self) -> pd.Series:
        """A-optimal influence weights w_j for each item."""
        return self._influence_weights.copy()

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Comprehensive diagnostic information about the design."""
        return self._diagnostics.copy()

    def select(
        self, budget: int, method: Literal["deterministic", "greedy"] = "deterministic"
    ) -> SelectionResult:
        """
        Select items using deterministic algorithms.

        Args:
            budget: Number of items to select.
            method: Selection algorithm: `"deterministic"` (batch) or `"greedy"` (sequential).

        Returns:
            Selection result with items, influence weights, and diagnostics.

        Raises:
            ValidationError: If the method name is unknown.
        """
        from .results import SelectionResult

        budget = validate_budget(budget, self.n_items, "budget")

        if budget == 0:
            empty_selected = pd.Index([], name="selected_items")
            empty_weights = pd.Series([], dtype=float, name="influence_weights")
            return SelectionResult(
                selected=empty_selected,
                influence_weights=empty_weights,
                diagnostics={"method": method, "budget": budget},
            )

        if method == "deterministic":
            return self._select_deterministic(budget)
        elif method == "greedy":
            return self._select_greedy(budget)
        else:
            raise ValidationError(
                f"Unknown selection method: {method}", "Use 'deterministic' or 'greedy'"
            )

    def _select_deterministic(self, budget: int) -> SelectionResult:
        """Deterministic A-optimal selection (equivalent to items_to_label)."""
        from .results import SelectionResult
        from .selection import topk

        items_index = pd.Index(self._influence.cols)
        selected_items = topk(self._influence.w, budget, index=items_index)
        selected_items.name = "selected_items"

        diagnostics = {
            "method": "deterministic",
            "budget": budget,
        }

        return SelectionResult(
            selected=selected_items,
            influence_weights=self._influence_weights,
            diagnostics=diagnostics,
        )

    def _select_greedy(self, budget: int) -> SelectionResult:
        """Greedy sequential A-optimal selection."""
        from .greedy import greedy_aopt_selection

        # Use existing greedy function and return its result directly
        return greedy_aopt_selection(
            self._counts,
            self._X,
            budget,
            ensure_full_rank=self._ensure_full_rank,
            ridge=self._ridge_value,
        )

    def inclusion_probabilities(
        self,
        budget: int,
        *,
        pi_min: float = PI_MIN_DEFAULT,
        method: Literal["aopt", "row_se"] = "aopt",
        **kwargs: Any,
    ) -> ProbabilityResult:
        """
        Compute inclusion probabilities for a given budget.

        Args:
            budget: Expected total budget (sum of inclusion probabilities).
            pi_min: Minimum inclusion probability per item.
            method: Probability computation strategy, `"aopt"` or `"row_se"`.
            \\*\\*kwargs: Additional method-specific arguments (e.g., `eps2` for `"row_se"`).

        Returns:
            Probability result with inclusion probabilities and diagnostics.

        Raises:
            ValidationError: If the method name is unknown.
        """
        from .results import ProbabilityResult

        budget = validate_budget(budget, self.n_items, "budget")

        if method == "aopt":
            probabilities = self._inclusion_probabilities_aopt(budget, pi_min)
            diagnostics = {"method": "aopt", "budget": budget, "pi_min": pi_min}
            # Add budget violation info if present
            if hasattr(self, "_last_budget_violation") and self._last_budget_violation:
                diagnostics["budget_violation"] = self._last_budget_violation
        elif method == "row_se":
            rowse_result = self._inclusion_probabilities_row_se(
                budget, pi_min, **kwargs
            )
            probabilities = rowse_result.probabilities
            diagnostics = {"method": "row_se", "budget": budget, "pi_min": pi_min}
            diagnostics.update(kwargs)
            diagnostics.update(rowse_result.to_dict())
        else:
            raise ValidationError(
                f"Unknown probability method: {method}", "Use 'aopt' or 'row_se'"
            )

        return ProbabilityResult(
            probabilities=probabilities,
            influence_projections=self._influence.g,
            diagnostics=diagnostics,
        )

    def _inclusion_probabilities_aopt(self, budget: int, pi_min: float) -> pd.Series:
        """A-optimal inclusion probabilities (equivalent to pi_aopt_for_budget)."""
        import warnings

        from .constants import (
            BINARY_SEARCH_HI,
            BINARY_SEARCH_LO,
            MAX_ITER_BINARY_SEARCH,
        )

        sqrtw = np.sqrt(np.maximum(self._influence.w, 0.0))
        if budget <= 0:
            return pd.Series(
                np.full_like(sqrtw, pi_min), index=self._influence.cols, name="pi"
            )

        m = sqrtw.size
        min_possible_budget = m * pi_min

        # Check if budget is feasible given pi_min constraint
        if budget < min_possible_budget:
            warnings.warn(
                f"Budget {budget} is infeasible with pi_min={pi_min:.3e} for {m} items. "
                f"Minimum possible budget is {min_possible_budget:.2f}. "
                f"Returning all probabilities as pi_min, which gives sum(pi)={min_possible_budget:.2f}.",
                UserWarning,
                stacklevel=4,
            )
            # Store violation info in diagnostics (will be added to result later)
            self._last_budget_violation = {
                "requested_budget": budget,
                "actual_budget": min_possible_budget,
                "pi_min": pi_min,
                "n_items": m,
            }
            return pd.Series(
                np.full_like(sqrtw, pi_min), index=self._influence.cols, name="pi"
            )

        # Clear any previous violation
        self._last_budget_violation = None
        budget = min(budget, m)

        def sum_pi(c: float) -> tuple[float, np.ndarray]:
            pi_array = np.clip(c * sqrtw, pi_min, 1.0)
            return pi_array.sum(), pi_array

        lo = BINARY_SEARCH_LO
        hi = BINARY_SEARCH_HI
        for _ in range(MAX_ITER_BINARY_SEARCH):
            c = (lo * hi) ** 0.5
            s, _ = sum_pi(c)
            if s > budget:
                hi = c
            else:
                lo = c
        _, pi_array = sum_pi(hi)
        return pd.Series(pi_array, index=self._influence.cols, name="pi")

    def _inclusion_probabilities_row_se(
        self, budget: int, pi_min: float, **kwargs: Any
    ):
        """
        Row-wise SE constrained probabilities (equivalent to `row_se_min_labels`).

        Args:
            budget: Expected total budget.
            pi_min: Minimum allowable inclusion probability.
            \\*\\*kwargs: Additional arguments; must include `eps2` (row-wise SE^2 constraints).

        Returns:
            Inclusion probabilities that satisfy the row-wise SE constraints.

        Raises:
            ValidationError: If the required `eps2` parameter is missing.
        """
        from .rowse import row_se_min_labels

        eps2 = kwargs.get("eps2")
        if eps2 is None:
            raise ValidationError(
                "Must provide 'eps2' parameter for row_se method",
                "Specify eps2=<float> for SE^2 tolerance per row",
            )

        return row_se_min_labels(
            self._counts,
            eps2,
            pi_min=pi_min,
            return_result=True,
            **{k: v for k, v in kwargs.items() if k != "eps2"},
        )

    def sample(
        self,
        budget: int,
        method: Literal["balanced", "core_plus_tail", "adaptive"] = "balanced",
        *,
        random_state: None | int | np.random.Generator = None,
        **kwargs: Any,
    ) -> SamplingResult | CoreTailResult:
        """
        Generate probabilistic samples using various methods.

        Args:
            budget: Number of items to sample.
            method: Sampling method (`"balanced"`, `"core_plus_tail"`, or `"adaptive"`).
            random_state: Random state for reproducible sampling. Can be None, int, or Generator.
            \\*\\*kwargs: Method-specific parameters (e.g., `tail_frac`, `pi_min`, tolerances).

        Returns:
            Sampled item identifiers.

        Raises:
            ValidationError: If the method name is unknown.
        """
        budget = validate_budget(budget, self.n_items, "budget")

        if budget == 0:
            from .results import CoreTailResult, SamplingResult

            empty_selected = pd.Index([], name="sampled_items")
            empty_pi = pd.Series([], dtype=float, name="pi")
            empty_weights = pd.Series([], dtype=float, name="weights")

            if method == "balanced":
                return SamplingResult(
                    sample=empty_selected,
                    probabilities=empty_pi,
                    weights=empty_weights,
                    diagnostics={"method": method, "budget": budget},
                )
            else:  # core_plus_tail or adaptive
                return CoreTailResult(
                    selected=empty_selected,
                    probabilities=empty_pi,
                    core=empty_selected,
                    tail=empty_selected,
                    ht_weights=empty_weights,
                    mixed_weights=empty_weights,
                    diagnostics={"method": method, "budget": budget},
                )

        if method == "balanced":
            return self._sample_balanced(budget, random_state=random_state, **kwargs)
        elif method == "core_plus_tail":
            return self._sample_core_plus_tail(
                budget, random_state=random_state, **kwargs
            )
        elif method == "adaptive":
            return self._sample_adaptive(budget, random_state=random_state, **kwargs)
        else:
            raise ValidationError(
                f"Unknown sampling method: {method}",
                "Use 'balanced', 'core_plus_tail', or 'adaptive'",
            )

    def _sample_balanced(
        self,
        budget: int,
        *,
        random_state: None | int | np.random.Generator = None,
        pi_min: float = PI_MIN_DEFAULT,
        **kwargs: Any,
    ) -> SamplingResult:
        """Balanced fixed-size sampling."""
        from .balanced import balanced_fixed_size
        from .results import SamplingResult
        from .utils import compute_horvitz_thompson_weights

        # Compute A-optimal probabilities
        pi = self._inclusion_probabilities_aopt(budget, pi_min)

        # Use cached g matrix
        sample = balanced_fixed_size(
            pi, self._influence.g, budget, random_state=random_state, **kwargs
        )
        sample.name = "sampled_items"

        # Compute suggested weights for sampled items
        weights = compute_horvitz_thompson_weights(pi, sample)

        diagnostics = {
            "method": "balanced",
            "budget": budget,
            "pi_min": pi_min,
            "random_state": random_state,
        }
        diagnostics.update(kwargs)

        return SamplingResult(
            sample=sample, probabilities=pi, weights=weights, diagnostics=diagnostics
        )

    def _sample_core_plus_tail(
        self,
        budget: int,
        *,
        tail_frac: float = 0.2,
        random_state: None | int | np.random.Generator = None,
        **kwargs: Any,
    ) -> CoreTailResult:
        """
        Hybrid core+tail sampling.

        Args:
            budget: Total sample size.
            tail_frac: Fraction allocated to the probabilistic tail.
            random_state: Random state for the tail sampling step. Can be None, int, or Generator.
            \\*\\*kwargs: Extra arguments forwarded to the balanced sampler.

        Returns:
            Core+tail result with selected items, probabilities, weights, and diagnostics.
        """
        from .hybrid import core_plus_tail

        # Use the hybrid function directly, which returns CoreTailResult
        return core_plus_tail(
            self._counts,
            self._X,
            budget,
            tail_frac=tail_frac,
            random_state=random_state,
            ensure_full_rank=self._ensure_full_rank,
            ridge=self._ridge_value,
            **kwargs,
        )

    def _sample_adaptive(
        self,
        budget: int,
        *,
        min_tail_frac: float = 0.1,
        max_tail_frac: float = 0.4,
        condition_threshold: float = 1e6,
        random_state: None | int | np.random.Generator = None,
        **kwargs: Any,
    ) -> CoreTailResult:
        """Adaptive core+tail with data-driven tail fraction."""
        from .hybrid import adaptive_core_tail

        # Use the hybrid function directly, which returns CoreTailResult
        return adaptive_core_tail(
            self._counts,
            self._X,
            budget,
            min_tail_frac=min_tail_frac,
            max_tail_frac=max_tail_frac,
            condition_threshold=condition_threshold,
            random_state=random_state,
            **kwargs,
        )

    def calibrate_weights(
        self,
        selected: pd.Index | list[str],
        pop_totals: np.ndarray | None = None,
        *,
        distance: str = "chi2",
        ridge: float = SMALL_RIDGE,
        nonneg: bool = True,
    ) -> pd.Series:
        """
        Compute calibrated weights for selected items.

        Args:
            selected: Identifiers of sampled items.
            pop_totals: Optional population totals; defaults to sums of the `g` matrix.
            distance: Calibration distance measure (e.g., `"chi2"`).
            ridge: Ridge regularization parameter.
            nonneg: Whether to enforce non-negative calibrated weights.

        Returns:
            Calibrated weights indexed by the selected items.
        """
        from .calibration import calibrate_weights

        # Compute A-optimal probabilities if needed
        pi = self._inclusion_probabilities_aopt(len(selected), PI_MIN_DEFAULT)

        return calibrate_weights(
            pi,
            self._influence.g,
            selected,
            pop_totals,
            distance=distance,
            ridge=ridge,
            nonneg=nonneg,
        )

    def estimate(
        self,
        selected: pd.Index | list[str],
        labels: pd.Series,
        weights: pd.Series | None = None,
        *,
        normalize_by_total: bool = True,
    ) -> EstimationResult:
        """
        Compute calibrated Horvitz-Thompson estimates for row shares.

        Args:
            selected: Identifiers of sampled items.
            labels: Observed labels for the selected items.
            weights: Optional calibrated weights; if omitted they are computed internally.
            normalize_by_total: Whether to divide by row totals to produce shares.

        Returns:
            Estimation result with estimates, weights, and diagnostics.
        """
        from .calibration import calibrated_ht_estimator
        from .results import EstimationResult

        if weights is None:
            weights = self.calibrate_weights(selected)

        estimates = calibrated_ht_estimator(
            self._counts, labels, weights, normalize_by_total=normalize_by_total
        )

        selected_index = (
            selected if isinstance(selected, pd.Index) else pd.Index(selected)
        )

        diagnostics = {
            "normalize_by_total": normalize_by_total,
            "n_selected": len(selected_index),
            "n_labeled": len(labels),
        }

        return EstimationResult(
            estimates=estimates,
            weights=weights,
            selected=selected_index,
            diagnostics=diagnostics,
        )

    def __repr__(self) -> str:
        """String representation of Design object."""
        ridge_str = f"{self._ridge_value:.2e}" if self._ridge_value else "None"
        return (
            f"Design(n_units={self.n_units}, n_items={self.n_items}, "
            f"condition_number={self.diagnostics['condition_number']:.2e}, "
            f"ridge={ridge_str})"
        )

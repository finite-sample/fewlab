"""
Fewlab: Optimal item selection for efficient labeling and survey sampling.

Main API functions:
- items_to_label: Deterministic A-optimal selection
- pi_aopt_for_budget: A-optimal inclusion probabilities
- balanced_fixed_size: Balanced sampling with fixed size
- row_se_min_labels: Row-wise SE minimization
- calibrate_weights: GREG-style weight calibration
- core_plus_tail: Hybrid deterministic core + balanced tail
- adaptive_core_tail: Data-driven hybrid selection
"""

from importlib.metadata import version

from .balanced import balanced_fixed_size
from .calibration import calibrate_weights, calibrated_ht_estimator
from .core import items_to_label, pi_aopt_for_budget
from .design import Design
from .greedy import greedy_aopt_selection
from .hybrid import adaptive_core_tail, core_plus_tail
from .results import (
    CoreTailResult,
    EstimationResult,
    ProbabilityResult,
    RowSEResult,
    SamplingResult,
    SelectionResult,
)
from .rowse import row_se_min_labels
from .selection import topk

__version__ = version("fewlab")

__all__ = [
    # Primary interface
    "Design",
    # Core methods
    "items_to_label",
    "pi_aopt_for_budget",
    "balanced_fixed_size",
    "row_se_min_labels",
    "topk",
    "calibrate_weights",
    "calibrated_ht_estimator",
    "core_plus_tail",
    "adaptive_core_tail",
    "greedy_aopt_selection",
    "CoreTailResult",
    "SamplingResult",
    "SelectionResult",
    "ProbabilityResult",
    "EstimationResult",
    "RowSEResult",
]

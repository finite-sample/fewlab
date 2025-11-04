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

from .core import items_to_label, pi_aopt_for_budget
from .balanced import balanced_fixed_size
from .rowse import row_se_min_labels
from .selection import topk
from .calibration import calibrate_weights, calibrated_ht_estimator
from .hybrid import core_plus_tail, adaptive_core_tail

__version__ = "0.3.1"

__all__ = [
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
]

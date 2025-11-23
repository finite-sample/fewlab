"""
Numerical constants and algorithm parameters for fewlab.

This module centralizes all magic numbers used throughout the codebase,
providing clear documentation for their purpose and mathematical significance.
"""

# Numerical stability epsilons
SMALL_RIDGE: float = (
    1e-8  # Small ridge value added to X^T X when matrix is ill-conditioned.
)

DIVISION_EPS: float = (
    1e-18  # Small epsilon added to denominators to prevent division by zero.
)

TOLERANCE_STRICT: float = 1e-12  # Very strict tolerance for numerical comparisons.

TOLERANCE_DEFAULT: float = (
    1e-6  # Default tolerance for convergence checks and optimizations.
)

TOLERANCE_LOOSE: float = 1e-2  # Loose tolerance for final validation checks.

# Condition number thresholds
CONDITION_THRESHOLD: float = (
    1e12  # Maximum acceptable condition number for matrix inversion.
)

# Algorithm parameters
MAX_SWAPS_BALANCED: int = (
    5_000  # Maximum number of swaps allowed in balanced sampling optimization.
)

MAX_ITER_ROWSE: int = (
    8_000  # Maximum iterations for row-wise standard error minimization.
)

MAX_ITER_BINARY_SEARCH: int = (
    100  # Maximum iterations for binary search in pi_aopt_for_budget.
)

SEARCH_LIMIT_BALANCED: int = (
    256  # Maximum candidates to try in each direction during balanced sampling.
)

# Default probability bounds
PI_MIN_DEFAULT: float = (
    1e-4  # Default minimum inclusion probability to prevent zero probabilities.
)

DUAL_START_VALUE: float = (
    1e-4  # Initial value for dual variables in optimization algorithms.
)

NOISE_SCALE: float = (
    1e-6  # Scale of random noise added to prevent cycling in iterative algorithms.
)

# Binary search bounds
BINARY_SEARCH_LO: float = (
    1e-12  # Lower bound for binary search in probability optimization.
)

BINARY_SEARCH_HI: float = (
    1e12  # Upper bound for binary search in probability optimization.
)

# Normalization epsilon
NORMALIZATION_EPS: float = (
    1e-9  # Small value added to standard deviation for data normalization.
)

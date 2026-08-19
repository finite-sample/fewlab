"""Fixed-size balanced sampling via the cube method.

Draws a sample of exactly the requested size while honouring the given
inclusion probabilities and keeping the regression projections balanced.
"""

from typing import cast

import numpy as np
import pandas as pd

from .constants import DIVISION_EPS
from .cube import cube_sample, scale_pi_to_budget
from .utils import _get_random_generator
from .validation import (
    ValidationError,
    validate_budget,
    validate_probability_series,
)


def balanced_fixed_size(
    pi: pd.Series,
    g: np.ndarray,
    budget: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> pd.Index:
    """Fixed-size balanced sampling with exact inclusion probabilities.

    Draws exactly `budget` items by the cube method, so that three things hold at
    once: item j is included with probability exactly `pi_delivered[j]`, the sample
    size never varies, and the calibration residual `sum_S (I_j/pi_j - 1) g_j` is
    driven to zero as far as the first two allow. The balance is what reduces the
    variance of Horvitz-Thompson estimators; the exact probabilities are what make
    them unbiased in the first place.

    The delivered probabilities are `scale_pi_to_budget(pi, budget)` rather than
    `pi` itself, because `sum(pi)` is the expected sample size and a fixed-size
    design of `budget` items is incoherent unless the probabilities sum to
    `budget`. When they already do -- as they do for `pi_aopt_for_budget(...,
    budget)` -- the rescaling is a no-op. **Weight by the delivered probabilities,
    not by the input**, or call `Design.sample`, which reports them.

    Args:
        pi: Inclusion probabilities for items. Index contains item identifiers.
        g: Regression projections g_j = X^T v_j for each item j (shape (p, m)).
        budget: Fixed sample size (number of items to select).
        random_state: Random state for reproducible sampling. Accepts None, an
            int seed, or a numpy Generator.

    Returns:
        Index of selected items. Length equals `budget`.

    Raises:
        ValidationError: If `pi`, `g`, or `budget` fail validation checks.

    See Also:
        scale_pi_to_budget: The probabilities this sampler actually delivers.
        pi_aopt_for_budget: Compute optimal inclusion probabilities.
        core_plus_tail: Hybrid deterministic + balanced sampling.
        calibrate_weights: Post-stratification weight adjustment.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import balanced_fixed_size, pi_aopt_for_budget
        >>> rng = np.random.default_rng(0)
        >>> counts = pd.DataFrame(rng.poisson(5, (1000, 100)))
        >>> X = pd.DataFrame(rng.standard_normal((1000, 3)))
        >>> probs = pi_aopt_for_budget(counts, X, budget=30)
        >>> selected = balanced_fixed_size(
        ...     probs.probabilities, probs.influence_projections, 30, random_state=42
        ... )
        >>> len(selected)
        30

    Notes:
        Earlier releases drew the sample proportional to `pi` without replacement
        and then swapped items greedily to improve balance. Neither step preserves
        `pi`: on the package's own test fixture, items with `pi == 1` were included
        75% of the time before the swaps and 45% after, which biased every
        estimator weighted by `1 / pi`.
    """
    # Validate inputs
    pi = validate_probability_series(pi)
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        raise ValidationError(
            "g must be a 2D numpy array", "Ensure g has shape (n_features, n_items)"
        )

    m: int = len(pi)
    if g.shape[1] != m:
        raise ValidationError(
            f"g has {g.shape[1]} columns but pi has {m} items",
            "Ensure g and pi represent the same items in the same order",
        )

    budget = validate_budget(budget, m)

    rng: np.random.Generator = _get_random_generator(random_state)
    cols: pd.Index = pi.index

    target: np.ndarray = scale_pi_to_budget(pi.to_numpy(float), budget)
    inv_pi: np.ndarray = 1.0 / np.maximum(target, DIVISION_EPS)

    # Rows the cube walk must preserve. Balancing on g/pi is what forces
    # sum_S (I_j/pi_j - 1) g_j to zero; the row of ones is the fixed-size
    # constraint, sum_j I_j = sum_j pi_j = budget. Order matters: the walk drops
    # rows from the top when it can no longer move, so the sample-size
    # constraint goes last and is the one never given up.
    constraints: np.ndarray = np.vstack([g * inv_pi[None, :], np.ones(m)])

    selected: np.ndarray = cube_sample(target, constraints, rng)
    return cast("pd.Index", cols[selected])

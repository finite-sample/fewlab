"""fewlab: buy the labels that buy the smallest standard error.

One label belongs to an entity -- a website, a merchant, a surname -- that
recurs across many units, so what a label is worth depends on where the entity
appears and on what you intend to estimate. That has an exact answer, not a
heuristic one: writing `V` for row shares and `A` for entity labels, the
complete-label regression coefficient is

    B = (X'X)^-1 X' V A = sum_j h_j a_j'

so entity `j`'s entire contribution is the rank-one matrix `h_j a_j'`, and `h_j`
is computable before a single label is bought.

Two goals, neither traded for the other. Unbiasedness is a constraint: every
contributing entity gets positive probability and reciprocal weight. Minimum
standard error is the objective under it.
"""

from .covariate import ShareEffect, estimate_share_effect
from .designs import (
    LabelDesign,
    core_plus_tail,
    design_risk,
    poisson_pps,
    simple_random,
)
from .estimator import Estimate, LabelSample, estimate, trim_alpha
from .influence import (
    Influence,
    contribution_scores,
    direction_spectrum,
    influence,
)
from .study import Ranking, Study

__version__ = "3.0.0"

__all__ = [
    "Estimate",
    "Influence",
    "LabelDesign",
    "LabelSample",
    "Ranking",
    "ShareEffect",
    "Study",
    "__version__",
    "contribution_scores",
    "core_plus_tail",
    "design_risk",
    "direction_spectrum",
    "estimate",
    "estimate_share_effect",
    "influence",
    "poisson_pps",
    "simple_random",
    "trim_alpha",
]

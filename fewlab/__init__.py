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

from importlib import metadata as _metadata

from .covariate import ShareEffect, estimate_share_effect
from .designs import (
    LabelDesign,
    core_plus_tail,
    design_risk,
    poisson_pps,
    robust_pps,
    simple_random,
)
from .estimator import Estimate, LabelSample, estimate, trim_alpha
from .influence import (
    Influence,
    contribution_scores,
    direction_spectrum,
    influence,
)
from .rowse import RowSEResult, row_se_min_labels
from .study import Ranking, Study
from .superpopulation import (
    SuperpopulationEstimate,
    estimate_superpopulation,
)

# Single source of truth is pyproject.toml; reading it back avoids a second
# copy that nothing keeps in sync.
try:
    _installed_version = _metadata.version("fewlab")
except _metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    _installed_version = "0.0.0.dev0"

__version__ = _installed_version

__all__ = [
    "Estimate",
    "Influence",
    "LabelDesign",
    "LabelSample",
    "Ranking",
    "RowSEResult",
    "ShareEffect",
    "Study",
    "SuperpopulationEstimate",
    "__version__",
    "contribution_scores",
    "core_plus_tail",
    "design_risk",
    "direction_spectrum",
    "estimate",
    "estimate_share_effect",
    "estimate_superpopulation",
    "influence",
    "poisson_pps",
    "robust_pps",
    "row_se_min_labels",
    "simple_random",
    "trim_alpha",
]

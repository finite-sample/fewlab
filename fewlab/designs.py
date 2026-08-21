"""Probability designs: which entities to label, and with what probability.

Two goals, and neither is traded for the other. Unbiasedness is a hard
constraint: every entity that contributes to the estimand must have a strictly
positive chance of being labeled, and must be weighted by the reciprocal of that
chance. Minimum standard error is the objective, optimized subject to it.

The square-root rule below satisfies the constraint automatically -- it assigns
`pi_j > 0` to every entity with `b_j > 0` without needing a floor -- and
minimizes the exact design risk among all Poisson designs of the same expected
cost. An entity with `b_j = 0` contributes nothing to the estimand, so leaving it
unlabeled costs no bias at all, and it is the one case where `pi_j = 0` is
correct rather than merely convenient.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CERTAINTY = 1.0 - 1e-12


@dataclass(frozen=True, slots=True)
class LabelDesign:
    """A probability design over labelable entities.

    This is a plan, not a draw. It fixes the probability with which each entity
    would be labeled; realizing it produces a `LabelSample`, which is the only
    thing an estimator will accept.

    Attributes:
        probabilities: First-order inclusion probabilities, indexed by entity.
        scores: The contribution scores `b_j` the design was built from.
        risk: Exact expected loss `sum_j (1 - pi_j) / pi_j * b_j` under Poisson
            sampling with these probabilities.
        method: Name of the rule that produced the probabilities.
        diagnostics: Method-specific detail, such as the certainty-core size.
    """

    probabilities: pd.Series
    scores: pd.Series
    risk: float
    method: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def budget(self) -> float:
        """Expected number of labels purchased.

        Returns:
            Sum of the inclusion probabilities.
        """
        return float(self.probabilities.sum())

    @property
    def certain(self) -> pd.Index:
        """Entities labeled with probability one.

        Returns:
            Index of the certainty core.
        """
        return pd.Index(self.probabilities.index[self.probabilities >= CERTAINTY])


def design_risk(scores: pd.Series | np.ndarray, pi: pd.Series | np.ndarray) -> float:
    """Compute the Poisson design risk `sum_j (1 - pi_j) / pi_j * b_j`.

    This is exact when entities are drawn as independent Bernoulli trials. Any
    fixed-size design with the same first-order probabilities -- a simple random
    tail, a cube draw -- has a smaller variance, because fixing the sample size
    and balancing on auxiliaries both remove variation this expression still
    counts. Treat it as exact for Poisson and as an upper bound otherwise.

    Args:
        scores: Contribution scores `b_j`.
        pi: Inclusion probabilities, aligned with `scores`.

    Returns:
        The expected loss. Infinite if any entity with `b_j > 0` has `pi_j = 0`,
        which is the arithmetic signature of a biased design.
    """
    b = np.asarray(scores, dtype=float)
    p = np.asarray(pi, dtype=float)
    contributes = b > 0
    if not contributes.any():
        return 0.0
    if (p[contributes] <= 0).any():
        return float("inf")
    return float(np.sum((1.0 - p[contributes]) / p[contributes] * b[contributes]))


def _root_ratios(
    scores: pd.Series, cost: pd.Series | np.ndarray | float | None
) -> np.ndarray:
    """Compute `sqrt(b_j / c_j)`, the quantity probabilities are proportional to.

    Args:
        scores: Contribution scores `b_j`.
        cost: Per-entity labeling cost, or None for uniform cost.

    Returns:
        The ratios, with zeros wherever `b_j` is zero.

    Raises:
        ValueError: If costs are not strictly positive or are misaligned.
    """
    b = scores.to_numpy(dtype=float)
    if cost is None:
        c = np.ones_like(b)
    else:
        c = np.asarray(cost, dtype=float)
        if c.ndim == 0:
            c = np.full_like(b, float(c))
        if c.size != b.size:
            raise ValueError(f"cost must have {b.size} entries, got {c.size}")
    if (c <= 0).any():
        raise ValueError("cost must be strictly positive for every entity")
    return np.sqrt(np.maximum(b, 0.0) / c)


def poisson_pps(
    scores: pd.Series,
    budget: float,
    *,
    cost: pd.Series | np.ndarray | float | None = None,
    pi_min: float = 0.0,
) -> LabelDesign:
    """Build the risk-minimizing design for a fixed labeling budget.

    Minimizing `sum_j (1 - pi_j) / pi_j * b_j` subject to `sum_j c_j pi_j = B`
    and `pi_j <= 1` gives

        pi_j* = min(1, lambda * sqrt(b_j / c_j))

    for the `lambda` that exhausts the budget. The cap at one is not a separate
    rule for "big" entities: it falls out of the constraint, and the set of
    entities it binds on *is* the certainty core. In a heavy-tailed population --
    website visits, merchant transactions, surname frequencies -- that core
    absorbs most of the budget, which is the whole reason the design pays.

    `lambda` is solved exactly rather than by search. Sorting the ratios
    descending, if the top `k` are capped then `B = k + lambda * sum_{j>k} r_j`,
    so `lambda` is determined by `k`, and `k` is the smallest value consistent
    with itself.

    Args:
        scores: Contribution scores `b_j` from `contribution_scores`.
        budget: Total labeling budget. With uniform cost this is the expected
            number of entities labeled.
        cost: Per-entity cost `c_j` of acquiring one label. None means uniform
            cost, under which the rule reduces to `pi_j proportional to
            sqrt(b_j)`.
        pi_min: Optional floor applied to entities with `b_j > 0`. The default
            of zero is deliberate: the square-root rule already gives every
            contributing entity a strictly positive probability, so no floor is
            needed for unbiasedness, and any floor moves the design away from
            minimum risk. Set it only to bound the largest weight.

    Returns:
        The design, carrying its probabilities and its exact risk.

    Raises:
        ValueError: If the budget is not positive or exceeds what the
            contributing entities can absorb.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import poisson_pps
        >>> b = pd.Series(np.arange(1, 21, dtype=float) ** 2)
        >>> design = poisson_pps(b, budget=5)
        >>> round(design.budget, 6)
        5.0
        >>> bool((design.probabilities > 0).all())
        True
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    r = _root_ratios(scores, cost)
    c = (
        np.ones_like(r)
        if cost is None
        else np.broadcast_to(np.asarray(cost, dtype=float), r.shape)
    )
    contributes = r > 0
    max_spend = float(c[contributes].sum())
    if budget > max_spend:
        raise ValueError(
            f"budget {budget} exceeds the cost {max_spend} of labeling every "
            f"entity that contributes to the estimand; nothing is left to buy"
        )

    pi = np.zeros_like(r)
    idx = np.flatnonzero(contributes)
    order = idx[np.argsort(-r[idx], kind="stable")]
    r_sorted = r[order]
    c_sorted = c[order]

    # Grow the certainty core until the next entity would not be capped.
    tail_mass = float(np.sum(r_sorted * c_sorted))
    spent = 0.0
    k = 0
    while k < order.size:
        tail_mass -= r_sorted[k] * c_sorted[k]
        spent += c_sorted[k]
        remaining = budget - spent
        if remaining <= 0:
            break
        if tail_mass <= 0:
            k += 1
            break
        lam = remaining / tail_mass
        if k + 1 < order.size and lam * r_sorted[k + 1] < 1.0:
            k += 1
            break
        k += 1

    pi[order[:k]] = 1.0
    rest = order[k:]
    if rest.size:
        tail_mass = float(np.sum(r[rest] * c[rest]))
        remaining = budget - float(c[order[:k]].sum())
        lam = remaining / tail_mass if tail_mass > 0 else 0.0
        pi[rest] = np.minimum(1.0, lam * r[rest])

    if pi_min > 0:
        pi[contributes] = np.maximum(pi[contributes], pi_min)

    probabilities = pd.Series(pi, index=scores.index, name="pi")
    n_certain = int((pi >= CERTAINTY).sum())
    return LabelDesign(
        probabilities=probabilities,
        scores=scores,
        risk=design_risk(scores, probabilities),
        method="poisson_pps",
        diagnostics={
            "n_certainty": n_certain,
            "n_contributing": int(contributes.sum()),
            "max_weight": float(1.0 / pi[contributes].min())
            if contributes.any()
            else 0.0,
            "pi_min": pi_min,
        },
    )


def core_plus_tail(scores: pd.Series, budget: int) -> LabelDesign:
    """Build the best certainty-core-plus-uniform-tail design.

    Label the `H` highest-contribution entities with certainty and give every
    remaining entity the same probability `p_H = (B - H) / (m - H)`. With
    `S_H = sum_{j>H} b_j` the Poisson risk collapses to a closed form,

        R(H) = (m - B) / (B - H) * S_H

    so the optimal core size is found by evaluating at most `B` cutoffs rather
    than guessed at. Promoting one more entity into the core pays exactly when
    `b_{H+1} > S_H / (B - H)`.

    `R(H)` is a *planning criterion*, and the distinction matters. It is the
    exact risk when the tail is drawn as independent Bernoulli trials at
    `p_H`. Draw the tail instead as a fixed-size simple random sample of
    `B - H` and the true variance is smaller, because fixing the count removes
    the sample-size variability and the contributions enter centered on the tail
    mean. So `R(H)` is an upper bound for the fixed-size draw, and the core size
    it selects is optimal for the Poisson design and near-optimal for the SRS
    one. `tests/test_designs.py` checks that gap by simulation rather than
    assuming it.

    This design is never better than `poisson_pps`, which is free to vary the
    tail probabilities. It is worth having because a uniform tail is a simple
    random sample: it is easy to explain to a labeling vendor, easy to audit,
    and its variance can be estimated by anyone with an introductory survey
    course. Use it when transparency is worth more than the last few percent of
    efficiency.

    Args:
        scores: Contribution scores `b_j` from `contribution_scores`.
        budget: Number of entities to label. Must be a positive integer no
            larger than the number of contributing entities.

    Returns:
        The design, with the chosen core size and the full `R(H)` curve in its
        diagnostics.

    Raises:
        ValueError: If the budget is not a positive integer, or exceeds the
            number of entities that contribute to the estimand.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import core_plus_tail
        >>> b = pd.Series(np.arange(1, 101, dtype=float)[::-1] ** 3)
        >>> design = core_plus_tail(b, budget=20)
        >>> design.diagnostics["core_size"] > 0
        True
        >>> round(design.budget, 6)
        20.0
    """
    if int(budget) != budget or budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget}")
    budget = int(budget)

    b_all = scores.to_numpy(dtype=float)
    contributes = b_all > 0
    m = int(contributes.sum())
    if budget > m:
        raise ValueError(
            f"budget {budget} exceeds the {m} entities that contribute to the "
            f"estimand; the rest can be left unlabeled at no cost in bias"
        )

    idx = np.flatnonzero(contributes)
    order = idx[np.argsort(-b_all[idx], kind="stable")]
    b_sorted = b_all[order]

    # S_H = sum of scores strictly below the core, for H = 0 .. budget - 1.
    suffix = np.concatenate([np.cumsum(b_sorted[::-1])[::-1], [0.0]])
    cores = np.arange(budget)
    risks = (m - budget) / (budget - cores) * suffix[cores]

    best = int(np.argmin(risks))
    tail_p = (budget - best) / (m - best) if m > best else 1.0

    pi = np.zeros_like(b_all)
    pi[order[:best]] = 1.0
    pi[order[best:]] = tail_p

    probabilities = pd.Series(pi, index=scores.index, name="pi")
    return LabelDesign(
        probabilities=probabilities,
        scores=scores,
        risk=design_risk(scores, probabilities),
        method="core_plus_tail",
        diagnostics={
            "core_size": best,
            "tail_probability": float(tail_p),
            "n_contributing": m,
            "risk_curve": pd.Series(risks, index=cores, name="risk"),
        },
    )


def simple_random(scores: pd.Series, budget: int) -> LabelDesign:
    """Build the uniform-probability design, as a benchmark.

    Every contributing entity gets the same probability `B / m`. This is what
    you do when you ignore the fact that entities differ in how much they matter
    to the estimand, and it is the baseline any efficiency claim should be
    measured against.

    Args:
        scores: Contribution scores `b_j`, used for the risk calculation and to
            identify which entities contribute at all.
        budget: Number of entities to label.

    Returns:
        The uniform design and its exact risk.

    Raises:
        ValueError: If the budget is not a positive integer, or exceeds the
            number of contributing entities.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import poisson_pps, simple_random
        >>> b = pd.Series(np.arange(1, 201, dtype=float) ** 2)
        >>> srs = simple_random(b, budget=20)
        >>> pps = poisson_pps(b, budget=20)
        >>> pps.risk < srs.risk
        True
    """
    if int(budget) != budget or budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget}")
    budget = int(budget)

    b = scores.to_numpy(dtype=float)
    contributes = b > 0
    m = int(contributes.sum())
    if budget > m:
        raise ValueError(f"budget {budget} exceeds the {m} contributing entities")

    pi = np.zeros_like(b)
    pi[contributes] = budget / m
    probabilities = pd.Series(pi, index=scores.index, name="pi")
    return LabelDesign(
        probabilities=probabilities,
        scores=scores,
        risk=design_risk(scores, probabilities),
        method="simple_random",
        diagnostics={"n_contributing": m},
    )

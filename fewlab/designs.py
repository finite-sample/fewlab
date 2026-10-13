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

    This is exact when entities are drawn as independent Bernoulli trials and
    scores include true squared label norms. Fixed-size and balanced designs
    involve joint inclusion probabilities; this expression is a planning
    criterion for them, not a universal upper bound.

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
    return float(np.sum((b[contributes] / p[contributes]) * (1.0 - p[contributes])))


def _design_inputs(
    scores: pd.Series, cost: pd.Series | np.ndarray | float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate scores and align costs to their entities.

    Args:
        scores: Nonnegative contribution scores with unique entity identifiers.
        cost: Positive costs; indexed costs are aligned by entity.

    Returns:
        Score and cost arrays in the score index order.

    Raises:
        ValueError: If scores or costs are invalid or indexes do not match.
    """
    if not scores.index.is_unique:
        raise ValueError("scores must have unique entity identifiers")
    b = scores.to_numpy(dtype=float)
    if not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("scores must be finite and nonnegative")
    if isinstance(cost, pd.Series):
        if (
            not cost.index.is_unique
            or len(cost) != len(scores)
            or not scores.index.isin(cost.index).all()
        ):
            raise ValueError("cost index must match the score entities")
        cost = cost.reindex(scores.index).to_numpy(dtype=float)
    c = np.asarray(1.0 if cost is None else cost, dtype=float)
    if c.ndim == 0:
        if not np.isfinite(c) or c <= 0:
            raise ValueError("cost must be finite and strictly positive")
        c = np.full_like(b, float(c))
    if c.shape != b.shape:
        raise ValueError(f"cost must have shape {b.shape}, got {c.shape}")
    if not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("cost must be finite and strictly positive")
    return b, c


def min_cost_for_risk(
    scores: pd.Series,
    risk_limit: float,
    *,
    cost: pd.Series | np.ndarray | float | None = None,
) -> LabelDesign:
    """Minimize expected labeling cost for a Poisson risk ceiling.

    Solve `min sum(c_j pi_j)` subject to
    `sum(b_j (1 - pi_j) / pi_j) <= risk_limit`. The solution is
    `pi_j = min(1, lambda sqrt(b_j / c_j))`. For an uncapped tail T,
    `lambda = sum_T sqrt(b_j c_j) / (risk_limit + sum_T b_j)`.
    Sorting the ratios identifies the certainty core, including an empty core.

    The risk is exact for independent Bernoulli draws when the scores include
    the true squared label norms. Unit norms are exact for a complete one-hot
    taxonomy and conservative for an unknown binary indicator. Estimated norms
    give a planning criterion. A scalar contrast targets its variance; joint
    coefficient loss targets the sum of coefficient variances. This ceiling
    excludes regression sampling uncertainty and does not bound realized error.

    Args:
        scores: Finite, nonnegative contribution scores indexed by unique entity.
        risk_limit: Finite, nonnegative ceiling in squared estimand units.
        cost: Positive finite labeling costs. None means unit costs. A Series
            must contain exactly the score entities and is aligned by index.

    Returns:
        A design for `Study.draw(..., fixed_size=False)`. Its `budget` is the
        expected label count; diagnostics also report expected cost and risk.
        Zero-score entities have zero probability. Guarantees apply only to
        the scored estimand, not to other coefficients or to balanced draws.

    Raises:
        ValueError: If scores, costs, or the ceiling are invalid.
        ArithmeticError: If finite probabilities satisfying the ceiling cannot
            be represented numerically.

    Examples:
        >>> import pandas as pd
        >>> from fewlab import min_cost_for_risk
        >>> design = min_cost_for_risk(pd.Series([1.0] * 10), risk_limit=40.0)
        >>> round(design.budget, 6), round(design.risk, 6)
        (2.0, 40.0)
    """
    b, c = _design_inputs(scores, cost)
    if not np.isfinite(risk_limit) or risk_limit < 0:
        raise ValueError("risk_limit must be finite and nonnegative")
    pi = np.zeros_like(b)
    active = np.flatnonzero(b > 0)
    if risk_limit == 0:
        pi[active] = 1.0
    elif active.size:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Use extended precision where available; fail on arithmetic overflow.
            weights = b[active].astype(np.longdouble)
            costs = c[active].astype(np.longdouble)
            ratios = np.sqrt(weights) / np.sqrt(costs)
            order = np.argsort(-ratios, kind="stable")
            mass = np.sqrt(weights[order]) * np.sqrt(costs[order])
            mass_tail = np.cumsum(mass[::-1])[::-1]
            score_tail = np.cumsum(weights[order][::-1])[::-1]
            for k, position in enumerate(order):
                lam = mass_tail[k] / (np.longdouble(risk_limit) + score_tail[k])
                if lam * ratios[position] <= 1:
                    tail = order[k:]
                    pi[active[tail]] = np.asarray(lam * ratios[tail], dtype=float)
                    break
                pi[active[position]] = 1.0
    if not np.isfinite(pi).all() or (pi[active] <= 0).any():
        raise ArithmeticError("risk ceiling requires unrepresentable probabilities")
    # Round toward feasibility when converting the analytic solution to float.
    pi[active] = np.minimum(1.0, np.nextafter(pi[active], 1.0))
    risk = design_risk(b, pi)
    expected_cost = float(c @ pi)
    if not np.isfinite(expected_cost) or not np.isfinite(risk):
        raise ArithmeticError("design cost or risk is not finite")
    if risk > risk_limit and not np.isclose(risk, risk_limit, rtol=1e-12, atol=0):
        raise ArithmeticError("computed design does not meet the risk ceiling")
    return LabelDesign(
        probabilities=pd.Series(pi, index=scores.index, name="pi"),
        scores=scores.copy(),
        risk=risk,
        method="min_cost_for_risk",
        diagnostics={
            "expected_cost": expected_cost,
            "risk_limit": float(risk_limit),
            "achieved_risk": risk,
            "n_certainty": int((pi >= CERTAINTY).sum()),
            "n_contributing": int(active.size),
        },
    )


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
    descending, a proposed core leaves a tail with
    `lambda = (B - sum_core c_j) / sum_tail sqrt(b_j c_j)`. Start with an empty
    core and promote an entity only when its proposed probability exceeds one.

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
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError(f"budget must be finite and positive, got {budget}")

    b, c = _design_inputs(scores, cost)
    r = np.sqrt(b) / np.sqrt(c)
    contributes = r > 0
    max_spend = float(c[contributes].sum())
    if budget > max_spend and not np.isclose(
        budget, max_spend, rtol=8 * np.finfo(float).eps, atol=0
    ):
        raise ValueError(
            f"budget {budget} exceeds the cost {max_spend} of labeling every "
            f"entity that contributes to the estimand; nothing is left to buy"
        )

    budget = min(budget, max_spend)
    pi = np.zeros_like(r)
    idx = np.flatnonzero(contributes)
    order = idx[np.argsort(-r[idx], kind="stable")]
    r_sorted = r[order]
    c_sorted = c[order]

    # Check the empty core first: low budgets need not make any item certain.
    tail_mass = np.cumsum((r_sorted * c_sorted)[::-1])[::-1]
    spent = 0.0
    for k, position in enumerate(order):
        lam = (budget - spent) / tail_mass[k]
        if lam * r_sorted[k] <= 1.0:
            pi[order[k:]] = np.minimum(1.0, lam * r_sorted[k:])
            break
        pi[position] = 1.0
        spent += c_sorted[k]

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
    `p_H`. For a fixed-size simple random tail, the true variance instead uses
    contributions centered on their tail mean and a finite-population correction.
    Depending on the contributions it can exceed the Poisson criterion. The core
    size selected here is optimal for the Poisson design only.

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
        >>> b = pd.Series(1.0 / np.arange(1, 101, dtype=float) ** 2)
        >>> design = core_plus_tail(b, budget=20)
        >>> design.diagnostics["core_size"]
        10
        >>> round(design.budget, 6)
        20.0

        A certainty core is not automatic. It appears only when the head is
        concentrated enough that `b_{H+1}` exceeds the average remaining
        contribution; on a population that decays more gently the whole budget
        goes to the tail, and the design says so rather than forcing a core.

        >>> gentle = pd.Series(np.arange(1, 101, dtype=float)[::-1] ** 3)
        >>> core_plus_tail(gentle, budget=20).diagnostics["core_size"]
        0
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


def _hinge_optimum(
    w: np.ndarray, c: np.ndarray, lam: float, tau: float, pi_min: float
) -> np.ndarray:
    """Minimize the per-entity Lagrangian of the robust program.

    For fixed `tau` and multiplier `lam` the program separates, and entity `j`
    solves `min_p (w_j/p - w_j - tau)_+ + lam c_j p`. The hinge turns off at
    `s_j = w_j / (w_j + tau)`, above which the objective only increases, so the
    optimum is either `s_j` or the stationary point `sqrt(w_j / (lam c_j))` of
    the active branch, whichever value is smaller.

    Args:
        w: Label-free contributions `h_j' W h_j`.
        c: Per-entity costs.
        lam: Budget multiplier.
        tau: Epigraph variable.
        pi_min: Lower clip.

    Returns:
        The minimizing probabilities.
    """
    safe = w / (w + tau) if tau > 0 else np.ones_like(w)
    interior = np.sqrt(w / (lam * c))
    pi = np.minimum(interior, safe)
    active = interior < safe
    value_interior = 2.0 * np.sqrt(w * lam * c) - w - tau
    value_safe = lam * c * safe
    pi = np.where(active & (value_safe <= value_interior), safe, pi)
    return np.clip(pi, pi_min, 1.0)


def _budget_solve(
    w: np.ndarray, c: np.ndarray, tau: float, budget: float, pi_min: float
) -> np.ndarray:
    """Find the multiplier that exhausts the budget at a given `tau`.

    Args:
        w: Label-free contributions.
        c: Per-entity costs.
        tau: Epigraph variable.
        budget: Total budget.
        pi_min: Lower clip.

    Returns:
        Probabilities spending the budget.
    """
    lo, hi = 1e-12, 1e12
    for _ in range(200):
        lam = float(np.sqrt(lo * hi))
        if float((c * _hinge_optimum(w, c, lam, tau, pi_min)).sum()) > budget:
            lo = lam
        else:
            hi = lam
    return _hinge_optimum(w, c, hi, tau, pi_min)


def robust_pps(
    scores: pd.Series,
    budget: float,
    *,
    prevalence: float,
    cost: pd.Series | np.ndarray | float | None = None,
    pi_min: float = 1e-4,
) -> LabelDesign:
    """Build the design that minimizes worst-case risk under a prevalence cap.

    `poisson_pps` needs the squared label norms, which for a binary trait means
    knowing which entities carry it. When that is unknown but the trait is known
    to be rare, the worst case is tractable. Suppose at most a fraction `alpha`
    of entities are positive, so at most `M = ceil(alpha m)` of them, and write
    `t_j = (1 - pi_j)/pi_j * w_j` for the label-free contribution. Then

        max over label configurations of the risk = sum of the M largest t_j,

    since an adversary spends its `M` positives on the largest terms. That
    maximum is convex in `pi`, despite being a maximum over a combinatorial set,
    by the epigraph identity

        sum of the M largest t = min over tau of [ M tau + sum_j (t_j - tau)_+ ],

    a minimum of jointly convex functions. Minimizing it under the budget is
    therefore a convex program, solved here by bisecting the budget multiplier
    inside a golden-section search on `tau`.

    Two consequences follow in a line each and both are checked by the test
    suite rather than assumed. At `alpha = 1` the sum of the `m` largest terms
    is the whole sum, so the program becomes `poisson_pps` with unit label
    norms and returns the same design. And on the worst case this design is
    never beaten by `poisson_pps`, since it minimizes exactly that quantity;
    it is beaten on the average case, which is what the insurance costs.

    Args:
        scores: Label-free contributions `w_j = h_j' W h_j`, from
            `contribution_scores` with the default unit label norms.
        budget: Total labeling budget.
        prevalence: Upper bound `alpha` on the fraction of entities carrying the
            trait, in `(0, 1]`.
        cost: Per-entity labeling cost, or None for uniform.
        pi_min: Lower clip. Unlike `poisson_pps`, a small positive floor is the
            default here, because the worst-case objective is flat in entities
            the adversary will not choose and so nothing else keeps their
            probabilities away from zero.

    Returns:
        The design, with the fitted `tau` and the worst-case risk among its
        diagnostics.

    Raises:
        ValueError: If `prevalence` is outside `(0, 1]` or the budget is not
            positive.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from fewlab import robust_pps
        >>> w = pd.Series(np.exp(np.linspace(0, 4, 200)))
        >>> design = robust_pps(w, budget=40, prevalence=0.05)
        >>> round(design.budget, 4)
        40.0
        >>> design.diagnostics["n_positive"]
        10
    """
    if not 0 < prevalence <= 1:
        raise ValueError(f"prevalence must be in (0, 1], got {prevalence}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    w = np.maximum(scores.to_numpy(dtype=float), 0.0)
    c = (
        np.ones_like(w)
        if cost is None
        else np.broadcast_to(np.asarray(cost, dtype=float), w.shape).astype(float)
    )
    m = int((w > 0).sum())
    n_positive = max(1, int(np.ceil(prevalence * m)))

    def worst_case(tau: float) -> tuple[float, np.ndarray]:
        pi = _budget_solve(w, c, tau, budget, pi_min)
        t = np.where(pi > 0, w / np.maximum(pi, 1e-300) - w, np.inf)
        return n_positive * tau + float(np.maximum(t - tau, 0.0).sum()), pi

    lo, hi = 0.0, float(w.max() / max(pi_min, 1e-6)) if w.size else 1.0
    for _ in range(120):
        a = lo + 0.382 * (hi - lo)
        b = lo + 0.618 * (hi - lo)
        if worst_case(a)[0] <= worst_case(b)[0]:
            hi = b
        else:
            lo = a
    tau = 0.5 * (lo + hi)
    _, pi = worst_case(tau)

    probabilities = pd.Series(pi, index=scores.index, name="pi")
    t = np.where(pi > 0, w / np.maximum(pi, 1e-300) - w, 0.0)
    return LabelDesign(
        probabilities=probabilities,
        scores=scores,
        risk=design_risk(scores, probabilities),
        method="robust_pps",
        diagnostics={
            "prevalence": prevalence,
            "n_positive": n_positive,
            "tau": tau,
            "worst_case_risk": float(np.sort(t)[::-1][:n_positive].sum()),
            "n_certainty": int((pi >= CERTAINTY).sum()),
        },
    )

"""The user-facing workflow: data, then a plan, then a draw, then an estimate.

Four objects, and the boundaries between them are the point:

    Study        the data. Counts, features, and the influence decomposition.
    Ranking      an ordering of entities by how much labeling them would help.
    LabelDesign  a probability for every entity. A plan, not yet a draw.
    LabelSample  a realized draw, carrying the probabilities delivered.
    Estimate     coefficients and their uncertainty.

Only a `LabelSample` reaches an estimator. A `Ranking` cannot, and that is
deliberate: the top entities by influence are an excellent operational priority
list, but they are not a probability sample, and no probability attached to them
afterward makes them one. Version 2 blurred these -- `calibrate_weights` invented
inclusion probabilities from the *size* of whatever set it was handed -- and the
result was a workflow the documentation called Horvitz-Thompson while it was
neither unbiased nor design-based.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .calibration import calibrate_weights
from .counts import CountMatrix, as_counts
from .cube import cube_sample
from .designs import (
    LabelDesign,
    core_plus_tail,
    min_cost_for_risk,
    poisson_pps,
    simple_random,
)
from .estimator import Estimate, LabelSample, estimate
from .influence import Influence, contribution_scores, direction_spectrum, influence
from .utils import _get_random_generator


def _frame_fingerprint(counts: CountMatrix, X: pd.DataFrame, infl: Influence) -> str:
    """Identify the ordered data and influence decomposition behind a draw."""
    digest = hashlib.sha256()
    unit_order = sorted(
        range(len(counts.index)),
        key=lambda i: (type(counts.index[i]).__name__, repr(counts.index[i])),
    )
    canonical_units = counts.index[unit_order]
    metadata = (
        list(canonical_units),
        list(counts.columns),
        list(X.columns),
        list(infl.items),
        infl.ridge,
    )
    digest.update(repr(metadata).encode("utf-8"))
    row_positions = np.empty(len(unit_order), dtype=int)
    row_positions[unit_order] = np.arange(len(unit_order))
    canonical_rows = row_positions[counts.rows]
    order = np.lexsort((counts.cols, canonical_rows))
    rows = canonical_rows[order]
    cols = counts.cols[order]
    starts = np.r_[True, (np.diff(rows) != 0) | (np.diff(cols) != 0)]
    positions = np.flatnonzero(starts)
    values = np.add.reduceat(counts.values[order], positions)
    nonzero = values != 0
    for part in (
        rows[positions][nonzero],
        cols[positions][nonzero],
        values[nonzero],
        X.reindex(canonical_units).to_numpy(float),
    ):
        array = np.ascontiguousarray(part, dtype="<f8")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Ranking:
    """An ordering of entities by contribution, for operational use only.

    A ranking answers "if I can only look at a few, which ones?". It does not
    answer "what is B, and how sure am I?" -- for that the selection has to be
    random, because unbiasedness comes from every entity having had a chance to
    be chosen. Passing a `Ranking` where a `LabelSample` belongs raises rather
    than silently producing a number that looks like an estimate.

    Attributes:
        items: Entities in decreasing order of contribution.
        scores: The contribution scores `b_j` that produced the order.
        loss: The estimand the scores were computed for.
        coverage: Share of total contribution mass the ranked entities carry.
    """

    items: pd.Index
    scores: pd.Series
    loss: str
    coverage: float

    def __len__(self) -> int:
        """Number of ranked entities.

        Returns:
            Length of the ranking.
        """
        return len(self.items)

    def to_design(self, budget: int, **kwargs: object) -> LabelDesign:
        """Point at the supported route from a ranking to an estimable design.

        Args:
            budget: Labeling budget.
            **kwargs: Forwarded to the design constructor.

        Returns:
            Never returns.

        Raises:
            NotImplementedError: Always. Build the design from `Study.design`,
                which starts from the scores rather than from a truncated list.
        """
        raise NotImplementedError(
            "A ranking cannot be converted into a probability design after the "
            "fact: truncating at the top B gives probability one to the entities "
            "above the cut and zero to everything below it, and no probability "
            "assigned afterward changes what happened. Call Study.design(budget) "
            "to build a design from the scores directly."
        )


@dataclass(frozen=True, slots=True)
class Study:
    """Counts, features, and everything derived from them.

    Attributes:
        counts: Count matrix, units by labelable entities.
        X: Unit feature matrix.
        influence: The exact decomposition `B = sum_j h_j a_j'`.
        diagnostics: Conditioning and degeneracy detail.
    """

    counts: CountMatrix
    X: pd.DataFrame
    influence: Influence
    diagnostics: dict = field(default_factory=dict)
    frame_fingerprint: str = ""

    def __init__(
        self,
        counts: pd.DataFrame | CountMatrix,
        X: pd.DataFrame,
        *,
        ridge: float | None = None,
        ensure_full_rank: bool = True,
    ) -> None:
        """Build a study from counts and features.

        Args:
            counts: Count matrix, units as rows and labelable entities as
                columns. The columns define the frame: the population is the
                entities someone actually touched, not every entity in the
                world.
            X: Unit feature matrix aligned with `counts.index`.
            ridge: Ridge term for `X'X`, or None to choose automatically.
            ensure_full_rank: Whether to apply the automatic ridge heuristic.
        """
        counts = as_counts(counts)
        infl = influence(counts, X, ridge=ridge, ensure_full_rank=ensure_full_rank)
        spectrum = direction_spectrum(infl)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "influence", infl)
        object.__setattr__(
            self, "frame_fingerprint", _frame_fingerprint(counts, X, infl)
        )
        object.__setattr__(
            self,
            "diagnostics",
            {
                "n_units": counts.shape[0],
                "density": counts.density,
                "n_entities": infl.n_items,
                "condition_number": float(np.linalg.cond(infl.XtX)),
                "ridge": infl.ridge,
                "direction_spectrum": spectrum,
                "effective_directions": int((spectrum > 0.05).sum()),
            },
        )

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        X: pd.DataFrame,
        *,
        unit: str,
        entity: str,
        count: str,
        entities: pd.Index | None = None,
        **kwargs: object,
    ) -> "Study":
        """Build a study from long-format usage data.

        This is the natural entry point for a real panel, which arrives as one
        row per unit-entity pair rather than as a dense matrix. A browsing panel
        of 1,134 users by 34,078 domains is 0.27% dense: 309 MB dense, a couple
        of megabytes as triples.

        Args:
            frame: Long-format counts.
            X: Unit feature matrix indexed by unit identifier.
            unit: Column naming the unit.
            entity: Column naming the labelable entity.
            count: Column holding the count.
            entities: Optional full entity frame, when some entities should
                belong to the population despite having no recorded activity.
            **kwargs: Forwarded to the constructor.

        Returns:
            The study.

        Examples:
            >>> import pandas as pd
            >>> from fewlab import Study
            >>> long = pd.DataFrame(
            ...     {"user": ["a", "a", "b"], "site": ["x", "y", "x"],
            ...      "visits": [3.0, 1.0, 5.0]}
            ... )
            >>> X = pd.DataFrame({"const": [1.0, 1.0], "z": [0.0, 1.0]},
            ...                  index=["a", "b"])
            >>> Study.from_long(long, X, unit="user", entity="site",
            ...                 count="visits").influence.n_items
            2
        """
        counts = CountMatrix.from_long(
            frame, unit=unit, entity=entity, count=count, entities=entities
        )
        return cls(counts, X, **kwargs)  # type: ignore[arg-type]

    def scores(
        self,
        *,
        loss: str = "coefficient",
        contrast: np.ndarray | None = None,
        label_norms: np.ndarray | pd.Series | float = 1.0,
    ) -> pd.Series:
        """Compute each entity's contribution `b_j` to the design risk.

        Args:
            loss: `"coefficient"` or `"prediction"`. See `contribution_scores`.
            contrast: Optional `(p, r)` matrix targeting `L'B` only.
            label_norms: Squared label norms `a_j' a_j`; 1.0 is exact for a
                one-hot taxonomy.

        Returns:
            Contribution scores indexed by entity.
        """
        return contribution_scores(
            self.influence, loss=loss, contrast=contrast, label_norms=label_norms
        )

    def rank(self, budget: int | None = None, **score_kwargs: object) -> Ranking:
        """Order entities by contribution, for operational triage.

        Args:
            budget: How many to return, or None for all.
            **score_kwargs: Forwarded to `scores`.

        Returns:
            The ranking. It cannot be estimated from; see `design`.
        """
        b = self.scores(**score_kwargs)  # type: ignore[arg-type]
        ordered = b.sort_values(ascending=False)
        if budget is not None:
            ordered = ordered.iloc[:budget]
        total = float(b.sum())
        return Ranking(
            items=ordered.index,
            scores=ordered,
            loss=str(score_kwargs.get("loss", "coefficient")),
            coverage=float(ordered.sum() / total) if total > 0 else 0.0,
        )

    def design(
        self,
        budget: float,
        *,
        method: Literal["pps", "core_plus_tail", "srs"] = "pps",
        cost: pd.Series | np.ndarray | float | None = None,
        **score_kwargs: object,
    ) -> LabelDesign:
        """Build a probability design for a labeling budget.

        Args:
            budget: Total labeling budget. With uniform cost this is the
                expected number of entities labeled.
            method: `"pps"` for the risk-minimizing square-root rule,
                `"core_plus_tail"` for a certainty core with a uniform tail, or
                `"srs"` for the uniform benchmark.
            cost: Per-entity labeling cost. Only `"pps"` uses it.
            **score_kwargs: Forwarded to `scores`.

        Returns:
            The design.

        Raises:
            ValueError: If `method` is unrecognized.
        """
        b = self.scores(**score_kwargs)  # type: ignore[arg-type]
        if method == "pps":
            return poisson_pps(b, budget, cost=cost)
        if method == "core_plus_tail":
            return core_plus_tail(b, int(budget))
        if method == "srs":
            return simple_random(b, int(budget))
        raise ValueError(f"unknown method {method!r}")

    def design_for_risk(
        self,
        risk_limit: float,
        *,
        cost: pd.Series | np.ndarray | float | None = None,
        **score_kwargs: object,
    ) -> LabelDesign:
        """Buy the least expected cost for a label-sampling risk ceiling.

        Use the default independent draw via `draw(design)`. A single
        contrast uses a variance ceiling (SE squared); the default coefficient
        loss bounds the sum of coefficient variances. With unit label norms,
        risk is exact for a one-hot taxonomy and bounds an unknown binary label.
        The ceiling excludes regression uncertainty and realized-sample error.
        A contrast design protects only that contrast, not other coefficients.

        Args:
            risk_limit: Finite, nonnegative Poisson risk ceiling.
            cost: Positive costs, or None for minimum expected label count.
            **score_kwargs: Forwarded to `scores`, including `loss`, `contrast`,
                and `label_norms`. Estimated norms give a planning criterion.

        Returns:
            The design, with achieved risk and expected cost in diagnostics.
        """
        scores = self.scores(**score_kwargs)  # type: ignore[arg-type]
        return min_cost_for_risk(scores, risk_limit, cost=cost)

    def draw(
        self,
        design: LabelDesign,
        *,
        random_state: int | np.random.Generator | None = None,
        fixed_size: bool = False,
    ) -> LabelSample:
        """Realize a design into an actual set of entities to label.

        Args:
            design: The probability design to realize.
            random_state: Seed or generator.
            fixed_size: False draws independent Bernoulli trials, giving an
                expected rather than exact sample size and a design-unbiased
                variance estimator. True draws exactly `budget` entities with
                the cube method, balancing on the influence projections.

        Returns:
            The realized sample, carrying the delivered probabilities.

        Raises:
            ValueError: If a fixed-size draw is asked for a non-integer budget.
        """
        rng = _get_random_generator(random_state)
        pi = design.probabilities
        items = pd.Index(pi.index)
        frame = pd.Index(self.influence.items)
        if not items.equals(frame) or items.has_duplicates:
            raise ValueError(
                "design probabilities must cover this Study frame in order"
            )
        values = pi.to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("design probabilities must be finite and in [0, 1]")
        provenance = (
            {"random_seed": random_state} if isinstance(random_state, int) else {}
        )

        if not fixed_size:
            drawn = rng.random(pi.size) < values
            return LabelSample(
                items=items[drawn],
                probabilities=pi,
                method=f"{design.method}+poisson",
                diagnostics={"expected_size": design.budget, **provenance},
                frame_fingerprint=self.frame_fingerprint,
            )

        size = design.budget
        if abs(size - round(size)) > 1e-9:
            raise ValueError(
                f"a fixed-size draw needs an integer budget, got {size}. Use "
                f"fixed_size=False for a Poisson draw."
            )

        active = values > 0
        pi_active = values[active]
        G_active = self.influence.G[:, active]
        constraints = np.vstack(
            [G_active / np.maximum(pi_active, 1e-18)[None, :], np.ones(pi_active.size)]
        )
        picked = cube_sample(pi_active, constraints, rng)
        chosen = items[active][picked]
        return LabelSample(
            items=chosen,
            probabilities=pi,
            method=f"{design.method}+cube",
            diagnostics={
                "balanced_on": "G",
                "target_size": round(size),
                **provenance,
            },
            frame_fingerprint=self.frame_fingerprint,
        )

    def calibrate(self, sample: LabelSample) -> pd.Series:
        """Calibrate a realized sample so it reproduces the influence totals.

        The estimator weights a sampled entity by `1 / pi_j`, which is unbiased
        but leaves the realized sample's influence total `sum_{j in S} g_j /
        pi_j` scattered around its population value `sum_j g_j`. Calibration
        adjusts the weights by the smallest chi-square distance that makes that
        equality hold exactly, which removes the part of the error attributable
        to an unlucky draw.

        This matters for a Poisson draw and much less for a cube draw, because
        `Study.draw(fixed_size=True)` already balances on `G` at selection time
        and so arrives calibrated. Use it when you took the simpler sampler and
        want the balance back.

        Returned as shrinkage factors rather than weights so they compose with
        `estimate`: the calibrated weight for entity `j` is `alpha_j / pi_j`.
        Note that calibration is *asymptotically* unbiased, not exactly so --
        the adjustment depends on the realized sample -- which is why the
        resulting estimate reports itself as no longer strictly unbiased.

        Args:
            sample: The realized draw to calibrate.

        Returns:
            Shrinkage factors indexed by the sampled entities, to pass as
            `alpha` to `estimate`.
        """
        self._check_sample_frame(sample)
        pi = sample.probabilities
        weights = calibrate_weights(pi, self.influence.G, sample.items)
        return (weights * pi.reindex(sample.items)).rename("alpha")

    def estimate(
        self,
        sample: LabelSample,
        labels: pd.Series | pd.DataFrame,
        *,
        proxy: pd.DataFrame | None = None,
        alpha: pd.Series | float = 1.0,
        variance: str = "auto",
        row_model_se: bool = False,
    ) -> Estimate:
        """Estimate `B` from a realized sample.

        Args:
            sample: The realized draw. A `Ranking` is rejected here on purpose.
            labels: Observed labels for the sampled entities. Arbitrary
                `q`-vectors are accepted, so soft labels -- a coder
                distribution, a probability vector over classes -- need no
                special handling.
            proxy: Optional cheap prediction for every entity, enabling the
                difference estimator.
            alpha: Shrinkage on the sampled correction; 1.0 is unbiased.
            variance: `"auto"` matches the estimator to the draw: design-unbiased
                `"poisson"` for independent draws and approximate `"balanced"`
                for cube draws. Explicit choices are available for comparisons.
            row_model_se: Also compute the homoskedastic row-level model SE.
                False keeps inference conditional on the observed rows.

        Returns:
            The estimate and design uncertainty, with model uncertainty if asked.

        Raises:
            TypeError: If handed a `Ranking` or a bare list of entities.
        """
        if isinstance(sample, Ranking):
            raise TypeError(
                "estimate() needs a LabelSample, not a Ranking. A ranking has no "
                "inclusion probabilities, so there is no unbiased weight to "
                "apply. Build a design with Study.design(budget) and realize it "
                "with Study.draw()."
            )
        if not isinstance(sample, LabelSample):
            raise TypeError(
                f"estimate() needs a LabelSample, got {type(sample).__name__}. A "
                f"bare set of entities does not say whether it came from top-B, "
                f"a random draw, or a hand edit, and the weights differ."
            )
        self._check_sample_frame(sample)
        return estimate(
            self.influence,
            sample,
            labels,
            counts=self.counts,
            X=self.X,
            proxy=proxy,
            alpha=alpha,
            variance=variance,
            row_model_se=row_model_se,
        )

    def _check_sample_frame(self, sample: LabelSample) -> None:
        """Reject a draw made for different counts, features, or entity frame."""
        if sample.frame_fingerprint != self.frame_fingerprint:
            raise ValueError(
                "sample was not drawn from this Study frame; use the same "
                "counts and features used to draw it"
            )

    def __repr__(self) -> str:
        """Summarize the study.

        Returns:
            A one-line description including the degeneracy diagnostic.
        """
        d = self.diagnostics
        return (
            f"Study(units={d['n_units']}, entities={d['n_entities']}, "
            f"density={d['density']:.4f}, cond={d['condition_number']:.2e}, "
            f"directions={d['effective_directions']})"
        )

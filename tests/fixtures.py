"""Shared populations for the design tests.

Two generators, and the difference between them is the whole point. `panel`
gives every entity its own direction of influence, which is what a real usage
panel looks like. `rank_one` makes every entity's influence point the same way,
which is what the pre-3.0 evaluation generated without noticing -- and on that
population every design criterion ranks entities identically, so a benchmark run
there reports every method as equivalent no matter what it measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def panel(
    n: int = 400,
    m: int = 250,
    p: int = 3,
    *,
    zipf: float = 1.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a usage panel whose entities appeal to different kinds of unit.

    Args:
        n: Number of units.
        m: Number of entities.
        p: Number of features, including the intercept.
        zipf: Dispersion of entity popularity on the log scale; larger is more
            heavy-tailed, as browsing data is.
        seed: Seed.

    Returns:
        The count matrix and the feature matrix.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p - 1))
    X = pd.DataFrame(
        np.c_[np.ones(n), Z],
        columns=["const", *[f"x{i}" for i in range(1, p)]],
        index=[f"u{i}" for i in range(n)],
    )
    loadings = rng.normal(size=(m, p - 1))
    popularity = np.exp(rng.normal(0, zipf, size=m))
    rate = np.exp(0.4 + Z @ loadings.T + np.log(popularity) - np.log(popularity).mean())
    counts = pd.DataFrame(
        rng.poisson(rate), index=X.index, columns=[f"e{j}" for j in range(m)]
    )
    return counts, X


def rank_one(
    n: int = 400, m: int = 250, p: int = 3, *, mean_count: float = 20.0, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a panel whose expected counts are `activity_i * popularity_j`.

    Args:
        n: Number of units.
        m: Number of entities.
        p: Number of features.
        mean_count: Average cell count. The diagnostic this fixture exercises
            has a noise floor of roughly `1/sqrt(mean_count)`, so the comparison
            is only meaningful at a matched count scale.
        seed: Seed.

    Returns:
        The count matrix and the feature matrix.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        np.c_[np.ones(n), rng.normal(size=(n, p - 1))],
        columns=["const", *[f"x{i}" for i in range(1, p)]],
        index=[f"u{i}" for i in range(n)],
    )
    activity = np.exp(rng.normal(0, 0.5, size=n))
    popularity = np.exp(rng.normal(0, 1.2, size=m))
    rate = np.outer(activity, popularity)
    rate = rate * mean_count / rate.mean()
    counts = pd.DataFrame(
        rng.poisson(rate),
        index=X.index,
        columns=[f"e{j}" for j in range(m)],
    )
    return counts, X


def one_hot_labels(
    entities: list[str], classes: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Assign each entity to exactly one class.

    Args:
        entities: Entity identifiers.
        classes: Number of mutually exclusive classes.
        seed: Seed.

    Returns:
        A one-hot label frame.
    """
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, classes, size=len(entities))
    return pd.DataFrame(
        np.eye(classes)[draw],
        index=pd.Index(entities),
        columns=[f"c{k}" for k in range(classes)],
    )


def exact_truth(study: object, labels: pd.DataFrame) -> np.ndarray:
    """Compute the complete-label coefficient matrix.

    Args:
        study: A `Study`.
        labels: True labels for every entity in the frame.

    Returns:
        The `(p, q)` finite-population truth.
    """
    infl = study.influence  # type: ignore[attr-defined]
    return infl.H @ labels.loc[infl.items].to_numpy(float)

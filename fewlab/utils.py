"""
Common utility functions for fewlab.

This module provides shared helper functions to reduce code duplication
across the library.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .constants import DIVISION_EPS


def _get_random_generator(
    random_state: None | int | np.random.Generator,
) -> np.random.Generator:
    """
    Create or validate a numpy random generator.

    Args:
        random_state: Random state specification. Can be:
            - None: Creates a new generator with default seed
            - int: Creates a generator with the given seed
            - np.random.Generator: Returns the generator as-is

    Returns:
        A numpy random number generator.

    Examples:
        >>> rng = _get_random_generator(42)
        >>> rng = _get_random_generator(None)
        >>> existing_rng = np.random.default_rng(123)
        >>> rng = _get_random_generator(existing_rng)
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    else:
        return np.random.default_rng(random_state)


def compute_g_matrix(counts: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """
    Compute the regression projection matrix G = X^T V.

    Args:
        counts: Count matrix with units as rows and items as columns (shape (n, m)).
        X: Covariate matrix aligned with `counts.index` (shape (n, p)).

    Returns:
        Regression projections g_j = X^T v_j for all items (shape (p, m)).

    Raises:
        ValueError: If indices are misaligned or every row sum is zero.

    Notes:
        This helper normalizes counts into the matrix V where v_j = counts_j / row_totals.
    """
    if not counts.index.equals(X.index):
        raise ValueError("counts.index must align with X.index")

    T = counts.sum(axis=1).to_numpy(float)
    if np.any(T == 0):
        # Filter out zero-sum rows
        keep = T > 0
        counts = counts.loc[keep]
        X = X.loc[keep]
        T = T[keep]
        if len(T) == 0:
            raise ValueError("All rows have zero totals; nothing to compute.")

    V = counts.to_numpy(float) / (T[:, None] + DIVISION_EPS)  # (n, m)
    Xn = X.to_numpy(float)
    G: np.ndarray = Xn.T @ V  # (p, m)

    return G


def validate_fraction(value: float, name: str = "fraction") -> None:
    """
    Validate that a value is a proper fraction in (0, 1).

    Args:
        value: Value to validate.
        name: Label for error messages.

    Raises:
        TypeError: If the value is not numeric.
        ValueError: If the value is not strictly between 0 and 1.
    """
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def compute_horvitz_thompson_weights(
    pi: pd.Series, selected: pd.Index | Sequence[str]
) -> pd.Series:
    """
    Compute Horvitz-Thompson weights (1/pi) for selected items.

    Args:
        pi: Inclusion probabilities for all items.
        selected: Identifiers for the sampled items.

    Returns:
        Horvitz-Thompson weights indexed by the selected items.
    """
    ht_weights = 1.0 / (pi + DIVISION_EPS)
    assert isinstance(ht_weights, pd.Series), "Expected pd.Series from division"
    if isinstance(selected, pd.Index):
        return ht_weights.reindex(selected)
    else:
        return ht_weights.reindex(list(selected))


def align_indices(*dataframes: pd.DataFrame | pd.Series) -> bool:
    """
    Check if all dataframes/series have aligned indices.

    Args:
        *dataframes: Data objects to compare.

    Returns:
        True if all indices match, otherwise False.
    """
    if len(dataframes) < 2:
        return True

    first_index = dataframes[0].index
    return all(df.index.equals(first_index) for df in dataframes[1:])


def get_item_positions(
    items: pd.Index | Sequence[str], reference: pd.Index
) -> np.ndarray:
    """
    Map item names to their positions in a reference index.

    Args:
        items: Item identifiers to map.
        reference: Index containing the full set of items.

    Returns:
        Integer positions of `items` within `reference`.

    Raises:
        ValueError: If any item is missing from the reference index.
    """
    col_to_pos = {col: i for i, col in enumerate(reference)}
    try:
        positions = np.array([col_to_pos[item] for item in items], dtype=int)
    except KeyError as e:
        raise ValueError(f"Item {e} not found in reference index") from e
    return positions

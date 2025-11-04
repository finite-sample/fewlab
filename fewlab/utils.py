"""
Common utility functions for fewlab.

This module provides shared helper functions to reduce code duplication
across the library.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence
from .constants import DIVISION_EPS


def compute_g_matrix(counts: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """
    Compute the regression projection matrix G = X^T V.

    Parameters
    ----------
    counts : pd.DataFrame, shape (n, m)
        Count matrix with rows=units, columns=items.
    X : pd.DataFrame, shape (n, p)
        Covariate matrix, index must align with counts.index.

    Returns
    -------
    np.ndarray, shape (p, m)
        Regression projections g_j = X^T v_j for all items.

    Notes
    -----
    This is a common computation used in multiple modules.
    V is the normalized count matrix where v_j = counts_j / row_totals.
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
    G = Xn.T @ V  # (p, m)

    return G


def validate_fraction(value: float, name: str = "fraction") -> None:
    """
    Validate that a value is a proper fraction in (0, 1).

    Parameters
    ----------
    value : float
        The value to validate.
    name : str
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If value is not in the interval (0, 1).
    """
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def compute_horvitz_thompson_weights(
    pi: pd.Series, selected: pd.Index | Sequence[str]
) -> pd.Series:
    """
    Compute Horvitz-Thompson weights (1/pi) for selected items.

    Parameters
    ----------
    pi : pd.Series
        Inclusion probabilities for all items.
    selected : pd.Index or sequence of str
        Selected item identifiers.

    Returns
    -------
    pd.Series
        HT weights indexed by selected items.
    """
    return (1.0 / (pi + DIVISION_EPS)).reindex(selected)


def align_indices(*dataframes: pd.DataFrame | pd.Series) -> bool:
    """
    Check if all dataframes/series have aligned indices.

    Parameters
    ----------
    *dataframes : pd.DataFrame or pd.Series
        DataFrames or Series to check.

    Returns
    -------
    bool
        True if all indices are equal, False otherwise.
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

    Parameters
    ----------
    items : pd.Index or sequence of str
        Items to map.
    reference : pd.Index
        Reference index containing all items.

    Returns
    -------
    np.ndarray
        Integer positions of items in reference.

    Raises
    ------
    ValueError
        If any item is not found in reference.
    """
    col_to_pos = {col: i for i, col in enumerate(reference)}
    try:
        positions = np.array([col_to_pos[item] for item in items], dtype=int)
    except KeyError as e:
        raise ValueError(f"Item {e} not found in reference index")
    return positions

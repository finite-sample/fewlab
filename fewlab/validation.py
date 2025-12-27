"""
Input validation utilities for fewlab functions.

This module provides comprehensive validation for function parameters,
data alignment checks, and helpful error messages with suggestions.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Custom exception for validation errors with helpful suggestions."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        if suggestion:
            full_message = f"{message}\n\nSuggestion: {suggestion}"
        else:
            full_message = message
        super().__init__(full_message)


def validate_counts_matrix(counts: Any, name: str = "counts") -> pd.DataFrame:
    """
    Validate counts matrix input.

    Args:
        counts: Candidate counts matrix to validate.
        name: Parameter name for error messages.

    Returns:
        Counts matrix with invalid rows filtered.

    Raises:
        ValidationError: If the matrix is empty, non-numeric, or otherwise invalid.
    """
    if not isinstance(counts, pd.DataFrame):
        raise ValidationError(
            f"{name} must be a pandas DataFrame, got {type(counts).__name__}",
            "Convert your counts data to a pandas DataFrame with "
            "rows=units and columns=items",
        )

    if counts.empty:
        raise ValidationError(
            f"{name} DataFrame is empty",
            "Ensure your counts data has both rows and columns",
        )

    if not pd.api.types.is_numeric_dtype(counts.values.dtype):
        if not all(
            pd.api.types.is_numeric_dtype(counts[col]) for col in counts.columns
        ):
            raise ValidationError(
                f"{name} must contain only numeric values",
                "Check for string values or convert non-numeric columns to numeric types",
            )

    # Check for negative values
    if (counts < 0).any().any():
        raise ValidationError(
            f"{name} contains negative values",
            "Counts should be non-negative. Check your data for errors.",
        )

    # Check for all-zero rows/columns
    zero_rows = (counts == 0).all(axis=1)
    zero_cols = (counts == 0).all(axis=0)

    if zero_rows.any():
        n_zero = zero_rows.sum()
        warnings.warn(
            f"{name} has {n_zero} rows with all zeros. "
            "These rows will be excluded from analysis.",
            UserWarning,
            stacklevel=3,
        )
        # Filter out zero rows automatically
        counts = counts.loc[~zero_rows]
        if counts.empty:
            raise ValidationError(
                f"All rows in {name} are zero",
                "Ensure your count matrix has non-zero usage data",
            )

    if zero_cols.any():
        n_zero = zero_cols.sum()
        warnings.warn(
            f"{name} has {n_zero} columns with all zeros. "
            "These items will not be selectable.",
            UserWarning,
            stacklevel=3,
        )

    return counts


def validate_features_matrix(X: Any, name: str = "X") -> pd.DataFrame:
    """
    Validate features/covariates matrix.

    Args:
        X: Candidate features matrix.
        name: Parameter name for error messages.

    Returns:
        Validated features matrix.

    Raises:
        ValidationError: If the matrix is empty, non-numeric, or has missing data.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValidationError(
            f"{name} must be a pandas DataFrame, got {type(X).__name__}",
            "Convert your features data to a pandas DataFrame with "
            "rows=units and columns=features",
        )

    if X.empty:
        raise ValidationError(
            f"{name} DataFrame is empty",
            "Ensure your features data has both rows and columns",
        )

    # Check for numeric data
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        non_numeric = [
            col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        ]
        raise ValidationError(
            f"{name} contains non-numeric columns: {non_numeric}",
            "Use pd.get_dummies() for categorical variables or convert to numeric types",
        )

    # Check for missing values
    has_nulls = X.isnull().any()
    if isinstance(has_nulls, pd.Series) and has_nulls.any():
        missing_cols = X.columns[has_nulls].tolist()
        raise ValidationError(
            f"{name} contains missing values in columns: {missing_cols}",
            "Use fillna() or drop missing values before calling fewlab functions",
        )

    # Check for constant columns
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        warnings.warn(
            f"{name} has constant columns: {constant_cols}. "
            "These provide no information for the analysis.",
            UserWarning,
            stacklevel=3,
        )

    return X


def validate_data_alignment(
    counts: pd.DataFrame, X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate that counts and features matrices are properly aligned and align them.

    Args:
        counts: Counts matrix.
        X: Features matrix.

    Returns:
        Tuple of `(counts, X)` aligned on the same index.

    Raises:
        ValidationError: If the matrices cannot be aligned.
    """
    # If counts was filtered (e.g., zero rows removed), align X accordingly
    if not counts.index.equals(X.index):
        # Check if counts index is a subset of X index (common after filtering)
        if counts.index.isin(X.index).all():
            # Align X to match counts
            X = X.loc[counts.index]
        elif X.index.isin(counts.index).all():
            # Align counts to match X (less common)
            counts = counts.loc[X.index]
        else:
            # More complex mismatch
            common_index = counts.index.intersection(X.index)
            if len(common_index) == 0:
                raise ValidationError(
                    "counts and X have no common index values",
                    "Ensure both DataFrames represent the same units (users, respondents, etc.)",
                )
            elif len(common_index) < max(len(counts), len(X)) * 0.5:
                raise ValidationError(
                    f"Only {len(common_index)} of {max(len(counts), len(X))} indices match between counts and X",
                    "Check that both DataFrames represent the same units",
                )
            else:
                # Use common subset
                warnings.warn(
                    f"Using {len(common_index)} common rows from counts ({len(counts)}) and X ({len(X)})",
                    UserWarning,
                    stacklevel=3,
                )
                counts = counts.loc[common_index]
                X = X.loc[common_index]

    return counts, X


def validate_budget(budget: Any, max_budget: int, name: str = "budget") -> int:
    """
    Validate budget parameter.

    Args:
        budget: Proposed sample size.
        max_budget: Maximum allowable sample size.
        name: Parameter name for error messages.

    Returns:
        Validated integer budget.

    Raises:
        ValidationError: If the value is not an integer within the allowed range.
    """
    if not isinstance(budget, int | np.integer):
        raise ValidationError(
            f"{name} must be an integer, got {type(budget).__name__}",
            f"Use an integer value between 1 and {max_budget}",
        )

    budget = int(budget)

    if budget <= 0:
        raise ValidationError(
            f"{name} must be positive, got {budget}",
            f"Use a value between 1 and {max_budget}",
        )

    if budget > max_budget:
        raise ValidationError(
            f"{name} ({budget}) exceeds maximum possible ({max_budget})",
            f"Reduce {name} to at most {max_budget} (the number of available items)",
        )

    return budget


def validate_probability_series(
    pi: Any, expected_index: pd.Index | None = None, name: str = "pi"
) -> pd.Series:
    """
    Validate probability series.

    Args:
        pi: Candidate probability vector.
        expected_index: Optional index that must match.
        name: Parameter name for error messages.

    Returns:
        Probability series with validated values and index.

    Raises:
        ValidationError: If the data is non-numeric, empty, or misaligned.
    """
    if not isinstance(pi, pd.Series):
        raise ValidationError(
            f"{name} must be a pandas Series, got {type(pi).__name__}",
            "Use a pandas Series with item identifiers as index and probabilities as values",
        )

    if pi.empty:
        raise ValidationError(
            f"{name} Series is empty", "Ensure the probability series has values"
        )

    # Check for numeric values
    if not pd.api.types.is_numeric_dtype(pi.dtype):
        raise ValidationError(
            f"{name} must contain numeric values",
            "Probabilities should be between 0 and 1",
        )

    # Check for valid probability range
    if (pi < 0).any() or (pi > 1).any():
        invalid_count = ((pi < 0) | (pi > 1)).sum()
        raise ValidationError(
            f"{name} has {invalid_count} values outside [0,1] range",
            "Probabilities must be between 0 and 1 inclusive",
        )

    # Check index alignment if expected
    if expected_index is not None:
        if not pi.index.equals(expected_index):
            missing = expected_index.difference(pi.index)
            extra = pi.index.difference(expected_index)

            msg_parts = [f"{name} index doesn't match expected items"]
            if len(missing) > 0:
                missing_list = list(missing)[:5]
                msg_parts.append(
                    f"Missing items: {missing_list}{' ...' if len(missing) > 5 else ''}"
                )
            if len(extra) > 0:
                extra_list = list(extra)[:5]
                msg_parts.append(
                    f"Extra items: {extra_list}{' ...' if len(extra) > 5 else ''}"
                )

            raise ValidationError(
                ". ".join(msg_parts),
                "Ensure probability series has the same items as the counts matrix columns",
            )

    return pi


def validate_item_selection(
    selected: Any, available_items: pd.Index, name: str = "selected"
) -> list[str]:
    """
    Validate item selection input.

    Args:
        selected: Items requested for inclusion.
        available_items: Items available to select from.
        name: Parameter name for error messages.

    Returns:
        Selected items as a list of strings.

    Raises:
        ValidationError: If the selection is empty, malformed, or references unknown items.
    """
    # Convert to list of strings
    if isinstance(selected, pd.Index):
        selected_list = selected.tolist()
    elif isinstance(selected, list | tuple | np.ndarray):
        selected_list = list(selected)
    elif isinstance(selected, pd.Series):
        selected_list = (
            selected.index.tolist() if selected.dtype == "bool" else selected.tolist()
        )
    else:
        raise ValidationError(
            f"{name} must be a list, pd.Index, or array-like of item identifiers",
            "Use a list of item names or a pandas Index",
        )

    if len(selected_list) == 0:
        raise ValidationError(f"{name} is empty", "Select at least one item")

    # Convert to strings for comparison
    selected_list = [str(item) for item in selected_list]
    available_list = [str(item) for item in available_items]

    # Check for unknown items
    unknown_items = set(selected_list) - set(available_list)
    if unknown_items:
        unknown_sample = list(unknown_items)[:5]
        raise ValidationError(
            f"{name} contains unknown items: {unknown_sample}{' ...' if len(unknown_items) > 5 else ''}",
            "Ensure all selected items exist in the counts matrix columns",
        )

    return selected_list


def warn_deprecated_parameter(
    old_name: str, new_name: str, version: str = "2.0.0"
) -> None:
    """
    Issue deprecation warning for renamed parameters.

    Args:
        old_name: Deprecated parameter name.
        new_name: Replacement parameter name.
        version: Version where the old name will be removed.
    """
    warnings.warn(
        f"Parameter '{old_name}' is deprecated and will be removed in version {version}. "
        f"Use '{new_name}' instead.",
        FutureWarning,
        stacklevel=4,  # Go up to the calling function
    )

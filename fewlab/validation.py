"""Input validation utilities for fewlab functions.

This module provides comprehensive validation for function parameters,
data alignment checks, and helpful error messages with suggestions.
"""

import warnings
from typing import Any

import pandas as pd


class ValidationError(ValueError):
    """Custom exception for validation errors with helpful suggestions."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Build the error, appending ``suggestion`` to the message when given."""
        if suggestion:
            full_message = f"{message}\n\nSuggestion: {suggestion}"
        else:
            full_message = message
        super().__init__(full_message)


def validate_counts_matrix(counts: Any, name: str = "counts") -> pd.DataFrame:
    """Validate counts matrix input.

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

    if not pd.api.types.is_numeric_dtype(counts.values.dtype) and not all(
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
    """Validate features/covariates matrix.

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
            "Use pd.get_dummies() for categorical variables or convert to "
            "numeric types",
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
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]

    if constant_cols:
        warnings.warn(
            f"{name} has constant columns: {constant_cols}. "
            "These provide no information for the analysis.",
            UserWarning,
            stacklevel=3,
        )

    return X

"""Input validation, which is the only user-facing surface most people hit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fewlab.counts import CountMatrix
from fewlab.validation import (
    ValidationError,
    validate_counts_matrix,
    validate_features_matrix,
)


def test_a_non_frame_is_rejected() -> None:
    """An array has no entity names, so it cannot name what to label."""
    with pytest.raises(ValidationError, match="DataFrame"):
        validate_counts_matrix(np.zeros((4, 3)))


def test_an_empty_matrix_is_rejected() -> None:
    """There is nothing to design over."""
    with pytest.raises(ValidationError, match="empty"):
        validate_counts_matrix(pd.DataFrame())


def test_negative_counts_are_rejected() -> None:
    """A negative share is not a share."""
    frame = pd.DataFrame({"a": [1.0, -2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValidationError, match="negative"):
        validate_counts_matrix(frame)


def test_non_numeric_counts_are_rejected() -> None:
    """Counting requires numbers."""
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    with pytest.raises(ValidationError, match="numeric"):
        validate_counts_matrix(frame)


def test_units_with_no_activity_are_dropped_with_a_warning() -> None:
    """They contribute no shares, but the caller should know they went."""
    frame = pd.DataFrame({"a": [1.0, 0.0], "b": [2.0, 0.0]})
    with pytest.warns(UserWarning, match="zero"):
        cleaned = validate_counts_matrix(frame)
    assert len(cleaned) == 1


def test_missing_features_are_rejected() -> None:
    """An imputed covariate would silently change the estimand."""
    X = pd.DataFrame({"const": [1.0, 1.0], "x": [0.5, np.nan]})
    with pytest.raises(ValidationError, match=r"missing|NaN"):
        validate_features_matrix(X)


def test_non_numeric_features_are_rejected() -> None:
    """Categoricals have to be made explicit as dummies by the caller."""
    X = pd.DataFrame({"const": [1.0, 1.0], "group": ["a", "b"]})
    with pytest.raises(ValidationError, match="numeric"):
        validate_features_matrix(X)


def test_long_format_rejects_negative_counts() -> None:
    """The sparse path validates what the dense path does."""
    long = pd.DataFrame({"u": ["a"], "e": ["x"], "n": [-1.0]})
    with pytest.raises(ValueError, match="non-negative"):
        CountMatrix.from_long(long, unit="u", entity="e", count="n")


def test_long_format_rejects_entities_outside_the_declared_frame() -> None:
    """Silently extending the population would change what B means."""
    long = pd.DataFrame({"u": ["a", "a"], "e": ["x", "z"], "n": [1.0, 2.0]})
    with pytest.raises(ValueError, match="absent"):
        CountMatrix.from_long(
            long, unit="u", entity="e", count="n", entities=pd.Index(["x", "y"])
        )


def test_the_declared_frame_may_include_untouched_entities() -> None:
    """They belong to the population and simply contribute nothing."""
    long = pd.DataFrame({"u": ["a", "b"], "e": ["x", "y"], "n": [1.0, 2.0]})
    cm = CountMatrix.from_long(
        long, unit="u", entity="e", count="n", entities=pd.Index(["x", "y", "z"])
    )
    assert cm.shape == (2, 3)
    assert cm.dense_columns(np.array([2])).sum() == 0.0

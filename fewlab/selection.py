from __future__ import annotations

import numpy as np
import pandas as pd


def topk(arr: np.ndarray, k: int, *, index: pd.Index | None = None) -> pd.Index:
    """
    Return indices of the top-k entries of ``arr`` in descending order.

    Args:
        arr: Array of scores to rank.
        k: Number of entries to keep.
        index: Optional index to map positions back to labels.

    Returns:
        Index of the top-k entries ordered by decreasing value.
    """
    if k <= 0:
        return pd.Index([], name="topk_indices")

    if k >= arr.size:
        idx = np.argsort(-arr)
    else:
        # partial select is O(n)
        idx = np.argpartition(-arr, kth=k - 1)[:k]
        # sort those k by value
        idx = idx[np.argsort(-arr[idx])]

    if index is not None:
        result = index[idx]
        result.name = "topk_indices"
        return result
    else:
        return pd.Index(idx, name="topk_indices")

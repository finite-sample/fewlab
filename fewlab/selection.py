from __future__ import annotations
import numpy as np

def topk(arr: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k entries of arr (descending)."""
    if k <= 0:
        return np.empty(0, dtype=int)
    if k >= arr.size:
        return np.argsort(-arr)
    # partial select is O(n)
    idx = np.argpartition(-arr, kth=k-1)[:k]
    # sort those k by value
    return idx[np.argsort(-arr[idx])]

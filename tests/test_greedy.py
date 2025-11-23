import numpy as np
import pandas as pd

from fewlab import greedy_aopt_selection, items_to_label


def test_greedy_selection_basic():
    """Test that greedy selection returns correct number of items."""
    n, m, p = 100, 20, 3
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    counts = pd.DataFrame(
        rng.poisson(5, size=(n, m)), columns=[f"item{j}" for j in range(m)]
    )

    K = 5
    selected = greedy_aopt_selection(counts, X, K)

    assert len(selected) == K
    assert len(set(selected)) == K  # No duplicates
    assert all(item in counts.columns for item in selected)


def test_greedy_vs_topk_performance():
    """
    Test that greedy selection performs at least as well as Top-K
    in terms of A-optimality (trace of inverse information matrix).
    Note: Greedy is not guaranteed to be globally optimal, but usually beats Top-K
    when items are correlated.
    """
    n, m, p = 100, 50, 5
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])

    # Create correlated items
    # Items 0 and 1 are identical -> Top-K might pick both, Greedy should pick one
    counts_vals = rng.poisson(2, size=(n, m))
    counts_vals[:, 1] = counts_vals[:, 0]
    counts = pd.DataFrame(counts_vals, columns=[f"item{j}" for j in range(m)])

    K = 10

    # Run both methods
    sel_greedy = greedy_aopt_selection(counts, X, K)
    sel_topk = items_to_label(counts, X, K)

    # Compute A-optimality objective: Trace((X'X + sum g g')^-1)
    # We need to reconstruct g vectors
    from fewlab.core import _influence

    inf = _influence(counts, X)
    G = inf.g
    col_to_idx = {c: i for i, c in enumerate(counts.columns)}

    def compute_trace_inv(selected_items):
        indices = [col_to_idx[c] for c in selected_items]
        G_sel = G[:, indices]
        # M = X'X (approx) + G G'
        # Actually, the objective we optimized was adding G G' to initial M
        # Let's just look at the information matrix of the selected items alone?
        # Or the posterior variance?
        # The greedy step minimized Trace((M + g g')^-1).
        # So we compare Trace((Ridge + G_sel G_sel')^-1).

        ridge = 1e-4 * np.eye(p)
        # We are selecting items to LABEL. The information gain is from the LABELS.
        # If we assume the labels y are related to X via y ~ X beta,
        # then observing item j gives us information... wait.
        # The formulation in `core.py` is about "shares".
        # The "influence" w_j is defined as g_j' (X'X)^-1 g_j.
        # This is the leverage of the *projection* of item j onto X space.
        # If we want to estimate beta from the selected items, we want to maximize information.
        # Information matrix ~ sum_j g_j g_j^T.

        M = ridge + G_sel @ G_sel.T
        return np.trace(np.linalg.inv(M))

    trace_greedy = compute_trace_inv(sel_greedy)
    trace_topk = compute_trace_inv(sel_topk)

    print(f"Trace Greedy: {trace_greedy}")
    print(f"Trace Top-K: {trace_topk}")

    # Greedy should be better (lower trace)
    assert trace_greedy <= trace_topk + 1e-9


def test_sparse_support():
    """Test that the code handles sparse dataframes correctly."""
    n, m, p = 50, 20, 3
    rng = np.random.default_rng(42)

    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])

    # Create sparse counts
    counts_dense = pd.DataFrame(
        rng.poisson(0.5, size=(n, m)), columns=[f"item{j}" for j in range(m)]
    )
    counts_sparse = counts_dense.astype(pd.SparseDtype(int, 0))

    K = 5
    # Should not raise error
    sel = greedy_aopt_selection(counts_sparse, X, K)
    assert len(sel) == K

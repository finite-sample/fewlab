import numpy as np
import pandas as pd
from fewlab import pi_aopt_for_budget, balanced_fixed_size, row_se_min_labels
from .data_synth import make_synth


def test_pi_aopt_budget_sums():
    counts, X = make_synth(n=80, m=120, p=5, seed=11)
    K = 30
    pi = pi_aopt_for_budget(counts, X, K, pi_min=1e-4)
    assert abs(pi.sum() - K) < 1e-2
    assert (pi.values >= 1e-4 - 1e-12).all() and (pi.values <= 1.0 + 1e-12).all()


def test_balanced_fixed_size_returns_K():
    counts, X = make_synth(n=60, m=100, p=4, seed=7)
    K = 25
    # get g via influence path (private) using the pi function as shortcut
    from fewlab.core import _influence

    inf = _influence(counts, X)
    pi = pd.Series(np.clip(np.sqrt(np.maximum(inf.w, 0)), 1e-4, 1.0), index=inf.cols)
    sel = balanced_fixed_size(pi, inf.g, K, seed=1)
    assert len(sel) == K
    assert set(sel).issubset(set(inf.cols))


def test_row_se_min_labels_basic():
    counts, X = make_synth(n=50, m=90, p=5, seed=21)
    T = counts.sum(axis=1).to_numpy(float)
    sumq = ((counts.to_numpy(float) / np.maximum(T[:, None], 1e-12)) ** 2).sum(axis=1)
    # lax caps => tiny budget
    eps2 = 0.2 * ((1.0 / 1e-4) - 1.0) * sumq
    pi = row_se_min_labels(counts, eps2, pi_min=1e-4, max_iter=2000)
    assert (pi.values >= 1e-4).all() and (pi.values <= 1.0).all()

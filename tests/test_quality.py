import numpy as np
import pandas as pd
from fewlab import items_to_label
from .data_synth import make_synth

def compute_w(counts, X, ridge=None):
    T = counts.sum(axis=1).to_numpy(float)
    keep = T > 0
    counts = counts.loc[keep]; X = X.loc[keep]; T = T[keep]
    V = counts.to_numpy(float) / T[:, None]
    Xn = X.to_numpy(float)
    XtX = Xn.T @ Xn
    if ridge is None:
        cond = np.linalg.cond(XtX)
        if not np.isfinite(cond) or cond > 1e12:
            ridge = 1e-8
    if ridge:
        XtX = XtX + ridge * np.eye(XtX.shape[0])
    XtX_inv = np.linalg.inv(XtX)
    G = Xn.T @ V
    w = np.einsum("jp,pk,kj->j", G.T, XtX_inv, G)
    return w, counts.columns

def test_selected_items_have_higher_w():
    counts, X = make_synth(n=120, m=200, p=6, seed=1234)
    K = 30
    chosen = items_to_label(counts, X, K=K)
    w, cols = compute_w(counts, X)
    w_map = {c: w[i] for i, c in enumerate(cols)}
    w_chosen = np.median([w_map[c] for c in chosen])
    w_rest = np.median([w_map[c] for c in cols if c not in chosen])
    assert w_chosen >= w_rest, "Selected items should have larger median w"

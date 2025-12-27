from __future__ import annotations

import numpy as np
import pandas as pd


def make_synth(n=300, m=400, p=6, random_state=123):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.normal(size=(n, p - 1)), columns=[f"x{k}" for k in range(p - 1)]
    )
    X = (X - X.mean()) / (X.std() + 1e-9)
    X.insert(0, "intercept", 1.0)

    base = rng.lognormal(mean=0.0, sigma=1.0, size=m)
    Theta = rng.normal(scale=0.6, size=(m, p))
    L = 0.6 * (X.to_numpy() @ Theta.T) + np.log(base)[None, :]
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    T = rng.poisson(120, size=n) + 1
    C = np.vstack([rng.multinomial(T[i], P[i]) for i in range(n)])
    counts = pd.DataFrame(C, columns=[f"item{j}" for j in range(m)])
    counts.index = X.index
    return counts, X

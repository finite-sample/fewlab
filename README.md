## fewlab: fewest items to label for unbiased OLS on shares

[![PyPI version](https://img.shields.io/pypi/v/fewlab.svg)](https://pypi.org/project/fewlab/)
[![Downloads](https://pepy.tech/badge/fewlab)](https://pepy.tech/project/fewlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Pick the **few**est items to **lab**el so you can run an OLS regression on per‑row trait shares with the highest precision.

## Why this exists

You have a nonnegative counts matrix $C \in \mathbb{R}_+^{n\times m}$: rows are units (people/devices/firms), columns are items (domains/apps/products).
Each item $j$ has an unknown binary label $a_j\in\{0,1\}$ (e.g., “unsafe”, “brand X”). For each row $i$, the true trait share is

$$
y_i \;=\; \sum_{j=1}^m \frac{c_{ij}}{T_i}\, a_j, \qquad T_i=\sum_j c_{ij}.
$$

You want to regress $y$ on covariates $X$ to get coefficients $\beta^\*=(X^\top X)^{-1}X^\top y$.
Labeling all $m$ items is expensive, so the question is: **which items should we label** to keep the OLS precise (and, if desired, design‑unbiased)?

`fewlab` provides a tiny API that takes your `counts` and `X`, computes how much each item matters for the regression, and returns a **ranked set of items** to label under a fixed budget $K$. The ranking comes from an A‑optimal design criterion: items whose label noise projects strongly onto the span of $X$ get priority.

## What it computes (intuition, not heavy math)

* Normalize exposures: $v_j = c_{\cdot j}/T \in \mathbb{R}^n$.
* Project onto regression directions: $g_j = X^\top v_j \in \mathbb{R}^p$.
* Weight by OLS geometry: $w_j = g_j^\top (X^\top X)^{-1} g_j \ge 0$.

Large $w_j$ means item $j$ can move the OLS coefficients the most.
The **square‑root rule** says that under a fixed labeling budget $K$, inclusion probabilities should scale like $\pi_j \propto \sqrt{w_j}$. As a simple, deterministic approximation, **pick the top‑$K$** items by $w_j$.

> **Unbiasedness note.** Design‑unbiased OLS via Horvitz–Thompson (HT) requires **randomization** with positive inclusion probability $\pi_j>0$ for *every* item that can contribute. A purely deterministic top‑$K$ list is a great prioritization heuristic, but strictly speaking it doesn’t by itself give HT‑unbiased shares for the full item universe. See “Design‑unbiased pipeline” below for a short, practical recipe.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `pandas`. Tests use `pytest`.


## Minimal usage (deterministic recommendation list)

This gets you a concrete list of **K item IDs** to send to the labeling team.

```python
import numpy as np
import pandas as pd
from fewlab import items_to_label

# 1) counts (n x m) and X (n x p)
# counts.index must equal X.index
# Example: quick synthetic data
rng = np.random.default_rng(0)
n, m, p = 200, 500, 5

X = pd.DataFrame(rng.normal(size=(n, p-1)), columns=[f"x{k}" for k in range(p-1)])
X = (X - X.mean()) / (X.std() + 1e-9)
X.insert(0, "intercept", 1.0)

base_pop = rng.lognormal(mean=0.0, sigma=1.0, size=m)
Theta = rng.normal(scale=0.5, size=(m, p))         # item × feature tilt
L = 0.5 * (X.to_numpy() @ Theta.T) + np.log(base_pop)[None, :]
P = np.exp(L - L.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)
T = rng.poisson(120, size=n) + 1
C = np.vstack([rng.multinomial(T[i], P[i]) for i in range(n)])
counts = pd.DataFrame(C, columns=[f"item{j}" for j in range(m)], index=X.index)

# 2) choose K items to label (deterministically)
K = 80
item_ids = items_to_label(counts, X, K=K)
print(f"{len(item_ids)} items to label:", item_ids[:10])
```

**When to use this**

* You want a **ranked, concrete list** under a fixed budget.
* You’re comfortable with a pragmatic approach (great for ops planning and when strict design‑unbiasedness is not required).
* If you later want formal unbiasedness, add a small randomized “cover” over the remaining items (see next section).


## Design‑unbiased pipeline (HT shares)

If you want **design‑unbiased OLS**, use randomized inclusion probabilities and HT weighting. A simple, practical variant is **Poisson (independent) sampling** with first‑order inclusion probabilities

$$
\pi_j \;=\; \min\{1,\; c\,\sqrt{w_j}\},
$$

choosing the scale $c$ so that $\sum_j \pi_j \approx K$. This yields a random label count with $\mathbb{E}[|S|]\approx K$; if you truly need a fixed size, use a fixed‑size balanced sampler (e.g., cube method) with the same $\pi_j$ as targets.

Here’s a self‑contained example showing (i) computing $\pi$, (ii) drawing the item sample, (iii) computing HT shares, and (iv) running OLS:

```python
import numpy as np
import pandas as pd

# Assume counts, X from the prior snippet
Xn = X.to_numpy(float)
T = counts.sum(axis=1).to_numpy(float)
keep = T > 0
counts2 = counts.loc[keep]
X2 = X.loc[keep]
T2 = T[keep]

# Compute w_j (same as inside fewlab)
V = counts2.to_numpy(float) / T2[:, None]       # (n x m)
XtX = Xn[keep].T @ Xn[keep]
ridge = None
cond = np.linalg.cond(XtX)
if not np.isfinite(cond) or cond > 1e12:
    ridge = 1e-8
if ridge:
    XtX = XtX + ridge * np.eye(XtX.shape[0])
XtX_inv = np.linalg.inv(XtX)
G = Xn[keep].T @ V                               # (p x m)
w = np.einsum("jp,pk,kj->j", G.T, XtX_inv, G)    # length m

# First-order inclusions: pi_j ∝ sqrt(w_j), scaled to expected K
K = 80
sqrtw = np.sqrt(np.maximum(w, 0.0))
def scale_to_budget(sqrtw, K, pi_min=1e-4):
    lo, hi = 1e-12, 1e12
    for _ in range(60):
        c = np.sqrt(lo*hi)
        pi = np.clip(c * sqrtw, pi_min, 1.0)
        s = pi.sum()
        if s > K: hi = c
        else:     lo = c
    return np.clip(c * sqrtw, pi_min, 1.0)

pi = scale_to_budget(sqrtw, K)
rng = np.random.default_rng(123)
I = rng.binomial(1, np.clip(pi, 0.0, 1.0)).astype(bool)   # Poisson sampling
S = np.where(I)[0]                                        # item indices sampled

# After labeling, suppose you obtain a_j for sampled items (demo here)
# In production, replace this mock with real labels
a = np.zeros(counts2.shape[1], dtype=int)
a[S] = 1 * (w[S] > np.median(w))  # fake: just for demonstration

# Horvitz–Thompson share estimates:
# yhat_i = sum_j (I_j / pi_j) (c_ij / T_i) a_j
wj = np.zeros_like(pi, dtype=float)
sel = I & (pi > 0)
wj[sel] = a[sel] / pi[sel]
s = counts2.to_numpy(float) @ wj
yhat = s / T2

# OLS on HT shares (design-unbiased for beta*)
beta_hat = np.linalg.solve(X2.to_numpy(float).T @ X2.to_numpy(float),
                           X2.to_numpy(float).T @ yhat)
print("beta_hat shape:", beta_hat.shape)
```

**Notes**

* Poisson sampling gives you unbiased HT shares and a **random** label count near $K$.
* For **fixed size** and smaller realized $X^\top u$, use a **balanced** fixed‑size sampler (cube method / conditional Poisson). Feed it the same $g_j=X^\top v_j$ as auxiliaries and $\pi_j$ as first‑order targets.

## API

```python
from fewlab import items_to_label

items_to_label(
    counts: pd.DataFrame,   # (n x m) nonnegative, index aligns with X.index
    X: pd.DataFrame,        # (n x p), include an intercept if desired
    K: int,                 # number of items to return
    *,
    item_axis: int = 1,     # reserved for future
    ensure_full_rank: bool = True,
    ridge: float | None = None
) -> list[str]
```

* Returns a deterministic list of **K item IDs** (column names of `counts`) ranked by the A‑optimal influence score.
* If $X^\top X$ is ill‑conditioned, a tiny ridge is added by default (`ensure_full_rank=True`).
* Rows with $T_i=0$ are dropped internally for stability.

---

## What this package does and does not do

* ✅ **Ranks items for labeling** under a fixed budget based on a principled A‑optimal criterion tied to your $X$.
* ✅ Plays nicely with balanced sampling: you can use the same $g_j=X^\top(c_{\cdot j}/T)$ as auxiliaries.
* ⚠️ **Does not** implement a full random sampling engine (cube method / conditional Poisson) — those are small, separate libraries.
* ⚠️ A purely deterministic top‑$K$ list is **not** HT‑unbiased for the full population. Add a small randomization layer (as shown) if strict design‑unbiasedness is required.

## Practical tips

* **Intercept:** Include an explicit intercept column in `X` if you want overall level effects.
* **Scaling of covariates:** Standardizing non‑intercept columns can help stability when computing $(X^\top X)^{-1}$.
* **Very sparse rows:** If many rows have small $T_i$, expect more volatile shares; consider per‑row variance caps (a future extension).
* **Label errors:** If labels are noisy (misclassification), HT is unbiased for the noisy trait. For de‑noised coefficients you’ll need a measurement‑error correction.


## License

MIT

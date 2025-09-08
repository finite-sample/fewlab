# fewlab

Pick the **few**est items to **lab**el for unbiased OLS on shares — fast, deterministic, and simple.

## What it does

Given:
- a counts matrix `C` (units × items), and
- a covariate matrix `X` (units × predictors),

`fewlab.items_to_label(C, X, K)` returns **which K items** to label to make OLS on the trait share unbiased (Horvitz–Thompson over items) and **variance-efficient** in the A-optimal sense.

It computes, for each item `j`:
- `v_j = c_{·j} / T` (row-normalized exposure),
- `g_j = X^T v_j`,
- `w_j = g_j^T (X^T X)^{-1} g_j`,
then picks the **top-K** items by `w_j`. This is the deterministic fixed-budget version of the square-root A-optimal allocation.

## Install

```bash
pip install -e .

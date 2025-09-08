## fewlab: fewest items to label for unbiased OLS on shares

[![PyPI version](https://img.shields.io/pypi/v/fewlab.svg)](https://pypi.org/project/fewlab/)
[![Downloads](https://pepy.tech/badge/rmcp)](https://pepy.tech/project/fewlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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

## fewlab: fewest items to label for most efficient unbiased OLS on shares

[![Python application](https://github.com/finite-sample/fewlab/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/fewlab/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/fewlab/)
[![PyPI version](https://img.shields.io/pypi/v/fewlab.svg)](https://pypi.org/project/fewlab/)
[![Downloads](https://pepy.tech/badge/fewlab)](https://pepy.tech/project/fewlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Problem**: You have usage data (users × items) and want to understand how user traits relate to item preferences. But you can't afford to label every item. This tool tells you which items to label first to get the most accurate analysis.

## When You Need This

You have:
- A usage matrix: rows are users, columns are items (websites, products, apps)
- User features you want to analyze (demographics, behavior patterns)
- Limited budget to label items (safe/unsafe, brand affiliation, category)

You want to run a regression to understand relationships between user features and item traits, but labeling is expensive. Random sampling wastes budget on items that don't affect your analysis.

## How It Works

The tool identifies items that most influence your regression coefficients. It prioritizes items that:
1. Are used by many people
2. Show different usage patterns across your user segments
3. Would most change your conclusions if mislabeled

Think of it as "statistical leverage"—some items matter more for understanding user-trait relationships.

## Quick Start

```python
from fewlab import items_to_label
import pandas as pd

# Your data: user features and item usage
user_features = pd.DataFrame(...)  # User characteristics
item_usage = pd.DataFrame(...)     # Usage counts per user-item

# Get top 100 items to label
priority_items = items_to_label(
    counts=item_usage,
    X=user_features,
    K=100
)

# Send priority_items to your labeling team
print(f"Label these items first: {priority_items}")
```

## Advanced Usage

```python
from fewlab import pi_aopt_for_budget, balanced_fixed_size, row_se_min_labels

# Get inclusion probabilities for expected budget
probabilities = pi_aopt_for_budget(
    counts=item_usage,
    X=user_features,
    K=100
)

# Balanced sampling with probability constraints
selected_items = balanced_fixed_size(
    pi=probabilities,
    g=influence_projections,
    K=100,
    seed=42
)

# Minimize row-wise standard errors
optimal_items = row_se_min_labels(
    counts=item_usage,
    eps2=error_budget_per_row
)
```

## What You Get

**Multiple approaches** for optimal item selection:

- **`items_to_label()`**: Deterministic top-K items for maximum precision
- **`pi_aopt_for_budget()`**: Inclusion probabilities for randomized sampling
- **`balanced_fixed_size()`**: Balanced sampling with probability constraints
- **`row_se_min_labels()`**: Minimize row-wise standard errors
- **`topk()`**: Efficient O(n) top-k selection algorithm

All methods consider:
- Item usage patterns across user segments
- Statistical leverage for regression coefficients
- Optimal allocation of labeling budget

## Practical Considerations

**Choosing K**: Start with 10-20% of items. You can always label more if needed.

**Validation**: Compare regression stability with different K values. When coefficients stop changing significantly, you have enough labels.

**Limitations**:
- Works best when usage patterns correlate with user features
- Assumes item labels are binary (has trait / doesn't have trait)
- Most effective for sparse usage matrices

## Advanced: Ensuring Unbiased Estimates

The basic approach gives you optimal items to label but technically requires some randomization for completely unbiased statistical estimates. If you need formal statistical guarantees, add a small random sample on top of the priority list. See the [statistical details](link) for more.

## Installation

```bash
pip install fewlab
```

**Requirements**: Python 3.11+, numpy ≥1.23, pandas ≥1.5

**Development**:
```bash
pip install -e ".[dev]"  # Includes testing, linting, pre-commit hooks
pip install -e ".[docs]" # Includes documentation building
```

## What's New in v0.3.0

- 🐍 **Modern Python**: Requires Python 3.11+ (breaking change)
- 📋 **Smart Config**: Docs automatically sync with pyproject.toml metadata
- 🚀 **Performance**: O(n) top-k selection algorithm (vs O(n log n))
- 🔧 **Code Quality**: Type hints, constants, eliminated dead code
- 📚 **Modern Docs**: Furo theme with dark/light mode support
- 🧪 **Developer Experience**: Pre-commit hooks, comprehensive testing
- 📦 **Expanded API**: 5 functions for different sampling strategies

## Development

For contributors, see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions including required pre-commit hooks.

## License

MIT

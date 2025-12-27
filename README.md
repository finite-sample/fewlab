## fewlab: fewest items to label for most efficient unbiased OLS on shares

[![Python application](https://github.com/finite-sample/fewlab/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/fewlab/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/fewlab/)
[![PyPI version](https://img.shields.io/pypi/v/fewlab.svg)](https://pypi.org/project/fewlab/)
[![Downloads](https://pepy.tech/badge/fewlab)](https://pepy.tech/project/fewlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/fewlab.svg)](https://www.python.org/downloads/)

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
from fewlab import Design
import pandas as pd

# Your data: user features and item usage
user_features = pd.DataFrame(...)  # User characteristics
item_usage = pd.DataFrame(...)     # Usage counts per user-item

# Create design (caches expensive computations)
design = Design(item_usage, user_features)

# Get top 100 items to label
priority_items = design.select(budget=100)

# Send priority_items to your labeling team
print(f"Label these items first: {priority_items}")
```

## Advanced Usage

```python
from fewlab import Design
import pandas as pd

# Create design with automatic ridge detection
design = Design(item_usage, user_features, ridge="auto")

# Multiple selection strategies
deterministic_items = design.select(budget=100, method="deterministic")
greedy_items = design.select(budget=100, method="greedy")

# Probabilistic sampling methods
balanced_sample = design.sample(budget=100, method="balanced", seed=42)
hybrid_sample = design.sample(budget=100, method="core_plus_tail", tail_frac=0.3)
adaptive_sample = design.sample(budget=100, method="adaptive")

# Get inclusion probabilities
pi_aopt = design.inclusion_probabilities(budget=100, method="aopt")
pi_rowse = design.inclusion_probabilities(budget=100, method="row_se", eps2=0.01)

# Complete workflow: select, weight, estimate
selected = design.select(budget=50)
labels = pd.Series([1, 0, 1, ...], index=selected)  # Your labels
weights = design.calibrate_weights(selected)
estimates = design.estimate(selected, labels)

# Access diagnostics
print(f"Condition number: {design.diagnostics['condition_number']:.2e}")
print(f"Influence weights: {design.influence_weights.head()}")
```

## What You Get

**Primary Interface**:

- **`Design`**: Object-oriented API that caches expensive computations and provides unified access to all methods

**Selection Methods**:

- **`.select(method="deterministic")`**: Batch A-optimal top-budget items (fastest)
- **`.select(method="greedy")`**: Sequential greedy A-optimal selection
- **`.sample(method="balanced")`**: Balanced probabilistic sampling
- **`.sample(method="core_plus_tail")`**: Hybrid deterministic + probabilistic
- **`.sample(method="adaptive")`**: Data-driven hybrid with automatic parameters

**Probability Methods**:

- **`.inclusion_probabilities(method="aopt")`**: A-optimal square-root rule
- **`.inclusion_probabilities(method="row_se")`**: Row-wise standard error minimization

**Complete Workflow**:

- **`.calibrate_weights()`**: GREG-style weight calibration
- **`.estimate()`**: Calibrated Horvitz-Thompson estimation
- **`.diagnostics`**: Comprehensive design diagnostics

All methods leverage cached influence computations for efficiency and provide consistent, structured results.

## Practical Considerations

**Choosing budget**: Start with 10-20% of items. You can always label more if needed.

**Validation**: Compare regression stability with different budget values. When coefficients stop changing significantly, you have enough labels.

**Performance**: The `Design` class caches expensive influence computations, making multiple method calls efficient.

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

**Requirements**: Python 3.12-3.14, numpy ≥1.23, pandas ≥1.5

**Development**:
```bash
pip install -e ".[dev]"  # Includes testing, linting, pre-commit hooks
pip install -e ".[docs]" # Includes documentation building
```

## What's New in v1.0.0

- 🎯 **Object-Oriented API**: New `Design` class caches expensive computations and provides unified interface
- 🚀 **Performance**: Eliminate redundant influence computations across multiple method calls
- 📊 **Structured Results**: Typed result classes replace loose tuples for better API consistency
- 🔧 **Standardized Parameters**: All functions use `budget` parameter (was `K`), no backward compatibility
- 📈 **Comprehensive Diagnostics**: Automatic condition number monitoring and ridge selection
- 🧪 **Enhanced Testing**: Full test coverage for new Design class and edge cases
- 🐍 **Modern Python**: Requires Python 3.12-3.14, uses latest type annotations
- 🛡️ **Robust Validation**: Enhanced input validation with helpful error messages

## Development

To contribute to this project, install dependencies and set up pre-commit hooks:

```bash
uv sync --all-groups
uv run pre-commit install
```

## License

MIT

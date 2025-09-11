## fewlab: fewest items to label for most efficient unbiased OLS on shares

[![PyPI version](https://img.shields.io/pypi/v/fewlab.svg)](https://pypi.org/project/fewlab/)
[![Downloads](https://pepy.tech/badge/fewlab)](https://pepy.tech/project/fewlab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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

## Basic Usage

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

## What You Get

A ranked list of K items that will give you the most precise regression estimates. The tool considers:
- How much each item is used
- Which user segments use which items  
- The statistical relationship between items and your analysis goals

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

Requires: numpy, pandas

## License

MIT

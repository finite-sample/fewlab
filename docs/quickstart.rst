Quick Start Guide
=================

This guide will get you started with fewlab in just a few minutes.

Installation
------------

Install fewlab using pip:

.. code-block:: bash

   pip install fewlab

Basic Usage
-----------

The main function you'll use is :func:`fewlab.items_to_label`, which tells you which items to prioritize for labeling.

Simple Example
^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   import numpy as np
   from fewlab import items_to_label

   # Create sample data
   # users × items usage matrix
   np.random.seed(42)
   n_users, n_items = 100, 50

   # User features (e.g., demographics)
   user_features = pd.DataFrame({
       'age': np.random.normal(35, 10, n_users),
       'income': np.random.normal(50000, 15000, n_users),
       'urban': np.random.binomial(1, 0.6, n_users)
   })

   # Item usage counts (sparse matrix)
   item_usage = pd.DataFrame(
       np.random.poisson(2, (n_users, n_items)),
       columns=[f'item_{i}' for i in range(n_items)]
   )

   # Get top 15 items to label
   priority_items = items_to_label(
       counts=item_usage,
       X=user_features,
       K=15
   )

   print(f"Priority items to label: {priority_items}")

What You Get
^^^^^^^^^^^^

The function returns a list of item names (column names from your usage matrix) ranked by importance. These are the items that will give you the most statistical power for understanding the relationship between user features and item characteristics.

Understanding the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithm prioritizes items based on:

1. **High usage**: Items used by many users provide more information
2. **Differential usage**: Items with different usage patterns across user segments
3. **Statistical leverage**: Items that most influence your regression estimates

Choosing K (Number of Items)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start with 10-20% of your total items:

.. code-block:: python

   total_items = item_usage.shape[1]
   K = max(10, total_items // 5)  # At least 10, or 20% of items

   priority_items = items_to_label(counts=item_usage, X=user_features, K=K)

You can always label more items later if your analysis needs higher precision.

Handling Real Data
------------------

Data Preparation
^^^^^^^^^^^^^^^^

1. **User alignment**: Ensure your user features and usage data have the same index:

.. code-block:: python

   # Make sure indices align
   assert user_features.index.equals(item_usage.index)

2. **Zero totals**: Users with zero usage are automatically dropped:

.. code-block:: python

   # Check for users with no usage
   total_usage = item_usage.sum(axis=1)
   print(f"Users with zero usage: {(total_usage == 0).sum()}")

3. **Sparse data**: The algorithm works well with sparse usage matrices.

Advanced Options
^^^^^^^^^^^^^^^^

Control numerical stability:

.. code-block:: python

   # Add ridge regularization if needed
   priority_items = items_to_label(
       counts=item_usage,
       X=user_features,
       K=15,
       ridge=1e-6  # Small regularization
   )

Next Steps
----------

- See :doc:`examples` for more detailed use cases
- Check :doc:`api` for complete function documentation
- Review :doc:`mathematical_background` for the statistical theory

Common Issues
-------------

**ValueError: counts.index must align with X.index**
   Make sure your user features and item usage DataFrames have the same row indices.

**Singular matrix errors**
   Try adding ridge regularization or check for perfectly correlated features.

**Empty results**
   Ensure you have users with non-zero usage totals.

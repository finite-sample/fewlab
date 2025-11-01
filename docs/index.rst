fewlab: fewest items to label for most efficient unbiased OLS
==============================================================

.. image:: https://github.com/finite-sample/fewlab/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/finite-sample/fewlab/actions/workflows/ci.yml
   :alt: Python application

.. image:: https://img.shields.io/pypi/v/fewlab.svg
   :target: https://pypi.org/project/fewlab/
   :alt: PyPI version

.. image:: https://pepy.tech/badge/fewlab
   :target: https://pepy.tech/project/fewlab
   :alt: Downloads

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

**Problem**: You have usage data (users × items) and want to understand how user traits relate to item preferences. But you can't afford to label every item. This tool tells you which items to label first to get the most accurate analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   examples
   api
   mathematical_background

Overview
--------

When You Need This
^^^^^^^^^^^^^^^^^^

You have:

- A usage matrix: rows are users, columns are items (websites, products, apps)
- User features you want to analyze (demographics, behavior patterns)
- Limited budget to label items (safe/unsafe, brand affiliation, category)

You want to run a regression to understand relationships between user features and item traits, but labeling is expensive. Random sampling wastes budget on items that don't affect your analysis.

How It Works
^^^^^^^^^^^^

The tool identifies items that most influence your regression coefficients. It prioritizes items that:

1. Are used by many people
2. Show different usage patterns across your user segments
3. Would most change your conclusions if mislabeled

Think of it as "statistical leverage"—some items matter more for understanding user-trait relationships.

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

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

Installation
------------

.. code-block:: bash

   pip install fewlab

Requires: numpy, pandas

License
-------

MIT

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

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

.. image:: https://img.shields.io/pypi/pyversions/fewlab.svg
   :target: https://www.python.org/downloads/
   :alt: Python versions

**Buy the labels that buy the smallest standard error.**

You have a usage matrix: rows are people, columns are things they used. Websites,
merchants, products, surnames. You want to regress something about the column on
features of the row, but finding out what the column *is* costs money, and you
cannot afford all of them.

On a real browsing panel, labeling 250 of 34,078 domains recovers the same
regression coefficients as labeling all of them, with a ninth of the error that
spending the same 250 labels at random would give. The estimate stays exactly
unbiased and reports a standard error you can put in a table.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   examples
   api
   mathematical_background
   EVALUATION_RESULTS

Why it works
------------

Write :math:`V` for row shares and :math:`A` for entity labels. The
complete-label regression coefficient is

.. math::
   \beta^* = (X^\top X)^{-1} X^\top V A = \sum_j h_j a_j^\top

exactly, with no error term. Entity :math:`j`'s entire contribution is
:math:`h_j a_j^\top`, and :math:`h_j` is computable from usage counts and
features before a single label is bought. Everything else follows: the optimal
inclusion probabilities, the certainty core, and the standard errors.

Two goals, and neither is traded for the other. Unbiasedness is a constraint:
every contributing entity gets positive probability and reciprocal weight.
Minimum standard error is what you optimize subject to it.

Installation
------------

.. code-block:: bash

   pip install fewlab

Requires Python 3.12 or later, numpy and pandas.

Evidence
--------

The numbers above come from :doc:`EVALUATION_RESULTS`, which scores every design
against a population whose labels are all known, so there is a fixed truth to be
measured against. Reproduce it with ``python benchmarks/make_exhibits.py``.

License
-------

MIT

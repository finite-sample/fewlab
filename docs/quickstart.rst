Quickstart
==========

Every code block on this page is a doctest and runs in CI, so an example that
stops working fails the build rather than sitting here misleading you.

Build a study
-------------

A study holds the usage counts and the unit features, and computes the influence
decomposition once.

.. doctest::

   >>> import numpy as np
   >>> import pandas as pd
   >>> from fewlab import Study
   >>> rng = np.random.default_rng(0)
   >>> counts = pd.DataFrame(
   ...     rng.poisson(3, (300, 120)),
   ...     columns=[f"site{j}" for j in range(120)],
   ... )
   >>> X = pd.DataFrame(
   ...     np.c_[np.ones(300), rng.normal(size=(300, 2))],
   ...     columns=["const", "age", "income"],
   ... )
   >>> study = Study(counts, X)
   >>> study.influence.n_items
   120

Real panels arrive in long format, one row per unit and entity. Use
``Study.from_long`` and the dense matrix is never materialized.

.. doctest::

   >>> long = pd.DataFrame(
   ...     {"user": ["a", "a", "b"], "site": ["x", "y", "x"], "visits": [3.0, 1.0, 5.0]}
   ... )
   >>> features = pd.DataFrame(
   ...     {"const": [1.0, 1.0], "z": [0.0, 1.0]}, index=["a", "b"]
   ... )
   >>> small = Study.from_long(
   ...     long, features, unit="user", entity="site", count="visits"
   ... )
   >>> small.influence.n_items
   2

Design, draw, estimate
----------------------

A design is a plan: a probability for every entity. Drawing it produces a
sample, and only a sample can be estimated from.

.. doctest::

   >>> design = study.design(budget=30)
   >>> round(design.budget, 6)
   30.0
   >>> bool((design.probabilities > 0).all())
   True
   >>> sample = study.draw(design, random_state=1)
   >>> len(sample)
   30

The entities at probability one are the certainty core. You do not choose it;
the budget constraint produces it.

.. doctest::

   >>> len(design.certain) > 0
   True

Now label what was drawn and estimate.

.. doctest::

   >>> labels = pd.Series(
   ...     rng.integers(0, 2, len(sample)).astype(float), index=sample.items
   ... )
   >>> est = study.estimate(sample, labels)
   >>> est.coefficients.shape
   (3, 1)
   >>> bool((est.design_se.to_numpy() > 0).all())
   True

Two standard errors
-------------------

One component comes from having labeled only some entities, the other from
having only so many units. The budget moves the first and not the second, and
their ratio tells you when to stop paying.

.. doctest::

   >>> est.regression_se is not None
   True
   >>> 0.0 <= est.diagnostics["variance_ratio"] <= 1.0
   True

Rankings are not samples
------------------------

A ranking is useful for triage and carries no inclusion probabilities, so it
cannot be weighted. Passing one to the estimator raises rather than returning a
number that looks like an estimate.

.. doctest::

   >>> ranking = study.rank(30)
   >>> round(ranking.coverage, 2) > 0
   True
   >>> study.estimate(ranking, labels)
   Traceback (most recent call last):
       ...
   TypeError: estimate() needs a LabelSample, not a Ranking. ...

Choosing the criterion
----------------------

There is no target-free notion of which entities are worth labeling.

.. doctest::

   >>> coef = study.scores(loss="coefficient")
   >>> pred = study.scores(loss="prediction")
   >>> bool((coef > 0).all() and (pred > 0).all())
   True

Check whether the choice can matter for your data. A spectrum that falls to the
noise floor immediately means every criterion ranks entities identically.

.. doctest::

   >>> spectrum = study.diagnostics["direction_spectrum"]
   >>> round(float(spectrum.iloc[0]), 6)
   1.0

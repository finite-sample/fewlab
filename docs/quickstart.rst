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
   >>> sample.method.endswith("+poisson")
   True

The default draw uses independent Bernoulli trials. The budget is the expected
number of labels; the realized count can differ. Set ``fixed_size=True`` when
the count must be exact. That cube draw preserves the inclusion probabilities
and balances on the influence totals, but its variance estimator is approximate.
At a very small budget, an independent draw can contain no labels: the point
estimate still follows the design, but its standard error is unavailable.
A cube draw with no residual degrees of freedom likewise has no estimable
standard error; fewlab reports ``NaN`` rather than zero.

The entities at probability one are the certainty core. You do not choose it;
the budget constraint produces it. This nearly uniform example has no core.

.. doctest::

   >>> len(design.certain)
   0

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

For a labeling handoff, call ``sample.save("label_sample.json")`` and send
``sample.items`` to the labelers. The manifest records the complete probability
frame, selected IDs, draw method, and study fingerprint. Later, load it with
``LabelSample.load("label_sample.json")``, rebuild the same ``Study`` from the
saved counts and features, and pass labels indexed by entity ID to
``study.estimate``. A different study frame raises an error. The selected IDs
and probabilities in the manifest are the record of the draw; a seed alone is
not a replacement for them.

Choose a risk ceiling instead of a budget
-----------------------------------------

``Study.design_for_risk`` finds the minimum expected labeling cost for a chosen
Poisson risk ceiling. Unit costs minimize the expected number of labels. The
result can have a fractional expected size, so draw independently with
the default independent draw.

For one coefficient, pass its contrast vector and square the desired design SE.
The default joint coefficient loss instead bounds the sum of variances over all
coefficients and label columns. These ceilings concern label-sampling variance;
they exclude regression sampling uncertainty and do not bound realized errors.

.. doctest::

   >>> contrast = np.array([0.0, 1.0, 0.0])
   >>> limited = study.design_for_risk(0.01**2, contrast=contrast)
   >>> limited.risk <= 0.01**2 * (1 + 1e-12)
   True
   >>> cost = limited.diagnostics["expected_cost"]
   >>> abs(cost - limited.budget) <= 1e-12 * limited.budget  # unit costs: cost = labels
   True
   >>> draw = study.draw(limited, fixed_size=False, random_state=2)
   >>> observed = pd.Series(rng.integers(0, 2, len(draw)), index=draw.items)
   >>> result = study.estimate(draw, observed)
   >>> result.coefficients.shape
   (3, 1)

Default unit label norms give exact aggregate risk for a complete one-hot
taxonomy and a conservative ceiling for an unknown binary indicator. Estimated
label norms give a planning criterion. A contrast design protects only the
chosen contrast; other coefficients may lose unbiasedness if their contributing
entities have zero scores for that contrast. The ceiling does not extend to
balanced draws or to a proxy or shrinkage estimator with different contributions.

Pass ``cost=`` to minimize expected expenditure with unequal label costs. An
indexed cost Series is matched by entity; ``design.budget`` remains the expected
number of labels, while ``design.diagnostics["expected_cost"]`` reports cost.

Design and model standard errors
--------------------------------

``design_se`` measures the uncertainty due to labeling only some entities,
conditional on the observed rows and entity frame. This is the default basis of
``est.summary()``. The sample records its draw method, so estimation uses the
exact Poisson variance formula after an independent draw and the balanced
residual approximation after a cube draw. Ordinary survey regression weights
for the rows do not represent this entity-label draw.

Pass ``row_model_se=True`` to request ``regression_se``, which additionally
assumes the observed rows come from a
homoskedastic row-level model. ``total_se`` combines the model and design
components; these are secondary to inference for the observed frame. The
budget moves the design component, and ``variance_ratio`` compares it with
the combined variance under that model.

.. doctest::

   >>> model_est = study.estimate(sample, labels, row_model_se=True)
   >>> model_est.regression_se is not None
   True
   >>> 0.0 <= model_est.diagnostics["variance_ratio"] <= 1.0
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

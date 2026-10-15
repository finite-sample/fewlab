Examples
========

Every block runs as a doctest.

Try it in your browser
----------------------

`Which websites should you pay to label? <demo/index.html>`_ runs the whole
decision on the real browsing panel, with a slider for the labeling budget. It
runs Python in your browser, so there is nothing to install; the first load
takes about half a minute. The source is ``docs/examples/label_budget.py`` in
the repository, a `marimo <https://marimo.io>`_ notebook that also runs as a
plain script.

A browsing panel
----------------

The motivating case. Rows are panelists, columns are websites, and a website's
category costs money to determine once and then applies to everyone who visited
it.

.. doctest::

   >>> import numpy as np
   >>> import pandas as pd
   >>> from fewlab import Study
   >>> rng = np.random.default_rng(7)
   >>> n, m = 400, 500
   >>> # Sites differ in popularity and in the kind of user they attract.
   >>> taste = rng.normal(size=(m, 2))
   >>> popularity = np.exp(rng.normal(0, 1.4, size=m))
   >>> Z = rng.normal(size=(n, 2))
   >>> rate = np.exp(0.4 + Z @ taste.T + np.log(popularity) - np.log(popularity).mean())
   >>> visits = pd.DataFrame(
   ...     rng.poisson(rate), columns=[f"site{j}" for j in range(m)]
   ... )
   >>> users = pd.DataFrame(
   ...     np.c_[np.ones(n), Z], columns=["const", "age", "income"]
   ... )
   >>> panel = Study(visits, users)
   >>> panel.influence.n_items
   500

Labeling 60 of the 500 sites, chosen by influence:

.. doctest::

   >>> design = panel.design(budget=60)
   >>> sample = panel.draw(design, random_state=3)
   >>> len(sample), len(design.certain) > 0
   (60, True)

Because the population is heavy-tailed, the certainty core absorbs a large share
of the budget and the sampled tail covers the rest.

.. doctest::

   >>> core = len(design.certain)
   >>> 0 < core < 60
   True

Costs that differ by entity
---------------------------

The rule is value per dollar, so an entity that is expensive to label is bought
less often at the same influence.

.. doctest::

   >>> cost = pd.Series(1.0, index=panel.influence.items)
   >>> biggest = panel.scores().idxmax()
   >>> cost[biggest] = 50.0
   >>> cheap = panel.design(60).probabilities[biggest]
   >>> dear = panel.design(60, cost=cost).probabilities[biggest]
   >>> bool(dear < cheap)
   True

A mutually exclusive taxonomy
-----------------------------

When each site belongs to exactly one category, the squared label norm is one,
so the design needs no pilot labeling at all. Labels are supplied as one column
per class.

.. doctest::

   >>> classes = ["news", "shopping", "social"]
   >>> draw = rng.integers(0, 3, len(sample))
   >>> labels = pd.DataFrame(
   ...     np.eye(3)[draw], index=sample.items, columns=classes
   ... )
   >>> est = panel.estimate(sample, labels)
   >>> est.coefficients.shape
   (3, 3)
   >>> list(est.coefficients.columns)
   ['news', 'shopping', 'social']

Soft labels need no special handling: a coder's probability distribution over
classes is just a row that does not happen to be one-hot.

.. doctest::

   >>> soft = pd.DataFrame(
   ...     rng.dirichlet(np.ones(3), size=len(sample)),
   ...     index=sample.items,
   ...     columns=classes,
   ... )
   >>> panel.estimate(sample, soft).coefficients.shape
   (3, 3)

A transparent design for a labeling vendor
------------------------------------------

A certainty core with a uniform random tail is easier to explain and audit than
unequal tail probabilities, and its core size is chosen by arithmetic rather
than judgment.

.. doctest::

   >>> simple = panel.design(60, method="core_plus_tail")
   >>> simple.diagnostics["core_size"] > 0
   True
   >>> round(simple.diagnostics["tail_probability"], 4) > 0
   True

It cannot beat the unrestricted design, which is free to vary the tail.

.. doctest::

   >>> bool(panel.design(60).risk <= simple.risk)
   True

Fewest labels for a fixed precision
------------------------------------

The reverse question: rather than fixing a budget and minimizing error, fix the
per-unit standard error and minimize the number of labels.

.. doctest::

   >>> from fewlab.rowse import row_se_min_labels
   >>> pi = row_se_min_labels(visits, eps2=0.05**2)
   >>> bool(0 < pi.sum() < visits.shape[1])
   True

Comparing against a random baseline
-----------------------------------

Uniform sampling is the honest benchmark, and the risk ordering is guaranteed
rather than empirical.

.. doctest::

   >>> pps = panel.design(60)
   >>> srs = panel.design(60, method="srs")
   >>> bool(pps.risk < srs.risk)
   True

Mathematical background
=======================

This page states the identity the package rests on and the design results that
follow from it. The manuscript in ``ms/`` gives the same material with proofs,
related work, and the empirical evaluation.

The estimand
------------

Rows are units and columns are entities that carry an expensive label. With
counts :math:`C`, row totals :math:`T_i = \sum_j c_{ij}`, and row shares
:math:`V_{ij} = c_{ij}/T_i`, entity :math:`j` carries a label :math:`a_j` and the
unit-level outcome is the share of activity on labeled entities:

.. math::
   y_i = \sum_{j=1}^m V_{ij}\, a_j.

With unit features :math:`X`, the estimand is the coefficient you would obtain by
labeling every entity in the frame,
:math:`\beta^* = (X^\top X)^{-1} X^\top y`.

The decomposition
-----------------

Writing :math:`H = (X^\top X)^{-1} X^\top V` with columns :math:`h_j`,

.. math::
   \beta^* = (X^\top X)^{-1} X^\top V a = \sum_{j=1}^m h_j a_j.

This is an identity. There is no error term, no distributional assumption, and
no covariance to approximate. Entity :math:`j`'s entire contribution is
:math:`h_j a_j`, and :math:`h_j` depends only on :math:`(C, X)`, so it is known
before any label is purchased.

The algebra carries a vector label :math:`a_j \in \mathbb{R}^q` unchanged, giving
:math:`\beta^* = \sum_j h_j a_j^\top`. A mutually exclusive taxonomy and a
coder's probability distribution over classes therefore need no special handling.

Unbiasedness
------------

Sample entities with inclusion probabilities :math:`\pi_j` and indicators
:math:`I_j` that do not depend on the unknown labels. Then

.. math::
   \widehat\beta = \sum_{j=1}^m \frac{I_j}{\pi_j} h_j a_j

is design-unbiased, since :math:`\mathbb{E}[I_j/\pi_j] = 1` and everything else is
fixed. Unbiasedness is a constraint on the design rather than a property to hope
for: it holds exactly when every entity with :math:`h_j \neq 0` gets positive
probability and enters with weight :math:`1/\pi_j`.

An entity nobody touched has :math:`h_j = 0` and contributes nothing, so it may
be left unlabeled at no cost in bias. That is the only case in which
:math:`\pi_j = 0` is correct.

Selecting the top :math:`K` entities by any criterion assigns probability one
above the cut and zero below it. Its bias is :math:`-\sum_{j \notin S} h_j a_j`,
which depends on labels nobody bought and so cannot be estimated or bounded from
the data in hand. This is why :class:`fewlab.Ranking` cannot be passed to an
estimator.

Which entities are worth labeling
---------------------------------

For a positive semidefinite :math:`W`, the loss
:math:`\operatorname{tr}[W(\widehat\beta - \beta^*)(\widehat\beta - \beta^*)^\top]`
has exact design risk under independent Bernoulli sampling

.. math::
   \mathbb{E}_p L = \sum_{j=1}^m \frac{1-\pi_j}{\pi_j} b_j,
   \qquad b_j = (h_j^\top W h_j)\, a_j^2.

The criterion enters only through :math:`b_j`, and it has to be declared.

.. list-table::
   :header-rows: 1

   * - ``loss=``
     - :math:`W`
     - :math:`b_j` for a one-hot taxonomy
   * - ``"coefficient"``
     - :math:`I`
     - :math:`h_j^\top h_j = g_j^\top (X^\top X)^{-2} g_j`
   * - ``"prediction"``
     - :math:`X^\top X`
     - :math:`g_j^\top (X^\top X)^{-1} g_j`
   * - ``contrast=L``
     - :math:`LL^\top`
     - :math:`\lVert L^\top h_j \rVert^2`
   * - ``"regressor"``
     - see below
     - :math:`\lVert v_j \rVert^2 - h_j^\top X^\top X h_j`

On a well-conditioned :math:`X^\top X` the first three rank entities almost
identically. They diverge when :math:`X^\top X` is far from a scaled identity and
the :math:`h_j` span several directions, where they can disagree on roughly a
fifth of a budget. There is no target-free answer.

When the choice cannot matter
-----------------------------

If every :math:`h_j` points in the same direction and entities differ only in
magnitude, then :math:`b_j \propto \lVert h_j \rVert^2` under every :math:`W`,
every criterion induces the same ordering, and the estimand stops mattering.
Expected counts of the form :math:`\text{activity}_i \times \text{popularity}_j`
produce exactly this degeneracy.

:func:`fewlab.direction_spectrum` reports it. Read it against a noise floor of
roughly :math:`1/\sqrt{\bar c}` for average count :math:`\bar c`, because integer
counts inject spurious directions at that scale; an absolute cutoff would call
the same population more degenerate simply for being observed more heavily.

The optimal design
------------------

Minimizing the risk subject to :math:`\sum_j c_j \pi_j = K` and
:math:`\pi_j \le 1` gives

.. math::
   \pi_j^\star = \min\left\{1,\ \lambda \sqrt{b_j / c_j}\right\}

for the :math:`\lambda` that exhausts the budget. The cap is not a separate rule
for large entities: the set on which it binds is the certainty core. The square
root rule also settles the unbiasedness constraint by itself, since it gives
strictly positive probability to every entity with :math:`b_j > 0`. A probability
floor is therefore unnecessary, and any floor moves the design away from minimum
risk.

Minimum cost for a risk ceiling
-------------------------------

:func:`fewlab.min_cost_for_risk` solves the reverse problem:

.. math::
   \min_{\pi} \sum_j c_j\pi_j
   \quad\text{subject to}\quad
   \sum_j b_j(1/\pi_j - 1) \le R,\qquad 0 < \pi_j \le 1\ (b_j>0).

Convexity and the stationarity conditions give the same capped square-root rule.
For a proposed uncapped tail :math:`T`, the binding risk constraint determines

.. math::
   \lambda = \frac{\sum_{j\in T}\sqrt{b_jc_j}}
                       {R + \sum_{j\in T}b_j}.

Sort by :math:`\sqrt{b_j/c_j}`, starting with an empty certainty core. Promote
an entity only if its proposed probability exceeds one, and recompute the tail
expression. This gives the global optimum in :math:`O(m\log m)` time. With
:math:`R=0`, label every positive-score entity with certainty. Zero-score
entities never need labels for the declared loss.

Unit label norms give exact aggregate risk for a complete one-hot taxonomy.
For an unknown binary indicator they bound its squared norm and therefore its
risk. Estimated norms give a planning criterion rather than a guaranteed bound.
A single contrast uses its variance (SE squared) as the ceiling; joint
coefficient loss sums variances over coefficient and label columns. A zero
contrast score does not imply zero contribution to every other coefficient, so
the guarantee applies only to the declared estimand.

Use :meth:`fewlab.Study.design_for_risk` followed by
``study.draw(design, fixed_size=False)`` and the plain Horvitz-Thompson estimator.
The ceiling bounds label-sampling risk under independent draws. It excludes
regression uncertainty and realized errors and does not transfer automatically
to balanced draws, proxy residuals, or shrunken weights.

Certainty core with a uniform tail
----------------------------------

Order :math:`b_1 \ge \cdots \ge b_m`, take the first :math:`H` with certainty,
and give the remainder :math:`p_H = (K-H)/(m-H)`. With
:math:`S_H = \sum_{j>H} b_j`,

.. math::
   R(H) = \frac{m-K}{K-H} S_H,

so the boundary follows from evaluating at most :math:`K` cutoffs. Promoting one
more entity pays exactly when :math:`b_{H+1} > S_H/(K-H)`.

This is exact for a Bernoulli tail. A fixed-size simple random tail instead
uses centered contributions and a finite-population correction. Its variance
can exceed this criterion when contributions have opposite signs.

Variance
--------

The design covariance of :math:`\operatorname{vec}(\widehat\beta)` is
:math:`\sum_j \frac{1-\pi_j}{\pi_j} (a_j a_j^\top) \otimes (h_j h_j^\top)`,
estimated without bias from the sample by
:math:`\sum_{j \in S} \frac{1-\pi_j}{\pi_j^2} (a_j a_j^\top) \otimes (h_j h_j^\top)`.

This is exact under Poisson sampling. Fixed-size and balanced draws also depend
on joint inclusion probabilities; the Poisson expression is not a universal
upper bound for their variance. Use variance estimators appropriate to the draw,
and do not interpret a variance ceiling as a confidence-interval coverage
guarantee.

The total standard error has a second component the budget cannot touch:

.. math::
   \operatorname{Var}(\widehat\beta)
   = \sigma^2 (X^\top X)^{-1} + \sum_j \frac{1-\pi_j}{\pi_j} b_j.

Their ratio is a stopping rule. ``Estimate.diagnostics["variance_ratio"]``
reports the share owed to the labeling budget.

When the label is a covariate
-----------------------------

If the constructed share appears on the right-hand side of a regression rather
than the left, the guarantee fails: an unbiased noisy covariate attenuates its
own coefficient, and the attenuation shrinks with the labeling budget rather
than with the size of the panel. Because the design chose :math:`\pi`, the error
variance is known and the attenuation can be corrected. See
:func:`fewlab.estimate_share_effect`.

References
----------

- Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without
  replacement from a finite universe.
- Deville, J.-C., & Tillé, Y. (2004). Efficient balanced sampling: the cube
  method.
- Deville, J.-C., & Särndal, C.-E. (1992). Calibration estimators in survey
  sampling.
- Clark, R. G. (2022). Sample design for analysis using high-influence
  probability sampling.
- Potter, F. (1990). A study of procedures to identify and trim extreme sampling
  weights.

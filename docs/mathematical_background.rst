Mathematical Background
=======================

This section explains the mathematical foundations behind fewlab's item selection algorithms.

Problem Setup
-------------

Suppose we have:

- :math:`n` users (rows) and :math:`m` items (columns)
- Usage count matrix :math:`C \in \mathbb{R}^{n \times m}` where :math:`C_{ij}` is the number of times user :math:`i` used item :math:`j`
- User feature matrix :math:`X \in \mathbb{R}^{n \times p}` containing :math:`p` features for each user
- Budget to label :math:`K` items out of :math:`m` total items

Our goal is to select which :math:`K` items to label to minimize the variance of regression coefficients when estimating:

.. math::
   y_j = X\beta_j + \epsilon_j

where :math:`y_j` is the (unknown) label vector for item :math:`j`.

Statistical Framework
---------------------

A-Optimal Design
^^^^^^^^^^^^^^^^

We use A-optimal experimental design theory. For a given item :math:`j`, if we observe its labels with probability :math:`\pi_j`, the variance of the regression coefficient estimator is proportional to:

.. math::
   \text{Var}(\hat{\beta}_j) \propto \text{tr}((X^T X)^{-1} X^T \Sigma_j X (X^T X)^{-1}) / \pi_j

where :math:`\Sigma_j` is the covariance matrix of the response for item :math:`j`.

The A-optimal weights are:

.. math::
   w_j = \text{tr}((X^T X)^{-1} X^T \Sigma_j X (X^T X)^{-1})

For usage data, we approximate :math:`\Sigma_j` using the empirical usage patterns.

Usage-Based Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

We approximate the response covariance using usage proportions. For user :math:`i` and item :math:`j`:

.. math::
   V_{ij} = \frac{C_{ij}}{T_i}

where :math:`T_i = \sum_{j'} C_{ij'}` is the total usage for user :math:`i`.

The weight becomes:

.. math::
   w_j = g_j^T (X^T X)^{-1} g_j

where :math:`g_j = X^T v_j` and :math:`v_j` is the :math:`j`-th column of :math:`V`.

Algorithm Details
-----------------

Main Algorithm (:func:`fewlab.items_to_label`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Compute usage proportions**: :math:`V_{ij} = C_{ij} / T_i` for users with :math:`T_i > 0`

2. **Calculate regression projections**: :math:`G = X^T V` where :math:`G \in \mathbb{R}^{p \times m}`

3. **Compute A-optimal weights**:

   .. math::
      w_j = g_j^T (X^T X)^{-1} g_j

   where :math:`g_j` is the :math:`j`-th column of :math:`G`.

4. **Select top K items**: Return items with largest :math:`w_j` values.

Budget-Constrained Optimization (:func:`fewlab.pi_aopt_for_budget`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a given labeling budget :math:`B`, solve:

.. math::
   \min_{\pi} \sum_{j=1}^m \frac{w_j}{\pi_j}

subject to:

.. math::
   \sum_{j=1}^m \pi_j \leq B, \quad \pi_j \geq 0

The optimal solution has the form:

.. math::
   \pi_j^* = \frac{\sqrt{w_j}}{\lambda} \text{ if } \sqrt{w_j} > \lambda \pi_{\min}, \text{ else } \pi_{\min}

where :math:`\lambda` is chosen to satisfy the budget constraint.

Balanced Fixed-Size Sampling (:func:`fewlab.balanced_fixed_size`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This heuristic algorithm:

1. **Initial sampling**: Draw :math:`K` items proportional to :math:`\pi_j`
2. **Greedy improvement**: Iteratively swap items to minimize :math:`\|\sum_{j \in S} (I_j/\pi_j - 1) g_j\|_2^2`

where :math:`S` is the selected set and :math:`I_j` is the indicator that item :math:`j` is selected.

Row-Wise Standard Error Constraints (:func:`fewlab.row_se_min_labels`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimizes total labeling subject to per-row standard error constraints:

.. math::
   \min_{\pi} \sum_{j=1}^m \pi_j

subject to:

.. math::
   \sum_{j=1}^m \frac{q_{ij}}{\pi_j} \leq \epsilon_i^2 + \sum_{j=1}^m q_{ij} \quad \forall i

where :math:`q_{ij} = (C_{ij}/T_i)^2`.

This is solved using a dual optimization approach with gradient descent.

Numerical Considerations
------------------------

Ridge Regularization
^^^^^^^^^^^^^^^^^^^^^

When :math:`X^T X` is ill-conditioned, we add ridge regularization:

.. math::
   (X^T X + \lambda I)^{-1}

The default strategy chooses :math:`\lambda = 10^{-8}` when :math:`\text{cond}(X^T X) > 10^{12}`.

Handling Sparse Data
^^^^^^^^^^^^^^^^^^^^^

- Users with zero total usage (:math:`T_i = 0`) are automatically removed
- Numerical stability is maintained through careful matrix conditioning checks
- Small regularization prevents divide-by-zero issues

Computational Complexity
-------------------------

- **Time complexity**: :math:`O(np^2 + mp^2)` for the main algorithm
- **Space complexity**: :math:`O(nm + mp + p^2)`
- **Bottleneck**: Matrix inversion :math:`(X^T X)^{-1}` when :math:`p` is large

The algorithm scales well for typical usage scenarios where :math:`p \ll n` and :math:`p \ll m`.

Theoretical Properties
----------------------

Optimality
^^^^^^^^^^^

Under the assumption that item labels follow the usage-proportion model, the A-optimal weights :math:`w_j` minimize the expected trace of the covariance matrix of regression coefficient estimates.

Robustness
^^^^^^^^^^

The algorithm is robust to:

- **Missing data**: Handled through zero usage entries
- **Outliers**: Proportional scaling reduces sensitivity to extreme usage patterns
- **Collinearity**: Ridge regularization prevents numerical instability

Limitations
^^^^^^^^^^^

- Assumes usage patterns are informative about labeling importance
- Optimal for trace-of-covariance criterion (A-optimality), not other design criteria
- Performance depends on the assumption that high-influence items should be prioritized

References
----------

The mathematical framework builds on classical optimal experimental design theory:

- Fedorov, V.V. (1972). *Theory of Optimal Experiments*
- Pukelsheim, F. (2006). *Optimal Design of Experiments*
- Atkinson, A.C., Donev, A.N., and Tobias, R.D. (2007). *Optimum Experimental Designs*

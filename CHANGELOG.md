# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - Unreleased

The estimand has an exact finite-population identity that the package was
approximating instead of using. Writing `V` for row shares and `A` for entity
labels, the complete-label regression coefficient is

    B = (X'X)^-1 X' V A = sum_j h_j a_j'

with no error term, so an entity's entire contribution to the answer is known
before any label is bought. Rebuilding on that identity fixed four defects and
changed the public API throughout; there are no compatibility shims.

Anyone who took a deterministic selection from `items_to_label` or
`Design.select`, passed it to `calibrate_weights` and `estimate`, and reported
the result as a Horvitz-Thompson estimate should treat that estimate as biased by
an amount that cannot be recovered from the data they have, and re-draw.

### Fixed

- **Deterministic selection could be weighted as though it were a sample.**
  `Design.calibrate_weights` computed inclusion probabilities from the *size* of
  whatever set it was handed, regardless of how that set was chosen, so a top-K
  ranking flowed into a function named `calibrated_ht_estimator`. Deterministic
  selection has probability one above the cut and zero below it, and its bias is
  the contribution of everything omitted, which depends on labels nobody bought.
  `Ranking`, `LabelDesign` and `LabelSample` are now separate types, and only a
  realized sample reaches an estimator.

- **The influence score targeted a different loss than the documentation
  claimed.** The code computed `g_j'(X'X)^-1 g_j`, which is the fitted-value
  score, while `mathematical_background.rst` asserted coefficient-variance
  optimality. That page contradicted itself, deriving the coefficient score at
  line 39 and writing the prediction score at line 56. The loss is now declared
  through `loss=`, and on an ill-conditioned `X'X` with spread influence
  directions the two choices disagree on roughly a fifth of a budget.

- **`row_se_min_labels` never minimized its objective.** Its dual step clipped
  the gradient at zero, so a multiplier could never fall when its constraint went
  slack; probabilities ratcheted upward and the search stopped at whatever design
  first became feasible. The objective `sum_j pi_j` was never evaluated. The step
  is now signed, and a scalar repair guarantees feasibility, which always exists
  because `pi = 1` everywhere gives variance zero. Closes #21.

- **`mixed_weights` was the truncated estimator under a misleading name.** It set
  weight 1.0 for both core and tail while its docstring described `1/pi` for the
  core, and its bias, called "tiny", is the entire omitted tail mass. On the
  browsing panel that bias runs from 14 to 53 percent of the size of the
  coefficient. It is replaced by explicit shrinkage that reports its own
  worst-case bias bound.

- **The published efficiency figures did not measure efficiency.** The 1.14x
  headline was the mean of five per-coefficient ratios, one of 1.6972 and four
  between 0.9994 and 1.0003. The evaluation drew a `beta_true` it never used to
  generate anything, and its counts were `activity_i x popularity_j`, a rank-one
  mean on which every entity's influence points the same way, so no method could
  differ from any other. All prior efficiency claims are withdrawn.

### Added

- `Study`, the entry point, with `from_long` for panels that arrive as one row
  per unit and entity. A browsing panel of 1,134 users by 34,078 domains is 0.27
  percent dense: 309 MB as a matrix, a few MB as triples.
- Standard errors, which did not previously exist, reported in two parts: the
  design component a budget buys down and the regression component it cannot
  touch. Their ratio is a stopping rule.
- `poisson_pps`, the risk-minimizing design, and `core_plus_tail`, whose
  certainty-core boundary now comes from a closed form rather than a
  `tail_frac` parameter. An entity enters the certainty core only when the
  budget requires it: ten equal scores at a budget of two get probability 0.2
  each.
- `min_cost_for_risk` and `Study.design_for_risk`, the inverse of the budget
  problem: the cheapest design that meets a risk ceiling, with optional entity
  costs and contrasts. Designs report achieved risk, expected cost, and
  certainty-core size.
- A proxy-assisted difference estimator that is exactly unbiased whatever the
  proxy says, so a poor classifier costs variance and never bias.
- `estimate_share_effect`, for the case where the purchased label builds a
  covariate rather than an outcome. There an unbiased constructed variable
  attenuates its own coefficient, and the design's knowledge of its own error
  variance supplies the correction.
- `direction_spectrum`, which reports when a population cannot distinguish one
  design criterion from another. It is the diagnostic that would have caught the
  broken evaluation.
- A benchmark against a browsing panel whose labels are all known, shipped
  hashed so the numbers can be rechecked rather than believed.

### Removed

- `greedy_aopt_selection`, which started from `1e8 * I` rather than `X'X` and so
  optimized a sensor-design objective unrelated to the batch criterion it shared
  a name with, while supplying no inclusion probabilities.
- `adaptive_core_tail`, whose tail fraction averaged two ad-hoc scores and is
  obsoleted by the closed-form core size.
- `mixed_weights`, `items_to_label`, `pi_aopt_for_budget`, `Design`, and the
  `core`, `selection`, `balanced`, `hybrid`, `greedy` and `results` modules.


## [2.0.0] - 2026-08-07

The sampler did not deliver the inclusion probabilities it reported, so every
estimator weighted by `1 / pi` was biased. Anyone who has used `Design.sample`,
`balanced_fixed_size` or `core_plus_tail` for a real estimate should re-draw the
sample and re-estimate.

### Fixed

- **`balanced_fixed_size` now delivers its stated inclusion probabilities.** It
  drew `budget` items with `rng.choice(..., replace=False, p=pi)` and then swapped
  items greedily to improve balance. Neither step preserves `pi`: sequential
  without-replacement draws condition on what came before, and the swap decisions
  depend on the data. On the package's own test fixture (`make_synth(n=60, m=120)`,
  budget 40) items with `pi == 1` were included 74-78% of the time before the
  swaps and 45-93% after, and the correlation between empirical and stated
  inclusion rates was 0.836. `calibrated_ht_estimator` was measurably biased as a
  result -- row 0 was off by 4.28 Monte Carlo standard errors over 400 replicates.

  The sampler is now the cube method (Deville and Tille 2004), which holds the
  inclusion probabilities, the fixed sample size and the balancing constraints
  simultaneously. On the same fixture: certainty items included exactly 100% of
  the time, correlation 0.9998, largest standardised deviation 2.08 sigma across
  120 items over 4000 draws, sample size exactly the budget every time. It is also
  roughly 3x faster, being O(m) rather than a bounded local search.

- **`core_plus_tail` reports the design it runs.** It returned the A-optimal `pi`
  it optimised over, but selects its core deterministically, so core items are
  included with probability one. Weighting them by `1 / pi_j > 1` inflated exactly
  the items chosen for having the most influence. `probabilities` and `ht_weights`
  now describe the executed design: one for the core, and the tail's own rescaled
  probabilities for the tail.

- **`Design.sample(method="balanced")` reports the delivered probabilities.**
  `pi_aopt_for_budget` hits its budget to a solver tolerance rather than exactly;
  the reported vector is now the one the fixed-size design realises.

- **`calibrate_weights(nonneg=True)` no longer breaks the constraint it solves.**
  It computed the exact chi-square calibration and then clipped negative weights
  to zero, which silently violates `G_S w = t` -- the equation the function
  exists to satisfy. On a 100-item fixture the raw solution went negative in 41
  draws out of 200, and in 10 of those the clipped weights reproduced the
  population totals *worse* than the unadjusted Horvitz-Thompson weights. It now
  pins offenders at the floor and re-solves over the rest, restoring the
  constraint exactly (largest residual over 200 draws: 2.8e-07) with no negative
  weights. When too few weights remain free to carry the constraints it warns
  rather than returning a silently wrong answer.

- **`scale_pi_to_budget` rejects a budget its positive entries cannot carry.**
  Scaling leaves a zero at zero, so `scale_pi_to_budget([0, 0, 0.5], 2)` returned
  a vector summing to 1 and produced a sample quietly smaller than requested. It
  now raises.

- **The cube landing phase settles leftovers by systematic pi-ps** rather than
  independent Bernoulli draws. Both preserve the marginals, but only the former
  preserves the count, so accumulated floating-point drift in the constraint rows
  can no longer cost or gain a whole item.

### Added

- `scale_pi_to_budget`, the rescaling the sampler applies. `sum(pi)` is the
  expected sample size, so a fixed-size design of `budget` items is only coherent
  when the probabilities sum to `budget`. Exported so callers can compute the
  delivered probabilities themselves.
- `tests/test_sampling_design.py`: Monte Carlo tests, via
  [simcheck](https://github.com/finite-sample/simcheck), that the empirical
  inclusion rate of every item matches its stated probability, that certainty
  items are always selected, and that `calibrated_ht_estimator` is unbiased row by
  row. Each gate derives its tolerance from the replicate count, and the
  unbiasedness test has a falsification partner that must be caught as biased.
  The suite previously checked only that the sample had the right *length*, which
  a sampler ignoring `pi` entirely also passes.

### Removed

- **Breaking:** `balanced_fixed_size` no longer takes `max_swaps` or `tol`. Both
  parameterised the greedy search that has been removed; the cube method has no
  corresponding knobs. No caller in the package passed either.

## [1.1.1] and earlier

See the git history.

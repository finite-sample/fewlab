# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

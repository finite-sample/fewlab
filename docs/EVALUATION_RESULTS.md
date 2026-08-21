# Evaluation

Probability sampling proportional to influence recovers the complete-label
regression coefficients about nine times more accurately than spending the same
budget at random, and the gap holds at every budget tested. Labeling 250 of
34,078 domains gives a root mean squared error of 5.97 against a coefficient
matrix whose own size is 11.1, with no bias and 94.2% coverage of a nominal 95%
interval.

![Three panels: error, efficiency relative to simple random sampling, and
interval coverage, each against the number of domains labeled, for four
designs.](results/yg_benchmark.png)

## What the benchmark measures

The population is the YouGov Pulse browsing panel: 1,132 US adults metered for
June 2022, and the 34,078 domains whose Blacklight tracking scan succeeded. The
frame is 0.27% dense.

Every domain's label is known, which is the point. The complete-label
coefficient matrix

```
B = (X'X)⁻¹X'VA
```

is therefore computable exactly, and each design is scored against that same
fixed target. The outcome `Y = VA` is the published per-user tracker exposure
rate, and the regression is the published specification: exposure on gender,
race, education and age group. Running that regression on all 34,078 labels
reproduces the four coefficients the manuscript reports to within 0.004, against
its own replication tolerance of 0.02, which is how we know the pipeline computes
the same quantity the paper does.

The estimand has two label classes, ad trackers and third-party cookies, so `B`
is 13 by 2 and `‖B‖ = 11.1`. Each row of the table below is 300 independent
Poisson draws from the same finite population, so the spread is design variance
and nothing else.

Reproduce it with:

```bash
python benchmarks/run_yg_benchmark.py --data benchmarks/data --out docs/results/yg_benchmark.csv
```

## Results

| budget | design | RMSE | RMSE / SRS | bias | coverage | certainty core |
|---|---|---|---|---|---|---|
| 100 | simple random | 121 | 1.000 | 8.7 | 0.838 | 0 |
| 100 | core + uniform tail | 44.1 | 0.365 | 2.8 | 0.868 | 41 |
| 100 | PPS | 10.6 | 0.088 | 0.77 | 0.938 | 6 |
| 100 | PPS, known label norms | 3.99 | 0.033 | 0.23 | 0.954 | 7 |
| 250 | simple random | 52.9 | 1.000 | 2.6 | 0.869 | 0 |
| 250 | core + uniform tail | 25.6 | 0.483 | 1.2 | 0.912 | 105 |
| 250 | PPS | 5.97 | 0.113 | 0.24 | 0.942 | 26 |
| 250 | PPS, known label norms | 2.13 | 0.040 | 0.14 | 0.949 | 38 |
| 500 | simple random | 40.0 | 1.000 | 1.2 | 0.868 | 0 |
| 500 | core + uniform tail | 15.2 | 0.379 | 0.96 | 0.927 | 242 |
| 500 | PPS | 3.86 | 0.096 | 0.23 | 0.945 | 68 |
| 500 | PPS, known label norms | 1.19 | 0.030 | 0.072 | 0.951 | 111 |
| 1,000 | simple random | 26.4 | 1.000 | 1.5 | 0.872 | 0 |
| 1,000 | core + uniform tail | 7.02 | 0.266 | 0.47 | 0.937 | 552 |
| 1,000 | PPS | 2.29 | 0.087 | 0.13 | 0.945 | 185 |
| 1,000 | PPS, known label norms | 0.604 | 0.023 | 0.028 | 0.950 | 311 |
| 2,000 | simple random | 19.1 | 1.000 | 0.79 | 0.877 | 0 |
| 2,000 | core + uniform tail | 3.10 | 0.163 | 0.15 | 0.943 | 1,176 |
| 2,000 | PPS | 1.18 | 0.062 | 0.047 | 0.952 | 550 |
| 2,000 | PPS, known label norms | 0.268 | 0.014 | 0.017 | 0.947 | 791 |

RMSE and bias are Frobenius norms of the error in `B`, in the same units as `B`
itself, whose norm is 11.1. Coverage is the share of nominal 95% intervals
containing the truth, pooled over the 26 coefficients. The certainty core is the
number of domains the cap at probability one selects, which the design produces
rather than the user choosing.

Four things in that table are worth reading.

Every design is unbiased, including the bad one. Simple random sampling has bias
2.6 at budget 250 against a Monte Carlo standard error of about 3.1, so it is not
distinguishable from zero. The designs differ in variance, not in bias, which is
what a probability design guarantees.

The efficiency ratio is flat in budget. PPS sits between 0.062 and 0.113 of the
random-sampling error across a twentyfold range of budgets, so the gain is a
property of the design rather than a lucky operating point.

Intervals from a good design cover and intervals from a bad one do not. Simple
random sampling covers at 0.84 to 0.88 against a nominal 0.95. The cause is not a
broken variance estimator: over 4,000 replicates the estimator is unbiased for the
true variance to within 1%. What breaks is the normal approximation. Under simple
random sampling a handful of very high influence domains either land in the sample
or do not, which leaves the estimator skewed at ±1.6 and its variance estimate
swinging by 200% of its own mean. Under PPS the skew is 0.1 and the variance
estimate varies by 30%. The good design does not only shrink the interval, it is
what makes the interval mean anything.

Knowing the label norms is worth another factor of three. The two PPS rows differ
only in whether `a_j'a_j` is known. Blacklight labels are counts, not one-hot
class indicators, and their squared norms run from a median of 10 to a maximum of
72,262, so the shortcut that makes `b_j` exact for a mutually exclusive taxonomy
does not apply here. That gap is what a pilot sample or a proxy is worth on this
population.

## Buying bias on purpose

Shrinking the weight by `α` trades bias for variance. The same domains are
labeled in every row below; only the weight differs. At `α = 1` the weight is
`1/π_j`. At `α = 0` the weight is 1, which is what zero-filling unlabeled
entities amounts to.

| budget | α | bias | sd | RMSE |
|---|---|---|---|---|
| 250 | 0.00 | 5.86 | 1.01 | 5.95 |
| 250 | 0.25 | 4.38 | 2.01 | 4.82 |
| 250 | 0.50 | 2.91 | 3.29 | **4.39** |
| 250 | 0.75 | 1.44 | 4.62 | 4.84 |
| 250 | 0.90 | 0.579 | 5.43 | 5.46 |
| 250 | 1.00 | 0.245 | 5.97 | 5.97 |
| 1,000 | 0.00 | 3.72 | 0.599 | 3.76 |
| 1,000 | 0.25 | 2.79 | 0.943 | 2.95 |
| 1,000 | 0.50 | 1.87 | 1.38 | 2.32 |
| 1,000 | 0.75 | 0.947 | 1.83 | **2.06** |
| 1,000 | 0.90 | 0.409 | 2.12 | 2.16 |
| 1,000 | 1.00 | 0.159 | 2.31 | 2.31 |
| 4,000 | 0.00 | 1.53 | 0.188 | 1.55 |
| 4,000 | 0.25 | 1.15 | 0.260 | 1.18 |
| 4,000 | 0.50 | 0.772 | 0.349 | 0.848 |
| 4,000 | 0.75 | 0.392 | 0.446 | 0.594 |
| 4,000 | 0.90 | 0.165 | 0.506 | **0.532** |
| 4,000 | 1.00 | 0.0331 | 0.546 | 0.547 |

Bold marks the smallest error in each block. The optimum is interior and moves
toward 1 as the budget grows: shrinkage cuts mean squared error by 26% at 250
labels, 11% at 1,000, and 3% at 4,000. Regularization helps most where the
variance is worst, which is what regularization does.

Two cautions. The optimal `α` depends on the truth you are trying to estimate, so
it cannot be read off in advance; the bias term is estimable from the sample, but
that plug-in is itself noisy. And `α = 0` is never optimal at any budget here,
with a bias running from 14% to 53% of the size of `B`. Dropping the unlabeled
entities is a real point on the frontier, just never a good one.

## Fixed-size draws are covered conservatively

The reported variance is the Poisson expression `Σ (1−π_j)/π_j · b_j`, which is
exact for independent Bernoulli draws. Fixing the sample size and balancing on the
influence totals both remove variation that expression still counts, so a cube
draw has a smaller true variance than the number reported. Measured on a synthetic
panel, the reported standard error exceeds the actual spread by 18% to 43%, and
coverage runs 0.96 to 0.99 against a nominal 0.95. Intervals from
`draw(fixed_size=True)` are therefore conservative rather than wrong.

## What the previous evaluation measured

Versions up to 2.0 reported a 1.14 times efficiency gain, a 9.2% gain at moderate
budgets, and a roughly 33% reduction in standard errors. All three have been
withdrawn.

The 1.14 figure was the mean of five per-coefficient ratios: 1.6972, 0.99993,
1.00016, 1.00026 and 0.99941. Four of those are noise at ±0.05%, and their mean
with the fifth is 1.1394. The accompanying text claimed the gain was consistent
across all coefficients; the results file in the same directory showed it was one
intercept.

The reason nothing else moved is that the simulation could not have detected a
difference. Its counts were generated as activity times popularity, a rank-one
mean, so every entity's influence vector pointed the same way. The normalized
influence matrix had singular values 1.000, 0.001, 0.000, 0.000. On such a
population every criterion ranks entities identically and every method selects the
same set, which is why all four methods reported errors agreeing to four
significant figures. The simulation also drew a `beta_true`, never used it to
generate labels, counts or outcomes, and then measured distance to it, which added
roughly 2.25 of irrelevant variance to every reported number.

The 33% figure cited a notebook that imported the package only to print its
version number, never called a selection function, subsampled rows rather than
entities, and reported ratios averaging 1.00 in its own stored output, one of them
below 1.

`Study.diagnostics["direction_spectrum"]` now reports the degeneracy that hid the
first problem, and the benchmark asserts on it, so a population that cannot
distinguish designs fails the harness instead of flattering it.

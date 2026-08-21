# Evaluation

Probability sampling proportional to influence recovers the complete-label
regression coefficients about nine times more accurately than spending the same
budget at random, and the gap holds at every budget tested.

The full results, with the tables and the figure, are in the manuscript under
`ms/`. This page describes what the benchmark measures and how to re-run it,
and does not restate the numbers, because a second copy of a computed value is
a copy that goes stale.

## What is measured

The population is the YouGov Pulse browsing panel: US adults metered for June
2022, and the domains whose Blacklight tracking scan succeeded. Every domain in
that frame has a known label, which is the point. The complete-label coefficient
matrix

```
B = (X'X)⁻¹X'VA
```

is therefore computable exactly, and each design is scored against that same
fixed target rather than against an estimate of one.

The outcome `Y = VA` is the published per-user tracker exposure rate, and the
regression is the published specification: exposure on gender, race, education
and age group. Running that regression on every label reproduces the four
coefficients the source manuscript reports, well inside its own replication
tolerance, which is how we know the pipeline computes the same quantity.

Each row of the results is many independent Poisson draws from the same finite
population, so the spread is design variance and nothing else.

## Reproducing it

```bash
python benchmarks/make_exhibits.py --data benchmarks/data --out ms/generated
```

That writes the tables, the figure, and every number the manuscript quotes, as
`ms/generated/tab_evidence.tex`, `tab_frontier.tex`, `fig_efficiency.pdf`, and
`macros.tex`. The raw results land beside them as `benchmark.csv` and
`frontier.csv`. Nothing computed is typed into the prose anywhere.

The extract itself is regenerated from the private source repository by
`benchmarks/build_yg_extract.py`; see `benchmarks/data/README.md` for its
provenance and for what the hashing does and does not protect.

## What the previous evaluation measured

Versions up to 2.0 reported a 1.14 times efficiency gain, a 9.2% gain at
moderate budgets, and a roughly 33% reduction in standard errors. All three are
withdrawn.

The 1.14 figure was the mean of five per-coefficient ratios: 1.6972, 0.99993,
1.00016, 1.00026 and 0.99941. Four of those are noise at a twentieth of a
percent, and their mean with the fifth is 1.1394. The accompanying text claimed
the gain was "consistent across all coefficients"; the results file in the same
directory showed it was one intercept.

The reason nothing else moved is that the simulation could not have detected a
difference. Its counts were generated as activity times popularity, a rank-one
mean, so every entity's influence vector pointed the same way and the normalized
influence matrix had singular values 1.000, 0.001, 0.000, 0.000. On such a
population every criterion ranks entities identically and every method selects
the same set, which is why all four methods reported errors agreeing to four
significant figures. The simulation also drew a `beta_true`, never used it to
generate labels, counts or outcomes, and then measured distance to it, adding
roughly 2.25 of irrelevant variance to every reported number.

The 33% figure cited a notebook that imported the package only to print its
version number, never called a selection function, subsampled rows rather than
entities, and reported ratios averaging 1.00 in its own stored output, one of
them below 1. That notebook has been rewritten as a walkthrough that uses the
package.

`Study.diagnostics["direction_spectrum"]` now reports the degeneracy that hid the
first problem, and the test suite asserts on it, so a population that cannot
distinguish designs fails the harness instead of flattering it.

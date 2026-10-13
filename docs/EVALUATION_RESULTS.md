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

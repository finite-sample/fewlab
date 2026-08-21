# YouGov Pulse browsing extract

A hashed extract of the browsing panel used in *Blacklight*, kept here so the
efficiency numbers in the documentation can be reproduced rather than taken on
trust.

## Provenance

YouGov fielded a browsing panel metered by RealityMine in June 2022, matched to
US population margins. 1,134 panelists with usable browsing and profile data
contributed 6,236,834 visits across 64,074 registered domains. Each domain was
scanned by Blacklight (The Markup) for seven tracking measures; 34,078 scans
succeeded, covering 53.19% of domains and 76.43% of visits.

Source data: Harvard Dataverse `doi:10.7910/DVN/VIV4TS` (panel) and
`doi:10.7910/DVN/3N7TDZ` (Blacklight scans).

## What this extract is

The evaluation frame is the 34,078 domains whose label is *known*, because a
benchmark needs a truth to measure against. On that frame the complete-label
coefficient `B = (X'X)^-1 X' V A` is computable exactly, so every design is
scored against the same fixed target.

| file | rows | contents |
|---|---|---|
| `visits.csv.gz` | 103,853 | `user, entity, visits, duration` |
| `labels.csv.gz` | 34,078 | `entity` plus the seven Blacklight measures |
| `features.csv.gz` | 1,132 | demographic design matrix, one row per user |

`user` and `entity` are salted SHA-256 digests; no domain name or panelist
identifier is recoverable from this directory. The feature matrix reproduces the
published specification -- gender, race, education and age group, with Male,
White, "HS or Below" and under-25 as reference levels.

Hashing identifiers is not the same as anonymity. Browsing vectors are
distinctive, and a determined party with auxiliary data could in principle
re-identify panelists from visit patterns even without domain names. The extract
is published deliberately, on the judgment that the reproducibility is worth it
given the underlying visit file is already public.

## Regenerating it

    python benchmarks/build_yg_extract.py --source ../private_blacklight --out benchmarks/data

## Running the benchmark

    python benchmarks/run_yg_benchmark.py --data benchmarks/data --out docs/results/yg_benchmark.csv

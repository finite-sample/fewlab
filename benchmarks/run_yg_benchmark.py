"""Score label-acquisition designs against a known finite-population truth.

The evaluation frame is the 34,078 YouGov-panel domains whose Blacklight scan
succeeded. Because their labels are known, the complete-label coefficient

    B = (X'X)^-1 X' V A

is computable exactly, and every design can be scored against that same fixed
target rather than against a guess.

Each design is realized many times over the *same* finite population, so the
spread reported is design variance and nothing else.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fewlab

MEASURES = ["ddg_join_ads", "third_party_cookies"]


def load(data: Path) -> tuple[fewlab.Study, pd.DataFrame]:
    """Load the hashed extract into a study and its labels.

    Args:
        data: Directory holding the extract.

    Returns:
        The study and the true label frame.
    """
    visits = pd.read_csv(data / "visits.csv.gz")
    labels = pd.read_csv(data / "labels.csv.gz").set_index("entity")[MEASURES]
    X = pd.read_csv(data / "features.csv.gz", index_col=0)
    study = fewlab.Study.from_long(
        visits, X, unit="user", entity="entity", count="visits", entities=labels.index
    )
    return study, labels.astype(float)


def truth(study: fewlab.Study, labels: pd.DataFrame) -> np.ndarray:
    """Compute the exact complete-label coefficient matrix.

    Args:
        study: The study.
        labels: True labels for every entity in the frame.

    Returns:
        The `(p, q)` finite-population coefficient matrix.
    """
    return study.influence.H @ labels.loc[study.influence.items].to_numpy(float)


def run(
    study: fewlab.Study,
    labels: pd.DataFrame,
    B: np.ndarray,
    budget: int,
    reps: int,
    seed: int,
) -> list[dict]:
    """Score every design at one budget.

    Args:
        study: The study.
        labels: True labels.
        B: The finite-population truth.
        budget: Number of entities labeled.
        reps: Monte Carlo replicates per design.
        seed: Base seed.

    Returns:
        One record per design.
    """
    norms = (labels**2).sum(axis=1).reindex(study.influence.items)
    variants = {
        "srs": {"method": "srs"},
        "core_plus_tail": {"method": "core_plus_tail"},
        "pps": {"method": "pps"},
        "pps_oracle_norms": {"method": "pps", "label_norms": norms.to_numpy()},
    }
    out = []
    for name, kwargs in variants.items():
        design = study.design(budget, **kwargs)
        rng = np.random.default_rng(seed)
        errs, covered, sizes = [], [], []
        for _ in range(reps):
            sample = study.draw(design, random_state=rng, fixed_size=False)
            if len(sample) == 0:
                continue
            est = fewlab.estimate(study.influence, sample, labels.loc[sample.items])
            E = est.coefficients.to_numpy()
            S = est.design_se.to_numpy()
            errs.append(E - B)
            covered.append(np.abs(E - B) <= 1.96 * S)
            sizes.append(len(sample))
        err = np.array(errs)
        out.append(
            {
                "budget": budget,
                "design": name,
                "reps": len(err),
                "mean_size": float(np.mean(sizes)),
                "rmse": float(np.sqrt((err**2).sum(axis=(1, 2)).mean())),
                "bias_norm": float(np.linalg.norm(err.mean(axis=0))),
                "coverage": float(np.mean(covered)),
                "certainty_core": len(design.certain),
            }
        )
    return out


def frontier(
    study: fewlab.Study,
    labels: pd.DataFrame,
    B: np.ndarray,
    budgets: list[int],
    reps: int,
) -> pd.DataFrame:
    """Trace the bias-variance frontier of shrinking the Horvitz-Thompson weight.

    The same entities are labeled in every row; only the weight differs. At
    `alpha = 1` the weight is `1/pi_j` and the estimator is unbiased. At
    `alpha = 0` the weight is one, which is what zero-filling the unlabeled
    entities amounts to.

    Args:
        study: The study.
        labels: True labels.
        B: The finite-population truth.
        budgets: Budgets to trace.
        reps: Replicates per point.

    Returns:
        One row per budget and shrinkage level.
    """
    rows = []
    for budget in budgets:
        design = study.design(budget)
        rng = np.random.default_rng(4242 + budget)
        samples = [
            study.draw(design, random_state=rng, fixed_size=False) for _ in range(reps)
        ]
        for alpha in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            a = 1.0 if alpha == 1.0 else alpha + (1.0 - alpha) * design.probabilities
            err = np.array(
                [
                    fewlab.estimate(
                        study.influence, s, labels.loc[s.items], alpha=a
                    ).coefficients.to_numpy()
                    - B
                    for s in samples
                ]
            )
            rows.append(
                {
                    "budget": budget,
                    "alpha": alpha,
                    "bias": float(np.linalg.norm(err.mean(axis=0))),
                    "sd": float(np.sqrt(err.var(axis=0).sum())),
                    "rmse": float(np.sqrt((err**2).sum(axis=(1, 2)).mean())),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the benchmark and print the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument(
        "--budgets", type=int, nargs="+", default=[100, 250, 500, 1000, 2000]
    )
    args = ap.parse_args()

    study, labels = load(args.data)
    B = truth(study, labels)
    print(f"frame: {study!r}")
    print(f"truth B: {B.shape}, ||B||_F = {np.linalg.norm(B):.4f}\n")

    rows = []
    for budget in args.budgets:
        rows.extend(run(study, labels, B, budget, args.reps, seed=20250820 + budget))
    table = pd.DataFrame(rows)
    table["rmse_vs_srs"] = table.groupby("budget")["rmse"].transform(
        lambda s: s / s[table.loc[s.index, "design"] == "srs"].iloc[0]
    )
    pd.set_option("display.width", 140)
    print(table.to_string(index=False))
    curve = frontier(study, labels, B, [250, 1000, 4000], reps=min(args.reps, 200))
    print("\nShrinkage frontier (same entities labeled; only the weight differs)")
    print(curve.to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        curve.to_csv(args.out.parent / "shrinkage_frontier.csv", index=False)
        (args.out.parent / "truth.json").write_text(
            json.dumps(
                {
                    "frobenius_norm": float(np.linalg.norm(B)),
                    "n_units": int(study.counts.shape[0]),
                    "n_entities": int(study.influence.n_items),
                    "density": float(study.counts.density),
                    "measures": MEASURES,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

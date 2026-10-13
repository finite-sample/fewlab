"""Build a hashed, redistributable extract of the YouGov Pulse browsing panel.

The panel is 1,134 US adults metered for June 2022 (YouGov Pulse / RealityMine),
6.2M visits across 64,074 registered domains. Each domain was scanned by
Blacklight for seven tracking measures; 34,078 scans succeeded. That scan is the
expensive label this package exists to allocate, and the published analysis
regresses per-user tracker exposure -- which is exactly `Y = V A`, a visit-share
weighted average of domain labels -- on demographics.

This script reads the private source repository and writes an extract in which
every user id and every domain name is replaced by a salted hash. Nothing here
reconstructs a domain name or a panelist.

Run:
    python benchmarks/build_yg_extract.py --source ../private_blacklight --out DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "fewlab-benchmark-v1"
LABELS = [
    "ddg_join_ads",
    "third_party_cookies",
    "canvas_fingerprinting",
    "session_recording",
    "key_logging",
    "fb_pixel",
    "google_analytics",
]


def _hash(value: object, prefix: str) -> str:
    digest = hashlib.sha256(f"{SALT}|{value}".encode()).hexdigest()[:12]
    return f"{prefix}{digest}"


def build_features(profile: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the published demographic design matrix.

    Reference levels match the manuscript: Male, White, HS or Below, under 25.

    Args:
        profile: Raw YouGov profile file.

    Returns:
        A design matrix with an intercept, indexed by raw caseid.
    """
    p = profile.dropna(subset=["caseid"]).copy()
    p["caseid"] = p["caseid"].astype(np.int64)
    p = p.set_index("caseid")
    gender = p["gender"].map({1: "Male", 2: "Female"})
    race = (
        p["race"]
        .map({1: "White", 2: "Black", 3: "Hispanic", 4: "Asian"})
        .fillna("Other")
    )
    educ = p["educ"].map(
        {
            1: "HS or Below",
            2: "HS or Below",
            3: "Some college",
            4: "Some college",
            5: "College",
            6: "Postgrad",
        }
    )
    age = pd.cut(
        p["birthyr"],
        bins=[1929, 1958, 1973, 1988, 1998, 2003],
        labels=["65+", "50-64", "35-49", "25-34", "<25"],
    )
    frame = pd.DataFrame(
        {"gender": gender, "race": race, "educ": educ, "age": age}
    ).dropna()
    dummies = pd.get_dummies(frame, drop_first=False).astype(float)
    drop = ["gender_Male", "race_White", "educ_HS or Below", "age_<25"]
    dummies = dummies.drop(columns=[c for c in drop if c in dummies])
    dummies.insert(0, "const", 1.0)
    return dummies


def main() -> None:
    """Build and write the extract."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src, out = args.source, args.out
    out.mkdir(parents=True, exist_ok=True)

    visits = pd.read_csv(src / "data/yg/yg_ind_domain.csv")
    bl = pd.read_csv(src / "data/blacklight_domain.csv")
    profile = pd.read_csv(src / "data/yg/profile.csv")

    bl["private_domain"] = bl["filename"].str.replace("_", ".", regex=False)
    bl = bl.drop(columns=["filename"]).drop_duplicates("private_domain")

    X = build_features(profile)

    panel_domains = set(visits["private_domain"].unique())
    labeled = bl[bl["private_domain"].isin(panel_domains)].copy()

    stats = {
        "panel_users": int(visits["caseid"].nunique()),
        "panel_domains": len(panel_domains),
        "panel_visits": int(visits["visits"].sum()),
        "labeled_domains": len(labeled),
        "label_coverage_domains": round(len(labeled) / len(panel_domains), 4),
    }

    # The evaluation frame is the domains whose true label is known, because a
    # benchmark needs a truth to be measured against. Restricting to them makes
    # the finite-population coefficient B exactly computable, so every design can
    # be scored against the same fixed target rather than against a guess.
    keep = visits[visits["private_domain"].isin(set(labeled["private_domain"]))].copy()
    keep = keep[keep["caseid"].isin(X.index)]
    stats["labeled_visit_share"] = round(
        float(keep["visits"].sum() / visits["visits"].sum()), 4
    )
    stats["eval_users"] = int(keep["caseid"].nunique())
    stats["eval_domains"] = int(keep["private_domain"].nunique())
    stats["eval_rows"] = len(keep)
    stats["density"] = round(
        len(keep) / (stats["eval_users"] * stats["eval_domains"]), 6
    )

    keep["user"] = [_hash(c, "u") for c in keep["caseid"]]
    keep["entity"] = [_hash(d, "e") for d in keep["private_domain"]]
    labeled = labeled[
        labeled["private_domain"].isin(set(keep["private_domain"]))
    ].copy()
    labeled["entity"] = [_hash(d, "e") for d in labeled["private_domain"]]

    X_out = X.loc[sorted(set(keep["caseid"]))].copy()
    X_out.index = [_hash(c, "u") for c in X_out.index]
    X_out.index.name = "user"

    keep[["user", "entity", "visits", "duration"]].to_csv(
        out / "visits.csv.gz", index=False, compression="gzip"
    )
    labeled[["entity", *LABELS]].to_csv(
        out / "labels.csv.gz", index=False, compression="gzip"
    )
    X_out.to_csv(out / "features.csv.gz", compression="gzip")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

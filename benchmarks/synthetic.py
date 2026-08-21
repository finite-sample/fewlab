"""Two synthetic experiments the manuscript quotes, so it need not quote them by hand.

Both concern claims that cannot be measured on the browsing panel. The first
needs a known coefficient on a constructed covariate, which the panel does not
supply. The second needs an ill-conditioned design matrix with influence vectors
spanning several directions, which is the case where declaring the loss changes
the answer; the panel is better conditioned than that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fewlab

TRUE_GAMMA = 2.0


def covariate_experiment(reps: int = 300, budget: int = 60, seed: int = 77) -> dict:
    """Measure attenuation when the label builds a covariate.

    Args:
        reps: Monte Carlo replicates.
        budget: Labeling budget.
        seed: Seed for the population.

    Returns:
        The naive and corrected means, the attenuation and overshoot in percent,
        the correlation between the outcome and covariate scores, and the
        percentage gain from targeting the covariate score.
    """
    rng = np.random.default_rng(seed)
    n, m, p = 800, 300, 3
    Z = rng.normal(size=(n, p - 1))
    X = pd.DataFrame(np.c_[np.ones(n), Z], columns=["const", "x1", "x2"])
    loadings = rng.normal(size=(m, p - 1))
    popularity = np.exp(rng.normal(0, 1.3, size=m))
    rate = np.exp(0.4 + Z @ loadings.T + np.log(popularity) - np.log(popularity).mean())
    counts = pd.DataFrame(rng.poisson(rate), columns=[f"e{j}" for j in range(m)])

    study = fewlab.Study(counts, X)
    labels = pd.Series(rng.integers(0, 2, m).astype(float), index=study.influence.items)
    V = counts.to_numpy(float) / counts.sum(axis=1).to_numpy(float)[:, None]
    y = V @ labels.to_numpy(float)
    z = pd.Series(
        TRUE_GAMMA * y
        + X.to_numpy(float) @ np.array([0.5, -0.3, 0.2])
        + rng.normal(0, 0.05, n),
        index=counts.index,
    )

    def run(loss: str, correct: bool) -> np.ndarray:
        design = study.design(budget, loss=loss)
        gen = np.random.default_rng(3)
        out = []
        for _ in range(reps):
            sample = study.draw(design, random_state=gen, fixed_size=False)
            if len(sample) == 0:
                continue
            out.append(
                fewlab.estimate_share_effect(
                    study.influence,
                    study.counts,
                    study.X,
                    sample,
                    labels.loc[sample.items],
                    z,
                    correct=correct,
                ).coefficient
            )
        return np.array(out)

    naive = run("regressor", correct=False)
    corrected = run("regressor", correct=True)
    outcome_design = run("coefficient", correct=False)

    outcome_score = study.scores(loss="coefficient")
    regressor_score = study.scores(loss="regressor")
    rho = float(outcome_score.corr(regressor_score, method="spearman"))
    gain = 1 - np.sqrt(((naive - TRUE_GAMMA) ** 2).mean()) / np.sqrt(
        ((outcome_design - TRUE_GAMMA) ** 2).mean()
    )

    return {
        "NaiveGamma": f"{naive.mean():.1f}",
        "TrueGamma": f"{TRUE_GAMMA:g}",
        "AttenuationPct": f"{100 * (1 - naive.mean() / TRUE_GAMMA):.0f}",
        "OvershootPct": f"{100 * (corrected.mean() / TRUE_GAMMA - 1):.0f}",
        "ScoreCorrelation": f"{rho:.3f}",
        "RegressorGainPct": f"{100 * gain:.0f}",
    }


def loss_divergence(budget: int = 40, seed: int = 7) -> dict:
    """Measure how much the top-K set moves when the declared loss changes.

    Args:
        budget: Size of the top-K set compared.
        seed: Seed for the population.

    Returns:
        The percentage of the top-K set that differs between the two losses.
    """
    rng = np.random.default_rng(seed)
    n, m = 600, 400
    Z = rng.normal(size=(n, 2))
    # Strongly correlated covariates push (X'X)^-1 away from a scaled identity,
    # which is what lets the two quadratic forms rank items differently.
    X = pd.DataFrame(
        np.c_[np.ones(n), Z[:, 0], 0.97 * Z[:, 0] + np.sqrt(1 - 0.97**2) * Z[:, 1]],
        columns=["const", "x1", "x2"],
    )
    loadings = rng.normal(size=(m, 2)) * 1.5
    popularity = np.exp(rng.normal(0, 1.0, size=m))
    rate = np.exp(0.5 + Z @ loadings.T + np.log(popularity) - np.log(popularity).mean())
    counts = pd.DataFrame(rng.poisson(rate), columns=[f"e{j}" for j in range(m)])

    study = fewlab.Study(counts, X)
    a = set(study.scores(loss="coefficient").nlargest(budget).index)
    b = set(study.scores(loss="prediction").nlargest(budget).index)
    return {"LossDisagreementPct": f"{100 * (1 - len(a & b) / budget):.0f}"}


def variance_comparison(reps: int = 1200, budget: int = 45, seed: int = 31) -> dict:
    """Compare the three variance estimators against the realized spread.

    Each is derived for a different draw, so each should be right on its own and
    wrong on the others. Measuring that is the only way to say so honestly.

    Args:
        reps: Replicates for the Bernoulli draw; the cube draw uses half, being
            slower.
        budget: Labeling budget.
        seed: Seed for the population.

    Returns:
        Median ratio of estimated to actual variance, per draw and estimator.
    """
    rng = np.random.default_rng(seed)
    n, m = 300, 120
    Z = rng.normal(size=(n, 2))
    X = pd.DataFrame(np.c_[np.ones(n), Z], columns=["const", "x1", "x2"])
    loadings = rng.normal(size=(m, 2))
    popularity = np.exp(rng.normal(0, 1.3, size=m))
    rate = np.exp(0.4 + Z @ loadings.T + np.log(popularity) - np.log(popularity).mean())
    counts = pd.DataFrame(rng.poisson(rate), columns=[f"e{j}" for j in range(m)])

    study = fewlab.Study(counts, X)
    draw = rng.integers(0, 2, m)
    labels = pd.DataFrame(
        np.eye(2)[draw], index=study.influence.items, columns=["c0", "c1"]
    )

    design = study.design(budget)
    out = {}
    for fixed, tag, n_reps in ((False, "Poisson", reps), (True, "Cube", reps // 2)):
        gen = np.random.default_rng(21)
        values: list[np.ndarray] = []
        var: dict[str, list[np.ndarray]] = {
            "poisson": [],
            "hajek": [],
            "balanced": [],
        }
        for _ in range(n_reps):
            sample = study.draw(design, random_state=gen, fixed_size=fixed)
            if len(sample) == 0:
                continue
            obs = labels.loc[sample.items]
            values.append(
                fewlab.estimate(study.influence, sample, obs)
                .coefficients.to_numpy()
                .ravel()
            )
            for name in var:
                var[name].append(
                    fewlab.estimate(study.influence, sample, obs, variance=name)
                    .design_se.to_numpy()
                    .ravel()
                    ** 2
                )
        actual = np.array(values).var(axis=0)
        for name, v in var.items():
            ratio = float(np.median(np.array(v).mean(axis=0) / actual))
            out[f"Var{tag}{name.capitalize()}"] = f"{ratio:.2f}"
    return out

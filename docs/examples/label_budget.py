# /// script
# requires-python = ">=3.12"
# dependencies = ["fewlab>=4.0", "marimo", "numpy", "pandas"]
# ///
"""Which websites should you pay to label? A marimo notebook."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import fewlab

    return fewlab, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Which websites should you pay to label?

    A browsing panel records who visited what. You want to regress something
    about the *sites* on features of the *people*: whether older adults
    encounter more trackers, whether education predicts news consumption. But
    determining what a site is costs money, and you cannot afford all of them.

    This notebook runs the whole decision on a real panel where every label is
    already known, so the answer can be checked against the truth instead of
    argued about.

    ## The data

    1,132 US panelists metered for June 2022 and 34,078 domains whose
    Blacklight tracking scan succeeded. Identifiers are salted hashes; no
    domain name or panelist is recoverable. The matrix is 0.27% dense, so it
    arrives in long format and stays that way.
    """)
    return


@app.cell
def _(mo, pd):
    data = mo.notebook_location() / "public"
    visits = pd.read_csv(str(data / "visits.csv.gz"))
    labels = pd.read_csv(str(data / "labels.csv.gz")).set_index("entity")
    features = pd.read_csv(str(data / "features.csv.gz"), index_col=0)

    MEASURES = ["ddg_join_ads", "third_party_cookies"]
    labels = labels[MEASURES].astype(float)
    mo.md(
        f"{visits.user.nunique():,} panelists, {len(labels):,} domains, "
        f"{len(visits):,} nonzero cells"
    )
    return MEASURES, features, labels, visits


@app.cell
def _(features, fewlab, labels, visits):
    study = fewlab.Study.from_long(
        visits,
        features,
        unit="user",
        entity="entity",
        count="visits",
        entities=labels.index,
    )
    study
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Before buying anything, check that a design can help

    If every site influences the regression in the same direction and they
    differ only in magnitude, then every criterion ranks sites identically and
    the choice of estimand stops mattering. Read the spectrum against a noise
    floor of about `1/sqrt(mean count)`, not against a fixed cutoff.
    """)
    return


@app.cell
def _(mo, np, study):
    spectrum = study.diagnostics["direction_spectrum"]
    mo.md(
        f"Spectrum: `{np.round(spectrum.to_numpy(), 3)}`\n\n"
        f"Effective directions: **{study.diagnostics['effective_directions']}**,"
        " so this population has real structure to exploit and the declared"
        " loss will matter."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What each site is worth

    `b_j` is site `j`'s contribution to the design risk. It needs no labels:
    only the visit counts and the panelist features.
    """)
    return


@app.cell
def _(pd, study):
    b = study.scores(loss="coefficient")
    share = b.sort_values(ascending=False).cumsum() / b.sum()
    tops = [10, 50, 250, 1000]
    pd.DataFrame(
        {"share of contribution mass": [share.iloc[k - 1] for k in tops]},
        index=pd.Index(tops, name="top sites"),
    ).round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The head is extraordinarily concentrated, which is exactly the situation
    the design is for.

    ## Choose a budget

    Everything below this slider is recomputed for the number of labels you
    can afford. In the browser each change takes several seconds, and the
    slider responds once the page has finished loading Python.
    """)
    return


@app.cell
def _(mo):
    budget = mo.ui.slider(
        steps=[100, 250, 500, 1000, 2000, 4000],
        value=250,
        label="labels to buy",
        show_value=True,
        debounce=True,
    )
    budget
    return (budget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the design

    `pi_j = min(1, lambda * sqrt(b_j / c_j))`. The cap at one is not a separate
    rule for big sites: the set it binds on *is* the certainty core. Nothing
    is assigned probability zero except sites nobody visited, which contribute
    nothing and so cost no bias. That is what keeps the estimate unbiased.
    """)
    return


@app.cell
def _(budget, mo, study):
    design = study.design(budget.value)
    core = design.certain
    tail = (design.probabilities > 0).sum() - len(core)
    mo.md(
        f"""
    | | |
    |---|---|
    | budget | {design.budget:,.0f} |
    | certainty core | {len(core)} sites labeled with probability 1 |
    | sampled tail | {tail:,} sites at pi < 1 |
    | largest weight | {design.diagnostics["max_weight"]:,.0f} |
    """
    )
    return core, design


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw the sample, label it, estimate

    A design is a plan. Drawing it gives the list you actually send to be
    labeled. The cube method lands on the budget exactly and balances on the
    influence totals, which reduces variance without touching the
    probabilities. Here the labels are already known, which is what makes this
    checkable; in practice labeling is the step that costs money.
    """)
    return


@app.cell
def _(core, design, labels, mo, study):
    sample = study.draw(design, random_state=0, fixed_size=True)
    est = study.estimate(sample, labels.loc[sample.items])
    mo.md(
        f"{len(sample)} sites to label; the core is in every draw: "
        f"{set(core) <= set(sample.items)}.\n\n"
        f"Mean design SE: **{est.design_se.to_numpy().mean():.3f}**. "
        "This uncertainty comes from labeling a subset of sites; the panel "
        "and its features are held fixed."
    )
    return (est,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check it against the truth

    Because every label in this frame is known, the coefficient you would get
    by labeling all 34,078 sites is computable exactly. Nothing else in this
    notebook uses it. `z` is how many standard errors each estimate sits from
    the truth; values scattered around zero are what an unbiased estimator
    with an honest standard error looks like.
    """)
    return


@app.cell
def _(MEASURES, est, labels, pd, study):
    H = study.influence.H
    truth = pd.DataFrame(
        H @ labels.loc[study.influence.items].to_numpy(float),
        index=est.coefficients.index,
        columns=MEASURES,
    )
    comparison = pd.DataFrame(
        {
            "truth": truth["ddg_join_ads"],
            "estimate": est.coefficients["ddg_join_ads"],
            "design se": est.design_se["ddg_join_ads"],
        }
    )
    comparison["z"] = (comparison["estimate"] - comparison["truth"]) / comparison[
        "design se"
    ]
    comparison.round(3)
    return (truth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the design bought

    The same budget spent uniformly at random, against the influence design,
    over 60 independent draws of each.
    """)
    return


@app.cell
def _(budget, fewlab, labels, mo, np, study, truth):
    def rmse(method, reps=60, seed=1):
        d = study.design(budget.value, method=method)
        rng = np.random.default_rng(seed)
        err = []
        for _ in range(reps):
            s = study.draw(d, random_state=rng, fixed_size=False)
            if len(s) == 0:
                continue
            e = fewlab.estimate(study.influence, s, labels.loc[s.items])
            err.append(e.coefficients.to_numpy() - truth.to_numpy())
        err = np.array(err)
        return float(np.sqrt((err**2).sum(axis=(1, 2)).mean()))

    pps, srs = rmse("pps"), rmse("srs")
    mo.md(
        f"""
    | | RMSE |
    |---|---|
    | uniform random | {srs:.2f} |
    | influence design | {pps:.2f} ({srs / pps:.1f}x better) |

    For scale, the coefficient matrix itself has norm
    {np.linalg.norm(truth.to_numpy()):.1f}.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What not to do

    The tempting shortcut is to label the top sites and treat the rest as
    zero. That is a real point on the bias-variance frontier and it is never
    the best one: on this panel its bias runs from 14 to 53 percent of the
    size of the coefficient, and unlike variance it cannot be estimated from
    the data you have. The package will not let a ranking reach an estimator
    by accident:
    """)
    return


@app.cell
def _(budget, labels, mo, study):
    ranking = study.rank(budget.value)
    try:
        study.estimate(ranking, labels.loc[ranking.items])
        refusal = mo.md("The ranking was accepted.")
    except TypeError as exc:
        refusal = mo.md(f"`TypeError: {exc}`")
    refusal
    return


if __name__ == "__main__":
    app.run()

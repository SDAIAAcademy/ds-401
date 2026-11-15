# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 1.2 · Prior predictive checks
#
# Day 1 · Session 2 (Priors you can defend) · 25 min.
#
# A contractor proposed a "harmless" flat prior for the log burst rate; a vendor's template came with a very tight one. You simulate a year of bursts in all 140 districts from each candidate prior, before looking at any burst count, and compare what each prior predicts with what Shabib knows about pipes. You keep one prior and write a reason for it, and you write the prior of the age effect.
#
# **AI policy.** The tutor Mirsad may explain; it writes no model code.
#
# **Reference state.** `data/districts.csv` only: a prior predictive check uses no outcome data.
#
# **Tasks**
# - [ ] 1 · Three priors in bursts per km-year (5')
# - [ ] 2 · Simulate a year in every district from each prior (6')
# - [ ] 3 · Compare with what the engineers know (5')
# - [ ] 4 · Keep one prior, reject two, with a reason each (5')
# - [ ] 5 · The prior of the age effect (4')
# - [ ] ✱ Stretch · Prior predictive share of district-years without a burst

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-401.git"
    if not Path("/content/ds-401").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-401"], check=True)
    os.chdir("/content/ds-401/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("yanabi.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import yanabi as U
print("mode:", U.MODE)

dist_tbl = U.districts()
km = dist_tbl["network_km"].to_numpy()
PRIORS = {  # normal priors on the log rate (bursts per km-year): (centre, spread)
    "flat-looking N(0, 10)": (0.0, 10.0),
    "weakly informative N(log 0.1, 1)": (np.log(0.10), 1.0),
    "tight N(log 0.5, 0.05)": (np.log(0.50), 0.05),
}
N_DRAWS = U.pick(4000, 1000)

# %% [markdown]
# ## 1 · Three priors in bursts per km-year (5')
#
# A prior on a log scale is hard to read. Translate it: draw log rates, exponentiate, and report the 5th, 50th and 95th percentiles of the rate.
#
# ✏️ **Exercise** · Return a DataFrame indexed by prior name with columns `p05`, `p50`, `p95` of the rate, from `N_DRAWS` draws each (use `U.rng(1)`).

# %%
def rate_percentiles(priors, n):
    # BEGIN SOLUTION
    r = U.rng(1)
    rows = {}
    for name, (m, s) in priors.items():
        rate = np.exp(r.normal(m, s, n))
        rows[name] = dict(zip(["p05", "p50", "p95"], np.quantile(rate, [0.05, 0.5, 0.95])))
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    pct = rate_percentiles(PRIORS, N_DRAWS)
except NotImplementedError:
    pct = U.reference("d1_2", "pct")
U.keep("d1_2", "pct", pct)
with pd.option_context("display.float_format", "{:.3g}".format):
    print(pct.to_string())
U.check(pct.loc["flat-looking N(0, 10)", "p95"] > 1e6, "the flat-looking prior puts real weight on more than a million bursts per km-year")

# %% [markdown]
# ## 2 · Simulate a year in every district from each prior (6')
#
# The prior predictive distribution: draw a rate from the prior, then a year of bursts for each district with its own `network_km`. One draw of the rate is shared by all districts here (no district effects yet).
#
# ✏️ **Exercise** · Write `prior_predictive(m, s, n)` that returns an array of shape `(n, 140)` of simulated burst counts. Clip the Poisson mean at 1e9 so that NumPy does not overflow.

# %%
def prior_predictive(m, s, n):
    # BEGIN SOLUTION
    r = U.rng(2)
    rate = np.exp(r.normal(m, s, n))
    mean = np.minimum(rate[:, None] * km[None, :], 1e9)
    return r.poisson(mean)
    # END SOLUTION


# %%
sims = {}
for name, (m, s) in PRIORS.items():
    try:
        sims[name] = prior_predictive(m, s, N_DRAWS)
    except NotImplementedError:
        U.safety("prior_predictive")
        r = U.rng(2)
        sims[name] = r.poisson(np.minimum(np.exp(r.normal(m, s, N_DRAWS))[:, None] * km[None, :], 1e9))
print({k: v.shape for k, v in sims.items()})
U.check(all(v.shape == (N_DRAWS, 140) for v in sims.values()), "one simulated year for each of the 140 districts, per prior draw")

# %% [markdown]
# ## 3 · Compare with what the engineers know (5')
#
# Shabib's statements, written down before anyone looked at the data:
# - a district-year with more than 500 bursts has never happened anywhere in the region (statement A);
# - across the whole network, between 300 and 3,000 bursts a year is plausible (statement B).
#
# ✏️ **Exercise** · For each prior, return the share of simulated district-years above 500 (`share_over_500`) and the share of simulated network totals inside 300 to 3,000 (`share_total_plausible`).

# %%
def compare(sims):
    # BEGIN SOLUTION
    rows = {}
    for name, y in sims.items():
        tot = y.sum(axis=1)
        rows[name] = {"share_over_500": float((y > 500).mean()),
                      "share_total_plausible": float(((tot >= 300) & (tot <= 3000)).mean())}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    cmp_tbl = compare(sims)
except NotImplementedError:
    cmp_tbl = U.reference("d1_2", "cmp")
U.keep("d1_2", "cmp", cmp_tbl)
print(cmp_tbl.round(3).to_string())
U.check(cmp_tbl.loc["flat-looking N(0, 10)", "share_over_500"] > 0.2, "the flat-looking prior predicts impossible district-years often")
U.check(cmp_tbl.loc["weakly informative N(log 0.1, 1)", "share_total_plausible"] > 0.5, "the weakly informative prior keeps most network totals plausible")

# %% [markdown]
# **Look at** the tight prior's row. It never breaks statement A, yet it puts almost all its weight on totals around 0.5 × 7,000 km. Does it pass statement B? A prior can be wrong by being too sure as well as by being too vague.
#
# ## 4 · Keep one prior, reject two, with a reason each (5')
#
# ✏️ **Exercise** · Fill `decision` with `keep` or `reject` for each prior and a reason of one sentence that quotes a number from your table.

# %%
def decide():
    # BEGIN SOLUTION
    return {
        "flat-looking N(0, 10)": ("reject", f"{cmp_tbl.iloc[0, 0]:.0%} of simulated district-years exceed 500 bursts, which statement A rules out"),
        "weakly informative N(log 0.1, 1)": ("keep", f"{cmp_tbl.iloc[1, 1]:.0%} of network totals fall within 300 to 3,000 and almost no district-year exceeds 500"),
        "tight N(log 0.5, 0.05)": ("reject", f"only {cmp_tbl.iloc[2, 1]:.0%} of network totals fall within 300 to 3,000: it is sure of a rate nobody measured"),
    }
    # END SOLUTION


# %%
try:
    decision = decide()
except NotImplementedError:
    decision = U.reference("d1_2", "decision")
U.keep("d1_2", "decision", decision)
for k, (v, why) in decision.items():
    print(f"{v:7s} {k}: {why}")
U.check([v[0] for v in decision.values()] == ["reject", "keep", "reject"], "the weakly informative prior is kept, the two others rejected")

# %% [markdown]
# ## 5 · The prior of the age effect (4')
#
# The model will have log rate = α + β × (age − 30) / 10. Shabib: "ten more years of age might double the burst rate; I have never seen it multiply it by five."
#
# ✏️ **Exercise** · Choose a normal prior for β (centre and spread) whose 95th percentile of exp(β) is between 2 and 5 and whose median is 1 or above. Return `(centre, spread)`.

# %%
def age_prior():
    # BEGIN SOLUTION
    return (0.0, 0.5)
    # END SOLUTION


# %%
try:
    b_m, b_s = age_prior()
except NotImplementedError:
    b_m, b_s = U.reference("d1_2", "age_prior")
U.keep("d1_2", "age_prior", [b_m, b_s])
q = np.exp(np.quantile(U.rng(3).normal(b_m, b_s, 100000), [0.5, 0.95]))
print(f"β ~ Normal({b_m}, {b_s}): rate ratio per 10 years, median {q[0]:.2f}, 95th percentile {q[1]:.2f}")
U.check(q[0] >= 0.99 and 2 <= q[1] <= 5, "the age prior matches what the engineer said")

# %% [markdown]
# ## ✱ Stretch · Share of district-years without a burst
#
# Under the kept prior, what share of simulated district-years have no burst? Compare with the share in `data/bursts.csv` only after you have written your prediction down.

# %%
def stretch():
    # BEGIN SOLUTION
    y = sims["weakly informative N(log 0.1, 1)"]
    print(f"prior predictive share of zero district-years: {(y == 0).mean():.2f}; observed: {(U.bursts()['bursts_reported'] == 0).mean():.2f}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: each prior was judged by what it predicts, before any burst count was used.",
       "A reason is written for every prior that the model will carry; LO1's evidence asks for it.",
       "Not checked yet: whether the data and this prior together give a sound posterior. Lab 1.3 fits it.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is N(0, 10) on the log rate not "uninformative"?
#    <details><summary>Answer</summary>On the rate scale it puts most of its weight on absurdly small or absurdly large rates: it says a lot, most of it impossible.</details>
# 2. Why may a prior predictive check use statements written before the data were seen, but not the burst counts themselves?
#    <details><summary>Answer</summary>Tuning the prior to the observed counts uses the data twice and makes the posterior over-confident.</details>
# 3. The tight prior never predicts more than 500 bursts in a district-year. Why reject it anyway?
#    <details><summary>Answer</summary>It is far more certain than anyone's knowledge: it rules out the network totals the engineers call plausible.</details>
#
# ## Carry forward
#
# Lab 1.3 fits the model with the kept prior for α and the age prior of task 5.

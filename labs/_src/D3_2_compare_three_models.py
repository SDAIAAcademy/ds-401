# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 3.2 · Three models by leave-one-out
#
# Day 3 · Session 2 (Comparing models by predictive accuracy) · 25 min.
#
# Three candidate models of the same burst records: complete pooling (Lab 1.3), the Poisson hierarchy with the reporting term (Lab 2.4) and the negative binomial hierarchy (Lab 3.1). You estimate how well each predicts a district-year it has not seen, by Pareto-smoothed importance sampling, compare the differences with their standard errors, find the district-years that sway the comparison and choose a model for the decision.
#
# **AI policy.** The tutor Mirsad may explain leave-one-out output; the choice is yours.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`; the three models are keywords of `U.burst_model`, so earlier labs need not be finished.
#
# **Tasks**
# - [ ] 1 · Three candidates and their log likelihoods (5')
# - [ ] 2 · Approximate leave-one-out for each (5')
# - [ ] 3 · The difference and its standard error (5')
# - [ ] 4 · Pareto k: the influential district-years (6')
# - [ ] 5 · Choose a model for the decision (4')
# - [ ] ✱ Stretch · Leave out whole districts instead of district-years

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

import warnings
import numpy as np
import pandas as pd
import arviz as az
import yanabi as U
print("mode:", U.MODE)
warnings.filterwarnings("ignore", message="Estimated shape parameter")  # the lab reads Pareto k itself

p = U.panel()
D = U.design(p)
md = U.model_data(D)
CANDIDATES = {
    "complete Poisson": {"pooling": "complete"},
    "hierarchical Poisson": {"reporting": True},
    "hierarchical negative binomial": {"reporting": True, "family": "negbin"},
}

# %% [markdown]
# ## 1 · Three candidates and their log likelihoods (5')
#
# Leave-one-out needs, for every draw, the log likelihood of every district-year: `U.sample` stores it (`log_likelihood=True`).
#
# ✏️ **Exercise** · Fit the three candidates and return a dict `name -> idata`. Check that each has a `log_likelihood` group of shape (chains, draws, 700).

# %%
def fit_candidates():
    # BEGIN SOLUTION
    out = {}
    for name, kw in CANDIDATES.items():
        _, out[name] = U.sample(U.burst_model, y=D["y"], **kw, **md)
    return out
    # END SOLUTION


# %%
try:
    fits = fit_candidates()
except NotImplementedError:
    U.safety("fit_candidates")
    fits = {name: U.sample(U.burst_model, y=D["y"], **kw, **md)[1] for name, kw in CANDIDATES.items()}
for name, f in fits.items():
    ll = f["log_likelihood"]["y"]
    dg = U.diagnose(f, ["alpha", "beta"])
    print(f"{name:32s} log likelihood {tuple(ll.shape)}, R-hat {dg['max_rhat']:.3f}, divergences {dg['divergences']}")
U.check(all(f["log_likelihood"]["y"].shape[-1] == 700 for f in fits.values()), "a log likelihood for each of the 700 district-years")

# %% [markdown]
# ## 2 · Approximate leave-one-out for each (5')
#
# `az.loo(idata, pointwise=True)` returns the estimated expected log predictive density (elpd), its standard error, the effective number of parameters and a Pareto k for every district-year.
#
# ✏️ **Exercise** · Return a dict `name -> loo result`, and print elpd, its standard error and p_loo for each.

# %%
def loo_all():
    # BEGIN SOLUTION
    return {name: az.loo(f, pointwise=True) for name, f in fits.items()}
    # END SOLUTION


# %%
try:
    loos = loo_all()
except NotImplementedError:
    U.safety("loo_all")
    loos = {name: az.loo(f, pointwise=True) for name, f in fits.items()}
for name, lo in loos.items():
    print(f"{name:32s} elpd {lo.elpd:9.1f}  SE {lo.se:6.1f}  p_loo {lo.p:6.1f}")
U.check(loos["hierarchical negative binomial"].elpd > loos["complete Poisson"].elpd, "the negative binomial hierarchy predicts better than complete pooling")

# %% [markdown]
# ## 3 · The difference and its standard error (5')
#
# ✏️ **Exercise** · Return `az.compare(fits, round_to="none")` (unrounded numbers) and read the elpd difference of each model from the best with its standard error (`elpd_diff`, `dse`). In ArviZ 1.x a worse model's `elpd_diff` is negative. A difference smaller than about twice its standard error is not a clear win.

# %%
def compare():
    # BEGIN SOLUTION
    return az.compare(fits, round_to="none")
    # END SOLUTION


# %%
try:
    cmp_tbl = compare()
except NotImplementedError:
    U.safety("compare")
    cmp_tbl = az.compare(fits, round_to="none")
cols = [c for c in ["rank", "elpd", "p", "elpd_diff", "dse", "weight", "warning"] if c in cmp_tbl.columns]
print(cmp_tbl[cols].round(1).to_string())
U.keep("d3_2", "compare", cmp_tbl[[c for c in ["elpd", "elpd_diff", "dse"] if c in cmp_tbl.columns]].astype(float))
best = cmp_tbl.index[0]
second = cmp_tbl.index[1]
U.check(best == "hierarchical negative binomial" and abs(cmp_tbl.loc[second, "elpd_diff"]) > 2 * cmp_tbl.loc[second, "dse"],
        "the negative binomial hierarchy is best by more than twice the standard error of the difference")

# %% [markdown]
# ## 4 · Pareto k: the influential district-years (6')
#
# A Pareto k above about 0.7 says the importance-sampling estimate for that district-year is unreliable: leaving it out changes the posterior a lot. That is a property of the data point under the model, and it is worth looking at.
#
# ✏️ **Exercise** · For each model, return the district-years with Pareto k above 0.7 as a DataFrame (`model`, `district_id`, `year`, `bursts`, `k`).

# %%
def influential():
    # BEGIN SOLUTION
    rows = []
    for name, lo in loos.items():
        k = np.asarray(lo.pareto_k).reshape(-1)
        for i in np.where(k > 0.7)[0]:
            rows.append({"model": name, "district_id": p["district_id"].iloc[i], "year": int(p["year"].iloc[i]),
                         "bursts": int(D["y"][i]), "k": float(k[i])})
    return pd.DataFrame(rows, columns=["model", "district_id", "year", "bursts", "k"])
    # END SOLUTION


# %%
try:
    infl = influential()
except NotImplementedError:
    infl = U.reference("d3_2", "influential")
U.keep("d3_2", "influential", infl)
print(infl.round(2).to_string(index=False))
d017 = p[p["district_id"] == "D017"][["year", "bursts_reported"]].set_index("year")["bursts_reported"].to_dict()
print("D017's five years:", d017)
U.check(((infl["district_id"] == "D017") & (infl["year"] == 2023)).any(), "D017 in 2023 is flagged")
U.check((infl["model"] == "hierarchical negative binomial").sum() < (infl["model"] == "hierarchical Poisson").sum(),
        "the negative binomial has fewer influential points: it expects some unusual years")

# %% [markdown]
# **Look at** D017's years. Shabib confirms that a trunk main failed in 2023 and its repairs caused a cluster of bursts. The year is real, not an error: keep it, and prefer a model that is not overturned by one such year.
#
# ## 5 · Choose a model for the decision (4')
#
# ✏️ **Exercise** · Write the comparison table's last line for Nouran: the model chosen, the evidence from Lab 3.1's checks and from leave-one-out, and what the choice does not prove.

# %%
def choose():
    # BEGIN SOLUTION
    d = cmp_tbl.loc[second]
    return (f"Chosen: the hierarchical negative binomial with the reporting term. It passed the zeros and dispersion checks of Lab 3.1 "
            f"and predicts held-out district-years better than the next model by {abs(d['elpd_diff']):.0f} points of elpd (standard error {d['dse']:.0f}); "
            "the trunk-main year of D017 does not overturn it. This does not show it predicts a district it has never seen, nor that the audit prior on reporting is right.")
    # END SOLUTION


# %%
try:
    line = choose()
except NotImplementedError:
    line = U.reference("d3_2", "choice")
U.keep("d3_2", "choice", line)
print(line)
U.check("negative binomial" in line, "the choice names the model")

# %% [markdown]
# ## ✱ Stretch · Leave out whole districts
#
# Leaving out one district-year still lets the model see the other four years of that district. For a district the model has never seen, sum the log likelihood over each district's five years and run `az.loo` on that. Then count the districts with Pareto k above 0.7: what do they say about trusting this approximation, and what would you run instead?

# %%
def stretch():
    # BEGIN SOLUTION
    for name, f in fits.items():
        ll = np.asarray(f["log_likelihood"]["y"])
        grouped = np.stack([ll[..., D["d"] == j].sum(axis=-1) for j in range(D["J"])], axis=-1)
        g = az.from_dict({"posterior": {"alpha": np.asarray(f["posterior"]["alpha"])},
                          "log_likelihood": {"y": grouped}})
        lo = az.loo(g, pointwise=True)
        print(f"{name:32s} leave-one-district-out elpd {lo.elpd:8.1f}, districts with k > 0.7: {(np.asarray(lo.pareto_k) > 0.7).sum()}")
    print("Many districts above 0.7 under the hierarchies: removing a whole district moves its own effect too much for importance sampling.",
          "These numbers are not reliable; refit with K-fold cross-validation over districts (for example 10 folds) to compare.")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: predictive accuracy by approximate leave-one-out, with the standard error of each difference.",
       "Found: one real but unusual year (D017, 2023) is influential under the Poisson models.",
       "Carry forward: the chosen model is the one the supplied Day 4 posterior comes from.")

# %% [markdown]
# ## Checkpoint
#
# 1. Model A's elpd is 12 higher than B's, with a standard error of the difference of 9. Is A clearly better?
#    <details><summary>Answer</summary>No: the difference is less than twice its standard error; prefer the simpler or the better-checked model, or gather more evidence.</details>
# 2. What should you do with a district-year whose Pareto k is 1.2?
#    <details><summary>Answer</summary>Look at it: check the record, and refit without it (exact leave-one-out) if the comparison depends on it; do not delete it to improve the score.</details>
# 3. Why is leave-one-district-out a different question?
#    <details><summary>Answer</summary>It asks how well the model predicts a new district, which matters for a district joining the network, not for next year in a known one.</details>
#
# ## Carry forward
#
# Lab 3.3 checks that the code of a model is right, which no comparison can show.

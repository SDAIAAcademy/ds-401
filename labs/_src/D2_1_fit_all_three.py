# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 2.1 · Three poolings and the rural districts
#
# Day 2 · Session 1 (No, complete and partial pooling) · 25 min.
#
# Fifty-one districts recorded fewer than five bursts in five years. You write the partial-pooling model yourself, fit it beside no pooling and complete pooling, and plot how far each district's estimate moves. Then you answer the responsible-use question of the day: does pooling pull the rural districts towards the urban average, and by how much?
#
# **AI policy.** AI-off: this is the day's first model. Write it yourself.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, loaded fresh. Nothing from Day 1 is needed.
#
# **Tasks**
# - [ ] 1 · Three models from one function (5')
# - [ ] 2 · Fit no, complete and partial pooling (6')
# - [ ] 3 · The shrinkage plot (5')
# - [ ] 4 · Are rural districts pulled towards the urban average? (6')
# - [ ] 5 · A defence of the pooling in three sentences (3')
# - [ ] ✱ Stretch · Plot shrinkage against each district's number of recorded bursts

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
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import matplotlib
import matplotlib.pyplot as plt
import yanabi as U
print("mode:", U.MODE)
if not IN_COLAB:
    matplotlib.use("Agg")

D = U.design()
kind = np.array(D["kind"])
totals = np.bincount(D["d"], weights=D["y"], minlength=D["J"])
print(f"districts with fewer than 5 recorded bursts: {(totals < 5).sum()} of {D['J']}")

# %% [markdown]
# ## 1 · Three models from one function (5')
#
# No pooling and complete pooling come from `U.burst_model(pooling="none")` and `pooling="complete"`. Write the third yourself:
#
# - α ~ Normal(log 0.1, 1), the district spread τ ~ HalfNormal(0.5)
# - for each district j: η_j ~ Normal(α, τ)
# - bursts ~ Poisson(exp(log km + η_j)); keep `rate = exp(η)` as a deterministic site
#
# ✏️ **Exercise** · Complete `partial_model`. Name the sites `alpha`, `tau`, `eta`, `rate`, `y`.

# %%
def partial_model(d, log_km, J, y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("district", J):
        eta = numpyro.sample("eta", dist.Normal(alpha, tau))
    numpyro.deterministic("rate", jnp.exp(eta))
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
    # END SOLUTION


# %%
try:
    numpyro.handlers.seed(partial_model, 0)(D["d"], D["log_km"], D["J"])
    partial = partial_model
except NotImplementedError:
    U.safety("partial_model")
    partial = lambda d, log_km, J, y=None: U.burst_model(d, log_km, D["X"], D["sensors"], J, y, predictors=False, centered=True)
print("three models ready")

# %% [markdown]
# ## 2 · Fit no, complete and partial pooling (6')
#
# ✏️ **Exercise** · Fit the three models with `U.sample`, check each with `U.diagnose`, and return a dict `name -> idata` with the names `none`, `complete`, `partial`.

# %%
def fit_three():
    # BEGIN SOLUTION
    md = U.model_data(D)
    out = {}
    _, out["none"] = U.sample(U.burst_model, y=D["y"], pooling="none", **md)
    _, out["complete"] = U.sample(U.burst_model, y=D["y"], pooling="complete", predictors=False, **md)
    _, out["partial"] = U.sample(partial, y=D["y"], d=D["d"], log_km=D["log_km"], J=D["J"])
    return out
    # END SOLUTION


# %%
try:
    fits = fit_three()
except NotImplementedError:
    U.safety("fit_three")
    md = U.model_data(D)
    fits = {"none": U.sample(U.burst_model, y=D["y"], pooling="none", **md)[1],
            "complete": U.sample(U.burst_model, y=D["y"], pooling="complete", predictors=False, **md)[1],
            "partial": U.sample(partial, y=D["y"], d=D["d"], log_km=D["log_km"], J=D["J"])[1]}
for k, f in fits.items():
    dg = U.diagnose(f, ["alpha"] + (["tau"] if "tau" in f["posterior"] else []))
    print(f"{k:9s} R-hat {dg['max_rhat']:.3f}, bulk ESS {dg['min_ess_bulk']:.0f}, divergences {dg['divergences']}")
    U.check(U.trustworthy(dg), f"{k}: the fit can be read")
est = pd.DataFrame({k: np.median(U.draws(f, "rate"), axis=0) for k, f in fits.items()}, index=D["ids"])
est["kind"], est["bursts"], est["km"] = kind, totals, D["km"]
print(f"spread between districts, τ: {U.describe(U.draws(fits['partial'], 'tau'))}")

# %% [markdown]
# ## 3 · The shrinkage plot (5')
#
# Shrinkage of district j: how far its partial-pooling estimate moved from its own (no-pooling) estimate towards the complete-pooling one, on the log scale. 0 means it kept its own estimate, 1 means it took the common rate.
#
# ✏️ **Exercise** · Add a column `shrinkage` to `est`, then save a scatter of shrinkage against `km` (log x axis) to `artifacts/shrinkage.png`, coloured by kind.

# %%
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def shrinkage(est):
    # BEGIN SOLUTION
    est = est.copy()
    ln, lc, lp = np.log(est["none"]), np.log(est["complete"]), np.log(est["partial"])
    est["shrinkage"] = (ln - lp) / (ln - lc)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in [("urban", "tab:blue"), ("suburban", "tab:green"), ("rural", "tab:red")]:
        m = est["kind"] == k
        ax.scatter(est.loc[m, "km"], est.loc[m, "shrinkage"].clip(-0.2, 1.2), s=14, c=c, label=k)
    ax.set_xscale("log"); ax.set_xlabel("network km"); ax.set_ylabel("shrinkage towards the common rate"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "shrinkage.png", dpi=110); plt.close(fig)
    return est
    # END SOLUTION


# %%
try:
    est = shrinkage(est)
except NotImplementedError:
    U.safety("shrinkage")
    ln, lc, lp = np.log(est["none"]), np.log(est["complete"]), np.log(est["partial"])
    est["shrinkage"] = (ln - lp) / (ln - lc)
by_kind = est.groupby("kind")["shrinkage"].median()
print("median shrinkage by kind:", by_kind.round(2).to_dict())
U.check(by_kind["rural"] > by_kind["urban"] + 0.3, "rural districts are shrunk much more than urban ones")

# %% [markdown]
# ## 4 · Are rural districts pulled towards the urban average? (6')
#
# Most of the network's kilometres, and so most of its bursts, are urban. The common rate of the complete-pooling model is therefore close to the urban rate.
#
# ✏️ **Exercise** · Return a DataFrame by kind with `true_gap`, the mean of log(true rate / common rate) using `U.truth()["true_rate_per_km_year"]` (real data has no such column), `estimated_gap`, the same for the partial-pooling estimates, and `kept`, the share of the true gap the estimates keep.

# %%
def rural_pull(est):
    # BEGIN SOLUTION
    tr = pd.Series(U.truth()["true_rate_per_km_year"]).reindex(est.index)
    common = float(est["complete"].median())
    e = est.assign(true_gap=np.log(tr / common), estimated_gap=np.log(est["partial"] / common))
    g = e.groupby("kind")[["true_gap", "estimated_gap"]].mean()
    g["kept"] = g["estimated_gap"] / g["true_gap"]
    return g
    # END SOLUTION


# %%
try:
    pull = rural_pull(est)
except NotImplementedError:
    pull = U.reference("d2_1", "pull")
U.keep("d2_1", "pull", pull)
print(pull.round(3).to_string())
U.check(pull.loc["rural", "true_gap"] > 0.4 and pull.loc["rural", "kept"] < 0.3, "rural districts truly burst far more often than the common rate, and the estimates keep little of that gap")

# %% [markdown]
# **Look at** the rural row. On real data you would not have `true_gap`, but you would see from task 3 that rural estimates are mostly borrowed. Here the truth shows the borrowing hides a real difference. Who would be harmed if the re-piping plan used these rural estimates?
#
# ## 5 · A defence of the pooling in three sentences (3')
#
# ✏️ **Exercise** · Write three sentences for Nouran: why partial pooling and not no pooling; what it does to the rural districts; what you will change in Lab 2.2.

# %%
def defence():
    # BEGIN SOLUTION
    return [
        f"No pooling gives the 51 districts with fewer than five bursts estimates driven by one or two counts; partial pooling borrows strength from the other districts in proportion to how little each one knows (τ = {np.median(U.draws(fits['partial'], 'tau')):.2f}).",
        f"Because the districts are shrunk towards one rate dominated by the urban networks, the rural estimates keep only {pull.loc['rural', 'kept']:.0%} of their true gap above that rate, which understates their risk.",
        "In Lab 2.2 the districts are shrunk towards a rate predicted from their pipe material and age instead of towards one urban-dominated average.",
    ]
    # END SOLUTION


# %%
try:
    lines = defence()
except NotImplementedError:
    lines = U.reference("d2_1", "defence")
U.keep("d2_1", "defence", lines)
for s in lines:
    print("-", s)
U.check(len(lines) == 3, "three sentences")

# %% [markdown]
# ## ✱ Stretch · Shrinkage against the number of bursts

# %%
def stretch():
    # BEGIN SOLUTION
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(est["bursts"] + 0.5, est["shrinkage"].clip(-0.2, 1.2), s=12)
    ax.set_xscale("log"); ax.set_xlabel("recorded bursts in five years (+0.5)"); ax.set_ylabel("shrinkage")
    fig.tight_layout(); fig.savefig(OUT / "shrinkage_by_bursts.png", dpi=110); plt.close(fig)
    print("correlation of shrinkage with log bursts:", round(float(np.corrcoef(np.log(est["bursts"] + 0.5), est["shrinkage"].clip(-0.2, 1.2))[0, 1]), 2))
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: three fits, each diagnosed before use.",
       "Found: partial pooling without predictors pulls rural districts towards the urban-dominated common rate.",
       "Carry forward: the partial-pooling model; Lab 2.2 adds material and age to its mean.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is a district with 2 bursts shrunk more than one with 200?
#    <details><summary>Answer</summary>Its own likelihood is flat, so the population distribution, set by τ and α, dominates its posterior.</details>
# 2. What does τ measure?
#    <details><summary>Answer</summary>How much districts' log rates differ beyond what the model's mean explains; a small τ means strong pooling.</details>
# 3. Is pulling rural districts down "conservative"?
#    <details><summary>Answer</summary>No: it understates their burst risk, so a plan built on it would under-invest where the pipes are oldest.</details>
#
# ## Carry forward
#
# Lab 2.2 starts from the same files and adds district-level predictors.

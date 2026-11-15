# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 1.3 · A Poisson model with an offset
#
# Day 1 · Session 3 (Fitting with Hamiltonian Monte Carlo) · 25 min.
#
# Dr. Nouran Al-Ateeq, director of asset management, wants to know how much material and pipe age change the burst rate. You write the model in NumPyro with network length as the exposure, run four chains of the No-U-Turn sampler, and read the posterior as rate ratios with intervals. Because the data are synthetic, you can also check the answer against the generating values.
#
# **AI policy.** The tutor Mirsad may explain NumPyro errors; it writes no model code.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, loaded fresh; the priors of Lab 1.2 are written in below.
#
# **Tasks**
# - [ ] 1 · Write the model (8')
# - [ ] 2 · Run four chains (4')
# - [ ] 3 · Rate ratios with intervals (6')
# - [ ] 4 · Against the generating values (3')
# - [ ] 5 · Expected bursts next year for two districts (4')
# - [ ] ✱ Stretch · Remove the offset and see what the model then says

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
import arviz as az
import yanabi as U
print("mode:", U.MODE)

D = U.design()
print(f"{len(D['y'])} district-years, {D['J']} districts; predictors: {U.X_NAMES}")

# %% [markdown]
# ## 1 · Write the model (8')
#
# log E[bursts of district j in a year] = log km_j + α + X_j β
#
# - α ~ Normal(log 0.1, 1): the kept prior of Lab 1.2
# - β ~ Normal(0, 0.5) for each of the four predictors (age per 10 years, ductile iron, cast iron, asbestos cement; PVC is the base)
# - the counts are Poisson. `log_km` is an **offset**: it enters with a coefficient fixed at 1.
#
# ✏️ **Exercise** · Complete `pooled_model`. Name the sample sites `alpha`, `beta` and `y`.

# %%
def pooled_model(d, log_km, X, y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
    eta = alpha + X @ beta
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
    # END SOLUTION


# %%
args = {"d": D["d"], "log_km": D["log_km"], "X": D["X"]}
try:
    numpyro.handlers.seed(pooled_model, 0)(**args)  # a dry run: raises if the exercise is not done yet
    model = pooled_model
except NotImplementedError:
    U.safety("pooled_model")
    model = lambda d, log_km, X, y=None: U.burst_model(d, log_km, X, D["sensors"], D["J"], y, pooling="complete")
print("model ready")

# %% [markdown]
# ## 2 · Run four chains (4')
#
# `U.sample` runs NUTS with four chains, 1,000 warm-up and 1,000 kept draws each (400 and 400 in quick mode), and returns the NumPyro run and an ArviZ object. The first run compiles the model: about half a minute.
#
# ✏️ **Exercise** · Call `U.sample(model, y=D["y"], **args)` and keep both results. Then print `U.diagnose(idata)`.

# %%
def fit():
    # BEGIN SOLUTION
    return U.sample(model, y=D["y"], **args)
    # END SOLUTION


# %%
try:
    mcmc, idata = fit()
except NotImplementedError:
    U.safety("fit")
    mcmc, idata = U.sample(model, y=D["y"], **args)
diag = U.diagnose(idata, ["alpha", "beta"])
print({k: round(v, 3) for k, v in diag.items()})
U.check(U.trustworthy(diag), "R-hat below 1.01, enough effective draws, no divergence: the fit can be read")

# %% [markdown]
# ## 3 · Rate ratios with intervals (6')
#
# ✏️ **Exercise** · Return a DataFrame indexed by `U.X_NAMES` with the posterior median of exp(β) and its 90% interval (`ratio`, `low`, `high`), and a row `"base rate, PVC age 30"` for exp(α).

# %%
def rate_ratios(idata):
    # BEGIN SOLUTION
    b = np.exp(U.draws(idata, "beta"))
    a = np.exp(U.draws(idata, "alpha"))
    rows = {n: np.quantile(b[:, i], [0.5, 0.05, 0.95]) for i, n in enumerate(U.X_NAMES)}
    rows["base rate, PVC age 30"] = np.quantile(a, [0.5, 0.05, 0.95])
    return pd.DataFrame(rows, index=["ratio", "low", "high"]).T
    # END SOLUTION


# %%
try:
    rr = rate_ratios(idata)
except NotImplementedError:
    rr = U.reference("d1_3", "rr")
U.keep("d1_3", "rr", rr)
print(rr.round(3).to_string())
U.check(rr.loc["asbestos cement", "low"] > 1.3, "asbestos-cement networks burst clearly more often than PVC networks of the same age")

# %% [markdown]
# **Look at** the intervals. They are narrow: 700 district-years and one rate for all districts of the same material and age. Day 2 asks whether one rate per material and age is believable.
#
# ## 4 · Against the generating values (3')
#
# ✏️ **Exercise** · Return the share of the five generating values (α and the four β, from `U.truth()`) that fall inside the 90% posterior intervals.

# %%
def coverage(idata):
    # BEGIN SOLUTION
    t = U.truth()
    true = [t["alpha"], t["beta_age_per_10y"], *[t["beta_material"][m] for m in U.MATERIALS[1:]]]
    est = np.column_stack([U.draws(idata, "alpha"), U.draws(idata, "beta")])
    lo, hi = np.quantile(est, [0.05, 0.95], axis=0)
    return float(np.mean((lo <= np.array(true)) & (np.array(true) <= hi)))
    # END SOLUTION


# %%
try:
    cov = coverage(idata)
except NotImplementedError:
    cov = U.reference("d1_3", "cov")
U.keep("d1_3", "cov", cov)
print(f"generating values inside the 90% intervals: {cov:.0%}")
U.check(cov >= 0.4, "the pooled model recovers some of the generating values")

# %% [markdown]
# Not all five need be inside. The pooled model leaves out the district effects and the under-reporting that generated the data, so its intervals are too narrow. Sessions 2.1 and 2.4 add both.
#
# ## 5 · Expected bursts next year for two districts (4')
#
# ✏️ **Exercise** · For the districts `D017` and the first rural district with asbestos cement, return a dict `{district_id: (median, low, high)}` of next year's expected bursts (rate × km) with a 90% interval.

# %%
rural_ac = next(i for i, (k, x) in enumerate(zip(D["kind"], D["X"])) if k == "rural" and x[3] == 1)
pair = [D["ids"].index("D017"), rural_ac]


def expected_next_year(idata):
    # BEGIN SOLUTION
    a = U.draws(idata, "alpha")
    b = U.draws(idata, "beta")
    out = {}
    for j in pair:
        mu = np.exp(a + b @ D["X"][j]) * D["km"][j]
        out[D["ids"][j]] = tuple(float(v) for v in np.quantile(mu, [0.5, 0.05, 0.95]))
    return out
    # END SOLUTION


# %%
try:
    nxt = expected_next_year(idata)
except NotImplementedError:
    nxt = U.reference("d1_3", "next")
U.keep("d1_3", "next", nxt)
for k, (m, lo, hi) in nxt.items():
    print(f"{k}: {m:.1f} expected bursts next year (90% interval {lo:.1f} to {hi:.1f})")
U.check(len(nxt) == 2, "two districts predicted")

# %% [markdown]
# ## ✱ Stretch · Without the offset
#
# Fit the same model without `log_km` (put zeros in its place) and compare the asbestos-cement ratio. What does the model now confuse with material?

# %%
def stretch():
    # BEGIN SOLUTION
    _, i2 = U.sample(model, y=D["y"], d=D["d"], log_km=np.zeros_like(D["log_km"]), X=D["X"])
    r2 = rate_ratios(i2)
    print(f"asbestos cement ratio without the offset: {r2.loc['asbestos cement', 'ratio']:.2f}, with it: {rr.loc['asbestos cement', 'ratio']:.2f}")
    print("rural networks are short and often asbestos cement: without exposure, material is read through length")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the sampler's diagnostics before any reading of the posterior.",
       "Checked: the estimates against the generating values; the pooled model is too sure.",
       "Carry forward: the model function and its priors; Lab 1.4 reads diagnostics of four fits like this one.")

# %% [markdown]
# ## Checkpoint
#
# 1. The coefficient of log km is not estimated. Why?
#    <details><summary>Answer</summary>It is an offset fixed at 1: twice the network length gives twice the expected bursts at the same rate.</details>
# 2. What does a rate ratio of 2.0 for asbestos cement mean here?
#    <details><summary>Answer</summary>Asbestos-cement networks have about twice the bursts per km-year of PVC networks of the same mean age, under this model.</details>
# 3. Why are these intervals too narrow?
#    <details><summary>Answer</summary>The model has no district effects and no extra year-to-year variation, so it treats 700 district-years as more information than they are.</details>
#
# ## Carry forward
#
# The model function of task 1 is the base of Day 2's hierarchical models (`U.burst_model` holds the reference version).

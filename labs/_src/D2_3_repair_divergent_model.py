# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 2.3 · Repair a divergent model
#
# Day 2 · Session 3 (The funnel and non-centred models) · 25 min.
#
# Fit B of Lab 1.4 came from a contractor who fitted the hierarchical model to the small districts only, in centred form. You reproduce its divergences, find where in the posterior they sit, rewrite the model in non-centred form and show that the repaired fit can be trusted.
#
# **AI policy.** The tutor Mirsad may explain the funnel; it writes no model code.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, restricted to the districts under 4 km. Nothing from Lab 2.2 is needed.
#
# **Tasks**
# - [ ] 1 · Reproduce the divergences (5')
# - [ ] 2 · Where they sit: the spread against one district (5')
# - [ ] 3 · Write the non-centred model (7')
# - [ ] 4 · Refit and compare the diagnostics (5')
# - [ ] 5 · The same answer, now trustworthy? (3')
# - [ ] ✱ Stretch · Raise the target acceptance rate of the centred model instead, and compare

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

p = U.panel()
small = p[p["network_km"] < 4]
S = U.design(small)
args = {"d": S["d"], "log_km": S["log_km"], "X": S["X"], "J": S["J"]}
print(f"{S['J']} small districts, {S['y'].sum()} reported bursts in five years")


def centred_model(d, log_km, X, J, y=None):
    """The contractor's model: η_j ~ Normal(α + X_j β, τ), written directly."""
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))
    with numpyro.plate("district", J):
        eta = numpyro.sample("eta", dist.Normal(alpha + X @ beta, tau))
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)


# %% [markdown]
# ## 1 · Reproduce the divergences (5')
#
# ✏️ **Exercise** · Fit `centred_model` with `U.sample` (`warmup=300`, as the contractor did) and return the idata. Print its diagnostics.

# %%
def fit_centred():
    # BEGIN SOLUTION
    _, idata = U.sample(centred_model, warmup=300, y=S["y"], **args)
    return idata
    # END SOLUTION


# %%
try:
    cen = fit_centred()
except NotImplementedError:
    U.safety("fit_centred")
    _, cen = U.sample(centred_model, warmup=300, y=S["y"], **args)
dc = U.diagnose(cen, ["alpha", "beta", "tau"])
print({k: round(v, 3) for k, v in dc.items()})
U.check(dc["divergences"] > 10, "the centred model diverges on the small districts")

# %% [markdown]
# ## 2 · Where they sit: the spread against one district (5')
#
# In the centred model, η_j must lie within about τ of its mean. When τ is small the region allowed for η_j becomes a narrow neck: a funnel. A step size tuned for the wide mouth is far too large for the neck.
#
# ✏️ **Exercise** · Plot log τ against η of the first small district, marking the divergent draws, save it to `artifacts/funnel.png`, and return the share of divergent draws that fall below the median of τ.

# %%
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def funnel():
    # BEGIN SOLUTION
    tau = U.draws(cen, "tau")
    eta0 = U.draws(cen, "eta")[:, 0]
    div = np.asarray(cen["sample_stats"]["diverging"]).reshape(-1).astype(bool)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(eta0[~div], np.log(tau[~div]), s=4, alpha=0.4, label="draws")
    ax.scatter(eta0[div], np.log(tau[div]), s=10, c="red", label="divergent")
    ax.set_xlabel("η of one small district"); ax.set_ylabel("log τ"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "funnel.png", dpi=110); plt.close(fig)
    return float(np.mean(tau[div] < np.median(tau)))
    # END SOLUTION


# %%
try:
    low_share = funnel()
except NotImplementedError:
    low_share = U.reference("d2_3", "low_share")
U.keep("d2_3", "low_share", low_share)
print(f"share of divergent draws with τ below its median: {low_share:.0%}")
U.check(low_share > 0.6, "the divergences sit in the neck of the funnel, where τ is small")

# %% [markdown]
# ## 3 · Write the non-centred model (7')
#
# Sample a standard normal z_j for each district and build η_j = α + X_j β + τ z_j. The posterior of z has no funnel: its shape no longer depends on τ.
#
# ✏️ **Exercise** · Complete `noncentred_model`, with the same priors as the centred one. Keep `eta` as a deterministic site so both models report the same quantity.

# %%
def noncentred_model(d, log_km, X, J, y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1.0))
    with numpyro.plate("district", J):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    eta = numpyro.deterministic("eta", alpha + X @ beta + tau * z)
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
    # END SOLUTION


# %%
try:
    numpyro.handlers.seed(noncentred_model, 0)(**args)
    nc_model = noncentred_model
except NotImplementedError:
    U.safety("noncentred_model")
    nc_model = lambda d, log_km, X, J, y=None: U.burst_model(d, log_km, X, np.ones(J, bool), J, y, tau_scale=1.0)
print("non-centred model ready")

# %% [markdown]
# ## 4 · Refit and compare the diagnostics (5')
#
# ✏️ **Exercise** · Fit the non-centred model with the same warm-up and return a DataFrame with one row per model (`centred`, `non-centred`) and the columns of `U.diagnose`.

# %%
def compare():
    # BEGIN SOLUTION
    global nc
    _, nc = U.sample(nc_model, warmup=300, y=S["y"], **args)
    return pd.DataFrame({"centred": dc, "non-centred": U.diagnose(nc, ["alpha", "beta", "tau"])}).T
    # END SOLUTION


# %%
try:
    tbl = compare()
except NotImplementedError:
    U.safety("compare")
    _, nc = U.sample(nc_model, warmup=300, y=S["y"], **args)
    tbl = pd.DataFrame({"centred": dc, "non-centred": U.diagnose(nc, ["alpha", "beta", "tau"])}).T
print(tbl.round(3).to_string())
U.check(tbl.loc["non-centred", "divergences"] <= 2 and tbl.loc["non-centred", "min_ess_bulk"] > 3 * tbl.loc["centred", "min_ess_bulk"],
        "the non-centred model has (almost) no divergence and many more effective draws")

# %% [markdown]
# ## 5 · The same answer, now trustworthy? (3')
#
# ✏️ **Exercise** · Return the 90% intervals of τ from both fits as a dict `{model: (low, high)}`. Which reaches small values of τ, and why does that matter for the small districts' rates?

# %%
def tau_intervals():
    # BEGIN SOLUTION
    return {k: tuple(float(v) for v in np.quantile(U.draws(f, "tau"), [0.05, 0.95])) for k, f in [("centred", cen), ("non-centred", nc)]}
    # END SOLUTION


# %%
try:
    ti = tau_intervals()
except NotImplementedError:
    ti = U.reference("d2_3", "tau_intervals")
U.keep("d2_3", "tau_intervals", ti)
for k, (lo, hi) in ti.items():
    print(f"{k:12s} τ 90% interval {lo:.3f} to {hi:.3f}")
U.check(ti["non-centred"][0] < ti["centred"][0], "the non-centred fit reaches the small values of τ the centred one missed")

# %% [markdown]
# ## ✱ Stretch · Only raise target_accept
#
# Refit the centred model with `target_accept=0.99`. Count the divergences and the time. Fewer divergences is not the same as a correct posterior: compare its τ interval with the non-centred one.

# %%
def stretch():
    # BEGIN SOLUTION
    import time
    t0 = time.time()
    _, c99 = U.sample(centred_model, warmup=300, target_accept=0.99, y=S["y"], **args)
    d99 = U.diagnose(c99, ["alpha", "beta", "tau"])
    print(f"target_accept 0.99: divergences {d99['divergences']}, R-hat {d99['max_rhat']:.3f}, {time.time() - t0:.0f} s; τ 90% interval {np.quantile(U.draws(c99, 'tau'), [0.05, 0.95]).round(3)}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the divergences of fit B come from the funnel, not from the data.",
       "Repaired: the same model in non-centred form samples the whole posterior.",
       "Not every centred model fails: with all 140 districts and their strong data, the centred form samples well. Choose the form by the data, and check.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the funnel appear for the small districts and not for the large ones?
#    <details><summary>Answer</summary>With little data, a district's η is held only by τ, so its range shrinks with τ; with a lot of data the likelihood holds η wherever τ is.</details>
# 2. Is the non-centred model a different model?
#    <details><summary>Answer</summary>No: it implies the same joint distribution; only the parameters the sampler moves in change.</details>
# 3. What does a centred fit with divergences tend to get wrong?
#    <details><summary>Answer</summary>It misses small values of τ, so it overstates the spread between districts and under-pools the small ones.</details>
#
# ## Carry forward
#
# Every hierarchical model from here on is written in non-centred form, as `U.burst_model` does.

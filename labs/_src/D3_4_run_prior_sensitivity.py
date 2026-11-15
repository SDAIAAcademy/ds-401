# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 3.4 · Prior sensitivity
#
# Day 3 · Session 4 (How much the priors matter) · 25 min.
#
# Before Nouran signs a plan, she asks: "would another reasonable analyst, with other priors, have told me something different?" You answer in two ways: a power-scaling diagnostic that weakens and strengthens the priors of the chosen model without refitting, and two refits with other priors on the spread between districts that someone could defend. Then you write what changes for the decision and what does not.
#
# **AI policy.** The tutor Mirsad may explain power-scaling; the summary is yours.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`; the chosen model is `U.burst_model(family="negbin", reporting=True)`, so earlier labs need not be finished.
#
# **Tasks**
# - [ ] 1 · The base fit and the power-scaling diagnostic (6')
# - [ ] 2 · Two alternative priors on the spread (7')
# - [ ] 3 · What changes: spread, rural rates, top 15 (7')
# - [ ] 4 · A summary for the director (5')
# - [ ] ✱ Stretch · Do the same for the prior on the negative binomial shape

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
from scipy import stats
import yanabi as U
print("mode:", U.MODE)

D = U.design()
md = U.model_data(D)
kind = np.array(D["kind"])
rural_ns = (kind == "rural") & ~D["sensors"]


def fit(tau_scale=0.5):
    return U.sample(lambda **kw: U.burst_model(family="negbin", reporting=True, tau_scale=tau_scale, **kw), y=D["y"], **md)[1]


def quantities(f):
    """The three quantities the decision leans on."""
    rate = U.draws(f, "rate")
    med = np.median(rate, axis=0)
    return {"tau": float(np.median(U.draws(f, "tau"))),
            "rural_no_sensor_rate": float(np.median(med[rural_ns])),
            "top15": set(np.array(D["ids"])[np.argsort(-med)[:15]].tolist())}


base = fit()
print("base fit:", {k: round(v, 3) for k, v in U.diagnose(base, ["alpha", "beta", "tau", "phi"]).items()})
qb = quantities(base)

# %% [markdown]
# ## 1 · The base fit and the power-scaling diagnostic (6')
#
# Power-scaling raises the prior to a power a: a < 1 weakens it, a > 1 strengthens it. Without refitting, each draw is reweighted by prior(θ)^(a − 1). If a quantity moves a lot when the prior is scaled by 0.8 or 1.25, the prior is doing much of the work.
#
# ✏️ **Exercise** · Compute the log prior of each draw for α, β, τ and φ (the priors a reviewer would question), then return a DataFrame with, for a in (0.8, 1.25), the weighted mean of τ and of the median rural rate minus their unweighted means, divided by the posterior standard deviation.

# %%
def log_prior(f):
    # BEGIN SOLUTION
    a, b, t, ph = U.draws(f, "alpha"), U.draws(f, "beta"), U.draws(f, "tau"), U.draws(f, "phi")
    return (stats.norm.logpdf(a, np.log(0.10), 1.0) + stats.norm.logpdf(b, 0, 0.5).sum(axis=1)
            + stats.halfnorm.logpdf(t, scale=0.5) + stats.gamma.logpdf(ph, 2.0, scale=1 / 0.2))
    # END SOLUTION


def power_scaling(f):
    # BEGIN SOLUTION
    lp = log_prior(f)
    q = {"tau": U.draws(f, "tau"), "rural rate": np.median(U.draws(f, "rate")[:, rural_ns], axis=1)}
    rows = {}
    for a in (0.8, 1.25):
        lw = (a - 1) * lp
        w = np.exp(lw - lw.max()); w /= w.sum()
        rows[f"prior^{a}"] = {k: float((np.sum(w * v) - v.mean()) / v.std()) for k, v in q.items()}
    return pd.DataFrame(rows)
    # END SOLUTION


# %%
try:
    ps = power_scaling(base)
except NotImplementedError:
    ps = U.reference("d3_4", "power_scaling")
U.keep("d3_4", "power_scaling", ps)
print("shift of the posterior mean, in posterior standard deviations:")
print(ps.round(3).to_string())
U.check(ps.abs().max().max() < 0.5, "scaling the priors by 0.8 or 1.25 moves no quantity by half a posterior sd: the data dominate these priors")

# %% [markdown]
# ## 2 · Two alternative priors on the spread (7')
#
# A tight prior, τ ~ HalfNormal(0.2) ("districts of the same material and age are alike"), and a wide one, τ ~ HalfNormal(2) ("districts can differ by a factor of 50"). Both could be defended by someone.
#
# ✏️ **Exercise** · Refit the model with `fit(tau_scale=0.2)` and `fit(tau_scale=2.0)`; return a dict `scale -> idata` including the base (0.5).

# %%
def refits():
    # BEGIN SOLUTION
    return {0.2: fit(0.2), 0.5: base, 2.0: fit(2.0)}
    # END SOLUTION


# %%
try:
    fits = refits()
except NotImplementedError:
    U.safety("refits")
    fits = {0.2: fit(0.2), 0.5: base, 2.0: fit(2.0)}
for s, f in fits.items():
    dg = U.diagnose(f, ["alpha", "beta", "tau", "phi"])
    print(f"τ ~ HalfNormal({s}): R-hat {dg['max_rhat']:.3f}, divergences {dg['divergences']}")
U.check(len(fits) == 3, "three fits")

# %% [markdown]
# ## 3 · What changes: spread, rural rates, top 15 (7')
#
# ✏️ **Exercise** · Return a DataFrame, one row per prior, with the posterior median of τ, the median rate of the rural districts without sensors, and how many of the base fit's 15 highest-rate districts are also in this fit's top 15.

# %%
def what_changes():
    # BEGIN SOLUTION
    rows = {}
    for s, f in fits.items():
        q = quantities(f)
        rows[f"HalfNormal({s})"] = {"tau": q["tau"], "rural_no_sensor_rate": q["rural_no_sensor_rate"], "top15_shared": len(q["top15"] & qb["top15"])}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    ch = what_changes()
except NotImplementedError:
    ch = U.reference("d3_4", "changes")
U.keep("d3_4", "changes", ch)
print(ch.round(3).to_string())
U.check(ch["top15_shared"].min() >= 12, "at least 12 of the 15 highest-rate districts stay the same under every prior")

# %% [markdown]
# ## 4 · A summary for the director (5')
#
# ✏️ **Exercise** · Write three sentences for Nouran: what changes, what does not, and what that means for the plan.

# %%
def summary():
    # BEGIN SOLUTION
    lo, hi = ch["rural_no_sensor_rate"].min(), ch["rural_no_sensor_rate"].max()
    return [f"Across three defensible priors on how much districts differ, the spread τ ranges from {ch['tau'].min():.2f} to {ch['tau'].max():.2f}, and the typical rate of rural districts without sensors from {lo:.3f} to {hi:.3f} bursts per km-year.",
            f"{int(ch['top15_shared'].min())} of the 15 highest-rate districts are the same under every prior, and weakening or strengthening the priors moves no key quantity by more than {ps.abs().max().max():.2f} posterior standard deviations.",
            "The plan's choice of districts does not hinge on the prior on τ; the audit prior on reporting (Lab 2.4) matters more and is the one to improve."]
    # END SOLUTION


# %%
try:
    lines = summary()
except NotImplementedError:
    lines = U.reference("d3_4", "summary")
U.keep("d3_4", "summary", lines)
for s in lines:
    print("-", s)
U.check(len(lines) == 3, "three sentences")

# %% [markdown]
# ## ✱ Stretch · The shape prior
#
# Refit with φ ~ Gamma(2, 1) (mean 2: very irregular years) and compare φ and the rural rate.

# %%
def stretch():
    # BEGIN SOLUTION
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    def m(d, log_km, X, sensors, J, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
        beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))
        with numpyro.plate("district", J):
            z = numpyro.sample("z", dist.Normal(0.0, 1.0))
            pr = numpyro.sample("p_report", dist.Beta(14.0, 6.0))
        eta = alpha + X @ beta + tau * z
        numpyro.deterministic("rate", jnp.exp(eta))
        phi = numpyro.sample("phi", dist.Gamma(2.0, 1.0))
        numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_km + eta[d] + jnp.log(jnp.where(sensors, 1.0, pr))[d]), phi), obs=y)
    _, f = U.sample(m, y=D["y"], **md)
    print("φ with Gamma(2, 1):", U.describe(U.draws(f, "phi")), "| base:", U.describe(U.draws(base, "phi")))
    print("rural no-sensor rate:", round(quantities(f)["rural_no_sensor_rate"], 3), "| base:", round(qb["rural_no_sensor_rate"], 3))
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the decision quantities under weaker, stronger and alternative priors.",
       "Found: the ranking of districts is robust to the prior on τ.",
       "Carry forward: the base model's posterior is the one supplied to Day 4 (data/posterior/rates.npz).")

# %% [markdown]
# ## Checkpoint
#
# 1. Power-scaling moved τ by 0.05 posterior sd. What does that say?
#    <details><summary>Answer</summary>The data, not the prior, determine τ here: weakening or strengthening the prior hardly moves it.</details>
# 2. Why report sensitivity on the top 15 districts rather than on β?
#    <details><summary>Answer</summary>The decision uses the ranking of districts; a coefficient can move while the plan stays the same, or the reverse.</details>
# 3. Which prior in this course can power-scaling not judge well, and why?
#    <details><summary>Answer</summary>The audit prior on reporting: the data carry almost no information about it, so the posterior simply follows it; only better data, such as the survey, can test it.</details>
#
# ## Carry forward
#
# Day 4 starts from the supplied posterior of this model.

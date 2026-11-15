# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 3.3 · Simulation-based calibration
#
# Day 3 · Session 3 (Is the code right) · 25 min.
#
# A colleague sent a small burst model for the 20 districts of the northern zone and a simulator that "generates data exactly as the model assumes". The model fits, the diagnostics are clean and one parameter-recovery run looks fine. Somewhere in the model code is one wrong line. You find it with a short simulation-based calibration, fix it, and show that the corrected model passes.
#
# **AI policy.** The tutor Mirsad may explain ranks and histograms; it writes no model code and does not point at the line.
#
# **Reference state.** The model, the simulator and the 20 districts below. Nothing from earlier labs is needed.
#
# **Tasks**
# - [ ] 1 · Parameter recovery on one dataset (5')
# - [ ] 2 · The calibration loop (8')
# - [ ] 3 · Ranks for each parameter (5')
# - [ ] 4 · Find and fix the line (4')
# - [ ] 5 · Calibrate again (3')
# - [ ] ✱ Stretch · Calibrate a derived quantity: the rate of an old asbestos-cement district

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

import time
import numpy as np
import pandas as pd
from scipy import stats
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib
import matplotlib.pyplot as plt
import yanabi as U
print("mode:", U.MODE)
if not IN_COLAB:
    matplotlib.use("Agg")

north = U.districts().query("kind != 'urban'").head(20)
KM = north["network_km"].to_numpy()
AGE = (north["mean_age_years"].to_numpy() - 30) / 10
YEARS = 5
print(f"{len(north)} districts, {KM.sum():.0f} km in all")


def simulate(rng):
    """The simulator: draws the parameters from the priors, then five years of bursts."""
    alpha = rng.normal(np.log(0.15), 0.5)
    beta = rng.normal(0.3, 0.2)
    rate = np.exp(alpha + beta * AGE)
    y = rng.poisson(np.repeat(rate * KM, YEARS))
    return {"alpha": alpha, "beta": beta}, y


D_IDX = np.repeat(np.arange(len(KM)), YEARS)
LOG_KM = np.log(np.repeat(KM, YEARS))


def colleague_model(y=None):
    """The colleague's model. One line differs from what the simulator assumes."""
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.15), 0.5))
    beta = numpyro.sample("beta", dist.Normal(0.3, 0.2 ** 2))
    rate = jnp.exp(alpha + beta * AGE)
    numpyro.sample("y", dist.Poisson(jnp.exp(LOG_KM) * rate[D_IDX]), obs=y)


def make_fitter(model):
    """One compiled sampler reused for every simulated dataset: 1 chain, 300 warm-up, 99 kept draws (thinned by 2)."""
    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=198, thinning=2, num_chains=1,
                progress_bar=False, jit_model_args=True)

    def fit(y, seed):
        mcmc.run(jax.random.PRNGKey(seed), y=jnp.asarray(y))
        return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return fit


# %% [markdown]
# ## 1 · Parameter recovery on one dataset (5')
#
# ✏️ **Exercise** · Simulate one dataset with `simulate(U.rng(6))` (the dataset your colleague tested on), fit it with the colleague's model, and return a dict with, for each parameter, the true value and the 90% posterior interval.

# %%
fit_colleague = make_fitter(colleague_model)


def recovery():
    # BEGIN SOLUTION
    true, y = simulate(U.rng(6))
    post = fit_colleague(y, 1)
    return {k: (float(true[k]), *np.quantile(post[k], [0.05, 0.95]).tolist()) for k in true}
    # END SOLUTION


# %%
try:
    rec = recovery()
except NotImplementedError:
    rec = U.reference("d3_3", "recovery")
U.keep("d3_3", "recovery", rec)
for k, (t, lo, hi) in rec.items():
    print(f"{k:6s} true {t:+.3f}   90% interval {lo:+.3f} to {hi:+.3f}   {'inside' if lo <= t <= hi else 'OUTSIDE'}")
U.check(all(lo <= t <= hi for t, lo, hi in rec.values()), "one recovery run looks fine: both true values are inside")

# %% [markdown]
# One run proves little: a wrong model can cover the truth by luck, and a right one can miss it 10% of the time.
#
# ## 2 · The calibration loop (8')
#
# Simulation-based calibration: repeat N times: draw parameters and data from the simulator, fit, and record the **rank** of each true value among the 99 posterior draws (how many draws are below it, 0 to 99). If the code is right, every rank is equally likely.
#
# ✏️ **Exercise** · Write `sbc(fit, n)` that returns a DataFrame with one row per simulation and one column of ranks per parameter. Seed simulation i with `U.rng(100 + i)` and the fit with `i`.

# %%
N_SIM = U.pick(200, 100)


def sbc(fit, n):
    # BEGIN SOLUTION
    rows = []
    for i in range(n):
        true, y = simulate(U.rng(100 + i))
        post = fit(y, i)
        rows.append({k: int(np.sum(post[k] < true[k])) for k in true})
    return pd.DataFrame(rows)
    # END SOLUTION


# %%
t0 = time.time()
try:
    ranks = sbc(fit_colleague, N_SIM)
except NotImplementedError:
    ranks = U.reference("d3_3", "ranks_colleague")
U.keep("d3_3", "ranks_colleague", ranks)
print(f"{len(ranks)} simulations in {time.time() - t0:.0f} s")
U.check(len(ranks) == N_SIM and ranks.max().max() <= 99, "one rank per parameter per simulation, between 0 and 99")

# %% [markdown]
# ## 3 · Ranks for each parameter (5')
#
# ✏️ **Exercise** · Draw a histogram of ranks in 10 bins for each parameter into `artifacts/sbc_colleague.png`, and return a dict of chi-square p-values for uniformity (`stats.chisquare` on the 10 bin counts) and the share of ranks in the two outer bins (expected 20%).

# %%
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def rank_summary(ranks, path):
    # BEGIN SOLUTION
    out = {}
    fig, axes = plt.subplots(1, ranks.shape[1], figsize=(8, 3))
    for ax, k in zip(axes, ranks.columns):
        counts, _ = np.histogram(ranks[k], bins=10, range=(0, 100))
        ax.bar(range(10), counts); ax.axhline(len(ranks) / 10, color="k", lw=1); ax.set_title(k)
        out[k] = {"p_value": float(stats.chisquare(counts).pvalue), "outer": float((counts[0] + counts[-1]) / len(ranks))}
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return out
    # END SOLUTION


# %%
try:
    summ = rank_summary(ranks, OUT / "sbc_colleague.png")
except NotImplementedError:
    summ = U.reference("d3_3", "summary_colleague")
U.keep("d3_3", "summary_colleague", summ)
for k, v in summ.items():
    print(f"{k:6s} chi-square p = {v['p_value']:.4f}; share of ranks in the outer bins {v['outer']:.0%} (expected 20%)")
U.check(summ["beta"]["p_value"] < 0.01 and summ["beta"]["outer"] > 0.35, "β's ranks pile at both ends: its posterior is too narrow and in the wrong place")

# %% [markdown]
# **Look at** the two histograms. A U shape means the posterior is too narrow: the truth falls in its tails too often. Which line of `colleague_model` makes β's posterior too narrow, while α's is fine?
#
# ## 4 · Find and fix the line (4')
#
# ✏️ **Exercise** · Write `fixed_model`, identical to `colleague_model` except for the one wrong line, so that it matches the simulator.

# %%
def fixed_model(y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.15), 0.5))
    beta = numpyro.sample("beta", dist.Normal(0.3, 0.2))  # the colleague wrote the variance, 0.2 ** 2, where NumPyro expects the scale
    rate = jnp.exp(alpha + beta * AGE)
    numpyro.sample("y", dist.Poisson(jnp.exp(LOG_KM) * rate[D_IDX]), obs=y)
    # END SOLUTION


# %%
try:
    numpyro.handlers.seed(fixed_model, 0)()
    model_ok = fixed_model
except NotImplementedError:
    U.safety("fixed_model")
    model_ok = colleague_model
print("fixed model ready")

# %% [markdown]
# ## 5 · Calibrate again (3')
#
# ✏️ **Exercise** · Run the same calibration with the fixed model and summarise the ranks into `artifacts/sbc_fixed.png`.

# %%
def calibrate_fixed():
    # BEGIN SOLUTION
    r = sbc(make_fitter(model_ok), N_SIM)
    return r, rank_summary(r, OUT / "sbc_fixed.png")
    # END SOLUTION


# %%
try:
    ranks_ok, summ_ok = calibrate_fixed()
except NotImplementedError:
    summ_ok = U.reference("d3_3", "summary_fixed")
U.keep("d3_3", "summary_fixed", summ_ok)
for k, v in summ_ok.items():
    print(f"{k:6s} chi-square p = {v['p_value']:.3f}; outer bins {v['outer']:.0%}")
U.check(all(v["p_value"] > 0.01 for v in summ_ok.values()), "the fixed model's ranks are consistent with uniform")

# %% [markdown]
# ## ✱ Stretch · A derived quantity
#
# The decision uses rates, not α and β. Add the rate of a 40-year-old district, exp(α + β), to the ranked quantities and calibrate both models again. Does the error show in the quantity the decision uses?

# %%
def stretch():
    # BEGIN SOLUTION
    for name, model in [("colleague", colleague_model), ("fixed", model_ok)]:
        fit = make_fitter(model)
        ranks_r = []
        for i in range(U.pick(100, 60)):
            true, y = simulate(U.rng(100 + i))
            post = fit(y, i)
            ranks_r.append(int(np.sum(np.exp(post["alpha"] + post["beta"]) < np.exp(true["alpha"] + true["beta"]))))
        counts, _ = np.histogram(ranks_r, bins=10, range=(0, 100))
        print(f"{name:9s} rate at age 40: chi-square p = {stats.chisquare(counts).pvalue:.3f}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the implementation, not only one fit, by ranks over many simulated datasets.",
       "Found: a variance written where the scale belongs makes β's prior five times too narrow.",
       "Every capstone model gets the same test: a short calibration before its numbers are used.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why did one parameter-recovery run not reveal the error?
#    <details><summary>Answer</summary>The true β of that run happened to lie near 0.3, where the too-narrow prior still covers it; only many runs show how often it misses.</details>
# 2. What would ranks piled at the low end only mean?
#    <details><summary>Answer</summary>The posterior sits systematically above the truth: a bias, for example a wrong offset or a sign error.</details>
# 3. Does a calibration that passes prove the model is right for the data?
#    <details><summary>Answer</summary>No: it proves the code fits the model it claims to; whether the model fits the data is the job of posterior predictive checks.</details>
#
# ## Carry forward
#
# The capstone README reports a short calibration of the participant's own model.

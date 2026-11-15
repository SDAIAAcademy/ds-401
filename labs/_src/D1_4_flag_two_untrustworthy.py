# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 1.4 · Four fits, two untrustworthy
#
# Day 1 · Session 4 (Reading the computation) · 25 min.
#
# Four fits of Yanabi's burst rates reached the analytics team, two from Zaina's own runs and two from a contractor. Each has a run log. Before anyone reads a posterior from them, you compute the diagnostics of each, find the two that cannot be trusted, and write the reason in one sentence each. One fit looks alarming in one diagnostic and is still sound: say why.
#
# **AI policy.** The tutor Mirsad may explain what a diagnostic measures; the verdicts are yours.
#
# **Reference state.** `data/fits/fit_A.npz` to `fit_D.npz` and `data/fits/run_log.json`, written by the course's reference run. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Load the fits and read the run log (3')
# - [ ] 2 · R-hat and effective sample sizes (6')
# - [ ] 3 · Divergences, tree depth and energy (6')
# - [ ] 4 · Rank plots of the two suspects (5')
# - [ ] 5 · A verdict per fit, with its reason (5')
# - [ ] ✱ Stretch · Fit Lab 1.3's model with a variational approximation and compare the intervals with NUTS

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

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import yanabi as U
print("mode:", U.MODE)
if not IN_COLAB:
    matplotlib.use("Agg")  # a script run writes files; in a notebook the plots show inline

NAMES = ["fit_A", "fit_B", "fit_C", "fit_D"]
log = json.loads((U.DATA / "fits" / "run_log.json").read_text(encoding="utf-8"))

# %% [markdown]
# ## 1 · Load the fits and read the run log (3')
#
# ✏️ **Exercise** · Load the four fits with `U.load_fit(name)` into a dict, and print each run log line. Which fits were not run the way you ran Lab 1.3?

# %%
def load_all():
    # BEGIN SOLUTION
    fits = {n: U.load_fit(n) for n in NAMES}
    for n in NAMES:
        print(n, "·", log[n])
    return fits
    # END SOLUTION


# %%
try:
    fits = load_all()
except NotImplementedError:
    U.safety("load_all")
    fits = {n: U.load_fit(n) for n in NAMES}
U.check(len(fits) == 4 and all("posterior" in f.children for f in fits.values()), "four fits with a posterior each")

# %% [markdown]
# ## 2 · R-hat and effective sample sizes (6')
#
# The course's thresholds, for four chains: rank-normalised R-hat below 1.01; bulk and tail effective sample size of at least 100 per chain (400).
#
# ✏️ **Exercise** · Return a DataFrame indexed by fit with `max_rhat`, `min_ess_bulk`, `min_ess_tail` over all parameters, using `az.rhat` and `az.ess` (`method="bulk"` and `"tail"`).

# %%
def mixing(fits):
    # BEGIN SOLUTION
    rows = {}
    for n, f in fits.items():
        rh, eb, et = az.rhat(f), az.ess(f, method="bulk"), az.ess(f, method="tail")
        rows[n] = {"max_rhat": max(float(np.max(rh[v])) for v in rh.data_vars),
                   "min_ess_bulk": min(float(np.min(eb[v])) for v in eb.data_vars),
                   "min_ess_tail": min(float(np.min(et[v])) for v in et.data_vars)}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    mix = mixing(fits)
except NotImplementedError:
    mix = U.reference("d1_4", "mix")
U.keep("d1_4", "mix", mix)
print(mix.round(3).to_string())
U.check(mix.loc["fit_C", "max_rhat"] > 1.1 and mix.loc["fit_A", "max_rhat"] < 1.01, "fit C's chains disagree; fit A's agree")

# %% [markdown]
# ## 3 · Divergences, tree depth and energy (6')
#
# - A **divergence** is a trajectory the integrator could not follow: the sampler missed part of the posterior. Any number above zero needs explaining.
# - Hitting the **maximum tree depth** costs time, not validity.
# - **E-BFMI** below 0.3 says the momentum resampling cannot move the sampler across energy levels.
#
# ✏️ **Exercise** · Return a DataFrame indexed by fit with `divergences`, `share_capped` (share of draws whose tree depth reached the cap in the run log) and `min_ebfmi` (the lowest E-BFMI over chains; `U.diagnose` computes it, and you may use it).

# %%
CAPS = {"fit_A": 10, "fit_B": 10, "fit_C": 10, "fit_D": 4}


def sampler_health(fits):
    # BEGIN SOLUTION
    rows = {}
    for n, f in fits.items():
        dg = U.diagnose(f, max_tree_depth=CAPS[n])
        rows[n] = {"divergences": dg["divergences"], "share_capped": dg["tree_depth_hits"] / dg["draws"], "min_ebfmi": dg["min_ebfmi"]}
    return pd.DataFrame(rows).T
    # END SOLUTION


# %%
try:
    health = sampler_health(fits)
except NotImplementedError:
    health = U.reference("d1_4", "health")
U.keep("d1_4", "health", health)
print(health.round(3).to_string())
U.check(health.loc["fit_B", "divergences"] > 50 and health.loc["fit_D", "divergences"] == 0, "fit B diverges; fit D does not")

# %% [markdown]
# ## 4 · Rank plots of the two suspects (5')
#
# A rank plot pools all draws, ranks them, and shows each chain's ranks as a histogram. Chains that explore the same distribution give flat, similar histograms.
#
# ✏️ **Exercise** · Draw `az.plot_rank` of `alpha` and `tau` for fits B and C, save each figure to `artifacts/rank_<fit>.png`, and return the two paths.

# %%
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)


def rank_plots():
    # BEGIN SOLUTION
    paths = []
    for n in ["fit_B", "fit_C"]:
        pc = az.plot_rank(fits[n], var_names=["alpha", "tau"])
        path = OUT / f"rank_{n}.png"
        pc.savefig(path)
        plt.close("all")
        paths.append(path)
    return paths
    # END SOLUTION


# %%
try:
    paths = rank_plots()
except NotImplementedError:
    U.safety("rank_plots")
    paths = []
print("saved:", [str(x) for x in paths])
per_chain_alpha = np.asarray(fits["fit_C"]["posterior"]["alpha"]).mean(axis=1)
print("fit C, mean of alpha in each chain:", per_chain_alpha.round(3))
U.check(np.ptp(per_chain_alpha) > 0.3, "one chain of fit C sits somewhere else")

# %% [markdown]
# ## 5 · A verdict per fit, with its reason (5')
#
# ✏️ **Exercise** · Return a dict `fit -> ("trust" or "do not trust", reason)`. Each reason names the diagnostic and its number.

# %%
def verdicts():
    # BEGIN SOLUTION
    return {
        "fit_A": ("trust", f"R-hat {mix.loc['fit_A', 'max_rhat']:.3f}, tail ESS {mix.loc['fit_A', 'min_ess_tail']:.0f}, no divergence"),
        "fit_B": ("do not trust", f"{int(health.loc['fit_B', 'divergences'])} divergences: the centred model cannot reach the narrow neck of the posterior for small districts"),
        "fit_C": ("do not trust", f"R-hat {mix.loc['fit_C', 'max_rhat']:.2f}: one chain never left its starting region, although no divergence was reported"),
        "fit_D": ("trust", f"{health.loc['fit_D', 'share_capped']:.0%} of trees hit the depth cap of 4, which costs efficiency, but R-hat {mix.loc['fit_D', 'max_rhat']:.3f} and ESS {mix.loc['fit_D', 'min_ess_bulk']:.0f} are sound"),
    }
    # END SOLUTION


# %%
try:
    verdict = verdicts()
except NotImplementedError:
    verdict = U.reference("d1_4", "verdict")
U.keep("d1_4", "verdict", verdict)
for k, (v, why) in verdict.items():
    print(f"{k}: {v:13s} {why}")
bad = sorted(k for k, (v, _) in verdict.items() if v != "trust")
U.check(bad == ["fit_B", "fit_C"], "fits B and C are the two untrustworthy ones")

# %% [markdown]
# ## ✱ Stretch · When an approximation is enough
#
# Fit Lab 1.3's pooled model with stochastic variational inference (`AutoNormal` guide, 3,000 steps) and compare the 90% intervals of β with those of NUTS. For a model with 5 parameters and 700 counts, is the approximation good enough for a rate ratio? Would you trust it for the 140 district effects of Day 2?

# %%
def stretch():
    # BEGIN SOLUTION
    import jax
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam
    D = U.design()
    md = U.model_data(D)
    m = lambda **kw: U.burst_model(pooling="complete", **kw)
    svi = SVI(m, AutoNormal(m), Adam(0.02), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), U.pick(3000, 1500), y=D["y"], progress_bar=False, **md)
    post = Predictive(svi.guide, params=res.params, num_samples=2000)(jax.random.PRNGKey(1), y=D["y"], **md)
    b_svi = np.asarray(post["beta"])
    _, idt = U.sample(m, y=D["y"], **md)
    b_nuts = U.draws(idt, "beta")
    for i, n in enumerate(U.X_NAMES):
        print(f"{n:16s} NUTS {np.quantile(b_nuts[:, i], [0.05, 0.95]).round(3)}  SVI {np.quantile(b_svi[:, i], [0.05, 0.95]).round(3)}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: every fit's mixing and sampler health before any of its numbers is used.",
       "No divergence is not enough (fit C); many capped trees is not a failure (fit D).",
       "Carry forward: the four-line check (R-hat, ESS, divergences, E-BFMI) runs after every fit for the rest of the course.")

# %% [markdown]
# ## Checkpoint
#
# 1. Fit C reported no divergence. Why is it still untrustworthy?
#    <details><summary>Answer</summary>One chain stayed near where it started; the chains disagree (R-hat far above 1.01). Divergences detect one failure, not all.</details>
# 2. Fit D hit its tree-depth cap in a third of its draws. Why is it trusted?
#    <details><summary>Answer</summary>Capped trees make sampling slower and draws more correlated, but its R-hat and effective sample sizes show the chains still mixed.</details>
# 3. What would you ask the contractor to change for fit B?
#    <details><summary>Answer</summary>Rewrite the model in non-centred form (Day 2, session 3) rather than only raising the target acceptance rate.</details>
#
# ## Carry forward
#
# Fit B's model returns in Lab 2.3, where you repair it.

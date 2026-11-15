# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 3.1 · Zeros and dispersion
#
# Day 3 · Session 1 (Posterior predictive checks) · 25 min.
#
# Yesterday's model (partial pooling, material and age, the reporting term) is Poisson: once a district's rate is known, its yearly counts vary only as much as a Poisson allows. Shabib doubts it: "some years a district has nothing, the next year a cluster". You simulate replicated burst records from the fitted model, compare them with the real ones on two statistics that matter for the decision, then write the negative binomial version and check it the same way.
#
# **AI policy.** AI-off: this is the day's first model. Write it yourself.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`; the Poisson model is `U.burst_model(reporting=True)`, so Day 2's labs need not be finished.
#
# **Tasks**
# - [ ] 1 · Replicated data from the Poisson model (6')
# - [ ] 2 · Share of zeros and dispersion (6')
# - [ ] 3 · Fit a negative binomial (6')
# - [ ] 4 · The same checks for the negative binomial (5')
# - [ ] 5 · A verdict in two sentences (2')
# - [ ] ✱ Stretch · Draw a rootogram of both models

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
md = U.model_data(D)
poisson_model = lambda **kw: U.burst_model(reporting=True, **kw)
mc_po, po = U.sample(poisson_model, y=D["y"], **md)
print("Poisson model:", {k: round(v, 3) for k, v in U.diagnose(po, ["alpha", "beta", "tau"]).items()})

# %% [markdown]
# ## 1 · Replicated data from the Poisson model (6')
#
# A posterior predictive check: for each posterior draw, simulate a full set of 700 district-years from the model with that draw's parameters. Real data that look unlike the replicates expose a misfit.
#
# ✏️ **Exercise** · Return an array of shape `(draws, 700)` of replicated counts with `U.predict(poisson_model, mc_po, **md)`.

# %%
def replicate():
    # BEGIN SOLUTION
    return U.predict(poisson_model, mc_po, **md)
    # END SOLUTION


# %%
try:
    yrep_po = replicate()
except NotImplementedError:
    U.safety("replicate")
    yrep_po = U.predict(poisson_model, mc_po, **md)
print("replicates:", yrep_po.shape, "observed total", D["y"].sum(), "replicated totals median", int(np.median(yrep_po.sum(axis=1))))
U.check(yrep_po.shape[1] == len(D["y"]), "one replicate of every district-year per draw")

# %% [markdown]
# ## 2 · Share of zeros and dispersion (6')
#
# Two test statistics, chosen for the decision:
# - `zeros_busy`: the share of district-years with no burst among the districts with at least 10 km of mains (where a quiet year is informative);
# - `dispersion`: the average, over those districts, of the variance of their five yearly counts divided by their mean.
#
# ✏️ **Exercise** · Write `stats_of(y)` returning both statistics for one set of 700 counts, then return the observed values, the replicated values and the posterior predictive p-values (the share of replicates at least as extreme as the observed value).

# %%
busy = D["km"][D["d"]] >= 10


def stats_of(y):
    # BEGIN SOLUTION
    yy = np.asarray(y)[busy]
    dd = D["d"][busy]
    zeros = float(np.mean(yy == 0))
    m = np.bincount(dd, weights=yy, minlength=D["J"])[np.unique(dd)] / 5
    m2 = np.bincount(dd, weights=yy ** 2, minlength=D["J"])[np.unique(dd)] / 5
    var = (m2 - m ** 2) * 5 / 4
    disp = float(np.mean(var[m > 0] / m[m > 0]))
    return {"zeros_busy": zeros, "dispersion": disp}
    # END SOLUTION


def ppc(yrep):
    obs = stats_of(D["y"])
    rep = pd.DataFrame([stats_of(r) for r in yrep])
    p = {k: float(np.mean(rep[k] >= obs[k])) for k in obs}
    return obs, rep, p


# %%
try:
    obs, rep_po, p_po = ppc(yrep_po)
except NotImplementedError:
    obs, rep_po, p_po = U.reference("d3_1", "obs"), None, U.reference("d3_1", "p_po")
U.keep("d3_1", "obs", obs)
U.keep("d3_1", "p_po", p_po)
for k in obs:
    rq = np.quantile(rep_po[k], [0.05, 0.95]) if rep_po is not None else (np.nan, np.nan)
    print(f"{k:11s} observed {obs[k]:.3f}; Poisson replicates 90% {rq[0]:.3f} to {rq[1]:.3f}; p = {p_po[k]:.3f}")
U.check(p_po["dispersion"] < 0.01, "the Poisson model cannot produce the observed dispersion")

# %% [markdown]
# **Look at** both lines. A Poisson has variance equal to its mean, so the dispersion replicates sit near 1; the real yearly counts vary more than twice as much. The share of quiet years, by contrast, sits at the top of the Poisson range without leaving it: not every statistic exposes a misfit, which is why you choose more than one, and choose them for the decision.
#
# ## 3 · Fit a negative binomial (6')
#
# Replace the Poisson with a negative binomial with mean μ and shape φ: variance μ + μ²/φ. `dist.NegativeBinomial2(mean, phi)` takes exactly these. Prior: φ ~ Gamma(2, 0.2) (mean 10: mild extra variation expected).
#
# ✏️ **Exercise** · Write `negbin_model` with the same arguments, priors and reporting term as `U.burst_model(reporting=True)` and a negative binomial likelihood. Name the shape `phi`.

# %%
def negbin_model(d, log_km, X, sensors, J, y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("district", J):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        p_raw = numpyro.sample("p_report", dist.Beta(14.0, 6.0))
    eta = alpha + X @ beta + tau * z
    numpyro.deterministic("rate", jnp.exp(eta))
    p = jnp.where(sensors, 1.0, p_raw)
    phi = numpyro.sample("phi", dist.Gamma(2.0, 0.2))
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_km + eta[d] + jnp.log(p)[d]), phi), obs=y)
    # END SOLUTION


# %%
try:
    numpyro.handlers.seed(negbin_model, 0)(**md)
    nb_model = negbin_model
except NotImplementedError:
    U.safety("negbin_model")
    nb_model = lambda **kw: U.burst_model(reporting=True, family="negbin", **kw)
mc_nb, nb = U.sample(nb_model, y=D["y"], **md)
dn = U.diagnose(nb, ["alpha", "beta", "tau", "phi"])
print({k: round(v, 3) for k, v in dn.items()})
print("φ:", U.describe(U.draws(nb, "phi")), "| generating value", U.truth()["phi"])
U.check(U.trustworthy(dn), "the negative binomial fit can be read")

# %% [markdown]
# ## 4 · The same checks for the negative binomial (5')
#
# ✏️ **Exercise** · Replicate from the negative binomial model and return its p-values for the two statistics.

# %%
def check_nb():
    # BEGIN SOLUTION
    yrep = U.predict(nb_model, mc_nb, **md)
    _, rep, p = ppc(yrep)
    return rep, p
    # END SOLUTION


# %%
try:
    rep_nb, p_nb = check_nb()
except NotImplementedError:
    rep_nb, p_nb = None, U.reference("d3_1", "p_nb")
U.keep("d3_1", "p_nb", p_nb)
for k in obs:
    print(f"{k:11s} observed {obs[k]:.3f}; negative binomial p = {p_nb[k]:.3f}")
U.check(all(0.05 < v < 0.95 for v in p_nb.values()), "the negative binomial reproduces both statistics")

# %% [markdown]
# ## 5 · A verdict in two sentences (2')
#
# ✏️ **Exercise** · Write the two sentences for the model comparison table: what the Poisson model gets wrong, and why it matters for Day 4.

# %%
def verdict():
    # BEGIN SOLUTION
    return [f"The Poisson model cannot reproduce the observed dispersion ({obs['dispersion']:.2f} against replicates near 1, p = {p_po['dispersion']:.3f}); the quiet years of districts over 10 km sit at the edge of its range (p = {p_po['zeros_busy']:.2f}). The negative binomial reproduces both.",
            "A model that understates year-to-year variation is too sure of every district's rate, so the Day 4 plan would look less risky than it is."]
    # END SOLUTION


# %%
try:
    v = verdict()
except NotImplementedError:
    v = U.reference("d3_1", "verdict")
U.keep("d3_1", "verdict", v)
for s in v:
    print("-", s)
U.check(len(v) == 2, "two sentences")

# %% [markdown]
# ## ✱ Stretch · A rootogram
#
# For counts 0 to 15, plot the square root of the observed frequency against the square root of the mean replicated frequency for both models. Predict first: will the Poisson curve miss the bars? Then explain what you see with task 2's dispersion statistic, which is computed within each district.

# %%
def stretch():
    # BEGIN SOLUTION
    yrep_nb = U.predict(nb_model, mc_nb, **md)
    k = np.arange(16)
    f_obs = np.array([np.sum(D["y"] == i) for i in k])
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, yr, c in [("Poisson", yrep_po, "tab:red"), ("negative binomial", yrep_nb, "tab:blue")]:
        f_rep = np.array([np.mean(np.sum(yr == i, axis=1)) for i in k])
        ax.plot(k, np.sqrt(f_rep), marker="o", c=c, label=name)
    ax.bar(k, np.sqrt(f_obs), alpha=0.3, color="grey", label="observed")
    ax.set_xlabel("bursts in a district-year"); ax.set_ylabel("sqrt(frequency)"); ax.legend()
    Path("artifacts").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig("artifacts/rootogram.png", dpi=110); plt.close(fig)
    print("saved artifacts/rootogram.png")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the model against its own replicated data, on statistics chosen for the decision.",
       f"Found: φ around {np.median(U.draws(nb, 'phi')):.1f}: years vary well beyond a Poisson.",
       "Carry forward: three candidates for Lab 3.2: complete pooling, the Poisson hierarchy and the negative binomial hierarchy.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why compare statistics of replicated data rather than the parameters?
#    <details><summary>Answer</summary>The question is whether the model could have produced data like ours; parameters of a wrong model can look perfectly reasonable.</details>
# 2. Why choose the zeros of districts over 10 km and not all zeros?
#    <details><summary>Answer</summary>Tiny districts have zeros under any model; a quiet year where several bursts are expected is what separates the models.</details>
# 3. A posterior predictive p-value of 0.5 on one statistic: is the model right?
#    <details><summary>Answer</summary>No: it reproduces that statistic. Other features may still fail, which is why the statistics are chosen for the decision.</details>
#
# ## Carry forward
#
# Lab 3.2 compares the models by how well they predict a district-year they did not see.

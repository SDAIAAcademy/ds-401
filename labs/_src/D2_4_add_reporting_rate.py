# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 2.4 · A reporting rate for districts without sensors
#
# Day 2 · Session 4 (Under-reporting as measurement error) · 25 min.
#
# Districts with sensors record every burst. In the 76 districts without sensors, a burst is recorded only when someone reports it. In 2024 Ouda Al-Shahwan's leak detection team listened to six districts without sensors for a season: of the bursts they found, 70% had been reported. You turn that audit into a prior on the share reported, add a reporting term to the model, and see what changes for the districts without sensors.
#
# **AI policy.** The tutor Mirsad may explain; it writes no model code.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, loaded fresh; the model of Lab 2.2 is available as `U.burst_model`, so Lab 2.2 need not be finished.
#
# **Tasks**
# - [ ] 1 · Reported bursts with and without sensors (4')
# - [ ] 2 · The audit prior on the reporting share (5')
# - [ ] 3 · Add the reporting term (7')
# - [ ] 4 · Corrected rates for districts without sensors (6')
# - [ ] 5 · What the data cannot tell apart (3')
# - [ ] ✱ Stretch · Change the audit prior and watch the corrected rates follow it

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
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import yanabi as U
print("mode:", U.MODE)

D = U.design()
md = U.model_data(D)
kind = np.array(D["kind"])
sens = D["sensors"]
true_rate = np.array([U.truth()["true_rate_per_km_year"][k] for k in D["ids"]])
print(f"districts without sensors: {(~sens).sum()} (rural {((~sens) & (kind == 'rural')).sum()})")

# %% [markdown]
# ## 1 · Reported bursts with and without sensors (4')
#
# ✏️ **Exercise** · For rural districts only, return the reported bursts per km-year of districts with sensors and of districts without (total bursts over total km-years, a dict with keys `sensors` and `no_sensors`).

# %%
def raw_rates():
    # BEGIN SOLUTION
    tot = np.bincount(D["d"], weights=D["y"], minlength=D["J"])
    exp = D["km"] * 5
    r = kind == "rural"
    return {"sensors": float(tot[r & sens].sum() / exp[r & sens].sum()),
            "no_sensors": float(tot[r & ~sens].sum() / exp[r & ~sens].sum())}
    # END SOLUTION


# %%
try:
    raw = raw_rates()
except NotImplementedError:
    raw = U.reference("d2_4", "raw")
U.keep("d2_4", "raw", raw)
print(f"rural districts, reported bursts per km-year: with sensors {raw['sensors']:.3f}, without {raw['no_sensors']:.3f}")
U.check(raw["no_sensors"] < raw["sensors"], "rural districts without sensors report fewer bursts per km-year")

# %% [markdown]
# The gap could be real (different pipes) or unrecorded bursts. The counts alone cannot say which: a district with rate λ and reporting share p produces exactly the same reported counts as one with rate λp and full reporting.
#
# ## 2 · The audit prior on the reporting share (5')
#
# The audit found 28 of 40 bursts had been reported. A Beta(14, 6) prior carries the audit's 70% with the weight of about 20 bursts, half the audit's 40, because six districts in one season may not represent all 76.
#
# ✏️ **Exercise** · Return the prior mean and 90% interval of the reporting share under Beta(14, 6), and the same for Beta(28, 12), the audit taken at full weight.

# %%
def audit_prior():
    # BEGIN SOLUTION
    out = {}
    for name, (a, b) in {"Beta(14, 6)": (14, 6), "Beta(28, 12)": (28, 12)}.items():
        out[name] = (a / (a + b), *stats.beta.ppf([0.05, 0.95], a, b))
    return out
    # END SOLUTION


# %%
try:
    ap = audit_prior()
except NotImplementedError:
    ap = U.reference("d2_4", "audit_prior")
U.keep("d2_4", "audit_prior", ap)
for k, (m, lo, hi) in ap.items():
    print(f"{k}: mean {m:.2f}, 90% interval {lo:.2f} to {hi:.2f}")
U.check(abs(ap["Beta(14, 6)"][0] - 0.70) < 1e-9, "both priors are centred on the audit's 70%")

# %% [markdown]
# ## 3 · Add the reporting term (7')
#
# Reported bursts ~ Poisson(km × rate × p_j), with p_j = 1 where there are sensors and p_j ~ Beta(14, 6) where there are none. On the log scale: add log p_j to the log mean.
#
# ✏️ **Exercise** · Complete `reporting_model`: the Lab 2.2 model (non-centred, material and age) plus the reporting term. Name the reporting site `p_report`.

# %%
def reporting_model(d, log_km, X, sensors, J, y=None):
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
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d] + jnp.log(p)[d])), obs=y)
    # END SOLUTION


# %%
try:
    numpyro.handlers.seed(reporting_model, 0)(**md)
    model = reporting_model
except NotImplementedError:
    U.safety("reporting_model")
    model = lambda **kw: U.burst_model(reporting=True, **kw)
_, rep = U.sample(model, y=D["y"], **md)
_, base = U.sample(U.burst_model, y=D["y"], **md)  # Lab 2.2's model, without the reporting term
dr = U.diagnose(rep, ["alpha", "beta", "tau"])
print({k: round(v, 3) for k, v in dr.items()})
U.check(U.trustworthy(dr), "the model with the reporting term can be read")

# %% [markdown]
# ## 4 · Corrected rates for districts without sensors (6')
#
# ✏️ **Exercise** · Return a DataFrame by kind and sensors with the mean of log(estimate / true rate) for both models, and the mean width of the 90% interval of the rate relative to its median (`rel_width`) for both.

# %%
def corrected():
    # BEGIN SOLUTION
    rows = []
    for name, f in [("base", base), ("reporting", rep)]:
        r = U.draws(f, "rate")
        med = np.median(r, axis=0)
        lo, hi = np.quantile(r, [0.05, 0.95], axis=0)
        rows.append(pd.DataFrame({"model": name, "kind": kind, "sensors": sens,
                                  "bias": np.log(med / true_rate), "rel_width": (hi - lo) / med}))
    t = pd.concat(rows).groupby(["kind", "sensors", "model"])[["bias", "rel_width"]].mean()
    return t.unstack("model")
    # END SOLUTION


# %%
try:
    cor = corrected()
except NotImplementedError:
    cor = U.reference("d2_4", "corrected")
U.keep("d2_4", "corrected", cor)
print(cor.round(3).to_string())
rb = cor.loc[("rural", False)]
U.check(abs(rb[("bias", "reporting")]) < abs(rb[("bias", "base")]) - 0.15, "the reporting term removes most of the rural districts' bias")
ub = cor.loc[("urban", False)]
U.check(ub[("rel_width", "reporting")] > ub[("rel_width", "base")] + 0.05, "urban districts without sensors get wider intervals: the uncertainty about reporting is now in the answer")

# %% [markdown]
# **Look at** the widths. Urban and suburban districts without sensors had precise counts, so the reporting share is now the main uncertainty and their intervals widen. Rural districts' intervals were already wide from having few bursts; the correction moves them up more than it widens them.
#
# ## 5 · What the data cannot tell apart (3')
#
# ✏️ **Exercise** · For the districts without sensors, return the average ratio of the posterior standard deviation of p_report to its prior standard deviation (Beta(14, 6)). A ratio near 1 means the data taught the model almost nothing about reporting.

# %%
def learned():
    # BEGIN SOLUTION
    sd_post = U.draws(rep, "p_report")[:, ~sens].std(axis=0)
    return float(np.mean(sd_post / stats.beta.std(14, 6)))
    # END SOLUTION


# %%
try:
    ratio = learned()
except NotImplementedError:
    ratio = U.reference("d2_4", "learned")
U.keep("d2_4", "learned", ratio)
print(f"posterior sd of the reporting share over its prior sd: {ratio:.2f}")
U.check(ratio > 0.8, "the reporting share is carried by the audit prior, not by the counts")

# %% [markdown]
# The correction is only as good as the audit. Nouran's memo must say so, and a better audit (or the survey priced on Day 4) is the way to improve it.
#
# ## ✱ Stretch · Another audit prior
#
# Refit with Beta(9, 1) (reporting is nearly complete) and with Beta(5, 5). How do the corrected rural rates follow the prior?

# %%
def stretch():
    # BEGIN SOLUTION
    def model_ab(a, b):
        def m(d, log_km, X, sensors, J, y=None):
            alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
            beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
            tau = numpyro.sample("tau", dist.HalfNormal(0.5))
            with numpyro.plate("district", J):
                z = numpyro.sample("z", dist.Normal(0.0, 1.0))
                pr = numpyro.sample("p_report", dist.Beta(a, b))
            eta = alpha + X @ beta + tau * z
            numpyro.deterministic("rate", jnp.exp(eta))
            numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d] + jnp.log(jnp.where(sensors, 1.0, pr))[d])), obs=y)
        return m
    m_rural = (kind == "rural") & ~sens
    for a, b in [(9, 1), (5, 5)]:
        _, f = U.sample(model_ab(a, b), y=D["y"], **md)
        print(f"Beta({a}, {b}): rural no-sensor median rate {np.median(np.median(U.draws(f, 'rate'), axis=0)[m_rural]):.3f}")
    print(f"Beta(14, 6): {np.median(np.median(U.draws(rep, 'rate'), axis=0)[m_rural]):.3f}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: reported counts are now modelled as true counts thinned by a reporting share.",
       "Found: the correction comes from the audit prior; the counts cannot identify it.",
       "Carry forward: this model, with a negative binomial in place of the Poisson (Lab 3.1), is the one Day 4 decides with.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why can't the counts identify the reporting share by themselves?
#    <details><summary>Answer</summary>Only the product of rate and reporting share reaches the counts; many pairs give the same product.</details>
# 2. Why did the intervals of the urban districts without sensors widen?
#    <details><summary>Answer</summary>Their counts were precise; the uncertainty about how many bursts went unrecorded is now part of the posterior instead of being assumed away.</details>
# 3. Why use the audit at half weight?
#    <details><summary>Answer</summary>Six districts in one season may not represent all 76 without sensors; a prior worth 20 bursts instead of 40 says so.</details>
#
# ## Carry forward
#
# Lab 3.1 checks whether the Poisson part of this model fits the year-to-year variation.

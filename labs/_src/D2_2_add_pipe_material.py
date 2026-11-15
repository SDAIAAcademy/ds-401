# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 2.2 · Material and age at the district level
#
# Day 2 · Session 2 (Group-level predictors) · 25 min.
#
# Lab 2.1 shrank every district towards one common rate, which the urban networks dominate. Here each district is shrunk towards the rate predicted for its pipe material and mean age. You compare the spread between districts and the rural pull with and without the predictors.
#
# **AI policy.** The tutor Mirsad may explain; it writes no model code.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, loaded fresh; the Lab 2.1 model without predictors is refitted here from `U.burst_model`, so Lab 2.1 need not be finished.
#
# **Tasks**
# - [ ] 1 · District-level predictors in the mean (6')
# - [ ] 2 · Fit and diagnose (5')
# - [ ] 3 · The spread between districts, before and after (4')
# - [ ] 4 · The rural districts again (6')
# - [ ] 5 · What the predictors explain (4')
# - [ ] ✱ Stretch · Add the district kind as a predictor and discuss what it stands for

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
import yanabi as U
print("mode:", U.MODE)

D = U.design()
md = U.model_data(D)
kind = np.array(D["kind"])
true_rate = np.array([U.truth()["true_rate_per_km_year"][k] for k in D["ids"]])
_, base = U.sample(U.burst_model, y=D["y"], predictors=False, **md)  # Lab 2.1's partial pooling, as the baseline
print("baseline τ:", U.describe(U.draws(base, "tau")))

# %% [markdown]
# ## 1 · District-level predictors in the mean (6')
#
# The district effect now has a mean that depends on the district: η_j ~ Normal(α + X_j β, τ), with β ~ Normal(0, 0.5) for age per 10 years and the three materials (PVC is the base). Write it in non-centred form, as `U.burst_model` does: z_j ~ Normal(0, 1) and η_j = α + X_j β + τ z_j (session 3 says why).
#
# ✏️ **Exercise** · Complete `predictor_model`. Name the sites `alpha`, `beta`, `tau`, `z`, `eta`, `rate`, `y`.

# %%
def predictor_model(d, log_km, X, J, y=None):
    # BEGIN SOLUTION
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    with numpyro.plate("district", J):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    eta = numpyro.deterministic("eta", alpha + X @ beta + tau * z)
    numpyro.deterministic("rate", jnp.exp(eta))
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
    # END SOLUTION


# %%
args = {"d": D["d"], "log_km": D["log_km"], "X": D["X"], "J": D["J"]}
try:
    numpyro.handlers.seed(predictor_model, 0)(**args)
    model = predictor_model
except NotImplementedError:
    U.safety("predictor_model")
    model = lambda d, log_km, X, J, y=None: U.burst_model(d, log_km, X, D["sensors"], J, y)
print("model ready")

# %% [markdown]
# ## 2 · Fit and diagnose (5')
#
# ✏️ **Exercise** · Fit the model with `U.sample`, print the four diagnostics and return the idata.

# %%
def fit():
    # BEGIN SOLUTION
    _, idata = U.sample(model, y=D["y"], **args)
    print({k: round(v, 3) for k, v in U.diagnose(idata, ["alpha", "beta", "tau"]).items()})
    return idata
    # END SOLUTION


# %%
try:
    pred = fit()
except NotImplementedError:
    U.safety("fit")
    _, pred = U.sample(model, y=D["y"], **args)
U.check(U.trustworthy(U.diagnose(pred, ["alpha", "beta", "tau"])), "the fit can be read")

# %% [markdown]
# ## 3 · The spread between districts, before and after (4')
#
# ✏️ **Exercise** · Return a dict with the posterior medians of τ without predictors (`base`) and with them (`predictors`), and the share of the between-district variance (τ²) the predictors explain.

# %%
def spread():
    # BEGIN SOLUTION
    t0, t1 = float(np.median(U.draws(base, "tau"))), float(np.median(U.draws(pred, "tau")))
    return {"base": t0, "predictors": t1, "explained": 1 - t1 ** 2 / t0 ** 2}
    # END SOLUTION


# %%
try:
    sp = spread()
except NotImplementedError:
    sp = U.reference("d2_2", "spread")
U.keep("d2_2", "spread", sp)
print(f"τ without predictors {sp['base']:.2f}, with them {sp['predictors']:.2f}: material and age explain {sp['explained']:.0%} of the variance between districts")
U.check(sp["predictors"] < sp["base"], "the predictors explain part of the spread")

# %% [markdown]
# ## 4 · The rural districts again (6')
#
# ✏️ **Exercise** · Return a DataFrame by kind with the mean of log(estimate / true rate) for the baseline (`base`) and for the model with predictors (`predictors`), using posterior medians of `rate`.

# %%
def rural_again():
    # BEGIN SOLUTION
    rows = {}
    for name, f in [("base", base), ("predictors", pred)]:
        r = np.median(U.draws(f, "rate"), axis=0)
        rows[name] = pd.Series(np.log(r / true_rate)).groupby(kind).mean()
    return pd.DataFrame(rows)
    # END SOLUTION


# %%
try:
    bias = rural_again()
except NotImplementedError:
    bias = U.reference("d2_2", "bias")
U.keep("d2_2", "bias", bias)
print(bias.round(3).to_string())
U.check(bias.loc["rural", "predictors"] > bias.loc["rural", "base"] + 0.2, "the rural pull is much smaller with the predictors")
U.check(bias.loc["rural", "predictors"] < -0.1, "but rural districts are still underestimated: something else is going on")

# %% [markdown]
# **Look at** the rural row. The predictors removed much of the pull, yet the rural districts are still estimated below their true rates. Most rural districts have no sensors. Lab 2.4 asks whether they report every burst.
#
# ## 5 · What the predictors explain (4')
#
# ✏️ **Exercise** · Return the posterior median rate ratios exp(β) by predictor name (`U.X_NAMES`) and compare them with Lab 1.3's complete-pooling ratios: asbestos cement 1.85, age per 10 years 1.35.

# %%
def ratios():
    # BEGIN SOLUTION
    return dict(zip(U.X_NAMES, np.median(np.exp(U.draws(pred, "beta")), axis=0).round(3).tolist()))
    # END SOLUTION


# %%
try:
    rr = ratios()
except NotImplementedError:
    rr = U.reference("d2_2", "ratios")
U.keep("d2_2", "ratios", rr)
print(rr)
U.check(rr["asbestos cement"] > 1.5, "asbestos cement still bursts more often once districts have their own effects")

# %% [markdown]
# ## ✱ Stretch · The district kind as a predictor
#
# Add an indicator for rural districts to `X` and refit. Predict first: will the rural estimates move towards their true rates? Then read the coefficient. Is "rural" a cause of bursts, or does it stand for something the model does not have (here: missing sensors)?

# %%
def stretch():
    # BEGIN SOLUTION
    X2 = np.column_stack([D["X"], (kind == "rural").astype(float)])
    _, f = U.sample(model, y=D["y"], d=D["d"], log_km=D["log_km"], X=X2, J=D["J"])
    r = np.median(U.draws(f, "rate"), axis=0)
    print("rural coefficient:", U.describe(U.draws(f, "beta")[:, -1]))
    print("rural mean log(estimate / truth):", round(float(np.mean(np.log(r / true_rate)[kind == "rural"])), 3))
    print("the rural coefficient comes out negative: it learns the low reported counts of rural districts, most without sensors, and moves the estimates further from the truth")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: districts are now shrunk towards a prediction for their material and age.",
       "Found: the rural pull shrinks but does not vanish.",
       "Carry forward: this model is the one Lab 2.4 extends with a reporting term.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why did τ fall when material and age entered the model?
#    <details><summary>Answer</summary>Part of what made districts differ is now explained by the predictors, so less is left to the district effects.</details>
# 2. Towards what is a rural asbestos-cement district shrunk now?
#    <details><summary>Answer</summary>Towards the rate predicted for an asbestos-cement network of its mean age, not towards the network-wide average.</details>
# 3. Why is a "rural" indicator a weak fix?
#    <details><summary>Answer</summary>It absorbs whatever differs about rural districts, here the missing sensors, without saying what it is; the decision then cannot tell real risk from unrecorded bursts.</details>
#
# ## Carry forward
#
# Lab 2.3 turns to the computation: what happens to this model on the small districts in centred form.

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Pre-work · Likelihoods, simulation and where data may go
#
# About three hours before Day 1, at your own pace. An entry quiz at the start of Day 1 checks it.
#
# Part A refreshes what the course assumes: simulating from a distribution, computing a likelihood and finding its maximum. Part B asks a data-handling question you will meet at work: may Yanabi's real burst and customer records be analysed in a hosted notebook?
#
# **Tasks**
# - [ ] A1 · Simulate counts and check their mean and variance (30')
# - [ ] A2 · The Poisson log likelihood of a rate (40')
# - [ ] A3 · The maximum and the curvature of the likelihood (40')
# - [ ] B1 · Classify four files and decide where each may be processed (50')
# - [ ] ✱ Stretch · Repeat A2 and A3 with a negative binomial and two unknowns

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
from scipy import stats, optimize
import yanabi as U
print("mode:", U.MODE)

# %% [markdown]
# ## A1 · Simulate counts and check their mean and variance (30')
#
# ✏️ **Exercise** · Draw 10,000 Poisson counts with mean 3 and 10,000 negative binomial counts with mean 3 and shape 2 (`rng.negative_binomial(n=2, p=2 / (2 + 3))`). Return the mean and variance of each as a dict.

# %%
def simulate_two():
    # BEGIN SOLUTION
    r = U.rng(1)
    po = r.poisson(3, 10000)
    nb = r.negative_binomial(2, 2 / 5, 10000)
    return {"poisson": (po.mean(), po.var()), "negbin": (nb.mean(), nb.var())}
    # END SOLUTION


# %%
try:
    mv = simulate_two()
except NotImplementedError:
    U.safety("simulate_two")
    mv = {"poisson": (3.0, 3.0), "negbin": (3.0, 7.5)}
print({k: tuple(round(float(x), 2) for x in v) for k, v in mv.items()})
U.check(abs(mv["poisson"][1] - 3) < 0.3 and mv["negbin"][1] > 6, "Poisson variance equals the mean; the negative binomial's is mean + mean²/shape = 7.5")

# %% [markdown]
# ## A2 · The Poisson log likelihood of a rate (40')
#
# A district of 12 km recorded 2, 4, 1, 3 and 5 bursts in five years.
#
# ✏️ **Exercise** · Write `loglik(rate)`, the Poisson log likelihood of these counts for a rate in bursts per km-year (mean per year = rate × 12).

# %%
COUNTS = np.array([2, 4, 1, 3, 5])
KM = 12.0


def loglik(rate):
    # BEGIN SOLUTION
    return float(stats.poisson.logpmf(COUNTS, rate * KM).sum())
    # END SOLUTION


# %%
try:
    vals = [loglik(r) for r in (0.1, 0.25, 0.5)]
except NotImplementedError:
    U.safety("loglik")
    vals = [float(stats.poisson.logpmf(COUNTS, r * KM).sum()) for r in (0.1, 0.25, 0.5)]
print("log likelihood at 0.10, 0.25, 0.50:", [round(v, 2) for v in vals])
U.check(vals[1] > vals[0] and vals[1] > vals[2], "0.25 is more likely than 0.10 or 0.50")

# %% [markdown]
# ## A3 · The maximum and the curvature of the likelihood (40')
#
# ✏️ **Exercise** · Find the maximum-likelihood rate numerically (`optimize.minimize_scalar` on minus the log likelihood, bounds 0.001 to 2) and compare with the closed form, total bursts / km-years. Then give an approximate standard error from the curvature: for a Poisson rate it is sqrt(total) / km-years.

# %%
def mle():
    # BEGIN SOLUTION
    res = optimize.minimize_scalar(lambda r: -loglik(r), bounds=(0.001, 2), method="bounded")
    exposure = KM * len(COUNTS)
    return float(res.x), float(COUNTS.sum() / exposure), float(np.sqrt(COUNTS.sum()) / exposure)
    # END SOLUTION


# %%
try:
    num, closed, se = mle()
except NotImplementedError:
    U.safety("mle")
    num, closed, se = 0.25, 0.25, float(np.sqrt(15) / 60)
print(f"numerical {num:.4f}, closed form {closed:.4f}, standard error {se:.4f}")
U.check(abs(num - closed) < 1e-3, "the numerical maximum matches total bursts over km-years")

# %% [markdown]
# ## B1 · Classify four files and decide where each may be processed (50')
#
# Read the national Data Classification Policy and the Personal Data Protection Law (addresses in the course README). Then, for each of Yanabi's real files below, give the classification level you would propose and say whether it may be uploaded to a hosted notebook service outside Yanabi, with one sentence of reason. Discuss your answers with Wurud Al-Samhan, the data protection officer, in the entry session; there is no automatic check.
#
# 1. Burst counts per district and year, with network length and pipe material.
# 2. Burst tickets: date, address of the burst, name and phone number of the caller.
# 3. The capital plan with costs per district, before the board approves it.
# 4. This course's synthetic files in `labs/data/`.

# %%
def classify():
    # BEGIN SOLUTION
    return {
        "burst counts per district": "internal operational data: process inside Yanabi's approved environment unless the data owner approves release",
        "burst tickets with callers": "contains personal data (names, phones, addresses): personal data law applies; never to a hosted notebook outside Yanabi without a lawful basis and approval",
        "capital plan before approval": "restricted until the board decides: inside Yanabi only",
        "course synthetic files": "invented data with no real person or asset: may be used in Colab",
    }
    # END SOLUTION


# %%
try:
    cls = classify()
except NotImplementedError:
    cls = {}
    print("B1 not done yet: bring your answers to the entry session")
for k, v in cls.items():
    print(f"- {k}: {v}")

# %% [markdown]
# ## ✱ Stretch · Two unknowns
#
# Maximise the negative binomial log likelihood of the same counts over the mean rate and the shape together. Why is the shape so poorly determined by five counts?

# %%
def stretch():
    # BEGIN SOLUTION
    def nll(p):
        rate, phi = np.exp(p)
        mu = rate * KM
        return -stats.nbinom.logpmf(COUNTS, phi, phi / (phi + mu)).sum()
    res = optimize.minimize(nll, x0=[np.log(0.25), 0.0])
    print("rate and shape at the maximum:", np.exp(res.x).round(3), "| five counts say little about extra variation")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Before Day 1
#
# Run `00_setup_check.ipynb`. Bring your answers to B1.

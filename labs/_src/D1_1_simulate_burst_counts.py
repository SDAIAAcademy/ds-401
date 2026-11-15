# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 1.1 · Simulate and recover a burst rate
#
# Day 1 · Session 1 (Models that generate data) · 25 min.
#
# Shabib Al-Ruqaish, Yanabi's pipe network engineer, keeps the burst records of 140 districts. Before any sampler, you write the story that produces a district's bursts as code, simulate five years from a rate you choose, and recover that rate on a grid of candidate values. Then you do the same for two real districts, one small and one large, and see how much five years of data can say.
#
# **AI policy.** AI-off: this is the day's first model. Write it yourself.
#
# **Reference state.** `data/districts.csv` and `data/bursts.csv`, loaded fresh. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · The generative story as a simulator (6')
# - [ ] 2 · The log likelihood on a grid of rates (5')
# - [ ] 3 · Prior times likelihood: the grid posterior (5')
# - [ ] 4 · A small district and a large one (6')
# - [ ] 5 · Where the grid stops working (3')
# - [ ] ✱ Stretch · Simulate with year-to-year variation and see the grid posterior become too sure

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

dist_tbl = U.districts()
p = U.panel()
print(f"{len(dist_tbl)} districts, {len(p)} district-years, {p['bursts_reported'].sum()} reported bursts")

# %% [markdown]
# ## 1 · The generative story as a simulator (6')
#
# One district, one year: bursts ~ Poisson(rate × km), with the rate in bursts per kilometre of main per year. Five years are five independent draws with the same rate.
#
# ✏️ **Exercise** · Write `simulate(rate, km, years, rng)` that returns an array of `years` burst counts.

# %%
def simulate(rate, km, years, rng):
    # BEGIN SOLUTION
    return rng.poisson(rate * km, size=years)
    # END SOLUTION


# %%
try:
    sim = simulate(0.20, 12.0, 5, U.rng(1))
except NotImplementedError:
    U.safety("simulate")
    sim = U.rng(1).poisson(0.20 * 12.0, size=5)
print("five simulated years:", sim, "total", sim.sum())
U.check(len(sim) == 5 and sim.min() >= 0, "five non-negative counts")

# %% [markdown]
# ## 2 · The log likelihood on a grid of rates (5')
#
# For every candidate rate on a grid, add up the log probability of each year's count. The rate that makes the data most probable is the peak.
#
# ✏️ **Exercise** · Write `log_lik(grid, counts, km)` returning one log likelihood per grid value. `stats.poisson.logpmf` does one count at a time; broadcast the grid against the counts.

# %%
GRID = np.linspace(0.001, 1.5, 3000)


def log_lik(grid, counts, km):
    # BEGIN SOLUTION
    return stats.poisson.logpmf(np.asarray(counts)[None, :], grid[:, None] * km).sum(axis=1)
    # END SOLUTION


# %%
try:
    ll = log_lik(GRID, sim, 12.0)
except NotImplementedError:
    U.safety("log_lik")
    ll = stats.poisson.logpmf(np.asarray(sim)[None, :], GRID[:, None] * 12.0).sum(axis=1)
mle = GRID[np.argmax(ll)]
print(f"peak of the likelihood: {mle:.3f} bursts per km-year (the simulated total over km-years: {sim.sum() / (12.0 * 5):.3f})")
U.check(abs(mle - sim.sum() / 60) < 0.002, "the peak is the total bursts over the kilometre-years")

# %% [markdown]
# ## 3 · Prior times likelihood: the grid posterior (5')
#
# Shabib's experience: about one burst per 10 km a year, rarely below one per 100 km or above 2 per km. A lognormal prior centred on 0.10 with a log-scale spread of 1 says that.
#
# ✏️ **Exercise** · Write `grid_posterior(grid, counts, km)` that returns posterior weights on the grid that add up to 1. Work on the log scale and subtract the maximum before exponentiating.

# %%
def log_prior(grid):
    return stats.norm.logpdf(np.log(grid), np.log(0.10), 1.0) - np.log(grid)  # lognormal density


def grid_posterior(grid, counts, km):
    # BEGIN SOLUTION
    lp = log_prior(grid) + log_lik(grid, counts, km)
    w = np.exp(lp - lp.max())
    return w / w.sum()
    # END SOLUTION


def summarise(grid, w):
    cdf = np.cumsum(w)
    return {"median": float(grid[np.searchsorted(cdf, 0.5)]),
            "low": float(grid[np.searchsorted(cdf, 0.05)]),
            "high": float(grid[np.searchsorted(cdf, 0.95)])}


# %%
try:
    w = grid_posterior(GRID, sim, 12.0)
except NotImplementedError:
    U.safety("grid_posterior")
    lp = log_prior(GRID) + stats.poisson.logpmf(np.asarray(sim)[None, :], GRID[:, None] * 12.0).sum(axis=1)
    w = np.exp(lp - lp.max()); w = w / w.sum()
s_sim = summarise(GRID, w)
print(f"posterior median {s_sim['median']:.3f}, 90% interval {s_sim['low']:.3f} to {s_sim['high']:.3f}; the rate you chose: 0.200")
U.check(abs(w.sum() - 1) < 1e-9 and s_sim["low"] < 0.20 < s_sim["high"], "the weights add up to 1 and the interval covers the rate you chose")

# %% [markdown]
# ## 4 · A small district and a large one (6')
#
# Two real districts with sensors (so every burst was recorded): the smallest rural network and the largest urban one.
#
# ✏️ **Exercise** · For each, compute the grid posterior of its rate from its five reported counts and its `network_km`. Return a DataFrame with one row per district: `district_id`, `km`, `bursts`, `median`, `low`, `high`, `width` (high minus low).

# %%
small = dist_tbl[(dist_tbl["kind"] == "rural") & dist_tbl["sensors"]].nsmallest(1, "network_km")["district_id"].item()
large = dist_tbl[(dist_tbl["kind"] == "urban") & dist_tbl["sensors"]].nlargest(1, "network_km")["district_id"].item()


def two_districts(ids):
    # BEGIN SOLUTION
    rows = []
    for did in ids:
        g = p[p["district_id"] == did]
        km = float(g["network_km"].iloc[0])
        s = summarise(GRID, grid_posterior(GRID, g["bursts_reported"].to_numpy(), km))
        rows.append({"district_id": did, "km": km, "bursts": int(g["bursts_reported"].sum()), **s, "width": s["high"] - s["low"]})
    return pd.DataFrame(rows)
    # END SOLUTION


# %%
try:
    two = two_districts([small, large])
except NotImplementedError:
    two = U.reference("d1_1", "two")
U.keep("d1_1", "two", two)
print(two.round(3).to_string(index=False))
tr = U.truth()["true_rate_per_km_year"]
print(f"true rates (known only because the data are synthetic): {small} {tr[small]:.3f}, {large} {tr[large]:.3f}")
U.check(two["width"].iloc[0] > 5 * two["width"].iloc[1], "the small district's interval is many times wider than the large one's")

# %% [markdown]
# **Look at** the two widths. Same prior, same five years: the large district's data overwhelm the prior; the small one's posterior still leans on it. Day 2 is about borrowing strength for districts like the small one.
#
# ## 5 · Where the grid stops working (3')
#
# ✏️ **Exercise** · A grid of 100 points per parameter. How many points does a model with one rate per district need for all 140 districts at once? Return the number as a power of ten (an integer exponent).

# %%
def grid_points_exponent(points_per_axis=100, parameters=140):
    # BEGIN SOLUTION
    return int(round(parameters * np.log10(points_per_axis)))
    # END SOLUTION


# %%
try:
    e = grid_points_exponent()
except NotImplementedError:
    U.safety("grid_points_exponent")
    e = 280
print(f"a 140-dimensional grid of 100 points per axis has 10^{e} points")
U.check(e == 280, "10^280 points: no computer evaluates that; session 3 samples instead")

# %% [markdown]
# ## ✱ Stretch · Year-to-year variation
#
# Simulate 5 years where each year's rate is itself drawn around 0.20 (a gamma with shape 6), recover the rate with the Poisson grid, and repeat 200 times. How often does the 90% interval cover 0.20?

# %%
def stretch():
    # BEGIN SOLUTION
    r = U.rng(7)
    hits = 0
    for _ in range(U.pick(200, 60)):
        yr = r.gamma(6.0, 0.20 / 6.0, size=5)
        c = r.poisson(yr * 12.0)
        s = summarise(GRID, grid_posterior(GRID, c, 12.0))
        hits += s["low"] < 0.20 < s["high"]
    print(f"90% intervals that cover the rate: {hits / U.pick(200, 60):.0%} (a calibrated model would give about 90%)")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note(f"Checked: the grid recovers a rate you chose, and the small district {small} has an interval {two['width'].iloc[0] / two['width'].iloc[1]:.0f} times wider than {large}.",
       "Not checked yet: whether the prior is sensible for every district. Lab 1.2 simulates from it.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the small district's posterior depend more on the prior than the large one's?
#    <details><summary>Answer</summary>Its five years hold a handful of bursts, so the likelihood is flat over a wide range of rates and the prior shapes the result; the large district's likelihood is sharp.</details>
# 2. The peak of the likelihood was the total bursts over the kilometre-years. Why?
#    <details><summary>Answer</summary>For Poisson counts with exposure, the maximum-likelihood rate is the total count divided by the total exposure.</details>
# 3. Why can't the 140-district model be fitted on a grid?
#    <details><summary>Answer</summary>The grid grows exponentially with the number of parameters: 100 points per axis and 140 axes is 10^280 points.</details>
#
# ## Carry forward
#
# The lognormal prior of task 3 is the starting point of Lab 1.2, which tests it by simulation.

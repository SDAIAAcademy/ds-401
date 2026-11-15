# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 4.1 · Replace, repair or wait
#
# Day 4 · Session 1 (Loss, utility and the decision table) · 25 min.
#
# Juhaina Al-Zahim has costed three actions for every district over a 20-year horizon. You compute the cost of each action in every posterior draw, choose the action with the lowest expected cost for each district, and compare with what a report of point estimates would have recommended.
#
# **AI policy.** AI-off: this is the day's first model of the decision. Write it yourself.
#
# **Reference state.** `data/posterior/rates.npz` (2,000 draws of each district's true burst rate from the Day 3 model: negative binomial, material and age, reporting term) and `data/costs.json`. Nothing from Day 3 is needed.
#
# **Tasks**
# - [ ] 1 · Load the posterior and the costs (3')
# - [ ] 2 · The cost of each action in each draw (7')
# - [ ] 3 · Expected cost and the best action per district (6')
# - [ ] 4 · Plug-in decisions against expected utility (6')
# - [ ] 5 · Where the two disagree (3')
# - [ ] ✱ Stretch · Decide on the 90th percentile of cost instead of the mean, and say who would choose that

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
import yanabi as U
print("mode:", U.MODE)

ACTIONS = ["wait", "repair", "replace"]

# %% [markdown]
# ## 1 · Load the posterior and the costs (3')
#
# ✏️ **Exercise** · Load the rates with `U.posterior_rates()`, the costs with `U.costs()`, and the districts in the same order as the posterior's columns. Print the number of draws and the actions with their cost per km, reduction and years.

# %%
def load():
    # BEGIN SOLUTION
    rate, ids = U.posterior_rates()
    c = U.costs()
    d = U.districts().set_index("district_id").loc[ids]
    for a in ACTIONS:
        print(a, c["actions"][a])
    return rate, ids, c, d
    # END SOLUTION


# %%
try:
    rate, ids, c, dist_tbl = load()
except NotImplementedError:
    U.safety("load")
    rate, ids = U.posterior_rates()
    c = U.costs()
    dist_tbl = U.districts().set_index("district_id").loc[ids]
km = dist_tbl["network_km"].to_numpy()
print(f"{rate.shape[0]} draws × {rate.shape[1]} districts; cost per burst SAR {c['cost_per_burst']:,}; horizon {c['horizon_years']} years")
U.check(rate.shape[1] == 140 and list(dist_tbl.index) == ids, "the districts are in the posterior's order")

# %% [markdown]
# ## 2 · The cost of each action in each draw (7')
#
# Over the horizon of H years, for a district of km kilometres with true rate r:
#
# cost(action) = cost_per_km × km + r × km × cost_per_burst × (H − years × reduction)
#
# The second term is the cost of the bursts that still happen: an action removes `reduction` of the bursts for `years` years.
#
# ✏️ **Exercise** · Return an array of shape (3, draws, 140): the cost of each action (in the order of `ACTIONS`) in each draw for each district.

# %%
def costs_per_draw(r):
    # BEGIN SOLUTION
    H, B = c["horizon_years"], c["cost_per_burst"]
    out = []
    for a in ACTIONS:
        A = c["actions"][a]
        out.append(A["cost_per_km"] * km + r * km * B * (H - A["years"] * A["reduction"]))
    return np.stack(out)
    # END SOLUTION


# %%
try:
    C = costs_per_draw(rate)
except NotImplementedError:
    U.safety("costs_per_draw")
    H, B = c["horizon_years"], c["cost_per_burst"]
    C = np.stack([c["actions"][a]["cost_per_km"] * km + rate * km * B * (H - c["actions"][a]["years"] * c["actions"][a]["reduction"]) for a in ACTIONS])
print("cost array:", C.shape)
U.check(C.shape == (3, rate.shape[0], 140) and np.all(C[0] >= 0), "a cost for every action, draw and district")

# %% [markdown]
# ## 3 · Expected cost and the best action per district (6')
#
# ✏️ **Exercise** · Return a DataFrame indexed by district with the expected cost of each action (columns `wait`, `repair`, `replace`, in SAR), the best action (`best`) and its expected saving against waiting (`saving`).

# %%
def decide(C):
    # BEGIN SOLUTION
    ec = pd.DataFrame(C.mean(axis=1).T, index=ids, columns=ACTIONS)
    ec["best"] = ec[ACTIONS].idxmin(axis=1)
    ec["saving"] = ec["wait"] - ec[ACTIONS].min(axis=1)
    return ec
    # END SOLUTION


# %%
try:
    eu = decide(C)
except NotImplementedError:
    eu = U.reference("d4_1", "eu")
U.keep("d4_1", "eu", eu)
print(eu["best"].value_counts().to_string())
print(pd.crosstab(dist_tbl["kind"], eu["best"]).to_string())
print(f"expected saving of the best actions, all districts: SAR {eu['saving'].sum() / 1e6:.1f} million")
U.check((eu["best"] == "replace").sum() > 5 and (eu["best"] == "wait").sum() > 50, "a few districts to re-pipe, many to leave")

# %% [markdown]
# ## 4 · Plug-in decisions against expected utility (6')
#
# A report of point estimates gives one rate per district, usually the posterior median. Deciding on it is a plug-in decision.
#
# ✏️ **Exercise** · Choose the best action per district from the costs evaluated at the posterior **median** rate, and from the costs at the posterior **mean** rate. Return a dict with the number of districts where each plug-in choice differs from the expected-utility choice.

# %%
def plug_in():
    # BEGIN SOLUTION
    out = {}
    for name, r in [("median", np.median(rate, axis=0)), ("mean", rate.mean(axis=0))]:
        best = np.array(ACTIONS)[costs_per_draw(r[None, :])[:, 0, :].argmin(axis=0)]
        out[name] = best
    return out
    # END SOLUTION


# %%
try:
    pi = plug_in()
except NotImplementedError:
    pi = {k: np.array(v) for k, v in U.reference("d4_1", "plug_in").items()}
U.keep("d4_1", "plug_in", {k: list(v) for k, v in pi.items()})
diff = {k: int((v != eu["best"].to_numpy()).sum()) for k, v in pi.items()}
print("districts where the plug-in choice differs from expected utility:", diff)
U.check(diff["median"] > 0 and diff["mean"] == 0, "the median plug-in disagrees with expected utility; the mean plug-in does not")

# %% [markdown]
# **Look at** the mean row. Every cost here is a straight line in the rate, so the expected cost is the cost at the mean rate: the mean plug-in gives the same decision. The median does not, because the posterior of a rate is skewed: its median is below its mean. Lab 4.2's budget and Lab 4.3's survey are where the whole distribution matters even with straight-line costs.
#
# ## 5 · Where the two disagree (3')
#
# ✏️ **Exercise** · Return the districts where the median plug-in and expected utility disagree, with their kind, sensors, both choices and the ratio of posterior mean to median.

# %%
def disagreements():
    # BEGIN SOLUTION
    m = pi["median"] != eu["best"].to_numpy()
    return pd.DataFrame({"kind": dist_tbl["kind"].to_numpy()[m], "sensors": dist_tbl["sensors"].to_numpy()[m],
                         "median_choice": pi["median"][m], "expected_utility": eu["best"].to_numpy()[m],
                         "mean_over_median": (rate.mean(axis=0) / np.median(rate, axis=0))[m]}, index=np.array(ids)[m])
    # END SOLUTION


# %%
try:
    dis = disagreements()
except NotImplementedError:
    dis = U.reference("d4_1", "disagreements")
U.keep("d4_1", "disagreements", dis)
print(dis.round(3).to_string())
U.check((dis["mean_over_median"] > 1).all(), "they are districts whose posterior is skewed upwards: the median understates their expected bursts")

# %% [markdown]
# ## ✱ Stretch · A cautious decision maker
#
# Choose, for each district, the action with the lowest 90th percentile of cost. Who in Yanabi would want that rule, and what does it cost on average?

# %%
def stretch():
    # BEGIN SOLUTION
    q90 = np.quantile(C, 0.9, axis=1)
    best90 = np.array(ACTIONS)[q90.argmin(axis=0)]
    changed = best90 != eu["best"].to_numpy()
    extra = C.mean(axis=1)[q90.argmin(axis=0), np.arange(140)].sum() - C.mean(axis=1).min(axis=0).sum()
    print(f"the cautious rule changes {changed.sum()} districts and costs SAR {extra / 1e6:.2f} million more on average")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the cost of every action in every draw, then the expectation.",
       "Found: a point-estimate report would leave some skewed (mostly rural) districts under-served.",
       "Carry forward: the expected costs and savings; Lab 4.2 fits them into the capital budget.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why compute the cost in each draw and then average, rather than average the rate first?
#    <details><summary>Answer</summary>In general the expected cost is not the cost at the expected rate; it is only here because the costs are straight lines in the rate. Averaging last is always right.</details>
# 2. Why does the median plug-in under-serve some districts?
#    <details><summary>Answer</summary>Their posterior is skewed upwards, so the median rate is below the mean and understates the bursts to expect.</details>
# 3. What does the 90th-percentile rule express?
#    <details><summary>Answer</summary>Aversion to bad outcomes: it pays more on average to avoid the worst cases, a choice for the risk owner, not the analyst.</details>
#
# ## Carry forward
#
# Lab 4.2 starts from the same supplied posterior and costs.

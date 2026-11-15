# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 4.3 · The price of the leak survey
#
# Day 4 · Session 3 (What more data is worth) · 25 min.
#
# Ouda Al-Shahwan brings an offer: an acoustic leak-detection contractor will listen to every main of a district without sensors for one season and return its true burst-prone rate with about 15% error, for SAR 45,000 a district. Survey all 76 such districts? Some? None? You price the survey by what it could change in the decisions of Lab 4.1, before buying it.
#
# **AI policy.** The tutor Mirsad may explain; the verdict is yours.
#
# **Reference state.** `data/posterior/rates.npz` and `data/costs.json`; the decision of Lab 4.1 is recomputed below. To keep the lesson clear, each district is decided on its own here (no budget); the stretch item discusses the budget.
#
# **Tasks**
# - [ ] 1 · The districts the survey covers (3')
# - [ ] 2 · Perfect information, district by district (6')
# - [ ] 3 · Simulate the survey before buying it (8')
# - [ ] 4 · The survey's value against its price (5')
# - [ ] 5 · The verdict (3')
# - [ ] ✱ Stretch · Buy the survey only where its value exceeds its price, and price that smaller offer

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
rate, ids = U.posterior_rates()
c = U.costs()
dist_tbl = U.districts().set_index("district_id").loc[ids]
km = dist_tbl["network_km"].to_numpy()
H, B = c["horizon_years"], c["cost_per_burst"]
C = np.stack([c["actions"][a]["cost_per_km"] * km + rate * km * B * (H - c["actions"][a]["years"] * c["actions"][a]["reduction"]) for a in ACTIONS])
EC = C.mean(axis=1)                    # expected cost of each action, per district
now = EC.min(axis=0)                   # expected cost of today's best action
SURVEY = c["survey"]
print("the offer:", SURVEY["what"], f"| SAR {SURVEY['price_per_district']:,} a district")

# %% [markdown]
# ## 1 · The districts the survey covers (3')
#
# ✏️ **Exercise** · Return the positions (column indices) of the districts without sensors, and print how many there are and what surveying all of them would cost.

# %%
def covered():
    # BEGIN SOLUTION
    return np.where(~dist_tbl["sensors"].to_numpy())[0]
    # END SOLUTION


# %%
try:
    cov = covered()
except NotImplementedError:
    U.safety("covered")
    cov = np.where(~dist_tbl["sensors"].to_numpy())[0]
full_price = len(cov) * SURVEY["price_per_district"]
print(f"{len(cov)} districts without sensors; the full survey costs SAR {full_price / 1e6:.2f} million")
U.check(len(cov) == 76, "76 districts without sensors")

# %% [markdown]
# ## 2 · Perfect information, district by district (6')
#
# If you knew a district's true rate, you would always pick the cheapest action for that rate. The expected value of perfect information (EVPI) is how much cheaper that is on average than deciding now:
#
# EVPI = min over actions of E[cost] − E[min over actions of cost]
#
# It is an upper bound: no survey is worth more.
#
# ✏️ **Exercise** · Return the EVPI of every covered district as a Series indexed by district id.

# %%
def evpi():
    # BEGIN SOLUTION
    v = now - C.min(axis=0).mean(axis=0)
    return pd.Series(v[cov], index=np.array(ids)[cov])
    # END SOLUTION


# %%
try:
    ev_pi = evpi()
except NotImplementedError:
    ev_pi = U.reference("d4_3", "evpi")
U.keep("d4_3", "evpi", ev_pi)
print(f"EVPI over the 76 districts: SAR {ev_pi.sum() / 1e6:.2f} million; districts with EVPI above the price: {(ev_pi > SURVEY['price_per_district']).sum()}")
U.check(ev_pi.min() >= -1e-6 and ev_pi.sum() < full_price, "perfect information is worth less than the full survey costs")

# %% [markdown]
# Already a verdict on the full offer: even perfect knowledge of all 76 rates would save less than the survey costs. The question becomes which districts.
#
# ## 3 · Simulate the survey before buying it (8')
#
# Preposterior analysis: for each posterior draw s, treat its rate as the truth, simulate what the survey would report (log rate plus noise with sd 0.15), update the posterior by weighting every draw by the likelihood of that report, pick the best action under the updated posterior, and record its cost under the truth of draw s. Compare with today's action in the same draw.
#
# ✏️ **Exercise** · Write `evsi(j)` that returns the expected value of the survey for district position j: the average over draws of (cost of today's action − cost of the action chosen after the survey), both under the draw's truth.

# %%
SD = SURVEY["log_sd"]
L = np.log(rate)
N_OUTER = U.pick(2000, 600)
outer = U.rng(43).choice(rate.shape[0], N_OUTER, replace=False)


def evsi(j):
    # BEGIN SOLUTION
    a_now = EC[:, j].argmin()
    obs = L[outer, j] + U.rng(44 + j).normal(0, SD, N_OUTER)
    lw = -0.5 * ((obs[:, None] - L[None, :, j]) / SD) ** 2          # (outer, draws)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    post_cost = w @ C[:, :, j].T                                       # (outer, actions)
    a_after = post_cost.argmin(axis=1)
    return float(np.mean(C[a_now, outer, j] - C[a_after, outer, j]))
    # END SOLUTION


# %%
try:
    ev_si = pd.Series([evsi(j) for j in cov], index=np.array(ids)[cov])
except NotImplementedError:
    ev_si = U.reference("d4_3", "evsi")
U.keep("d4_3", "evsi", ev_si)
print(f"EVSI over the 76 districts: SAR {ev_si.sum() / 1e6:.2f} million (EVPI {ev_pi.sum() / 1e6:.2f})")
U.check((ev_si <= ev_pi + 0.05 * ev_pi.max()).all(), "no district's survey is worth more than perfect information, up to simulation noise")

# %% [markdown]
# ## 4 · The survey's value against its price (5')
#
# ✏️ **Exercise** · Return a DataFrame of the covered districts with `kind`, `km`, `evpi`, `evsi`, `worth_it` (EVSI above the price per district), sorted by EVSI.

# %%
def against_price():
    # BEGIN SOLUTION
    t = pd.DataFrame({"kind": dist_tbl["kind"].iloc[cov].to_numpy(), "km": km[cov], "evpi": ev_pi.to_numpy(), "evsi": ev_si.to_numpy()},
                     index=ev_si.index)
    t["worth_it"] = t["evsi"] > SURVEY["price_per_district"]
    return t.sort_values("evsi", ascending=False)
    # END SOLUTION


# %%
try:
    tbl = against_price()
except NotImplementedError:
    tbl = U.reference("d4_3", "table")
U.keep("d4_3", "table", tbl)
print(tbl.head(12).round(0).to_string())
n_yes = int(tbl["worth_it"].sum())
print(f"\ndistricts where the survey is worth its price: {n_yes}; their EVSI SAR {tbl.loc[tbl['worth_it'], 'evsi'].sum() / 1e6:.2f} million for SAR {n_yes * SURVEY['price_per_district'] / 1e6:.2f} million")
U.check(0 < n_yes < len(cov), "the survey is worth buying for some districts, not for all")

# %% [markdown]
# ## 5 · The verdict (3')
#
# ✏️ **Exercise** · Write the verdict for Nouran in three sentences: on the full offer, on a targeted survey, and on what the value depends on.

# %%
def verdict():
    # BEGIN SOLUTION
    yes = tbl[tbl["worth_it"]]
    return [f"The full survey of 76 districts costs SAR {full_price / 1e6:.2f} million and could save at most SAR {ev_si.sum() / 1e6:.2f} million in better decisions: decline it as offered.",
            f"A survey of the {len(yes)} districts where it is worth more than its price (mostly {yes['kind'].mode().iloc[0]}) is expected to save SAR {yes['evsi'].sum() / 1e6:.2f} million for SAR {len(yes) * SURVEY['price_per_district'] / 1e6:.2f} million: propose it.",
            "The value comes from districts close to a decision threshold with uncertain rates; it would fall if the audit prior on reporting were sharpened first, and it assumes the contractor's 15% error."]
    # END SOLUTION


# %%
try:
    v = verdict()
except NotImplementedError:
    v = U.reference("d4_3", "verdict")
U.keep("d4_3", "verdict", v)
for s in v:
    print("-", s)
U.check(len(v) == 3, "three sentences")

# %% [markdown]
# ## ✱ Stretch · The targeted offer, and the budget
#
# Price the targeted survey as a counter-offer to Ouda. Then discuss: Lab 4.2's plan is limited by the budget; can a survey that changes no funded decision still be worth its price?

# %%
def stretch():
    # BEGIN SOLUTION
    yes = tbl[tbl["worth_it"]]
    net = yes["evsi"].sum() - len(yes) * SURVEY["price_per_district"]
    print(f"targeted survey: {len(yes)} districts, expected net value SAR {net / 1e6:.2f} million")
    print("under the budget, a survey is worth something only if it can move a district across the funding line; this lab's per-district values are an upper guide")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the survey priced by the decisions it could change, before buying it.",
       "Found: the full offer is not worth its price; a targeted one is.",
       "Carry forward: the verdict on the survey is the third part of the capstone memo.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is EVPI an upper bound on the value of the survey?
#    <details><summary>Answer</summary>No data can do better than knowing the true rate; a noisy survey changes fewer decisions.</details>
# 2. A district with a very uncertain rate has an EVSI near zero. How?
#    <details><summary>Answer</summary>Its best action is the same over almost all plausible rates, so no survey result would change the decision.</details>
# 3. Why simulate the survey instead of buying it and seeing?
#    <details><summary>Answer</summary>The decision to buy must be made before the results exist; the posterior already says what the results could be.</details>
#
# ## Carry forward
#
# Lab 4.4 reviews someone else's analysis, including how they priced further data.

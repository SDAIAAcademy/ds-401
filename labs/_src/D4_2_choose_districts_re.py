# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 4.2 · A plan within the budget
#
# Day 4 · Session 2 (Choosing under a budget) · 25 min.
#
# Lab 4.1's best actions would spend more than Juhaina's capital budget. You rank the actions by expected saving per riyal invested, fill the budget, and report the plan as Nouran needs it: what it costs, how many bursts it avoids with an interval, who carries the remaining risk, and which districts the model knows least about.
#
# **AI policy.** The tutor Mirsad may explain; it writes no code for the plan.
#
# **Reference state.** `data/posterior/rates.npz` and `data/costs.json`; the costs of Lab 4.1 are recomputed below, so Lab 4.1 need not be finished.
#
# **Tasks**
# - [ ] 1 · The plan without a budget, and its cost (4')
# - [ ] 2 · Benefit per riyal (5')
# - [ ] 3 · Fill the budget (6')
# - [ ] 4 · Avoided bursts with an interval (6')
# - [ ] 5 · Who carries the remaining risk (4')
# - [ ] ✱ Stretch · Solve the budget exactly by dynamic programming and compare with the greedy plan

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
invest = {a: c["actions"][a]["cost_per_km"] * km for a in ACTIONS}
avoided_per_draw = {a: rate * km * c["actions"][a]["years"] * c["actions"][a]["reduction"] for a in ACTIONS}  # bursts avoided
saving = {a: (avoided_per_draw[a] * B).mean(axis=0) - invest[a] for a in ACTIONS}  # expected net saving against waiting
BUDGET = c["capital_budget"]
print(f"capital budget: SAR {BUDGET / 1e6:.0f} million")

# %% [markdown]
# ## 1 · The plan without a budget, and its cost (4')
#
# ✏️ **Exercise** · For each district take the action with the largest expected net saving (waiting saves 0). Return the total investment of that plan and the number of districts per action.

# %%
def unconstrained():
    # BEGIN SOLUTION
    s = np.stack([saving[a] for a in ACTIONS])
    best = np.array(ACTIONS)[s.argmax(axis=0)]
    total = sum(invest[a][best == a].sum() for a in ACTIONS)
    return best, float(total)
    # END SOLUTION


# %%
try:
    best_u, spend_u = unconstrained()
except NotImplementedError:
    best_u, spend_u = (np.array(v) if i == 0 else v for i, v in enumerate(U.reference("d4_2", "unconstrained")))
U.keep("d4_2", "unconstrained", [list(best_u), spend_u])
print(pd.Series(best_u).value_counts().to_string())
print(f"investment without a budget: SAR {spend_u / 1e6:.1f} million against a budget of {BUDGET / 1e6:.0f}")
U.check(spend_u > BUDGET, "the unconstrained plan does not fit the budget")

# %% [markdown]
# ## 2 · Benefit per riyal (5')
#
# Rank the steps of investment by expected net saving per riyal. A district can take two steps: from waiting to repair, then from repair to replacement (the extra cost buys the extra saving). Where replacement is worth more per riyal than repair, the district's step is straight from waiting to replacement.
#
# ✏️ **Exercise** · Return a DataFrame of steps with `district_id`, `step` (`repair`, `replace` or `repair->replace`), `invest` and `saving` (both increments) and `ratio`, keeping only steps with a positive saving, sorted by `ratio`, largest first.

# %%
def options():
    # BEGIN SOLUTION
    rows = []
    for j, did in enumerate(ids):
        r_c, r_s = invest["repair"][j], saving["repair"][j]
        p_c, p_s = invest["replace"][j], saving["replace"][j]
        if r_s > 0 and r_s / r_c >= (p_s - r_s) / (p_c - r_c):
            rows.append({"district_id": did, "step": "repair", "invest": r_c, "saving": r_s})
            if p_s > r_s:
                rows.append({"district_id": did, "step": "repair->replace", "invest": p_c - r_c, "saving": p_s - r_s})
        elif p_s > 0:
            rows.append({"district_id": did, "step": "replace", "invest": p_c, "saving": p_s})
    t = pd.DataFrame(rows)
    t["ratio"] = t["saving"] / t["invest"]
    return t.sort_values("ratio", ascending=False).reset_index(drop=True)
    # END SOLUTION


# %%
try:
    opts = options()
except NotImplementedError:
    opts = U.reference("d4_2", "options")
U.keep("d4_2", "options", opts)
print(opts.head(8).round(2).to_string(index=False))
U.check(len(opts) > 0 and opts["ratio"].is_monotonic_decreasing, "steps sorted by saving per riyal")

# %% [markdown]
# ## 3 · Fill the budget (6')
#
# Go down the list and take each step that fits in what is left of the budget; an upgrade to replacement only for a district already given repair. (A greedy rule: fast and close to the best, not always the best; the stretch checks it.)
#
# ✏️ **Exercise** · Return the plan as a dict `district_id -> action` (`repair` or `replace`), its total investment and its total expected net saving.

# %%
def fill_budget():
    # BEGIN SOLUTION
    plan, spent, saved = {}, 0.0, 0.0
    for _, o in opts.iterrows():
        did, step = o["district_id"], o["step"]
        if spent + o["invest"] > BUDGET:
            continue
        if step == "repair->replace":
            if plan.get(did) != "repair":
                continue
            plan[did] = "replace"
        else:
            plan[did] = step
        spent += o["invest"]
        saved += o["saving"]
    return plan, spent, saved
    # END SOLUTION


# %%
try:
    plan, spent, saved = fill_budget()
except NotImplementedError:
    plan, spent, saved = U.reference("d4_2", "plan")
U.keep("d4_2", "plan", [plan, spent, saved])
print(pd.Series(plan).value_counts().to_string())
print(f"plan: {len(plan)} districts, SAR {spent / 1e6:.1f} million invested, expected net saving SAR {saved / 1e6:.1f} million")
U.check(spent <= BUDGET and spent > 0.9 * BUDGET, "the plan uses most of the budget without exceeding it")

# %% [markdown]
# ## 4 · Avoided bursts with an interval (6')
#
# Bursts avoided over the horizon differ from draw to draw: the plan's uncertainty is the posterior's.
#
# ✏️ **Exercise** · Return, per draw, the total bursts the plan avoids, and print its median and 90% interval.

# %%
def avoided():
    # BEGIN SOLUTION
    tot = np.zeros(rate.shape[0])
    for did, a in plan.items():
        tot += avoided_per_draw[a][:, ids.index(did)]
    return tot
    # END SOLUTION


# %%
try:
    av = avoided()
except NotImplementedError:
    av = np.array(U.reference("d4_2", "avoided"))
U.keep("d4_2", "avoided", av[:200])
lo, md_, hi = np.quantile(av, [0.05, 0.5, 0.95])
print(f"bursts avoided over {H} years: {md_:.0f} (90% interval {lo:.0f} to {hi:.0f})")
U.check(hi > lo * 1.1, "the avoided bursts carry real uncertainty")

# %% [markdown]
# ## 5 · Who carries the remaining risk (4')
#
# ✏️ **Exercise** · Return two tables: the 8 unfunded districts with the most expected bursts over the horizon (who carries the remaining risk), and the 8 districts with the widest posterior relative to their median (what the model knows least about), each with kind, sensors and whether the plan funds them.

# %%
def remaining_risk():
    # BEGIN SOLUTION
    exp_bursts = (rate * km).mean(axis=0) * H
    lo_, hi_ = np.quantile(rate, [0.05, 0.95], axis=0)
    t = pd.DataFrame({"kind": dist_tbl["kind"], "sensors": dist_tbl["sensors"], "funded": [plan.get(i, "no") for i in ids],
                      "expected_bursts": exp_bursts, "relative_width": (hi_ - lo_) / np.median(rate, axis=0)}, index=ids)
    unfunded = t[t["funded"] == "no"].sort_values("expected_bursts", ascending=False).head(8)
    unknown = t.sort_values("relative_width", ascending=False).head(8)
    return unfunded, unknown
    # END SOLUTION


# %%
try:
    unfunded, unknown = remaining_risk()
except NotImplementedError:
    unfunded, unknown = U.reference("d4_2", "unfunded"), U.reference("d4_2", "unknown")
U.keep("d4_2", "unfunded", unfunded)
U.keep("d4_2", "unknown", unknown)
print("unfunded, most expected bursts:\n" + unfunded.round(2).to_string())
print("\nleast known:\n" + unknown.round(2).to_string())
U.check((~unknown["sensors"]).mean() >= 0.5, "most of the least-known districts have no sensors")

# %% [markdown]
# **Look at** the two tables. The unfunded urban districts carry most of the bursts the plan leaves; their residents and the repair crews carry that risk. The least-known districts are mostly rural without sensors: the plan's choices for them rest on the audit prior, which is what Lab 4.3's survey would sharpen.
#
# ## ✱ Stretch · The exact optimum
#
# Solve the same problem exactly: a dynamic programme over budget units of SAR 4,000 (every investment is a multiple of it), where each district chooses wait, repair or replace. How much more does the exact plan save than the greedy one?

# %%
def stretch():
    # BEGIN SOLUTION
    unit = 4_000  # every investment is a multiple of SAR 4,000 (km has one decimal), so this is exact
    cap = int(BUDGET // unit)
    best = np.zeros(cap + 1)
    for j in range(len(ids)):
        new = best.copy()
        for a in ["repair", "replace"]:
            w = int(round(invest[a][j] / unit))
            if saving[a][j] > 0 and w <= cap:
                new[w:] = np.maximum(new[w:], best[:cap + 1 - w] + saving[a][j])
        best = new
    print(f"exact optimum: SAR {best[-1] / 1e6:.2f} million; greedy: SAR {saved / 1e6:.2f} million")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: the plan fits the budget and carries its uncertainty in avoided bursts.",
       "Named: who carries the remaining risk, and where the model knows least.",
       "Carry forward: the plan; Lab 4.3 asks whether better data on the least-known districts is worth its price.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why rank by saving per riyal rather than by saving?
#    <details><summary>Answer</summary>The budget is the constraint: a district that saves less but costs far less can give more saving per riyal spent.</details>
# 2. Why report avoided bursts with an interval?
#    <details><summary>Answer</summary>The rates are uncertain, so the plan's effect is too; a single number hides how wrong it could be.</details>
# 3. Who carries the remaining risk?
#    <details><summary>Answer</summary>The residents and crews of the districts the plan leaves, named in the memo with their expected bursts.</details>
#
# ## Carry forward
#
# The memo of the capstone reports the same four things: the plan, its cost, its expected effect with an interval, and the remaining risk.

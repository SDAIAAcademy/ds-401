# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Lab 4.4 · Review an analysis, choose a brief
#
# Day 4 · Session 4 (Reviewing a Bayesian analysis) · 25 min.
#
# Before the course, Yanabi paid a consultant, Sakhr Al-Nassar, for a burst analysis and a re-piping list. Nouran asks you to review it. You read it, go through the course's checklist item by item, and rank what you find by its effect on the decision: a flaw that changes which districts are re-piped matters more than one that only makes a table less tidy. Then you choose your capstone brief.
#
# **AI policy.** The tutor Mirsad may explain a checklist item; the review is yours.
#
# **Reference state.** `data/review/consultant_analysis.md`, `data/bursts.csv`, `data/districts.csv` and the supplied posterior `data/posterior/rates.npz` (to compare with); `capstone/briefs.json`.
#
# **Tasks**
# - [ ] 1 · Read the analysis (6')
# - [ ] 2 · The checklist, item by item (8')
# - [ ] 3 · Rank the flaws by their effect on the decision (6')
# - [ ] 4 · Choose your capstone brief (5')
# - [ ] ✱ Stretch · Rerun one of the consultant's claims and show the number changes

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
import yanabi as U
print("mode:", U.MODE)

report = (U.DATA / "review" / "consultant_analysis.md").read_text(encoding="utf-8")
CHECKLIST = {
    "priors": "Is every prior stated, with a reason, and checked by prior predictive simulation?",
    "computation": "Enough chains and draws; R-hat below 1.01; ESS; divergences explained or removed?",
    "structure": "Does the model's structure (pooling, predictors) suit districts with very unequal data?",
    "measurement": "Is the data what it claims to measure (under-reporting, changes in recording)?",
    "checking": "Posterior predictive checks on statistics that matter; comparison of alternatives?",
    "decision": "Decision from the whole posterior, with uncertainty, remaining risk and the value of more data?",
}

# %% [markdown]
# ## 1 · Read the analysis (6')
#
# ✏️ **Exercise** · Print the report, then return the list of its section headings (lines that start with `## `).

# %%
def read_report():
    # BEGIN SOLUTION
    print(report)
    return [line[3:] for line in report.splitlines() if line.startswith("## ")]
    # END SOLUTION


# %%
try:
    sections = read_report()
except NotImplementedError:
    U.safety("read_report")
    print(report)
    sections = [line[3:] for line in report.splitlines() if line.startswith("## ")]
print("sections:", sections)
U.check(len(sections) == 6, "six sections read")

# %% [markdown]
# ## 2 · The checklist, item by item (8')
#
# ✏️ **Exercise** · For each checklist item, return the flaw you find (one sentence, quoting the report) or `None` if the item is sound. Fill `review` as a dict `item -> sentence or None`.

# %%
def review_items():
    # BEGIN SOLUTION
    return {
        "priors": "Normal(0, 100) on a log rate is called uninformative but puts its weight on absurd rates, and no prior predictive check was run.",
        "computation": "Two short chains, R-hat judged against 1.1 instead of 1.01, and 37 divergences dismissed as a nuisance: the fit is not shown to be trustworthy.",
        "structure": "Each district is estimated separately: districts with 2 to 7 bursts on 1.5 to 4.7 km top the list on noise, and 13 districts with no recorded burst are declared safe.",
        "measurement": "Districts with and without sensors are treated alike, so unrecorded bursts in the 76 districts without sensors are read as low risk.",
        "checking": "R² between observed and fitted counts is offered as fit; no posterior predictive check of zeros or dispersion and no comparison with a pooled or negative binomial model.",
        "decision": "The ranking of posterior means is taken as the decision: no costs, no intervals, no remaining risk, and the survey dismissed because the model 'has converged'.",
    }
    # END SOLUTION


# %%
try:
    review = review_items()
except NotImplementedError:
    review = U.reference("d4_4", "review")
U.keep("d4_4", "review", review)
for k, v in review.items():
    print(f"{k:12s} {'sound' if v is None else v}")
found = [k for k, v in review.items() if v]
U.check(len(found) >= 4, f"at least four flaws found (LO6 asks for four of the six): {len(found)}")

# %% [markdown]
# ## 3 · Rank the flaws by their effect on the decision (6')
#
# A flaw's effect on the decision: would correcting it change which districts are re-piped? Compare the consultant's top six with the supplied posterior (`U.posterior_rates()`), which has pooling, predictors, the reporting term and the negative binomial.
#
# ✏️ **Exercise** · Return `ranking`, the found checklist items ordered from largest to smallest effect on the decision, and `evidence`, a DataFrame of the consultant's six top districts with their raw rate and the supplied posterior's median and 90% interval.

# %%
def rank_flaws():
    # BEGIN SOLUTION
    rate, ids = U.posterior_rates()
    d = U.districts().set_index("district_id")
    tot = U.bursts().groupby("district_id")["bursts_reported"].sum()
    top = ["D010", "D017", "D073", "D113", "D067", "D118"]
    ev = pd.DataFrame({"kind": d.loc[top, "kind"], "sensors": d.loc[top, "sensors"],
                       "raw_rate": (tot[top] / (d.loc[top, "network_km"] * 5)).round(3),
                       "posterior_median": [float(np.median(rate[:, ids.index(i)])) for i in top],
                       "low": [float(np.quantile(rate[:, ids.index(i)], 0.05)) for i in top],
                       "high": [float(np.quantile(rate[:, ids.index(i)], 0.95)) for i in top]}, index=top)
    ranking = ["structure", "measurement", "decision", "computation", "priors", "checking"]
    return [k for k in ranking if k in found], ev
    # END SOLUTION


# %%
try:
    ranking, evidence = rank_flaws()
except NotImplementedError:
    ranking, evidence = U.reference("d4_4", "ranking"), U.reference("d4_4", "evidence")
U.keep("d4_4", "ranking", ranking)
U.keep("d4_4", "evidence", evidence)
print("ranked by effect on the decision:", ranking)
print(evidence.round(3).to_string())
U.check(set(ranking) == set(found) and len(ranking) == len(found), "every flaw found is ranked once")

# %% [markdown]
# **Look at** D017. Its raw rate of 0.50 includes the trunk-main year of 2023; the pooled negative binomial model reads it as a district with a real but lower rate and one bad year. Re-piping a 58 km district costs SAR 23 million: more than half of the capital budget on one unusual year.
#
# ## 4 · Choose your capstone brief (5')
#
# Three briefs, each with synthetic data and known generating values: water losses (Yanabi's metered zones), school inspections and clinic waiting times.
#
# ✏️ **Exercise** · Print each brief's title, question and decision, then set `choice` to the key of the brief you will take and `why` to one sentence on the modelling problem you expect it to raise.

# %%
briefs = json.loads((Path("capstone") / "briefs.json").read_text(encoding="utf-8"))


def choose_brief():
    # BEGIN SOLUTION
    for k, b in briefs.items():
        print(f"[{k}] {b['title']}\n   {b['question']}\n   decision: {b['decision']}\n")
    return "school_inspections", "Schools with one visit on record are the small districts of this brief: the pooling across schools and regions decides who gets a follow-up."
    # END SOLUTION


# %%
try:
    choice, why = choose_brief()
except NotImplementedError:
    choice, why = U.reference("d4_4", "choice")
U.keep("d4_4", "choice", [choice, why])
print("capstone brief:", choice, "|", why)
U.check(choice in briefs, "a brief chosen from the three")

# %% [markdown]
# ## ✱ Stretch · Rerun a claim
#
# The report says the 13 districts with no recorded burst "need no action for the foreseeable future". Take them from `data/bursts.csv` and print the supplied posterior's median rate and the probability that each exceeds 0.2 bursts per km-year (the rate at which repair starts to pay).

# %%
def stretch():
    # BEGIN SOLUTION
    rate, ids = U.posterior_rates()
    tot = U.bursts().groupby("district_id")["bursts_reported"].sum()
    zero = tot[tot == 0].index.tolist()
    for i in zero:
        r = rate[:, ids.index(i)]
        print(f"{i}: median {np.median(r):.3f}, P(rate > 0.2) = {np.mean(r > 0.2):.2f}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %%
U.note("Checked: an analysis against priors, computation, structure, measurement, checking and the decision.",
       "Ranked: flaws by whether correcting them changes the districts chosen.",
       "Carry forward: the checklist is the follow-up task's template, and your capstone is reviewed against it by a peer.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does "estimate each district separately" rank above the flat prior?
#    <details><summary>Answer</summary>It decides the list: tiny districts with a few bursts top it by noise. The flat prior makes that worse but is not the main cause.</details>
# 2. The report says R-hat is below 1.1. What do you ask for?
#    <details><summary>Answer</summary>Four chains with full warm-up, R-hat below 1.01, ESS, and the divergences removed or explained, before any number is used.</details>
# 3. Is a review that finds six flaws a rejection of the analysis?
#    <details><summary>Answer</summary>It is a list of what to fix, ranked by effect on the decision; some of the consultant's list may survive the corrected model.</details>
#
# ## Carry forward
#
# The capstone starts from the brief you chose; its data and README template are in `capstone/`.

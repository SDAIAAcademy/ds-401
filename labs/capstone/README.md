# Capstone: a Bayesian model and a decision memo

You answer one of three briefs with a Bayesian model and a two-page decision memo, in a public GitHub repository of your own, and defend it in a 15-minute oral: an 8-minute presentation, then 7 minutes of questions in which the assessor changes a prior or a cost and you predict the effect on the decision before re-running, and a peer who reviewed your model asks their question.

## The briefs

Choose one in Lab 4.4. The datasets, in `capstone/data/`, are synthetic, written with a fixed seed by `labs/prepare_data.py` from known parameters. Model the data, not the generator: the assessor compares your estimates with those parameters, and a README that quotes them instead of estimating them scores nothing on the model criterion.

| Brief | Question | Data | Decision |
|---|---|---|---|
| Water losses | Yanabi can fit pressure-reducing valves in 12 of 60 metered zones. Each costs SAR 150,000 and cuts night-flow losses by 20%; water lost costs SAR 4 per cubic metre. Which zones, and with what expected saving and uncertainty? | `data/water_losses.csv`: 60 zones, 3 to 39 nights each, connections, pressure, loss per connection per day | 12 zones, or fewer |
| School inspections | An inspectorate can make 30 follow-up visits at SAR 9,000 each; a serious finding left unaddressed for a year is valued at SAR 4,000. Which schools, and how sure are you about the small ones with one visit on record? | `data/school_inspections.csv`: 200 schools in 8 regions, 1 to 4 visits each, pupils, serious findings | 30 visits |
| Clinic waiting times | A health cluster can add one doctor to 5 of 40 clinics. The target is a median wait under 30 minutes. Which clinics, and what is the probability that each meets the target afterwards? | `data/clinic_waits.csv`: 40 clinics in 4 cities, 15 to 159 patients each, doctors, wait in minutes | 5 clinics |

The costs and values are synthetic planning numbers. State in the memo any assumption you add (a horizon, the effect of an extra doctor, a discount rate).

## What the repository contains

- The model code and a notebook or script that reruns from the raw data to every number in the README, with a fixed seed, and its run log.
- The environment: `pyproject.toml` with `uv.lock`, or an equivalent with pinned versions.
- A README in this order:
  1. **Brief.** The question in the decision maker's words.
  2. **Model and priors.** The generative story, the structure (pooling, predictors), and every prior with its reason and a prior predictive check.
  3. **Computation.** Chains, draws, R-hat, effective sample sizes, divergences, and a short simulation-based calibration of your model.
  4. **Checking.** Posterior predictive checks on statistics that matter for the decision, and a comparison with at least one alternative model by approximate leave-one-out.
  5. **Decision.** The choice by expected utility, its cost, its expected effect with an interval, who carries the remaining risk and which units the model knows least about, and the value of one piece of further data.
  6. **Sensitivity.** What changes under one other defensible prior and one other cost.
- `memo.md`: the two-page decision memo for a manager who will not read the README.

## How it is scored

| Criterion | Weight | Full marks |
|---|---|---|
| Model and priors | 20 | Structure suited to unequal groups and defended; every prior with a reason and a prior predictive check |
| Computation | 20 | Diagnostics reported with numbers; a calibration that passes, or a found and fixed error |
| Checking | 25 | Predictive checks targeted at the decision; a comparison of models with the standard error of the difference |
| Decision | 25 | Expected utility over posterior draws; uncertainty, remaining risk and the value of further data stated |
| Reproducibility | 10 | A fresh clone reruns from the raw data to the same numbers |

The capstone is 60% of the course grade, the supervised check 20% and the oral with the peer-model review 20%. Passing is 70% overall with at least 60% on the capstone rubric and on the supervised check. AI assistance is allowed in the capstone; you must be able to defend every line in the oral.

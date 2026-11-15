<!-- academy:title -->
# Bayesian Modelling and Decisions under Uncertainty

النمذجة البايزية واتخاذ القرارات في ظل عدم اليقين

Data Science Track · SDAIA Academy · DS-401 · Level 400

REPLACE: one line under the title: what participants can do after the course

Participants build Bayesian models that support decisions and judge models built by others. On synthetic burst records from 140 districts of a fictional water utility, they test priors by simulation, fit and diagnose Hamiltonian Monte Carlo, pool small districts hierarchically, check and compare models by predictive accuracy, and verify code by simulation-based calibration. The last day turns posteriors into a replacement plan by expected utility and prices a proposed leak survey by its value of information.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ds-401/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Implement a Bayesian workflow from prior predictive simulation to sampler diagnosis, given a burst-count model with network length as exposure | Applied | a notebook with prior draws, diagnostics and a reason for each prior |
| LO2 | Design a hierarchical model for districts with very unequal data and defend the pooling it implies, given 140 districts, a third with fewer than five recorded bursts | Performance | no-, complete- and partial-pooling estimates with a written defence |
| LO3 | Evaluate competing models by posterior predictive checks and approximate leave-one-out cross-validation and choose one for the decision, given three candidate models of the same data | Performance | a comparison table with check plots and the reasoning |
| LO4 | Diagnose an error in a model implementation by parameter recovery and simulation-based calibration, given a model with a seeded coding error | Performance | calibration plots that locate the error and a corrected model that passes |
| LO5 | Decide a replacement plan by expected utility under the posterior and price further data by its value of information, given costs, a capital budget and an offered acoustic leak survey | Performance | a two-page memo with the plan, its expected loss and a verdict on the survey |
| LO6 | Review another analyst's Bayesian analysis for priors, computation, checking and reporting, given an analysis with six planted flaws | Performance | a review that finds at least four flaws and ranks them by effect on the decision |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | From priors to posteriors | Models that generate data; Priors you can defend; Fitting with Hamiltonian Monte Carlo; Reading the computation | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Pooling across districts | No, complete and partial pooling; Group-level predictors; The funnel and non-centred models; Under-reporting as measurement error | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Is the model good enough | Posterior predictive checks; Comparing models by predictive accuracy; Is the code right; How much the priors matter | [`slides/day3.html`](slides/day3.html) |
| Day 4 | From posterior to decision | Loss, utility and the decision table; Choosing under a budget; What more data is worth; Reviewing a Bayesian analysis | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Simulate burst counts and recover the rate on a grid | [`D1_1_simulate_burst_counts.ipynb`](labs/D1_1_simulate_burst_counts.ipynb) |
| 1.2 | Day 1 · Session 2 | Run prior predictive checks and reject an implausible prior | [`D1_2_run_prior_predictive.ipynb`](labs/D1_2_run_prior_predictive.ipynb) |
| 1.3 | Day 1 · Session 3 | Fit a Poisson model with a network-length offset | [`D1_3_fit_poisson_model.ipynb`](labs/D1_3_fit_poisson_model.ipynb) |
| 1.4 | Day 1 · Session 4 | Flag the two untrustworthy fits among four from their diagnostics | [`D1_4_flag_two_untrustworthy.ipynb`](labs/D1_4_flag_two_untrustworthy.ipynb) |
| 2.1 | Day 2 · Session 1 | Fit all three and plot shrinkage for rural districts | [`D2_1_fit_all_three.ipynb`](labs/D2_1_fit_all_three.ipynb) |
| 2.2 | Day 2 · Session 2 | Add pipe material and age and compare shrinkage | [`D2_2_add_pipe_material.ipynb`](labs/D2_2_add_pipe_material.ipynb) |
| 2.3 | Day 2 · Session 3 | Repair a divergent model by non-centred parameterisation | [`D2_3_repair_divergent_model.ipynb`](labs/D2_3_repair_divergent_model.ipynb) |
| 2.4 | Day 2 · Session 4 | Add a reporting-rate term for districts without sensors | [`D2_4_add_reporting_rate.ipynb`](labs/D2_4_add_reporting_rate.ipynb) |
| 3.1 | Day 3 · Session 1 | Expose Poisson misfit on zeros, then fit a negative binomial | [`D3_1_expose_poisson_misfit.ipynb`](labs/D3_1_expose_poisson_misfit.ipynb) |
| 3.2 | Day 3 · Session 2 | Compare three models by approximate leave-one-out and inspect influential districts | [`D3_2_compare_three_models.ipynb`](labs/D3_2_compare_three_models.ipynb) |
| 3.3 | Day 3 · Session 3 | Locate a seeded error by a short simulation-based calibration | [`D3_3_locate_seeded_error.ipynb`](labs/D3_3_locate_seeded_error.ipynb) |
| 3.4 | Day 3 · Session 4 | Run a prior sensitivity analysis and summarise what changes | [`D3_4_run_prior_sensitivity.ipynb`](labs/D3_4_run_prior_sensitivity.ipynb) |
| 4.1 | Day 4 · Session 1 | Compute expected utility of replace, repair and wait per district | [`D4_1_compute_expected_utility.ipynb`](labs/D4_1_compute_expected_utility.ipynb) |
| 4.2 | Day 4 · Session 2 | Choose districts to re-pipe within budget, with avoided bursts and uncertainty | [`D4_2_choose_districts_re.ipynb`](labs/D4_2_choose_districts_re.ipynb) |
| 4.3 | Day 4 · Session 3 | Price the leak survey by its value of information | [`D4_3_price_leak_survey.ipynb`](labs/D4_3_price_leak_survey.ipynb) |
| 4.4 | Day 4 · Session 4 | Review a supplied analysis by checklist and choose the capstone brief | [`D4_4_review_supplied_analysis.ipynb`](labs/D4_4_review_supplied_analysis.ipynb) |
<!-- /academy:labs -->

## Capstone

A Bayesian model and decision memo for one of three synthetic briefs (water losses, school inspections, clinic waiting times) in a public repository whose README reports diagnostics, checks and the decision; 15-minute oral with a peer-model review

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- DS-301 Regression and Generalised Linear Modelling
- Fits and interprets generalised linear models in code
- Simulates from probability distributions and computes a likelihood

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Probabilistic programming:** Stan through CmdStanPy or CmdStanR, PyMC with Bambi, NumPyro, brms for R
- **Diagnostics and model comparison:** ArviZ, loo and bayesplot for R
- **Analysis environment:** Python through uv with Jupyter or marimo, R with Quarto
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- NDMO Data Classification Policy: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- National AI Risk Management Framework: REPLACE: the address of the official text

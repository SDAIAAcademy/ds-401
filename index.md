---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/index.html"
---

# Bayesian Modelling and Decisions under Uncertainty

النمذجة البايزية واتخاذ القرارات في ظل عدم اليقين

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 400

Build Bayesian models of uneven data, prove the computation and the code are sound, and turn the posterior into a decision with a price on further data

Participants build Bayesian models that support decisions and judge models built by others. On synthetic burst records from 140 districts of a fictional water utility, they test priors by simulation, fit and diagnose Hamiltonian Monte Carlo, pool small districts hierarchically, check and compare models by predictive accuracy, and verify code by simulation-based calibration. The last day turns posteriors into a replacement plan by expected utility and prices a proposed leak survey by its value of information.

**Who it is for:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Yanabi Water Authority.** Yanabi Water Authority is a fictional water utility with 140 districts, from 200 km urban networks to rural networks of one or two kilometres, some with burst sensors and most rural ones without. Participants model five years of synthetic burst records with known generating parameters (seed 401), with planted problems to find: very unequal districts, under-reporting where no sensors listen, extra year-to-year variation, a trunk-main failure year, a seeded coding error and a consultant analysis with six flaws; the posterior then decides a re-piping plan within a capital budget and the price of an acoustic leak survey.

## The course

### Day 1: From priors to posteriors

Slides: https://sdaiaacademy.github.io/ds-401/slides/day1.md

- **Session 1: Models that generate data**
  - Write a generative model of burst counts with network length as exposure
  - Compute a posterior on a grid from a prior and a likelihood
  - Explain why the posterior of a small district leans on its prior
  - [Lab 1.1: Simulate and recover a burst rate (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D1_1_simulate_burst_counts.ipynb)
- **Session 2: Priors you can defend**
  - Translate a prior on a log scale into the units the engineers use
  - Run a prior predictive simulation for every district
  - Write a reason for each prior a model carries
  - [Lab 1.2: Prior predictive checks (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D1_2_run_prior_predictive.ipynb)
- **Session 3: Fitting with Hamiltonian Monte Carlo**
  - Explain how Hamiltonian Monte Carlo uses the gradient of the log posterior
  - Write a Poisson model with an offset in a probabilistic programming language
  - Report rate ratios with posterior intervals
  - [Lab 1.3: A Poisson model with an offset (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D1_3_fit_poisson_model.ipynb)
- **Session 4: Reading the computation**
  - Read R-hat, bulk and tail effective sample size and divergences
  - Tell an efficiency warning from a validity failure
  - Judge when approximate inference is good enough for a question
  - [Lab 1.4: Four fits, two untrustworthy (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D1_4_flag_two_untrustworthy.ipynb)

### Day 2: Pooling across districts

Slides: https://sdaiaacademy.github.io/ds-401/slides/day2.md

- **Session 1: No, complete and partial pooling**
  - Fit no-pooling, complete-pooling and partial-pooling models of the same counts
  - Read shrinkage as a function of how much data a district has
  - Check whether pooling pulls one group of districts towards another’s average
  - [Lab 2.1: Three poolings and the rural districts (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D2_1_fit_all_three.ipynb)
- **Session 2: Group-level predictors**
  - Add district-level predictors to the mean of a hierarchical model
  - Compare the between-district spread before and after
  - Explain why predictors change who is shrunk towards what
  - [Lab 2.2: Material and age at the district level (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D2_2_add_pipe_material.ipynb)
- **Session 3: The funnel and non-centred models**
  - Recognise the funnel geometry behind divergences in a hierarchical model
  - Rewrite a centred model in non-centred form
  - Show that the repaired model gives a trustworthy answer
  - [Lab 2.3: Repair a divergent model (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D2_3_repair_divergent_model.ipynb)
- **Session 4: Under-reporting as measurement error**
  - Write reported counts as true counts thinned by a reporting share
  - Use an audit to set an informative prior on what the data cannot identify
  - Report corrected rates for districts without sensors with their wider uncertainty
  - [Lab 2.4: A reporting rate for districts without sensors (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D2_4_add_reporting_rate.ipynb)

### Day 3: Is the model good enough

Slides: https://sdaiaacademy.github.io/ds-401/slides/day3.md

- **Session 1: Posterior predictive checks**
  - Simulate replicated data from a fitted model
  - Choose test statistics that target the decision, such as the share of zeros
  - Replace a model whose checks fail and check the replacement
  - [Lab 3.1: Zeros and dispersion (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D3_1_expose_poisson_misfit.ipynb)
- **Session 2: Comparing models by predictive accuracy**
  - Estimate expected log predictive density by Pareto-smoothed importance sampling
  - Compare models by the difference and its standard error, not by the ranking alone
  - Find influential observations from Pareto k and decide what to do about them
  - [Lab 3.2: Three models by leave-one-out (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D3_2_compare_three_models.ipynb)
- **Session 3: Is the code right**
  - Recover known parameters from data simulated by the model
  - Run a short simulation-based calibration and read its rank histograms
  - Locate and fix an implementation error from the parameter that fails
  - [Lab 3.3: Simulation-based calibration (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D3_3_locate_seeded_error.ipynb)
- **Session 4: How much the priors matter**
  - Run a power-scaling sensitivity diagnostic
  - Refit with alternative priors that someone could defend
  - Summarise what changes for the decision and what does not
  - [Lab 3.4: Prior sensitivity (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D3_4_run_prior_sensitivity.ipynb)

### Day 4: From posterior to decision

Slides: https://sdaiaacademy.github.io/ds-401/slides/day4.md

- **Session 1: Loss, utility and the decision table**
  - Write a decision table of actions, costs and consequences
  - Compute expected cost over posterior draws for each action
  - Show where a decision on the posterior mean differs from one on the whole posterior
  - [Lab 4.1: Replace, repair or wait (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D4_1_compute_expected_utility.ipynb)
- **Session 2: Choosing under a budget**
  - Rank actions by expected benefit per riyal
  - Choose a plan that fits a capital budget
  - Report avoided bursts with an interval, and who carries the remaining risk
  - [Lab 4.2: A plan within the budget (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D4_2_choose_districts_re.ipynb)
- **Session 3: What more data is worth**
  - Compute the expected value of perfect information for a decision
  - Compute the expected value of a survey by preposterior simulation
  - Give a verdict on an offer of data by comparing its value with its price
  - [Lab 4.3: The price of the leak survey (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D4_3_price_leak_survey.ipynb)
- **Session 4: Reviewing a Bayesian analysis**
  - Review an analysis against a checklist of priors, computation, checking and reporting
  - Rank flaws by their effect on the decision
  - Choose and scope a capstone brief
  - [Lab 4.4: Review an analysis, choose a brief (25 min)](https://sdaiaacademy.github.io/ds-401/labs/D4_4_review_supplied_analysis.ipynb)

### Capstone: Capstone project presentations

Slides: https://sdaiaacademy.github.io/ds-401/slides/capstone.md

## Capstone

A Bayesian model and decision memo for one of three synthetic briefs (water losses, school inspections, clinic waiting times) in a public repository whose README reports diagnostics, checks and the decision; 15-minute oral with a peer-model review

**What you submit:** A public GitHub repository link: the model, the run log, the checks and the two-page decision memo, with a README that reports the diagnostics, the checks and the decision

**Rubric:**

- Model and priors: 20%
- Computation: 20%
- Checking: 25%
- Decision: 25%
- Reproducibility: 10%

## Outcomes

- **LO1** Implement a Bayesian workflow from prior predictive simulation to sampler diagnosis, given a burst-count model with network length as exposure (Applied) Evidence: a notebook with prior draws, diagnostics and a reason for each prior
- **LO2** Design a hierarchical model for districts with very unequal data and defend the pooling it implies, given 140 districts, a third with fewer than five recorded bursts (Performance) Evidence: no-, complete- and partial-pooling estimates with a written defence
- **LO3** Evaluate competing models by posterior predictive checks and approximate leave-one-out cross-validation and choose one for the decision, given three candidate models of the same data (Performance) Evidence: a comparison table with check plots and the reasoning
- **LO4** Diagnose an error in a model implementation by parameter recovery and simulation-based calibration, given a model with a seeded coding error (Performance) Evidence: calibration plots that locate the error and a corrected model that passes
- **LO5** Decide a replacement plan by expected utility under the posterior and price further data by its value of information, given costs, a capital budget and an offered acoustic leak survey (Performance) Evidence: a two-page memo with the plan, its expected loss and a verdict on the survey
- **LO6** Review another analyst's Bayesian analysis for priors, computation, checking and reporting, given an analysis with six planted flaws (Performance) Evidence: a review that finds at least four flaws and ranks them by effect on the decision

## Tools

- **Probabilistic programming:** Stan through CmdStanPy or CmdStanR, PyMC with Bambi, NumPyro, brms for R
- **Diagnostics and model comparison:** ArviZ, loo and bayesplot for R
- **Analysis environment:** Python through uv with Jupyter or marimo, R with Quarto

## Prerequisites

- DS-301 Regression and Generalised Linear Modelling
- Fits and interprets generalised linear models in code
- Simulates from probability distributions and computes a likelihood

## Sources

### Workflow and computation

- [Gelman et al. (2020), Bayesian Workflow](https://arxiv.org/abs/2011.01808)
- [Betancourt (2017), A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434)
- [Betancourt and Girolami (2013), Hamiltonian Monte Carlo for Hierarchical Models](https://arxiv.org/abs/1312.0906)
- [Vehtari et al. (2019), Rank-normalization, folding, and localization: an improved R-hat](https://arxiv.org/abs/1903.08008)
- [Stan User’s Guide](https://mc-stan.org/docs/stan-users-guide/)

### Checking and comparing

- [Gabry et al. (2017), Visualization in Bayesian workflow](https://arxiv.org/abs/1709.01449)
- [Vehtari, Gelman and Gabry (2015), Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC](https://arxiv.org/abs/1507.04544)
- [Talts et al. (2018), Validating Bayesian Inference Algorithms with Simulation-Based Calibration](https://arxiv.org/abs/1804.06788)
- [Modrák et al. (2022), Simulation-Based Calibration Checking for Bayesian Computation](https://arxiv.org/abs/2211.02383)
- [Kallioinen et al. (2021), Detecting and diagnosing prior and likelihood sensitivity with power-scaling](https://arxiv.org/abs/2107.14054)

### Tools

- [NumPyro documentation](https://num.pyro.ai/en/stable/)
- [ArviZ documentation](https://python.arviz.org/en/stable/)
- [CmdStanPy](https://mc-stan.org/cmdstanpy/)
- [Bambi](https://bambinos.github.io/bambi/)
- [loo for R](https://mc-stan.org/loo/)
- [bayesplot for R](https://mc-stan.org/bayesplot/)
- [marimo](https://marimo.io/)
- [Quarto](https://quarto.org/)

### National references

- [Data Classification Policy (National Data Management Office, SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
- [Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [AI Ethics Principles (SDAIA)](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [National AI Risk Management Framework, English executive summary (SDAIA)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryEn.pdf)

---

SDAIA Academy · Data Science Track · Bayesian Modelling and Decisions under Uncertainty

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit c70a408

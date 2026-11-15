<!-- academy:title -->
# Bayesian Modelling and Decisions under Uncertainty

النمذجة البايزية واتخاذ القرارات في ظل عدم اليقين

Data Science Track · SDAIA Academy · DS-401 · Level 400

Build Bayesian models of uneven data, prove the computation and the code are sound, and turn the posterior into a decision with a price on further data

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

**Yanabi Water Authority** is a fictional water utility with 140 districts, from 200 km urban networks to rural networks of one or two kilometres. Most rural districts have no burst sensors, so some of their bursts go unrecorded. Participants work on five years of synthetic burst records (2021 to 2025) written by `labs/prepare_data.py` with a fixed seed and known generating values, so every estimate can be checked against the truth.

The people of the case: Dr. Nouran Al-Ateeq, director of asset management, owns the decision; Shabib Al-Ruqaish, pipe network engineer, owns the burst records; Juhaina Al-Zahim, capital planning lead, owns the costs and the budget; Ouda Al-Shahwan, leak detection manager, brings an acoustic survey offer; Wurud Al-Samhan is the data protection officer; Sakhr Al-Nassar is the consultant whose analysis is reviewed on Day 4. Participants play Zaina Al-Habdan, senior data scientist.

Planted in the data, each found by a named lab: an implausible prior (Lab 1.2), two untrustworthy fits (Lab 1.4), rural districts pulled towards the urban average (Labs 2.1 and 2.2), a funnel in the small districts (Lab 2.3), under-reporting where no sensors listen (Lab 2.4), extra year-to-year variation (Lab 3.1), a trunk-main failure year (Lab 3.2), a seeded coding error (Lab 3.3) and a consultant analysis with six flaws (Lab 4.4). Costs, the budget and the survey price in `labs/data/costs.json` are synthetic planning numbers.

## How the course runs

- **Pre-work (three hours):** likelihoods and simulation, and a decision on whether real asset and customer records may enter a hosted notebook under the data classification levels and the Personal Data Protection Law (`labs/P0_prework_likelihood_simulation.ipynb`); an entry quiz checks it.
- **Four teaching days** of four sessions; each session ends with a 25-minute lab that starts from supplied files, so no lab depends on another. Day 4 starts from a supplied posterior.
- **AI policy:** the supervised check and the first model of each day (Labs 1.1, 2.1, 3.1, 4.1) are done without AI. In the other labs the tutor Mirsad (`labs/tutor_instruction.md`) explains diagnostics and errors and writes no model code. The capstone may use AI fully and is defended orally.
- **Follow-up (within three weeks):** a review of one workplace or public uncertainty analysis with the Day 4 checklist (`labs/followup/README.md`).

## Assessment

- A supervised check without AI: read diagnostics, choose a pooling structure, compute an expected loss by hand (20%).
- The capstone, graded on the rubric: model and priors 20, computation 20, checking 25, decision 25, reproducibility 10 (60%).
- The oral with a peer-model review (20%): the assessor changes a prior or a cost and you predict the effect on the decision before re-running.
- Passing: 70% overall, with at least 60% on the capstone rubric and on the supervised check. Run logs and commit history are part of the evidence. Certificate: completion with assessed work.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Simulate and recover a burst rate | [`D1_1_simulate_burst_counts.ipynb`](labs/D1_1_simulate_burst_counts.ipynb) |
| 1.2 | Day 1 · Session 2 | Prior predictive checks | [`D1_2_run_prior_predictive.ipynb`](labs/D1_2_run_prior_predictive.ipynb) |
| 1.3 | Day 1 · Session 3 | A Poisson model with an offset | [`D1_3_fit_poisson_model.ipynb`](labs/D1_3_fit_poisson_model.ipynb) |
| 1.4 | Day 1 · Session 4 | Four fits, two untrustworthy | [`D1_4_flag_two_untrustworthy.ipynb`](labs/D1_4_flag_two_untrustworthy.ipynb) |
| 2.1 | Day 2 · Session 1 | Three poolings and the rural districts | [`D2_1_fit_all_three.ipynb`](labs/D2_1_fit_all_three.ipynb) |
| 2.2 | Day 2 · Session 2 | Material and age at the district level | [`D2_2_add_pipe_material.ipynb`](labs/D2_2_add_pipe_material.ipynb) |
| 2.3 | Day 2 · Session 3 | Repair a divergent model | [`D2_3_repair_divergent_model.ipynb`](labs/D2_3_repair_divergent_model.ipynb) |
| 2.4 | Day 2 · Session 4 | A reporting rate for districts without sensors | [`D2_4_add_reporting_rate.ipynb`](labs/D2_4_add_reporting_rate.ipynb) |
| 3.1 | Day 3 · Session 1 | Zeros and dispersion | [`D3_1_expose_poisson_misfit.ipynb`](labs/D3_1_expose_poisson_misfit.ipynb) |
| 3.2 | Day 3 · Session 2 | Three models by leave-one-out | [`D3_2_compare_three_models.ipynb`](labs/D3_2_compare_three_models.ipynb) |
| 3.3 | Day 3 · Session 3 | Simulation-based calibration | [`D3_3_locate_seeded_error.ipynb`](labs/D3_3_locate_seeded_error.ipynb) |
| 3.4 | Day 3 · Session 4 | Prior sensitivity | [`D3_4_run_prior_sensitivity.ipynb`](labs/D3_4_run_prior_sensitivity.ipynb) |
| 4.1 | Day 4 · Session 1 | Replace, repair or wait | [`D4_1_compute_expected_utility.ipynb`](labs/D4_1_compute_expected_utility.ipynb) |
| 4.2 | Day 4 · Session 2 | A plan within the budget | [`D4_2_choose_districts_re.ipynb`](labs/D4_2_choose_districts_re.ipynb) |
| 4.3 | Day 4 · Session 3 | The price of the leak survey | [`D4_3_price_leak_survey.ipynb`](labs/D4_3_price_leak_survey.ipynb) |
| 4.4 | Day 4 · Session 4 | Review an analysis, choose a brief | [`D4_4_review_supplied_analysis.ipynb`](labs/D4_4_review_supplied_analysis.ipynb) |
<!-- /academy:labs -->

## Capstone

A Bayesian model and decision memo for one of three synthetic briefs (water losses, school inspections, clinic waiting times) in a public repository whose README reports diagnostics, checks and the decision; 15-minute oral with a peer-model review

**Submission:** a public GitHub repository link: the model, the run log, the checks and the two-page decision memo, with a README that reports the diagnostics, the checks and the decision. The talk is 8 minutes and the questions 7, inside the 15-minute oral. The three briefs and their data are in `labs/capstone/`.

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
<!-- /academy:sources -->

### National references

- NDMO Data Classification Policy: https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf
- Personal Data Protection Law and its implementing regulations: https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf and https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf
- SDAIA AI Ethics Principles: https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
- National AI Risk Management Framework: https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryEn.pdf (the English executive summary; the full text is published in Arabic at https://sdaia.gov.sa/ar/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryAr.pdf)

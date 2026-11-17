---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/"
---

# Bayesian Modelling and Decisions under Uncertainty

Build Bayesian models of uneven data, prove the computation and the code are sound, and turn the posterior into a decision with a price on further data

Participants build Bayesian models that support decisions and judge models built by others. On synthetic burst records from 140 districts of a fictional water utility, they test priors by simulation, fit and diagnose Hamiltonian Monte Carlo, pool small districts hierarchically, check and compare models by predictive accuracy, and verify code by simulation-based calibration. The last day turns posteriors into a replacement plan by expected utility and prices a proposed leak survey by its value of information.

- **Audience:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.
- **Prerequisites:** DS-301 Regression and Generalised Linear Modelling; Fits and interprets generalised linear models in code; Simulates from probability distributions and computes a likelihood
- **Case study:** Yanabi Water Authority: Yanabi Water Authority is a fictional water utility with 140 districts, from 200 km urban networks to rural networks of one or two kilometres, some with burst sensors and most rural ones without. Participants model five years of synthetic burst records with known generating parameters (seed 401), with planted problems to find: very unequal districts, under-reporting where no sensors listen, extra year-to-year variation, a trunk-main failure year, a seeded coding error and a consultant analysis with six flaws; the posterior then decides a re-piping plan within a capital budget and the price of an acoustic leak survey.

## Outcomes

- LO1 (Applied): Implement a Bayesian workflow from prior predictive simulation to sampler diagnosis, given a burst-count model with network length as exposure
- LO2 (Performance): Design a hierarchical model for districts with very unequal data and defend the pooling it implies, given 140 districts, a third with fewer than five recorded bursts
- LO3 (Performance): Evaluate competing models by posterior predictive checks and approximate leave-one-out cross-validation and choose one for the decision, given three candidate models of the same data
- LO4 (Performance): Diagnose an error in a model implementation by parameter recovery and simulation-based calibration, given a model with a seeded coding error
- LO5 (Performance): Decide a replacement plan by expected utility under the posterior and price further data by its value of information, given costs, a capital budget and an offered acoustic leak survey
- LO6 (Performance): Review another analyst's Bayesian analysis for priors, computation, checking and reporting, given an analysis with six planted flaws

## Day 1: From priors to posteriors

Write the story that makes the data, test the priors by simulation, fit with Hamiltonian Monte Carlo and read whether the computation can be trusted

#### [day1#title] From priors to posteriors

*SDAIA Academy · Bayesian Modelling and Decisions under Uncertainty*

Day 1: Write the story that makes the data, test the priors by simulation, fit with Hamiltonian Monte Carlo and read whether the computation can be trusted

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-401

#### [day1#agenda] From priors to posteriors

*Today*

- **Session 1 · Models that generate data**: A burst count as a generative story, and Bayes’ rule computed on a grid
- **Session 2 · Priors you can defend**: Priors judged by what they predict, before any outcome is seen
- **Session 3 · Fitting with Hamiltonian Monte Carlo**: Why sampling replaces the grid, and how HMC and NUTS move through a posterior
- **Session 4 · Reading the computation**: The diagnostics that decide whether a fit can be read, and when an approximation is enough

#### [day1#the-decision] Every model this week ends in one decision about pipes

*Yanabi Water Authority*

- **The question.** Dr. Nouran Al-Ateeq, director of asset management, must choose which of 140 districts to re-pipe or repair within a capital budget.
- **The data.** Five years of reported bursts, 2021 to 2025, from networks of 1.2 km to 198 km. Synthetic, with known generating values.
- **The difficulty.** A third of the districts recorded fewer than five bursts in five years, and most rural districts have no sensors.

Day 1 builds the workflow on one model. Day 2 pools districts, Day 3 checks the models, Day 4 decides.

#### [day1#prework-classification] Real asset and customer records stay out of a hosted notebook until classified

*From the pre-work*

- The pre-work asked you to classify Yanabi's real burst and customer files under the national data classification levels.
- Burst counts per district: an internal record. Customer addresses on a burst ticket: personal data under the Personal Data Protection Law.
- This course uses only synthetic files, so Colab is allowed here. At work, the classification decides where a notebook may run.

- **Ask Wurud first.** Wurud Al-Samhan, the data protection officer, signs off before a real extract leaves Yanabi's systems.

#### [day1#setup-check] One notebook proves your setup: run it now

*Before the first lab*

```bash
# file: terminal
cd ds-401/labs
uv sync
uv run jupyter execute 00_setup_check.ipynb
```

- Or open `00_setup_check.ipynb` in Colab and run all cells.
- It imports NumPyro, JAX and ArviZ, fits a tiny model and prints the versions.
- Everything runs on a laptop processor. No model is downloaded and no key is needed.

> **AI policy.** The first model of each day (Labs 1.1, 2.1, 3.1, 4.1) is written without AI. In the other labs the tutor Mirsad explains diagnostics and errors; it writes no model code.

### Session 1: Models that generate data

#### [day1#s1] Models that generate data

*Session 1*

- Write a generative model of burst counts with network length as exposure
- Compute a posterior on a grid from a prior and a likelihood
- Explain why the posterior of a small district leans on its prior

#### [day1#recap-glm] You have fitted this count model before, by maximum likelihood

*Recap from DS-301*

- Counts with an exposure: log E[bursts] = log km + α + Xβ.
- The offset fixes the coefficient of log km at 1, so β describes a rate.
- Overdispersion and the negative binomial: Day 3 returns to them.

- **What changes this week.** Each unknown gets a distribution, not one best value. The decision on Day 4 uses the whole distribution.

#### [day1#generative-story] Write the story that produces the bursts, then run it forwards

*A model is a simulator*

1. **1 · Rate** bursts per km of main per year
2. **2 · Exposure** times the district's km, for one year
3. **3 · Count** a Poisson draw with that mean
4. **4 · Record** what was reported (Day 2)

If you can simulate the data, you can test every later step against a truth you chose.

#### [day1#bayes-rule] The posterior weighs every candidate rate by prior and data

*Prior times likelihood*

```python
# file: Lab 1.1, task 3
lp = log_prior(grid) + log_lik(grid, counts, km)
w = np.exp(lp - lp.max())
w = w / w.sum()   # posterior weights on the grid
```

- Work on the log scale: products of many probabilities underflow.
- Subtract the maximum before exponentiating.
- The normalising constant is the sum of the weights.

#### [day1#small-and-large] Five years say little about a 1.2 km district

*Lab 1.1, task 4*

| District | km | Bursts, 5 years | Median rate | 90% interval |
| --- | --- | --- | --- | --- |
| D043, rural | 1.2 | 2 | 0.194 | 0.058 to 0.523 |
| D011, urban | 198.0 | 87 | 0.087 | 0.073 to 0.104 |

Same prior, same five years: the small district's interval is 15 times wider, and its median still leans on the prior's centre.

#### [day1#grid-limit] A grid works for one rate and fails for 140

*Why we need a sampler*

**10^280** grid points for 140 rates at 100 points each

- The grid grows exponentially with the number of unknowns.
- Almost all of its points have negligible posterior weight.
- A sampler spends its effort where the weight is: session 3.

#### [day1#workflow-map] A Bayesian workflow is a loop, not a fit

*The week as a workflow*

1. **Prior predictive** Day 1
2. **Fit and diagnose** Days 1 and 2
3. **Check and compare** Day 3
4. **Calibrate the code** Day 3
5. **Decide** Day 4

> **A failed check sends you back.** A model is expanded or replaced, and every step runs again.

#### [day1#check-grid] A narrow posterior

*Checkpoint*

**Question:** Two districts share the prior. District P has 200 km and 90 bursts in five years; district Q has 2 km and 3 bursts. Which statement about their posteriors holds?

- A. P's posterior depends more on the prior than Q's
- B. Q's posterior is wider and closer to the prior's centre
- C. Both have the same width because the prior is the same
- D. Q's posterior is narrow because 3 bursts is a small number

#### [day1#lab-1-1] Simulate and recover a burst rate

Lab 1.1 · 25 min

Simulate burst counts from a rate you choose, recover it on a grid, then compare the posteriors of a small and a large district.

[D1_1_simulate_burst_counts.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D1_1_simulate_burst_counts.ipynb)

- [ ] The generative story as a simulator (6 min)
- [ ] The log likelihood on a grid of rates (5 min)
- [ ] Prior times likelihood: the grid posterior (5 min)
- [ ] A small district and a large one (6 min)
- [ ] Where the grid stops working (3 min)
- [ ] ✱ Stretch: Simulate with year-to-year variation and see the grid posterior become too sure

### Session 2: Priors you can defend

#### [day1#s2] Priors you can defend

*Session 2*

- Translate a prior on a log scale into the units the engineers use
- Run a prior predictive simulation for every district
- Write a reason for each prior a model carries

#### [day1#log-scale] A prior on the log rate means something only on the rate scale

*Reading a prior*

| Prior on the log rate | 5th percentile | Median | 95th percentile |
| --- | --- | --- | --- |
| N(0, 10), "flat-looking" | 0.00000013 | 0.78 | 17,000,000 |
| N(log 0.1, 1), weakly informative | 0.019 | 0.10 | 0.52 |
| N(log 0.5, 0.05), tight | 0.46 | 0.50 | 0.54 |

Bursts per km of main per year, from 1,000 prior draws (Lab 1.2, task 1, quick run).

#### [day1#flat-not-harmless] The "harmless" flat prior predicts impossible district-years

*Lab 1.2, tasks 2 and 3*

**37%** of simulated district-years above 500 bursts under N(0, 10)

- No district in the region has ever had 500 bursts in a year.
- With little data, such a prior pulls small districts towards absurd values.
- With a lot of data it hardly matters, which is how it hides.

#### [day1#prior-predictive] Simulate data from the prior alone, then ask an engineer

*Prior predictive simulation*

1. **1 · Draw** parameters from the prior
2. **2 · Simulate** a year in all 140 districts
3. **3 · Compare** with statements written before the data

> **Not the burst counts.** Tuning a prior to the observed data uses the data twice and makes the posterior too sure.

#### [day1#too-tight] A prior can also be wrong by being too sure

*The other failure*

| Prior | District-years above 500 | Network totals in 300 to 3,000 |
| --- | --- | --- |
| N(0, 10) | 37% | 10% |
| N(log 0.1, 1) | 0% | 73% |
| N(log 0.5, 0.05) | 0% | 0% |

The vendor's tight prior never breaks the first statement, yet it rules out every network total the engineers call plausible.

#### [day1#reason-per-prior] Every prior in the notebook carries one sentence of reason

*The evidence LO1 asks for*

- **α ~ Normal(log 0.1, 1).** Centred on the engineers' "one burst per 10 km a year"; 73% of simulated network totals within their plausible range.
- **β age ~ Normal(0, 0.5).** Median rate ratio 1.0 per 10 years, 95th percentile 2.3: "might double, never five times".

> **A reason quotes a number or a statement.** "Weakly informative" alone is a label, not a reason.

#### [day1#weakly-informative] Weakly informative: rules out the impossible, stays open on the plausible

*A working definition*

- **Too vague.** Puts weight on rates no pipe ever had. Harms small districts.
- **Weakly informative.** Spans every rate an engineer would accept, and little beyond.
- **Informative.** Carries a measurement: an audit, an earlier study. Day 2 uses one for reporting.

#### [day1#check-prior] Choosing the prior

*Checkpoint*

**Question:** A colleague sets the prior on the log burst rate by fitting it to the five years of observed counts, then fits the model to the same counts. What is the problem?

- A. Priors may never use data of any kind
- B. Nothing: with 700 counts the prior does not matter
- C. The data are used twice, so the posterior is too sure
- D. The sampler cannot handle a prior estimated from data

#### [day1#lab-1-2] Prior predictive checks

Lab 1.2 · 25 min

Simulate a year of bursts in all 140 districts from three candidate priors, compare with the engineers’ statements, keep one and reject two with reasons.

[D1_2_run_prior_predictive.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D1_2_run_prior_predictive.ipynb)

- [ ] Three priors in bursts per km-year (5 min)
- [ ] Simulate a year in every district from each prior (6 min)
- [ ] Compare with what the engineers know (5 min)
- [ ] Keep one prior, reject two, with a reason each (5 min)
- [ ] The prior of the age effect (4 min)
- [ ] ✱ Stretch: Prior predictive share of district-years without a burst

### Session 3: Fitting with Hamiltonian Monte Carlo

#### [day1#s3] Fitting with Hamiltonian Monte Carlo

*Session 3*

- Explain how Hamiltonian Monte Carlo uses the gradient of the log posterior
- Write a Poisson model with an offset in a probabilistic programming language
- Report rate ratios with posterior intervals

#### [day1#draws-not-formulas] A pile of posterior draws answers any question you can compute

*Monte Carlo*

- A median, an interval, a probability: functions of the draws.
- A derived quantity, such as next year's bursts in D017, is computed draw by draw.
- The decision on Day 4 is a cost computed in every draw, then averaged.

```python
# file: Lab 1.3, task 5
mu = np.exp(a + b @ X[j]) * km[j]   # one value per draw
np.quantile(mu, [0.5, 0.05, 0.95])
```

#### [day1#hmc-intuition] HMC rolls a particle across the posterior using its gradient

*Hamiltonian Monte Carlo*

- **Position.** The current values of all parameters.
- **Momentum.** A random push, drawn fresh at each iteration.
- **Gradient.** The slope of the log posterior steers the path towards high density.

The path is followed in small leapfrog steps; the end point is accepted or rejected so that the draws come from the posterior.

#### [day1#nuts] NUTS chooses the path length; warm-up chooses the step size

*The No-U-Turn sampler*

| Setting | Who sets it | What goes wrong |
| --- | --- | --- |
| Path length | NUTS: doubles the path until it turns back | Paths that hit the maximum tree depth: slow |
| Step size | Warm-up: tuned to a target acceptance rate | Too large for a curved region: divergences |
| Mass matrix | Warm-up: the scale of each parameter | Very different scales: slow mixing |

#### [day1#numpyro-model] The model is the generative story, written as code

*Lab 1.3, task 1*

```python
# file: Lab 1.3
def pooled_model(d, log_km, X, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([4]).to_event(1))
    eta = alpha + X @ beta                      # log rate per district
    numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
```

The same function simulates data when `y` is missing and conditions on it when `y` is given.

#### [day1#tool-slot] The labs use NumPyro; the workflow is the same in every tool

*Probabilistic programming: interchangeable examples*

- **Stan.** Through CmdStanPy or CmdStanR; compiles a model to C++.
- **PyMC.** With Bambi for formula-style models.
- **NumPyro.** On JAX; installs without a compiler, runs on a laptop or Colab.
- **brms.** Formula models in R, fitted by Stan.

Diagnostics and comparison: ArviZ in Python; loo and bayesplot in R. Your capstone may use any of them.

#### [day1#pooled-result] Asbestos cement bursts nearly twice as often as PVC of the same age

*Lab 1.3, task 3*

| Item | Value |
| --- | --- |
| Asbestos cement | 1.85 (1.70 to 2.02) |
| Age, per 10 years | 1.35 (1.30 to 1.39) |
| Cast iron | 1.33 (1.23 to 1.44) |
| Ductile iron | 1.22 (1.13 to 1.31) |

Rate ratios against PVC, posterior medians with 90% intervals, complete-pooling Poisson model (quick run).

#### [day1#check-offset] The offset

*Checkpoint*

**Question:** Juhaina asks why the model does not estimate a coefficient for log km, as it does for age. What do you answer?

- A. Because bursts are assumed proportional to length, so its coefficient is fixed at 1
- B. Because network length does not affect the number of bursts
- C. Because the sampler cannot estimate a coefficient on a logarithm
- D. Because its prior is centred on 1 and the data agree

#### [day1#lab-1-3] A Poisson model with an offset

Lab 1.3 · 25 min

Write the burst model with network length as exposure in NumPyro, run four chains, and read material and age effects as rate ratios with intervals.

[D1_3_fit_poisson_model.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D1_3_fit_poisson_model.ipynb)

- [ ] Write the model (8 min)
- [ ] Run four chains (4 min)
- [ ] Rate ratios with intervals (6 min)
- [ ] Against the generating values (3 min)
- [ ] Expected bursts next year for two districts (4 min)
- [ ] ✱ Stretch: Remove the offset and see what the model then says

### Session 4: Reading the computation

#### [day1#s4] Reading the computation

*Session 4*

- Read R-hat, bulk and tail effective sample size and divergences
- Tell an efficiency warning from a validity failure
- Judge when approximate inference is good enough for a question

#### [day1#four-checks] Four numbers decide whether a fit can be read

*Before reading any posterior*

| Diagnostic | Course threshold, four chains | What it detects |
| --- | --- | --- |
| Rank-normalised R-hat | below 1.01 | chains that disagree |
| Bulk and tail ESS | at least 400 | too few independent draws for the centre or the tails |
| Divergences | zero, or explained | regions the integrator could not follow |
| E-BFMI | above 0.3 | momentum that cannot cross energy levels |

#### [day1#rhat-story] No divergence, and still one chain is somewhere else

*Fit C*

| Chain | Mean of α |
| --- | --- |
| 1 | -2.417 |
| 2 | -2.411 |
| 3 | -2.403 |
| 4 | -1.229 |

**1.60** R-hat of fit C

#### [day1#divergences] 741 divergences: the sampler could not reach part of the posterior

*Fit B*

- A divergence is a path whose energy blew up: the step size was too large for the curvature there.
- The draws then miss that region, and every summary is biased.
- Fit B is the centred hierarchical model on the small districts: a funnel.

> **Do not just raise target_accept.** It shrinks the step size, often leaving fewer divergences and the same bias. Lab 2.3 changes the model instead.

#### [day1#efficiency-validity] A capped tree costs time; it does not make the draws wrong

*Fit D*

- **33%** of draws hit the tree-depth cap of 4
- **1.002** R-hat
- **1,452** smallest bulk ESS

Zaina capped the tree depth to save time. The chains still mixed, so fit D can be read.

#### [day1#rank-plots] Rank plots show chains that disagree better than trace plots do

*Looking at chains*

- Rank all draws of a parameter together, then plot each chain's ranks as a histogram.
- Chains from the same distribution: flat, similar histograms.
- A chain stuck high or low: a histogram piled at one end.

```python
# file: Lab 1.4, task 4
pc = az.plot_rank(fits["fit_C"], var_names=["alpha", "tau"])
pc.savefig("artifacts/rank_fit_C.png")
```

#### [day1#approximate-inference] An approximation is fine when the posterior is simple and the question is its centre

*When approximate inference suffices*

| 90% interval of β | NUTS | Variational (AutoNormal) |
| --- | --- | --- |
| Age per 10 years | 0.265 to 0.334 | 0.253 to 0.326 |
| Asbestos cement | 0.533 to 0.699 | 0.551 to 0.698 |

Five parameters and 700 counts: close enough for a rate ratio. For 140 district effects, skewed spreads and tail decisions, sample.

#### [day1#check-diagnostics] Which fit to send back

*Checkpoint*

**Question:** A fit reports R-hat 1.003, bulk ESS 2,100, no divergences, and 40% of draws at the maximum tree depth. Another reports no divergences and R-hat 1.21. Which do you send back?

- A. The first, because 40% of trees are capped
- B. Neither, because neither has divergences
- C. Both, because each has a warning
- D. The second, because its chains disagree

#### [day1#lab-1-4] Four fits, two untrustworthy

Lab 1.4 · 25 min

Compute the diagnostics of four supplied fits, flag the two that cannot be trusted and write the reason for each verdict.

[D1_4_flag_two_untrustworthy.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D1_4_flag_two_untrustworthy.ipynb)

- [ ] Load the fits and read the run log (3 min)
- [ ] R-hat and effective sample sizes (6 min)
- [ ] Divergences, tree depth and energy (6 min)
- [ ] Rank plots of the two suspects (5 min)
- [ ] A verdict per fit, with its reason (5 min)
- [ ] ✱ Stretch: Fit Lab 1.3’s model with a variational approximation and compare the intervals with NUTS

#### [day1#wrap-up] Three things to keep

*Day 1 · wrap-up*

- **Simulate first.** A model you can run forwards can be tested against a truth you chose, and its priors against what engineers know.
- **Diagnose before reading.** R-hat, ESS, divergences and E-BFMI, every fit, before any number leaves the notebook.
- **Small districts lean on the prior.** Five years of 2 bursts say little; tomorrow they borrow from the other districts.

#### [day1#next] Day 2: Pooling across districts

*Next*

Finish any lab you did not complete; each one starts from the supplied files. Read the run log of fit B again: tomorrow you repair it.

## Day 2: Pooling across districts

Let districts with little data borrow strength from the others, and say what that borrowing does to the rural ones

#### [day2#title] Pooling across districts

*SDAIA Academy · Bayesian Modelling and Decisions under Uncertainty*

Day 2: Let districts with little data borrow strength from the others, and say what that borrowing does to the rural ones

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-401

#### [day2#agenda] Pooling across districts

*Today*

- **Session 1 · No, complete and partial pooling**: Three ways to treat 140 districts, and the shrinkage a hierarchy implies
- **Session 2 · Group-level predictors**: Shrinking towards a prediction instead of towards one grand mean
- **Session 3 · The funnel and non-centred models**: Why hierarchical models diverge when data are thin, and the reparameterisation that repairs them
- **Session 4 · Under-reporting as measurement error**: Bursts that happen but are not recorded, modelled instead of ignored

#### [day2#review-day1] Before reading a fit

*Review of Day 1*

**Question:** A colleague's fit reports R-hat 1.004 and bulk ESS 900, but 23 divergences. She wants to publish the rate ratios. What do you advise?

- A. Not yet: find where the divergences sit and change the model or its form, then refit
- B. Publish: R-hat and ESS are fine
- C. Run twice as many draws and publish
- D. Drop the divergent draws and publish the rest

### Session 1: No, complete and partial pooling

#### [day2#s1] No, complete and partial pooling

*Session 1*

- Fit no-pooling, complete-pooling and partial-pooling models of the same counts
- Read shrinkage as a function of how much data a district has
- Check whether pooling pulls one group of districts towards another’s average

#### [day2#recap-random-intercepts] You met random intercepts in DS-301; today they get priors and a defence

*Recap from DS-301*

- DS-301 Day 4 fitted random intercepts by maximum likelihood for 30 districts.
- Here: 140 districts, a prior on the spread τ, the whole posterior of every district.
- New today: predictors at the district level, the funnel, and measurement error.

- **The question LO2 asks.** Not "is pooling allowed?" but "what does this pooling do to which districts, and can you defend it?"

#### [day2#three-poolings] No pooling trusts each district; complete pooling trusts none

*Three ways to treat 140 districts*

- **No pooling.** One rate per district, each from its own record. Tiny districts get wild estimates; 13 with no recorded burst look perfectly safe.
- **Complete pooling.** One rate for all (Lab 1.3). Every district is average, including the old asbestos-cement ones.
- **Partial pooling.** District rates drawn from a common distribution with spread τ. The data decide how much each district borrows.

#### [day2#hierarchy-code] One line makes the districts a population

*Lab 2.1, task 1*

```python
# file: Lab 2.1
alpha = numpyro.sample("alpha", dist.Normal(np.log(0.10), 1.0))
tau = numpyro.sample("tau", dist.HalfNormal(0.5))
with numpyro.plate("district", J):
    eta = numpyro.sample("eta", dist.Normal(alpha, tau))   # the population
numpyro.sample("y", dist.Poisson(jnp.exp(log_km + eta[d])), obs=y)
```

τ near 0 is complete pooling; τ very large is no pooling. The posterior of τ chooses between them.

#### [day2#shrinkage-by-kind] Rural districts borrow most of their estimate; urban ones keep their own

*Lab 2.1, task 3*

| Item | Value |
| --- | --- |
| Rural | 0.55 |
| Suburban | 0.12 |
| Urban | 0.04 |

Median shrinkage towards the common rate (0: own estimate, 1: common rate). τ = 0.56, 90% interval 0.49 to 0.65 (quick run).

#### [day2#rural-pull] Pooling towards one rate hides how much riskier the rural districts are

*Responsible use: Lab 2.1, task 4*

- **0.73** true gap of rural districts above the common rate (log scale)
- **0.11** the gap the estimates keep
- **16%** of the real difference survives

> **Who is harmed.** Most kilometres are urban, so the common rate is an urban rate. A plan built on these estimates under-invests where the pipes are oldest.

#### [day2#defend-pooling] A defence of pooling says what it does to whom

*LO2's evidence*

1. Why partial pooling: 51 districts recorded fewer than five bursts; alone, their estimates are noise.
2. What it does: small districts borrow from the population, here up to half their estimate.
3. Whom it affects: the rural districts, whose risk it understates while one urban-dominated rate is the target.
4. What changes next: shrink towards a prediction for each district's material and age.

#### [day2#check-pooling] A district with no recorded burst

*Checkpoint*

**Question:** D120 is a rural district of 2 km with no burst recorded in five years. What does each model say about its rate?

- A. No pooling and complete pooling both give it a rate near zero
- B. Partial pooling gives it exactly the common rate
- C. No pooling: near zero; complete: the common rate; partial: between, close to the common rate
- D. No model can say anything until it has a burst

#### [day2#lab-2-1] Three poolings and the rural districts

Lab 2.1 · 25 min

Fit the three models, plot how far each district’s estimate moves, and test whether partial pooling pulls rural districts towards the urban average.

[D2_1_fit_all_three.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D2_1_fit_all_three.ipynb)

- [ ] Three models from one function (5 min)
- [ ] Fit no, complete and partial pooling (6 min)
- [ ] The shrinkage plot (5 min)
- [ ] Are rural districts pulled towards the urban average? (6 min)
- [ ] A defence of the pooling in three sentences (3 min)
- [ ] ✱ Stretch: Plot shrinkage against each district’s number of recorded bursts

### Session 2: Group-level predictors

#### [day2#s2] Group-level predictors

*Session 2*

- Add district-level predictors to the mean of a hierarchical model
- Compare the between-district spread before and after
- Explain why predictors change who is shrunk towards what

#### [day2#shrink-to-prediction] Shrink each district towards what its pipes predict, not towards the average

*Group-level predictors*

```python
# file: Lab 2.2
mu = alpha + X @ beta          # material and age
eta = mu + tau * z             # district effect around it
```

- A rural asbestos-cement district of 40 years is now pulled towards the rate of such networks.
- τ now measures what material and age leave unexplained.
- The district-level predictors cost no data from the small districts: they are known for all 140.

#### [day2#tau-falls] Material and age explain about half of what made districts differ

*Lab 2.2, task 3*

- **0.56** τ without predictors
- **0.41** τ with material and age
- **47%** of the variance between districts explained

#### [day2#rural-bias-table] The predictors remove much of the rural bias, not all of it

*Lab 2.2, task 4*

| Mean log(estimate / true rate) | Without predictors | With material and age |
| --- | --- | --- |
| Rural | -0.61 | -0.25 |
| Suburban | -0.11 | -0.12 |
| Urban | -0.06 | -0.07 |

Rural estimates are still about 22% below the truth. Most rural districts have no sensors: session 4.

#### [day2#rural-indicator] A "rural" predictor fits the records better and the truth worse

*Lab 2.2, stretch*

- Its coefficient comes out negative: -0.21, 90% interval -0.48 to 0.04.
- It learns the low *reported* counts of rural districts, most without sensors.
- Rural estimates move further from their true rates: -0.32 on the log scale.

- **A predictor that stands for a missing variable.** "Rural" here means "no sensors". Model the recording process instead (session 4).

#### [day2#group-predictor-cost] A good group-level predictor is known for every group and plausibly causal

*Choosing group-level predictors*

- **Material, age.** Known for all 140; engineering reasons to expect an effect.
- **Pressure, soil.** Plausible; Yanabi has them for the urban districts only. A missing value is a modelling problem of its own.
- **District kind.** Known for all, but stands for several things at once, including how bursts are recorded.

#### [day2#check-predictors] What τ means now

*Checkpoint*

**Question:** After adding material and age, τ fell from 0.56 to 0.41. Nouran asks what that means for a rural asbestos-cement district with 3 recorded bursts.

- A. It is now pulled towards the rate predicted for old asbestos-cement networks, and somewhat more strongly
- B. It is pooled less because the model now knows more
- C. Its own three bursts no longer matter
- D. Its rate has fallen by 0.15

#### [day2#lab-2-2] Material and age at the district level

Lab 2.2 · 25 min

Add pipe material and age to the hierarchical model and compare the spread between districts and the rural shrinkage with Lab 2.1.

[D2_2_add_pipe_material.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D2_2_add_pipe_material.ipynb)

- [ ] District-level predictors in the mean (6 min)
- [ ] Fit and diagnose (5 min)
- [ ] The spread between districts, before and after (4 min)
- [ ] The rural districts again (6 min)
- [ ] What the predictors explain (4 min)
- [ ] ✱ Stretch: Add the district kind as a predictor and discuss what it stands for

### Session 3: The funnel and non-centred models

#### [day2#s3] The funnel and non-centred models

*Session 3*

- Recognise the funnel geometry behind divergences in a hierarchical model
- Rewrite a centred model in non-centred form
- Show that the repaired model gives a trustworthy answer

#### [day2#the-funnel] When τ is small, every district effect must squeeze into a narrow neck

*Why fit B diverged*

Figure: Sketch. In Lab 2.3's run, every one of the centred model's 59 divergences had τ below its median.

#### [day2#centred-vs-noncentred] Non-centred: sample standard district effects, then scale them by τ

*Same model, different coordinates*

```python
# file: centred
eta = sample("eta", Normal(mu, tau))
```

```python
# file: non-centred
z = sample("z", Normal(0, 1))
eta = deterministic("eta", mu + tau * z)
```

The joint distribution is identical. In z, the posterior has no funnel: its shape no longer depends on τ.

#### [day2#repair-numbers] The non-centred model samples the 49 small districts cleanly

*Lab 2.3, task 4*

|  | Divergences | R-hat | Bulk ESS | E-BFMI |
| --- | --- | --- | --- | --- |
| Centred | 59 | 1.037 | 49 | 0.10 |
| Non-centred | 0 | 1.004 | 563 | 0.82 |

Same priors, same 300 warm-up iterations, 2,000 kept draws (quick run).

#### [day2#tau-interval] The centred fit missed the small values of τ

*Lab 2.3, task 5*

| Item | Value |
| --- | --- |
| Centred | 0.095 to 0.491 |
| Non-centred | 0.018 to 0.514 |

90% intervals of τ for the small districts. Missing small τ means pooling them less than the data allow.

#### [day2#target-accept-trap] Raising target_accept removed the divergences and not the problem

*Lab 2.3, stretch*

- **0** divergences at target_accept 0.99
- **1.144** R-hat, still far above 1.01

- A smaller step reaches a little further into the neck, slowly.
- The chains still disagree, and τ's interval still starts at 0.061.
- Change the geometry; do not only shrink the step.

#### [day2#when-centred] With strong data in every group, the centred form can be the better one

*Not a rule of thumb*

- **All 140 districts, centred.** No divergence, bulk ESS above 1,000 in Lab 2.1: large districts pin their own effects, so there is no funnel.
- **49 small districts, centred.** 59 divergences: little data per district, the prior's funnel dominates.

> **Choose the form by the data, then check.** The diagnostics, not a habit, decide.

#### [day2#check-funnel] A divergent hierarchical model

*Checkpoint*

**Question:** A hierarchical model of 50 small districts gives 60 divergences, all at small τ. What is the best first step?

- A. Raise target_accept to 0.99 until the divergences stop
- B. Drop the districts with fewer than three bursts
- C. Switch to complete pooling, which never diverges
- D. Rewrite the district effects in non-centred form and refit

#### [day2#lab-2-3] Repair a divergent model

Lab 2.3 · 25 min

Reproduce the divergences of the centred model on the small districts, locate them in the funnel, and repair the model by non-centred parameterisation.

[D2_3_repair_divergent_model.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D2_3_repair_divergent_model.ipynb)

- [ ] Reproduce the divergences (5 min)
- [ ] Where they sit: the spread against one district (5 min)
- [ ] Write the non-centred model (7 min)
- [ ] Refit and compare the diagnostics (5 min)
- [ ] The same answer, now trustworthy? (3 min)
- [ ] ✱ Stretch: Raise the target acceptance rate of the centred model instead, and compare

### Session 4: Under-reporting as measurement error

#### [day2#s4] Under-reporting as measurement error

*Session 4*

- Write reported counts as true counts thinned by a reporting share
- Use an audit to set an informative prior on what the data cannot identify
- Report corrected rates for districts without sensors with their wider uncertainty

#### [day2#reported-not-happened] Rural districts without sensors report 30% fewer bursts per kilometre

*Lab 2.4, task 1*

| Item | Value |
| --- | --- |
| With sensors | 0.244 |
| Without sensors | 0.173 |

- Reported bursts per km-year, rural districts only.
- Real difference, or bursts nobody recorded?
- The counts alone cannot say: rate λ with reporting p gives the same counts as rate λp fully reported.

#### [day2#measurement-model] Model what was recorded as a share of what happened

*Under-reporting as measurement error*

1. **True bursts** Poisson(km × rate)
2. **Each one reported** with probability p
3. **Recorded bursts** Poisson(km × rate × p)

```python
# file: Lab 2.4
p = jnp.where(sensors, 1.0, p_raw)      # p_raw ~ Beta(14, 6)
y ~ Poisson(exp(log_km + eta[d] + log(p)[d]))
```

#### [day2#audit-prior] The 2024 audit sets the prior the data cannot replace

*An informative prior, with its source*

- Ouda's team listened to 6 districts without sensors for a season: 28 of 40 bursts found had been reported.
- Beta(14, 6): mean 0.70, 90% interval 0.52 to 0.85; the audit at half weight.
- Half weight because six districts in one season may not represent all 76.

- **The reason for this prior.** Cites a measurement, its size and why it was discounted: the standard of Day 1, applied to an informative prior.

#### [day2#bias-removed] The reporting term removes the bias of the districts without sensors

*Lab 2.4, task 4*

| Without sensors, mean log(estimate / truth) | Lab 2.2 model | With reporting term |
| --- | --- | --- |
| Rural | -0.28 | -0.02 |
| Suburban | -0.27 | 0.03 |
| Urban | -0.37 | -0.05 |

Urban districts without sensors had precise counts; their relative interval widens from 0.55 to 0.69. The reporting share is now their main uncertainty.

#### [day2#prior-carries-it] The counts taught the model almost nothing about reporting

*Lab 2.4, task 5*

**0.98** posterior sd of the reporting share over its prior sd, districts without sensors

- The correction is exactly as good as the audit.
- The memo must say so, and name the districts that depend on it.
- Better data on reporting is what Day 4's survey would buy.

#### [day2#check-reporting] Reading the correction

*Checkpoint*

**Question:** After adding the reporting term, a rural district without sensors moves from 0.18 to 0.26 bursts per km-year. Juhaina asks what the increase rests on.

- A. On the district's own burst counts, which reveal its reporting rate
- B. Mainly on the audit prior: the counts cannot separate rate from reporting
- C. On the sampler having more draws
- D. On an error in the burst register that the model fixed

#### [day2#lab-2-4] A reporting rate for districts without sensors

Lab 2.4 · 25 min

Add a reporting-rate term with an audit-based prior for districts without sensors and compare their corrected rates with the uncorrected ones.

[D2_4_add_reporting_rate.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D2_4_add_reporting_rate.ipynb)

- [ ] Reported bursts with and without sensors (4 min)
- [ ] The audit prior on the reporting share (5 min)
- [ ] Add the reporting term (7 min)
- [ ] Corrected rates for districts without sensors (6 min)
- [ ] What the data cannot tell apart (3 min)
- [ ] ✱ Stretch: Change the audit prior and watch the corrected rates follow it

#### [day2#wrap-up] Three things to keep

*Day 2 · wrap-up*

- **Say whom pooling moves.** Small districts borrow most; check whether they borrow from the right population.
- **Fix the geometry.** Divergences at small τ: non-centre the model, then check again.
- **Model the recording.** What was reported is not what happened; an audit-based prior carries what the counts cannot.

#### [day2#next] Day 3: Is the model good enough

*Next*

Tomorrow's first lab is AI-off: you write the negative binomial version of today's model. Reread Lab 2.4's model before you come.

## Day 3: Is the model good enough

Check the model against its own data, compare candidates by prediction, prove the code right and measure what the priors decide

#### [day3#title] Is the model good enough

*SDAIA Academy · Bayesian Modelling and Decisions under Uncertainty*

Day 3: Check the model against its own data, compare candidates by prediction, prove the code right and measure what the priors decide

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-401

#### [day3#agenda] Is the model good enough

*Today*

- **Session 1 · Posterior predictive checks**: Replicated data from the posterior, and the test statistics that expose a misfit
- **Session 2 · Comparing models by predictive accuracy**: Expected log predictive density, approximate leave-one-out and its diagnostics
- **Session 3 · Is the code right**: Parameter recovery and simulation-based calibration as tests of a model implementation
- **Session 4 · How much the priors matter**: Prior sensitivity by refitting and by power-scaling, reported on the quantities the decision uses

#### [day3#review-day2] What a pooled estimate rests on

*Review of Day 2*

**Question:** A rural district of 1.5 km recorded one burst in five years. Its posterior rate under the Lab 2.4 model is 0.21. What contributes most to that number?

- A. Its own record: 1 burst over 7.5 km-years
- B. The rate predicted for its material and age, plus the audit prior on reporting
- C. The sampler's starting values
- D. The common rate of all 140 districts

### Session 1: Posterior predictive checks

#### [day3#s1] Posterior predictive checks

*Session 1*

- Simulate replicated data from a fitted model
- Choose test statistics that target the decision, such as the share of zeros
- Replace a model whose checks fail and check the replacement

#### [day3#ppc-idea] A model that fits should be able to produce data like yours

*Posterior predictive checks*

1. **1 · Draw** parameters from the posterior
2. **2 · Replicate** 700 district-years from the model with those parameters
3. **3 · Compare** a statistic of the replicates with the same statistic of the data

> **Choose the statistic for the decision.** The total number of bursts fits almost any model; the variation from year to year decides how sure Day 4 can be.

#### [day3#two-statistics] The Poisson model cannot produce the year-to-year variation Yanabi records

*Lab 3.1, task 2*

| Districts over 10 km | Observed | Poisson replicates, 90% | p |
| --- | --- | --- | --- |
| Dispersion: variance of the five yearly counts over their mean | 2.32 | 0.87 to 1.13 | 0.000 |
| Share of district-years with no burst | 0.048 | 0.021 to 0.050 | 0.12 |

The dispersion fails outright. The quiet years sit at the edge of the Poisson range without leaving it: not every statistic exposes a misfit.

#### [day3#negbin] A negative binomial lets each year vary more than a Poisson

*Lab 3.1, task 3*

```python
# file: Lab 3.1
phi = numpyro.sample("phi", dist.Gamma(2.0, 0.2))
numpyro.sample("y", dist.NegativeBinomial2(mean, phi), obs=y)
# variance = mean + mean**2 / phi
```

- φ = 7.0, 90% interval 5.6 to 9.0 (the generating value was 6).
- Both statistics now sit inside the replicates: p = 0.34 and 0.48.
- Weather, pressure surges and repair campaigns move a district's year; the Poisson has no room for them.

#### [day3#rootogram] A histogram of all district-years cannot tell the two models apart

*Lab 3.1, stretch*

Figure: Square root of the frequencies, from the course's quick run. The dispersion within each district, 2.32 against Poisson replicates of 0.87 to 1.13, separates the same two models at once.

#### [day3#check-ppc] Choosing a test statistic

*Checkpoint*

**Question:** Shabib proposes checking the Poisson model on the total number of bursts per year across the network. What do you tell him?

- A. The model was fitted to reproduce totals; check the variation within districts from year to year instead
- B. A good check: if the totals match, the model fits
- C. Checks are only valid on data the model has not seen
- D. If p is near 0.5 the model is proven correct

#### [day3#lab-3-1] Zeros and dispersion

Lab 3.1 · 25 min

Expose the Poisson model’s misfit on district-years without a burst and on dispersion, then fit a negative binomial and check it the same way.

[D3_1_expose_poisson_misfit.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D3_1_expose_poisson_misfit.ipynb)

- [ ] Replicated data from the Poisson model (6 min)
- [ ] Share of zeros and dispersion (6 min)
- [ ] Fit a negative binomial (6 min)
- [ ] The same checks for the negative binomial (5 min)
- [ ] A verdict in two sentences (2 min)
- [ ] ✱ Stretch: Draw a rootogram of both models

### Session 2: Comparing models by predictive accuracy

#### [day3#s2] Comparing models by predictive accuracy

*Session 2*

- Estimate expected log predictive density by Pareto-smoothed importance sampling
- Compare models by the difference and its standard error, not by the ranking alone
- Find influential observations from Pareto k and decide what to do about them

#### [day3#loo-idea] Score each model on the district-years it did not see

*Comparing models by predictive accuracy*

- **elpd**: the expected log predictive density of a new district-year; higher is better.
- Leave-one-out refits 700 times; PSIS-LOO approximates all of them from one fit by importance sampling.
- **Pareto k** says, for each point, whether the approximation can be trusted (below about 0.7).

- **Why not R² or the fit to the data.** Any model with more parameters fits the data it saw better. Predicting what it did not see is the fair contest.

#### [day3#loo-table] The negative binomial hierarchy predicts best, by far more than its error

*Lab 3.2, tasks 2 and 3*

| Model | elpd | p_loo | Difference from best | SE of difference |
| --- | --- | --- | --- | --- |
| Hierarchical negative binomial | -1,393 | 66 | 0 | – |
| Hierarchical Poisson | -1,503 | 158 | -110 | 23 |
| Complete pooling, Poisson | -1,710 | 22 | -317 | 54 |

Read the difference against its own standard error, not against the elpd's: about 5 standard errors here.

#### [day3#pareto-k] D017's 2023 is the year a trunk main failed: keep it, choose a model it does not overturn

*Lab 3.2, task 4*

- The Poisson hierarchy flags six district-years above 0.7, D017 in 2023 among them: 54 bursts against 18 to 27 in its other years.
- The negative binomial expects some unusual years: it flags one, at 0.71.
- Shabib confirms the failure. The year is real; deleting it would understate risk.

- **A flagged point is a question.** Ask what happened before deciding what to do. An error is corrected; an event is modelled.

#### [day3#check-loo] Reading a model comparison

*Checkpoint*

**Question:** Model A has elpd -1,402 (SE 35) and model B -1,408 (SE 36). The difference is 6 with a standard error of 5. Juhaina asks which to use.

- A. A: it has the higher elpd
- B. Neither can be told apart because 6 is small next to 35
- C. No clear winner on prediction: choose by the checks, simplicity and what the decision needs
- D. The one with more parameters

#### [day3#lab-3-2] Three models by leave-one-out

Lab 3.2 · 25 min

Compare three candidate models by approximate leave-one-out, inspect the influential district-years, and choose one for the decision.

[D3_2_compare_three_models.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D3_2_compare_three_models.ipynb)

- [ ] Three candidates and their log likelihoods (5 min)
- [ ] Approximate leave-one-out for each (5 min)
- [ ] The difference and its standard error (5 min)
- [ ] Pareto k: the influential district-years (6 min)
- [ ] Choose a model for the decision (4 min)
- [ ] ✱ Stretch: Leave out whole districts instead of district-years

### Session 3: Is the code right

#### [day3#s3] Is the code right

*Session 3*

- Recover known parameters from data simulated by the model
- Run a short simulation-based calibration and read its rank histograms
- Locate and fix an implementation error from the parameter that fails

#### [day3#one-run-proves-little] One successful recovery proves little

*Is the code right*

- A colleague simulated one dataset and recovered both parameters: α and β inside their 90% intervals.
- The model still has an error. One run passes by luck as easily as by being right.
- A correct model misses the truth 10% of the time; a wrong one can cover it often.

- **Lab 3.3's starting point.** 20 northern districts, 445 km, a Poisson rate with an age effect. The simulator is right; one line of the model is not.

#### [day3#sbc-steps] If the code is right, the truth's rank among the draws is uniform

*Simulation-based calibration*

1. **1 · Simulate** parameters from the prior, then data
2. **2 · Fit** the model to those data: 99 draws
3. **3 · Rank** count draws below the true value: 0 to 99
4. **4 · Repeat** 100 times; plot each parameter's ranks

> **Shapes to read.** Flat: right. U: posterior too narrow. Hump: too wide. Slope: biased.

#### [day3#sbc-result] β's ranks pile up at both ends: its prior is five times too narrow

*Lab 3.3, tasks 3 to 5*

| Share of ranks in the outer fifth (expected 20%) | α | β |
| --- | --- | --- |
| Colleague's model | 33% (p = 0.09) | 69% (p < 0.001) |
| Fixed model | 23% (p = 0.44) | 25% (p = 0.66) |

```python
# file: the line
beta = numpyro.sample("beta", dist.Normal(0.3, 0.2 ** 2))   # a variance where the scale belongs
```

#### [day3#sbc-limits] Calibration tests the code against the simulator, not the model against the world

*What calibration does not do*

- **It finds.** Coding errors, wrong parameterisations, a sampler that cannot fit the model.
- **It misses.** The same mistake in both the simulator and the model; a model that is wrong about Yanabi.
- **It costs.** 100 fits per calibration. Small models and short chains make it a few minutes on a laptop.

#### [day3#check-sbc] Reading rank histograms

*Checkpoint*

**Question:** In a calibration of 200 simulations, a parameter's ranks form a hump: few near 0 or 99, many in the middle. What does it say?

- A. The posterior is approximately normal, as expected
- B. The posterior is too narrow
- C. The posterior is too wide: the truth sits near its middle too often
- D. The sampler needs more warm-up

#### [day3#lab-3-3] Simulation-based calibration

Lab 3.3 · 25 min

Run a short simulation-based calibration of a supplied model, locate its seeded coding error from the ranks, fix it and show the corrected model passes.

[D3_3_locate_seeded_error.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D3_3_locate_seeded_error.ipynb)

- [ ] Parameter recovery on one dataset (5 min)
- [ ] The calibration loop (8 min)
- [ ] Ranks for each parameter (5 min)
- [ ] Find and fix the line (4 min)
- [ ] Calibrate again (3 min)
- [ ] ✱ Stretch: Calibrate a derived quantity: the rate of a district whose pipes average 40 years

### Session 4: How much the priors matter

#### [day3#s4] How much the priors matter

*Session 4*

- Run a power-scaling sensitivity diagnostic
- Refit with alternative priors that someone could defend
- Summarise what changes for the decision and what does not

#### [day3#power-scaling] Make the priors a little weaker and a little stronger, and see what moves

*How much the priors matter*

- Raise the prior density to a power: 0.8 weakens it, 1.25 strengthens it.
- Reweight the existing draws by importance sampling: no refit needed.
- Report the shift of each decision quantity in posterior standard deviations.

**0.03** largest shift, in posterior sd (rural rate)

#### [day3#three-priors] Three priors on the spread, one list of districts

*Lab 3.4, tasks 2 and 3*

| Prior on τ | τ (median) | Rural, no sensors, rate | Top 15 shared with base |
| --- | --- | --- | --- |
| HalfNormal(0.2) | 0.31 | 0.235 | 15 |
| HalfNormal(0.5), base | 0.32 | 0.232 | 15 |
| HalfNormal(2.0) | 0.32 | 0.229 | 15 |

With 140 districts, the data pin τ; the prior on the spread does not change the plan.

#### [day3#which-prior-matters] The prior that matters is the one on reporting, and it rests on one audit

*The summary for Nouran*

1. The priors on spread and shape barely move the decision quantities.
2. The reporting prior moves the rural no-sensor rates almost one for one (Lab 2.4's stretch).
3. So the money for better information goes to reporting, not to refining the model: Day 4, session 3.

#### [day3#check-sensitivity] Reporting sensitivity

*Checkpoint*

**Question:** Power-scaling shows that no quantity shifts by more than 0.03 posterior sd. Nouran asks whether the priors can now be ignored in the memo.

- A. Yes: the analysis is insensitive to its priors
- B. Yes, and the analysis could have used maximum likelihood instead
- C. No: sensitivity cannot be known until new data arrive
- D. No: state the priors tested, and name the reporting prior, which the data cannot check

#### [day3#lab-3-4] Prior sensitivity

Lab 3.4 · 25 min

Run a power-scaling diagnostic, refit with two other priors on the district spread, and summarise what changes for the rural rates and the ranking of districts.

[D3_4_run_prior_sensitivity.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D3_4_run_prior_sensitivity.ipynb)

- [ ] The base fit and the power-scaling diagnostic (6 min)
- [ ] Two alternative priors on the spread (7 min)
- [ ] What changes: spread, rural rates, top 15 (7 min)
- [ ] A summary for the director (5 min)
- [ ] ✱ Stretch: Do the same for the prior on the negative binomial shape

#### [day3#wrap-up] Three things to keep

*Day 3 · wrap-up*

- **Check what the decision needs.** Choose statistics the fit does not force; the variation from year to year sank the Poisson.
- **Test the code, not one run.** Calibration over many simulations found a line that one recovery missed.
- **Name the prior that matters.** Spread and shape priors barely move the plan; the reporting prior does.

#### [day3#next] Day 4: From posterior to decision

*Next*

Tomorrow turns the posterior into a capital plan. Read the consultant's report in labs/data/review before you come: Lab 4.4 reviews it.

## Day 4: From posterior to decision

Turn the posterior into a re-piping plan within budget, price the survey that would sharpen it, and review someone else’s analysis

#### [day4#title] From posterior to decision

*SDAIA Academy · Bayesian Modelling and Decisions under Uncertainty*

Day 4: Turn the posterior into a re-piping plan within budget, price the survey that would sharpen it, and review someone else’s analysis

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-401

#### [day4#agenda] From posterior to decision

*Today*

- **Session 1 · Loss, utility and the decision table**: Actions, costs and outcomes, and expected utility computed over posterior draws
- **Session 2 · Choosing under a budget**: A plan for the capital budget with avoided bursts and their uncertainty
- **Session 3 · What more data is worth**: Expected value of perfect and of sample information, against the price of the data
- **Session 4 · Reviewing a Bayesian analysis**: A checklist for priors, computation, checking and reporting, and flaws ranked by their effect on the decision

#### [day4#review-day3] What the model choice rests on

*Review of Day 3*

**Question:** Day 4 uses the hierarchical negative binomial with the reporting term. Which piece of yesterday's evidence would most change today's plan had it gone the other way?

- A. The replicated network total matching 3,894 bursts
- B. The power-scaling of the prior on τ
- C. The dispersion check: a Poisson would make every district look more certain than it is
- D. The calibration of the colleague's model

### Session 1: Loss, utility and the decision table

#### [day4#s1] Loss, utility and the decision table

*Session 1*

- Write a decision table of actions, costs and consequences
- Compute expected cost over posterior draws for each action
- Show where a decision on the posterior mean differs from one on the whole posterior

#### [day4#decision-table] Three actions, one cost each, computed for every posterior draw

*Loss, utility and the decision table*

```python
# file: Lab 4.1
cost(action) = cost_per_km * km + rate * km * 80_000 * (20 - years * reduction)
```

| Action | SAR per km | Fewer bursts | For |
| --- | --- | --- | --- |
| Wait | 0 | 0% | – |
| Repair weak joints, manage pressure | 40,000 | 30% | 10 years |
| Re-pipe the mains | 400,000 | 90% | 20 years |

Synthetic planning numbers (data/costs.json); a burst costs SAR 80,000; horizon 20 years.

#### [day4#eu-steps] Average the cost over the posterior, then choose

*Step by step*

1. **1 · Draws** 2,000 posterior rates for each of 140 districts
2. **2 · Costs** each action's cost in each draw: 3 × 2,000 × 140
3. **3 · Choose** the action with the lowest expected cost in each district

> **Result.** Wait in 87 districts, repair 38, re-pipe 15.

#### [day4#plug-in] Deciding on the median rate gets three rural districts wrong

*Lab 4.1, tasks 4 and 5*

| District | Kind | Sensors | Median plug-in | Expected utility | Mean / median |
| --- | --- | --- | --- | --- | --- |
| D045 | rural | yes | wait | repair | 1.05 |
| D069 | rural | no | repair | re-pipe | 1.03 |
| D091 | rural | no | wait | repair | 1.03 |

The mean plug-in agrees everywhere: these costs are straight lines in the rate. The median sits below the mean of a skewed posterior.

#### [day4#check-eu] Expected cost or cost at the estimate

*Checkpoint*

**Question:** An extra cost applies only when a district exceeds 40 bursts in a year: a fine from the regulator. Does deciding on the posterior mean rate still give the expected-utility choice?

- A. Yes: the mean plug-in is always the same as expected utility
- B. Not in general: the fine is not linear in the rate, so its expectation depends on the whole posterior
- C. No, but the median plug-in would
- D. No decision can be made with a threshold cost

#### [day4#lab-4-1] Replace, repair or wait

Lab 4.1 · 25 min

Compute the expected 20-year cost of replacing, repairing or waiting for every district over the supplied posterior and compare with decisions made on the posterior mean.

[D4_1_compute_expected_utility.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D4_1_compute_expected_utility.ipynb)

- [ ] Load the posterior and the costs (3 min)
- [ ] The cost of each action in each draw (7 min)
- [ ] Expected cost and the best action per district (6 min)
- [ ] Plug-in decisions against expected utility (6 min)
- [ ] Where the two disagree (3 min)
- [ ] ✱ Stretch: Decide on the 90th percentile of cost instead of the mean, and say who would choose that

### Session 2: Choosing under a budget

#### [day4#s2] Choosing under a budget

*Session 2*

- Rank actions by expected benefit per riyal
- Choose a plan that fits a capital budget
- Report avoided bursts with an interval, and who carries the remaining risk

#### [day4#budget-bind] The best plan for each district costs SAR 61 million; the budget is 40

*Choosing under a budget*

- Each district's best action ignores the others; the budget couples them.
- Rank steps by expected net saving per riyal: wait to repair, repair to re-pipe.
- Fill the budget in that order; a district's second step comes only after its first.

- **61.4** SAR million for the unconstrained plan
- **40.0** SAR million capital budget

#### [day4#the-plan] 43 districts within the budget, with an interval on what they avoid

*Lab 4.2, tasks 3 and 4*

- **34 + 9** repairs and re-pipings, SAR 40.0 million
- **18.2** SAR million expected net saving over 20 years
- **719** bursts avoided, 90% interval 607 to 870

> **The interval is part of the plan.** A board that hears "719" will remember it; a board that hears "between 600 and 870" can plan for both.

#### [day4#remaining-risk] Name who carries the risk the plan leaves, and where the model knows least

*Responsible use: Lab 4.2, task 5*

- **Largest unfunded risks.** Large urban districts with sensors (D116, D087, D014): many bursts expected, but each kilometre is cheap to leave. Their residents and crews carry the bursts the plan does not prevent.
- **Least known.** Mostly rural districts without sensors: their rates rest on the audit prior. The memo names them, and session 3 prices finding out.

#### [day4#check-budget] Filling a budget

*Checkpoint*

**Question:** District A: repair costs 1 million and saves 1.5 million; re-piping costs 10 million and saves 4 million in all. Funds are short. Which step is ranked by which ratio?

- A. Re-pipe at 0.4 per riyal, as one step
- B. Only the repair is ever considered
- C. Re-pipe first because it saves the most
- D. Repair at 1.5 per riyal; then the upgrade to re-pipe at 2.5 / 9 ≈ 0.28 per riyal

#### [day4#lab-4-2] A plan within the budget

Lab 4.2 · 25 min

Choose the districts to re-pipe or repair within the capital budget, with avoided bursts and their uncertainty, and name the districts the model knows least about.

[D4_2_choose_districts_re.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D4_2_choose_districts_re.ipynb)

- [ ] The plan without a budget, and its cost (4 min)
- [ ] Benefit per riyal (5 min)
- [ ] Fill the budget (6 min)
- [ ] Avoided bursts with an interval (6 min)
- [ ] Who carries the remaining risk (4 min)
- [ ] ✱ Stretch: Solve the budget exactly by dynamic programming and compare with the greedy plan

### Session 3: What more data is worth

#### [day4#s3] What more data is worth

*Session 3*

- Compute the expected value of perfect information for a decision
- Compute the expected value of a survey by preposterior simulation
- Give a verdict on an offer of data by comparing its value with its price

#### [day4#voi-idea] Information is worth what it would change in the decision

*What more data is worth*

1. **Perfect information** Decide in each draw knowing the true rate; average; subtract today's best
2. **The real survey** Simulate its result (15% error), update, decide, average
3. **Against its price** Buy where the value exceeds SAR 45,000 per district

> **No change, no value.** A district whose best action holds across its whole posterior gains nothing from a survey.

#### [day4#survey-numbers] Decline the full survey; propose 16 districts

*Lab 4.3, tasks 2 to 4*

|  | Price, SAR million | Value, SAR million |
| --- | --- | --- |
| Perfect information, 76 districts | – | 2.38 |
| Full survey as offered, 76 districts | 3.42 | 1.81 |
| Targeted survey, 16 districts | 0.72 | 1.21 |

The 16 are mostly rural districts close to a decision threshold, with uncertain rates.

#### [day4#check-voi] Which district to survey

*Checkpoint*

**Question:** Two districts without sensors. X: rate very uncertain, but re-piping is best in every draw. Y: rate moderately uncertain, and repair or wait each wins in about half the draws. Where is a survey worth more?

- A. Y: the survey can change its decision; X's decision is already settled
- B. X: it is the more uncertain
- C. Both the same: a survey costs the same in each
- D. Neither: surveys only confirm a model

#### [day4#lab-4-3] The price of the leak survey

Lab 4.3 · 25 min

Price the acoustic leak survey offered for districts without sensors by its value of information and give a verdict on the offer.

[D4_3_price_leak_survey.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D4_3_price_leak_survey.ipynb)

- [ ] The districts the survey covers (3 min)
- [ ] Perfect information, district by district (6 min)
- [ ] Simulate the survey before buying it (8 min)
- [ ] The survey’s value against its price (5 min)
- [ ] The verdict (3 min)
- [ ] ✱ Stretch: Buy the survey only where its value exceeds its price, and price that smaller offer

### Session 4: Reviewing a Bayesian analysis

#### [day4#s4] Reviewing a Bayesian analysis

*Session 4*

- Review an analysis against a checklist of priors, computation, checking and reporting
- Rank flaws by their effect on the decision
- Choose and scope a capstone brief

#### [day4#review-checklist] Six questions, in the order of the workflow

*Reviewing a Bayesian analysis*

- **Priors.** Stated, justified, checked by prior predictive simulation?
- **Computation.** Four chains, R-hat under 1.01, ESS, divergences explained?
- **Structure.** Pooling suited to unequal groups, and defended?
- **Measurement.** Is what was recorded what happened?
- **Checking.** Predictive checks and a comparison of models?
- **Decision.** Costs, uncertainty, remaining risk, value of data?

#### [day4#consultant-top] The consultant's top six, raw and pooled

*Lab 4.4, task 3*

| District | Kind | Sensors | Raw rate | Posterior median (90%) |
| --- | --- | --- | --- | --- |
| D010 | rural | no | 0.51 | 0.39 (0.26 to 0.60) |
| D017 | suburban | yes | 0.50 | 0.40 (0.30 to 0.54) |
| D073 | rural | yes | 0.50 | 0.25 (0.15 to 0.39) |
| D113 | rural | no | 0.42 | 0.32 (0.19 to 0.53) |
| D067 | rural | no | 0.41 | 0.38 (0.23 to 0.60) |
| D118 | rural | no | 0.40 | 0.41 (0.25 to 0.67) |

#### [day4#rank-flaws] Rank the flaws by what they do to the decision, not by how wrong they look

*Lab 4.4, task 3*

1. **Structure.** No pooling: small districts top the list on noise; 13 zero-burst districts are declared safe.
2. **Measurement.** Unrecorded bursts read as low risk in 76 districts.
3. **Decision.** A ranking of means taken as the plan: no costs, intervals, remaining risk or survey price.
4. **Computation.** Two chains, R-hat against 1.1, 37 divergences dismissed.
5. **Priors** and **checking**: Normal(0, 100) called uninformative; R² as fit.

#### [day4#check-review] Writing one finding

*Checkpoint*

**Question:** Which sentence is a usable review finding?

- A. The priors are not good enough
- B. The report uses Normal(0, 100)
- C. Each district is estimated alone, so D073 (a few bursts on a short network) ranks third; pooled, its rate halves and it leaves the list
- D. The analysis should have used NumPyro

#### [day4#lab-4-4] Review an analysis, choose a brief

Lab 4.4 · 25 min

Review a consultant’s Bayesian analysis of Yanabi’s bursts by checklist, rank the flaws you find by their effect on the decision, and choose your capstone brief.

[D4_4_review_supplied_analysis.ipynb](https://sdaiaacademy.github.io/ds-401/labs/D4_4_review_supplied_analysis.ipynb)

- [ ] Read the analysis (6 min)
- [ ] The checklist, item by item (8 min)
- [ ] Rank the flaws by their effect on the decision (6 min)
- [ ] Choose your capstone brief (5 min)
- [ ] ✱ Stretch: Rerun one of the consultant’s claims and show the number changes

#### [day4#wrap-up] Three things to keep

*Day 4 · wrap-up*

- **Average the cost, then choose.** And report the plan's outcome as an interval, with who carries the risk it leaves.
- **Price information by the decision.** Data are worth buying only where they can change what you do.
- **Review by effect.** Quote, show the evidence, say what changes.

#### [day4#next] Capstone: your own model and decision memo

*Next*

Start your capstone repository on the brief you chose in Lab 4.4. Read labs/capstone/README.md for the structure and the rubric.

## Capstone: Capstone project presentations

An eight-minute presentation of the model and the decision, then seven minutes of questions: the assessor changes a prior or a cost, and a peer reviews the model

#### [capstone#title] Capstone project presentations

*SDAIA Academy · Bayesian Modelling and Decisions under Uncertainty*

Capstone: An eight-minute presentation of the model and the decision, then seven minutes of questions: the assessor changes a prior or a cost, and a peer reviews the model

Data Science Track · 4 teaching days + capstone day · 8 min talk and demo · 7 min Q&A · DS-401

#### [capstone#agenda] How the orals run

*Capstone*

- **Expectations and final prep**: the rubric, the oral's two questions, a last rehearsal
- **Orals**: a presentation, then questions with a changed prior or cost and the peer's review
- **Review and closing**: the trainer's review and the follow-up task

#### [capstone#rubric] The rubric

*How the repository is scored*

| Item | Value |
| --- | --- |
| Model and priors | 20% |
| Computation | 20% |
| Checking | 25% |
| Decision | 25% |
| Reproducibility | 10% |

> **Every claim carries a number from the run log.** R-hat, divergences, a predictive p-value, an elpd difference with its standard error, an interval on the decision.

#### [capstone#format] 8 minutes to present, 7 to defend

*Format*

- **1 · Brief and model.** The decision maker's question; the structure and the priors you would defend.
- **2 · Evidence.** Diagnostics, the check that mattered, the comparison of models.
- **3 · Decision.** The choice, its interval, the remaining risk, what more data is worth.

> **The questions.** The assessor changes one prior or one cost; you predict the effect on the decision, then rerun. Then your peer asks the question from their review of your model.

#### [capstone#what-good-looks-like] Predict in words first, then check with the model

*What a strong defence sounds like*

- **Strong.** "Doubling the value of a serious finding raises the stakes in schools with one visit; I expect three or four of them to enter the 30 visits, at the cost of large schools with known moderate rates." Then the rerun.
- **Weak.** "The results should be robust." Or: rerunning before saying anything, then describing what changed.

#### [capstone#running-order] Who presents when, and who reviews whom

*Running order*

| # | Presenter | Brief | Peer reviewer |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |
| 6 |  |  |  |
| 7 |  |  |  |
| 8 |  |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first oral*

Rehearse once; check that one cell of your notebook reruns in under two minutes with a changed prior or cost.

#### [capstone#now-presenting] Presentation

*Now presenting*

##### Questions

#### [capstone#review] What the orals showed

*Trainer review*

- **Strongest patterns.** What several participants got right.
- **Common gaps.** What most repositories would fix next.
- **One surprise.** A prediction that the rerun overturned.

#### [capstone#what-next] The follow-up task, within three weeks

*After the course*

- **Review one real uncertainty analysis** from your workplace or a public report, with the checklist of Lab 4.4.
- **Submit at most 800 words:** the findings, quoting the analysis, ranked by their effect on the decision, and a three-sentence verdict. labs/followup/README.md has the details.
- **Keep your capstone repository reproducible:** it is the evidence of the course's outcomes.

#### [capstone#thanks] Thank you

*SDAIA Academy*

Bayesian Modelling and Decisions under Uncertainty

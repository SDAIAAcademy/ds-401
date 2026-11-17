---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/slides/day1.html"
---

# Bayesian Modelling and Decisions under Uncertainty

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

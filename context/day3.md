---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/slides/day3.html"
---

# Bayesian Modelling and Decisions under Uncertainty

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

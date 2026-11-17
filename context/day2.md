---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/slides/day2.html"
---

# Bayesian Modelling and Decisions under Uncertainty

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

---
course: "Bayesian Modelling and Decisions under Uncertainty"
code: "DS-401"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-401/slides/day4.html"
---

# Bayesian Modelling and Decisions under Uncertainty

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

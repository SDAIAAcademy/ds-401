# DS-401 answer key, for the trainer

Everything in `labs/data/` is synthetic, written by `labs/prepare_data.py` with seed 401; the supplied fits and the Day 4 posterior are written by `labs/_tools/make_reference.py`. The numbers below come from the course's solution runs in quick mode (`DS401_QUICK=1`: four chains of 500 warm-up and 500 kept draws). A full run gives numbers close to these, not identical.

## The generating values (`data/truth.json`)

| Parameter | Value |
|---|---|
| Base rate, PVC network of mean age 30 | 0.10 bursts per km-year (α = log 0.10 = -2.303) |
| Age effect | 0.30 per 10 years on the log scale (ratio 1.35) |
| Material effects (log scale, PVC = 0) | ductile iron 0.15, cast iron 0.45, asbestos cement 0.70 |
| District spread after material and age, τ | 0.30 |
| Year-to-year variation, negative binomial shape φ | 6 |
| Reporting share where there are no sensors | Beta(14, 6) per district (mean 0.70); 1 where there are sensors |
| Trunk-main failure | D017, 2023: 38 extra bursts |

140 districts: 44 urban, 40 suburban, 56 rural; 76 without sensors (51 of them rural); 51 districts with fewer than five recorded bursts in five years; 205 of 700 district-years with none.

## Planted problems and the lab that finds each

| Problem | Where | Found in | Expected result |
|---|---|---|---|
| An absurd "flat" prior and a dogmatic one | Lab 1.2's candidates | Lab 1.2 | N(0, 10): 37% of district-years above 500; N(log 0.5, 0.05): 0% of network totals plausible |
| Two untrustworthy fits | `data/fits/fit_B.npz`, `fit_C.npz` | Lab 1.4 | B: 741 divergences, R-hat 1.37; C: R-hat 1.60, chain 4 stuck at α ≈ -1.23, no divergence. D is sound (33% capped trees) |
| Rural districts pulled towards the urban-dominated rate | partial pooling without predictors | Labs 2.1, 2.2 | rural estimates keep 16% of their true gap; mean log(estimate / truth) -0.61, then -0.25 with material and age |
| The funnel | centred model, the 49 districts under 4 km | Lab 2.3 | 59 divergences, all at small τ; non-centred: 0 |
| Under-reporting without sensors | 76 districts | Lab 2.4 | rural no-sensor bias -0.28 removed (-0.02); the reporting share is carried by the prior (sd ratio 0.98) |
| Extra year-to-year variation | all districts | Lab 3.1 | dispersion 2.32 against Poisson replicates 0.87 to 1.13 (p = 0.000); quiet years 0.048, at the edge of the Poisson range (p = 0.12); negative binomial φ = 7.0 (5.6 to 9.0), p = 0.34 and 0.48 |
| The trunk-main year | D017, 2023 (54 bursts against 18 to 27) | Lab 3.2 | k = 0.84 under the hierarchical Poisson (6 points above 0.7), not flagged by the negative binomial (1 point). elpd: negative binomial -1,393; Poisson hierarchy -110 (SE 23); complete pooling -317 (SE 54) |
| A seeded coding error | `colleague_model` in Lab 3.3: `dist.Normal(0.3, 0.2 ** 2)` for β, a variance where the scale belongs | Lab 3.3 | β: 69% of ranks in the outer fifth (expected 20%), p &lt; 0.001; α: 33%, p = 0.09 (pulled by β); fixed model: p = 0.44 and 0.66 |
| Six flaws in the consultant's analysis | `data/review/consultant_analysis.md` | Lab 4.4 | see below |

## Day 4

Costs (`data/costs.json`, synthetic): SAR 80,000 per burst; 20-year horizon; repair SAR 40,000 per km, 30% fewer bursts for 10 years; replace SAR 400,000 per km, 90% fewer for 20 years; capital budget SAR 40 million; survey SAR 45,000 per district, 15% error on the log rate.

- Lab 4.1: expected utility chooses wait 87, repair 38, replace 15; the median plug-in differs in 3 districts (D045, D069, D091), the mean plug-in in none (costs are linear in the rate).
- Lab 4.2: unconstrained plan SAR 61.4 million; within the budget, 34 repairs and 9 replacements, SAR 40.0 million, expected net saving SAR 18.2 million (the exact optimum by dynamic programming is the same); 719 bursts avoided over 20 years (90% interval 607 to 870). The largest unfunded risks are urban districts with sensors (D116, D087, D014).
- Lab 4.3: full survey SAR 3.42 million against EVSI SAR 1.81 million (EVPI 2.38): decline as offered; worth its price in 16 districts (EVSI SAR 1.21 million for SAR 0.72 million): propose a targeted survey.

## Lab 4.4: the six flaws, ranked by effect on the decision

1. **Structure.** No pooling: tiny districts top the list on noise (D073: raw 0.50, posterior median 0.25), 13 zero-burst districts declared safe although several have a posterior probability above 0.7 of exceeding 0.2.
2. **Measurement.** Districts without sensors treated as fully reported: their risk is understated by about a factor of 1.4.
3. **Decision.** Ranking of posterior means taken as the plan: no costs, no intervals, no remaining risk, the survey dismissed without pricing; D017's raw rate includes the trunk-main year, and re-piping it would take SAR 23 million of the 40.
4. **Computation.** Two short chains, R-hat against 1.1, 37 divergences dismissed.
5. **Priors.** Normal(0, 100) on the log rate called uninformative, no prior predictive check.
6. **Checking.** R² of observed against fitted offered as fit; no predictive check or model comparison.

A participant passes LO6 with at least four found and a ranking whose top two change the list of districts. Other rankings are acceptable when argued by effect on the decision.

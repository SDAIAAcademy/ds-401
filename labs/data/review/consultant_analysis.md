# Burst risk and re-piping priorities, Yanabi Water Authority

Prepared by Sakhr Al-Nassar, consultant analyst, for the asset management directorate. Synthetic
document written for the SDAIA Academy course DS-401; the analysis, its author and its numbers are
invented, and its flaws are deliberate.

## 1. Summary

A Bayesian model of five years of burst records (2021 to 2025) for all 140 districts identifies
the 20 districts with the highest burst rate. We recommend re-piping these 20 districts first.
The model converged and fits the data well (R² of 0.93 between observed and fitted counts), so the
estimates are reliable and no further data collection, such as the acoustic survey under
discussion, is needed.

## 2. Data

Reported bursts per district and year from the asset register, joined to network length, dominant
pipe material and mean pipe age. All 700 district-years were used as recorded.

## 3. Model

Each district's burst rate was estimated separately, so that every district is judged on its own
record and no district is "averaged away":

    log rate_j ~ Normal(0, 100)                 for each district j, independently
    bursts_jt ~ Poisson(rate_j × km_j)

The prior Normal(0, 100) on the log rate is uninformative and lets the data speak.

## 4. Computation

The model was fitted with NUTS, 2 chains of 200 warm-up and 500 draws each. All R-hat values were
below 1.1, which shows convergence. The sampler reported 37 divergent transitions; these are a
known nuisance of the sampler and do not affect the estimates.

## 5. Results

The highest estimated rates (posterior mean, bursts per km of main per year):

| District | Kind | km | Bursts, 5 years | Estimated rate |
|---|---|---|---|---|
| D010 | rural | 4.7 | 12 | 0.51 |
| D017 | suburban | 58.2 | 146 | 0.50 |
| D073 | rural | 2.8 | 7 | 0.50 |
| D113 | rural | 1.9 | 4 | 0.42 |
| D067 | rural | 2.9 | 6 | 0.41 |
| D118 | rural | 1.5 | 3 | 0.40 |

Thirteen districts recorded no burst in five years; their estimated rate is close to zero and they
need no action for the foreseeable future.

Districts with and without burst sensors were treated alike: a recorded burst is a burst.

## 6. Recommendation

Re-pipe the 20 districts with the highest posterior mean rate, starting with D010, D017 and D073.
The posterior means are the best estimates of each district's rate, so the ranking is the
decision. Uncertainty intervals are omitted from this summary because the model has converged.

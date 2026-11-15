# Bayesian Modelling and Decisions under Uncertainty: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Models that generate data | [`D1_1_simulate_burst_counts.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D1_1_simulate_burst_counts.ipynb) |
| Day 1 | Session 2 · Priors you can defend | [`D1_2_run_prior_predictive.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D1_2_run_prior_predictive.ipynb) |
| Day 1 | Session 3 · Fitting with Hamiltonian Monte Carlo | [`D1_3_fit_poisson_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D1_3_fit_poisson_model.ipynb) |
| Day 1 | Session 4 · Reading the computation | [`D1_4_flag_two_untrustworthy.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D1_4_flag_two_untrustworthy.ipynb) |
| Day 2 | Session 1 · No, complete and partial pooling | [`D2_1_fit_all_three.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D2_1_fit_all_three.ipynb) |
| Day 2 | Session 2 · Group-level predictors | [`D2_2_add_pipe_material.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D2_2_add_pipe_material.ipynb) |
| Day 2 | Session 3 · The funnel and non-centred models | [`D2_3_repair_divergent_model.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D2_3_repair_divergent_model.ipynb) |
| Day 2 | Session 4 · Under-reporting as measurement error | [`D2_4_add_reporting_rate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D2_4_add_reporting_rate.ipynb) |
| Day 3 | Session 1 · Posterior predictive checks | [`D3_1_expose_poisson_misfit.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D3_1_expose_poisson_misfit.ipynb) |
| Day 3 | Session 2 · Comparing models by predictive accuracy | [`D3_2_compare_three_models.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D3_2_compare_three_models.ipynb) |
| Day 3 | Session 3 · Is the code right | [`D3_3_locate_seeded_error.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D3_3_locate_seeded_error.ipynb) |
| Day 3 | Session 4 · How much the priors matter | [`D3_4_run_prior_sensitivity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D3_4_run_prior_sensitivity.ipynb) |
| Day 4 | Session 1 · Loss, utility and the decision table | [`D4_1_compute_expected_utility.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D4_1_compute_expected_utility.ipynb) |
| Day 4 | Session 2 · Choosing under a budget | [`D4_2_choose_districts_re.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D4_2_choose_districts_re.ipynb) |
| Day 4 | Session 3 · What more data is worth | [`D4_3_price_leak_survey.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D4_3_price_leak_survey.ipynb) |
| Day 4 | Session 4 · Reviewing a Bayesian analysis | [`D4_4_review_supplied_analysis.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/D4_4_review_supplied_analysis.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.

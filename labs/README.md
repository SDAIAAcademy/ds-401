# Bayesian Modelling and Decisions under Uncertainty: labs

Sixteen labs, one per session, on Yanabi Water Authority, a fictional water utility with 140 districts and five years of burst records. Every lab runs on a laptop CPU or in Google Colab. No language model is called and nothing is downloaded beyond the Python packages.

## Quick start

**Google Colab.** Open a notebook from the table below; its first cell clones this repository and installs NumPyro and ArviZ (`requirements-colab.txt`). The rest is already in Colab.

**Locally, with uv** (Python 3.12 or 3.13):

```bash
git clone https://github.com/SDAIAAcademy/ds-401.git
cd ds-401/labs
uv sync                                   # creates the environment from uv.lock
uv run jupyter lab
```

On Windows, if the course folder sits deep in your user folder, keep the environment on a short path: `set UV_PROJECT_ENVIRONMENT=C:\uvenv\ds401` (PowerShell: `$env:UV_PROJECT_ENVIRONMENT="C:\uvenv\ds401"`) before `uv sync`, and enable long paths in git with `git config core.longpaths true`.

Before Day 1, run `00_setup_check.ipynb` (a few minutes) and work through the pre-work, `P0_prework_likelihood_simulation.ipynb` (about three hours).

## The notebooks

| Unit | Session | Notebook |
|---|---|---|
| Pre-work | Likelihoods, simulation and where data may go | [`P0_prework_likelihood_simulation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/P0_prework_likelihood_simulation.ipynb) |
| Setup | Check the environment | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-401/blob/main/labs/00_setup_check.ipynb) |
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

Each lab starts from a supplied reference state (the data files, and for some labs supplied fits or a posterior), so a participant who did not finish an earlier lab starts level. If an exercise is left unfinished, the lab prints a notice and continues with a reference version: recomputed by the helper module, or read from the course's reference values in `data/reference/`. Each lab ends with a self-check, a checkpoint with answers, and a stretch item marked ✱.

## The data

All data are synthetic, written with a fixed seed (401) by `prepare_data.py`. No real utility, district or person is described.

| File | What it holds |
|---|---|
| `data/districts.csv` | 140 districts: kind (urban, suburban, rural), network length in km, main pipe material, mean pipe age, whether the district has acoustic sensors |
| `data/bursts.csv` | Recorded bursts per district and year, 2021 to 2025 |
| `data/truth.json` | The generating values, for checking recovery; real data never has this file |
| `data/fits/` | Four supplied fits of the same model, two of them untrustworthy, with the contractor's run log (Lab 1.4) |
| `data/posterior/rates.npz` | 2,000 posterior draws of every district's rate from the model chosen on Day 3 (Day 4) |
| `data/costs.json` | Synthetic planning costs: a burst, repair, re-piping, the capital budget and the leak survey offer (Day 4) |
| `data/review/consultant_analysis.md` | The consultant's analysis reviewed in Lab 4.4 |
| `data/reference/` | The course's reference values for every lab, used when an exercise is unfinished |
| `capstone/` | The three capstone briefs, their data and the capstone instructions |

## The helper module

`yanabi.py` holds what every lab shares: data loading (`districts`, `bursts`, `panel`, `design`), the burst model with its options (`burst_model`: pooling, predictors, Poisson or negative binomial, reporting term, centred or non-centred), sampling and diagnostics (`sample`, `diagnose`, `trustworthy`), and the self-checks (`check`, `note`). Two environment variables change how labs run:

| Variable | Effect |
|---|---|
| `DS401_QUICK=1` | Shorter chains (500 warm-up, 500 draws) and fewer simulations; the course's reference numbers come from this mode |
| `DS401_STRICT=1` | A failed self-check stops the notebook instead of printing a warning |

## AI tutor

Where a lab allows it, the tutor Mirsad explains diagnostics and errors; it writes no model code. `tutor_instruction.md` holds its instructions, to paste into an assistant your Academy has approved. The first lab of each day, the pre-work entry quiz and the supervised check are done without it.

## After the course

`followup/README.md` describes the follow-up task: a review of one real uncertainty analysis within three weeks.

## For trainers

`solutions/` holds every notebook with its solution, and `solutions/answer_key.md` lists the planted problems, the expected numbers and the key of the consultant review. The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions. Edit the sources, not the notebooks. `_tools/make_reference.py` rebuilds the supplied fits and the Day 4 posterior.

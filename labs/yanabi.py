"""Helpers for the DS-401 labs: the data of Yanabi Water Authority, sampling, diagnostics, checks.

    import yanabi as U

NumPy, pandas, JAX, NumPyro and ArviZ. Everything runs on a laptop processor or a Colab CPU
runtime. Nothing calls or downloads a language model.

Environment variables
    DS401_QUICK=1      fewer warm-up and posterior draws, fewer simulations, for a fast run
    DS401_STRICT=1     solution runs: a failed check, or a safety net taken, stops the run
    DS401_WRITE_REF=1  the trainer's run that writes data/reference/ (the safety nets read it)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):  # a Windows console or a redirected run may not default to UTF-8
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore", category=FutureWarning)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: E402
import arviz as az  # noqa: E402

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"

QUICK = os.environ.get("DS401_QUICK", "0") == "1"
STRICT = os.environ.get("DS401_STRICT", "0") == "1"
WRITE_REF = os.environ.get("DS401_WRITE_REF", "0") == "1"
MODE = "quick" if QUICK else "full"
SEED = 401
MATERIALS = ["PVC", "ductile_iron", "cast_iron", "asbestos_cement"]
LOG_RATE_CENTRE = float(np.log(0.10))  # Shabib's "about one burst per 10 km a year", the prior centre


def pick(full, quick):
    """The full value, or the small one in quick mode."""
    return quick if QUICK else full


def rng(offset: int = 0) -> np.random.Generator:
    """A seeded NumPy generator. The same offset gives the same draws on every run."""
    return np.random.default_rng(SEED + offset)


def key(offset: int = 0):
    """A seeded JAX key."""
    return jax.random.PRNGKey(SEED + offset)


# ---------------------------------------------------------------- checks and safety nets

def check(ok, what: str) -> bool:
    """Print a tick or a cross. In a solution run (DS401_STRICT=1) a cross stops the run."""
    ok = bool(ok)
    print(("✅ " if ok else "❌ ") + what)
    if STRICT and not ok:
        raise AssertionError(what)
    return ok


def note(*lines: str) -> None:
    """Print the lab's closing notes."""
    for line in lines:
        print("•", line)


def _jsonable(v):
    if isinstance(v, pd.DataFrame):
        return {"__frame__": json.loads(v.to_json(orient="split"))}
    if isinstance(v, pd.Series):
        return {"__series__": json.loads(v.to_json(orient="split"))}
    if isinstance(v, (np.ndarray, jnp.ndarray)):
        return {"__array__": np.asarray(v).tolist()}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _labels(xs):
    """Index labels from JSON: lists of lists were tuples of a MultiIndex."""
    if xs and all(isinstance(x, list) for x in xs):
        return pd.MultiIndex.from_tuples([tuple(x) for x in xs])
    return xs


def _unjson(v):
    if isinstance(v, dict):
        if "__array__" in v:
            return np.array(v["__array__"])
        if "__frame__" in v:
            f = v["__frame__"]
            return pd.DataFrame(f["data"], index=_labels(f["index"]), columns=_labels(f["columns"]))
        if "__series__" in v:
            f = v["__series__"]
            return pd.Series(f["data"], index=_labels(f["index"]), name=f.get("name"))
        return {k: _unjson(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unjson(x) for x in v]
    return v


def keep(lab: str, name: str, value) -> None:
    """In the trainer's reference run, store a result for the safety net of later cells."""
    if not WRITE_REF:
        return
    REF.mkdir(parents=True, exist_ok=True)
    path = REF / f"{lab}.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store[name] = _jsonable(value)
    path.write_text(json.dumps(store, indent=1) + "\n", encoding="utf-8")


def reference(lab: str, name: str):
    """The stored reference result: used when an exercise is not done yet."""
    print(f"↪ exercise not done yet: using the reference result {lab}/{name}")
    if STRICT:
        raise AssertionError(f"a solution run took the safety net for {lab}/{name}")
    store = json.loads((REF / f"{lab}.json").read_text(encoding="utf-8"))
    return _unjson(store[name])


def safety(name: str) -> None:
    """Say that a safety net is being taken (for nets that recompute instead of reading a file)."""
    print(f"↪ exercise not done yet: using the reference version of {name}")
    if STRICT:
        raise AssertionError(f"a solution run took the safety net for {name}")


# ---------------------------------------------------------------- the data

def districts() -> pd.DataFrame:
    """140 districts: kind, network length, dominant material, mean pipe age, sensors."""
    d = pd.read_csv(DATA / "districts.csv")
    d["material"] = pd.Categorical(d["material"], categories=MATERIALS)
    d["sensors"] = d["sensors"].astype(str).str.lower().eq("true")
    return d


def bursts() -> pd.DataFrame:
    """700 district-years, 2021 to 2025: the bursts that were reported."""
    return pd.read_csv(DATA / "bursts.csv")


def panel() -> pd.DataFrame:
    """bursts joined to districts: one row per district-year."""
    return bursts().merge(districts(), on="district_id", how="left")


def truth() -> dict:
    """The generating parameters. Labs compare their estimates with them; real data has none."""
    return json.loads((DATA / "truth.json").read_text(encoding="utf-8"))


def costs() -> dict:
    """Costs, horizon, capital budget and the survey offer (synthetic)."""
    return json.loads((DATA / "costs.json").read_text(encoding="utf-8"))


def design(p: pd.DataFrame | None = None) -> dict:
    """Arrays for the models: one row per district-year and one per district.

    d       district index of each row (0 to J-1)
    log_km  log network length of each row's district (the exposure of one year)
    y       reported bursts of each row
    X       district-level predictors: (age - 30) / 10 and three material indicators (PVC is the base)
    sensors district has sensors (True) or not
    """
    p = panel() if p is None else p
    dd = p.drop_duplicates("district_id").sort_values("district_id").reset_index(drop=True)
    idx = {k: i for i, k in enumerate(dd["district_id"])}
    age = (dd["mean_age_years"].to_numpy(float) - 30) / 10
    mats = np.column_stack([(dd["material"].astype(str) == m).to_numpy(float) for m in MATERIALS[1:]])
    return {
        "d": p["district_id"].map(idx).to_numpy(np.int32),
        "log_km": np.log(p["network_km"].to_numpy(float)),
        "y": p["bursts_reported"].to_numpy(np.int32),
        "X": np.column_stack([age, mats]),
        "sensors": dd["sensors"].to_numpy(bool),
        "km": dd["network_km"].to_numpy(float),
        "J": len(dd),
        "ids": dd["district_id"].tolist(),
        "kind": dd["kind"].tolist(),
    }


X_NAMES = ["age per 10 years", "ductile iron", "cast iron", "asbestos cement"]


# ---------------------------------------------------------------- the reference models

def burst_model(d, log_km, X, sensors, J, y=None, *, pooling="partial", predictors=True,
                family="poisson", reporting=False, centered=False, tau_scale=0.5,
                alpha_sd=1.0, beta_sd=0.5):
    """The course's burst model in one function, every variant a keyword.

    log E[bursts of district j in one year] = log km_j + eta_j (+ log p_j when reporting)
    pooling     "none": eta_j independent; "complete": one eta; "partial": eta_j ~ Normal(mu_j, tau)
    predictors  mu_j = alpha + X_j beta (otherwise alpha)
    family      "poisson" or "negbin" (year-to-year variation with shape phi)
    reporting   districts without sensors report a share p_j ~ Beta(14, 6) of their bursts
    centered    the centred form of the partial-pooling model (the non-centred one is the default)
    """
    alpha = numpyro.sample("alpha", dist.Normal(LOG_RATE_CENTRE, alpha_sd))
    mu = alpha
    if predictors and pooling != "none":
        beta = numpyro.sample("beta", dist.Normal(0.0, beta_sd).expand([X.shape[1]]).to_event(1))
        mu = alpha + X @ beta
    if pooling == "complete":
        eta = numpyro.deterministic("eta", jnp.broadcast_to(mu, (J,)))
    elif pooling == "none":
        with numpyro.plate("district", J):
            eta = numpyro.sample("eta", dist.Normal(LOG_RATE_CENTRE, 1.5))
    else:
        tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))
        if centered:
            with numpyro.plate("district", J):
                eta = numpyro.sample("eta", dist.Normal(mu, tau))
        else:
            with numpyro.plate("district", J):
                z = numpyro.sample("z", dist.Normal(0.0, 1.0))
            eta = numpyro.deterministic("eta", mu + tau * z)
    numpyro.deterministic("rate", jnp.exp(eta))
    log_mean = log_km + eta[d]
    if reporting:
        with numpyro.plate("district_r", J):
            p_raw = numpyro.sample("p_report", dist.Beta(14.0, 6.0))
        p = jnp.where(sensors, 1.0, p_raw)
        log_mean = log_mean + jnp.log(p)[d]
    if family == "negbin":
        phi = numpyro.sample("phi", dist.Gamma(2.0, 0.2))
        numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_mean), phi), obs=y)
    else:
        numpyro.sample("y", dist.Poisson(jnp.exp(log_mean)), obs=y)


def model_data(D: dict) -> dict:
    """The keyword arguments burst_model needs, from design()."""
    return {k: D[k] for k in ("d", "log_km", "X", "sensors", "J")}


# ---------------------------------------------------------------- sampling and diagnostics

def sample(model, *, warmup=None, draws=None, chains=4, seed=0, target_accept=0.8,
           max_tree_depth=10, log_likelihood=True, **data):
    """Run NUTS with four chains and return (mcmc, idata).

    The chains run vectorised in one process (one compilation). idata is an ArviZ DataTree with
    the posterior, sample statistics, observed data and, by default, the pointwise log likelihood.
    """
    warmup = pick(1000, 500) if warmup is None else warmup
    draws = pick(1000, 500) if draws is None else draws
    kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(key(seed), extra_fields=("diverging", "num_steps", "energy"), **data)
    idata = az.from_numpyro(mcmc, log_likelihood=log_likelihood)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    ss = idata["sample_stats"].to_dataset()
    ss["energy"] = (("chain", "draw"), np.asarray(ex["energy"]))
    ss["tree_depth"] = (("chain", "draw"), np.floor(np.log2(np.asarray(ex["num_steps"]) + 1)).astype(int))
    idata["sample_stats"] = ss
    return mcmc, idata


def diagnose(idata, var_names=None, max_tree_depth=10) -> dict:
    """The numbers that decide whether a fit can be trusted (max_tree_depth: the cap the run used)."""
    post = idata["posterior"]
    names = var_names or [v for v in post.data_vars if v not in ("rate",)]
    rh = az.rhat(idata, var_names=names)
    eb = az.ess(idata, var_names=names, method="bulk")
    et = az.ess(idata, var_names=names, method="tail")
    ss = idata["sample_stats"]
    div = int(np.asarray(ss["diverging"]).sum())
    out = {
        "max_rhat": float(max(np.nanmax(np.asarray(rh[v])) for v in names)),
        "min_ess_bulk": float(min(np.nanmin(np.asarray(eb[v])) for v in names)),
        "min_ess_tail": float(min(np.nanmin(np.asarray(et[v])) for v in names)),
        "divergences": div,
        "draws": int(np.asarray(ss["diverging"]).size),
    }
    if "tree_depth" in ss:
        td = np.asarray(ss["tree_depth"])
        out["tree_depth_hits"] = int((td >= max_tree_depth).sum())
    if "energy" in ss:
        e = np.asarray(ss["energy"])
        bfmi = np.sum(np.diff(e, axis=1) ** 2, axis=1) / np.sum((e - e.mean(axis=1, keepdims=True)) ** 2, axis=1)
        out["min_ebfmi"] = float(bfmi.min())
    return out


def trustworthy(diag: dict, n_chains: int = 4) -> bool:
    """The course's rule: R-hat below 1.01, bulk and tail ESS at least 100 per chain, no divergence.

    Quick mode runs chains half as long, so it accepts R-hat below 1.02 and 50 effective draws
    per chain, and says so. The course's rule is the full-mode one.
    """
    limit, per_chain = (1.02, 50) if QUICK else (1.01, 100)
    if QUICK:
        print(f"(quick mode: short chains, R-hat accepted below {limit}, ESS from {per_chain * n_chains})")
    return (diag["max_rhat"] < limit and diag["min_ess_bulk"] >= per_chain * n_chains
            and diag["min_ess_tail"] >= per_chain * n_chains and diag["divergences"] == 0)


def draws(idata, name: str) -> np.ndarray:
    """Posterior draws of one variable, chains stacked: shape (draws, ...)."""
    a = np.asarray(idata["posterior"][name])
    return a.reshape((-1,) + a.shape[2:])


def predict(model, mcmc, seed=1, **data) -> np.ndarray:
    """Posterior predictive draws of y: shape (draws, rows)."""
    pred = Predictive(model, mcmc.get_samples())
    return np.asarray(pred(key(seed), **data)["y"])


def load_fit(name: str):
    """One of the supplied fits (data/fits/<name>.npz) as an ArviZ DataTree."""
    z = np.load(DATA / "fits" / f"{name}.npz")
    post = {k[5:]: z[k] for k in z.files if k.startswith("post_")}
    stats = {k[6:]: z[k] for k in z.files if k.startswith("stats_")}
    return az.from_dict({"posterior": post, "sample_stats": stats})


def posterior_rates() -> tuple[np.ndarray, list[str]]:
    """The supplied Day 4 posterior: true bursts per km-year, shape (draws, 140), and district ids."""
    z = np.load(DATA / "posterior" / "rates.npz")
    return z["rate"].astype(float), [str(s) for s in z["district_id"]]


def describe(x: np.ndarray, prob=0.9) -> str:
    """Median and central interval of a vector of draws, as text."""
    lo, hi = np.quantile(x, [(1 - prob) / 2, 1 - (1 - prob) / 2])
    return f"{np.median(x):.3f} ({int(prob * 100)}% interval {lo:.3f} to {hi:.3f})"

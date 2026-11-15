"""The trainer's tool: write the supplied fits of Lab 1.4 and the Day 4 posterior.

    cd labs
    uv run python _tools/make_reference.py

Runs the course's reference models with NumPyro on a laptop processor (a few minutes) and writes
    data/fits/fit_A.npz ... fit_D.npz   four fits of the same question, two of them untrustworthy
    data/posterior/rates.npz            draws of each district's true burst rate (Day 4 starts here)
Fixed seeds: the same files on every run of the same library versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402

import yanabi as U  # noqa: E402

FITS = U.DATA / "fits"
POST = U.DATA / "posterior"


def _save(name: str, samples: dict, extra: dict) -> None:
    """samples: name -> array (chains, draws, ...); extra: sample stats with the same leading shape."""
    FITS.mkdir(parents=True, exist_ok=True)
    arrays = {f"post_{k}": np.asarray(v, dtype=np.float32) for k, v in samples.items()}
    arrays.update({f"stats_{k}": np.asarray(v) for k, v in extra.items()})
    np.savez_compressed(FITS / f"{name}.npz", **arrays)


def _run(model, chains=4, warmup=1000, draws=1000, seed=0, init=None, max_tree_depth=10, step=None, **data):
    opts = {"step_size": step, "adapt_step_size": False, "adapt_mass_matrix": False} if step else {}
    kernel = NUTS(model, max_tree_depth=max_tree_depth, init_strategy=init or numpyro.infer.init_to_uniform, **opts)
    mc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains,
              chain_method="vectorized", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps", "energy"), **data)
    s = mc.get_samples(group_by_chain=True)
    ex = mc.get_extra_fields(group_by_chain=True)
    stats = {"diverging": np.asarray(ex["diverging"]), "energy": np.asarray(ex["energy"]),
             "tree_depth": np.floor(np.log2(np.asarray(ex["num_steps"]) + 1)).astype(int)}
    return s, stats


def fits() -> None:
    D = U.design()
    md = U.model_data(D)
    keep = ("alpha", "beta", "tau")
    small = D["km"] < 4  # the small (mostly rural) districts: weak data, a funnel for the centred form
    p = U.panel()
    ps = p[p["network_km"] < 4]
    Ds = U.design(ps)
    mds = U.model_data(Ds)

    # A: the partial-pooling Poisson model with predictors, non-centred: trustworthy
    s, st = _run(U.burst_model, seed=11, y=D["y"], **md)
    _save("fit_A", {k: s[k] for k in keep}, st)
    print("A", U.diagnose(U.load_fit("fit_A")))

    # B: the same model, centred, on the small districts only: divergences in the funnel
    s, st = _run(lambda **kw: U.burst_model(centered=True, tau_scale=1.0, **kw), seed=12, warmup=300, y=Ds["y"], **mds)
    _save("fit_B", {k: s[k] for k in keep}, st)
    print("B", U.diagnose(U.load_fit("fit_B")))

    # C: four chains; the fourth was started far away and run without adaptation (a tiny fixed
    # step size and no warm-up): it barely moves, so R-hat is far above 1
    parts, stats = [], []
    for c in range(4):
        if c < 3:
            s, st = _run(U.burst_model, chains=1, draws=1000, seed=20 + c, y=D["y"], **md)
        else:
            s, st = _run(U.burst_model, chains=1, warmup=0, draws=1000, seed=23, step=2e-5, max_tree_depth=3,
                         init=init_to_value(values={"alpha": -1.2, "tau": 1.5}), y=D["y"], **md)
        parts.append(s)
        stats.append(st)
    s = {k: np.concatenate([q[k] for q in parts], axis=0) for k in keep}
    st = {k: np.concatenate([q[k] for q in stats], axis=0) for k in stats[0]}
    _save("fit_C", s, st)
    print("C", U.diagnose(U.load_fit("fit_C")))

    # D: the complete-pooling model with a tree depth capped at 4: many capped trees, still sound
    s, st = _run(lambda **kw: U.burst_model(pooling="complete", **kw), seed=14, max_tree_depth=4, y=D["y"], **md)
    _save("fit_D", {k: s[k] for k in ("alpha", "beta")}, st)
    print("D", U.diagnose(U.load_fit("fit_D"), max_tree_depth=4))
    del small


def posterior() -> None:
    """The Day 4 posterior: negative binomial, predictors, reporting term, non-centred."""
    D = U.design()
    md = U.model_data(D)
    kernel = NUTS(lambda **kw: U.burst_model(family="negbin", reporting=True, **kw))
    mc = MCMC(kernel, num_warmup=1000, num_samples=500, num_chains=4, chain_method="vectorized", progress_bar=False)
    mc.run(jax.random.PRNGKey(41), y=D["y"], **md)
    rate = np.asarray(mc.get_samples()["rate"], dtype=np.float32)
    POST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POST / "rates.npz", rate=rate, district_id=np.array(D["ids"]))
    print("posterior rates", rate.shape)


if __name__ == "__main__":
    what = sys.argv[1:] or ["fits", "posterior"]
    if "fits" in what:
        fits()
    if "posterior" in what:
        posterior()

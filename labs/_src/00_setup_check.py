# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "matplotlib>=3.9", "jax>=0.7", "numpyro>=0.19", "arviz>=1.0"]
# ///
# %% [markdown]
# # Setup check
#
# Run this once before Day 1. It imports every library the labs use, reads the case-study data, fits a tiny model with four chains and prints what is missing. No language model is called and no model is downloaded.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-401.git"
    if not Path("/content/ds-401").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-401"], check=True)
    os.chdir("/content/ds-401/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("yanabi.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")
sys.path.insert(0, os.getcwd())

missing = []
for name in ["numpy", "pandas", "scipy", "matplotlib", "jax", "numpyro", "arviz"]:
    try:
        mod = __import__(name)
        print(f"✅ {name} {getattr(mod, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"❌ {name} is missing")

# %%
import time
import yanabi as U

p = U.panel()
print(f"✅ data: {p['district_id'].nunique()} districts, {len(p)} district-years")
D = U.design()
t0 = time.time()
_, idata = U.sample(U.burst_model, warmup=200, draws=200, pooling="complete", y=D["y"], **U.model_data(D))
d = U.diagnose(idata, ["alpha", "beta"])
print(f"✅ four chains sampled in {time.time() - t0:.0f} s (the first run includes compiling); R-hat {d['max_rhat']:.3f}, divergences {d['divergences']}")
print("Everything is ready." if not missing else f"Install the missing libraries: {', '.join(missing)}")

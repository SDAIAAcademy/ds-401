# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
"""Write the synthetic data of Yanabi Water Authority into labs/data/.

    uv run prepare_data.py

NumPy and the standard library only, fixed seed: the same files on every run. The generating
parameters are written to data/truth.json, so every lab can check its estimates against them.
Everything is invented: the districts, the bursts, the costs, the survey offer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 401
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CAP = HERE / "capstone" / "data"

YEARS = [2021, 2022, 2023, 2024, 2025]
MATERIALS = ["PVC", "ductile_iron", "cast_iron", "asbestos_cement"]

# ---- the generating parameters (the truth the labs recover)
TRUTH = {
    "alpha": float(np.log(0.10)),          # log bursts per km-year of a PVC network of mean age 30
    "beta_material": {"PVC": 0.0, "ductile_iron": 0.15, "cast_iron": 0.45, "asbestos_cement": 0.70},
    "beta_age_per_10y": 0.30,             # on the log scale, per 10 years above 30
    "sigma_district": 0.30,               # district effect after material and age
    "phi": 6.0,                           # negative binomial shape of the year-to-year variation
    "reporting_no_sensor": {"a": 14, "b": 6},  # Beta of the share of bursts reported where no sensors
    "reporting_sensor": 1.0,
    "trunk_main": {"district": "D017", "year": 2023, "extra_bursts": 38},
}


def _districts(rng):
    rows = []
    kinds = ["urban"] * 44 + ["suburban"] * 40 + ["rural"] * 56
    rng.shuffle(kinds)
    for j, kind in enumerate(kinds, start=1):
        did = f"D{j:03d}"
        if kind == "urban":
            km = rng.lognormal(np.log(110), 0.45)
            mat_p = [0.35, 0.30, 0.25, 0.10]
            age = rng.normal(24, 7)
            sensors = rng.random() < 0.85
        elif kind == "suburban":
            km = rng.lognormal(np.log(45), 0.45)
            mat_p = [0.45, 0.25, 0.10, 0.20]
            age = rng.normal(22, 8)
            sensors = rng.random() < 0.55
        else:
            km = rng.lognormal(np.log(2.4), 0.5)
            mat_p = [0.20, 0.10, 0.15, 0.55]
            age = rng.normal(38, 8)
            sensors = rng.random() < 0.12
        rows.append({
            "district_id": did,
            "kind": kind,
            "network_km": round(float(np.clip(km, 1.2, 400)), 1),
            "material": MATERIALS[rng.choice(4, p=mat_p)],
            "mean_age_years": int(np.clip(round(age), 6, 60)),
            "sensors": bool(sensors),
        })
    return rows


def _bursts(rng, districts):
    t = TRUTH
    out, true_rates, report = [], {}, {}
    for d in districts:
        eta = (t["alpha"] + t["beta_material"][d["material"]]
               + t["beta_age_per_10y"] * (d["mean_age_years"] - 30) / 10
               + t["sigma_district"] * rng.normal())
        rate = float(np.exp(eta))
        true_rates[d["district_id"]] = rate
        p = 1.0 if d["sensors"] else float(rng.beta(t["reporting_no_sensor"]["a"], t["reporting_no_sensor"]["b"]))
        report[d["district_id"]] = p
        for y in YEARS:
            mu = rate * d["network_km"]
            lam = rng.gamma(t["phi"], mu / t["phi"])  # negative binomial through a gamma mixture
            true_n = int(rng.poisson(lam))
            if d["district_id"] == t["trunk_main"]["district"] and y == t["trunk_main"]["year"]:
                true_n += t["trunk_main"]["extra_bursts"]
            reported = int(rng.binomial(true_n, p))
            out.append({"district_id": d["district_id"], "year": y, "bursts_reported": reported, "bursts_true": true_n})
    return out, true_rates, report


COSTS = {
    "currency": "SAR",
    "note": "Synthetic planning numbers for the course, not prices of any real supplier.",
    "cost_per_burst": 80000,
    "horizon_years": 20,
    "actions": {
        "wait": {"cost_per_km": 0, "reduction": 0.0, "years": 0},
        "repair": {"cost_per_km": 40000, "reduction": 0.30, "years": 10,
                   "what": "targeted repair of known weak joints and pressure management"},
        "replace": {"cost_per_km": 400000, "reduction": 0.90, "years": 20,
                    "what": "re-pipe the district's mains"},
    },
    "capital_budget": 90000000,
    "survey": {
        "offered_by": "an acoustic leak-detection contractor",
        "what": "listens to every main of a district without sensors for one season and returns its true burst-prone rate with about 15% error",
        "log_sd": 0.15,
        "price_per_district": 45000,
    },
}


def _capstone(rng):
    """Three small synthetic briefs for the capstone, each with known generating parameters."""
    CAP.mkdir(parents=True, exist_ok=True)
    briefs = {}
    # 1 water losses: night-flow loss per district metered area, litres per connection per day
    rows = []
    for k in range(1, 61):
        zone = f"Z{k:02d}"
        conns = int(rng.integers(80, 2500))
        pressure = round(float(rng.normal(4.2, 0.8)), 2)
        base = np.exp(np.log(110) + 0.25 * (pressure - 4.2) + 0.35 * rng.normal())
        n_nights = int(rng.integers(3, 40))
        for n in range(n_nights):
            rows.append({"zone": zone, "connections": conns, "pressure_bar": pressure, "night": n + 1,
                         "loss_l_per_conn_day": round(float(base * np.exp(0.25 * rng.normal())), 1)})
    _write_csv(CAP / "water_losses.csv", rows)
    briefs["water_losses"] = {
        "title": "Water losses: which metered zones get pressure management",
        "file": "data/water_losses.csv",
        "question": "Yanabi can fit pressure-reducing valves in 12 of 60 metered zones this year. Each valve costs SAR 150,000 and is expected to cut night-flow losses by 20%. Water lost costs SAR 4 per cubic metre. Which zones, and what is the expected saving and its uncertainty?",
        "decision": "12 zones out of 60, or fewer if the expected saving does not cover the cost",
    }
    # 2 school inspections: number of serious findings per inspection, schools in 8 regions
    rows = []
    for s in range(1, 201):
        region = f"R{rng.integers(1, 9)}"
        size = int(rng.integers(120, 1600))
        eff = 0.4 * rng.normal()
        for v in range(int(rng.integers(1, 5))):
            rows.append({"school": f"S{s:03d}", "region": region, "pupils": size, "visit": v + 1,
                         "serious_findings": int(rng.poisson(np.exp(np.log(1.1) + 0.2 * np.log(size / 500) + eff)))})
    _write_csv(CAP / "school_inspections.csv", rows)
    briefs["school_inspections"] = {
        "title": "School inspections: which schools get a follow-up visit",
        "file": "data/school_inspections.csv",
        "question": "An inspectorate can make 30 follow-up visits. A visit costs SAR 9,000; a serious finding left unaddressed for a year is valued at SAR 4,000. Which schools, and how sure are you about the small schools with one visit on record?",
        "decision": "30 follow-up visits among 200 schools",
    }
    # 3 clinic waiting times: minutes from arrival to consultation, clinics in 4 cities
    rows = []
    for c in range(1, 41):
        city = ["north", "south", "east", "west"][int(rng.integers(0, 4))]
        doctors = int(rng.integers(2, 12))
        mu = np.log(35) - 0.08 * (doctors - 6) + 0.3 * rng.normal()
        for p in range(int(rng.integers(15, 160))):
            rows.append({"clinic": f"C{c:02d}", "city": city, "doctors": doctors,
                         "wait_minutes": round(float(np.exp(mu + 0.55 * rng.normal())), 1)})
    _write_csv(CAP / "clinic_waits.csv", rows)
    briefs["clinic_waits"] = {
        "title": "Clinic waiting times: where an extra doctor goes",
        "file": "data/clinic_waits.csv",
        "question": "A health cluster can add one doctor to each of 5 clinics out of 40. The target is a median wait under 30 minutes. Which clinics, and what is the probability that each one meets the target after the change?",
        "decision": "5 clinics out of 40",
    }
    (HERE / "capstone" / "briefs.json").write_text(json.dumps(briefs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    districts = _districts(rng)
    bursts, true_rates, report = _bursts(rng, districts)
    _write_csv(DATA / "districts.csv", districts)
    # the participants' file has the reported counts only; the true counts stay in the truth file
    _write_csv(DATA / "bursts.csv", [{k: r[k] for k in ("district_id", "year", "bursts_reported")} for r in bursts])
    truth = dict(TRUTH)
    truth["true_rate_per_km_year"] = {k: round(v, 5) for k, v in true_rates.items()}
    truth["reporting_rate"] = {k: round(v, 4) for k, v in report.items()}
    truth["true_bursts"] = {f"{r['district_id']}-{r['year']}": r["bursts_true"] for r in bursts}
    (DATA / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    (DATA / "costs.json").write_text(json.dumps(COSTS, indent=2) + "\n", encoding="utf-8")
    _capstone(np.random.default_rng(SEED + 1))

    tot = {}
    for r in bursts:
        tot[r["district_id"]] = tot.get(r["district_id"], 0) + r["bursts_reported"]
    few = sum(1 for v in tot.values() if v < 5)
    kinds = {}
    for d in districts:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"districts {len(districts)} {kinds}; district-years {len(bursts)}; reported bursts {sum(tot.values())}")
    print(f"districts with fewer than 5 recorded bursts in five years: {few}")
    print(f"without sensors: {sum(1 for d in districts if not d['sensors'])}")
    zeros = sum(1 for r in bursts if r["bursts_reported"] == 0)
    print(f"district-years with no reported burst: {zeros}")


if __name__ == "__main__":
    main()

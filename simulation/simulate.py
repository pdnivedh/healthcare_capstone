"""
Digital twin of the hospital (Part B).

Usage (from the project root, next to hospital.duckdb):
    from simulation.simulate import simulate
    df = simulate({"beds": 80, "arrival_scale": 20, "top_pct": 10, "effectiveness": 0.2,
                   "n_days": 365, "n_reps": 30, "seed": 42})

Returns one row per replication with columns:
    rep, mean_occupancy, max_occupancy, bed_days, blocked_admissions, readmissions,
    readmissions_prevented, bed_days_avoided, flagged_discharges
(the last two are extras: bed-days of the readmissions that were prevented, and how many
discharges the policy flagged).

Design (put these in docs/ASSUMPTIONS.md):
  * Poisson arrivals of NEW patients; rate fitted on new admissions only (2017-2025).
  * Each stay draws a (risk_lr, los_days) row from syn_risk_scores (bootstrap).
  * Loss system: if all beds are busy, the arriving patient is blocked (no queue).
  * At discharge the patient returns with probability = risk; if the policy flags the
    patient (top `top_pct`% by score) the probability becomes risk * (1 - effectiveness).
    The same uniform draw decides both worlds, so "prevented" is a paired quantity.
  * A returning patient comes back after an empirical gap and gets a fresh (risk, LOS) draw.
  * Common random numbers: every patient "lineage" has its own random stream, so scenarios
    with the same seed see the same arrivals and the same random numbers. Differences between
    scenarios are therefore due to the policy / parameters, not to luck.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import simpy

DEFAULTS = dict(
    beds=80, arrival_scale=20.0, top_pct=0, effectiveness=0.2,
    n_days=365, n_reps=30, seed=42,
    warmup_days=60, los_scale=1.0, score_noise=0.0,
    keep_occ=False, db_path="hospital.duckdb",
)

_D = {}   # cached inputs


def load_config(path="config.yaml"):
    """Read settings from a YAML file (needs `pip install pyyaml`)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_inputs(db_path="hospital.duckdb", force=False):
    """Load RISK, LOS, GAPS and the arrival rate from the database (cached)."""
    if _D and not force and _D.get("db_path") == str(db_path):
        return _D
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} not found. Put hospital.duckdb in the project root.")
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET TimeZone='UTC'")
    stays = con.execute("SELECT PATIENT, START, STOP FROM syn_stays").df()
    scores = con.execute("SELECT los_days, risk_lr, readmit_30d FROM syn_risk_scores").df()
    con.close()

    stays["START"] = pd.to_datetime(stays["START"], utc=True)
    stays["STOP"] = pd.to_datetime(stays["STOP"], utc=True)
    s = stays.sort_values(["PATIENT", "START"])
    gap = (s["START"] - s.groupby("PATIENT")["STOP"].shift(1)).dt.total_seconds() / 86400
    is_re = gap.between(0, 30)

    n_hist_days = (pd.Timestamp("2025-12-31") - pd.Timestamp("2017-01-01")).days + 1
    scores = scores.dropna(subset=["los_days", "risk_lr"]).reset_index(drop=True)

    _D.clear()
    _D.update(
        db_path=str(db_path),
        RATE=float((~is_re).sum() / n_hist_days),                 # new admissions per day
        GAPS=gap[is_re].to_numpy(),
        RISK=scores["risk_lr"].to_numpy(),
        LOS=scores["los_days"].to_numpy(),
        READMIT=scores["readmit_30d"].astype(bool).to_numpy(),
        n_readm=int(is_re.sum()), n_new=int((~is_re).sum()), n_hist_days=n_hist_days,
    )
    _D["RISK_SD"] = float(_D["RISK"].std())
    return _D


def run_once(cfg, rep):
    D = load_inputs(cfg["db_path"])
    RISK, LOS, GAPS, RATE, RISK_SD = D["RISK"], D["LOS"], D["GAPS"], D["RATE"], D["RISK_SD"]
    n = len(RISK)
    seed = int(cfg["seed"])
    rng_arr = np.random.default_rng([seed, rep, 0])      # arrivals
    rng_thr = np.random.default_rng([seed, rep, 1])      # flagging threshold (noisy score)

    env = simpy.Environment()
    beds = simpy.Resource(env, capacity=int(cfg["beds"]))
    warm = cfg["warmup_days"]
    st = dict(blocked=0, readm=0, prevented=0, bed_days=0.0, avoided=0.0, flagged=0, occ=[])

    lam, eff, los_scale = RATE * cfg["arrival_scale"], cfg["effectiveness"], cfg["los_scale"]
    sigma = cfg["score_noise"] * RISK_SD

    if cfg["top_pct"] > 0:
        score_all = RISK + rng_thr.normal(0, sigma, n) if sigma > 0 else RISK
        thr = np.quantile(score_all, 1 - cfg["top_pct"] / 100)
    else:
        thr = np.inf

    def visit(lrng, i):
        # Always draw the same 4 numbers per visit so lineages stay aligned across scenarios.
        u = lrng.random(); gap = lrng.choice(GAPS); nxt = lrng.integers(n); z = lrng.normal()
        risk, los = RISK[i], LOS[i] * los_scale

        if beds.count >= beds.capacity:                   # loss system
            if env.now >= warm: st["blocked"] += 1
            return
        with beds.request() as req:
            yield req
            if env.now >= warm: st["bed_days"] += los
            yield env.timeout(los)

        measuring = env.now >= warm
        flagged = (risk + sigma * z) >= thr
        p = risk * (1 - eff) if flagged else risk
        if measuring and flagged: st["flagged"] += 1
        if u < p:
            if measuring: st["readm"] += 1
            env.process(returner(lrng, gap, nxt))
        elif u < risk:                                    # would have returned, did not
            if measuring:
                st["prevented"] += 1
                st["avoided"] += LOS[nxt] * los_scale

    def returner(lrng, gap, nxt):
        yield env.timeout(gap)
        env.process(visit(lrng, nxt))

    def arrivals():
        k = 0
        while True:
            yield env.timeout(rng_arr.exponential(1 / lam))
            i = rng_arr.integers(n)
            env.process(visit(np.random.default_rng([seed, rep, 2, k]), i))
            k += 1

    def monitor():
        while True:
            if env.now >= warm: st["occ"].append(beds.count)
            yield env.timeout(1)

    env.process(arrivals()); env.process(monitor())
    env.run(until=warm + cfg["n_days"])
    return st


def simulate(config=None, **overrides):
    """Run n_reps replications. Returns a DataFrame, one row per replication."""
    cfg = {**DEFAULTS, **(config or {}), **overrides}
    rows, occs = [], []
    for rep in range(int(cfg["n_reps"])):
        st = run_once(cfg, rep)
        occ = np.array(st["occ"]); occs.append(occ)
        rows.append(dict(
            rep=rep, mean_occupancy=occ.mean(), max_occupancy=int(occ.max()),
            bed_days=st["bed_days"], blocked_admissions=st["blocked"],
            readmissions=st["readm"], readmissions_prevented=st["prevented"],
            bed_days_avoided=st["avoided"], flagged_discharges=st["flagged"],
        ))
    df = pd.DataFrame(rows)
    if cfg["keep_occ"]:
        df.attrs["occ"] = occs
    return df


if __name__ == "__main__":
    print(simulate(n_reps=3, n_days=200, top_pct=10).round(2))

"""one replicate m:
     derive two streams from (seed, m): reaction times and service draws (CRN, doc 5.7)
Repeat K times; report T̄, T₉₅, P(T ≤ 180); paired differences ΔT^(m) use the same m.

Parallel: graph + population (incl. LED plan) are prepared once; workers only simulate."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import PREP_LOGNORMAL, PREP_WINDOWS, REACTION_SPECS, Config
from .flow import simulate
from .plugins.randomness import PoissonRandomness
from .scenarios import SCENARIOS, Scenario, prepare

GRID_DT = 1.0          # time grid for stored curves
GRID_T = 900.0
TRACK_EDGES = (("FOY", "EXIT-Main"), ("ST1", "EXIT-Stage-W"), ("POR1", "EXIT-Side-W"))
N_TRACK = 40           # replicates whose queue paths are stored

_JOB: Dict = {}


def _streams(seed: int, m: int):
    ss = np.random.SeedSequence([seed, m])
    r1, r2 = ss.spawn(2)
    return np.random.default_rng(r1), np.random.default_rng(r2)


def _init(jobs):
    _JOB.clear()
    _JOB.update(jobs)


def _one(task) -> Dict:
    'one replicate. task = (job key, m).'
    key, m = task
    j = _JOB[key]
    r_react, r_serv = _streams(j["seed"], m)
    rnd = j["proto"].sample_streams(r_react, r_serv)
    if not j.get("motion", True):
        rnd.rng_motion = None
    res = simulate(j["b"], j["cfg"], j["classes"], j["v"], randomness=rnd,
                   discovery=j["discovery"], broadcast_delay=j["delay"])
    grid = np.arange(0, GRID_T + 1e-9, GRID_DT)
    t = np.concatenate([[0.0], res.t])
    rem = np.concatenate([[res.meta["n"]], res.remaining_inside])
    curve = rem[np.searchsorted(t, grid, side="right") - 1]
    out = dict(key=key, m=m, T=res.T, T_ground=res.T_ground, T90=res.time_to_fraction(0.9),
               T50=res.time_to_fraction(0.5), mean_time=res.mean_time,
               curve=curve.astype(np.int32) if j["keep_curves"] else None, exits=res.exit_groups())
    if j.get("wheel"):

        from .validation import wheelchair_report
        rep = wheelchair_report(res, j["cfg"])
        out.update(T_refuge=rep.get("T_refuge", np.nan), T_street=rep.get("T_street", np.nan))
    if m < N_TRACK and j.get("keep_diagnostics", True):
        qs = {}
        for e in TRACK_EDGES:
            if e in res.edges:
                q = res.edge_queue[:, res.edges.index(e)]
                qs[e] = np.concatenate([[0], q])[np.searchsorted(t, grid, side="right") - 1].astype(np.int32)
        out["queues"] = qs

        if rnd.draws is not None:
            allr = np.concatenate(list(rnd.draws.values()))
            out["starts_per_s"] = np.histogram(allr, bins=np.arange(0, 121, 1.0))[0]
    return out


def reaction_sampler(sc: Scenario, cfg: Config, seats: Dict[str, float],
                     dist: Optional[str] = None, service: str = "poisson",
                     dispersion: Optional[float] = None) -> PoissonRandomness:
    spec = dict(REACTION_SPECS[sc.prep_key])
    dist = dist or spec["family"]
    if dist not in (spec["family"], "mean-field"):
        raise ValueError("Scenario reaction family is configured centrally; use mean-field for a deterministic control")
    return PoissonRandomness(PREP_WINDOWS[sc.prep_key], seats, dist=dist, service=service, dt=cfg.phys.dt,
                             dispersion=dispersion, reaction_spec=spec)


def make_job(sc: Scenario, cfg: Config, seed: int = 2026, dist: Optional[str] = None, service: str = "poisson",
             keep_curves: bool = True, dispersion: Optional[float] = None, wheel: bool = False, motion: bool = True, keep_diagnostics: bool = True) -> Dict:
    b, classes, v, info = prepare(sc, cfg)
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    proto = reaction_sampler(sc, cfg, seats, dist, service, dispersion)
    dist = proto.dist
    return dict(b=b, cfg=cfg, classes=classes, v=v, discovery=sc.discovery, delay=sc.broadcast_delay,
                proto=proto, seed=seed, name=sc.name, dist=dist, service=service, keep_curves=keep_curves,
                n=int(sum(seats.values())), wheel=wheel, motion=motion, keep_diagnostics=keep_diagnostics,
                setup=dict(reaction_spec=dict(proto.reaction_spec),
                           **{k: value for k,value in info.items() if k.startswith("manager") or k.startswith("wheelchair")}))


def run_batch(jobs: Dict[str, Dict], K: int, workers: Optional[int] = None) -> Dict[str, Dict]:
    'Many scenarios / variants share one process pool (process start-up is expensive on Windows).'
    workers = workers or min(6, max(1, (os.cpu_count() or 2) - 2))
    tasks = [(k, m) for k in jobs for m in range(K)]
    if workers == 1:
        _init(jobs)
        outs = [_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(jobs,)) as ex:
            outs = list(ex.map(_one, tasks, chunksize=max(1, len(tasks) // (8 * workers))))
    res = {}
    for key, j in jobs.items():
        o = sorted([x for x in outs if x["key"] == key], key=lambda x: x["m"])
        T = np.array([x["T"] for x in o])
        r = dict(scenario=j["name"], K=K, seed=j["seed"], dist=j["dist"], service=j["service"], motion=j["motion"],
                 T=T, T90=np.array([x["T90"] for x in o]), T50=np.array([x["T50"] for x in o]),
                 T_ground=np.array([x["T_ground"] for x in o]),
                 mean_time=np.array([x["mean_time"] for x in o]),
                 grid=np.arange(0, GRID_T + 1e-9, GRID_DT),
                 exits={k: np.array([x["exits"].get(k, 0.0) for x in o]) for k in o[0]["exits"]},
                 queues=[x.get("queues", {}) for x in o if "queues" in x],
                 starts=[x["starts_per_s"] for x in o if "starts_per_s" in x], n=j["n"])
        if j["keep_curves"]:
            r["curves"] = np.stack([x["curve"] for x in o])
        if j.get("wheel"):
            r["T_refuge"] = np.array([x["T_refuge"] for x in o])
            r["T_street"] = np.array([x["T_street"] for x in o])
        r.update(summary(T))
        r["setup"] = j.get("setup", {})
        res[key] = r
    return res


def run_mc(sc: Scenario, cfg: Config, K: int = 200, seed: int = 2026, dist: Optional[str] = None,
           service: str = "poisson", workers: Optional[int] = None) -> Dict:
    'K replicates of one scenario.'
    return run_batch({sc.name: make_job(sc, cfg, seed, dist, service)}, K, workers)[sc.name]


def summary(T: np.ndarray) -> Dict:
    'estimates with 95% CIs.'
    K = len(T)
    p = float(np.mean(T <= 180))
    se_T = float(T.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0
    se_p = float(np.sqrt(max(p * (1 - p), 1e-12) / K))
    return dict(T_mean=float(T.mean()), T_sd=float(T.std(ddof=1)) if K > 1 else 0.0,
                T_mean_ci=(float(T.mean() - 1.96 * se_T), float(T.mean() + 1.96 * se_T)),
                T_95=float(np.percentile(T, 95)), T_05=float(np.percentile(T, 5)),
                T_median=float(np.median(T)),
                P_le_180=p, P_ci=(max(0.0, p - 1.96 * se_p), min(1.0, p + 1.96 * se_p)))


def run_mc_many(names: Sequence[str], cfg: Config, K: int = 200, seed: int = 2026,
                overrides: Optional[Dict[str, Dict]] = None, **kw) -> Dict[str, Dict]:
    'several scenarios with CRN.'
    out = {}
    for n in names:
        sc = SCENARIOS[n] if n in SCENARIOS else None
        if overrides and n in overrides:
            base = SCENARIOS[overrides[n].get("base", n)]
            sc = base.with_(name=n, **{k: v for k, v in overrides[n].items() if k != "base"})
        out[n] = run_mc(sc, cfg, K=K, seed=seed, **kw)
    return out


def paired_diff(a: Dict, b: Dict) -> np.ndarray:
    assert a["seed"] == b["seed"] and a["K"] == b["K"], "need the same CRN"
    return a["T"] - b["T"]


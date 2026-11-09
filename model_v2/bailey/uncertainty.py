"""/ Parameter uncertainty: tornado (one at a time) + parameter Monte Carlo.

Why this layer exists
----------------------------------------
Section 5 noise moves T by a few seconds; whether 180 s is met depends mostly on the
unconfirmed parameters. Here they are sampled within ASSUMED RANGES together with the
Poisson noise, giving P(T ≤ 180) under parameter uncertainty.

All ranges are [ASSUMED RANGE] — not data; state them as assumptions in the paper."""
from __future__ import annotations

import os
from math import log
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .config import BEHAVIOURS, MANAGERS, PREP_LOGNORMAL, PREP_WINDOWS, REACTION_SPECS, Config
from .montecarlo import _streams, reaction_sampler
from .scenarios import SCENARIOS, run


@dataclass(frozen=True)
class Range:
    key: str
    label: str            # label for the paper
    low: float
    high: float
    nominal: float
    log: bool = False     # multiplicative factors: log-uniform
    source: str = "ASSUMED RANGE"


RANGES: List[Range] = [
    Range("q", "specific flow q (p/m/s)", 0.8, 1.3, C.DEFAULT.phys.q, source="ASSUMED RANGE including prior 1.3 baseline"),
    Range("v", "walking speed (×)", 0.885, 1.115, 1.0, source="ASSUMED RANGE around baseline speeds"),
    Range("stair_factor", "stair speed factor", C.DEFAULT.phys.stair_factor - 0.10,
          C.DEFAULT.phys.stair_factor + 0.10, C.DEFAULT.phys.stair_factor),
    Range("len", "all cell lengths (×)", 0.8, 1.2, 1.0),
    Range("w_main_exterior", "main doors total width (m)", 5.6, 8.4, 7.0),
    Range("w_side_door", "side doors S width (m)", 1.0, 1.5, 1.2),
    Range("w_side_stair", "side stairs width (m)", 1.0, 1.5, 1.2),
    Range("w_stage_stair", "stage stairs ST width (m)", 1.0, 1.5, 1.2),
    Range("w_stage_door_mezz", "mezz. stage doors width (m)", 0.8, 1.2, 1.0),
    Range("w_foyer_door", "foyer doors F width (m)", 1.1, 1.5, 1.3),
    Range("beta_dist", "β1 distance aversion (×)", 0.5, 2.0, 1.0, log=True),
    Range("beta_habit", "β2 main-entrance habit (×)", 0.5, 1.5, 1.0, log=True),
    Range("beta_queue", "β3 queue avoidance (×)", 0.5, 2.0, 1.0, log=True),
    Range("beta_guide", "β4 instruction weight (×)", 0.5, 1.5, 1.0, log=True),
    Range("radius_m", "manager guidance radius (m)", *C.MANAGER_RADIUS_RANGE, C.MANAGERS["radius_m"], source="ASSUMED sensitivity around literature-adapted 10 m"),
    Range("p_follow", "share of instructed people who comply", 0.75, 0.95, 0.9),
    Range("prep", "pre-movement time scale (×)", 0.5, 2.0, 1.0, log=True),
]
RANGE = {r.key: r for r in RANGES}


def apply_theta(cfg: Config, theta: Dict[str, float]) -> Config:
    'See docs/IMPLEMENTATION.md for implementation details.'
    ph, geo = cfg.phys, cfg.geo
    ph_kw, geo_kw = {}, {}
    if "q" in theta:
        ph_kw["q"] = theta["q"]
    if "v" in theta:
        ph_kw["v_emergency"] = C.Physical.v_emergency * theta["v"]
        ph_kw["v_normal"] = C.Physical.v_normal * theta["v"]
    if "stair_factor" in theta:
        ph_kw["stair_factor"] = theta["stair_factor"]
    if "len" in theta:
        for f in fields(C.Geometry):
            if f.name.startswith("l_"):
                geo_kw[f.name] = getattr(C.Geometry, f.name) * theta["len"]
    for k in ("w_main_exterior", "w_side_door", "w_side_stair", "w_stage_stair",
              "w_stage_door_mezz", "w_foyer_door"):
        if k in theta:
            geo_kw[k] = theta[k]
    return cfg.with_(phys=replace(ph, **ph_kw), geo=replace(geo, **geo_kw))


@contextmanager
def patched(theta: Dict[str, float]):
    'See docs/IMPLEMENTATION.md for implementation details.'
    saved_b, saved_p, saved_m = dict(BEHAVIOURS), dict(PREP_WINDOWS), dict(MANAGERS)
    saved_log = dict(PREP_LOGNORMAL)
    saved_specs = {k:dict(v) for k,v in REACTION_SPECS.items()}
    try:
        for k, b in saved_b.items():
            kw = {}
            for par in ("beta_dist", "beta_habit", "beta_queue", "beta_guide"):
                if par in theta:
                    kw[par] = getattr(b, par) * theta[par]
            BEHAVIOURS[k] = replace(b, **kw)
        for key in ("radius_m", "p_follow"):
            if key in theta:
                MANAGERS[key] = theta[key]
        if "prep" in theta:
            for k, (a, bb) in saved_p.items():
                PREP_WINDOWS[k] = (a * theta["prep"], bb * theta["prep"])
            for k, (mu, sigma) in saved_log.items():
                PREP_LOGNORMAL[k] = (mu + log(theta["prep"]), sigma)
            for k, spec in saved_specs.items():
                updated = dict(spec); factor = theta["prep"]
                if spec["family"] == "lognormal": updated["mu"] += log(factor)
                elif spec["family"] == "gamma": updated["scale"] *= factor
                elif spec["family"] == "truncnorm":
                    for field in ("mu", "sigma", "low", "high"): updated[field] *= factor
                REACTION_SPECS[k] = updated
        yield
    finally:
        BEHAVIOURS.clear(); BEHAVIOURS.update(saved_b)
        PREP_WINDOWS.clear(); PREP_WINDOWS.update(saved_p)
        PREP_LOGNORMAL.clear(); PREP_LOGNORMAL.update(saved_log)
        REACTION_SPECS.clear(); REACTION_SPECS.update(saved_specs)
        MANAGERS.clear(); MANAGERS.update(saved_m)


def run_theta(name: str, cfg: Config, theta: Dict[str, float], randomness=None):
    'run a scenario under parameter set theta.'
    sc = SCENARIOS[name]
    with patched(theta):
        return run(sc, apply_theta(cfg, theta), randomness=randomness)[0]


# ---------------------------------------------------------------------------
# tornado: one parameter at a time at its bounds
# ---------------------------------------------------------------------------
def _tornado_job(args):
    name, key, val, cfg = args
    r = run_theta(name, cfg, {key: val})
    return dict(scenario=name, key=key, value=val, T=r.T, T90=r.time_to_fraction(0.9))


def tornado(cfg: Config, scenarios: Sequence[str] = ("concert", "lecture"),
            workers: Optional[int] = None) -> pd.DataFrame:
    jobs = []
    for name in scenarios:
        for r in RANGES:
            if r.key in ("radius_m", "p_follow") and SCENARIOS[name].population != "managers":
                continue
            for val in (r.low, r.high):
                jobs.append((name, r.key, val, cfg))
    workers = workers or min(6, max(1, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_tornado_job, jobs))
    df = pd.DataFrame(rows)
    base = {n: run(SCENARIOS[n], cfg)[0].T for n in scenarios}
    df["T_base"] = df["scenario"].map(base)
    df["label"] = df["key"].map(lambda k: RANGE[k].label)
    return df


# ---------------------------------------------------------------------------
# parameter MC: parameters + Poisson noise
# ---------------------------------------------------------------------------
def sample_theta(rng: np.random.Generator) -> Dict[str, float]:
    th = {}
    for r in RANGES:
        u = rng.uniform()
        th[r.key] = float(np.exp(np.log(r.low) + u * (np.log(r.high) - np.log(r.low)))) if r.log\
            else float(r.low + u * (r.high - r.low))
    return th


def _pmc_job(args):
    name, m, seed, cfg = args
    r_par = np.random.default_rng(np.random.SeedSequence([seed, m, 7]))
    theta = sample_theta(r_par)                      # same θ for every scenario
    sc = SCENARIOS[name]
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    with patched(theta):
        proto = reaction_sampler(sc, cfg, seats)
    r_react, r_serv = _streams(seed, m)
    r = run_theta(name, cfg, theta, randomness=proto.sample_streams(r_react, r_serv))
    return dict(scenario=name, m=m, T=r.T, T_ground=r.T_ground, T90=r.time_to_fraction(0.9),
                mean_time=r.mean_time, **theta)


def param_mc(cfg: Config, scenarios: Sequence[str] = ("normal", "concert", "blocked", "lecture"),
             K: int = 400, seed: int = 2026, workers: Optional[int] = None) -> pd.DataFrame:
    jobs = [(n, m, seed, cfg) for n in scenarios for m in range(K)]
    workers = workers or min(6, max(1, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_pmc_job, jobs, chunksize=4))
    return pd.DataFrame(rows)


def ranges_table() -> pd.DataFrame:
    return pd.DataFrame([dict(parameter=r.label, key=r.key, low=r.low, nominal=r.nominal, high=r.high,
                              sampling="log-uniform" if r.log else "uniform", source=r.source)
                         for r in RANGES])


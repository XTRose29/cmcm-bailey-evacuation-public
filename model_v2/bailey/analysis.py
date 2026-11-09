"""Sweeps and checks.

closed form + lower bounds
numerical checks: Δt, walking model"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .building import build_building, egress_view
from .config import BEHAVIOURS, Config, prep_min
from .markov import MarkovClass, bottleneck_lower_bound, lp_assignment, mixture_analysis
from .routing import Router, cell_times, exit_distances, leave_prob
from .scenarios import BLOCK_TARGETS, SCENARIOS, build_population, run


# ---------------------------------------------------------------------------
def markov_report(cfg: Config, scenario: str = "concert") -> Dict:
    """closed-form chain (3.5) + lower bounds.
uses the same population mix as the scenario."""
    sc = SCENARIOS[scenario]
    b = build_building(cfg, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
    classes, _ = build_population(sc, cfg, b)
    v = getattr(cfg.phys, sc.v_key)
    removed_all = set(sc.blocked)
    mcls = []
    for c in classes:
        stairs = {x.id for x in b.cells if x.stair} if not c.beh.uses_stairs else set()
        tau = cell_times(b, cfg, v, c.beh)
        x0 = np.zeros(b.nT)
        for s, n in c.seat_counts.items():
            x0[b.transient.index(s)] += n
        mcls.append(MarkovClass(c.name, Router(b, c.beh, tau, stairs | removed_all, c.plan, c.node_behaviours),
                                leave_prob(tau, cfg.phys.dt), x0))
    mix = mixture_analysis(b, cfg, mcls)
    #
    bv = egress_view(b)
    tau = cell_times(bv, cfg, v, classes[-1].beh)
    d = exit_distances(bv, tau, removed_all)
    start = prep_min(sc.prep_key)
    lb = bottleneck_lower_bound(bv, d, cfg.pop.n_total, start, removed_all)
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    lp = lp_assignment(bv, d, seats, start, removed_all)
    junc = b.junctions()
    passes = {j: float(mix["passes"][b.transient.index(j)]) for j in junc}
    mu = {j: b.cell[j].inflow_cap for j in junc}
    return dict(b=b, mix=mix, lb=lb, lp=lp, passes=passes, mu=mu, scenario=scenario)


# ---------------------------------------------------------------------------
def blocked_sweep(cfg: Config, targets: Sequence[str] = BLOCK_TARGETS,
                  modes: Sequence[str] = ("known", "broadcast", "arrival")) -> pd.DataFrame:
    rows = []
    base, _ = run(SCENARIOS["concert"], cfg)
    rows.append(dict(target="none", discovery="—", T=base.T, T90=base.time_to_fraction(.9),
                     mean_time=base.mean_time))
    for tg in targets:
        for m in modes:
            r, _ = run(SCENARIOS["concert"].with_(name=f"blocked_{tg}_{m}", blocked=(tg,), discovery=m), cfg)
            rows.append(dict(target=tg, discovery=m, T=r.T, T90=r.time_to_fraction(.9),
                             mean_time=r.mean_time, T_ground=r.T_ground))
    return pd.DataFrame(rows)


def barrier_layouts():
    """symmetric barrier pairs.
    k starts at 2: the fenced-off part must contain a side door (C1 has none)."""
    out = {"none": ()}
    for k in range(2, 5):
        out[f"C{k}|C{k+1}"] = ((f"C{k}", f"C{k+1}"), (f"C{10-k}", f"C{11-k}"))
    return out


def barrier_sweep(cfg: Config, with_led: bool = True) -> pd.DataFrame:
    rows = []
    for name, cut in barrier_layouts().items():
        for pop in (("managers", "led") if with_led else ("managers",)):
            for base_name in ("concert", "lecture"):
                base = SCENARIOS[base_name]
                population = pop if base_name == "concert" else ("lecture_led" if pop == "led" else "lecture")
                sc = base.with_(name=f"{base_name}_{pop}_{name}", barriers=cut, population=population)
                r, _ = run(sc, cfg)
                rows.append(dict(layout=name, guidance=pop, event=base_name, T=r.T,
                                 T90=r.time_to_fraction(.9), mean_time=r.mean_time))
    return pd.DataFrame(rows)


def compliance_sweep(cfg: Config, grid=np.linspace(0, 1, 11)) -> pd.DataFrame:
    rows = []
    for c in grid:
        for name in ("concert", "barriers", "led", "combo"):
            r, _ = run(SCENARIOS[name].with_(compliance=float(c)), cfg)
            rows.append(dict(scenario=name, compliance=float(c), T=r.T,
                             T90=r.time_to_fraction(.9), mean_time=r.mean_time))
    return pd.DataFrame(rows)


@contextmanager
def _patched_behaviours(**overrides):
    'temporarily override a β in every behaviour.'
    saved = dict(BEHAVIOURS)
    try:
        for k, b in saved.items():
            BEHAVIOURS[k] = replace(b, **overrides)
        yield
    finally:
        BEHAVIOURS.clear()
        BEHAVIOURS.update(saved)


def beta_sweep(cfg: Config) -> pd.DataFrame:
    rows = []
    grids = {"beta_habit": [0.0, 1.0, 2.0, 3.0, 4.0],
             "beta_dist": [0.10, 0.25, 0.50, 1.00],
             "beta_queue": [0.0, 0.05, 0.10, 0.30, 0.60]}
    for par, vals in grids.items():
        for val in vals:
            with _patched_behaviours(**{par: val}):
                for name in ("concert", "lecture"):
                    r, _ = run(SCENARIOS[name], cfg)
                    rows.append(dict(parameter=par, value=val, scenario=name, T=r.T,
                                     T90=r.time_to_fraction(.9)))
    return pd.DataFrame(rows)


def method_checks(cfg: Config) -> pd.DataFrame:
    'Numerical checks: how much results move with smaller Δt or the other walking model.'
    rows = []
    for dt in (0.5, 0.25, 0.125, 0.0625):
        for wm in ("erlang", "geometric"):
            c2 = cfg.with_(phys=replace(cfg.phys, dt=dt, walk_model=wm))
            for name in ("concert", "lecture"):
                r, _ = run(SCENARIOS[name], c2)
                rows.append(dict(dt=dt, walk_model=wm, scenario=name, T=r.T,
                                 T90=r.time_to_fraction(.9), mean_time=r.mean_time))
    return pd.DataFrame(rows)


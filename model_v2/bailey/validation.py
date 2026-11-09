"""validation.

                         engine vs closed form for one room + one door
                         achieved flow per metre of clear width vs theatre-drill and lab data
                         rerun with drill-based and lab-based capacities → bracket on T
                         wheelchair users: time to outdoor refuge + assisted descent to street

Data sources: config.VALIDATION_FLOWS and config.WHEELCHAIR_ASSIST."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .building import Building, Cell
from .config import BEHAVIOURS, VALIDATION_FLOWS, WHEELCHAIR_ASSIST, Config
from .flow import PopClass, SimResult, simulate
from .plugins.randomness import MeanFieldPlaceholder


# ---------------------------------------------------------------------------
# one-door analytic check
# ---------------------------------------------------------------------------
def single_door_check(cfg: Config, Ns: Sequence[int] = (50, 100, 200, 400),
                      widths: Sequence[float] = (0.9, 1.2, 1.8), room_len: float = 6.0) -> pd.DataFrame:
    """N people walk through a room of length room_len to one door of width w;
    the engine's T should be close to τ + N/μ."""
    rows = []
    v = cfg.phys.v_emergency
    beh = replace(BEHAVIOURS["concert_self"], beta_queue=0.0)
    for w in widths:
        mu = cfg.phys.q * w
        for N in Ns:
            cells = [Cell("R", "room", "seat", room_len, row_part=0.0),
                     Cell("DOOR", "door", "door", cfg.geo.l_door, w, inflow_cap=mu,
                          n_max=cfg.geo.l_door * w * cfg.phys.k_max + 1),
                     Cell("EXIT-Main", "exit", "exit", exit_group="EXIT-Main")]
            b = Building(cells, [("R", "DOOR"), ("DOOR", "EXIT-Main")])
            r = simulate(b, cfg, [PopClass("s", beh, {"R": float(N)})], v,
                         randomness=MeanFieldPlaceholder((0.0, 0.0)), threshold=0.5)
            tau = room_len / v
            T_an = tau + N / mu
            rows.append(dict(width=w, N=N, mu=mu, T_sim=r.T, T_analytic=T_an,
                             rel_err=(r.T - T_an) / T_an))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# achieved specific flow per metre of clear width
# ---------------------------------------------------------------------------
def specific_flows(res: SimResult, qa: Dict, window_s: float = 10.0) -> pd.DataFrame:
    'Peak window-averaged departures per second divided by the clear width.'
    b = res.b
    dt = res.meta["dt"]
    n = max(1, int(round(window_s / dt)))
    rows = []
    for j, s in qa["series"].items():
        if "→" in j:
            continue                         # links skipped
        cell = b.cell[j]
        if cell.width <= 0 or s["D"][-1] < 20:
            continue
        d = np.diff(np.concatenate([[0.0], s["D"]])) / dt
        peak = float(np.max(np.convolve(d, np.ones(n) / n, mode="valid"))) if len(d) >= n else float(d.max())
        rows.append(dict(junction=j, kind=cell.kind, clear_width=cell.width, peak_flow=peak,
                         specific_flow_clear=peak / cell.width, passed=float(s["D"][-1])))
    return pd.DataFrame(rows).sort_values("specific_flow_clear", ascending=False)


# ---------------------------------------------------------------------------
# capacity bracket
# ---------------------------------------------------------------------------
def capacity_variants(cfg: Config) -> Dict[str, Config]:
    """three capacity assumptions:
default
      prior baseline 1.30 per clear width                                  — comparison
upper end"""
    ph = cfg.phys
    return {
        "Model default (clear width)": cfg,
        "prior baseline 1.30 p/(m·s)": cfg.with_(phys=replace(ph, q=1.30)),
        "lab bottleneck 1.9 p/(m·s)": cfg.with_(phys=replace(ph, q=VALIDATION_FLOWS["lab_bottleneck"])),
    }


# ---------------------------------------------------------------------------
# wheelchair users to street level
# ---------------------------------------------------------------------------
def _last_pass(res: SimResult, edge, tol: float = 0.0) -> float:
    if edge not in res.edges:
        return float("nan")
    cum = np.cumsum(res.flows[:, res.edges.index(edge)])
    if cum[-1] == 0:
        return float("nan")
    return float(res.t[np.flatnonzero(cum >= cum[-1])[0]])


def wheelchair_report(res: SimResult, cfg: Config) -> Dict:
    """wheelchair users:
                arrival at an outdoor refuge (landing or portico) = class building-egress time
                per route: max(T_refuge, time the exterior stair is clear) + length / assisted speed
    Simplification: exterior stairs are not cells; "stair clear" = last person enters + normal stair walk."""
    if "wheelchair" not in res.class_exit:
        return {}
    A = WHEELCHAIR_ASSIST
    b = res.b
    used = dict(zip(b.exits, res.class_exit["wheelchair"]))
    n_w = sum(used.values())
    T_ref = res.class_T.get("wheelchair", float("nan"))
    v_stair_crowd = cfg.phys.v_emergency * cfg.phys.stair_factor
    routes = {}
    if used.get("EXIT-Main", 0) > 0.01 * n_w:
        clear = _last_pass(res, ("FOY", "EXIT-Main")) + A["L_main_stair"] / v_stair_crowd
        routes["main stairs"] = dict(share=used["EXIT-Main"] / n_w, stair_clear=clear,
                                     T_street=max(T_ref, clear) + A["L_main_stair"] / A["v_stair"])
    for side, por in (("W", "POR1"), ("E", "POR2")):
        e = f"EXIT-Side-{side}"
        if used.get(e, 0) > 0.01 * n_w:
            clear = _last_pass(res, (por, e)) + A["L_side_stair"] / v_stair_crowd
            routes[f"side stairs {side}"] = dict(share=used[e] / n_w, stair_clear=clear,
                                                 T_street=max(T_ref, clear) + A["L_side_stair"] / A["v_stair"])
    T_street = max((r["T_street"] for r in routes.values()), default=float("nan"))
    return dict(T_refuge=T_ref, T_street=T_street, routes=routes, T_all=res.T, T_all_ground=res.T_ground)


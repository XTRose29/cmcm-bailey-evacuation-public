"""How each question's situation is set up.

|----------------|------|------|
ordinary concert dismissal, main exit only |
lecture, no managers, familiar students |
baseline + corridor barriers |
baseline + LED guidance |
barriers + LED |
lecture + LED |
lecture + barriers |

All behaviour parameters are placeholders; read results together with the robustness sweep."""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .building import NORMAL_CLOSED, Building, build_building, egress_view
from .config import (BEHAVIOURS, BROADCAST_DELAY, COMPLIANCE, MANAGERS, PREP_LOGNORMAL, PREP_WINDOWS, REACTION_SPECS, WHEELCHAIR_SEATS,
                     Config, DEFAULT, MANAGER_DEFAULT_NODES, WHEELCHAIR_LOCATION_SOURCE, prep_min)
from .flow import PopClass, SimResult, simulate
from .guidance import guidance_info, resolve_stations, default_stations
from .integer import ENGINE_VERSION, integer_classes
from .markov import lp_assignment
from .plugins.queueing import FlowQueueModel
from .plugins.randomness import mean_field_release
from .routing import cell_times, exit_distances, plan_nearest

DEFAULT_BARRIERS: Tuple[Tuple[str, str], ...] = (("C2", "C3"), ("C8", "C9"))
"""Default barrier positions: the two far ends of the corridor are fenced off toward
compares all positions."""

BLOCK_TARGETS = ("EXIT-Main", "F2", "D5", "EXIT-Side-W", "EXIT-Stage-W", "BS1")
'See docs/IMPLEMENTATION.md for implementation details.'


@dataclass
class Scenario:
    name: str
    question: str
    zh: str
    v_key: str = "v_emergency"
    prep_key: str = "concert_emergency"
    closed: Tuple[str, ...] = ()
    barriers: Tuple[Tuple[str, str], ...] = ()
    blocked: Tuple[str, ...] = ()
    discovery: str = "known"
    population: str = "managers"   # managers | led | normal | lecture | lecture_led
    compliance: Optional[float] = None   #
    managers: Optional[int] = None       # crowd managers; None = law minimum
    manager_nodes: Optional[Tuple[str, ...]] = None
    manager_radius_m: Optional[float] = None
    manager_p_follow: Optional[float] = None
    wheelchairs: int = 0
    wheelchair_guide: str = "nearest"    #
                                         # where wheelchair users are guided: nearest step-free | main landing
    broadcast_delay: float = BROADCAST_DELAY   # broadcast delay (blocked scenarios)

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)


SCENARIOS: Dict[str, Scenario] = {
    "normal": Scenario("normal", "Q1", "Normal concert dismissal", v_key="v_normal",
                       prep_key="normal_dismissal", closed=NORMAL_CLOSED, population="normal"),
    "concert": Scenario("concert", "Q2", "Concert emergency with Crowd Managers (baseline)"),
    "blocked": Scenario("blocked", "Q2", "Concert emergency with an initially unknown main-exit blockage and a broadcast after 20 s",
                        blocked=("EXIT-Main",), discovery="broadcast"),
    "lecture": Scenario("lecture", "Q3", "Lecture emergency without Crowd Managers",
                        prep_key="lecture_emergency", population="lecture"),
    "barriers": Scenario("barriers", "Q4a", "Concert emergency with corridor barriers", barriers=DEFAULT_BARRIERS),
    "led": Scenario("led", "Q4b", "Concert emergency with LED guidance", population="led"),
    "combo": Scenario("combo", "Q4", "Concert emergency with barriers and LED guidance", barriers=DEFAULT_BARRIERS,
                      population="led"),
    "lecture_led": Scenario("lecture_led", "Q4", "Lecture emergency with LED guidance",
                            prep_key="lecture_emergency", population="lecture_led"),
    "lecture_barriers": Scenario("lecture_barriers", "Q4", "Lecture emergency with corridor barriers",
                                 prep_key="lecture_emergency", population="lecture",
                                 barriers=DEFAULT_BARRIERS),
    "wheelchair": Scenario("wheelchair", "—", "Concert emergency with three wheelchair users at the rear M2 platform (third position assumed)", wheelchairs=3),
}


def manager_compliance(seats: Dict[str, float], n: int, radius_m: Optional[float] = None,
                       p_follow: Optional[float] = None, cfg: Config = DEFAULT):
    """Initial seat-node coverage only; local guidance continues at later decision nodes.

    Kept for the count-sweep reporting API. No people-per-manager reach heuristic.
    """
    b = build_building(cfg)
    info = guidance_info(b, cfg, default_stations(b, n), radius_m, p_follow)
    return {z: info["compliance_by_zone"].get(z, 0.0) for z in seats}, info["manager_alloc"]


def _manager_setup(sc, cfg, b):
    if sc.population not in ("managers", "led"):
        if sc.manager_nodes or (sc.managers is not None and sc.managers != 0):
            raise ValueError("This scenario has no crowd managers")
        return {}
    stations = resolve_stations(b, sc.managers, sc.manager_nodes)
    p = sc.manager_p_follow
    if p is None and sc.population == "managers" and sc.compliance is not None:
        p = sc.compliance  # compliance sweep changes response to a manager, not coverage
    return guidance_info(b, cfg, stations, sc.manager_radius_m, p)


def _local_manager_classes(seats, b, cfg, v, removed, info):
    """Susceptible people change beta only within the union of station ranges."""
    base, guided = BEHAVIOURS["concert_self"], BEHAVIOURS["concert_managed"]
    p = info["manager_p_follow"] if info.get("manager_nodes") else 0.0
    plan = plan_nearest(b, exit_distances(b, cell_times(b, cfg, v, base), removed))
    local = {i: guided for i in info.get("manager_covered_nodes", [])}
    return [PopClass("manager_receptive", base, {z: n*p for z,n in seats.items()}, plan, local),
            PopClass("self", base, {z: n*(1-p) for z,n in seats.items()})]


def _split_counts(seats: Dict[str, float], share: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    a = {s: n * share for s, n in seats.items()}
    b = {s: n * (1 - share) for s, n in seats.items()}
    return a, b


_LP_CACHE: Dict = {}


def _cached_lp(b: Building, cfg: Config, beh, v: float, seats: Dict[str, float],
               prep_min: float, removed) -> Dict:
    'the LP takes seconds; cache per building layout.'
    key = (ENGINE_VERSION, tuple(b.edges), tuple(sorted(b.closed)), tuple(sorted(removed)), beh.name, v,
           tuple(sorted(seats.items())), prep_min, cfg.phys, cfg.geo)
    if key not in _LP_CACHE:
        disk = _disk_cache_path("lp", key)
        if disk and os.path.exists(disk):
            with open(disk, "rb") as fh:
                _LP_CACHE[key] = pickle.load(fh)
            return _LP_CACHE[key]
        #
        # Solve in the building-egress view, then map portico gates back to the real side exits
        bv = egress_view(b)
        tau = cell_times(bv, cfg, v, beh)
        d = exit_distances(bv, tau, set(removed))
        lp = lp_assignment(bv, d, seats, prep_min, set(removed))
        lp["exits"] = [bv.gate_to_exit.get(e, e) for e in lp["exits"]]
        _LP_CACHE[key] = lp
        if disk:
            os.makedirs(os.path.dirname(disk), exist_ok=True)
            with open(disk, "wb") as fh:
                pickle.dump(lp, fh)
    return _LP_CACHE[key]


def gate_finish(res: SimResult) -> Dict[str, float]:
    'Building-egress finish time per physical exit (side exits count when entering the portico).'
    b = res.b
    gate_of = {}
    for i, k in b.edges:
        ci, ck = b.cell[i], b.cell[k]
        if ci.kind != "exit" and not ci.outside and (ck.kind == "exit" or ck.outside):
            if ck.kind == "exit":
                gate_of[(i, k)] = k
            else:   # portico → the exit it leads to
                gate_of[(i, k)] = [x for x in b.out[k] if b.cell[x].kind == "exit"][0]
    out = {}
    for e in b.exits:
        cols = [m for m, ed in enumerate(res.edges) if gate_of.get(ed) == e]
        if not cols:
            continue
        cum = np.cumsum(res.flows[:, cols].sum(1))
        tot = cum[-1]
        if tot == 0:
            out[e] = float("nan")
            continue
        out[e] = float(res.t[np.flatnonzero(cum >= tot)[0]])
    return out


_REFINE_CACHE: Dict = {}

#
# Refining the LED plan takes ~80 s; results are cached on disk keyed by a parameter hash.
CACHE_DIR = os.environ.get("BAILEY_CACHE", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "outputs", "cache"))


def _disk_cache_path(kind: str, key) -> Optional[str]:
    if not CACHE_DIR:
        return None
    h = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{kind}_{h}.pkl")


def _cached_refine(sc: Scenario, cfg: Config, b: Building, lp: Dict, led_beh, self_beh,
                   seats: Dict[str, float], v: float, iters: int = 25,
                   c_design: Optional[Dict[str, float]] = None):
    """The LP is a fluid bound that ignores start-up and merging losses, so following it
    literally overloads the main exit. Starting from the LP split, we iterate with the
    simulator: shift a little load from exits that finish late to exits that finish early.
renormalised per seat zone."""
    c_design = c_design or {z: COMPLIANCE["led"] for z in seats}
    key = (ENGINE_VERSION, tuple(b.edges), tuple(sorted(b.closed)), tuple(sorted(b.blocked)), sc.discovery,
           led_beh, self_beh, v, tuple(sorted(seats.items())), sc.prep_key, tuple(sorted(REACTION_SPECS[sc.prep_key].items())),
           cfg.phys, cfg.geo, sc.manager_nodes, sc.managers, sc.manager_radius_m, sc.manager_p_follow,
           tuple(sorted(MANAGERS.items())), MANAGER_DEFAULT_NODES, BEHAVIOURS["concert_managed"],
           tuple(sorted((z, round(c, 4)) for z, c in c_design.items())))
    if key in _REFINE_CACHE:
        return _REFINE_CACHE[key]
    disk = _disk_cache_path("refine", key)
    if disk and os.path.exists(disk):
        with open(disk, "rb") as fh:
            _REFINE_CACHE[key] = pickle.load(fh)
        return _REFINE_CACHE[key]
    zones, exits = lp["zones"], lp["exits"]
    tau = cell_times(b, cfg, v, led_beh)
    d = exit_distances(b, tau, set(b.blocked) if sc.discovery == "known" else set())
    allow = np.array([[np.isfinite(d[b.idx[z], b.exits.index(e)]) for e in exits] for z in zones])
    n_z = np.array([seats[z] for z in zones])
    y = lp["y"] + 0.02 * n_z[:, None] * allow
    y = y / y.sum(1, keepdims=True) * n_z[:, None]
    rnd = mean_field_release(sc.prep_key)
    best, best_T, hist = y.copy(), float("inf"), []
    for it in range(iters):
        classes = []
        for ei, e in enumerate(exits):
            cnt = {z: c_design[z] * y[zi, ei] for zi, z in enumerate(zones) if y[zi, ei] > 1e-9}
            if sum(cnt.values()) > 1e-9:
                if sc.population == "managers":
                    local = {i: led_beh for i in _manager_setup(sc, cfg, b)["manager_covered_nodes"]}
                    classes.append(PopClass(f"mgr→{e}", self_beh, cnt, {i: e for i in b.transient}, local))
                else:
                    classes.append(PopClass(f"led→{e}", led_beh, cnt, {i: e for i in b.transient}))
        rest = {z: (1 - c_design[z]) * seats[z] for z in seats}
        if sc.population == "led":
            classes += _local_manager_classes(rest, b, cfg, v,
                        set(sc.blocked) if sc.discovery == "known" else set(), _manager_setup(sc, cfg, b))
        else:
            classes.append(PopClass("self", self_beh, rest))
        r = simulate(b, cfg, classes, v, randomness=rnd, discovery=sc.discovery,
                     broadcast_delay=sc.broadcast_delay)
        hist.append(r.T)
        if r.T < best_T:
            best, best_T = y.copy(), r.T
        F = gate_finish(r)
        Fv = np.array([F.get(e, np.nan) for e in exits])
        load = y.sum(0)
        ok = np.isfinite(Fv)
        if ok.sum() < 2:
            break
        Fbar = np.average(Fv[ok], weights=np.maximum(load[ok], 1e-9))
        kappa = 1.5 / (1 + 0.15 * it)
        g = np.where(ok, np.exp(-kappa * (Fv - Fbar) / max(Fbar, 1.0)), 1.0)
        y = y * g[None, :] * allow
        y = y / np.maximum(y.sum(1, keepdims=True), 1e-12) * n_z[:, None]
    _REFINE_CACHE[key] = (best, hist)
    if disk:
        os.makedirs(os.path.dirname(disk), exist_ok=True)
        with open(disk, "wb") as fh:
            pickle.dump((best, hist), fh)
    return best, hist


def build_population(sc: Scenario, cfg: Config, b: Building) -> Tuple[List[PopClass], Dict]:
    'Split the 1329 people into behaviour classes and build the instruction plan.'
    seats = {k: float(v) for k, v in cfg.pop.seat_counts().items()}
    v = getattr(cfg.phys, sc.v_key)
    info: Dict = {}
    classes: List[PopClass] = []

    #
    # Plans use the believed map: hidden blockages are not known to the planners either
    removed = set(sc.blocked) if sc.discovery == "known" else set()

    #
    # See docs/IMPLEMENTATION.md for implementation details.
    # Wheelchair users (positions in config.WHEELCHAIR_SEATS): own class with a separate speed and no stairs,
    # guided to the nearest step-free way out (landing outside the main doors, or the portico).
    if sc.wheelchairs:
        if isinstance(sc.wheelchairs, bool) or int(sc.wheelchairs) != sc.wheelchairs or not 0 <= sc.wheelchairs <= cfg.pop.wheelchairs:
            raise ValueError("Wheelchair count must be an integer from 0 to the problem maximum (3)")
        wz = {}; remaining = sc.wheelchairs
        for z, capacity in WHEELCHAIR_SEATS.items():
            k = min(remaining, capacity)
            if k:
                wz[z] = k; remaining -= k
        if remaining:
            raise ValueError("Wheelchair placement does not have enough modeled spaces")
        for z, k in wz.items():
            seats[z] -= k
        stairs = {x.id for x in b.cells if x.stair}
        tau_w = cell_times(b, cfg, v, BEHAVIOURS["wheelchair"])
        plan_w = plan_nearest(b, exit_distances(b, tau_w, removed | stairs))
        if sc.wheelchair_guide == "main" and "EXIT-Main" not in removed:
            plan_w = {i: "EXIT-Main" for i in b.transient}     # all to the main doors
        classes.append(PopClass("wheelchair", BEHAVIOURS["wheelchair"], {z: float(k) for z, k in wz.items()},
                                plan_w))
        info.update(wheelchair_seats=wz, wheelchair_location_source=dict(WHEELCHAIR_LOCATION_SOURCE))

    info.update(_manager_setup(sc, cfg, b))

    if sc.population == "normal":
        classes.append(PopClass("normal", BEHAVIOURS["normal"], seats))
    elif sc.population in ("lecture",):
        classes.append(PopClass("lecture", BEHAVIOURS["lecture"], seats))
    elif sc.population == "managers":
        classes += _local_manager_classes(seats, b, cfg, v, removed, info)
        initial = sum(n * info["compliance_by_zone"][z] for z, n in seats.items()) / sum(seats.values())
        info.update(compliance=initial, plan_kind="local managers: nearest available exit",
                    compliance_definition="initial seat-node coverage times response share; not all-time exposure")
    elif sc.population in ("led", "lecture_led"):
        #
        #
        # LED: colour-coded strips implement the LP system-optimal split; followers are
        # grouped by their assigned exit, and each group is guided to that exit everywhere.
        c = COMPLIANCE["led"] if sc.compliance is None else sc.compliance
        led_beh = BEHAVIOURS["concert_led" if sc.population == "led" else "lecture_led"]
        self_beh = BEHAVIOURS["concert_self" if sc.population == "led" else "lecture"]
        lp = _cached_lp(b, cfg, led_beh, v, seats, prep_min(sc.prep_key), removed)
        # LED following is separate from local managers. Managers do not magically
        # turn unexposed/non-LED people into permanent LED followers.
        c_design = {z: COMPLIANCE["led"] for z in seats}
        # Hardware routing stays fixed while manager placements/counts are compared.
        design_sc = sc.with_(managers=MANAGERS["law_min"] if sc.population == "led" else None,
                             manager_nodes=None, manager_radius_m=None, manager_p_follow=None,
                             compliance=None)
        y, hist = _cached_refine(design_sc, cfg, b, lp, led_beh, self_beh, seats, v, c_design=c_design)
        if not 0 <= c <= 1:
            raise ValueError("LED compliance must be in [0,1]")
        cz = {z: c for z in seats}
        for ei, e in enumerate(lp["exits"]):
            counts = {z: cz[z] * y[zi, ei] for zi, z in enumerate(lp["zones"]) if y[zi, ei] > 1e-9}
            if sum(counts.values()) > 1e-9:
                classes.append(PopClass(f"led→{e}", led_beh, counts, {i: e for i in b.transient}))
        rest = {z: n * (1 - cz[z]) for z, n in seats.items()}
        if sc.population == "led":
            classes += _local_manager_classes(rest, b, cfg, v, removed, info)
        else:
            classes.append(PopClass("self", self_beh, rest))
        c = 1 - sum(rest.values()) / max(sum(seats.values()), 1e-9)
        info.update(compliance=c, plan_kind="LP + simulation balancing", T_lp=lp["T_lp"], lp=lp,
                    y_led=y, refine_history=hist)
    else:
        raise ValueError(sc.population)
    classes = integer_classes(classes)
    if "compliance" in info:
        denom = sum(sum(c.seat_counts.values()) for c in classes if c.name != "wheelchair")
        if sc.population == "managers":
            guided = sum(n for c in classes if c.name == "manager_receptive"
                         for z, n in c.seat_counts.items() if z in c.node_behaviours)
        else:
            guided = sum(sum(c.seat_counts.values()) for c in classes if c.name.startswith("led→"))
        info["compliance_expected"] = info["compliance"]
        info["compliance"] = guided / max(denom, 1)
    return classes, info


def run(sc: Scenario, cfg: Config, randomness=None, queue_model=None) -> Tuple[SimResult, Dict]:
    'run one scenario.'
    b = build_building(cfg, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
    classes, info = build_population(sc, cfg, b)
    v = getattr(cfg.phys, sc.v_key)
    rnd = randomness or mean_field_release(sc.prep_key)
    res = simulate(b, cfg, classes, v, randomness=rnd, queue_model=queue_model,
                   discovery=sc.discovery, broadcast_delay=sc.broadcast_delay)
    res.meta.update(scenario=sc.name, question=sc.question, zh=sc.zh, prep=dict(REACTION_SPECS[sc.prep_key]),
                    prep_distribution=REACTION_SPECS[sc.prep_key]["family"], reaction_spec=dict(REACTION_SPECS[sc.prep_key]),
                    barriers=list(sc.barriers), closed=list(sc.closed),
                    compliance=info.get("compliance"), plan_kind=info.get("plan_kind"),
                    broadcast_delay=sc.broadcast_delay)
    res.meta.update({k: val for k, val in info.items() if k.startswith("manager") or k.startswith("wheelchair")})
    return res, info


def prepare(sc: Scenario, cfg: Config):
    'For Monte Carlo: build the graph and population (incl. the LED plan) once.'
    b = build_building(cfg, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
    classes, info = build_population(sc, cfg, b)
    return b, classes, getattr(cfg.phys, sc.v_key), info


def summary_row(res: SimResult) -> Dict:
    g = res.exit_groups()
    return dict(scenario=res.meta.get("scenario"), question=res.meta.get("question"),
                zh=res.meta.get("zh"), T=res.T, T90=res.time_to_fraction(0.9),
                mean_time=res.mean_time, meets_180=bool(res.T <= 180),
                **{f"n[{k}]": round(v, 1) for k, v in g.items()})


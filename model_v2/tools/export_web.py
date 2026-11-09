"""/ Export ready-made model data for the web engine (src/model-data.js).

exports two things:
              LED plans and blocked-exit rebalancing plans (people per seat zone → exit). The web page has
              no LP solver, so it uses the plans computed in Python.
              Deterministic results of the formal run; the web engine's test must reproduce each of them.

reads caches only, never simulates:
    A plan missing from the cache is skipped with [skip]; nothing is recomputed.
    Must run with the model code that produced the caches; stops if the code hash differs.

run:
    python tools/export_web.py --out path/to/model-data.js"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bailey import scenarios as S  # noqa: E402
from bailey.building import build_building  # noqa: E402
from bailey.config import BEHAVIOURS, DEFAULT as cfg, MANAGERS, prep_min  # noqa: E402
from bailey.validation import capacity_variants  # noqa: E402
from bailey import analysis  # noqa: E402

PAPER_CACHE = os.path.join(HERE, "outputs", "paper", "cache")
CAP_KEY = {"SFPE design (model default)": "sfpe", "theatre drill 0.96 p/(m·s)": "drill",
           "lab bottleneck 1.9 p/(m·s)": "lab"}


class NoCompute(RuntimeError):
    pass


def _guard(*_a, **_k):
    raise NoCompute("cache miss")


def plan_key(pop, barriers=(), blocked=(), discovery="known", cap="sfpe", wc="-"):
    'must match planKey() in src/engine.js.'
    bar = ",".join(sorted("-".join(p) for p in barriers)) or "none"
    blk = ",".join(sorted(blocked)) or "none"
    return f"{pop}|bar={bar}|blk={blk}|disc={discovery if blocked else '-'}|cap={cap}|wc={wc}"


def _num(x):
    x = float(x)
    return None if not math.isfinite(x) else round(x, 4)


def load(name):
    with open(os.path.join(PAPER_CACHE, name + ".pkl"), "rb") as fh:
        return pickle.load(fh)


def export_plans():
    'LED plans per population × barriers × capacity.'
    plans = {}
    saved = S.simulate, S.lp_assignment
    S.simulate, S.lp_assignment = _guard, _guard          # caches only
    try:
        for cap_label, c in capacity_variants(cfg).items():
            for lay, cut in analysis.barrier_layouts().items():
                for pop, base in (("led", "concert"), ("lecture_led", "lecture")):
                    sc = S.SCENARIOS[base].with_(name="x", barriers=cut, population=pop)
                    b = build_building(c, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
                    key = plan_key(pop, cut, cap=CAP_KEY[cap_label])
                    try:
                        _, info = S.build_population(sc, c, b)
                    except NoCompute:
                        print(f"[skip] {key}: not in cache")
                        continue
                    lp = info["lp"]
                    plans[key] = dict(zones=list(lp["zones"]), exits=list(lp["exits"]),
                                      y=np.round(info["y_led"], 6).tolist(), T_lp=_num(lp["T_lp"]))
                    print(f"[plan] {key}")
        #
        from dataclasses import replace
        sc = S.SCENARIOS["blocked"].with_(name="blocked_rebalanced", discovery="known")
        b = build_building(cfg, blocked=sc.blocked)
        v = cfg.phys.v_emergency
        seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
        c_mgr = {z: S._manager_setup(sc, cfg, b)["manager_p_follow"] for z in seats}
        for name, bg, bq in (("instructions", None, None), ("strict", 10.0, None), ("hold", 10.0, 0.0)):
            beh = BEHAVIOURS["concert_managed"]
            if bg is not None:
                beh = replace(beh, name=f"concert_managed_b4_{bg:g}", beta_guide=bg)
            if bq is not None:
                beh = replace(beh, name=beh.name + f"_b3_{bq:g}", beta_queue=bq)
            try:
                lp = S._cached_lp(b, cfg, beh, v, seats, prep_min(sc.prep_key), set(sc.blocked))
                y, _ = S._cached_refine(sc, cfg, b, lp, beh, BEHAVIOURS["concert_self"], seats, v, c_design=c_mgr)
            except NoCompute:
                print(f"[skip] rebalance {name}: not in cache")
                continue
            plans[f"rebalance:{name}"] = dict(zones=list(lp["zones"]), exits=list(lp["exits"]),
                                              y=np.round(y, 6).tolist(), T_lp=_num(lp["T_lp"]),
                                              beta_guide=beh.beta_guide, beta_queue=beh.beta_queue)
            print(f"[plan] rebalance:{name}")
    finally:
        S.simulate, S.lp_assignment = saved
    return plans


def export_refs():
    'deterministic results of the formal run.'
    refs = []
    det = load("det")
    for n, r in det["data"]["res"].items():
        refs.append(dict(group="main", spec=dict(scenario=n), T=_num(r.T), T_ground=_num(r.T_ground),
                         mean_time=_num(r.mean_time), exits={k: _num(v) for k, v in r.exit_groups().items()},
                         remaining_inside=[round(float(x), 3) for x in r.remaining_inside],
                         class_T={k: _num(v) for k, v in r.class_T.items()}))
    sw = load("sweeps")["data"]
    for row in sw["blocked"].itertuples():
        if row.target == "none":
            continue
        refs.append(dict(group="blocked", spec=dict(scenario="concert", blocked=[row.target], discovery=row.discovery),
                         T=_num(row.T), T_ground=_num(row.T_ground)))
    lays = analysis.barrier_layouts()
    for row in sw["barrier"].itertuples():
        pop = row.guidance if row.event == "concert" else ("lecture_led" if row.guidance == "led" else "lecture")
        refs.append(dict(group="barrier", spec=dict(scenario=row.event, barriers=[list(p) for p in lays[row.layout]],
                                                    population=pop), T=_num(row.T)))
    for row in sw["compliance"].itertuples():
        refs.append(dict(group="compliance", spec=dict(scenario=row.scenario, compliance=float(row.compliance)),
                         T=_num(row.T)))
    for row in sw["beta"].itertuples():
        refs.append(dict(group="beta", spec=dict(scenario=row.scenario, beta={row.parameter: float(row.value)}),
                         T=_num(row.T)))
    for row in sw["checks"].itertuples():
        refs.append(dict(group="numerics", spec=dict(scenario=row.scenario, dt=float(row.dt),
                                                     walk_model=row.walk_model), T=_num(row.T)))
    bc = load("bc")["data"]
    for row in bc["det"].itertuples():
        refs.append(dict(group="broadcast", spec=dict(scenario="blocked", broadcast_delay=float(row.delay)),
                         T=_num(row.T)))
    for mode, T in bc["refs"].items():
        refs.append(dict(group="broadcast", spec=dict(scenario="blocked", discovery=mode), T=_num(T)))
    dz = load("design")["data"]["runs"]
    for row in dz.itertuples():
        base = "combo" if (row.led and row.barriers) else "led" if row.led else "barriers" if row.barriers else "concert"
        refs.append(dict(group="managers", spec=dict(scenario=base, managers=int(row.managers)), T=_num(row.T_det)))
    from make_figures import WHEEL_LAYOUTS
    lay_by_label = {k: v for k, v in WHEEL_LAYOUTS.items()}
    preset = list(WHEEL_LAYOUTS.values())[0]
    for row in load("wheel")["data"].itertuples():
        wz = preset if row.layout == "preset" else lay_by_label[row.layout]
        refs.append(dict(group="wheelchair", spec=dict(scenario=row.scenario, wheelchairs=3, wheelchair_seats=wz,
                                                       wheelchair_guide=row.guide, wheelchair_delay=float(row.delay)),
                         T=_num(row.T), T_ground=_num(row.T_ground), T_refuge=_num(row.T_refuge),
                         T_street=_num(row.T_street)))
    rb = os.path.join(HERE, "outputs", "paper", "tables", "blocked_rebalanced.csv")
    if os.path.exists(rb):      # follow-up experiment
        import pandas as pd
        for row in pd.read_csv(rb).itertuples():
            v = row.variant
            kind = ("hold" if "queue held" in v else "strict" if "strictly" in v
                    else "instructions" if "re-balanced instructions" in v else None)
            if kind:
                refs.append(dict(group="rebalance", spec=dict(scenario="blocked", rebalance=kind), T=_num(row.T_det)))
    vd = load("valid")["data"]["bracket_det"]
    for row in vd.itertuples():
        refs.append(dict(group="capacity", spec=dict(scenario=row.scenario, capacity=CAP_KEY[row.capacity]),
                         T=_num(row.T)))
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    web = os.path.join(os.path.dirname(HERE), "web_v2")
    default_out = os.path.join(web, "src", "model-data.js") if os.path.isdir(web) else None
    ap.add_argument("--out", default=default_out, required=default_out is None)
    a = ap.parse_args()
    from make_figures import code_hash
    sig = load("det")["__sig__"]["code"]
    if sig != code_hash():
        sys.exit(f"cached results come from model code {sig}, current code is {code_hash()}: "
                 "rerun the formal pipeline first (or check out that code).")
    data = dict(source=dict(model_code_hash=sig, note="exported by model/tools/export_web.py from formal-v1 caches"),
                plans=export_plans(), refs=export_refs())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("/* Generated by model/tools/export_web.py — do not edit by hand.\n"
                 "   LED / rebalancing plans and reference results from the Python model (formal run). */\n")
        fh.write("(function(root){'use strict';const DATA=")
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";root.BaileyData=DATA;if(typeof module!=='undefined')module.exports=DATA;})"
                 "(typeof window!=='undefined'?window:globalThis);\n")
    print(f"wrote {a.out}: {len(data['plans'])} plans, {len(data['refs'])} reference results")


if __name__ == "__main__":
    main()


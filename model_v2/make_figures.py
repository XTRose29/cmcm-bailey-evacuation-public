"""/ One command: compute everything and draw every paper figure.

Monte Carlo replicates
redraw from cache only
recompute some stages

outputs:  outputs/paper/figures/*.png|pdf   outputs/paper/tables/*.csv
                 Each cache stores a signature (model-code hash + stage parameters); mismatch → recompute.
run records:
one entry appended per run"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bailey import analysis, paper, uncertainty  # noqa: E402
from bailey.building import build_building  # noqa: E402
from bailey.config import BEHAVIOURS, DEFAULT, PREP_LOGNORMAL, PREP_WINDOWS, REACTION_SPECS  # noqa: E402
from bailey.flow import PopClass, simulate  # noqa: E402
from bailey.markov import MarkovClass, absorbing_analysis, transition_matrix  # noqa: E402
from bailey.montecarlo import make_job, run_batch, summary  # noqa: E402
from bailey.plugins.queueing import FlowQueueModel  # noqa: E402
from bailey.plugins.randomness import MeanFieldPlaceholder  # noqa: E402
from bailey.routing import Router, cell_times, leave_prob  # noqa: E402
from bailey.scenarios import SCENARIOS, gate_finish, run, summary_row  # noqa: E402

ROOT = os.environ.get("BAILEY_PAPER_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "outputs", "paper")
FIG, TAB, CACHE = (os.path.join(ROOT, d) for d in ("figures", "tables", "cache"))
MAIN = ["normal", "concert", "blocked", "lecture", "barriers", "led", "combo", "lecture_led", "lecture_barriers"]
Q13 = ["normal", "concert", "blocked", "lecture"]
cfg = DEFAULT
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "outputs", "runs")
EXPLOG = os.path.join(HERE, "outputs", "EXPERIMENT_LOG.md")
STAGE_TIMES = {}

#
# Model code (not plotting) determines the results; its hash goes into every cache signature.
_MODEL_FILES = ["guidance.py", "integer.py", "config.py", "building.py", "routing.py", "markov.py", "flow.py", "scenarios.py", "analysis.py",
                "montecarlo.py", "uncertainty.py", "cost.py", "validation.py", "plugins/queueing.py",
                "plugins/randomness.py"]


def code_hash() -> str:
    h = hashlib.sha1()
    for f in _MODEL_FILES:
        with open(os.path.join(HERE, "bailey", f), "rb") as fh:
            h.update(fh.read())
    for fn in (stage_det, stage_markov, stage_sweeps, stage_mc, stage_mc_variants, stage_broadcast, stage_tornado,
               stage_pmc, stage_design, stage_dispersion, stage_wheelchair, stage_validation, stage_wheel_mc,
               stage_rowcap):
        h.update(inspect.getsource(fn).encode("utf-8"))
    return h.hexdigest()[:12]


def cached(name, fn, redo, params=()):
    'cache with a signature (code hash + stage parameters).'
    path = os.path.join(CACHE, name + ".pkl")
    sig = dict(code=code_hash(), params=list(params))
    if os.path.exists(path) and name not in redo and "all" not in redo:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict) and obj.get("__sig__") == sig:
            print(f"[cache]   {name}: reused (sig {sig})", flush=True)
            STAGE_TIMES[name] = "cached"
            return obj["data"]
        print(f"[cache]   {name}: signature changed → recompute", flush=True)
    t = time.time()
    print(f"[compute] {name} ... params={list(params)}", flush=True)
    out = fn()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"__sig__": sig, "data": out}, fh)
    STAGE_TIMES[name] = round(time.time() - t, 1)
    print(f"[compute] {name} done in {time.time() - t:.0f} s", flush=True)
    return out


def load_cached(name):
    path = os.path.join(CACHE, name + ".pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "__sig__" in obj:
        if obj["__sig__"]["code"] != code_hash():
            print(f"[warn] cache {name} was computed with other model code ({obj['__sig__']['code']})")
        return obj["data"]
    return obj


class Tee:
    'write to screen and log file.'

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


# ---------------------------------------------------------------------------
# computations
# ---------------------------------------------------------------------------
def stage_det():
    res, info, qa, fin = {}, {}, {}, {}
    for n in MAIN + ["wheelchair"]:
        r, inf = run(SCENARIOS[n], cfg, queue_model=FlowQueueModel())
        res[n], info[n] = r, {k: v for k, v in inf.items() if k not in ("plan",)}
        qa[n] = FlowQueueModel().analyze(r)
        fin[n] = gate_finish(r)
    return dict(res=res, info=info, qa=qa, finish=fin)


def stage_markov():
    out = {}
    for n in ("concert", "lecture", "normal"):
        rep = analysis.markov_report(cfg, n)
        out[n] = dict(mix=rep["mix"], lb=rep["lb"], lp_T=rep["lp"]["T_lp"], passes=rep["passes"], mu=rep["mu"],
                      exits=list(rep["b"].exits), groups=[rep["b"].cell[e].exit_group for e in rep["b"].exits])
    #
    b = build_building(cfg)
    beh = BEHAVIOURS["concert_self"]
    v = cfg.phys.v_emergency
    tau = cell_times(b, cfg, v, beh)
    a = absorbing_analysis(b, cfg, MarkovClass("all", Router(b, beh, tau, set()), leave_prob(tau, cfg.phys.dt),
                                               np.ones(b.nT)))
    out["t_cell"] = {c: float(a["t_sec"][i]) for i, c in enumerate(b.transient)}
    #
    beh0 = dataclasses.replace(beh, beta_queue=0.0)
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    r0 = Router(b, beh0, tau, set())
    p = leave_prob(tau, cfg.phys.dt)
    P = transition_matrix(b, r0, p)
    x = np.zeros(len(b.ids))
    for s, n_ in seats.items():
        x[b.idx[s]] = n_
    n_tot = x.sum()
    steps = int(600 / cfg.phys.dt)
    frac = []
    for _ in range(steps):
        x = x @ P
        frac.append(x[:b.nT].sum() / n_tot)
    t = cfg.phys.dt * np.arange(1, steps + 1)
    mc = absorbing_analysis(b, cfg, MarkovClass("s", r0, p, np.array([seats.get(c, 0.0) for c in b.transient])))
    curves = {"analytic expected population x₀Pᵗ (geometric stay)": dict(t=t, frac=np.array(frac), color=paper.ORANGE,
                                                               lw=3.0, mean=mc["mean_time"])}
    for wm, col, ls in (("geometric", paper.INK, (0, (3, 2))), ("erlang", paper.BLUE, "-")):
        c2 = cfg.with_(phys=dataclasses.replace(cfg.phys, walk_model=wm))
        sim = simulate(b, c2, [PopClass("s", beh0, seats)], v, capacities=False, threshold=1e-6, t_max=3000,
                       walk_model=wm)
        lab = ("integer reference, no capacities, geometric stages" if wm == "geometric"
               else "integer reference, no capacities, rounded walking times")
        curves[lab] = dict(t=sim.t, frac=sim.remaining / sim.meta["n"], color=col, ls=ls, lw=1.3,
                           mean=sim.meta["mean_time_ground"])
    rfull, _ = run(SCENARIOS["concert"], cfg)
    curves["full model: capacities + queues + reaction time"] = dict(
        t=rfull.t, frac=rfull.remaining / rfull.meta["n"], color=paper.AQUA, lw=1.6,
        mean=rfull.meta["mean_time_ground"])
    out["curves"] = curves
    return out


def _barrier_job(args):
    name, cut, pop, base_name = args
    base = SCENARIOS[base_name]
    population = pop if base_name == "concert" else ("lecture_led" if pop == "led" else "lecture")
    sc = base.with_(name=f"{base_name}_{pop}_{name}", barriers=cut, population=population)
    r, _ = run(sc, cfg)
    return dict(layout=name, guidance=pop, event=base_name, T=r.T, T90=r.time_to_fraction(.9),
                mean_time=r.mean_time)


def _comp_job(args):
    c, name = args
    r, _ = run(SCENARIOS[name].with_(compliance=float(c)), cfg)
    return dict(scenario=name, compliance=float(c), T=r.T, T90=r.time_to_fraction(.9), mean_time=r.mean_time)


def stage_sweeps(workers):
    out = {}
    jobs = [(n, cut, pop, ev) for n, cut in analysis.barrier_layouts().items()
            for pop in ("managers", "led") for ev in ("concert", "lecture")]
    with ProcessPoolExecutor(max_workers=min(workers, 6)) as ex:   # LP needs memory
        out["barrier"] = pd.DataFrame(list(ex.map(_barrier_job, jobs)))
        out["compliance"] = pd.DataFrame(list(ex.map(
            _comp_job, [(c, n) for c in np.linspace(0, 1, 11) for n in ("concert", "barriers", "led", "combo")])))
    out["blocked"] = analysis.blocked_sweep(cfg)
    out["beta"] = analysis.beta_sweep(cfg)
    out["checks"] = analysis.method_checks(cfg)
    return out


def stage_mc(K, workers):
    jobs = {n: make_job(SCENARIOS[n], cfg) for n in MAIN}
    return run_batch(jobs, K, workers)


VARIANTS = {"reaction only\n(baseline R)": (None, "hard", False),
            "service only": ("mean-field", "poisson", False),
            "movement only": ("mean-field", "hard", True)}


def stage_mc_variants(K, workers):
    jobs = {}
    for n in ("concert", "lecture", "normal"):
        for lab, (dist, serv, motion) in VARIANTS.items():
            jobs[f"{n}|{lab}"] = make_job(SCENARIOS[n], cfg, dist=dist, service=serv, keep_curves=False, motion=motion)
    # jobs["normal|both\n(Poisson stream)"] = make_job(SCENARIOS["normal"], cfg, dist="poisson-stream",
                                                     # keep_curves=False)
    out = run_batch(jobs, K, workers)
    return {tuple(k.split("|")): v for k, v in out.items()}


DELAYS = [0, 5, 10, 20, 30, 45, 60, 90]


def stage_broadcast(K, workers):
    rows = []
    for d in DELAYS:
        r, _ = run(SCENARIOS["blocked"].with_(broadcast_delay=float(d)), cfg)
        rows.append(dict(delay=d, T=r.T, T90=r.time_to_fraction(.9)))
    refs = {}
    for mode in ("known", "arrival"):
        r, _ = run(SCENARIOS["blocked"].with_(discovery=mode), cfg)
        refs[mode] = r.T
    jobs = {str(float(d)): make_job(SCENARIOS["blocked"].with_(name=f"blocked_d{d}", broadcast_delay=float(d)), cfg,
                                    keep_curves=False) for d in (0, 10, 20, 30, 45, 60, 90)}
    mcs = {float(k): v for k, v in run_batch(jobs, K, workers).items()}
    return dict(det=pd.DataFrame(rows), refs=refs, mc=mcs)


MGR_LIST = [0, 2, 4, 6, 7, 8, 10, 12, 15, 18]
DESIGN_MGR = [6, 7, 8, 10, 12, 15]


def _design_job(args):
    n, led, bar = args
    base = "combo" if (led and bar) else "led" if led else "barriers" if bar else "concert"
    sc = SCENARIOS[base].with_(name=f"design_{n}_{int(led)}{int(bar)}", managers=n)
    r, info = run(sc, cfg)
    return dict(managers=n, led=led, barriers=bar, T_det=r.T, T90_det=r.time_to_fraction(.9),
                compliance=info.get("compliance"))


def stage_design(K, workers):
    'Q5 + manager model: designs (n managers, LED, barriers) → deterministic T + small MC + yearly cost.'
    from bailey.scenarios import manager_compliance
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    comp = []
    for n in range(0, 19):
        c, k = manager_compliance(seats, n, cfg=cfg)
        comp.append(dict(managers=n, avg=sum(c[z] * seats[z] for z in seats) / sum(seats.values()),
                         **{f"c_{z}": c[z] for z in sorted(seats)}, **{f"k_{z}": k[z] for z in sorted(seats)}))
    # build LED plans once (disk cache) before the pool
    for base in ("led", "combo"):
        run(SCENARIOS[base], cfg)
    designs = [(n, led, bar) for n in DESIGN_MGR for led in (False, True) for bar in (False, True)]
    sweep = [(n, False, False) for n in MGR_LIST if n not in DESIGN_MGR] +\
            [(n, True, False) for n in MGR_LIST if n not in DESIGN_MGR and n >= 0]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        det_rows = list(ex.map(_design_job, designs + sweep))
    jobs = {}
    for n, led, bar in designs:
        base = "combo" if (led and bar) else "led" if led else "barriers" if bar else "concert"
        jobs[f"{n}|{int(led)}|{int(bar)}"] = make_job(SCENARIOS[base].with_(name=f"d{n}{led}{bar}", managers=n),
                                                      cfg, keep_curves=False)
    mcr = run_batch(jobs, K, workers)
    for r in det_rows:
        key = f"{r['managers']}|{int(r['led'])}|{int(r['barriers'])}"
        if key in mcr:
            m = mcr[key]
            r.update(T_mc=m["T_mean"], T_mc_ci_lo=m["T_mean_ci"][0], T_mc_ci_hi=m["T_mean_ci"][1],
                     T_mc05=m["T_05"], T_mc95=m["T_95"], P180=m["P_le_180"], K=m["K"])
    return dict(compliance=pd.DataFrame(comp), runs=pd.DataFrame(det_rows))


KAPPAS = [0.0, 0.5, 2.0]  # κ = 1.0 (Poisson) disabled
DISP_SCEN = ["concert", "barriers", "normal", "lecture"]


def stage_dispersion(K, workers):
    'service variability comparison.'
    jobs = {f"{n}|{k}": make_job(SCENARIOS[n], cfg, dispersion=k, keep_curves=False)
            for n in DISP_SCEN for k in KAPPAS}
    out = run_batch(jobs, K, workers)
    return {(k.split("|")[0], float(k.split("|")[1])): v for k, v in out.items()}


WHEEL_LAYOUTS = {"rear mezzanine ADA platform\n(M2) [preset]": {"M2": 3},
                 "mezzanine spread [hypothetical]": {"M1": 1, "M2": 1, "M3": 1},
                 "orchestra\n(O1, O2, O3)": {"O1": 1, "O2": 1, "O3": 1}}
WHEEL_DELAYS = [0.0, 10.0, 20.0, 30.0]


def _wc_run(base, wz, guide, delay):
    from bailey import config as C
    from bailey.validation import wheelchair_report
    saved_s, saved_b = dict(C.WHEELCHAIR_SEATS), C.BEHAVIOURS["wheelchair"]
    try:
        C.WHEELCHAIR_SEATS.clear(); C.WHEELCHAIR_SEATS.update(wz)
        C.BEHAVIOURS["wheelchair"] = dataclasses.replace(saved_b, delay=delay)
        r, info = run(SCENARIOS[base].with_(name=f"wc_{base}", wheelchairs=3, wheelchair_guide=guide), cfg)
        rep = wheelchair_report(r, cfg)
    finally:
        C.WHEELCHAIR_SEATS.clear(); C.WHEELCHAIR_SEATS.update(saved_s)
        C.BEHAVIOURS["wheelchair"] = saved_b
    sh = {k: v["share"] for k, v in rep["routes"].items()}
    return dict(scenario=base, guide=guide, delay=delay, T=r.T, T_ground=r.T_ground, T_refuge=rep["T_refuge"],
                T_street=rep["T_street"], share_main=sh.get("main stairs", 0.0),
                share_side=sh.get("side stairs W", 0.0) + sh.get("side stairs E", 0.0),
                routes=json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in rep["routes"].items()}))


def stage_wheelchair():
    'Wheelchairs (deterministic): placement × guidance × assistance delay; refuge and street times.'
    rows = []
    for lab, wz in WHEEL_LAYOUTS.items():
        for guide in ("nearest", "main"):
            r = _wc_run("concert", wz, guide, 10.0)
            rows.append(dict(kind="layout", layout=lab, **r))
        r = _wc_run("blocked", wz, "nearest", 10.0)
        rows.append(dict(kind="layout", layout=lab, **r))
    preset = list(WHEEL_LAYOUTS.values())[0]
    for guide in ("nearest", "main"):
        for d in WHEEL_DELAYS:
            rows.append(dict(kind="delay", layout="preset", **_wc_run("concert", preset, guide, d)))
    return pd.DataFrame(rows)


from contextlib import contextmanager


@contextmanager
def _wheel_patch(wz, delay):
    'temporarily set wheelchair seats and assistance delay.'
    from bailey import config as C
    saved_s, saved_b = dict(C.WHEELCHAIR_SEATS), C.BEHAVIOURS["wheelchair"]
    try:
        C.WHEELCHAIR_SEATS.clear(); C.WHEELCHAIR_SEATS.update(wz)
        C.BEHAVIOURS["wheelchair"] = dataclasses.replace(saved_b, delay=delay)
        yield
    finally:
        C.WHEELCHAIR_SEATS.clear(); C.WHEELCHAIR_SEATS.update(saved_s)
        C.BEHAVIOURS["wheelchair"] = saved_b


WHEEL_MC = {   # label → (scenario, seats, guidance, delay)
    "preset seats, nearest exit": ("concert", {"M2": 3}, "nearest", 10.0),
    "preset seats, main doors": ("concert", {"M2": 3}, "main", 10.0),
    "preset seats, main doors, 30 s wait": ("concert", {"M2": 3}, "main", 30.0),
    "orchestra seats, main doors": ("concert", {"O1": 1, "O2": 1, "O3": 1}, "main", 10.0),
    "preset seats, main exit blocked": ("blocked", {"M2": 3}, "nearest", 10.0),
}


def stage_wheel_mc(K, workers):
    'Wheelchair Monte Carlo: refuge and street times under random reaction times + service noise (CRN).'
    jobs = {}
    for lab, (base, wz, guide, delay) in WHEEL_MC.items():
        with _wheel_patch(wz, delay):
            sc = SCENARIOS[base].with_(name=f"wcmc_{base}", wheelchairs=3, wheelchair_guide=guide)
            jobs[lab] = make_job(sc, cfg, keep_curves=False, wheel=True)
    return run_batch(jobs, K, workers)


def stage_rowcap(K, workers):
    'Seat-row capacity sensitivity: deterministic (9 main scenarios) + paired Monte Carlo (Q1–Q3, CRN).'
    cfg_on = cfg.with_(geo=dataclasses.replace(cfg.geo, seat_row_capacity=True))
    rows = []
    for n in MAIN:
        r0, _ = run(SCENARIOS[n], cfg)
        r1, _ = run(SCENARIOS[n], cfg_on)
        rows.append(dict(scenario=n, T_off=r0.T, T_on=r1.T, dT=r1.T - r0.T, Tg_off=r0.T_ground, Tg_on=r1.T_ground))
    jobs = {}
    for n in Q13:
        jobs[f"off|{n}"] = make_job(SCENARIOS[n], cfg, keep_curves=False)
        jobs[f"on|{n}"] = make_job(SCENARIOS[n], cfg_on, keep_curves=False)
    mcr = run_batch(jobs, K, workers)
    mc_rows = []
    for n in Q13:
        a, b_ = mcr[f"off|{n}"], mcr[f"on|{n}"]
        d = b_["T"] - a["T"]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        mc_rows.append(dict(scenario=n, K=len(d), T_off=a["T_mean"], T_on=b_["T_mean"], dT_mean=float(d.mean()),
                            dT_ci_lo=float(d.mean() - 1.96 * se), dT_ci_hi=float(d.mean() + 1.96 * se),
                            P180_off=a["P_le_180"], P180_on=b_["P_le_180"]))
    return dict(det=pd.DataFrame(rows), mc=pd.DataFrame(mc_rows))


def _valid_job(args):
    label, cfgv, name = args
    r, info = run(SCENARIOS[name], cfgv)
    return dict(capacity=label, scenario=name, T=r.T, T90=r.time_to_fraction(.9), mean_time=r.mean_time)


def stage_validation(K, workers):
    """validation:
one door vs closed form"""
    from bailey.validation import capacity_variants, single_door_check, specific_flows
    out = dict(door=single_door_check(cfg))
    sf = {}
    for n in ("concert", "lecture", "normal"):
        r, _ = run(SCENARIOS[n], cfg)
        sf[n] = specific_flows(r, FlowQueueModel().analyze(r))
    out["specific"] = sf
    vars_ = capacity_variants(cfg)
    #
    # build the LED plans for every capacity variant first (disk cache) so workers do not repeat them
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_valid_job, [(lab, c, n) for lab, c in vars_.items() for n in ("led", "combo", "lecture_led")]))
        det = list(ex.map(_valid_job, [(lab, c, n) for lab, c in vars_.items() for n in MAIN]))
    out["bracket_det"] = pd.DataFrame(det)
    jobs = {f"{lab}|{n}": make_job(SCENARIOS[n], c, keep_curves=False)
            for lab, c in vars_.items() for n in Q13}
    mcr = run_batch(jobs, K, workers)
    out["bracket_mc"] = pd.DataFrame([dict(capacity=k.split("|")[0], scenario=k.split("|")[1], K=v["K"],
                                           T_mean=v["T_mean"], T_05=v["T_05"], T_95=v["T_95"],
                                           P_le_180=v["P_le_180"]) for k, v in mcr.items()])
    return out


def stage_tornado(workers):
    return uncertainty.tornado(cfg, scenarios=Q13, workers=workers)


def stage_pmc(K, workers):
    return uncertainty.param_mc(cfg, scenarios=Q13, K=K, workers=workers)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def wilson(k: float, n: int, z: float = 1.96):
    'See docs/IMPLEMENTATION.md for implementation details.'
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def write_tables(det, mk, mc, var, bc, torn, pmc, xt=None):
    os.makedirs(TAB, exist_ok=True)
    rows = []
    for n in MAIN + ["wheelchair"]:
        r = det["res"][n]
        row = summary_row(r)
        row["T_ground"] = r.T_ground
        if mc and n in mc:
            m = mc[n]
            row.update(MC_K=m["K"], MC_T_mean=m["T_mean"], MC_T_mean_ci_lo=m["T_mean_ci"][0],
                       MC_T_mean_ci_hi=m["T_mean_ci"][1], MC_T_sd=m["T_sd"], MC_T05=m["T_05"], MC_T95=m["T_95"],
                       MC_P_le_180=m["P_le_180"],
                       MC_P_wilson_lo=wilson(float(np.sum(m["T"] <= 180)), len(m["T"]))[0],
                       MC_P_wilson_hi=wilson(float(np.sum(m["T"] <= 180)), len(m["T"]))[1])
            #
            tg = np.asarray(m["T_ground"], float)
            kg = float(np.sum(tg <= 180))
            row.update(MC_Tg_mean=float(np.mean(tg)), MC_Tg05=float(np.percentile(tg, 5)),
                       MC_Tg95=float(np.percentile(tg, 95)), MC_Pg_le_180=kg / len(tg),
                       MC_Pg_wilson_lo=wilson(kg, len(tg))[0], MC_Pg_wilson_hi=wilson(kg, len(tg))[1])
        if pmc is not None and n in set(pmc["scenario"]):
            g = pmc[pmc["scenario"] == n]["T"]
            lo_, hi_ = wilson(float(np.sum(g <= 180)), len(g))
            row.update(paramMC_K=len(g), paramMC_T_median=float(g.median()),
                       paramMC_T05=float(np.percentile(g, 5)), paramMC_T95=float(np.percentile(g, 95)),
                       paramMC_P_le_180=float(np.mean(g <= 180)), paramMC_P_wilson_lo=lo_, paramMC_P_wilson_hi=hi_)
            if "T_ground" in pmc:      # only in runs after 2026-11-07
                gg = pmc[pmc["scenario"] == n]["T_ground"]
                row.update(paramMC_Tg_median=float(gg.median()), paramMC_Pg_le_180=float(np.mean(gg <= 180)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "scenario_summary.csv"), index=False)
    for n in Q13 + ["led", "combo"]:
        det["qa"][n]["table"].to_csv(os.path.join(TAB, f"queue_{n}.csv"), index=False)
    lb = mk["concert"]["lb"]
    pd.DataFrame([dict(bound="max-flow (per exit)", T=lb["LB_per_exit"]), dict(bound="max-flow (joint)", T=lb["LB_joint"]),
                  dict(bound="LP system optimum", T=mk["concert"]["lp_T"])]).to_csv(
        os.path.join(TAB, "lower_bounds_concert.csv"), index=False)
    uncertainty.ranges_table().to_csv(os.path.join(TAB, "assumed_parameter_ranges.csv"), index=False)
    if var:
        pd.DataFrame([dict(scenario=k[0], variant=k[1].replace("\n", " "), K=v["K"], T_mean=v["T_mean"],
                           T_sd=v["T_sd"], T95=v["T_95"], P_le_180=v["P_le_180"]) for k, v in var.items()]).to_csv(
            os.path.join(TAB, "mc_randomness_variants.csv"), index=False)
    if mc:
        pairs = [("barriers", "concert"), ("led", "concert"), ("combo", "concert"),
                 ("lecture_led", "lecture"), ("lecture_barriers", "lecture")]
        rows_ = []
        for a_, b_ in pairs:
            d_ = mc[a_]["T"] - mc[b_]["T"]
            se = d_.std(ddof=1) / np.sqrt(len(d_))
            rows_.append(dict(design=a_, baseline=b_, K=len(d_), dT_mean=float(d_.mean()),
                              dT_ci_lo=float(d_.mean() - 1.96 * se), dT_ci_hi=float(d_.mean() + 1.96 * se),
                              dT_sd=float(d_.std(ddof=1)), P_faster=float(np.mean(d_ < 0)),
                              P_slower=float(np.mean(d_ > 0))))
        pd.DataFrame(rows_).to_csv(os.path.join(TAB, "mc_paired_differences.csv"), index=False)
    if bc:
        d = bc["det"].copy()
        d["MC_T_median"] = d["delay"].map(lambda x: np.median(bc["mc"][float(x)]["T"]) if float(x) in bc["mc"] else np.nan)
        d["MC_P_le_180"] = d["delay"].map(lambda x: bc["mc"][float(x)]["P_le_180"] if float(x) in bc["mc"] else np.nan)
        d.to_csv(os.path.join(TAB, "blocked_broadcast_delay.csv"), index=False)
    if torn is not None:
        torn.to_csv(os.path.join(TAB, "tornado.csv"), index=False)
    if pmc is not None:
        pmc.to_csv(os.path.join(TAB, "parameter_mc_runs.csv"), index=False)
    xt = xt or {}
    if xt.get("design"):
        from bailey.cost import design_table
        d = xt["design"]
        d["compliance"].to_csv(os.path.join(TAB, "managers_compliance.csv"), index=False)
        runs = d["runs"]
        full = runs.dropna(subset=["T_mc"]) if "T_mc" in runs else runs
        design_table(full.to_dict("records"), cfg, obj="T_mc").to_csv(os.path.join(TAB, "q5_designs_pareto.csv"),
                                                                     index=False)
        runs.to_csv(os.path.join(TAB, "managers_sweep.csv"), index=False)
        #
        from bailey.cost import cost_sensitivity, led_cost_per_second
        cs = cost_sensitivity(full.to_dict("records"), cfg, obj="T_mc")
        cs["oat"].to_csv(os.path.join(TAB, "q5_cost_sensitivity_oat.csv"), index=False)
        cs["freq"].to_csv(os.path.join(TAB, "q5_cost_sensitivity_pareto_share.csv"), index=False)
        if mc and "lecture_led" in mc and "lecture" in mc:
            saved = -float(np.mean(mc["lecture_led"]["T"] - mc["lecture"]["T"]))
            lc = led_cost_per_second(cfg, saved)
            pd.DataFrame([dict(case="lecture + LED vs lecture", seconds_saved=saved,
                               usd_per_second_median=float(lc["usd_per_second"].median()),
                               usd_per_second_p05=float(lc["usd_per_second"].quantile(.05)),
                               usd_per_second_p95=float(lc["usd_per_second"].quantile(.95)))]).to_csv(
                os.path.join(TAB, "q5_led_lecture_cost_per_second.csv"), index=False)
    if xt.get("disp"):
        pd.DataFrame([dict(scenario=k[0], kappa=k[1], K=v["K"], T_mean=v["T_mean"], T_sd=v["T_sd"], T95=v["T_95"],
                           P_le_180=v["P_le_180"]) for k, v in xt["disp"].items()]).to_csv(
            os.path.join(TAB, "service_dispersion.csv"), index=False)
    if xt.get("wheel") is not None:
        xt["wheel"].to_csv(os.path.join(TAB, "wheelchair.csv"), index=False)
        old = os.path.join(TAB, "wheelchair_layouts.csv")
        if os.path.exists(old):
            os.remove(old)
    if xt.get("wheel_mc"):
        rows_w = []
        for lab, r_ in xt["wheel_mc"].items():
            for col in ("T", "T_refuge", "T_street"):
                a_ = np.asarray(r_[col], float)
                rows_w.append(dict(design=lab, quantity=col, K=len(a_), mean=float(np.nanmean(a_)),
                                   p05=float(np.nanpercentile(a_, 5)), p95=float(np.nanpercentile(a_, 95)),
                                   P_le_180=float(np.mean(a_ <= 180)), P_le_240=float(np.mean(a_ <= 240))))
        pd.DataFrame(rows_w).to_csv(os.path.join(TAB, "wheelchair_mc.csv"), index=False)
    if xt.get("rowcap"):
        xt["rowcap"]["det"].to_csv(os.path.join(TAB, "rowcap_deterministic.csv"), index=False)
        xt["rowcap"]["mc"].to_csv(os.path.join(TAB, "rowcap_mc_paired.csv"), index=False)
    if xt.get("valid"):
        v = xt["valid"]
        v["door"].to_csv(os.path.join(TAB, "validation_single_door.csv"), index=False)
        pd.concat([df.assign(scenario=n) for n, df in v["specific"].items()]).to_csv(
            os.path.join(TAB, "validation_specific_flows.csv"), index=False)
        v["bracket_det"].pivot(index="scenario", columns="capacity", values="T").reindex(MAIN).to_csv(
            os.path.join(TAB, "validation_capacity_bracket.csv"))
        v["bracket_mc"].to_csv(os.path.join(TAB, "validation_capacity_bracket_mc.csv"), index=False)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_all(det, mk, sw, mc, var, bc, torn, pmc, xt=None):
    xt = xt or {}
    P = paper
    res, qa = det["res"], det["qa"]
    made = []
    b0 = build_building(cfg)
    made.append(P.fig_floorplan(b0, FIG))

    # chain
    made.append(P.fig_markov_time_map(b0, mk["t_cell"], FIG))
    panels = {}
    for n, title in (("concert", "(a) Concert, crowd managers"),
                     ("lecture", "(b) Lecture, no guidance")):
        mix = mk[n]["mix"]
        groups = mk[n]["groups"]
        Bg = {g: np.array([sum(mix["B_seat"][s, e] for e, gg in enumerate(groups) if gg == g)
                           for s in range(len(mix["seats"]))]) for g in P.EXG}
        panels[title] = dict(seats=mix["seats"], B_group=Bg, t_seat=mix["t_seat"])
    made.append(P.fig_markov_exit_probs(panels, FIG))
    tab = qa["concert"]["table"].set_index("junction")
    rows = []
    for j, pas in mk["concert"]["passes"].items():
        mu = mk["concert"]["mu"][j]
        sim = tab.loc[j, "passed"] if j in tab.index else 0.0
        rows.append(dict(junction=j, label=j, chain_load=pas / mu, sim_load=sim / mu, wait_person_s=pas / mu))
    rows = pd.DataFrame(rows).sort_values("chain_load", ascending=False)
    rows = P.dedupe_mirrors(rows, col="chain_load").head(14)
    made.append(P.fig_markov_screening(rows, FIG))
    made.append(P.fig_chain_vs_engine(mk["curves"], FIG))

    # flow model
    made.append(P.fig_evac_curves(res, Q13, FIG, lb=mk["concert"]["lb"]["LB"],
                                  title="Evacuation curves, deterministic main model (Q1–Q3)"))
    made.append(P.fig_evac_curves(res, ["concert", "barriers", "led", "combo"], FIG, lb=mk["concert"]["lb"]["LB"],
                                  fname="C1b_evacuation_curves_Q4", title="Q4 designs vs. the concert baseline"))
    made.append(P.fig_evac_curves(res, ["lecture", "lecture_barriers", "lecture_led"], FIG,
                                  lb=mk["concert"]["lb"]["LB"], fname="C1c_evacuation_curves_Q4_lecture",
                                  title="Q4 designs in the lecture setting"))
    for n in ("concert", "blocked", "normal", "lecture", "combo"):
        made.append(P.fig_snapshots(res[n], FIG, n))
    made.append(P.fig_kymograph(res, Q13, FIG, cfg.phys.k_max))
    made.append(P.fig_exit_balance(res, det["finish"], MAIN, FIG))
    made.append(P.fig_exit_flow(res, Q13, FIG))

    # queueing
    for n in Q13:
        made.append(P.fig_ncurves(qa[n], FIG, n))
    made.append(P.fig_rho_heatmap(qa, Q13, FIG))
    made.append(P.fig_bottleneck_ranking(qa, Q13, FIG))
    made.append(P.fig_mm1(qa, ["concert", "lecture"], FIG, mu_ref=cfg.phys.q * cfg.geo.w_mezz_door))
    made.append(P.fig_wait_map(qa, res, Q13, FIG))

    # Poisson + MC
    if mc:
        windows = {n: REACTION_SPECS[SCENARIOS[n].prep_key] for n in ("concert", "lecture", "normal")}
        made.append(P.fig_reaction(windows, mc["normal"], FIG))
        made.append(P.fig_service_noise(mc["concert"], res["concert"], FIG, "concert"))
        made.append(P.fig_service_noise(mc["blocked"], res["blocked"], FIG, "blocked"))
        detT = {n: res[n].T for n in MAIN}
        made.append(P.fig_T_distribution(mc, MAIN, detT, FIG))
        made.append(P.fig_mc_bands(mc, res, Q13, FIG))
        made.append(P.fig_crn(mc, [("barriers", "concert"), ("led", "concert"), ("combo", "concert"),
                                   ("lecture_barriers", "lecture"), ("lecture_led", "lecture")], FIG))
        made.append(P.fig_convergence(mc, ["concert", "lecture", "blocked", "normal"], pmc, FIG))
        made.append(P.fig_T_definition(mc, ["concert", "lecture", "barriers", "led", "normal", "blocked"], FIG))
    if var and mc:
        groups = {}
        for n in ("concert", "lecture", "normal"):
            g = {"reaction only\n(baseline R)": var[(n, "reaction only\n(baseline R)")],
                 "service only": var[(n, "service only")],
                 "movement only": var[(n, "movement only")],
                 "all\n(baseline R)": mc[n]}
            # if (n, "both\n(Poisson stream)") in var:
                # g["both\n(Poisson stream)"] = var[(n, "both\n(Poisson stream)")]
            groups[n] = g
        made.append(P.fig_reaction_sensitivity(groups, {n: res[n].T for n in groups}, FIG))
    if bc:
        refs = {f"known from t = 0: {bc['refs']['known']:.0f} s": (bc["refs"]["known"], (0, (1, 2))),
                f"found at the door only: {bc['refs']['arrival']:.0f} s": (bc["refs"]["arrival"], (0, (6, 2))),
                f"no obstruction: {res['concert'].T:.0f} s": (res["concert"].T, "-")}
        rb = os.path.join(TAB, "blocked_rebalanced.csv")
        if os.path.exists(rb):      # follow-up experiment
            d_ = pd.read_csv(rb)
            hold = d_[d_["variant"].str.contains("queue held")]
            if len(hold):
                refs[f"re-balanced to stage stairs + queue held: {hold['T_det'].iloc[0]:.0f} s"] = (
                    float(hold["T_det"].iloc[0]), (0, (2, 1, 1, 1)))
        made.append(P.fig_broadcast(bc["det"], bc["mc"], refs, FIG))

    # sensitivity
    if torn is not None:
        made.append(P.fig_tornado(torn, FIG, Q13))
    if pmc is not None:
        made.append(P.fig_param_mc(pmc, Q13, FIG, {n: res[n].T for n in Q13}))
    nominal = {"concert": dict(beta_dist=BEHAVIOURS["concert_self"].beta_dist,
                               beta_habit=BEHAVIOURS["concert_self"].beta_habit,
                               beta_queue=BEHAVIOURS["concert_self"].beta_queue),
               "lecture": dict(beta_dist=BEHAVIOURS["lecture"].beta_dist, beta_habit=BEHAVIOURS["lecture"].beta_habit,
                               beta_queue=BEHAVIOURS["lecture"].beta_queue)}
    if sw:
        made.append(P.fig_beta(sw["beta"], FIG, nominal))
        made.append(P.fig_compliance(sw["compliance"], FIG))
        made.append(P.fig_barriers(sw["barrier"], b0, analysis.barrier_layouts(), FIG))
        made.append(P.fig_blocked_sweep(sw["blocked"], FIG))
        made.append(P.fig_method_checks(sw["checks"], FIG))

    # design and cost (Q5, managers, wheelchairs, service variability)
    if xt.get("design"):
        from bailey.cost import design_table
        d = xt["design"]
        made.append(P.fig_managers(d["compliance"], d["runs"], cfg, FIG))
        full = d["runs"].dropna(subset=["T_mc"]) if "T_mc" in d["runs"] else d["runs"]
        made.append(P.fig_pareto(design_table(full.to_dict("records"), cfg, obj="T_mc"), cfg, FIG))
        from bailey.cost import cost_sensitivity
        made.append(P.fig_cost_sensitivity(cost_sensitivity(full.to_dict("records"), cfg, obj="T_mc"), FIG))
    if xt.get("disp"):
        made.append(P.fig_dispersion(xt["disp"], DISP_SCEN, KAPPAS, {n: res[n].T for n in DISP_SCEN}, FIG))
    if xt.get("wheel") is not None:
        made.append(P.fig_wheelchair(xt["wheel"], FIG))
    if xt.get("wheel_mc"):
        made.append(P.fig_wheelchair_mc(xt["wheel_mc"], FIG))
    if xt.get("rowcap"):
        made.append(P.fig_rowcap(xt["rowcap"], FIG))
    if xt.get("valid"):
        made.append(P.fig_validation(xt["valid"], FIG))
        made.append(P.fig_capacity_bracket(xt["valid"], Q13 + ["barriers", "led"], FIG))
    return made


def _git_version():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                           timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=HERE, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return h + (" (+uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def key_results() -> str:
    'key numbers for summary.md and the log.'
    lines = []
    f = os.path.join(TAB, "scenario_summary.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        lines.append("| scenario | T det | MC mean T | MC T95 | P(T≤180) MC | K | T_ground det | MC mean T_ground |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in d.itertuples():
            mc_ = (f"{r.MC_T_mean:.1f} | {r.MC_T95:.1f} | {r.MC_P_le_180:.2f} | {int(r.MC_K)}"
                   if hasattr(r, "MC_T_mean") and pd.notna(r.MC_T_mean) else "– | – | – | –")
            tg_ = f"{r.MC_Tg_mean:.1f}" if hasattr(r, "MC_Tg_mean") and pd.notna(r.MC_Tg_mean) else "–"
            lines.append(f"| {r.scenario} | {r.T:.1f} | {mc_} | {r.T_ground:.1f} | {tg_} |")
    for name, title in (("mc_paired_differences.csv", "paired differences (design − baseline)"),
                        ("validation_capacity_bracket.csv", "capacity bracket (deterministic T)"),
                        ("wheelchair.csv", "wheelchair users"),
                        ("wheelchair_mc.csv", "wheelchair users, Monte Carlo"),
                        ("rowcap_deterministic.csv", "seat-row capacity check (deterministic)"),
                        ("rowcap_mc_paired.csv", "seat-row capacity check (paired MC)"),
                        ("q5_cost_sensitivity_oat.csv", "Q5 cost sensitivity, one parameter at a time"),
                        ("q5_led_lecture_cost_per_second.csv", "LED in lectures: $ per second saved")):
        f = os.path.join(TAB, name)
        if os.path.exists(f):
            lines.append(f"\n**{title}** (`{name}`)\n")
            lines.append("```\n" + pd.read_csv(f).round(2).head(40).to_string(index=False) + "\n```")
    return "\n".join(lines)


def record_run(run_dir, run_id, cmd, a, seconds, made):
    'See docs/IMPLEMENTATION.md for implementation details.'
    meta = dict(run_id=run_id, command=cmd, seconds=round(seconds, 1), model_code_hash=code_hash(),
                git=_git_version(), python=platform.python_version(), numpy=np.__version__,
                args=vars(a), stage_times=STAGE_TIMES, n_figures=len(made),
                config=dict(phys=dataclasses.asdict(cfg.phys), geo=dataclasses.asdict(cfg.geo),
                            costs=dataclasses.asdict(cfg.costs)))
    from bailey import config as C
    meta["config"].update(MANAGERS=C.MANAGERS, SERVICE_DISPERSION=C.SERVICE_DISPERSION, COMPLIANCE=C.COMPLIANCE,
                          WHEELCHAIR_SEATS=C.WHEELCHAIR_SEATS, WHEELCHAIR_ASSIST=C.WHEELCHAIR_ASSIST,
                          PREP_WINDOWS={k: list(v) for k, v in C.PREP_WINDOWS.items()},
                          PREP_LOGNORMAL=C.PREP_LOGNORMAL, REACTION_SPECS=C.REACTION_SPECS,
                          BEHAVIOURS={k: dataclasses.asdict(v) for k, v in C.BEHAVIOURS.items()})
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    if os.path.isdir(TAB):
        shutil.copytree(TAB, os.path.join(run_dir, "tables"), dirs_exist_ok=True)
    #
    code_dir = os.path.join(run_dir, "code")
    shutil.copytree(os.path.join(HERE, "bailey"), os.path.join(code_dir, "bailey"), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.md"))
    shutil.copy2(os.path.join(HERE, "make_figures.py"), code_dir)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    for p in made:
        shutil.copy2(p, os.path.join(run_dir, "figures", os.path.basename(p)))
    kr = key_results()
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {run_id}\n\n- command: `{cmd}`\n- note: {a.note or '—'}\n- duration: {seconds / 60:.1f} min\n"
                 f"- model code hash: {meta['model_code_hash']}, git: {meta['git']}\n- stage times (s): "
                 f"{STAGE_TIMES}\n\n{kr}\n")
    new = not os.path.exists(EXPLOG)
    with open(EXPLOG, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# Experiment log\n\nEach make_figures.py run appends a record. Full outputs are in "
                     "`outputs/runs/<run id>/`. Manual checks belong in each record's Review section.\n"
                     "One entry per run (auto); full records in `outputs/runs/<run id>/`.\n")
        fh.write(f"\n---\n\n## {run_id}\n\n- command: `{cmd}`\n- note: {a.note or '—'}\n"
                 f"- duration: {seconds / 60:.1f} min · model code {meta['model_code_hash']} · git {meta['git']}\n"
                 f"- stage times (s): {STAGE_TIMES}\n- outputs: `outputs/runs/{run_id}/`\n\n{kr}\n")
    print(f"[run] recorded in {run_dir} and {EXPLOG}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=1000, help="MC replicates for the main scenarios")
    ap.add_argument("--K-var", type=int, default=300, help="MC replicates for randomness variants")
    ap.add_argument("--K-bc", type=int, default=200, help="MC replicates per broadcast delay")
    ap.add_argument("--K-param", type=int, default=400, help="parameter-MC replicates per scenario")
    ap.add_argument("--K-design", type=int, default=100, help="MC replicates per Q5 design")
    ap.add_argument("--K-disp", type=int, default=200, help="MC replicates per dispersion κ")
    ap.add_argument("--K-valid", type=int, default=200, help="MC replicates per capacity variant (validation)")
    ap.add_argument("--K-wheel", type=int, default=200, help="MC replicates per wheelchair design")
    ap.add_argument("--K-row", type=int, default=200, help="MC replicates per scenario for the seat-row check")
    ap.add_argument("--run-name", default="run", help="name of this run (folder outputs/runs/<time>_<name>)")
    ap.add_argument("--note", default="", help="why this run was made (goes into EXPERIMENT_LOG.md)")
    ap.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--only", default="all", help="'plot' = redraw from cache only")
    ap.add_argument("--redo", default="", help="comma list of stages to recompute (det,markov,sweeps,mc,var,bc,tornado,pmc,design,disp,wheel,"
                         "valid,wheel_mc,rowcap,all)")
    a = ap.parse_args()
    redo = set(filter(None, a.redo.split(",")))
    w = a.workers
    t0 = time.time()
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + a.run_name
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(run_dir, "run.log"))
    cmd = "python " + " ".join(sys.argv)
    with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as fh:
        fh.write(cmd + "\n")
    print(f"[run] {run_id}\n[run] {cmd}\n[run] model code hash {code_hash()}", flush=True)
    if a.only == "plot":
        get = load_cached
        det, mk, sw, mc, var, bc, torn, pmc = (get(n) for n in ("det", "markov", "sweeps", "mc", "var", "bc",
                                                               "tornado", "pmc"))
        xt = dict(design=get("design"), disp=get("disp"), wheel=get("wheel"), valid=get("valid"),
                  wheel_mc=get("wheel_mc"), rowcap=get("rowcap"))
    else:
        det = cached("det", stage_det, redo)
        mk = cached("markov", stage_markov, redo)
        sw = cached("sweeps", lambda: stage_sweeps(w), redo)
        mc = cached("mc", lambda: stage_mc(a.K, w), redo, (a.K,))
        var = cached("var", lambda: stage_mc_variants(a.K_var, w), redo, (a.K_var,))
        bc = cached("bc", lambda: stage_broadcast(a.K_bc, w), redo, (a.K_bc,))
        torn = cached("tornado", lambda: stage_tornado(w), redo)
        pmc = cached("pmc", lambda: stage_pmc(a.K_param, w), redo, (a.K_param,))
        xt = dict(design=cached("design", lambda: stage_design(a.K_design, w), redo, (a.K_design,)),
                  disp=cached("disp", lambda: stage_dispersion(a.K_disp, w), redo, (a.K_disp,)),
                  wheel=cached("wheel", stage_wheelchair, redo),
                  valid=cached("valid", lambda: stage_validation(a.K_valid, w), redo, (a.K_valid,)),
                  wheel_mc=cached("wheel_mc", lambda: stage_wheel_mc(a.K_wheel, w), redo, (a.K_wheel,)),
                  rowcap=cached("rowcap", lambda: stage_rowcap(a.K_row, w), redo, (a.K_row,)))
    if det and mk:
        write_tables(det, mk, mc, var, bc, torn, pmc, xt)
    made = plot_all(det, mk, sw, mc, var, bc, torn, pmc, xt)
    print(f"\n{len(made)} figures in {FIG}  ({time.time() - t0:.0f} s)")
    for p in made:
        print("  ", os.path.basename(p))
    record_run(run_dir, run_id, cmd, a, time.time() - t0, made)


if __name__ == "__main__":
    main()


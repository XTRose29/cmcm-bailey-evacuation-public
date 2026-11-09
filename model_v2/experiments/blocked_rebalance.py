"""Follow-up: with the main exit blocked, how much does a deliberately
                                  re-balanced instruction plan (system optimum without the main exit) help?

why
  In the formal run the side stairs are the binding bottleneck when the main exit is blocked while the stage
  stairs empty at ~150 s; a later broadcast helped only because more people happened to use the stage stairs.

method
  Managers know at t = 0 and assign exits per seat zone with the LP + simulation-balancing method (same as the
  LED plan, but delivered only by the 6 managers, compliance from the coverage model). Baselines: nearest-exit
  instructions (known at t = 0) and the default discovery + 20 s broadcast. Deterministic + Monte Carlo (CRN).

run:   python experiments/blocked_rebalance.py --K 200
one entry appended）"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from bailey.building import build_building  # noqa: E402
from bailey.config import BEHAVIOURS, DEFAULT as cfg, MANAGERS, prep_min  # noqa: E402
from bailey.flow import PopClass, simulate  # noqa: E402
from bailey.montecarlo import make_job, reaction_sampler, run_batch  # noqa: E402
from bailey.plugins.randomness import mean_field_release  # noqa: E402
from bailey.scenarios import (SCENARIOS, _cached_lp, _cached_refine, gate_finish,  # noqa: E402
                              _manager_setup, run)


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def rebalanced_job(seed: int, beta_guide: float = None, beta_queue: float = None):
    """managers assign exits by the no-main-exit system optimum.
    instruction weight β4: None = model default 2.5; 10 = followed almost surely."""
    sc = SCENARIOS["blocked"].with_(name="blocked_rebalanced", discovery="known")
    b = build_building(cfg, blocked=sc.blocked)
    v = cfg.phys.v_emergency
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    removed = set(sc.blocked)
    local_info = _manager_setup(sc, cfg, b)
    c_mgr = {z: local_info["manager_p_follow"] for z in seats}
    from dataclasses import replace as _replace
    led_beh, self_beh = BEHAVIOURS["concert_managed"], BEHAVIOURS["concert_self"]
    if beta_guide is not None:
        led_beh = _replace(led_beh, name=f"concert_managed_b4_{beta_guide:g}", beta_guide=beta_guide)
    if beta_queue is not None:   # managers make people hold the queue
        led_beh = _replace(led_beh, name=led_beh.name + f"_b3_{beta_queue:g}", beta_queue=beta_queue)
    lp = _cached_lp(b, cfg, led_beh, v, seats, prep_min(sc.prep_key), removed)
    y, hist = _cached_refine(sc, cfg, b, lp, led_beh, self_beh, seats, v, c_design=c_mgr)
    classes = []
    for ei, e in enumerate(lp["exits"]):
        cnt = {z: c_mgr[z] * y[zi, ei] for zi, z in enumerate(lp["zones"]) if y[zi, ei] > 1e-9}
        if sum(cnt.values()) > 1e-9:
            classes.append(PopClass(f"mgr→{e}", self_beh, cnt, {i: e for i in b.transient},
                                    {i: led_beh for i in local_info["manager_covered_nodes"]}))
    classes.append(PopClass("self", self_beh, {z: n * (1 - c_mgr[z]) for z, n in seats.items()}))
    proto = reaction_sampler(sc, cfg, seats)
    job = dict(b=b, cfg=cfg, classes=classes, v=v, discovery=sc.discovery, delay=sc.broadcast_delay,
               proto=proto, seed=seed, name=sc.name, dist="uniform", service="poisson", keep_curves=False,
               n=int(sum(seats.values())), motion=True, setup=local_info)
    det = simulate(b, cfg, classes, v, randomness=mean_field_release(sc.prep_key),
                   discovery=sc.discovery)
    plan = pd.DataFrame(y, index=lp["zones"], columns=lp["exits"]).round(1)
    print(f"[plan] LP lower bound with the main exit removed: {lp['T_lp']:.1f} s (beta_guide={led_beh.beta_guide})")
    return job, det, plan, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_blocked-rebalance"
    rd = os.path.join(HERE, "outputs", "runs", run_id)
    os.makedirs(rd, exist_ok=True)
    sys.stdout = Tee(os.path.join(rd, "run.log"))
    cmd = "python " + " ".join(sys.argv)
    print(f"[run] {run_id}\n[run] {cmd}", flush=True)

    job_r, det_r, plan, hist = rebalanced_job(a.seed)
    job_s, det_s, plan_s, hist_s = rebalanced_job(a.seed, beta_guide=10.0)
    print("[plan] strict following (β4 = 10), refinement history:", [round(x, 1) for x in hist_s])
    job_h, det_h, plan_h, hist_h = rebalanced_job(a.seed, beta_guide=10.0, beta_queue=0.0)
    print("[plan] strict + hold the queue (β4 = 10, β3 = 0), refinement history:", [round(x, 1) for x in hist_h])
    print("[plan] people per seat zone → exit (strict + hold):\n" + plan_h.to_string())
    print("[plan] people per seat zone → exit (system optimum without the main exit):\n" + plan.to_string())
    print("[plan] refinement history of T (s):", [round(x, 1) for x in hist])
    variants = {
        "found at door + 20 s broadcast (default)": SCENARIOS["blocked"],
        "known at t = 0, nearest-exit instructions": SCENARIOS["blocked"].with_(name="bk", discovery="known"),
    }
    rows = []
    for lab, sc in variants.items():
        r, _ = run(sc, cfg)
        rows.append(dict(variant=lab, T_det=r.T, exits=json.dumps({k: round(x) for k, x in r.exit_groups().items()}),
                         finish=json.dumps({k: round(x, 1) for k, x in gate_finish(r).items()})))
    rows.append(dict(variant="known at t = 0, re-balanced instructions (system optimum)", T_det=det_r.T,
                     exits=json.dumps({k: round(x) for k, x in det_r.exit_groups().items()}),
                     finish=json.dumps({k: round(x, 1) for k, x in gate_finish(det_r).items()})))
    rows.append(dict(variant="known at t = 0, re-balanced, instructions followed strictly (β4 = 10)",
                     T_det=det_s.T, exits=json.dumps({k: round(x) for k, x in det_s.exit_groups().items()}),
                     finish=json.dumps({k: round(x, 1) for k, x in gate_finish(det_s).items()})))
    rows.append(dict(variant="known at t = 0, re-balanced, followed strictly AND queue held (β4 = 10, β3 = 0)",
                     T_det=det_h.T, exits=json.dumps({k: round(x) for k, x in det_h.exit_groups().items()}),
                     finish=json.dumps({k: round(x, 1) for k, x in gate_finish(det_h).items()})))
    jobs = {lab: make_job(sc, cfg, seed=a.seed, keep_curves=False) for lab, sc in variants.items()}
    jobs[rows[-3]["variant"]] = job_r
    jobs[rows[-2]["variant"]] = job_s
    jobs[rows[-1]["variant"]] = job_h
    mc = run_batch(jobs, a.K, a.workers)
    base = mc["known at t = 0, nearest-exit instructions"]["T"]
    for r in rows:
        m = mc[r["variant"]]
        d = m["T"] - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        r.update(K=m["K"], T_mc_mean=m["T_mean"], T_mc05=m["T_05"], T_mc95=m["T_95"], P_le_180=m["P_le_180"],
                 dT_vs_nearest=float(d.mean()), dT_ci_lo=float(d.mean() - 1.96 * se),
                 dT_ci_hi=float(d.mean() + 1.96 * se))
    df = pd.DataFrame(rows)
    print(df.drop(columns=["finish"]).round(2).to_string(index=False))
    tab = os.path.join(HERE, "outputs", "paper", "tables")
    os.makedirs(tab, exist_ok=True)
    df.to_csv(os.path.join(tab, "blocked_rebalanced.csv"), index=False)
    df.to_csv(os.path.join(rd, "results.csv"), index=False)
    plan.to_csv(os.path.join(rd, "rebalanced_plan.csv"))
    shutil.copy2(os.path.abspath(__file__), rd)
    secs = time.time() - t0
    json.dump(dict(run_id=run_id, command=cmd, seconds=round(secs, 1), K=a.K, seed=a.seed), open(
        os.path.join(rd, "meta.json"), "w", encoding="utf-8"), indent=2)
    with open(os.path.join(HERE, "outputs", "EXPERIMENT_LOG.md"), "a", encoding="utf-8") as fh:
        fh.write(f"\n---\n\n## {run_id}\n\n- command: `{cmd}`\n- note: follow-up to the broadcast-delay finding: "
                 f"deliberately re-balanced manager instructions when the main exit is blocked\n"
                 f"- duration: {secs / 60:.1f} min\n- outputs: `outputs/runs/{run_id}/`\n\n```\n"
                 f"{df.drop(columns=['finish', 'exits']).round(2).to_string(index=False)}\n```\n")
    print(f"[run] recorded in {rd}", flush=True)


if __name__ == "__main__":
    main()


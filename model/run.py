"""Command-line interface for individual scenarios and diagnostic plots.
Use run_overnight.py for the formal experiment population and reporting rules.
Example: python model/run.py concert --out outputs/concert"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from bailey import analysis, viz
from bailey.config import DEFAULT, PARAM_SOURCES
from bailey.cost import annual_costs
from bailey.export import export_json, export_timeseries_csv
from bailey.montecarlo import run_mc
from bailey.plugins.queueing import FlowQueueModel
from bailey.scenarios import SCENARIOS, gate_finish, run, summary_row

SCENARIO_CMDS = list(SCENARIOS)


def _fmt(x):
    return "inf" if not np.isfinite(x) else f"{x:.1f}"


def do_scenario(name, cfg, out, quiet=False):
    res, info = run(SCENARIOS[name], cfg)
    figdir = os.path.join(out, "figures", name)
    paths = [viz.plot_occupancy(res, figdir, name), viz.plot_bottlenecks(res, figdir, name),
             viz.plot_snapshots(res, figdir, name)]
    js = export_json(res, os.path.join(out, "data", f"{name}.json"))
    csv = export_timeseries_csv(res, os.path.join(out, "data", f"{name}_timeseries.csv"))
    if not quiet:
        row = summary_row(res)
        print(f"\n== {name} ({SCENARIOS[name].question}) {SCENARIOS[name].zh}")
        print(f"   T (last person leaves building) = {_fmt(res.T)} s   "
              f"T_ground (reaches street) = {_fmt(res.T_ground)} s")
        print(f"   T90 = {_fmt(row['T90'])} s   mean egress time = {res.mean_time:.1f} s   "
              f"meets 180 s: {row['meets_180']}")
        print("   exits:", {k: round(v) for k, v in res.exit_groups().items()})
        print("   per-exit finish (building egress):", {k: _fmt(v) for k, v in gate_finish(res).items()})
        if info.get("T_lp") is not None:
            print(f"   LP lower bound used for LED design = {info['T_lp']:.1f} s")
        if info.get("compliance") is not None:
            print(f"   compliance = {info['compliance']:.2f}  plan = {info.get('plan_kind')}")
        top = FlowQueueModel().analyze(res)["table"].head(3)
        print("   worst queues (Section 4):", "; ".join(
            f"{r.junction}: max wait {r.W_max:.0f} s, peak queue {r.peak_queue:.0f}" for r in top.itertuples()))
        print("   figures:", *paths, sep="\n     ")
        print("   data:", js, csv, sep="\n     ")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd")
    ap.add_argument("scenario", nargs="?")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--walk", choices=["erlang", "geometric"], default=None)
    ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args(argv)

    cfg = DEFAULT
    if a.dt or a.walk:
        cfg = cfg.with_(phys=replace(cfg.phys, dt=a.dt or cfg.phys.dt, walk_model=a.walk or cfg.phys.walk_model))
    out = a.out
    fig = os.path.join(out, "figures")
    tab = os.path.join(out, "tables")
    os.makedirs(tab, exist_ok=True)

    cmds = [a.cmd]
    if a.cmd == "all":
        cmds = ["params", "graph", "markov", *SCENARIO_CMDS, "compare", "sweep-blocked",
                "sweep-compliance", "sweep-beta", "checks", "sweep-barriers", "cost"]

    for cmd in cmds:
        if cmd == "params":
            print(pd.DataFrame(PARAM_SOURCES, columns=["parameter", "default", "source"]).to_string(index=False))
            print(f"\nk_max (derived) = {cfg.phys.k_max:.2f} persons/m²,  dt = {cfg.phys.dt} s,  "
                  f"walk model = {cfg.phys.walk_model}")
        elif cmd == "graph":
            from bailey.building import build_building
            print(viz.plot_building(build_building(cfg), fig))
        elif cmd == "markov":
            rep = analysis.markov_report(cfg, a.scenario or "concert")
            mix = rep["mix"]
            print(f"\n== Markov closed form ({rep['scenario']} population, no congestion)")
            print("   mean free-walk time per person:", f"{mix['mean_time']:.1f} s")
            print("   t per seat zone (s):", dict(zip(mix["seats"], np.round(mix["t_seat"], 1))))
            print("   people per exit:", dict(zip(rep["b"].exits, np.round(mix["exit_people"], 0))))
            print(f"   max-flow lower bound = {rep['lb']['LB']:.1f} s   LP lower bound = {rep['lp']['T_lp']:.1f} s")
            pd.DataFrame({"seat": mix["seats"], "t_free_s": mix["t_seat"],
                          **{e: mix["B_seat"][:, m] for m, e in enumerate(rep["b"].exits)}}
                         ).to_csv(os.path.join(tab, "markov_by_seat.csv"), index=False)
            print(*viz.plot_markov(rep, fig), sep="\n")
        elif cmd in SCENARIO_CMDS:
            do_scenario(cmd, cfg, out)
        elif cmd == "compare":
            results = {n: run(SCENARIOS[n], cfg)[0] for n in SCENARIO_CMDS}
            df = pd.DataFrame([summary_row(r) for r in results.values()])
            df.to_csv(os.path.join(tab, "scenario_summary.csv"), index=False)
            print(df.to_string(index=False))
            print(viz.plot_curves(results, fig))
            print(viz.plot_exit_usage(results, fig))
        elif cmd == "sweep-blocked":
            df = analysis.blocked_sweep(cfg)
            df.to_csv(os.path.join(tab, "blocked_sweep.csv"), index=False)
            print(df.to_string(index=False)); print(viz.plot_blocked_sweep(df, fig))
        elif cmd == "sweep-barriers":
            df = analysis.barrier_sweep(cfg)
            df.to_csv(os.path.join(tab, "barrier_sweep.csv"), index=False)
            print(df.to_string(index=False)); print(viz.plot_barrier_sweep(df, fig))
        elif cmd == "sweep-compliance":
            df = analysis.compliance_sweep(cfg)
            df.to_csv(os.path.join(tab, "compliance_sweep.csv"), index=False)
            print(df.pivot(index="compliance", columns="scenario", values="T").to_string())
            print(viz.plot_compliance(df, fig))
        elif cmd == "sweep-beta":
            df = analysis.beta_sweep(cfg)
            df.to_csv(os.path.join(tab, "beta_sweep.csv"), index=False)
            print(df.to_string(index=False)); print(viz.plot_beta(df, fig))
        elif cmd == "checks":
            df = analysis.method_checks(cfg)
            df.to_csv(os.path.join(tab, "method_checks.csv"), index=False)
            print(df.to_string(index=False)); print(viz.plot_method_checks(df, fig))
        elif cmd == "mc":
            name = a.scenario or "concert"
            r = run_mc(SCENARIOS[name], cfg, K=a.K)
            print(f"{name}: T̄ = {r['T_mean']:.1f} s, T95 = {r['T_95']:.1f} s, P(T≤180) = {r['P_le_180']:.2f}")
        elif cmd == "cost":
            print(annual_costs(cfg))
        else:
            sys.exit(f"unknown command: {cmd}\n\n{__doc__}")


if __name__ == "__main__":
    main()


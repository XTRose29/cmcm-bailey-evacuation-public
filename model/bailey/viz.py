"""Diagnostic plots of model populations, flows, queues and scenario comparisons."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .flow import SimResult


# palette

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

SCEN_COLOR = {   # colour follows the scenario everywhere
    "concert": SLOTS[0], "led": SLOTS[1], "barriers": SLOTS[2], "combo": SLOTS[3],
    "lecture": SLOTS[4], "normal": SLOTS[5], "blocked": SLOTS[6], "lecture_led": SLOTS[7],
    "lecture_barriers": "#86b6ef", "wheelchair": MUTED,
}
SCEN_LABEL = {
    "normal": "Q1 normal dismissal", "concert": "Q2 concert emergency (managers)",
    "blocked": "Q2 + main exit blocked", "lecture": "Q3 lecture emergency",
    "barriers": "Q4a barriers", "led": "Q4b LED guidance", "combo": "Q4 barriers + LED",
    "lecture_led": "Q4 lecture + LED", "lecture_barriers": "Q4 lecture + barriers",
    "wheelchair": "wheelchair assistance",
}
EXIT_COLOR = {"EXIT-Main": SLOTS[0], "EXIT-Side": SLOTS[1], "EXIT-Stage": SLOTS[2]}
KIND_STYLE = {   # (colour, size, legend label)
    "seat": (SLOTS[0], 260, "seat zones"),
    "access": (SLOTS[1], 70, "doors / access points"),
    "door": (SLOTS[1], 70, "doors / access points"),
    "corridor": ("#b4b2a9", 120, "corridor / foyer"),
    "foyer": ("#b4b2a9", 260, "corridor / foyer"),
    "stair": (SLOTS[2], 110, "stairs"),
    "portico": (SLOTS[5], 110, "portico (outside)"),
    "exit": (INK, 120, "exits (absorbing)"),
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.axisbelow": True,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    })


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _ref_line(ax, x=180.0, label="180 s requirement", vertical=True):
    if vertical:
        ax.axvline(x, color=INK2, lw=1.0)
        y0, y1 = ax.get_ylim()
        ax.text(x, y0 + 0.97 * (y1 - y0), f" {label}", color=INK2, va="top", ha="left", fontsize=8,
                backgroundcolor=SURFACE)
    else:
        ax.axhline(x, color=INK2, lw=1.0)
        ax.text(ax.get_xlim()[0], x, f"{label} ", color=INK2, va="bottom", ha="left", fontsize=8)


# building graph

def plot_building(b, outdir: str, name="fig01_building_graph.png", values: Optional[Dict] = None,
                  title="Bailey Hall as a state graph (schematic, not to scale)") -> str:
    _style()
    fig, ax = plt.subplots(figsize=(9, 7.6))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, k in b.edges:
        (x0, y0), (x1, y1) = b.cell[i].xy, b.cell[k].xy
        rev = (i, k) in b.reverse
        if rev and (k, i) in b.edges and (k, i) not in b.reverse:
            continue          # plain reverse of a forward link
        if rev and (k, i) in b.reverse and i > k:
            continue          # draw two-way fallback links once
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#c3c2b7" if rev else MUTED,
                                    lw=0.6 if rev else 0.9, shrinkA=6, shrinkB=6,
                                    linestyle=":" if rev else "-", mutation_scale=7))
    seen = set()
    for c in b.cells:
        col, size, lab = KIND_STYLE[c.kind]
        ax.scatter(*c.xy, s=size, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.5,
                   label=lab if lab not in seen else None,
                   marker="s" if c.kind == "exit" else "o")
        seen.add(lab)
        txt = c.id.replace("EXIT-", "") if c.kind == "exit" else c.id
        ax.text(c.xy[0], c.xy[1] + 1.25, txt, ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=4)
    ax.text(0, 6.5, "STAGE (north)", ha="center", color=MUTED, fontsize=9)
    ax.text(0, -40, "street (south)", ha="center", color=MUTED, fontsize=9)
    ax.text(10.5, -30.5, "Balcony zones B1–B3 (2nd floor) on the outer ring.\n"
            "Balcony stairs BS1/BS2 land in the foyer.\n"
            "Dotted: portico detour PORS, used only\nif a side stair is blocked.\n"
            "Turn-back links (e.g. FOY → F) exist but are not drawn.",
            fontsize=7, color=INK2, va="top", wrap=True)
    ax.set_xlim(-34, 34); ax.set_ylim(-42, 10)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, ncol=2, bbox_to_anchor=(0.0, -0.06))
    ax.set_title(title, loc="left")
    return _save(fig, outdir, name)


# closed-form Markov chain

def plot_markov(rep: Dict, outdir: str) -> List[str]:
    _style()
    b, mix = rep["b"], rep["mix"]
    groups = b.exit_groups
    Bg = np.zeros((len(mix["seats"]), len(groups)))
    for m, e in enumerate(b.exits):
        Bg[:, groups.index(b.cell[e].exit_group)] += mix["B_seat"][:, m]
    paths = []

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a1.grid(False)
    im = a1.imshow(Bg, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    a1.set_xticks(range(len(groups)), [g.replace("EXIT-", "") for g in groups])
    a1.set_yticks(range(len(mix["seats"])), mix["seats"])
    for (r, c), val in np.ndenumerate(Bg):
        a1.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if val > 0.55 else INK)
    a1.set_title("B = N·R: probability of leaving by each exit", loc="left")
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    y = np.arange(len(mix["seats"]))
    a2.barh(y, mix["t_seat"], color=SLOTS[0], height=0.6)
    a2.set_yticks(y, mix["seats"]); a2.invert_yaxis()
    for yy, val in zip(y, mix["t_seat"]):
        a2.text(val, yy, f" {val:.0f}", va="center", fontsize=8, color=INK2)
    a2.set_xlabel("expected time to exit without congestion, t = N·1 (s)")
    a2.set_title("Mean free-walk time from each seat zone", loc="left")
    fig.suptitle(f"Absorbing Markov chain, no congestion ({rep['scenario']} population)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    paths.append(_save(fig, outdir, "fig02_markov_exit_prob_and_time.png"))


    # Service time needed = expected passers / μ → who becomes the bottleneck
    fig, ax = plt.subplots(figsize=(8, 5.2))
    need = sorted(((rep["passes"][j] / rep["mu"][j], j, rep["passes"][j]) for j in rep["passes"]),
                  reverse=True)[:15]
    yy = np.arange(len(need))
    ax.barh(yy, [n[0] for n in need], color=SLOTS[1], height=0.6)
    ax.set_yticks(yy, [n[1] for n in need]); ax.invert_yaxis()
    for k, (val, j, p) in enumerate(need):
        ax.text(val, k, f" {p:.0f} people", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("expected people through junction ÷ capacity μ (s)")
    ax.set_title("Where congestion will form if nobody adapts (top 15 junctions)", loc="left")
    lb = rep["lb"]
    ax.text(0.99, 0.02, f"routing-independent lower bounds:\n"
            f"max-flow bound {lb['LB']:.0f} s,  LP bound {rep['lp']['T_lp']:.0f} s",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    fig.tight_layout()
    paths.append(_save(fig, outdir, "fig03_markov_bottleneck_demand.png"))
    return paths


# one scenario

def _cell_order(b):
    order = [c.id for c in b.cells if c.kind == "seat"]
    for k in ("access", "door", "corridor", "foyer", "stair", "portico"):
        order += [c.id for c in b.cells if c.kind == k and c.id not in order]
    return order


def plot_occupancy(res: SimResult, outdir: str, tag: str) -> str:
    _style()
    b = res.b
    order = [c for c in _cell_order(b) if c in b.transient]
    M = np.array([res.occ[:, b.idx[c]] for c in order])
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.grid(False)
    im = ax.imshow(M, aspect="auto", cmap=SEQ, interpolation="nearest",
                   extent=[res.t[0], res.t[-1], len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)), order, fontsize=6.5)
    ax.set_xlabel("time (s)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("people in cell")
    if res.T < np.inf:
        ax.axvline(res.T, color=INK2, lw=1.0)
        ax.text(res.T, -0.5, f" T = {res.T:.0f} s", color=INK2, va="bottom", fontsize=8)
    ax.set_title(f"Who is where over time — {SCEN_LABEL.get(tag, tag)}", loc="left")
    return _save(fig, outdir, f"{tag}_occupancy_heatmap.png")


def plot_bottlenecks(res: SimResult, outdir: str, tag: str, top: int = 6) -> str:
    _style()
    b = res.b
    junc = b.junctions() + [e for e in b.exits]
    peaks = sorted(((res.queue_in[:, b.idx[j]].max(), j) for j in junc), reverse=True)

    # The symmetric building yields identical W/E curves: merge them under one label
    groups = []
    for pk, j in peaks:
        y = res.queue_in[:, b.idx[j]]
        for g in groups:
            if np.max(np.abs(g["y"] - y)) < 0.5:
                g["names"].append(j)
                break
        else:
            groups.append(dict(y=y, pk=pk, names=[j]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    outside_q = {e for e in b.exits if any(b.cell[i].outside for i, k in b.edges if k == e)}
    for k, g in enumerate(groups[:top]):
        name = " = ".join(g["names"])
        where = " — on the portico, already outside" if g["names"][0] in outside_q else ""
        each = " each" if len(g["names"]) > 1 else ""
        ax.plot(res.t, g["y"], color=SLOTS[k], label=f"{name} (peak {g['pk']:.0f}{each}){where}",
                lw=2.0 if k < 2 else 1.4)
        if k == 0:   # direct-label only the largest
            m = int(np.argmax(g["y"]))
            ax.text(res.t[m], g["y"][m], f" {name}", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("time (s)"); ax.set_ylabel("people waiting to enter, L_j(t)")
    ax.set_title(f"Largest queues — {SCEN_LABEL.get(tag, tag)}", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return _save(fig, outdir, f"{tag}_bottlenecks.png")


def plot_snapshots(res: SimResult, outdir: str, tag: str,
                   times: Sequence[float] = (10, 30, 50, 75, 100, 130)) -> str:
    _style()
    b = res.b
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    vmax = 1.0
    for ax, tt in zip(axes.flat, times):
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        k = min(int(np.searchsorted(res.t, tt)), len(res.t) - 1)
        for i, kk in b.edges:
            if (i, kk) in b.reverse:
                continue
            (x0, y0), (x1, y1) = b.cell[i].xy, b.cell[kk].xy
            ax.plot([x0, x1], [y0, y1], color=GRID, lw=0.8, zorder=1)
        xs, ys, cs, ss = [], [], [], []
        for c in b.transient:
            n = res.occ[k, b.idx[c]]
            nm = b.cell[c].n_max
            frac = n / nm if np.isfinite(nm) and nm > 0 else n / 300.0
            xs.append(b.cell[c].xy[0]); ys.append(b.cell[c].xy[1])
            cs.append(min(frac, vmax)); ss.append(20 + 2.2 * n)
        sc = ax.scatter(xs, ys, c=cs, s=ss, cmap=SEQ, vmin=0, vmax=vmax, edgecolor=MUTED,
                        linewidth=0.4, zorder=2)
        out = max(0.0, res.meta["n"] - res.remaining_inside[k])
        ax.set_title(f"t = {res.t[k]:.0f} s   left building: {out:.0f}/{res.meta['n']:.0f}",
                     loc="left", fontsize=9)
        ax.set_xlim(-32, 32); ax.set_ylim(-39, 9); ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("occupancy ÷ cell capacity (dot area ∝ people)")
    fig.suptitle(f"Snapshots — {SCEN_LABEL.get(tag, tag)}", x=0.01, ha="left", fontsize=11)
    return _save(fig, outdir, f"{tag}_snapshots.png")


# scenario comparison

def plot_curves(results: Dict[str, SimResult], outdir: str) -> str:
    _style()
    panels = [("Q1–Q3 (no interventions)", ["normal", "concert", "blocked", "lecture"]),
              ("Q4 interventions", ["concert", "barriers", "led", "combo"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, names) in zip(axes, panels):
        for nm in names:
            if nm not in results:
                continue
            r = results[nm]
            pct = 100 * (1 - r.remaining_inside / r.meta["n"])
            ax.plot(r.t, pct, color=SCEN_COLOR[nm], label=f"{SCEN_LABEL[nm]} — T = {r.T:.0f} s")
        ax.set_ylim(0, 102)
        ax.set_xlim(left=0)
        ax.set_xlabel("time since the start (s)")
        _ref_line(ax)
        ax.set_title(title, loc="left")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("% of audience out of the building")
    return _save(fig, outdir, "fig04_evacuation_curves.png")


def plot_exit_usage(results: Dict[str, SimResult], outdir: str) -> str:
    _style()
    names = [n for n in SCEN_LABEL if n in results]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.5))
    y = np.arange(len(names))
    left = np.zeros(len(names))
    groups = ["EXIT-Main", "EXIT-Side", "EXIT-Stage"]
    for g in groups:
        vals = np.array([results[n].exit_groups().get(g, 0.0) for n in names])
        ax.barh(y, vals, left=left, color=EXIT_COLOR[g], height=0.6, edgecolor=SURFACE,
                linewidth=1.5, label=g.replace("EXIT-", ""))
        left += vals
    ax.set_yticks(y, [SCEN_LABEL[n] for n in names]); ax.invert_yaxis()
    ax.set_xlabel("people leaving by each exit group")
    ax.set_title("Exit usage by scenario", loc="left")
    ax.legend(ncol=3, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.grid(axis="y", visible=False)
    return _save(fig, outdir, "fig05_exit_usage.png")


# sweeps

def plot_blocked_sweep(df: pd.DataFrame, outdir: str) -> str:
    _style()
    base = float(df.loc[df.target == "none", "T"].iloc[0])
    d = df[df.target != "none"]
    targets = list(dict.fromkeys(d.target))
    modes = ["known", "broadcast", "arrival"]
    labels = {"known": "known from t = 0", "broadcast": "found on arrival + broadcast at 20 s",
              "arrival": "found on arrival only"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    x = np.arange(len(targets))
    for k, m in enumerate(modes):
        vals = [float(d[(d.target == t) & (d.discovery == m)]["T"].iloc[0]) for t in targets]
        ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=SLOTS[k], label=labels[m])
    ax.axhline(base, color=INK2, lw=1)
    ax.text(len(targets) - 0.5, base, f"no obstruction {base:.0f} s ", ha="right", va="bottom",
            fontsize=8, color=INK2)
    ax.axhline(180, color=SLOTS[7], lw=1)
    ax.text(-0.5, 180, " 180 s", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, targets)
    ax.set_ylabel("building egress time T (s)")
    ax.set_title("Q2: obstruction of one exit — where, and how people find out", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, outdir, "fig06_blocked_sweep.png")


def plot_barrier_sweep(df: pd.DataFrame, outdir: str) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, ev in zip(axes, ["concert", "lecture"]):
        d = df[df.event == ev]
        layouts = list(dict.fromkeys(d.layout))
        x = np.arange(len(layouts))
        for k, g in enumerate(["managers", "led"]):
            dd = d[d.guidance == g]
            if dd.empty:
                continue
            vals = [float(dd[dd.layout == L]["T"].iloc[0]) for L in layouts]
            lab = ("crowd managers" if ev == "concert" else "self-evacuation") if g == "managers" else "LED guidance"
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.35, color=SLOTS[k * 1], label=lab)
        ax.set_xticks(x, ["no barrier" if L == "none" else f"cut {L}\n(+ mirror)" for L in layouts], fontsize=8)
        ax.set_title(f"{ev}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("building egress time T (s)")
    fig.suptitle("Q4a: corridor barrier position", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, outdir, "fig07_barrier_sweep.png")


def plot_compliance(df: pd.DataFrame, outdir: str) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for nm in ["concert", "barriers", "led", "combo"]:
        d = df[df.scenario == nm]
        ax.plot(d.compliance * 100, d["T"], color=SCEN_COLOR[nm], marker="o", ms=4, label=SCEN_LABEL[nm])
    ax.set_xlabel("share of people following instructions (%)")
    ax.set_ylabel("building egress time T (s)")
    _ref_line(ax, 180, vertical=False)
    ax.set_title("Robustness: how T changes when fewer people comply", loc="left")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "fig08_compliance_robustness.png")


def plot_beta(df: pd.DataFrame, outdir: str) -> str:
    _style()
    pars = list(dict.fromkeys(df.parameter))
    names = {"beta_habit": "β2 habit (main-entrance bonus)", "beta_dist": "β1 distance (per s walked)",
             "beta_queue": "β3 queue avoidance (per s waited)"}
    fig, axes = plt.subplots(1, len(pars), figsize=(4 * len(pars), 3.8), sharey=True)
    for ax, p in zip(axes, pars):
        for k, sc in enumerate(["concert", "lecture"]):
            d = df[(df.parameter == p) & (df.scenario == sc)]
            ax.plot(d.value, d["T"], marker="o", ms=4, color=SCEN_COLOR[sc], label=SCEN_LABEL[sc])
        ax.set_xlabel(names.get(p, p))
        ax.axhline(180, color=INK2, lw=1)
    axes[0].set_ylabel("T (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensitivity to assumed behaviour parameters (all classes changed together)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, outdir, "fig09_beta_sensitivity.png")


def plot_method_checks(df: pd.DataFrame, outdir: str) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sc in zip(axes, ["concert", "lecture"]):
        d = df[df.scenario == sc]
        dts = sorted(d.dt.unique(), reverse=True)
        x = np.arange(len(dts))
        for k, wm in enumerate(["erlang", "geometric"]):
            vals = [float(d[(d.dt == t) & (d.walk_model == wm)]["T"].iloc[0]) for t in dts]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.35, color=SLOTS[k], label=f"{wm} walking")
            for xx, vv in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(xx, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"Δt = {t:g} s" for t in dts])
        ax.set_title(sc, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("T (s)")
    axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(1.05, -0.32), ncol=2)
    fig.suptitle("Numerical checks: time step and walking-time distribution", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return _save(fig, outdir, "fig10_method_checks.png")

"""grouped by paper section:
model set-up            (floor plan with our cells)
absorbing chain    (free-flow times, exit probabilities, bottleneck screening, check)
flow model        (curves, snapshots, space–time, exit balance, exit flow)
queueing                   (N-curves, ρ(t), bottleneck ranking, M/M/1, wait map)
Poisson + MC       (reaction times, service noise, T distributions, CRN, ...)
sensitivity        (tornado, parameter MC, β, compliance, barriers, blockage, numerics)

functions only plot; make_figures.py prepares data."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import floorplan as fp
from .plugins.queueing import mm1_wq


# reference palette)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
SEQ_O = LinearSegmentedColormap.from_list("seq_o", ["#fdf6f2", "#fbd9c8", "#f6b291", "#f08b5f", "#eb6834",
                                                    "#c4501f", "#933a14", "#62260b"])
DIV = LinearSegmentedColormap.from_list("div", ["#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f3a3a2",
                                                "#e34948", "#8f1f1e"])

SC = {   # scenario → (label, colour), fixed everywhere
    "normal": ("Q1 normal dismissal", GREEN),
    "concert": ("Q2 concert emergency", BLUE),
    "blocked": ("Q2 main exit blocked", RED),
    "lecture": ("Q3 lecture emergency", VIOLET),
    "barriers": ("Q4 barriers", AQUA),
    "led": ("Q4 LED strips", ORANGE),
    "combo": ("Q4 barriers + LED", YELLOW),
    "lecture_led": ("Q4 lecture + LED", MAGENTA),
    "lecture_barriers": ("Q4 lecture + barriers", "#86b6ef"),
    "wheelchair": ("wheelchair assistance", MUTED),
}
EXG = {"EXIT-Main": ("Main entrance", BLUE), "EXIT-Side": ("Side stairs", ORANGE),
       "EXIT-Stage": ("Stage-side stairs", AQUA)}
JDESC = {"A": "orchestra access", "D": "mezzanine door", "C": "corridor", "F": "foyer door",
         "S": "side door", "BS": "balcony stair", "ST": "stage stair", "POR": "portico"}


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
        "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 7.5, "lines.linewidth": 1.8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def ref180(ax, vertical=True, label=True, y=0.98):
    if vertical:
        ax.axvline(180, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        if label:
            ax.text(181.5, y, "180 s target", transform=ax.get_xaxis_transform(), fontsize=7,
                    color=INK, va="top", ha="left")
    else:
        ax.axhline(180, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        if label:
            ax.text(0.005, 181.5, "180 s target", transform=ax.get_yaxis_transform(), fontsize=7,
                    color=INK, va="bottom", ha="left")


_MIR = [(r"\bA(\d)\b", lambda m: f"A{7 - int(m.group(1))}"), (r"\bD(\d)\b", lambda m: f"D{10 - int(m.group(1))}"),
        (r"\bC(\d+)\b", lambda m: f"C{11 - int(m.group(1))}"), (r"\bS(\d)\b", lambda m: f"S{5 - int(m.group(1))}"),
        (r"\bF(\d)\b", lambda m: f"F{4 - int(m.group(1))}"), (r"\bBS(\d)\b", lambda m: f"BS{3 - int(m.group(1))}"),
        (r"\bST(\d)\b", lambda m: f"ST{3 - int(m.group(1))}"), (r"\bPOR(\d)\b", lambda m: f"POR{3 - int(m.group(1))}"),
        (r"\bM(\d)\b", lambda m: f"M{4 - int(m.group(1))}"), (r"\bB(\d)\b", lambda m: f"B{4 - int(m.group(1))}"),
        (r"\bO(\d)\b", lambda m: f"O{4 - int(m.group(1))}")]


def mirror(jid: str) -> str:
    'east–west mirror id.'
    s = jid.replace("→", " → ")
    for pat, fn in _MIR:
        s = re.sub(pat, fn, s)
    s = s.replace("-W", "-@").replace("-E", "-W").replace("-@", "-E")
    return s.replace(" → ", "→")


def jlabel(jid: str, label: str) -> str:
    if "→" in jid:
        return label
    m = re.match(r"([A-Z]+)\d", jid)
    return f"{jid} {JDESC.get(m.group(1), '')}".strip() if m else jid


def dedupe_mirrors(df: pd.DataFrame, col: str = "wait_person_s") -> pd.DataFrame:
    'merge symmetric pairs into one row.'
    rows, used = [], set()
    vals = dict(zip(df["junction"], df[col]))
    for _, r in df.iterrows():
        j = r["junction"]
        if j in used:
            continue
        mj = mirror(j)
        rr = r.copy()
        rr["display"] = jlabel(j, r["label"])
        rr["pair"] = False
        if mj != j and mj in vals and abs(vals[mj] - vals[j]) <= 0.02 * max(abs(vals[j]), 1e-9):
            used.add(mj)
            rr["pair"] = True
            if "→" in j:
                lab = r["label"]
                for a_, b_ in (("W ", ""), ("E ", ""), ("M1→ST1", "M1/M3→ST"), ("M3→ST2", "M1/M3→ST")):
                    lab = lab.replace(a_, b_)
                rr["display"] = lab if lab.startswith("M1/M3") else "W/E " + lab
            else:
                rr["display"] = f"{j} / {mj} {JDESC.get(re.match(r'([A-Z]+)', j).group(1), '')}"
        used.add(j)
        rows.append(rr)
    return pd.DataFrame(rows).reset_index(drop=True)


# model set-up

def fig_floorplan(b, outdir):
    style()
    fig = plt.figure(figsize=(7.2, 3.9))
    ax1 = fig.add_axes([0.0, 0.0, 0.60, 0.92])
    ax2 = fig.add_axes([0.61, 0.30, 0.39, 0.62])
    fp.draw_floor(ax1, b, 1, edges=True)
    fp.draw_floor(ax2, b, 2, edges=True)
    ax1.set_title("(a) Ground floor: Orchestra (O), Mezzanine (M), corridor, foyer", fontsize=8.5)
    ax2.set_title("(b) Balcony (2nd floor)", fontsize=8.5)
    kinds = [("seat", "seat zones (start states)"), ("door", "doors / access points (junctions)"),
             ("corridor", "corridor / foyer"), ("stair", "stairs (junctions)"),
             ("portico", "portico (outdoors)"), ("exit", "exits (absorbing states)")]
    h = [Patch(facecolor=fp.KIND_FILL[k], edgecolor=fp.KIND_EDGE[k], label=l) for k, l in kinds]
    h.append(Line2D([], [], color=INK2, lw=0.6, marker=">", ms=3, label="allowed forward move i→k"))
    ax2.legend(handles=h, loc="upper left", bbox_to_anchor=(0.02, -0.04), fontsize=6.6, ncol=1,
               handlelength=1.4)
    return save(fig, outdir, "A1_floorplan_cells")


# absorbing Markov chain

def fig_markov_time_map(b, t_cell: Dict[str, float], outdir, who="uninstructed concert-goer"):
    style()
    vmax = float(np.nanmax(list(t_cell.values())))
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(7.2, 3.6))
    ax1 = fig.add_axes([0.0, 0.0, 0.60, 0.92])
    ax2 = fig.add_axes([0.61, 0.33, 0.37, 0.58])
    fp.draw_floor(ax1, b, 1, values=t_cell, cmap=SEQ, norm=norm, exit_text={})
    fp.draw_floor(ax2, b, 2, values=t_cell, cmap=SEQ, norm=norm, exit_text={})
    ax1.set_title("(a) Ground floor", fontsize=8.5)
    ax2.set_title("(b) Balcony", fontsize=8.5)
    cax = fig.add_axes([0.64, 0.20, 0.32, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label(f"expected time to absorption t = N·1 (s)\nno congestion, {who}", fontsize=7)
    cb.outline.set_visible(False)
    return save(fig, outdir, "B1_markov_expected_time_map")


def fig_markov_exit_probs(panels: Dict[str, Dict], outdir):
    """panels: name → dict(seats, B_group (nseat×3), t_seat) / per-population closed form."""
    style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.2, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, d) in zip(axes, panels.items()):
        seats = d["seats"]
        y = np.arange(len(seats))[::-1]
        left = np.zeros(len(seats))
        for g, (lab, col) in EXG.items():
            v = d["B_group"][g]
            ax.barh(y, v, left=left, color=col, height=0.72, edgecolor=BG, linewidth=1.0, label=lab)
            for yy, l0, vv in zip(y, left, v):
                if vv >= 0.12:
                    ax.text(l0 + vv / 2, yy, f"{vv:.0%}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if col != YELLOW else INK)
            left += v
        for yy, t in zip(y, d["t_seat"]):
            ax.text(1.02, yy, f"{t:.0f} s", va="center", fontsize=6.8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(seats)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.text(1.02, 1.0, "t = N·1", transform=ax.transAxes, fontsize=6.3, color=INK2, va="bottom")
    axes[0].set_xlabel("absorption probability B = N·R, grouped by exit")
    axes[-1].set_xlabel("absorption probability B = N·R, grouped by exit")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.22), ncol=3)
    fig.subplots_adjust(wspace=0.22)
    return save(fig, outdir, "B2_markov_exit_probabilities")


def fig_markov_screening(rows: pd.DataFrame, outdir):
    """rows: junction, display, chain_load (s), sim_load (s), sim_wmax / dumbbell chain vs flow model."""
    style()
    rows = rows.sort_values("chain_load").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(rows) + 0.9))
    y = np.arange(len(rows))
    for yy, a, c in zip(y, rows["chain_load"], rows["sim_load"]):
        ax.plot([a, c], [yy, yy], color=AXIS, lw=1.2, zorder=1)
    ax.scatter(rows["chain_load"], y, s=26, color=ORANGE, zorder=3, edgecolor=BG, lw=1.0,
               label="Markov chain, nobody adapts: (x₀N)ⱼ·pⱼ / μⱼ")
    ax.scatter(rows["sim_load"], y, s=26, color=BLUE, zorder=3, edgecolor=BG, lw=1.0,
               label="flow model with queue avoidance: people through j / μⱼ")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["display"], fontsize=7)
    ax.set_xlabel("minimum time junction j must stay busy (s) = people through j ÷ capacity μⱼ")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Bottleneck screening: closed-form chain vs. full flow model (concert emergency)")
    return save(fig, outdir, "B3_markov_bottleneck_screening")


def fig_chain_vs_engine(curves: Dict[str, Dict], outdir):
    """curves: label → dict(t, frac, color, ls, mean) / validation."""
    style()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for lab, d in curves.items():
        ax.plot(d["t"], 100 * d["frac"], color=d["color"], ls=d.get("ls", "-"), lw=d.get("lw", 1.8),
                label=f"{lab}  (mean {d['mean']:.1f} s)")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 101)
    ax.set_xlabel("time since alarm (s)")
    ax.set_ylabel("% of occupants not yet absorbed")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Without capacities the flow engine reproduces the chain x(t+1) = x(t)·P")
    return save(fig, outdir, "B4_chain_vs_engine_check")


# flow model

def fig_evac_curves(results: Dict, names: Sequence[str], outdir, lb: Optional[float] = None,
                    fname="C1_evacuation_curves_Q1_Q3", title=None):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ends = []
    for n in names:
        r = results[n]
        lab, col = SC[n]
        frac = 100 * (1 - r.remaining_inside / r.meta["n"])
        ax.plot(np.concatenate([[0], r.t]), np.concatenate([[0], frac]), color=col, label=f"{lab}: T = {r.T:.0f} s")
        ends.append((r.T, col, lab))
    ymax = 100
    for T, col, lab in ends:
        ax.plot([T], [100], marker="o", ms=4.5, color=col, mec=BG, mew=1.0, zorder=4)
    ref180(ax, y=0.62)
    if lb:
        ax.axvline(lb, color=MUTED, lw=0.8, ls=":")
        ax.text(lb - 2, 0.03, f"lower bound {lb:.0f} s\n(max-flow / LP)", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0, max(e[0] for e in ends) * 1.08)
    ax.set_ylim(0, 103)
    ax.set_xlabel("time since alarm / dismissal (s)")
    ax.set_ylabel("% of 1,329 occupants out of the building")
    ax.legend(loc="lower right", fontsize=7)
    if title:
        ax.set_title(title)
    return save(fig, outdir, fname)


def load_values(r, step: int) -> Dict[str, float]:
    'cell load in [0, 1].'
    b = r.b
    out = {}
    x0 = r.occ[0, :b.nT]
    for n, c in enumerate(b.transient):
        cell = b.cell[c]
        occ = r.occ[step, n]
        if cell.kind == "seat":
            init = max(float(r.occ[0, n]), 1e-9)
            out[c] = min(occ / init, 1.0)
        else:
            out[c] = min(occ / cell.n_max, 1.0) if np.isfinite(cell.n_max) else 0.0
    return out


def fig_snapshots(r, outdir, name, times: Optional[Sequence[float]] = None):
    style()
    b = r.b
    if times is None:
        T = r.T
        times = [10, 30] + [float(np.round(f * T / 5) * 5) for f in (0.4, 0.6, 0.8)] + [float(np.floor(T - 2))]
    norm = Normalize(0, 1)
    fig = plt.figure(figsize=(7.2, 6.3))
    ncol, nrow = 2, 3
    W, H = 0.5, 0.31
    lab = SC.get(name, (name, INK))[0]
    for i, tt in enumerate(times[:6]):
        cx, cy = i % ncol, i // ncol
        ax = fig.add_axes([cx * W, 1 - (cy + 1) * H - 0.02, W - 0.01, H - 0.03])
        s = int(np.clip(np.searchsorted(r.t, tt), 0, len(r.t) - 1))
        vals = load_values(r, s)
        exits = {e: f"{r.occ[s, b.idx[e]]:.0f}" for e in b.exits}
        fp.draw_floor(ax, b, 1, values=vals, cmap=SEQ, norm=norm, label_size=4.2, exit_text=exits, lw=0.4)
        out = max(0.0, r.meta["n"] - r.remaining_inside[s])
        ax.set_title(f"t = {r.t[s]:.0f} s   ·   {out:.0f} out ({out / r.meta['n']:.0%})", fontsize=8)
        axb = ax.inset_axes([0.70, -0.02, 0.33, 0.30])
        fp.draw_floor(axb, b, 2, values=vals, cmap=SEQ, norm=norm, labels=False, exit_text={}, lw=0.3)
        axb.text(0.5, 1.0, "balcony", transform=axb.transAxes, fontsize=5.5, ha="center", va="bottom",
                 color=INK2)
    cax = fig.add_axes([0.25, 0.035, 0.5, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label("load: seat zones = share of occupants still there; passages = occupancy ÷ jam capacity "
                 "(k_max·area).\nNumbers at exits = people already at street level there; the title counts people "
                 "out of the BUILDING (portico = outside), so it can be larger.", fontsize=6.3)
    cb.outline.set_visible(False)
    fig.suptitle(f"{lab}: model state over time", x=0.01, ha="left", fontsize=9.5, fontweight="bold", y=1.0)
    return save(fig, outdir, f"C2_snapshots_{name}")


RING = ["POR1", "S1", "S2", *[f"C{k}" for k in range(1, 11)], "S3", "S4", "POR2"]


def fig_kymograph(results: Dict, names: Sequence[str], outdir, kmax: float):
    'density along the ring corridor (W→E) over time.'
    style()
    rows = ["POR1", *[f"C{k}" for k in range(1, 11)], "POR2", "FOY"]
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.55 * n + 0.5), sharex=True)
    axes = np.atleast_1d(axes)
    norm = Normalize(0, kmax)
    tmax = max(results[nm].t[-1] for nm in names)
    for ax, nm in zip(axes, names):
        r = results[nm]
        b = r.b
        M = np.full((len(rows), len(r.t)), np.nan)
        for i, c in enumerate(rows):
            if c in b.idx and np.isfinite(b.cell[c].n_max):
                area = b.cell[c].n_max / kmax
                M[i] = r.occ[:, b.idx[c]] / area
        ax.imshow(M, aspect="auto", cmap=SEQ, norm=norm, interpolation="nearest",
                  extent=[0, r.t[-1], len(rows) - 0.5, -0.5])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=5.8)
        ax.grid(False)
        ax.axhline(len(rows) - 1.5, color=BG, lw=1.5)
        ax.axvline(r.T, color=INK, lw=0.8, ls=":")
        ax.text(0.995, 0.92, f"{SC[nm][0]}  ·  T = {r.T:.0f} s", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color=INK, bbox=dict(facecolor=BG, edgecolor="none", pad=1.5, alpha=0.85))
        ax.set_xlim(0, tmax)
    axes[-1].set_xlabel("time (s)")
    fig.text(0.0, 0.5, "west portico → ring corridor C1…C10 → east portico; foyer", rotation=90,
             va="center", fontsize=7.5, color=INK2)
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.set_label("density (persons / m²)", fontsize=7)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.91, hspace=0.12)
    return save(fig, outdir, "C3_space_time_corridor")


def fig_exit_balance(results: Dict, finish: Dict[str, Dict[str, float]], names: Sequence[str], outdir):
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.33 * len(names) + 1.2), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.1, 1]))
    y = np.arange(len(names))[::-1]
    for yy, n in zip(y, names):
        g = results[n].exit_groups()
        left = 0
        for k, (lab, col) in EXG.items():
            v = g.get(k, 0.0)
            a1.barh(yy, v, left=left, color=col, height=0.7, edgecolor=BG, lw=1.0,
                    label=lab if yy == y[0] else None)
            if v > 110:
                a1.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="#ffffff")
            left += v
    a1.set_yticks(y)
    a1.set_yticklabels([SC[n][0] for n in names])
    a1.set_xlabel("occupants leaving by each exit")
    a1.grid(axis="y", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=6.5, handlelength=1.0,
              columnspacing=0.8)
    a1.set_title("(a) Exit choice")
    mk = {"EXIT-Main": ("o", BLUE), "EXIT-Side-W": ("s", ORANGE), "EXIT-Side-E": ("s", ORANGE),
          "EXIT-Stage-W": ("^", AQUA), "EXIT-Stage-E": ("^", AQUA)}
    for yy, n in zip(y, names):
        f = finish[n]
        vals = [v for v in f.values() if np.isfinite(v)]
        if vals:
            a2.plot([min(vals), max(vals)], [yy, yy], color=AXIS, lw=2.5, zorder=1, solid_capstyle="round")
        for e, v in f.items():
            if np.isfinite(v):
                m, c = mk[e]
                a2.scatter(v, yy, marker=m, s=26, color=c, zorder=3, edgecolor=BG, lw=0.6)
    ref180(a2, y=0.06)
    a2.set_xlabel("time the last person passes each exit (s)")
    a2.set_title("(b) Exit finishing times (spread = imbalance)")
    a2.grid(axis="y", visible=False)
    h = [Line2D([], [], marker=m, color=c, ls="", ms=5, label=l) for l, (m, c) in
         {"Main doors": ("o", BLUE), "side stairs (W, E)": ("s", ORANGE), "stage stairs (W, E)": ("^", AQUA)}.items()]
    a2.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.5, handlelength=1.0,
              columnspacing=0.8)
    fig.subplots_adjust(wspace=0.12)
    return save(fig, outdir, "C4_exit_usage_and_balance")


def fig_exit_flow(results: Dict, names: Sequence[str], outdir, smooth_s=5.0):
    'outflow rate per exit vs capacity.'
    style()
    panels = [("FOY", "EXIT-Main", "Main doors (5 × 1.4 m)"),
              ("POR1", "EXIT-Side-W", "West side stair (1.2 m)"),
              ("ST1", "EXIT-Stage-W", "West stage stair (1.2 m)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    for ax, (i, k, title) in zip(axes, panels):
        mu = None
        for n in names:
            r = results[n]
            if (i, k) not in r.edges:
                continue
            m = r.edges.index((i, k))
            dt = r.meta["dt"]
            w = max(1, int(smooth_s / dt))
            rate = np.convolve(r.flows[:, m] / dt, np.ones(w) / w, mode="same")
            ax.plot(r.t, rate, color=SC[n][1], lw=1.4, label=SC[n][0])
            mu = r.b.edge_cap.get((i, k))
        if (i, k) == ("FOY", "EXIT-Main") and names:
            bb = results[names[0]].b
            up = sum(bb.cell[c].inflow_cap for c in ("F1", "F2", "F3", "BS1", "BS2") if c in bb.cell)
            ax.axhline(up, color=INK2, lw=0.8, ls=":")
            ax.text(0.98, up - 0.25, f"upstream limit: 3 foyer doors\n+ 2 balcony stairs = {up:.2f} p/s",
                    transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=6.0, color=INK2)
        if mu:
            ax.axhline(mu, color=INK, lw=0.8, ls=(0, (4, 2)))
            ax.text(0.98, mu, f"capacity μ = {mu:.2f} p/s", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=6.5)
            ax.set_ylim(0, mu * 1.3)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("outflow (persons / s), 5 s average")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.25), ncol=len(names), fontsize=7)
    fig.subplots_adjust(wspace=0.28)
    return save(fig, outdir, "C5_exit_outflow_vs_capacity")


# queueing

def pick_junctions(qa: Dict, k: int = 4, must: Sequence[str] = ("FOY→EXIT-Main",)) -> List[str]:
    tab = dedupe_mirrors(qa["table"])
    out = [j for j in must if j in qa["series"]]
    for j in tab["junction"]:
        if len(out) >= k:
            break
        if j not in out:
            out.append(j)
    return out[:k]


def fig_ncurves(qa: Dict, outdir, name: str, js: Optional[Sequence[str]] = None):
    style()
    js = js or pick_junctions(qa, 4)
    fig, axes = plt.subplots(1, len(js), figsize=(7.2, 2.6))
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, js):
        s = qa["series"][j]
        t, A, D = s["t"], s["A"], s["D"]
        ax.plot(t, A, color=ORANGE, lw=1.5, label="cumulative arrivals A(t)")
        ax.plot(t, D, color=BLUE, lw=1.5, label="cumulative departures D(t)")
        ax.fill_between(t, D, A, color=ORANGE, alpha=0.13, lw=0)
        q = A - D
        iq = int(np.argmax(q))
        if q[iq] >= 1:
            ax.annotate("", xy=(t[iq], A[iq]), xytext=(t[iq], D[iq]),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
            ax.text(t[iq] + 2, (A[iq] + D[iq]) / 2, f"Q max\n{q[iq]:.0f}", fontsize=6.3, va="center")
        else:
            ax.text(0.05, 0.93, "no queue: A(t) = D(t)\nflow is limited upstream", transform=ax.transAxes,
                    fontsize=6.0, va="top", color=INK2)
        iw = int(np.argmax(s["w"]))
        if s["w"][iw] > 1:
            y0 = A[iw]
            ax.annotate("", xy=(t[iw] + s["w"][iw], y0), xytext=(t[iw], y0),
                        arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
            ax.text(t[iw] + s["w"][iw] / 2, y0 + 0.03 * A[-1], f"W max {s['w'][iw]:.0f} s", fontsize=6.3,
                    ha="center", va="bottom", color=INK2)
        tend = t[np.flatnonzero(D >= D[-1] - 0.5)[0]]
        ax.set_xlim(0, tend * 1.08)
        ax.set_ylim(0, A[-1] * 1.12)
        ax.set_title(jlabel(j, s["label"]), fontsize=7.6)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("persons")
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.2, -0.24), ncol=2)
    fig.suptitle(f"Queueing at the main junctions ({SC[name][0]}): vertical gap = queue Q(t), "
                 "horizontal gap = FIFO wait", x=0.01, ha="left", fontsize=9, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.3)
    return save(fig, outdir, f"D1_cumulative_arrival_departure_{name}")


def _junction_order(b, ids):
    def key(j):
        if "→" in j:
            return (9, j)
        m = re.match(r"([A-Z]+)(\d+)", j)
        order = {"A": 0, "D": 1, "C": 2, "S": 3, "F": 4, "BS": 5, "ST": 6, "POR": 7}
        return (order.get(m.group(1), 8), int(m.group(2))) if m else (9, j)
    return sorted(ids, key=key)


def fig_rho_heatmap(qas: Dict[str, Dict], names: Sequence[str], outdir):
    style()
    ids = set()
    for n in names:
        ids |= {j for j, s in qas[n]["series"].items() if s["D"][-1] > 0.5}
    ids = _junction_order(None, ids)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(7.2, 0.105 * len(ids) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    norm = LogNorm(0.1, 10)
    for ax, nm in zip(axes, names):
        ser = qas[nm]["series"]
        t = next(iter(ser.values()))["t"]
        M = np.full((len(ids), len(t)), np.nan)
        for i, j in enumerate(ids):
            if j in ser:
                rho = ser[j]["rho"].copy()
                busy = (ser[j]["A"] > 0.5) & (ser[j]["D"] < ser[j]["D"][-1] - 0.5)
                rho[~busy] = np.nan
                M[i] = np.clip(rho, 0.1, 10)
        ax.imshow(M, aspect="auto", cmap=DIV, norm=norm, interpolation="nearest",
                  extent=[0, t[-1], len(ids) - 0.5, -0.5])
        ax.set_title(SC[nm][0], fontsize=7.8)
        ax.grid(False)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, min(t[-1], 300))
    axes[0].set_yticks(range(len(ids)))
    axes[0].set_yticklabels([j.replace("→EXIT-", "→") for j in ids], fontsize=5.0)
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax)
    cb.set_label("ρⱼ(t) = λⱼ(t) / μⱼ  (5 s average; blank = idle)", fontsize=7)
    cb.set_ticks([0.1, 0.3, 1, 3, 10]); cb.set_ticklabels(["≤0.1", "0.3", "1", "3", "≥10"])
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.14)
    fig.suptitle("Traffic intensity at every junction: red = arrivals exceed capacity, the queue grows",
                 x=0.01, ha="left", fontsize=9, fontweight="bold", y=1.0)
    return save(fig, outdir, "D2_traffic_intensity_rho")


def fig_bottleneck_ranking(qas: Dict[str, Dict], names: Sequence[str], outdir, top: int = 8):
    style()
    n = len(names)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.15 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, nm in zip(axes, names):
        tab = dedupe_mirrors(qas[nm]["table"])
        tab["total"] = tab["wait_person_s"] * np.where(tab["pair"], 2, 1)
        tab = tab.sort_values("total", ascending=False).head(top).iloc[::-1]
        y = np.arange(len(tab))
        mult = np.where(tab["pair"], 2, 1)
        ax.barh(y, tab["wait_person_s"] * mult / 60, color=SC[nm][1], height=0.65)
        for yy, w, wm in zip(y, tab["wait_person_s"] * mult / 60, tab["W_max"]):
            ax.text(w, yy, f"  max wait {wm:.0f} s", va="center", fontsize=6.2, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(tab["display"], fontsize=6.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(SC[nm][0], fontsize=8)
        ax.set_xlim(0, (tab["wait_person_s"] * mult).max() / 60 * 1.45)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.text(0.5, 0.0, "total time spent queuing at the junction (person-minutes); W/E pairs summed",
             ha="center", fontsize=8, color=INK2)
    fig.suptitle("Where the waiting happens: junctions ranked by total queuing time ∫Qⱼ dt", x=0.01,
                 ha="left", fontsize=9, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.45, wspace=0.75)
    return save(fig, outdir, "D3_bottleneck_ranking")


def fig_mm1(qas: Dict[str, Dict], names: Sequence[str], outdir, mu_ref: float = 1.56,
            ts_scenario="concert"):
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    rho = np.linspace(0.01, 0.985, 300)
    a1.plot(rho, mm1_wq(rho, mu_ref), color=INK, lw=1.4, label=f"M/M/1 steady state, μ = {mu_ref} p/s")
    a1.axvspan(1.0, 1.6, color="#f3a3a2", alpha=0.25, lw=0)
    a1.text(1.3, 0.6, "ρ ≥ 1:\nno steady\nstate", ha="center", fontsize=6.8, color=INK2,
            transform=a1.get_xaxis_transform())
    for nm in names:
        tab = dedupe_mirrors(qas[nm]["table"])
        tab = tab[tab["passed"] > 20]
        a1.scatter(np.minimum(tab["rho_bar"], 1.55), np.maximum(tab["W_mean"], 0.05), s=16, color=SC[nm][1],
                   edgecolor=BG, lw=0.6, label=f"junctions, {SC[nm][0]} (simulated mean wait)", zorder=3)
    a1.set_yscale("log")
    a1.set_xlim(0, 1.6)
    a1.set_ylim(0.05, 300)
    a1.set_xlabel("period-average utilisation ρ̄ = λ̄ / μ")
    a1.set_ylabel("mean wait per person (s)")
    a1.legend(loc="upper left", fontsize=5.8)
    a1.text(0.02, 0.07, "points on the floor: no waiting", transform=a1.transAxes, fontsize=6, color=INK2)
    a1.set_title("(a) M/M/1 vs. simulated waits", fontsize=8.5)
    ser = qas[ts_scenario]["series"]
    js = pick_junctions(qas[ts_scenario], 3)
    cols = [BLUE, ORANGE, AQUA]
    for j, c in zip(js, cols):
        s = ser[j]
        busy = (s["A"] > 0.5) & (s["D"] < s["D"][-1] - 0.5)
        r_ = np.where(busy, s["rho"], np.nan)
        a2.plot(s["t"], r_, color=c, lw=1.3, label=jlabel(j, s["label"]))
    a2.axhline(1, color=INK, lw=0.8, ls=(0, (4, 2)))
    a2.set_yscale("log")
    a2.set_ylim(0.1, 10)
    a2.set_xlim(0, 160)
    a2.set_xlabel("time (s)")
    a2.set_ylabel("ρⱼ(t) = λⱼ(t)/μⱼ")
    a2.legend(loc="upper right", fontsize=6.5)
    a2.set_title(f"(b) ρ(t) is not stationary ({SC[ts_scenario][0]})", fontsize=8.5)
    fig.subplots_adjust(wspace=0.32)
    return save(fig, outdir, "D4_mm1_reference")


def fig_wait_map(qas: Dict[str, Dict], results: Dict, names: Sequence[str], outdir):
    style()
    vmax = 0
    vals = {}
    for nm in names:
        tab = qas[nm]["table"]
        v = {r.junction: r.W_max for r in tab.itertuples() if "→" not in r.junction}
        vals[nm] = v
        vmax = max(vmax, max(v.values()))
    norm = Normalize(0, np.ceil(vmax / 10) * 10)
    fig = plt.figure(figsize=(7.2, 2.75 * int(np.ceil(len(names) / 2)) + 0.4))
    for i, nm in enumerate(names):
        cx, cy = i % 2, i // 2
        rows = int(np.ceil(len(names) / 2))
        ax = fig.add_axes([cx * 0.5, 1 - (cy + 1) / rows * 0.92, 0.49, 0.92 / rows - 0.04])
        r = results[nm]
        tab = qas[nm]["table"].set_index("junction")
        ex = {}
        for e in r.b.exits:
            for j in tab.index:
                if j.endswith("→" + e):
                    ex[e] = f"{tab.loc[j, 'W_max']:.0f} s"
        fp.draw_floor(ax, r.b, 1, values=vals[nm], cmap=SEQ_O, norm=norm, label_size=4.2, exit_text=ex, lw=0.4)
        ax.set_title(SC[nm][0], fontsize=8)
        axb = ax.inset_axes([0.70, -0.02, 0.33, 0.30])
        fp.draw_floor(axb, r.b, 2, values=vals[nm], cmap=SEQ_O, norm=norm, label_size=3.6, exit_text={}, lw=0.3)
    cax = fig.add_axes([0.25, 0.0, 0.5, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ_O), cax=cax, orientation="horizontal")
    cb.set_label("longest FIFO wait to enter the cell (s); numbers at exits = longest wait at the exit door/stair",
                 fontsize=6.5)
    cb.outline.set_visible(False)
    return save(fig, outdir, "D5_longest_wait_map")


# Poisson + Monte Carlo

def fig_reaction(specs: Dict, mc_normal: Dict, outdir):
    from .plugins.randomness import reaction_distribution
    style()
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    a1, a2, a3 = axes
    rvs = {n: reaction_distribution(spec) for n, spec in specs.items()}
    x = np.linspace(0, max(rv.ppf(.995) for rv in rvs.values()), 1201)
    for n, rv in rvs.items():
        a1.plot(x, rv.pdf(x), color=SC[n][1], label=f"{n}: {specs[n]['family']}")
        a3.plot(x, rv.cdf(x), color=SC[n][1], label=n)
    a1.set(xlabel="Pre-movement time (s)", ylabel="Density", title="Configured distributions")
    a1.legend(fontsize=6)
    st = np.asarray(mc_normal["starts"])
    if st.size:
        tt = np.arange(st.shape[1])
        a2.bar(tt+.5, st[0], color=SC["normal"][1], alpha=.5, label="One replicate")
        a2.plot(tt+.5, st.mean(0), color=INK, label="Sample mean")
        a2.plot(tt+.5, mc_normal["n"]*np.diff(rvs["normal"].cdf(np.arange(st.shape[1]+1))),
                color=ORANGE, ls="--", label="Expected bin count")
        a2.legend(fontsize=6)
    a2.set(xlabel="Time since dismissal (s)", ylabel="Starts per second", title="Ordinary dismissal")
    a3.set(xlabel="Pre-movement time (s)", ylabel="Cumulative probability", title="Full distribution tails")
    a3.set_ylim(0, 1)
    fig.tight_layout()
    return save(fig, outdir, "E1_reaction_times_and_starts")


def fig_service_noise(mc: Dict, det, outdir, name="concert"):
    style()
    edges = [(("FOY", "EXIT-Main"), "Main doors"), (("POR1", "EXIT-Side-W"), "West side stair"),
             (("ST1", "EXIT-Stage-W"), "West stage exit")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
    g = mc["grid"]
    for ax, (e, title) in zip(axes, edges):
        paths = [q[e] for q in mc["queues"] if e in q]
        for p in paths:
            ax.plot(g, p, color=SC[name][1], lw=0.4, alpha=0.25)
        if paths:
            ax.plot(g, np.median(np.stack(paths), 0), color=SC[name][1], lw=1.4, label="random service: median")
        if e in det.edges:
            m = det.edges.index(e)
            ax.plot(det.t, det.edge_queue[:, m], color=INK, lw=1.2, ls=(0, (4, 2)), label="deterministic μΔt")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, 200)
    axes[0].set_ylabel("queue length Qⱼ(t) (persons)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"Random service at junctions (mean μⱼΔt, variability κ): {len(paths)} sample paths, {SC[name][0]}", x=0.01, ha="left",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.25)
    return save(fig, outdir, f"E2_queue_paths_service_noise_{name}")


def fig_T_distribution(mcs: Dict[str, Dict], names: Sequence[str], det: Dict, outdir,
                       fname="E3_T_distribution_MC", title=None):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 0.9))
    y = np.arange(len(names))[::-1]
    rng = np.random.default_rng(0)
    xmax = 0
    for yy, n in zip(y, names):
        T = mcs[n]["T"]
        col = SC[n][1]
        ax.scatter(T, yy + rng.uniform(-0.22, 0.22, len(T)), s=2.5, color=col, alpha=0.35, lw=0)
        q = np.percentile(T, [5, 25, 50, 75, 95])
        ax.plot([q[0], q[4]], [yy, yy], color=INK, lw=0.8, zorder=3)
        ax.add_patch(plt.Rectangle((q[1], yy - 0.16), q[3] - q[1], 0.32, facecolor="none", edgecolor=INK, lw=0.8,
                                   zorder=3))
        ax.plot([q[2]] * 2, [yy - 0.16, yy + 0.16], color=INK, lw=1.2, zorder=3)
        if n in det:
            ax.scatter(det[n], yy + 0.3, marker="v", s=18, color=INK, zorder=4)
        xmax = max(xmax, T.max())
        ax.text(1.005, yy, f"mean {mcs[n]['T_mean']:.0f}   T95 {mcs[n]['T_95']:.0f}   "
                           f"P(T≤180) {mcs[n]['P_le_180']:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=6.5, color=INK2)
    ref180(ax, y=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([SC[n][0] for n in names])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("evacuation time T: last person leaves the building (s)")
    K = mcs[names[0]]["K"]
    h = [Line2D([], [], marker="v", color=INK, ls="", ms=5, label="deterministic main model"),
         Line2D([], [], color=INK, lw=0.8, label="box: quartiles; whiskers: 5–95%")]
    ax.legend(handles=h, loc="lower right", fontsize=6.5)
    ax.set_title(title or f"Monte Carlo distribution of T (K = {K} each; random reaction times & door service)")
    return save(fig, outdir, fname)


def fig_mc_bands(mcs: Dict[str, Dict], results: Dict, names: Sequence[str], outdir):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for n in names:
        m = mcs[n]
        frac = 100 * (1 - m["curves"] / m["n"])
        lo, med, hi = np.percentile(frac, [5, 50, 95], axis=0)
        ax.fill_between(m["grid"], lo, hi, color=SC[n][1], alpha=0.2, lw=0)
        ax.plot(m["grid"], med, color=SC[n][1], lw=1.6, label=f"{SC[n][0]}")
        r = results[n]
        ax.plot(np.concatenate([[0], r.t]), 100 * (1 - np.concatenate([[r.meta['n']], r.remaining_inside]) / r.meta["n"]),
                color=SC[n][1], lw=0.8, ls=(0, (3, 2)))
    ref180(ax, y=0.25)
    ax.set_xlim(0, max(np.percentile(mcs[n]["T"], 99) for n in names) * 1.05)
    ax.set_ylim(0, 102)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("% out of the building")
    h = ax.get_legend_handles_labels()
    extra = [Patch(color=MUTED, alpha=0.3, label="5–95% band (MC)"),
             Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=0.8, label="deterministic")]
    ax.legend(handles=h[0] + extra, loc="lower right", fontsize=6.8)
    ax.set_title("Evacuation curves under stochastic sampling: median and 5–95% band")
    return save(fig, outdir, "E4_evacuation_bands_MC")


def fig_crn(mcs: Dict[str, Dict], pairs: Sequence[tuple], outdir):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(pairs) + 0.9))
    y = np.arange(len(pairs))[::-1]
    rng = np.random.default_rng(1)
    for yy, (a, b) in zip(y, pairs):
        d = mcs[a]["T"] - mcs[b]["T"]
        ax.scatter(d, yy + rng.uniform(-0.2, 0.2, len(d)), s=2.5, color=SC[a][1], alpha=0.4, lw=0)
        q = np.percentile(d, [5, 25, 50, 75, 95])
        ax.plot([q[0], q[4]], [yy, yy], color=INK, lw=0.8)
        ax.add_patch(plt.Rectangle((q[1], yy - 0.15), q[3] - q[1], 0.3, facecolor="none", edgecolor=INK, lw=0.8))
        ax.plot([q[2]] * 2, [yy - 0.15, yy + 0.15], color=INK, lw=1.2)
        sd_pair = d.std(ddof=1)
        sd_ind = np.sqrt(mcs[a]["T"].var(ddof=1) + mcs[b]["T"].var(ddof=1))
        ax.text(1.005, yy, f"mean ΔT {d.mean():+.1f} s   P(ΔT<0) {np.mean(d < 0):.2f}   "
                           f"sd paired {sd_pair:.1f} vs indep. {sd_ind:.1f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color=INK2)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SC[a][0]}  −  {SC[b][0].split(' ', 1)[1]}" for a, b in pairs], fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired difference ΔT⁽ᵐ⁾ = T_design⁽ᵐ⁾ − T_baseline⁽ᵐ⁾ (s); negative = design is faster")
    ax.set_title("Interventions compared with common random numbers (same people, same service draws)")
    return save(fig, outdir, "E5_paired_differences_CRN")


def fig_convergence(mcs: Dict[str, Dict], names: Sequence[str], pmc: Optional[pd.DataFrame], outdir):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for n in names:
        T = mcs[n]["T"]
        k = np.arange(1, len(T) + 1)
        mean = np.cumsum(T) / k
        sd = np.array([T[:i].std(ddof=1) if i > 1 else 0 for i in k])
        axes[0].plot(k, mean - mean[-1], color=SC[n][1], lw=1.2, label=SC[n][0])
        axes[0].fill_between(k, mean - mean[-1] - 1.96 * sd / np.sqrt(k), mean - mean[-1] + 1.96 * sd / np.sqrt(k),
                             color=SC[n][1], alpha=0.15, lw=0)
        p95 = np.array([np.percentile(T[:i], 95) for i in k])
        axes[1].plot(k, p95 - p95[-1], color=SC[n][1], lw=1.2)
    axes[0].set_ylim(-4, 4)
    axes[1].set_ylim(-6, 6)
    axes[0].set_title("(a) running mean − final (±95% CI)", fontsize=7.2)
    axes[1].set_title("(b) running T95 − final", fontsize=7.2)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("replicates K")
        ax.axhline(0, color=INK, lw=0.6)
    axes[0].set_ylabel("seconds")
    if pmc is not None:
        for n, g in pmc.groupby("scenario"):
            if n not in SC:
                continue
            g = g.sort_values("m")
            hit = (g["T"].to_numpy() <= 180).astype(float)
            k = np.arange(1, len(hit) + 1)
            p = np.cumsum(hit) / k
            se = np.sqrt(np.maximum(p * (1 - p), 1e-9) / k)
            axes[2].plot(k, p, color=SC[n][1], lw=1.2, label=SC[n][0])
            axes[2].fill_between(k, np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1), color=SC[n][1],
                                 alpha=0.15, lw=0)
        axes[2].set_xscale("log")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_xlabel("replicates K")
        axes[2].set_title("(c) running P(T ≤ 180), parameter MC", fontsize=7.6)
        axes[2].legend(fontsize=5.8, loc="lower left")
    fig.subplots_adjust(wspace=0.45)
    return save(fig, outdir, "E6_MC_convergence")


def fig_reaction_sensitivity(groups: Dict[str, Dict[str, Dict]], det: Dict, outdir):
    """groups: scenario → {variant label: mc}."""
    style()
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    cols = [MUTED, BLUE, ORANGE, AQUA, VIOLET]
    for ax, (n, var) in zip(axes, groups.items()):
        labs = list(var)
        data = [var[l]["T"] for l in labs]
        bp = ax.boxplot(data, vert=True, widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.2))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor(INK); p.set_linewidth(0.7)
        short = {"reaction only\n(baseline R)": "reaction\nonly", "service only": "service\nonly", "movement only": "movement\nonly",
                 "all\n(baseline R)": "all,\nbaseline R", "all\n(exponential R)": "all,\nexp. R",
                 "both\n(Poisson stream)": "both,\nPoisson\nstream"}
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels([short.get(l, l) for l in labs], fontsize=6.0)
        if n in det:
            ax.axhline(det[n], color=INK, lw=0.8, ls=":")
        ax.set_title(SC[n][0], fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("T (s)")
    axes[0].text(0.02, 0.02, "dotted: deterministic model", transform=axes[0].transAxes, fontsize=6, color=INK2)
    fig.suptitle("Randomness: reaction, movement and door service (whiskers 5–95%)",
                 x=0.01, ha="left", fontsize=9, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.3)
    return save(fig, outdir, "E7_reaction_vs_service_randomness")


def fig_broadcast(df_det: pd.DataFrame, mcs: Dict[float, Dict], refs: Dict[str, float], outdir):
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.plot(df_det["delay"], df_det["T"], color=RED, lw=1.6, marker="o", ms=3.5, label="deterministic main model")
    if mcs:
        d = sorted(mcs)
        lo = [np.percentile(mcs[x]["T"], 5) for x in d]
        hi = [np.percentile(mcs[x]["T"], 95) for x in d]
        med = [np.median(mcs[x]["T"]) for x in d]
        ax.fill_between(d, lo, hi, color=RED, alpha=0.18, lw=0, label="Monte Carlo 5–95%")
        ax.plot(d, med, color=RED, lw=0.9, ls=(0, (3, 2)), label="Monte Carlo median")
    for lab, (val, ls) in refs.items():
        ax.axhline(val, color=INK2, lw=0.8, ls=ls)
        ax.text(df_det["delay"].max(), val + 3, lab, ha="right", fontsize=6.5, color=INK2)
    ref180(ax, vertical=False)
    ax.set_xlabel("delay before crowd managers broadcast the blockage (s)")
    ax.set_ylabel("evacuation time T (s)")
    ax.set_ylim(100, None)
    ax.legend(loc="center left", fontsize=6.6, bbox_to_anchor=(0.0, 0.42))
    pmax = max(mcs[x]["P_le_180"] for x in mcs) if mcs else float("nan")
    ax.set_title(f"Q2: main entrance blocked and unknown — value of fast information (max P(T≤180) = {pmax:.2f})",
                 fontsize=8.6)
    return save(fig, outdir, "E8_blocked_exit_broadcast_delay")


# sensitivity and robustness

def fig_tornado(df: pd.DataFrame, outdir, names: Sequence[str]):
    style()
    n = len(names)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 3.2 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    order = None
    for ax, nm in zip(axes, names):
        g = df[df["scenario"] == nm]
        base = g["T_base"].iloc[0]
        rows = []
        for lab, sub in g.groupby("label"):
            sub = sub.sort_values("value")
            rows.append((lab, sub["T"].iloc[0] - base, sub["T"].iloc[-1] - base))
        rr = pd.DataFrame(rows, columns=["label", "lo", "hi"])
        if order is None:
            allr = []
            for lab, sub in df.groupby("label"):
                allr.append((lab, (sub["T"] - sub["T_base"]).abs().max()))
            order = [l for l, _ in sorted(allr, key=lambda x: x[1])]
        rr = rr.set_index("label").reindex(order)
        y = np.arange(len(order))
        ax.barh(y + 0.18, rr["lo"].fillna(0), color=BLUE, height=0.34, label="parameter at low end of range")
        ax.barh(y - 0.18, rr["hi"].fillna(0), color=ORANGE, height=0.34, label="parameter at high end of range")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=6.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{SC[nm][0]} (T = {base:.0f} s)", fontsize=8)
        lim = np.nanmax(np.abs(rr[["lo", "hi"]].to_numpy()))
        ax.set_xlim(-lim * 1.15, lim * 1.15)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncol):n]:
        ax.set_xlabel("change in T (s)")
    axes[0].legend(loc="lower left", fontsize=6.5)
    fig.suptitle("One-at-a-time sensitivity of T across the assumed parameter ranges (other parameters nominal)",
                 x=0.01, ha="left", fontsize=9, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.08, hspace=0.3)
    return save(fig, outdir, "F1_tornado_sensitivity")


def fig_param_mc(pmc: pd.DataFrame, names: Sequence[str], outdir, det: Dict):
    from scipy.stats import spearmanr
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for n in names:
        T = np.sort(pmc[pmc["scenario"] == n]["T"].to_numpy())
        p = np.arange(1, len(T) + 1) / len(T)
        a1.step(T, p, where="post", color=SC[n][1], lw=1.5,
                label=f"{SC[n][0]}: P(T≤180) = {np.mean(T <= 180):.2f}")
        if n in det:
            a1.scatter(det[n], 0.02, marker="v", color=SC[n][1], s=16, zorder=4)
    ref180(a1, y=0.08)
    a1.set_xscale("log")
    a1.set_xticks([60, 100, 150, 200, 300, 500, 800])
    a1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.set_xlabel("T (s), log scale")
    a1.set_ylabel("cumulative probability")
    a1.legend(fontsize=5.6, loc="lower right", handlelength=1.2)
    a1.set_title("(a) T under parameter uncertainty + random noise", fontsize=8)
    from .uncertainty import RANGE

    keys = [c for c in pmc.columns if c in RANGE]
    width = 0.8 / len(names)
    y = np.arange(len(keys))
    rho = {}
    for i, n in enumerate(names):
        g = pmc[pmc["scenario"] == n]
        rho[n] = [spearmanr(g[k], g["T"]).statistic if g[k].nunique() > 1 else 0 for k in keys]
    order = np.argsort(np.max(np.abs(np.array(list(rho.values()))), axis=0))
    for i, n in enumerate(names):
        a2.barh(y + (i - (len(names) - 1) / 2) * width, np.array(rho[n])[order], height=width, color=SC[n][1],
                label=SC[n][0])
    a2.set_yticks(y)
    a2.set_yticklabels([RANGE[keys[k]].label for k in order], fontsize=6.2)
    a2.axvline(0, color=INK, lw=0.7)
    a2.set_xlim(-1, 1)
    a2.set_xlabel("Spearman rank correlation with T")
    a2.grid(axis="y", visible=False)
    a2.set_title("(b) Which uncertain input drives T", fontsize=8)
    fig.subplots_adjust(wspace=0.75)
    return save(fig, outdir, "F2_parameter_uncertainty_MC")


def fig_beta(df: pd.DataFrame, outdir, nominal: Dict[str, Dict[str, float]]):
    style()
    pars = [("beta_dist", "β1 distance aversion (per s)"), ("beta_habit", "β2 main-entrance habit"),
            ("beta_queue", "β3 queue avoidance (per s)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (p, lab) in zip(axes, pars):
        g = df[df["parameter"] == p]
        for n in ("concert", "lecture"):
            h = g[g["scenario"] == n]
            ax.plot(h["value"], h["T"], marker="o", ms=3.5, color=SC[n][1], label=SC[n][0])
            if n in nominal and p in nominal[n]:
                ax.axvline(nominal[n][p], color=SC[n][1], lw=0.7, ls=":")
        ax.set_xlabel(lab)
        ref180(ax, vertical=False, label=(ax is axes[0]))
    axes[0].set_ylabel("T (s)")
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Route-choice parameters (dotted: values used)", x=0.01, ha="left", fontsize=9,
                 fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.1)
    return save(fig, outdir, "F3_route_choice_beta_sensitivity")


def fig_compliance(df: pd.DataFrame, outdir):
    style()
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    for n in ("concert", "barriers", "led", "combo"):
        g = df[df["scenario"] == n]
        ax.plot(100 * g["compliance"], g["T"], marker="o", ms=3, color=SC[n][1], label=SC[n][0])
    ax.set_xlabel("share of occupants following instructions / LED colour (%)")
    ax.set_ylabel("T (s)")
    ax.legend(fontsize=7)
    ax.set_title("Robustness to compliance (designs fixed at nominal compliance)")
    return save(fig, outdir, "F4_compliance_robustness")


def fig_barriers(df: pd.DataFrame, b, layouts: Dict[str, tuple], outdir):
    style()
    fig = plt.figure(figsize=(7.2, 3.0))
    ax0 = fig.add_axes([0.0, 0.05, 0.42, 0.85])
    fp.draw_floor(ax0, b, 1, labels=True, label_size=4.2)
    cols = {"C2|C3": AQUA, "C3|C4": ORANGE, "C4|C5": VIOLET}
    for lay, cut in layouts.items():
        if not cut:
            continue
        for a, c in cut:
            ka = int(a[1:])
            ang = 18 * ka
            p0, p1 = fp._pt(28.2, ang), fp._pt(31.6, ang)
            ax0.plot([p0[0], p1[0]], [p0[1], p1[1]], color=cols.get(lay, RED), lw=3.2, solid_capstyle="butt",
                     zorder=8)
    ax0.set_title("(a) Candidate barrier positions (symmetric pairs)", fontsize=8)
    h = [Line2D([], [], color=c, lw=3, label=l) for l, c in cols.items()]
    ax0.legend(handles=h, loc="lower left", fontsize=6.5, ncol=1)
    ax = fig.add_axes([0.52, 0.17, 0.46, 0.72])
    lays = list(layouts)
    x = np.arange(len(lays))
    mk = {("concert", "managers"): ("o", SC["concert"][1], "concert, crowd managers"),
          ("concert", "led"): ("s", SC["led"][1], "concert, LED strips"),
          ("lecture", "managers"): ("o", SC["lecture"][1], "lecture, no guidance"),
          ("lecture", "led"): ("s", SC["lecture_led"][1], "lecture, LED strips")}
    for (ev, gd), (m, c, lab) in mk.items():
        g = df[(df["event"] == ev) & (df["guidance"] == gd)].set_index("layout").reindex(lays)
        off = {"concert": -0.08, "lecture": 0.08}[ev]
        ax.plot(x + off, g["T"], marker=m, ms=4.5, color=c, lw=1.0, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["no barrier" if l == "none" else l for l in lays])
    ax.set_ylabel("T (s)")
    ax.set_title("(b) Evacuation time by barrier layout", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    return save(fig, outdir, "F5_barrier_layouts")


def fig_blocked_sweep(df: pd.DataFrame, outdir):
    style()
    d = df[df["target"] != "none"]
    base = df[df["target"] == "none"]["T"].iloc[0]
    targets = list(dict.fromkeys(d["target"]))
    modes = ["known", "broadcast", "arrival"]
    M = d.pivot_table(index="target", columns="discovery", values="T").reindex(index=targets, columns=modes)
    fig, ax = plt.subplots(figsize=(5.2, 0.38 * len(targets) + 1.0))
    values = M.to_numpy()
    finite = values[np.isfinite(values)]
    bounds = np.r_[finite, base if np.isfinite(base) else 180, 180]
    norm = TwoSlopeNorm(vcenter=180, vmin=bounds.min() - 1, vmax=max(bounds.max(), 200))
    cmap = DIV.copy()
    cmap.set_bad("#d9dde2")
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            label = f"{v:.0f} s" if np.isfinite(v) else ("not cleared" if np.isinf(v) else "no data")
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if np.isfinite(v) and abs(norm(v) - 0.5) > 0.33 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["known at t = 0", "found at door\n+ broadcast", "found at door only"], fontsize=7)
    names = {"EXIT-Main": "main exterior doors", "F2": "F2 central foyer door", "D5": "D5 central mezz. door",
             "EXIT-Side-W": "west side stair", "EXIT-Stage-W": "west stage exit", "BS1": "west balcony stair"}
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([names.get(t, t) for t in targets], fontsize=7)
    ax.grid(False)
    ax.set_title(f"Q2: which obstruction hurts, and does information help? (no obstruction: {base:.0f} s)",
                 fontsize=8.5)
    return save(fig, outdir, "F6_blocked_exit_sweep")


def fig_method_checks(df: pd.DataFrame, outdir):
    style()
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for (wm, n), g in df.groupby(["walk_model", "scenario"]):
        ax.plot(g["dt"], g["T"], marker="o" if wm == "erlang" else "s", ms=4, color=SC[n][1],
                ls="-" if wm == "erlang" else (0, (3, 2)), label=f"{SC[n][0]}, {wm} walking")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1.0]); ax.set_xticklabels(["0.25", "0.5", "1"])
    ax.set_xlabel("time step Δt (s)")
    ax.set_ylabel("T (s)")
    ax.legend(fontsize=6.3)
    ax.set_title("Numerical checks: Δt and walking-time distribution")
    return save(fig, outdir, "F7_numerical_checks")


# design and cost (Q5, crowd managers, service variability, wheelchairs)

def fig_managers(comp: pd.DataFrame, runs: pd.DataFrame, cfg, outdir):
    'crowd managers → compliance → T.'
    style()
    from .config import MANAGERS
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    groups = {"orchestra (O1–O3)": ("O", ORANGE), "mezzanine (M1–M3)": ("M", AQUA), "balcony (B1–B3)": ("B", VIOLET)}
    seats = {k: float(x) for k, x in cfg.pop.seat_counts().items()}
    for lab, (pre, col) in groups.items():
        zs = [z for z in seats if z.startswith(pre)]
        w = np.array([seats[z] for z in zs])
        avg = sum(comp[f"c_{z}"] * seats[z] for z in zs) / w.sum()
        a1.plot(comp["managers"], avg, color=col, lw=1.4, marker="o", ms=2.5, label=lab)
    a1.plot(comp["managers"], comp["avg"], color=INK, lw=2.0, label="whole hall")
    a1.axvline(MANAGERS["law_min"], color=INK2, lw=0.8, ls=":")
    a1.text(MANAGERS["law_min"] + 0.2, 0.03, "legal\nminimum", fontsize=6.3, color=INK2)
    a1.set_xlabel("number of crowd managers n")
    a1.set_ylabel("initial seat-node coverage × response share")
    a1.set_ylim(0, 1)
    a1.legend(fontsize=6.3, loc="lower right")
    a1.set_title(f"(a) Local radius {MANAGERS['radius_m']:g} m; response {MANAGERS['p_follow']}", fontsize=8)
    for (led, bar), (col, lab) in {(False, False): (BLUE, "managers only"),
                                   (True, False): (ORANGE, "managers + LED strips")}.items():
        g = runs[(runs["led"] == led) & (runs["barriers"] == bar)].sort_values("managers")
        a2.plot(g["managers"], g["T_det"], color=col, marker="o", ms=3.5, lw=1.4, label=lab)
    a2.axvline(MANAGERS["law_min"], color=INK2, lw=0.8, ls=":")
    ref180(a2, vertical=False)
    a2.set_xlabel("number of crowd managers n")
    a2.set_ylabel("T, concert emergency (s)")
    a2.legend(fontsize=6.5, loc="center right")
    a2.set_title("(b) More managers vs. LED strips (deterministic)", fontsize=8)
    fig.subplots_adjust(wspace=0.3)
    return save(fig, outdir, "G1_crowd_managers")


def fig_pareto(df: pd.DataFrame, cfg, outdir):
    'Q5: yearly cost vs evacuation time, Pareto front.'
    style()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    mk = {(False, False): ("o", "managers only"), (True, False): ("s", "+ LED strips"),
          (False, True): ("^", "+ barriers"), (True, True): ("D", "+ LED + barriers")}
    ns = sorted(df["managers"].unique())
    cmap = LinearSegmentedColormap.from_list("mgr", ["#9ec5f4", "#0d366b"])
    norm = Normalize(min(ns), max(ns))
    for (led, bar), (m, lab) in mk.items():
        g = df[(df["led"] == led) & (df["barriers"] == bar)]
        lo_c, hi_c = ("T_mc_ci_lo", "T_mc_ci_hi") if "T_mc_ci_lo" in g else ("T_mc05", "T_mc95")
        ax.errorbar(g["cost_total"] / 1000, g["T_mc"], yerr=[g["T_mc"] - g[lo_c], g[hi_c] - g["T_mc"]],
                    fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=1)
        ax.scatter(g["cost_total"] / 1000, g["T_mc"], marker=m, s=34, c=[cmap(norm(n)) for n in g["managers"]],
                   edgecolor=INK, lw=0.5, zorder=3, label=lab)
    front = df[df["pareto"]].sort_values("cost_total")
    ax.step(front["cost_total"] / 1000, front["T_mc"], where="post", color=RED, lw=1.2, zorder=2,
            label="Pareto front")
    for r in front.itertuples():
        ax.annotate(f"n={r.managers}", (r.cost_total / 1000, r.T_mc), xytext=(4, 4), textcoords="offset points",
                    fontsize=6.3, color=INK)
    ax.set_xlabel("yearly cost of the design (thousand USD / year; ASSUMED unit costs, see config.Costs)")
    ax.set_ylabel("mean T, concert emergency (s)\nMonte Carlo; bars = 95% CI of the mean")
    ax.legend(fontsize=6.5, loc="center right", ncol=1)
    cax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("crowd managers n", fontsize=7)
    cb.outline.set_visible(False)
    ax.set_title("Q5: cost vs. evacuation time for every design (n ≥ 6 by law)")
    fig.subplots_adjust(right=0.9)
    return save(fig, outdir, "G2_cost_pareto")


def fig_dispersion(disp: Dict, names: Sequence[str], kappas: Sequence[float], det: Dict, outdir):
    'how service variability κ changes the results.'
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    width = 0.8 / len(kappas)
    cols = [MUTED, AQUA, BLUE, RED]
    names_k = {0.0: "κ = 0 (hard capacity)", 0.5: "κ = 0.5 (orderly lanes, default)",
               # 1.0: "κ = 1 (Poisson)",
               2.0: "κ = 2 (bursty)"}
    for j, k in enumerate(kappas):
        color = RED if k == 2.0 else cols[j]
        data = [disp[(n, k)]["T"] for n in names]
        pos = np.arange(len(names)) + (j - (len(kappas) - 1) / 2) * width
        bp = a1.boxplot(data, positions=pos, widths=width * 0.85, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops=dict(color=INK, lw=1.0))
        for b_ in bp["boxes"]:
            b_.set_facecolor(color); b_.set_alpha(0.45); b_.set_edgecolor(INK); b_.set_linewidth(0.6)
        a1.plot([], [], color=color, lw=6, alpha=0.45, label=names_k.get(k, f"κ = {k}"))
    for i, n in enumerate(names):
        a1.plot([i - 0.42, i + 0.42], [det[n]] * 2, color=INK, lw=0.8, ls=":")
    a1.set_xticks(range(len(names)))
    a1.set_xticklabels([SC[n][0].replace(" emergency", "").replace(" dismissal", "") for n in names], fontsize=7)
    a1.set_ylabel("T (s)")
    a1.legend(fontsize=6.0, loc="upper left")
    a1.grid(axis="x", visible=False)
    a1.set_title("(a) T vs. service variability κ = Var/Mean", fontsize=8)
    a1.text(0.99, 0.02, "dotted: deterministic", transform=a1.transAxes, ha="right", fontsize=6, color=INK2)
    if "barriers" in names and "concert" in names:
        m, lo, hi = [], [], []
        for k in kappas:
            d = disp[("barriers", k)]["T"] - disp[("concert", k)]["T"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            m.append(d.mean()); lo.append(d.mean() - 1.96 * se); hi.append(d.mean() + 1.96 * se)
        m = np.array(m)
        a2.errorbar(kappas, m, yerr=[m - np.array(lo), np.array(hi) - m], color=AQUA, marker="o", ms=4, capsize=3)
        a2.axhline(0, color=INK, lw=0.8)
        a2.set_xlabel("service variability κ")
        a2.set_ylabel("mean ΔT, barriers − baseline (s)")
        a2.set_title("(b) Barrier penalty vs. κ", fontsize=8)
    fig.subplots_adjust(wspace=0.3)
    return save(fig, outdir, "E9_service_variability_kappa")


def fig_wheelchair(df: pd.DataFrame, outdir):
    'Wheelchairs: refuge vs street time; placement × guidance; assistance delay.'
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
    lay = df[(df["kind"] == "layout") & (df["scenario"] == "concert")]
    labs = list(dict.fromkeys(lay["layout"]))
    x = np.arange(len(labs))
    for j, (guide, col, nm) in enumerate((("nearest", MAGENTA, "nearest step-free exit"),
                                          ("main", VIOLET, "guided to the main doors"))):
        g = lay[lay["guide"] == guide].set_index("layout").reindex(labs)
        off = (j - 0.5) * 0.36
        a1.bar(x + off, g["T_refuge"], width=0.34, color=col, alpha=0.95, label=f"{nm}: outdoor refuge")
        a1.bar(x + off, g["T_street"] - g["T_refuge"], bottom=g["T_refuge"], width=0.34, color=col, alpha=0.35,
               label=f"{nm}: + assisted descent to street")
        for xx, r_, s_ in zip(x + off, g["T_refuge"], g["T_street"]):
            a1.text(xx, s_ + 3, f"{s_:.0f}", ha="center", fontsize=6.2)
            a1.text(xx, r_ - 3, f"{r_:.0f}", ha="center", va="top", fontsize=6.0, color="#ffffff")
    t_all = lay["T"].mean()
    a1.axhline(t_all, color=SC["concert"][1], lw=0.9, ls=(0, (4, 2)))
    a1.text(-0.55, t_all + 4, f"last person out of the building ≈ {t_all:.0f} s", ha="left", fontsize=6.0,
            color=SC["concert"][1], bbox=dict(facecolor=BG, edgecolor="none", pad=0.5))
    a1.set_xticks(x)
    a1.set_xticklabels(labs, fontsize=6.3)
    a1.set_ylabel("time (s), concert emergency")
    a1.set_ylim(0, lay["T_street"].max() * 1.55)
    a1.legend(fontsize=5.6, loc="upper left", ncol=1)
    a1.grid(axis="x", visible=False)
    a1.set_title("(a) Where to seat them, where to send them", fontsize=8)
    dl = df[df["kind"] == "delay"]
    for guide, col, nm in (("nearest", MAGENTA, "nearest"), ("main", VIOLET, "main doors")):
        g = dl[dl["guide"] == guide].sort_values("delay")
        a2.plot(g["delay"], g["T_refuge"], color=col, marker="o", ms=3.5, lw=1.4, label=f"{nm}: refuge")
        a2.plot(g["delay"], g["T_street"], color=col, marker="s", ms=3.5, lw=1.0, ls=(0, (3, 2)),
                label=f"{nm}: street")
    a2.set_xlabel("extra wait for an assistant (s)")
    a2.set_ylabel("time (s)")
    a2.legend(fontsize=6.0, loc="center right", bbox_to_anchor=(1.0, 0.68))
    a2.set_title("(b) Sensitivity to assistance delay (preset seats)", fontsize=8)
    fig.suptitle("Wheelchair users: 3 people, half speed, no stairs; elevators prohibited; assisted stair "
                 "descent 0.31 m/s (NIST)", x=0.01, ha="left", fontsize=8.6, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.3)
    return save(fig, outdir, "G3_wheelchair_placement")


def fig_validation(v: Dict, outdir):
    'engine vs closed form; specific flow vs data.'
    style()
    from .config import VALIDATION_FLOWS
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.3]))
    d = v["door"]
    cols = {0.9: BLUE, 1.2: ORANGE, 1.8: AQUA}
    for w, g in d.groupby("width"):
        a1.scatter(g["T_analytic"], g["T_sim"], s=22, color=cols.get(w, INK), edgecolor=BG, lw=0.6,
                   label=f"door {w:g} m (μ = {g['mu'].iloc[0]:.2f} p/s)", zorder=3)
    lim = [0, d[["T_analytic", "T_sim"]].max().max() * 1.05]
    a1.plot(lim, lim, color=INK, lw=0.8, ls=":")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("closed form τ + N/μ (s)")
    a1.set_ylabel("engine T (s)")
    a1.legend(fontsize=6.2, loc="upper left")
    a1.set_title(f"(a) One room, one door: max error {100 * d['rel_err'].abs().max():.1f}%", fontsize=8)
    sf = pd.concat([df.assign(scenario=n) for n, df in v["specific"].items()])
    sf = sf[sf["kind"].isin(["door", "access", "stair"])]
    order = ["concert", "lecture", "normal"]
    rng = np.random.default_rng(0)
    for i, n in enumerate(order):
        g = sf[sf["scenario"] == n]
        a2.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g["specific_flow_clear"], s=16, color=SC[n][1],
                   edgecolor=BG, lw=0.5, zorder=3)
    for key, lab, ls in (("theatre_drill", "theatre drill, peak exit flow (Imanishi 2018)", "-"),
                         ("lab_bottleneck", "laboratory bottleneck (Liddle et al.)", (0, (4, 2)))):
        a2.axhline(VALIDATION_FLOWS[key], color=INK, lw=0.9, ls=ls)
        a2.text(2.45, VALIDATION_FLOWS[key] + 0.03, f"{lab}: {VALIDATION_FLOWS[key]}", ha="right", fontsize=6.2)
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels([SC[n][0].split(" ", 1)[1] for n in order], fontsize=7)
    a2.set_ylabel("peak flow per metre of clear width\n(persons/(m·s), 10 s window)")
    a2.set_ylim(0, 2.2)
    a2.grid(axis="x", visible=False)
    a2.set_title("(b) Model door/stair peak flow (= capacity) vs. measured", fontsize=8)
    fig.subplots_adjust(wspace=0.32)
    return save(fig, outdir, "H1_validation")


def fig_capacity_bracket(v: Dict, names: Sequence[str], outdir):
    'T under three capacity assumptions: a bracket.'
    style()
    det = v["bracket_det"]
    mc = v["bracket_mc"]
    caps = list(dict.fromkeys(det["capacity"]))
    cols = [BLUE, GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(names) + 1.0))
    y = np.arange(len(names))[::-1]
    for j, (cap, col) in enumerate(zip(caps, cols)):
        g = det[det["capacity"] == cap].set_index("scenario").reindex(names)
        off = (j - 1) * 0.22
        ax.scatter(g["T"], y + off, color=col, s=26, marker="o", zorder=3, label=cap, edgecolor=BG, lw=0.6)
        m = mc[mc["capacity"] == cap].set_index("scenario")
        for yy, n in zip(y, names):
            if n in m.index:
                ax.plot([m.loc[n, "T_05"], m.loc[n, "T_95"]], [yy + off] * 2, color=col, lw=1.0, alpha=0.7)
    for yy, n in zip(y, names):
        g = det[det["scenario"] == n]["T"]
        ax.plot([g.min(), g.max()], [yy - 0.3] * 2, color=AXIS, lw=3, solid_capstyle="butt", zorder=1)
    ref180(ax, y=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([SC[n][0] for n in names])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("T (s): dots = deterministic; lines = Monte Carlo 5–95%; grey bar = bracket")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("How T depends on door capacity: model default vs. measured drill and lab flows")
    return save(fig, outdir, "H2_capacity_bracket")


# added: T definition, cost sensitivity, wheelchair MC, seat-row check

def fig_T_definition(mcs: Dict[str, Dict], names: Sequence[str], outdir):
    "T (leave the building, the problem's definition) next to T_ground (down the exterior stairs to the street)."
    style()
    names = [n for n in names if n in mcs]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.0))
    y = np.arange(len(names))[::-1]
    for yy, n in zip(y, names):
        col = SC[n][1]
        for arr, dy, alpha in ((mcs[n]["T"], 0.15, 1.0), (mcs[n]["T_ground"], -0.15, 0.45)):
            a = np.asarray(arr, float)
            q = np.percentile(a, [5, 50, 95])
            ax.plot([q[0], q[2]], [yy + dy] * 2, color=col, lw=4.5, alpha=alpha, solid_capstyle="butt")
            ax.plot([q[1]] * 2, [yy + dy - 0.12, yy + dy + 0.12], color=INK, lw=1.0)
        g = float(np.mean(np.asarray(mcs[n]["T_ground"]) - np.asarray(mcs[n]["T"])))
        ax.text(1.005, yy, f"T {np.mean(mcs[n]['T']):.0f} s → street {np.mean(mcs[n]['T_ground']):.0f} s "
                           f"(+{g:.0f} s)", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=INK2)
    ref180(ax, y=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([SC[n][0] for n in names])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s); bars = 5–95% of Monte Carlo replicates, tick = median")
    h = [Line2D([], [], color=MUTED, lw=4.5, label="T: last person leaves the building (the problem's definition)"),
         Line2D([], [], color=MUTED, lw=4.5, alpha=0.45, label="T_ground: last person reaches street level")]
    ax.legend(handles=h, loc="lower right", fontsize=6.5)
    ax.set_title("Which T? Leaving the building vs. reaching the street (gap = queue for the exterior stairs)")
    return save(fig, outdir, "E10_T_building_vs_street")


def fig_cost_sensitivity(cs: Dict[str, pd.DataFrame], outdir):
    'Q5: share of cost draws in which each design is Pareto-optimal; one-at-a-time changes of the front.'
    style()
    fr, oat = cs["freq"].copy(), cs["oat"]
    fr = fr.sort_values(["share_on_front", "T"], ascending=[True, False]).tail(14)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    cols = [ORANGE if (r.led or r.barriers) else BLUE for r in fr.itertuples()]
    a1.barh(np.arange(len(fr)), fr["share_on_front"], color=cols, height=0.62)
    for i, r in enumerate(fr.itertuples()):
        tag = " (dominated for any cost)" if r.dominated_for_any_cost else ""
        a1.text(r.share_on_front + 0.01, i, f"{r.share_on_front:.2f}{tag}", va="center", fontsize=6.0, color=INK2)
    a1.set_yticks(np.arange(len(fr)))
    a1.set_yticklabels([f"{d}  ({t:.1f} s)" for d, t in zip(fr["design"], fr["T"])], fontsize=6.4)
    a1.set_xlim(0, 1.45)
    a1.set_xlabel("share of joint cost draws on the Pareto front")
    a1.grid(axis="y", visible=False)
    a1.set_title("(a) Which designs stay efficient", fontsize=8)
    params = list(dict.fromkeys(oat["parameter"]))
    yy = np.arange(len(params))[::-1]
    for y_, p in zip(yy, params):
        for lvl, mk in (("low", "<"), ("high", ">")):
            r = oat[(oat["parameter"] == p) & (oat["level"] == lvl)].iloc[0]
            col = RED if r.device_on_front else (AQUA if r.same_front_as_default else YELLOW)
            x0 = 0 if lvl == "low" else 1
            a2.scatter(x0, y_, marker=mk, s=40, color=col, zorder=3)
            a2.text(x0 + (-0.08 if lvl == "low" else 0.08), y_, f"{r.value:g}",
                    ha="right" if lvl == "low" else "left", va="center", fontsize=5.8, color=INK2)
    a2.set_yticks(yy)
    a2.set_yticklabels([p.replace("_", " ") for p in params], fontsize=6.2)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["low end", "high end"])
    a2.set_xlim(-0.8, 1.8)
    a2.grid(False)
    h = [Line2D([], [], marker="o", color=AQUA, ls="", label="same front as default"),
         Line2D([], [], marker="o", color=YELLOW, ls="", label="front changes (manager counts only)"),
         Line2D([], [], marker="o", color=RED, ls="", label="a device design joins the front")]
    a2.legend(handles=h, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1)
    a2.set_title("(b) One cost at a time", fontsize=8)
    fig.suptitle("Q5 cost sensitivity: costs varied over assumed ranges (config.COST_RANGES); T fixed from Monte Carlo",
                 x=0.01, ha="left", fontsize=8.4, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.75)
    return save(fig, outdir, "G4_cost_sensitivity")


def fig_wheelchair_mc(wmc: Dict[str, Dict], outdir):
    'wheelchair Monte Carlo: refuge and street times.'
    style()
    labs = list(wmc)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(labs) + 1.0))
    y = np.arange(len(labs))[::-1]
    for yy, lab in zip(y, labs):
        r = wmc[lab]
        for key, dy, col in (("T_refuge", 0.17, VIOLET), ("T_street", -0.17, MAGENTA), ("T", 0.0, MUTED)):
            a = np.asarray(r[key], float)
            a = a[np.isfinite(a)]
            if not len(a):
                continue
            q = np.percentile(a, [5, 50, 95])
            ax.plot([q[0], q[2]], [yy + dy] * 2, color=col, lw=3.6 if key != "T" else 1.4, solid_capstyle="butt")
            ax.plot([q[1]] * 2, [yy + dy - 0.1, yy + dy + 0.1], color=INK, lw=1.0)
        st = np.asarray(r["T_street"], float)
        ax.text(1.005, yy, f"street: median {np.nanmedian(st):.0f} s, P(≤240 s) {np.mean(st <= 240):.2f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color=INK2)
    ref180(ax, y=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s); bars = 5–95% of replicates, tick = median")
    h = [Line2D([], [], color=VIOLET, lw=3.6, label="wheelchair users reach the outdoor refuge"),
         Line2D([], [], color=MAGENTA, lw=3.6, label="... assisted down to the street (0.31 m/s)"),
         Line2D([], [], color=MUTED, lw=1.4, label="everyone: last person leaves the building")]
    ax.legend(handles=h, loc="lower right", fontsize=6.2)
    K = len(np.asarray(wmc[labs[0]]["T"]))
    ax.set_title(f"Wheelchair users under randomness (K = {K} each, common random numbers)")
    return save(fig, outdir, "G5_wheelchair_mc")


def fig_rowcap(rc: Dict[str, pd.DataFrame], outdir):
    'seat-row capacity on vs off.'
    style()
    d, m = rc["det"], rc["mc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    x = np.arange(len(d))
    cols = [SC[n][1] for n in d["scenario"]]
    a1.bar(x - 0.18, d["T_off"], width=0.34, color=cols, alpha=0.45, label="off (formal run)")
    a1.bar(x + 0.18, d["T_on"], width=0.34, color=cols, label="row-end capacity on")
    for xx, a, b_ in zip(x, d["T_off"], d["T_on"]):
        a1.text(xx, max(a, b_) + 4, f"{b_ - a:+.1f}", ha="center", fontsize=6.2)
    a1.set_xticks(x)
    a1.set_xticklabels([SC[n][0].replace("Q4 ", "") for n in d["scenario"]], rotation=35, ha="right", fontsize=6.2)
    a1.set_ylabel("deterministic T (s)")
    a1.legend(fontsize=6.3, loc="upper left")
    a1.grid(axis="x", visible=False)
    a1.set_title("(a) Deterministic", fontsize=8)
    yy = np.arange(len(m))[::-1]
    a2.errorbar(m["dT_mean"], yy, xerr=[m["dT_mean"] - m["dT_ci_lo"], m["dT_ci_hi"] - m["dT_mean"]], fmt="o",
                color=INK, ms=4, capsize=2)
    a2.axvline(0, color=AXIS, lw=0.8)
    a2.set_yticks(yy)
    a2.set_yticklabels([SC[n][0] for n in m["scenario"]], fontsize=6.5)
    a2.set_xlabel("paired ΔT, on − off (s), 95% CI")
    a2.grid(axis="y", visible=False)
    a2.set_title(f"(b) Monte Carlo, K = {int(m['K'].iloc[0])}", fontsize=8)
    fig.suptitle("Seat-row capacity check: row ends × 1.0 p/(m·s) × 0.55 m (row model of the web prototype)", x=0.01,
                 ha="left", fontsize=8.4, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.45)
    return save(fig, outdir, "F8_seat_row_capacity")


"""Schematic floor-plan geometry and plotting helpers. Coordinates are illustrative, not surveyed."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib.patches import Polygon, Rectangle, Wedge


# polygon per cell

SECTORS: Dict[str, Tuple[float, float, float, float]] = {}


def _sec(cid, r0, r1, a1, a2):
    SECTORS[cid] = (r0, r1, a1, a2)


for k, (a1, a2) in enumerate([(0, 40), (40, 140), (140, 180)]):
    _sec(f"O{k+1}", 7.2, 19.4, a1, a2)
for k, a in enumerate([4, 40, 75, 105, 140, 176]):
    _sec(f"A{k+1}", 19.6, 21.2, a - 2.4, a + 2.4)
for k, (a1, a2) in enumerate([(0, 60), (60, 120), (120, 180)]):
    _sec(f"M{k+1}", 21.5, 26.5, a1, a2)
for k, a in enumerate([3, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 177]):
    _sec(f"D{k+1}", 26.8, 28.4, a - 1.9, a + 1.9)
for k in range(10):
    _sec(f"C{k+1}", 28.6, 31.2, 18 * k, 18 * (k + 1))
for k, a in enumerate([27, 45, 135, 153]):
    _sec(f"S{k+1}", 31.4, 33.2, a - 1.7, a + 1.7)
for k, a in enumerate([81, 90, 99]):
    _sec(f"F{k+1}", 31.4, 33.1, a - 1.8, a + 1.8)
_sec("POR1", 33.6, 36.4, 8, 64)
_sec("POR2", 33.6, 36.4, 116, 172)
_sec("ST1", 26.5, 31.0, -15, -3)
_sec("ST2", 26.5, 31.0, 183, 195)
_sec("EXIT-Stage-W", 31.6, 34.2, -14, -4)
_sec("EXIT-Stage-E", 31.6, 34.2, 184, 194)
_sec("EXIT-Side-W", 37.0, 40.0, 30, 42)
_sec("EXIT-Side-E", 37.0, 40.0, 138, 150)
for k, (a1, a2) in enumerate([(0, 60), (60, 120), (120, 180)]):
    _sec(f"B{k+1}", 21.5, 27.0, a1, a2)
_sec("BS1@2F", 27.3, 30.0, 57, 63)
_sec("BS2@2F", 27.3, 30.0, 117, 123)

RECTS: Dict[str, Tuple[float, float, float, float]] = {   # (x0, y0, w, h)
    "FOY": (-9.0, -41.4, 18.0, 8.0),
    "BS1": (-12.6, -40.6, 3.2, 6.4),
    "BS2": (9.4, -40.6, 3.2, 6.4),
}
POLYS: Dict[str, np.ndarray] = {
    "EXIT-Main": np.array([(-6.5, -41.8), (6.5, -41.8), (8.5, -46.0), (-8.5, -46.0)]),

    "PORS": np.array([(-17.0, -48.0), (17.0, -48.0), (17.0, -49.6), (-17.0, -49.6)]),
}

FLOOR1 = ["O1", "O2", "O3", "A1", "A2", "A3", "A4", "A5", "A6", "M1", "M2", "M3",
          *[f"D{k}" for k in range(1, 10)], *[f"C{k}" for k in range(1, 11)],
          "S1", "S2", "S3", "S4", "F1", "F2", "F3", "FOY", "POR1", "POR2", "PORS",
          "BS1", "BS2", "ST1", "ST2",
          "EXIT-Main", "EXIT-Side-W", "EXIT-Side-E", "EXIT-Stage-W", "EXIT-Stage-E"]
FLOOR2 = ["B1", "B2", "B3", "BS1@2F", "BS2@2F", "ST1", "ST2", "EXIT-Stage-W", "EXIT-Stage-E"]

KIND_FILL = {
    "seat": "#cde2fb", "access": "#f6c3ad", "door": "#f6c3ad", "corridor": "#e1e0d9",
    "foyer": "#e1e0d9", "portico": "#bfe3bf", "stair": "#b4e6d2", "exit": "#0b0b0b",
}
KIND_EDGE = {"seat": "#2a78d6", "access": "#eb6834", "door": "#eb6834", "corridor": "#8a8984",
             "foyer": "#8a8984", "portico": "#008300", "stair": "#1baf7a", "exit": "#0b0b0b"}


def _pt(r, a):
    a = np.deg2rad(a)
    return np.array([-r * np.cos(a), -r * np.sin(a)])


def centroid(cid: str) -> np.ndarray:
    if cid in SECTORS:
        r0, r1, a1, a2 = SECTORS[cid]
        return _pt(0.5 * (r0 + r1), 0.5 * (a1 + a2))
    if cid in RECTS:
        x, y, w, h = RECTS[cid]
        return np.array([x + w / 2, y + h / 2])
    return POLYS[cid].mean(0)


def patch(cid: str, **kw):
    'matplotlib patch for a cell.'
    if cid in SECTORS:
        r0, r1, a1, a2 = SECTORS[cid]
        return Wedge((0, 0), r1, 180 + a1, 180 + a2, width=r1 - r0, **kw)
    if cid in RECTS:
        x, y, w, h = RECTS[cid]
        return Rectangle((x, y), w, h, **kw)
    return Polygon(POLYS[cid], closed=True, **kw)


def base_id(pid: str) -> str:
    return pid.split("@")[0]


def draw_context(ax, floor: int = 1, ink="#8a8984"):
    'stage, barrier ring, orientation labels.'
    ax.add_patch(Rectangle((-17.5, 0.0), 35.0, 6.0, facecolor="#f0efec", edgecolor="#c3c2b7", lw=0.6,
                           zorder=0))
    if floor == 1:
        ax.add_patch(Wedge((0, 0), 6.6, 180, 360, facecolor="#f0efec", edgecolor="#c3c2b7", lw=0.6,
                           zorder=0))
        th = np.linspace(np.pi, 2 * np.pi, 200)
        ax.plot(19.5 * np.cos(th), 19.5 * np.sin(th), color=ink, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0, 3.0, "STAGE", ha="center", va="center", fontsize=7, color=ink, zorder=2)


def setup_axes(ax, floor: int = 1):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    if floor == 1:
        ax.set_xlim(-42, 42); ax.set_ylim(-51, 8)
    else:
        ax.set_xlim(-36, 36); ax.set_ylim(-33, 8)


def draw_floor(ax, b, floor: int = 1, values: Optional[Dict[str, float]] = None, cmap=None,
               norm=None, labels: bool = True, label_size: float = 5.2, edges: bool = False,
               closed_style: bool = True, highlight: Optional[Dict[str, str]] = None,
               exit_text: Optional[Dict[str, str]] = None, lw: float = 0.5):
    """draw one floor.
thick outline for chosen cells
text next to exits"""
    setup_axes(ax, floor)
    draw_context(ax, floor)
    ids = FLOOR1 if floor == 1 else FLOOR2
    if edges:
        _draw_edges(ax, b, ids)
    for pid in ids:
        cid = base_id(pid)
        present = cid in b.cell
        kind = b.cell[cid].kind if present else _kind_guess(cid)
        if values is not None and present and cid in values and kind != "exit":
            fc = cmap(norm(values[cid]))
            ec = "#52514e"
        else:
            fc = KIND_FILL[kind] if values is None else ("#0b0b0b" if kind == "exit" else "#f7f7f5")
            ec = KIND_EDGE[kind] if values is None else "#a8a7a1"
        style = {}
        if not present and closed_style:
            fc, ec = "#ffffff", "#c3c2b7"
            style = dict(hatch="////")
        if cid in b.blocked:
            ec = "#e34948"
        p = patch(pid, facecolor=fc, edgecolor=ec, lw=lw, zorder=3, **style)
        ax.add_patch(p)
        if cid in b.blocked:
            c = centroid(pid)
            ax.plot(*c, marker="x", color="#e34948", ms=9, mew=2.2, zorder=6)
        if highlight and cid in highlight:
            ax.add_patch(patch(pid, facecolor="none", edgecolor=highlight[cid], lw=1.8, zorder=5))
        if labels and kind != "exit" and pid not in ("PORS",):
            c = centroid(pid)
            txt = cid
            dark = values is not None and present and cid in values and norm(values[cid]) > 0.55
            ax.text(c[0], c[1], txt, ha="center", va="center", fontsize=label_size,
                    color="#ffffff" if dark else "#0b0b0b", zorder=6)
    if exit_text is None and values is None and floor == 1:
        exit_text = {"EXIT-Main": "Main", "EXIT-Side-W": "Side-W", "EXIT-Side-E": "Side-E",
                     "EXIT-Stage-W": "Stage-W", "EXIT-Stage-E": "Stage-E"}
    if exit_text and floor == 1:
        for e, txt in exit_text.items():
            if e not in ids:
                continue
            c = centroid(e)
            dx, dy, ha, col = {"EXIT-Main": (0, -0.3, "center", "#ffffff"),
                               "EXIT-Side-W": (-2.2, -2.6, "right", "#0b0b0b"),
                               "EXIT-Side-E": (2.2, -2.6, "left", "#0b0b0b"),
                               "EXIT-Stage-W": (-1.9, 0.0, "right", "#0b0b0b"),
                               "EXIT-Stage-E": (1.9, 0.0, "left", "#0b0b0b")}[e]
            ax.text(c[0] + dx, c[1] + dy, txt, ha=ha, va="center", fontsize=6.2, color=col, zorder=7,
                    fontweight="bold" if e == "EXIT-Main" else "normal")
    if floor == 1 and labels and label_size >= 5 and "PORS" in b.cell:   # only on large maps
        c = centroid("PORS")
        ax.text(c[0], c[1] - 1.6, "PORS: south portico detour (used only if a side stair is blocked)",
                ha="center", va="top", fontsize=4.6, color="#52514e", zorder=6)
    if floor == 1 and "PORS" in ids and "PORS" in b.cell:
        # dashed connectors for the detour
        for por, x in (("POR1", -17.0), ("POR2", 17.0)):
            c = centroid(por)
            ax.plot([c[0], x], [c[1], -48.8], color="#008300", lw=0.6, ls=(0, (2, 2)), zorder=2)


def _kind_guess(cid):
    if cid.startswith("EXIT"):
        return "exit"
    if cid.startswith(("ST", "BS")):
        return "stair"
    if cid.startswith("POR"):
        return "portico"
    if cid.startswith("C"):
        return "corridor"
    return "door"


def _draw_edges(ax, b, ids):
    pid_of = {base_id(p): p for p in ids}
    for i, k in b.edges:
        if (i, k) in b.reverse or i not in pid_of or k not in pid_of:
            continue
        if ids is FLOOR2 and not (i.startswith("B") and len(i) == 2) and not i.startswith("ST"):
            continue                       # balcony: only links leaving B
        p0, p1 = centroid(pid_of[i]), centroid(pid_of[k])
        ax.annotate("", xy=p1, xytext=p0, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.45, shrinkA=2.5, shrinkB=2.5,
                                    mutation_scale=4.5, alpha=0.75))

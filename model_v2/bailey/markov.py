"""/ Closed-form absorbing Markov chain (no congestion) + lower bounds + LED plan.

Implements main-model sections 3.1–3.5:
    P = [[Q, R], [0, I]]
fundamental matrix
expected steps to absorption
absorption probabilities
expected number of people passing cell j"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Set

import networkx as nx
import numpy as np

from .building import Building, INF
from .config import Config
from .routing import Router


@dataclass
class MarkovClass:
    'one population class for the closed-form analysis.'
    name: str
    router: Router
    p: np.ndarray        # leave prob per step (nT,)
    x0: np.ndarray       #


def transition_matrix(b: Building, router: Router, p: np.ndarray) -> np.ndarray:
    """exit rows = identity.
dead-end rows get a self-loop."""
    assert b.ids[:b.nT] == b.transient, "transient cells must come first"
    nA = len(b.ids)
    s = router.split(None)                     # no queue term
    P = np.zeros((nA, nA))
    for n in range(b.nT):
        if s[n].sum() == 0:
            P[n, n] = 1.0
            continue
        P[n, n] = 1.0 - p[n]
        P[n, :] += p[n] * s[n]
    for m in range(b.nT, nA):
        P[m, m] = 1.0
    return P


def _reachable(P: np.ndarray, start: np.ndarray, nT: int) -> np.ndarray:
    seen = np.zeros(nT, bool)
    stack = list(np.flatnonzero(start[:nT] > 0))
    seen[stack] = True
    while stack:
        i = stack.pop()
        for k in np.flatnonzero(P[i, :nT] > 0):
            if not seen[k]:
                seen[k] = True
                stack.append(k)
    return seen


def absorbing_analysis(b: Building, cfg: Config, cls: MarkovClass) -> Dict[str, np.ndarray]:
    """N, t, B for one class.
    Inversion restricted to cells reachable from x0, which keeps I − Q invertible."""
    dt = cfg.phys.dt
    P = transition_matrix(b, cls.router, cls.p)
    nT = b.nT
    reach = _reachable(P, cls.x0, nT)
    Q = P[:nT, :nT][np.ix_(reach, reach)]
    R = P[:nT, nT:][reach]
    # no trapped mass among reachable cells
    if np.any(np.isclose(np.diag(Q), 1.0) & np.isclose(R.sum(1), 0.0)):
        bad = [b.transient[i] for i in np.flatnonzero(reach)[np.isclose(np.diag(Q), 1.0)]]
        raise ValueError(f"{cls.name}: mass trapped in {bad}")
    Nsub = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    N = np.zeros((nT, nT))
    N[np.ix_(reach, reach)] = Nsub
    t_steps = np.zeros(nT)
    t_steps[reach] = Nsub.sum(axis=1)
    B = np.zeros((nT, b.nE))
    B[reach] = Nsub @ R
    visits = cls.x0 @ N                        #
    passes = visits * cls.p                    # people passing each cell
    return dict(P=P, N=N, t_sec=t_steps * dt, B=B, passes=passes,
                mean_time=float(cls.x0 @ t_steps * dt / max(cls.x0.sum(), 1e-12)),
                exit_people=cls.x0 @ B)


def mixture_analysis(b: Building, cfg: Config, classes: Sequence[MarkovClass]) -> Dict:
    'See docs/IMPLEMENTATION.md for implementation details.'
    res = [absorbing_analysis(b, cfg, c) for c in classes]
    x0_tot = sum(c.x0 for c in classes)
    seats = b.seats()
    sidx = [b.transient.index(s) for s in seats]
    t_seat = np.zeros(len(seats))
    B_seat = np.zeros((len(seats), b.nE))
    for c, r in zip(classes, res):
        w = np.divide(c.x0[sidx], x0_tot[sidx], out=np.zeros(len(sidx)), where=x0_tot[sidx] > 0)
        t_seat += w * r["t_sec"][sidx]
        B_seat += w[:, None] * r["B"][sidx]
    exit_people = sum(r["exit_people"] for r in res)
    passes = sum(r["passes"] for r in res)
    n = x0_tot.sum()
    return dict(per_class={c.name: r for c, r in zip(classes, res)},
                seats=seats, t_seat=t_seat, B_seat=B_seat,
                exit_people=exit_people, passes=passes,
                mean_time=float(sum(r["mean_time"] * c.x0.sum() for c, r in zip(classes, res)) / n))


# ---------------------------------------------------------------------------
# Max-flow capacity into exits
# ---------------------------------------------------------------------------
def max_flow_rates(b: Building, removed: Set[str] = frozenset()) -> Dict[str, float]:
    'Node-split graph: capacity of (i_in → i_out) = inflow cap μ_i.'
    G = nx.DiGraph()
    for c in b.cells:
        if c.id in removed:
            continue
        if c.inflow_cap < INF:
            G.add_edge(c.id + "|in", c.id + "|out", capacity=c.inflow_cap)
        else:
            G.add_edge(c.id + "|in", c.id + "|out")
    for i, k in b.edges:
        if i in removed or k in removed:
            continue
        cap = b.edge_cap.get((i, k), INF)
        if cap < INF:
            G.add_edge(i + "|out", k + "|in", capacity=cap)
        else:
            G.add_edge(i + "|out", k + "|in")
    for s in b.seats():
        G.add_edge("SRC", s + "|in")
    out = {}
    exits = [e for e in b.exits if e not in removed]
    for e in exits:
        H = G.copy()
        H.add_edge(e + "|out", "SNK")
        out[e] = float(nx.maximum_flow_value(H, "SRC", "SNK"))
    H = G.copy()
    for e in exits:
        H.add_edge(e + "|out", "SNK")
    out["_total"] = float(nx.maximum_flow_value(H, "SRC", "SNK"))
    return out


def bottleneck_lower_bound(b: Building, d: np.ndarray, n: float, prep_min: float,
                           removed: Set[str] = frozenset()) -> Dict[str, float]:
    'See docs/IMPLEMENTATION.md for implementation details.'
    mf = max_flow_rates(b, removed)
    exits = [e for e in b.exits if e not in removed]
    sidx = [b.idx[s] for s in b.seats()]
    t0 = {e: prep_min + float(np.min(d[sidx, b.exits.index(e)])) for e in exits}
    mu = {e: mf[e] for e in exits}

    def served(T):
        return sum(mu[e] * max(0.0, T - t0[e]) for e in exits)

    lo, hi = 0.0, 1e5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if served(mid) < n else (lo, mid)
    lb1 = hi
    lb2 = min(t0.values()) + n / mf["_total"]
    return dict(LB=max(lb1, lb2), LB_per_exit=lb1, LB_joint=lb2,
                mu_total=mf["_total"], **{f"mu[{e}]": mu[e] for e in exits},
                **{f"t0[{e}]": t0[e] for e in exits})


# ---------------------------------------------------------------------------
#
# System-optimal assignment (LP): split each seat zone over exits so the last
#
# It also yields a stronger lower bound, and serves as the LED plan.
# ---------------------------------------------------------------------------
def _seat_flow_graph(b: Building, removed: Set[str]):
    G = nx.DiGraph()
    for c in b.cells:
        if c.id in removed:
            continue
        if c.inflow_cap < INF:
            G.add_edge(c.id + "|in", c.id + "|out", capacity=c.inflow_cap)
        else:
            G.add_edge(c.id + "|in", c.id + "|out")
    for i, k in b.edges:
        if i in removed or k in removed:
            continue
        cap = b.edge_cap.get((i, k), INF)
        if cap < INF:
            G.add_edge(i + "|out", k + "|in", capacity=cap)
        else:
            G.add_edge(i + "|out", k + "|in")
    return G


def lp_assignment(b: Building, d: np.ndarray, seat_counts: Dict[str, float], prep_min: float,
                  removed: Set[str] = frozenset()) -> Dict:
    """objective : min T
constraints:
everyone assigned
                 max-flow rate from zones Z (respects every door/stair on the way)
earliest possible arrival
    The optimal T is a routing-independent lower bound (fluid relaxation)."""
    from itertools import combinations
    from scipy.optimize import linprog

    zones = [z for z in b.seats() if seat_counts.get(z, 0) > 0]
    exits = [e for e in b.exits if e not in removed]
    nz, ne = len(zones), len(exits)
    G0 = _seat_flow_graph(b, set(removed))
    for e in exits:
        G0.add_edge(e + "|out", "SNK_" + e)
        G0.add_edge("SNK_" + e, "SNK_ALL")

    def mflow(Z, sink):
        H = G0.copy()
        for z in Z:
            H.add_edge("SRC", z + "|in")
        return float(nx.maximum_flow_value(H, "SRC", sink))

    col = lambda zi, ei: zi * ne + ei
    nv = nz * ne + 1                      # last variable is T
    A_ub, b_ub = [], []
    for r in range(1, nz + 1):
        for Z in combinations(range(nz), r):
            Zn = [zones[i] for i in Z]
            for ei, e in enumerate(exits):
                mu = mflow(Zn, "SNK_" + e)
                t0 = prep_min + min(d[b.idx[z], b.exits.index(e)] for z in Zn)
                row = np.zeros(nv)
                for zi in Z:
                    row[col(zi, ei)] = 1.0
                if mu <= 1e-12 or not np.isfinite(t0):
                    for zi in Z:              # unreachable → forbidden
                        A_ub.append(np.eye(nv)[col(zi, ei)]); b_ub.append(0.0)
                    continue
                row[-1] = -mu
                A_ub.append(row); b_ub.append(-mu * t0)
            mu_all = mflow(Zn, "SNK_ALL")
            t0_all = prep_min + min(np.nanmin(np.where(np.isfinite(d[b.idx[z]]), d[b.idx[z]], np.nan))
                                    for z in Zn)
            row = np.zeros(nv)
            for zi in Z:
                for ei in range(ne):
                    row[col(zi, ei)] = 1.0
            row[-1] = -mu_all
            A_ub.append(row); b_ub.append(-mu_all * t0_all)
    A_eq = np.zeros((nz, nv)); b_eq = np.zeros(nz)
    for zi, z in enumerate(zones):
        for ei in range(ne):
            A_eq[zi, col(zi, ei)] = 1.0
        b_eq[zi] = seat_counts[z]
    cvec = np.zeros(nv); cvec[-1] = 1.0
    sol = linprog(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * nv, method="highs")
    if not sol.success:
        raise RuntimeError("LP failed: " + sol.message)
    T_star = float(sol.x[-1])
    #
    # Stage 2: keep T (+0.5 s slack) and minimise total walking, which removes
    # degenerate solutions such as sending the west seats to the east exits.
    c2 = np.zeros(nv)
    for zi, z in enumerate(zones):
        for ei, e in enumerate(exits):
            dd = d[b.idx[z], b.exits.index(e)]
            c2[col(zi, ei)] = dd if np.isfinite(dd) else 1e6
    bounds2 = [(0, None)] * (nv - 1) + [(0, T_star + 0.5)]
    sol2 = linprog(c2, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds2, method="highs")
    if sol2.success:
        sol = sol2
    y = sol.x[:-1].reshape(nz, ne)
    y[y < 1e-6] = 0.0
    return dict(T_lp=T_star, zones=zones, exits=exits, y=y,
                n_constraints=len(A_ub))

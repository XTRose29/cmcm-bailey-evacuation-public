"""How long people stay in a cell, and where they go next.

Implements main-model section 3.4(3):
free walking time through cell i
probability of leaving per step
logit split among neighbours

utility of stepping into neighbour k:
free-walk seconds from entering k to exit e
visible waiting time at k

progress rule:
    Only neighbours that get closer to at least one known exit are allowed,
    so the fluid does not wander back and forth along the corridor."""
from __future__ import annotations

import heapq
from typing import Dict, Iterable, Optional, Set

import numpy as np

from .building import Building, INF
from .config import Behaviour, Config


def cell_times(b: Building, cfg: Config, v: float, beh: Behaviour) -> np.ndarray:
    """free walking time per transient cell (seconds).
row part walked at v·row_factor
at v·stair_factor"""
    ph = cfg.phys
    vv = v * beh.speed_factor
    stair_factor = ph.stair_factor_normal if v == ph.v_normal else ph.stair_factor
    tau = np.zeros(b.nT)
    for n, cid in enumerate(b.transient):
        c = b.cell[cid]
        if c.kind == "seat" and not beh.uses_stairs and cid.startswith("M"):
            # Rear wheelchair platform: no travel through seat rows. [ESTIMATE]
            t = cfg.geo.l_wheelchair_access / vv
        elif c.kind == "seat":
            t = c.row_part / (vv * ph.row_factor) + max(c.length - c.row_part, 0.0) / vv
        elif c.stair:
            t = c.length / (vv * stair_factor)
        else:
            t = c.length / vv
        tau[n] = t
    return tau


def leave_prob(tau: np.ndarray, dt: float) -> np.ndarray:
    'See docs/IMPLEMENTATION.md for implementation details.'
    return np.minimum(1.0, dt / np.maximum(tau, 1e-12))


def exit_distances(b: Building, tau: np.ndarray, removed: Iterable[str] = ()) -> np.ndarray:
    """Shortest free-walk time from entering `node` to each exit (includes node's own τ).
    Cells in `removed` are treated as impassable (known blockage, stairs for wheelchairs).
returns inf where unreachable."""
    removed = set(removed)
    w = {cid: tau[n] for n, cid in enumerate(b.transient)}
    for e in b.exits:
        w[e] = 0.0
    rev: Dict[str, list] = {c: [] for c in b.ids}
    for i, k in b.edges:
        rev[k].append(i)
    d = np.full((len(b.ids), b.nE), INF)
    for col, e in enumerate(b.exits):
        if e in removed:
            continue
        dist = {e: 0.0}
        pq = [(0.0, e)]
        while pq:
            dk, k = heapq.heappop(pq)
            if dk > dist.get(k, INF):
                continue
            for i in rev[k]:
                if i in removed:
                    continue
                nd = dk + w[i]
                if nd < dist.get(i, INF):
                    dist[i] = nd
                    heapq.heappush(pq, (nd, i))
        for cid, val in dist.items():
            d[b.idx[cid], col] = val
    return d


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


class Router:
    """Route-choice rule for one population variant. Static parts are precomputed;
    only the queue term changes each step."""

    def __init__(self, b: Building, beh: Behaviour, tau: np.ndarray,
                 removed: Set[str], plan: Optional[Dict[str, str]] = None,
                 node_behaviours: Optional[Dict[str, Behaviour]] = None):
        self.b, self.beh = b, beh
        self.removed = set(removed)
        self.d = exit_distances(b, tau, self.removed)          # (nAll, nE)
        nT, nA = b.nT, len(b.ids)
        main = np.array([1.0 if b.cell[e].exit_group == "EXIT-Main" else 0.0 for e in b.exits])

        # allowed neighbours (mask over all cells)
        self.allowed = np.zeros((nT, nA), dtype=bool)
        # static utility per (i, k): logsum over exits
        self.static_u = np.full((nT, nA), -np.inf)
        self.beta_queue = np.full(nT, beh.beta_queue)
        for n, i in enumerate(b.transient):
            local = (node_behaviours or {}).get(i, beh)
            self.beta_queue[n] = local.beta_queue
            di = self.d[b.idx[i]]
            plan_vec = np.zeros(b.nE)
            target = plan.get(i) if plan else None
            for e in ((target,) if isinstance(target, str) else (target or ())):
                if e in b.exits:          # ties all count
                    plan_vec[b.exits.index(e)] = 1.0
            cands = [k for k in b.out[i] if k not in self.removed]
            fwd = [k for k in cands if (i, k) not in b.reverse]
            back = [k for k in cands if (i, k) in b.reverse]
            closer = lambda ks: [k for k in ks if np.any(self.d[b.idx[k]] < di - 1e-9)]
            prog = closer(fwd)
            if not prog:   # turn back only if nothing forward works
                prog = closer(back)
            if not prog:   # last resort
                prog = [k for k in cands if np.isfinite(self.d[b.idx[k]]).any()]
            for k in prog:
                kk = b.idx[k]
                score = (-local.beta_dist * self.d[kk] + local.beta_habit * main
                         + local.beta_guide * plan_vec)
                score = np.where(np.isfinite(self.d[kk]), score, -np.inf)
                self.allowed[n, kk] = True
                self.static_u[n, kk] = _logsumexp(score[None, :], axis=1)[0]
            if not self.allowed[n].any() and i not in self.removed:
                #
                # dead end for this variant; harmless as long as nobody enters it
                pass

    def split(self, wait: Optional[np.ndarray] = None) -> np.ndarray:
        """Split fractions; rows sum to 1 (or 0 for dead ends).
visible waiting seconds per cell, shape (nAll,)"""
        u = self.static_u.copy()
        if wait is not None:
            u = u - self.beta_queue[:, None] * wait[None, :]
        m = np.max(u, axis=1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        e = np.where(self.allowed, np.exp(u - m), 0.0)
        tot = e.sum(axis=1, keepdims=True)
        return np.divide(e, tot, out=np.zeros_like(e), where=tot > 0)


def plan_nearest(b: Building, d: np.ndarray) -> Dict[str, str]:
    """Default instruction: nearest exit(s) from each cell. Ties are kept as a tuple,
    otherwise the symmetric building would be biased toward the first (west) exit."""
    out = {}
    for i in b.transient:
        row = d[b.idx[i]]
        if np.isfinite(row).any():
            m = np.min(row)
            out[i] = tuple(e for e, val in zip(b.exits, row) if val <= m + 1e-6)
    return out


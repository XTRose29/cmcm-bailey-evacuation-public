"""-----------------------------------------------------------------
    The chain gives directions (s_ik), queueing gives the junction limits (μ_j),
    the flow ODE does the bookkeeping.
Capacity is applied once (doc 4.6): the engine's min(demand, μ, supply) already uses μ_j,
so capacity() returns μ_j unchanged; this module *analyses* the queue at every junction.

What a junction j is
------------------------------------
    cells with a finite inflow capacity: the queue waiting to enter j
    links with their own capacity (main doors, side stairs, stage exits, mezzanine stage doors)

------------------------------------------------------------------
cumulative departures
          max FIFO wait = largest horizontal gap between the cumulative curves
  M/M/1 reference with the period-average λ̄: W_q = ρ̄/(μ − λ̄), undefined for ρ̄ ≥ 1."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

INF = float("inf")


class QueueModel:
    'interface.'
    name = "base"

    def capacity(self, cell_id: str, base_cap: float, t: float) -> float:
        raise NotImplementedError

    def edge_capacity(self, edge, base_cap: float, t: float) -> float:
        raise NotImplementedError

    def analyze(self, result) -> Dict:
        raise NotImplementedError


# readable junction names for the paper
EDGE_LABEL = {
    ("FOY", "EXIT-Main"): "Main doors (5×1.4 m)",
    ("POR1", "EXIT-Side-W"): "W side stair",
    ("POR2", "EXIT-Side-E"): "E side stair",
    ("ST1", "EXIT-Stage-W"): "W stage exit",
    ("ST2", "EXIT-Stage-E"): "E stage exit",
    ("M1", "ST1"): "M1→ST1 stage door",
    ("M3", "ST2"): "M3→ST2 stage door",
}


def junctions(b) -> List[Dict]:
    'list junctions and the edges that feed them.'
    out = []
    for c in b.cells:
        if c.kind == "exit" or not np.isfinite(c.inflow_cap):
            continue
        cols = [m for m, (i, k) in enumerate(b.edges) if k == c.id]
        if cols:
            out.append(dict(id=c.id, label=c.id, mu=float(c.inflow_cap), cols=cols, kind=c.kind))
    for (i, k), cap in b.edge_cap.items():
        if (i, k) in b.edges and np.isfinite(cap):
            m = b.edges.index((i, k))
            out.append(dict(id=f"{i}→{k}", label=EDGE_LABEL.get((i, k), f"{i}→{k}"),
                            mu=float(cap), cols=[m], kind="link"))
    return out


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def fifo_wait(t: np.ndarray, A: np.ndarray, D: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """FIFO waiting time of the n-th person = D⁻¹(n) − A⁻¹(n) (horizontal gap of the N-curves).
wait for someone arriving at each time t."""
    w = np.zeros_like(t)
    if A[-1] <= tol:
        return w
    #
    idx = np.searchsorted(D, A - tol, side="left")
    idx = np.minimum(idx, len(t) - 1)
    w = np.maximum(t[idx] - t, 0.0)
    w[A <= tol] = 0.0
    return w


class FlowQueueModel(QueueModel):
    'Section 4: deterministic capacity (applied once) + per-junction queue analysis.'
    name = "flow queue analysis (Section 4)"

    def __init__(self, smooth_s: float = 5.0):
        self.smooth_s = smooth_s

    def capacity(self, cell_id, base_cap, t):
        return base_cap

    def edge_capacity(self, edge, base_cap, t):
        return base_cap

    def analyze(self, result) -> Dict:
        r = result
        b = r.b
        dt = r.meta["dt"]
        t = r.t
        nwin = max(1, int(round(self.smooth_s / dt)))
        rows, series = [], {}
        for j in junctions(b):
            if j["id"] in b.blocked:
                continue
            cols = j["cols"]
            a = r.edge_arr[:, cols].sum(1)              # arrivals per step
            d = r.flows[:, cols].sum(1)                 # departures per step
            A = np.cumsum(a)
            Dc = np.cumsum(d)
            Q = r.edge_queue[:, cols].sum(1)
            lam = _smooth(np.maximum(a, 0) / dt, nwin)  # λ_j(t)
            dep = _smooth(d / dt, nwin)
            mu = j["mu"]
            rho = lam / mu
            passed = float(Dc[-1])
            if passed < 0.5:
                continue
            wait_ps = float(Q.sum() * dt)               #
            W_mean = wait_ps / passed                   # Little's law
            w_fifo = fifo_wait(t, A, Dc)
            busy = np.flatnonzero((A > 0.5) & (Dc < passed - 0.5))
            t_on = float(t[busy[0]]) if busy.size else np.nan
            t_off = float(t[busy[-1]]) if busy.size else np.nan
            active = (t_off - t_on) if busy.size else 0.0
            lam_bar = passed / active if active > 0 else np.nan
            rho_bar = lam_bar / mu if active > 0 else np.nan
            Wq_mm1 = rho_bar / (mu - lam_bar) if (active > 0 and rho_bar < 1) else INF
            over = float((rho > 1.0).sum() * dt)
            rows.append(dict(junction=j["id"], label=j["label"], kind=j["kind"], mu=mu,
                             passed=passed, peak_queue=float(Q.max()),
                             t_peak=float(t[int(np.argmax(Q))]), wait_person_s=wait_ps,
                             W_mean=W_mean, W_max=float(w_fifo.max()),
                             peak_rho=float(rho.max()), time_rho_gt1=over,
                             t_first=t_on, t_last=t_off, lam_bar=lam_bar, rho_bar=rho_bar,
                             Wq_mm1=Wq_mm1, saturated_share=float((dep >= 0.95 * mu).sum() * dt
                                                                  / max(active, dt))))
            series[j["id"]] = dict(t=t, A=A, D=Dc, Q=Q, lam=lam, dep=dep, rho=rho, w=w_fifo, mu=mu,
                                   label=j["label"])
        df = pd.DataFrame(rows).sort_values("wait_person_s", ascending=False).reset_index(drop=True)
        return dict(table=df, series=series, status="ok")


class CapacityOnlyPlaceholder(FlowQueueModel):
    'kept for backward compatibility.'
    name = "flow queue analysis (Section 4)"


def mm1_wq(rho: np.ndarray, mu: float) -> np.ndarray:
    'steady-state wait.'
    rho = np.asarray(rho, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rho < 1, rho / (mu * (1 - rho)), INF)

"""Integer population simulation on the Bailey Hall network.
people are indivisible; rates may be real.
Deterministic runs use finite reaction-time quantiles and balanced integer moves.
Monte Carlo runs sample person reactions, walking-stage transitions, route choices,
and passage service. The analytic Markov chain remains an expectation reference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .building import Building, INF
from .config import Behaviour, Config
from .plugins.queueing import CapacityOnlyPlaceholder, QueueModel
from .plugins.randomness import MeanFieldPlaceholder, RandomnessModel
from .routing import Router, cell_times, leave_prob
from .integer import ENGINE_VERSION, ServiceCredit, RouteAllocator, integer_classes, limit_counts


JUNCTION_KINDS = ("access", "door", "stair")
'junction kinds of doc 4.2 (A, D, F, S, BS, ST).'


@dataclass
class PopClass:
    'one population class.'
    name: str
    beh: Behaviour
    seat_counts: Dict[str, int]                 # persons per seat zone
    plan: Optional[Dict[str, str]] = None       # instruction map cell → exit
    node_behaviours: Optional[Dict[str, Behaviour]] = None  # local manager effect at current decision node


@dataclass
class SimResult:
    b: Building
    t: np.ndarray               # time at end of each step
    remaining: np.ndarray       # persons not yet at street level
    remaining_inside: np.ndarray  #
    occ: np.ndarray             # occupancy; exit cols cumulative
    queue_in: np.ndarray        # queue to enter each cell
    flows: np.ndarray           # persons per edge per step
    edges: List[tuple]
    exit_cum: np.ndarray        # (steps, nE)
    class_exit: Dict[str, np.ndarray]   # final exit use per class
    T: float                    #
    T_ground: float             # last person reaches street level
    mean_time: float            # mean building-egress time per person
    meta: Dict = field(default_factory=dict)
    #
    edge_arr: Optional[np.ndarray] = None     # persons joining queue i→k
    edge_queue: Optional[np.ndarray] = None   # queue i→k at step end
    class_T: Dict[str, float] = field(default_factory=dict)  # per-class egress time

    def exit_groups(self) -> Dict[str, int]:
        'totals per Definition exit group.'
        out = {g: 0 for g in self.b.exit_groups}
        for m, e in enumerate(self.b.exits):
            out[self.b.cell[e].exit_group] += int(self.exit_cum[-1, m])
        return out

    def time_to_fraction(self, frac: float) -> float:
        'time until a fraction has left the building.'
        n = self.meta["n"]
        hit = np.flatnonzero(n - self.remaining_inside >= frac * n - 1e-9)
        return float(self.t[hit[0]]) if hit.size else INF


def simulate(b: Building, cfg: Config, classes: Sequence[PopClass], v: float,
             randomness: Optional[RandomnessModel] = None,
             queue_model: Optional[QueueModel] = None,
             discovery: str = "known", broadcast_delay: float = 20.0,
             capacities: bool = True, t_max: Optional[float] = None,
             threshold: Optional[float] = None, walk_model: Optional[str] = None) -> SimResult:
    """
    Integer simulation. Monte Carlo samples Erlang/geometric walking stages.
    Deterministic Erlang-reference walking takes ceil(tau/dt) steps; geometric
    reference runs carry fractional transition work and move only whole people. Capacities are
    real-valued rates with bounded service credit. Storage is floored to people.
    The legacy threshold argument is ignored: clearance always requires zero.
    Discovery modes: known, arrival, or arrival plus a delayed broadcast.
    """
    ph = cfg.phys
    dt = ph.dt
    t_max = ph.t_max if t_max is None else t_max
    # threshold is retained only for old callers; exact zero is always required.
    classes = integer_classes(classes)
    if not classes or dt <= 0 or t_max <= 0:
        raise ValueError("Need classes, positive dt and positive t_max")
    randomness = randomness or MeanFieldPlaceholder((0.0, 0.0))
    queue_model = queue_model or CapacityOnlyPlaceholder()
    walk_model = walk_model or ph.walk_model
    nT, nA, nE = b.nT, len(b.ids), b.nE
    blocked = set(b.blocked)
    hidden = bool(blocked) and discovery != "known"

    # informed & uninformed variants ----------
    variants = []   # (class_name, router, tau, informed: bool)
    twin = {}       # uninformed index -> informed index
    for c in classes:
        stairs = {x.id for x in b.cells if x.stair} if not c.beh.uses_stairs else set()
        tau = cell_times(b, cfg, v, c.beh)
        r_inf = Router(b, c.beh, tau, stairs | blocked, c.plan, c.node_behaviours)
        if hidden:
            r_unk = Router(b, c.beh, tau, stairs, c.plan, c.node_behaviours)
            variants.append((c.name, r_unk, tau, False))
            variants.append((c.name, r_inf, tau, True))
            twin[len(variants) - 2] = len(variants) - 1
        else:
            variants.append((c.name, r_inf, tau, True))
    V = len(variants)
    motion_rng = randomness.rng_motion
    #
    cls_delay = {c.name: c.beh.delay for c in classes}
    delays = np.array([cls_delay[v_[0]] for v_ in variants])

    # walking phases per cell ----------
    taus = np.array([v_[2] for v_ in variants])                       # (V, nT)
    if walk_model == "geometric":
        m_ph = np.ones((V, nT), dtype=int)
        p_ph = leave_prob(taus, dt)
    elif walk_model == "erlang":
        m_ph = np.maximum(1, np.floor(taus / dt + 1e-9)).astype(int)
        p_ph = np.minimum(1.0, m_ph * dt / np.maximum(taus, 1e-12))
    else:
        raise ValueError(walk_model)
    if walk_model == "erlang" and motion_rng is None:
        # Deterministic reference: each whole person walks for ceil(tau/dt) steps.
        # Fractional per-phase rounding would otherwise double a lone person's time.
        m_ph = np.maximum(1, np.ceil(taus / dt - 1e-10)).astype(int)
        p_ph = np.ones_like(taus)
    M = int(m_ph.max())
    h_idx = np.arange(M)[None, None, :]
    active = h_idx < m_ph[:, :, None]                                 # existing phases
    last = h_idx == (m_ph[:, :, None] - 1)                            # last phase

    # initial state ----------
    x0 = np.zeros((V, nT), dtype=np.int64)
    vi = 0
    for c in classes:
        for s, cnt in c.seat_counts.items():
            x0[vi, b.transient.index(s)] += cnt
        vi += 2 if hidden else 1
    n_total = int(x0.sum())
    if n_total == 0:
        raise ValueError("Population must be positive")
    seated = x0.copy()
    released = np.zeros((V, nT), dtype=np.int64)
    walk = np.zeros((V, nT, M), dtype=np.int64)
    queue = np.zeros((V, nT, nA), dtype=np.int64)
    done = np.zeros((V, nE), dtype=np.int64)

    # Assign each sampled reaction to exactly one class.
    starts = {}
    order_rng = np.random.default_rng(ph.population_order_seed)
    for n in np.flatnonzero(x0.sum(0)):
        times = randomness.reaction_times(b.transient[n], int(x0[:, n].sum()))
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Reaction times must be finite and nonnegative")
        # Fixed permutation avoids assigning all early quantiles to one group.
        times = times[order_rng.permutation(len(times))]
        offset = 0
        for vv in range(V):
            count = int(x0[vv, n])
            if count:
                starts[vv, n] = np.sort(times[offset:offset + count] + delays[vv])
                offset += count
    motion_rng = randomness.rng_motion
    route_allocator = RouteAllocator((V, nT, nA), motion_rng)
    phase_credit = np.zeros_like(walk, dtype=float)
    node_credit = ServiceCredit((nA,))
    edge_credit = ServiceCredit((nT, nA))
    row_credit = ServiceCredit((nT,))

    # static capacities ----------
    base_in = np.array([b.cell[c].inflow_cap for c in b.ids])
    n_max = np.floor(np.array([b.cell[c].n_max for c in b.ids]))
    edge_cap_mat = np.full((nT, nA), INF)
    for (i, k), cap in b.edge_cap.items():
        if i in b.idx and k in b.idx and b.idx[i] < nT:
            edge_cap_mat[b.idx[i], b.idx[k]] = cap
    cap_edges = [(i, k) for i, k in zip(*np.nonzero(np.isfinite(edge_cap_mat)))]
    seat_idx = [n for n in range(nT) if b.cell[b.transient[n]].kind == "seat"]
    out_cap = np.array([b.cell[c].outflow_cap for c in b.transient])
    row_capped = [n for n in seat_idx if np.isfinite(out_cap[n])]
    #
    mu_out = np.zeros(nA)
    for n, i in enumerate(b.ids):
        tot = 0.0
        for k in b.out[i]:
            tot += min(b.edge_cap.get((i, k), INF), b.cell[k].inflow_cap)
        mu_out[n] = tot if np.isfinite(tot) and tot > 0 else cfg.phys.q * max(b.cell[i].width, 1.0)

    inside = np.array([not b.cell[c].outside for c in b.transient])    # (nT,)
    steps = int(np.ceil(t_max / dt))
    rec_t, rec_rem, rec_in, rec_occ, rec_q, rec_f, rec_exit = [], [], [], [], [], [], []
    rec_ea, rec_eq, rec_cls = [], [], []
    e_i = np.array([b.idx[i] for i, _ in b.edges])                     # edge tails
    e_k = np.array([b.idx[k] for _, k in b.edges])                     # edge heads
    #
    # Junctions of doc 4.2 (doors, access points, stairs) draw S_j in the stochastic version;
    # corridor and portico segments are passages and keep their deterministic capacity.
    rand_cells = np.array([np.isfinite(base_in[n]) and b.cell[c].kind in JUNCTION_KINDS
                           for n, c in enumerate(b.ids)])
    T = INF
    T_ground = INF
    broadcast_done = False
    area = 0.0       # ∫ persons inside dt  → mean building-egress time
    area_all = 0.0   # ∫ persons not at street dt → mean time to street

    for step in range(steps):
        t0 = step * dt
        t1 = t0 + dt
        # release from seats
        for vv in range(V):
            for n in seat_idx:
                if x0[vv, n] > 0:
                    target = int(np.searchsorted(starts[vv, n], t1, side="right"))
                    rel = max(0, target - released[vv, n])
                    rel = min(rel, seated[vv, n])
                    seated[vv, n] -= rel
                    released[vv, n] += rel
                    walk[vv, n, 0] += rel

        # current occupancy and queues
        occ_T = seated.sum(0) + walk.sum(axis=(0, 2)) + queue.sum(axis=(0, 2))   # (nT,)
        occ = np.zeros(nA)
        occ[:nT] = occ_T
        L_in = queue.sum(axis=(0, 1))                                      #
        wait = np.divide(L_in, base_in, out=np.zeros(nA), where=np.isfinite(base_in) & (base_in > 0))
        wait += occ / mu_out
        wait[nT:] = 0.0

        # walkers reach the exit, choose, and queue
        q_before = queue.sum(0)                                            # for arrivals
        splits = [router.split(wait) for (_, router, _, _) in variants]   # each (nT, nA)
        has_out = np.array([s_.sum(1) > 0 for s_ in splits])               # (V, nT)
        if motion_rng is None:
            work = walk * p_ph[:, :, None] + phase_credit
            out = np.minimum(walk, np.floor(work + 1e-10).astype(np.int64)) * active
            phase_credit = np.clip(work - out, 0.0, 1.0 - 1e-12)
            phase_credit[walk == 0] = 0.0
        else:
            out = motion_rng.binomial(walk, p_ph[:, :, None]) * active
        out_last = out * last * has_out[:, :, None]                        # reach the exit
        moving = out * (~last)                                             # next phase
        walk -= out_last + moving
        walk[:, :, 1:] += moving[:, :, :-1]
        arrive = out_last.sum(2)                                           # (V, nT)
        for vv in range(V):
            queue[vv] += route_allocator.split(vv, arrive[vv], splits[vv])

        # discovery of the blockage
        if hidden:
            do_broadcast = discovery == "broadcast" and not broadcast_done and t1 >= broadcast_delay
            for u, k_inf in twin.items():
                s_inf = splits[k_inf]
                ok = s_inf.sum(1) > 0     # cells with a route once informed
                if do_broadcast:
                    # everyone is told
                    seated[k_inf] += seated[u]; seated[u] = 0
                    x0[k_inf] += x0[u]; released[k_inf] += released[u]
                    x0[u] = 0; released[u] = 0
                    walk[k_inf] += walk[u]; walk[u] = 0
                    phase_credit[k_inf] = np.maximum(phase_credit[k_inf], phase_credit[u])
                    phase_credit[u] = 0
                    for n in seat_idx:
                        if (u, n) in starts:
                            starts[k_inf, n] = starts.pop((u, n))
                    qtot = queue[u].sum(1) * ok
                    queue[k_inf] += route_allocator.split(k_inf, qtot, s_inf)
                    queue[u] *= (~ok)[:, None]          #
                else:                                   # people with no alternative stay put
                    # discovered at the blocked door
                    for k in blocked:
                        if k not in b.idx:
                            continue
                        kk = b.idx[k]
                        m = queue[u][:, kk] * ok
                        if m.sum() > 0:
                            queue[u][:, kk] -= m
                            queue[k_inf] += route_allocator.split(k_inf, m, s_inf)
            if do_broadcast:
                broadcast_done = True
        #
        # Section 4: net persons joining each edge queue this step (includes re-routed people)
        A_step = queue.sum(0) - q_before                                   # (nT, nA)

        # capacities and flows
        D = queue.copy()                                                   # (V, nT, nA)
        if capacities:
            mult = randomness.capacity_multiplier(t0)
            tot_edge = D.sum(0)                                            # (nT, nA)
            ecap = edge_cap_mat.copy()
            for (ei, ek) in cap_edges:      # only a few edges carry their own cap
                ecap[ei, ek] = queue_model.edge_capacity((b.ids[ei], b.ids[ek]), edge_cap_mat[ei, ek], t0)
            ecap = ecap * mult * dt
            cap_in = np.array([queue_model.capacity(b.ids[k], base_in[k], t0) for k in range(nA)])
            cap_in = cap_in * mult * dt
            #
            #
            #
            #
            #
            # Section 5: realised service S_j(t). One draw per physical move (doc 4.6):
            # junction cells draw S_j; a capped link into a junction reuses that junction's
            # factor S_j/(μ_jΔt); only links into an exit draw their own.
            eps = np.ones(nA)
            if rand_cells.any():
                exp_j = cap_in[rand_cells]
                s_j = randomness.service(exp_j, t0, "cell")
                eps[rand_cells] = np.divide(s_j, exp_j, out=np.ones_like(exp_j), where=exp_j > 0)
                cap_in[rand_cells] = s_j
            if cap_edges:
                ce = tuple(np.array(cap_edges).T)
                to_exit = ce[1] >= nT
                vals = ecap[ce] * eps[ce[1]]
                if to_exit.any():
                    vals[to_exit] = randomness.service(ecap[ce][to_exit], t0, "edge")
                ecap[ce] = vals
            for k in blocked:
                if k in b.idx:
                    ecap[:, b.idx[k]] = 0.0
            disabled_node = np.array([c in blocked or base_in[k] == 0
                                      for k, c in enumerate(b.ids)])
            disabled_edge = np.broadcast_to(disabled_node, ecap.shape) | (edge_cap_mat == 0)
            random_node = rand_cells if randomness.stochastic_service else np.zeros(nA, bool)
            random_edge = np.zeros_like(ecap, dtype=bool)
            if randomness.stochastic_service:
                for ei, ek in cap_edges:
                    random_edge[ei, ek] = rand_cells[ek] or ek >= nT
            disabled_node |= (cap_in == 0) & ~random_node
            disabled_edge |= (ecap == 0) & ~random_edge
            node_budget = node_credit.offer(cap_in, random_node, disabled_node)
            edge_budget = edge_credit.offer(ecap, random_edge, disabled_edge)
            row_budget = row_credit.offer(out_cap * dt)
            for ei, ek in cap_edges:
                D[:, ei, ek] = limit_counts(D[:, ei, ek], edge_budget[ei, ek])
            # Blocked destinations also include edges without a separate cap.
            D[:, :, disabled_node] = 0
            for n in row_capped:
                D[:, n, :] = limit_counts(D[:, n, :], row_budget[n])
            supply = np.full(nA, INF)
            supply[:nT] = np.maximum(n_max[:nT] - occ_T, 0)
            node_cap = np.minimum(node_budget, supply)
            F = D.copy()
            for k in np.flatnonzero(D.sum(axis=(0, 1))):
                F[:, :, k] = limit_counts(D[:, :, k], node_cap[k])
            # Debit only actual movement, after all receiving constraints.
            node_credit.consume(F.sum(axis=(0, 1)))
            edge_credit.consume(F.sum(0))
            row_credit.consume(F.sum(axis=(0, 2)))
        else:
            F = D

        # update
        queue -= F
        if (queue < 0).any():
            raise RuntimeError("Service exceeded queued population")
        inflow = F.sum(axis=1)                                             # (V, nA)
        walk[:, :, 0] += inflow[:, :nT]
        done += inflow[:, nT:]

        per_cell = seated.sum(0) + walk.sum(axis=(0, 2)) + queue.sum(axis=(0, 2))
        rem = int(per_cell.sum())
        rem_in = int(per_cell[inside].sum())
        area += rem_in * dt
        area_all += rem * dt
        Fe = F.sum(0)
        rec_t.append(t1)
        rec_rem.append(rem)
        rec_in.append(rem_in)
        o = np.zeros(nA, dtype=np.int64)
        o[:nT] = seated.sum(0) + walk.sum(axis=(0, 2)) + queue.sum(axis=(0, 2))
        o[nT:] = done.sum(0)
        rec_occ.append(o)
        rec_q.append(queue.sum(axis=(0, 1)))
        rec_f.append(Fe[e_i, e_k])
        rec_ea.append(A_step[e_i, e_k])
        rec_eq.append(queue.sum(0)[e_i, e_k])
        #
        rec_cls.append(((seated + walk.sum(2) + queue.sum(2)) * inside[None, :]).sum(1))
        rec_exit.append(done.sum(0).copy())
        if T == INF and rem_in == 0:
            T = t1
        if rem == 0:
            T_ground = t1
            break

    class_exit: Dict[str, np.ndarray] = {}
    for vv, (name, *_rest) in enumerate(variants):
        class_exit[name] = class_exit.get(name, np.zeros(nE, dtype=np.int64)) + done[vv]
    # Every class clears only at exactly zero.
    class_T: Dict[str, float] = {}
    if rec_cls:
        R = np.array(rec_cls)                                              # (steps, V)
        names = [v_[0] for v_ in variants]
        for name in dict.fromkeys(names):
            cols = [i for i, nm in enumerate(names) if nm == name]
            rem_c = R[:, cols].sum(1)
            hit = np.flatnonzero(rem_c == 0)
            class_T[name] = float(np.array(rec_t)[hit[0]]) if hit.size else INF
    #
    #
    # Mean exit time from t = 0: someone leaving in step s exits at (s+1)·Δt but is
    # counted in `remaining` for s steps only, hence the extra Δt.
    mean_time = (area / n_total + dt) if np.isfinite(T) else INF
    mean_time_ground = (area_all / n_total + dt) if np.isfinite(T_ground) else INF
    return SimResult(b=b, t=np.array(rec_t), remaining=np.array(rec_rem),
                     remaining_inside=np.array(rec_in), occ=np.array(rec_occ),
                     queue_in=np.array(rec_q), flows=np.array(rec_f), edges=list(b.edges),
                     exit_cum=np.array(rec_exit), class_exit=class_exit, T=T, T_ground=T_ground,
                     mean_time=mean_time,
                     edge_arr=np.array(rec_ea), edge_queue=np.array(rec_eq), class_T=class_T,
                     meta=dict(population_model=ENGINE_VERSION, clearance_threshold=0, completed=bool(np.isfinite(T_ground)),
                               stochastic_motion=motion_rng is not None, n=n_total, v=v, discovery=discovery, blocked=sorted(blocked),
                               randomness=randomness.name, queue_model=queue_model.name,
                               classes=[c.name for c in classes], dt=dt, walk_model=walk_model,
                               mean_time_ground=mean_time_ground))


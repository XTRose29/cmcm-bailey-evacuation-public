"""Stationary local guidance on a same-floor passage graph.

The radius uses estimated passage metres, never schematic drawing coordinates.
A manager influences decisions only at covered nodes. There is no persistence or
stacking of beta boosts outside/inside overlapping coverage areas."""
from __future__ import annotations

import heapq
import math
from typing import Dict, Optional, Sequence, Tuple

from .building import Building
from .config import (Config, MANAGERS, MANAGER_CANDIDATE_KINDS, MANAGER_DEFAULT_NODES)


def candidate_nodes(b: Building) -> Tuple[str, ...]:
    'Indoor stations; staff stand clear of doorways and flows.'
    return tuple(sorted(c.id for c in b.cells if c.kind in MANAGER_CANDIDATE_KINDS
                        and not c.stair and not c.outside and c.id not in b.closed | b.blocked))


def passage_distances(b: Building, station: str) -> Dict[str, float]:
    """Same-floor undirected network distance between cell centres (m).

    Each link costs half each endpoint's walking length. Doors may transmit
    guidance, but walls, stairs, closed cells, and removed barrier links do not.
    This is a network-distance approximation, not a line-of-sight/acoustic model.
    """
    valid = set(candidate_nodes(b))
    if station not in valid:
        raise ValueError(f"Invalid manager station: {station}")
    adj = {i: {} for i in valid}
    for i, j in b.edges:
        if i not in valid or j not in valid or b.cell[i].level != b.cell[j].level:
            continue
        w = (b.cell[i].length + b.cell[j].length) / 2
        adj[i][j] = adj[j][i] = w
    dist = {station: 0.0}; todo = [(0.0, station)]
    while todo:
        d, i = heapq.heappop(todo)
        if d != dist[i]:
            continue
        for j, w in adj[i].items():
            if d + w < dist.get(j, math.inf):
                dist[j] = d + w
                heapq.heappush(todo, (d + w, j))
    return dist


def coverage(b: Building, stations: Sequence[str], radius_m: Optional[float] = None):
    radius = MANAGERS["radius_m"] if radius_m is None else radius_m
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Manager radius must be finite and nonnegative")
    if len(set(stations)) != len(stations):
        raise ValueError("Manager stations must be distinct")
    nearest = {}; owner = {}
    for station in sorted(stations):
        for i, d in passage_distances(b, station).items():
            if d <= radius and d < nearest.get(i, math.inf):
                nearest[i] = d; owner[i] = station
    return tuple(sorted(nearest)), nearest, owner


def default_stations(b: Building, n: int) -> Tuple[str, ...]:
    """Reproducible reference, not an optimized claim.

    Counts above six extend the prefix with unused candidate nodes. Every count
    uses a nested placement, so initial geometric coverage cannot shrink."""
    valid = candidate_nodes(b)
    if isinstance(n, bool) or int(n) != n or not 0 <= n <= len(valid):
        raise ValueError(f"Manager count must be an integer in [0,{len(valid)}]")
    preferred = [s for s in MANAGER_DEFAULT_NODES if s in valid]
    ordered = preferred + [s for s in valid if s not in preferred]
    return tuple(ordered[:int(n)])


def resolve_stations(b: Building, count: Optional[int], nodes: Optional[Sequence[str]]):
    if nodes is None:
        return default_stations(b, MANAGERS["law_min"] if count is None else count)
    stations = tuple(nodes)
    if count is not None and count != len(stations):
        raise ValueError("Manager count must equal the number of explicit stations")
    coverage(b, stations, 0.0)   # validates uniqueness and eligibility
    return stations


def guidance_info(b: Building, cfg: Config, stations: Sequence[str], radius_m=None, p_follow=None):
    p = MANAGERS["p_follow"] if p_follow is None else p_follow
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Manager compliance must be in [0,1]")
    nodes, distances, owner = coverage(b, stations, radius_m)
    return dict(managers=len(stations), manager_nodes=list(stations),
                manager_radius_m=MANAGERS["radius_m"] if radius_m is None else radius_m,
                manager_p_follow=p, manager_covered_nodes=list(nodes),
                manager_distance_m=distances, manager_nearest_station=owner,
                manager_alloc={i: int(i in stations) for i in candidate_nodes(b)},
                compliance_by_zone={z: p if z in nodes else 0.0 for z in cfg.pop.seat_counts()},
                manager_distance_metric="same-floor passage-centre shortest path (estimated metres)")

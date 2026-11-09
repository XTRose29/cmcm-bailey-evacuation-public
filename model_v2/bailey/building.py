"""/ Turn Bailey Hall into a graph of cells (states) and links.

Mapping to the Definition section
---------------------------------------------------------

Three refinements vs. the Definition (results are still reported per 3 exits):
     POR split into west/east halves, one per side staircase.
     Added stage stair towers ST1/ST2, reachable from balcony AND mezzanine.
     EXIT-Main | EXIT-Side-W, EXIT-Side-E → EXIT-Side | EXIT-Stage-W, EXIT-Stage-E → EXIT-Stage
     Exits split into 5 physical absorbing states, grouped into the 3 Definition exits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import Config

INF = float("inf")


@dataclass
class Cell:
    id: str
    zh: str                     # Chinese label
    kind: str                   # seat | access | door | corridor | foyer | portico | stair | exit
    length: float = 0.0         # ℓ_i, m
    width: float = 0.0          #
    stair: bool = False         #
    row_part: float = 0.0       # part walked inside seat rows
    inflow_cap: float = INF     # max persons entering per second
    n_max: float = INF          # storage capacity
    outflow_cap: float = INF    #
    level: str = "1F"
    xy: Tuple[float, float] = (0.0, 0.0)   # schematic coordinates
    exit_group: Optional[str] = None       # Definition exit group
    outside: bool = False                  #


@dataclass
class Building:
    cells: List[Cell]                      # transient first, exits last
    edges: List[Tuple[str, str]]           # directed links
    edge_cap: Dict[Tuple[str, str], float] = field(default_factory=dict)  #
    reverse: Set[Tuple[str, str]] = field(default_factory=set)   #
                                                                  # fallback-only links (turning back)
    closed: Set[str] = field(default_factory=set)      # closed cells
    blocked: Set[str] = field(default_factory=set)     #

    # convenience indices ----
    def __post_init__(self):
        self.ids = [c.id for c in self.cells]
        self.idx = {c.id: n for n, c in enumerate(self.cells)}
        self.transient = [c.id for c in self.cells if c.kind != "exit"]
        self.exits = [c.id for c in self.cells if c.kind == "exit"]
        self.nT = len(self.transient)
        self.nE = len(self.exits)
        self.cell = {c.id: c for c in self.cells}
        self.out = {c.id: [] for c in self.cells}
        for i, k in self.edges:
            self.out[i].append(k)
        self.exit_groups = sorted({c.exit_group for c in self.cells if c.exit_group},
                                  key=["EXIT-Main", "EXIT-Side", "EXIT-Stage"].index)

    def seats(self) -> List[str]:
        return [c.id for c in self.cells if c.kind == "seat"]

    def junctions(self) -> List[str]:
        'junctions where queues form.'
        return [c.id for c in self.cells if c.inflow_cap < INF and c.kind != "exit"]


def _ring(r: float, deg: float) -> Tuple[float, float]:
    'Polar helper: 180° west, 270° south (foyer), 360° east.'
    a = np.deg2rad(deg)
    return (r * np.cos(a), r * np.sin(a))


def build_building(cfg: Config,
                   closed: Iterable[str] = (),
                   barriers: Sequence[Tuple[str, str]] = (),
                   blocked: Iterable[str] = ()) -> Building:
    """Build the graph.

               cells removed from the graph (e.g. side doors locked in normal dismissal)
               corridor barriers: remove both directions between two corridor segments
               obstructed cells: still in the graph, capacity 0, possibly unknown to people"""
    g, p = cfg.geo, cfg.phys
    q, km = p.q, p.k_max
    # capacity μ = q × clear width
    cells: List[Cell] = []

    # seat zones ----------------
    seat_n = cfg.pop.seat_counts()
    o_xy = [(-6.5, -3.5), (0.0, -7.0), (6.5, -3.5)]
    m_xy = [_ring(12.5, 205), _ring(12.5, 270), _ring(12.5, 335)]
    b_xy = [_ring(28.0, 222), (0.0, -32.0), _ring(28.0, 318)]
    side = ["West", "Centre", "East"]
    for k in range(3):
        cells.append(Cell(f"O{k+1}", f"Orchestra {side[k]}", "seat", g.l_orchestra, row_part=g.l_row_half, xy=o_xy[k]))
    for k in range(3):
        cells.append(Cell(f"M{k+1}", f"mezzanine{side[k]}", "seat", g.l_mezz, row_part=g.l_row_half,
                          n_max=seat_n[f"M{k+1}"] + g.a_mezz_aisle * km, xy=m_xy[k]))
    for k in range(3):
        cells.append(Cell(f"B{k+1}", f"Balcony {side[k]}", "seat", g.l_balcony, row_part=g.l_row_half,
                          n_max=seat_n[f"B{k+1}"] + g.a_balcony_aisle * km, level="2F", xy=b_xy[k]))
    if g.seat_row_capacity:
        # row ends split by seats per zone
        ends = {"O": g.row_ends_orchestra, "M": g.row_ends_mezz, "B": g.row_ends_balcony}
        tot = {p_: sum(n for z, n in seat_n.items() if z[0] == p_) for p_ in ends}
        for c in cells:
            c.outflow_cap = ends[c.id[0]] * seat_n[c.id] / tot[c.id[0]] * g.q_row * g.w_row

    # passages ----------------
    a_deg = [200, 225, 255, 285, 315, 340]
    for k in range(6):
        cells.append(Cell(f"A{k+1}", f"Access point A{k+1}", "access", g.l_access, g.w_aisle,
                          inflow_cap=q * g.w_aisle, n_max=g.l_access * g.w_aisle * km,
                          xy=_ring(9.0, a_deg[k])))
    for k in range(9):
        cells.append(Cell(f"D{k+1}", f"Mezzanine door D{k+1}", "door", g.l_door, g.w_mezz_door,
                          inflow_cap=q * g.w_mezz_door, n_max=g.l_door * g.w_mezz_door * km + 1,
                          xy=_ring(15.5, 190 + 20 * k)))
    for k in range(10):
        cells.append(Cell(f"C{k+1}", f"Corridor C{k+1}", "corridor", g.l_corridor_seg, g.w_corridor,
                          inflow_cap=q * g.w_corridor, n_max=g.l_corridor_seg * g.w_corridor * km,
                          xy=_ring(18.0, 189 + 18 * k)))
    f_deg = [258, 270, 282]
    for k in range(3):
        cells.append(Cell(f"F{k+1}", f"Foyer portal F{k+1}", "door", g.l_door, g.w_foyer_door,
                          inflow_cap=q * g.w_foyer_door, n_max=g.l_door * g.w_foyer_door * km + 1,
                          xy=_ring(20.0, f_deg[k])))
    cells.append(Cell("FOY", "Foyer", "foyer", g.l_foyer, 20.0,
                      n_max=g.a_foyer * km, xy=(0.0, -23.0)))
    s_deg = [207, 225, 315, 333]   # attached to C2, C3, C8, C9
    for k in range(4):
        cells.append(Cell(f"S{k+1}", f"Side door S{k+1}", "door", g.l_door, g.w_side_door,
                          inflow_cap=q * g.w_side_door, n_max=g.l_door * g.w_side_door * km + 1,
                          xy=_ring(20.5, s_deg[k])))
    #
    # The portico is an outdoor colonnade: entering it counts as leaving the building,
    # but people still descend the side steps to reach the street.
    cells.append(Cell("POR1", "Portico west", "portico", g.l_portico, g.w_portico,
                      inflow_cap=q * g.w_portico, n_max=g.l_portico * g.w_portico * km,
                      xy=_ring(23.0, 214), outside=True))
    cells.append(Cell("POR2", "Portico east", "portico", g.l_portico, g.w_portico,
                      inflow_cap=q * g.w_portico, n_max=g.l_portico * g.w_portico * km,
                      xy=_ring(23.0, 326), outside=True))
    #
    # South portico linking the two halves (the portico wraps the building); detour only
    cells.append(Cell("PORS", "Portico south", "portico", g.l_portico_detour, g.w_portico,
                      inflow_cap=q * g.w_portico, n_max=g.l_portico_detour * g.w_portico * km,
                      xy=(0.0, -36.5), outside=True))
    cells.append(Cell("BS1", "Balcony stairs west", "stair", g.l_balcony_stair, g.w_balcony_stair, stair=True,
                      inflow_cap=q * g.w_balcony_stair,
                      n_max=g.l_balcony_stair * g.w_balcony_stair * km, level="2F→1F", xy=(-7.0, -24.5)))
    cells.append(Cell("BS2", "Balcony stairs east", "stair", g.l_balcony_stair, g.w_balcony_stair, stair=True,
                      inflow_cap=q * g.w_balcony_stair,
                      n_max=g.l_balcony_stair * g.w_balcony_stair * km, level="2F→1F", xy=(7.0, -24.5)))
    cells.append(Cell("ST1", "Stage stairs west", "stair", g.l_stage_stair, g.w_stage_stair, stair=True,
                      inflow_cap=q * g.w_stage_stair,
                      n_max=g.l_stage_stair * g.w_stage_stair * km, level="2F→out", xy=(-21.0, 4.0)))
    cells.append(Cell("ST2", "Stage stairs east", "stair", g.l_stage_stair, g.w_stage_stair, stair=True,
                      inflow_cap=q * g.w_stage_stair,
                      n_max=g.l_stage_stair * g.w_stage_stair * km, level="2F→out", xy=(21.0, 4.0)))

    # exits (absorbing) ----------------
    cells += [
        Cell("EXIT-Main", "Main exterior landing", "exit", xy=(0.0, -27.5), exit_group="EXIT-Main"),
        Cell("EXIT-Side-W", "West exterior stairs", "exit", xy=_ring(30.0, 196), exit_group="EXIT-Side"),
        Cell("EXIT-Side-E", "East exterior stairs", "exit", xy=_ring(30.0, 344), exit_group="EXIT-Side"),
        Cell("EXIT-Stage-W", "Stage exit west", "exit", xy=(-25.0, 7.0), exit_group="EXIT-Stage"),
        Cell("EXIT-Stage-E", "Stage exit east", "exit", xy=(25.0, 7.0), exit_group="EXIT-Stage"),
    ]

    # links ----------------
    E: List[Tuple[str, str]] = []
    two = lambda a, b: E.extend([(a, b), (b, a)])
    E += [("O1", "A1"), ("O1", "A2"), ("O2", "A2"), ("O2", "A3"), ("O2", "A4"), ("O2", "A5"),
          ("O3", "A5"), ("O3", "A6")]
    E += [("A1", "M1"), ("A2", "M1"), ("A3", "M2"), ("A4", "M2"), ("A5", "M3"), ("A6", "M3")]
    two("M1", "M2"); two("M2", "M3")
    E += [("M1", "D1"), ("M1", "D2"), ("M1", "D3"), ("M2", "D4"), ("M2", "D5"), ("M2", "D6"),
          ("M3", "D7"), ("M3", "D8"), ("M3", "D9")]
    E += [("D1", "C1"), ("D2", "C2"), ("D3", "C3"), ("D4", "C4"), ("D5", "C5"), ("D5", "C6"),
          ("D6", "C7"), ("D7", "C8"), ("D8", "C9"), ("D9", "C10")]
    for k in range(1, 10):
        two(f"C{k}", f"C{k+1}")
    for c, f in (("C5", "F1"), ("C5", "F2"), ("C6", "F2"), ("C6", "F3")):
        E.append((c, f))
    for f in ("F1", "F2", "F3"):
        E.append((f, "FOY"))
    E += [("FOY", "EXIT-Main")]
    E += [("C2", "S1"), ("C3", "S2"), ("C8", "S3"), ("C9", "S4")]
    E += [("S1", "POR1"), ("S2", "POR1"), ("S3", "POR2"), ("S4", "POR2")]
    E += [("POR1", "EXIT-Side-W"), ("POR2", "EXIT-Side-E")]
    two("B1", "B2"); two("B2", "B3")
    E += [("B1", "BS1"), ("B2", "BS1"), ("B2", "BS2"), ("B3", "BS2"), ("BS1", "FOY"), ("BS2", "FOY")]
    E += [("B1", "ST1"), ("B3", "ST2"), ("M1", "ST1"), ("M3", "ST2")]
    E += [("ST1", "EXIT-Stage-W"), ("ST2", "EXIT-Stage-E")]

    #
    # Turning-back links: used only when no forward link is usable (e.g. a known blockage)
    R = [("D1", "M1"), ("D2", "M1"), ("D3", "M1"), ("D4", "M2"), ("D5", "M2"), ("D6", "M2"),
         ("D7", "M3"), ("D8", "M3"), ("D9", "M3"),
         ("F1", "C5"), ("F2", "C5"), ("F2", "C6"), ("F3", "C6"),
         ("FOY", "F1"), ("FOY", "F2"), ("FOY", "F3"),
         ("S1", "C2"), ("S2", "C3"), ("S3", "C8"), ("S4", "C9"),
         ("ST1", "B1"), ("ST1", "M1"), ("ST2", "B3"), ("ST2", "M3"),
         ("BS1", "B1"), ("BS1", "B2"), ("BS2", "B2"), ("BS2", "B3"),
         ("POR1", "PORS"), ("PORS", "POR2"), ("POR2", "PORS"), ("PORS", "POR1")]
    E += R

    edge_cap = {
        ("FOY", "EXIT-Main"): q * g.w_main_exterior,
        ("POR1", "EXIT-Side-W"): q * g.w_side_stair,
        ("POR2", "EXIT-Side-E"): q * g.w_side_stair,
        ("ST1", "EXIT-Stage-W"): q * g.w_stage_stair,
        ("ST2", "EXIT-Stage-E"): q * g.w_stage_stair,
        ("M1", "ST1"): q * g.w_stage_door_mezz,
        ("M3", "ST2"): q * g.w_stage_door_mezz,
    }

    # barriers ----
    cut = set()
    for a, b in barriers:
        cut |= {(a, b), (b, a)}
    E = [e for e in E if e not in cut]

    # closed cells ----
    closed = set(closed)
    keep = [c for c in cells if c.id not in closed]
    E = [(i, k) for i, k in E if i not in closed and k not in closed]

    rev = {e for e in R if e in E}
    b = Building(keep, E, edge_cap, reverse=rev, closed=closed, blocked=set(blocked))
    _validate(b)
    return b


def _validate(b: Building) -> None:
    'every transient cell must reach an exit.'
    import networkx as nx
    G = nx.DiGraph(b.edges)
    G.add_nodes_from(b.ids)
    for c in b.transient:
        if not any(nx.has_path(G, c, e) for e in b.exits):
            raise ValueError(f"cell {c} cannot reach any exit")


# common closure sets
NORMAL_CLOSED = ("S1", "S2", "S3", "S4", "POR1", "POR2", "PORS", "EXIT-Side-W", "EXIT-Side-E",
                 "ST1", "ST2", "EXIT-Stage-W", "EXIT-Stage-E")
"""emergency only）。
Ordinary dismissal: side doors and stage stairs are not used (problem statement)."""


def egress_view(b: Building) -> Building:
    """Building-egress view: outdoor cells (portico) become absorbing, so lower bounds and the
    LP refer to T = last person leaves the building, not = reaches the street."""
    import copy
    outside = {c.id for c in b.cells if c.outside}
    # which exit each portico cell leads to
    leads = {o: next((k for k in b.out[o] if b.cell[k].kind == "exit"), None) for o in outside}
    keep_out = {o for o, e in leads.items() if e is not None}
    cells = []
    for c in b.cells:
        if c.kind == "exit" and any(leads.get(o) == c.id for o in keep_out):
            continue                       # replaced by the portico gate
        if c.id in outside and c.id not in keep_out:
            continue                       # pure detour cells dropped
        cc = copy.copy(c)
        if c.id in keep_out:
            cc.kind = "exit"
            cc.exit_group = b.cell[leads[c.id]].exit_group
        cells.append(cc)
    ids = {c.id for c in cells}
    trans = [c for c in cells if c.kind != "exit"]
    exits = [c for c in cells if c.kind == "exit"]
    edges = [(i, k) for i, k in b.edges if i in ids and k in ids and i not in keep_out]
    v = Building(trans + exits, edges, {e: c for e, c in b.edge_cap.items() if e in edges},
                 reverse={e for e in b.reverse if e in edges}, closed=set(b.closed), blocked=set(b.blocked))
    v.gate_to_exit = {o: leads[o] for o in keep_out}
    return v

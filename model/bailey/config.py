"""Model parameters and source notes. Dimensions are in metres and times in seconds.
Source tags distinguish literature values, problem data and modeling assumptions."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


# Physical parameters

@dataclass(frozen=True)
class Physical:
    dt: float = 0.1                # Production time step, seconds
                                    # Finite walking times and whole-person gates need a fresh dt check.
    q: float = 0.96                 # clear-width theatre door flow (Imanishi & Sano, 2019)
    v_emergency: float = 1.34       # free-flow speed, m/s (Weidmann, 1993)
    v_normal: float = 1.10
    stair_factor: float = 0.44 / 1.34  # emergency stairs, 0.44 m/s at baseline (NIST TN 1839)
    stair_factor_normal: float = 0.58  # ordinary-dismissal stairs [TEAM-WEB]
    row_factor: float = 0.50        # speed multiplier inside seat rows [TEAM-WEB]
    t_max: float = 1800.0           # simulation cap [TEAM-WEB]
    empty_threshold: float = 0.0    # legacy compatibility field; integer engine clears only at zero
    population_order_seed: int = 2026  # fixed shuffle of deterministic reaction ranks across groups
    walk_model: str = "erlang"
                                    # erlang (default, thin tail) or geometric (doc's P_ii) — see flow.py

    @property
    def k_max(self) -> float:
        'Jam density derived from q = v0·k_max/4 with v0 = emergency speed.'
        return 4.0 * self.q / self.v_emergency


# Geometry (widths w in m, lengths ℓ in m)

@dataclass(frozen=True)
class Geometry:
    # widths ----
    w_aisle: float = 1.2
    w_mezz_door: float = 1.2        # mezzanine doors D [TEAM-WEB "orchestra doors 1.2"]
    w_foyer_door: float = 1.3       # foyer doors F [TEAM-WEB]
    w_main_exterior: float = 7.0    # 5 exterior doors [TEAM-WEB]
    n_main_doors: int = 5           # number of main door leaves [PROBLEM]
    w_side_door: float = 1.2
    w_side_stair: float = 1.2       # portico side steps [TEAM-WEB]
    w_balcony_stair: float = 1.3    # balcony stairs [TEAM-WEB]
    w_stage_stair: float = 1.2      # stage-side stair towers [PLACEHOLDER]
    w_stage_door_mezz: float = 1.0  # mezzanine stage escape door [PLACEHOLDER]
    w_corridor: float = 2.2         # ring corridor [TEAM-WEB]
    w_portico: float = 3.0          # portico [ESTIMATE]


    l_orchestra: float = 12.0       # seat → access point [ESTIMATE]
    l_mezz: float = 10.0            # seat → door [ESTIMATE]
    l_balcony: float = 14.0         # seat → stair head [ESTIMATE]
    l_row_half: float = 4.0
    l_wheelchair_access: float = 2.0  # rear ADA platform to local aisle, m [ESTIMATE; not surveyed]
    l_access: float = 3.0           # steps up to mezzanine [ESTIMATE]
    l_door: float = 0.5
    l_corridor_seg: float = 8.5
    l_foyer: float = 10.0           # foyer crossing [ESTIMATE]
    l_portico: float = 15.0         # side door → side steps [ESTIMATE]
    l_portico_detour: float = 40.0  # walk around between portico halves [ESTIMATE]
    l_balcony_stair: float = 5.0    # balcony stair [TEAM-WEB]
    l_stage_stair: float = 8.0      # stage stair [PLACEHOLDER]
    l_mezz_lateral: float = 12.0    # lateral walk between M zones [ESTIMATE]
    l_balcony_lateral: float = 15.0 # lateral walk between B zones [ESTIMATE]


    a_mezz_aisle: float = 30.0      # aisle area per M zone [ESTIMATE]
    a_balcony_aisle: float = 20.0   # per B zone [ESTIMATE]
    a_foyer: float = 160.0          # foyer [ESTIMATE]


    #   Seat-zone outflow ≤ row ends × q_row × row width. Row ends from the initial seating layout
    #   (orchestra 19 rows × 5 blocks, mezzanine 7 × 8, balcony 3 × 3, two ends per block), split by seats.
    #   Back-of-envelope it never binds (each zone's doors/stairs are narrower); checked by the rowcap stage.
    seat_row_capacity: bool = False
    q_row: float = 1.0              # persons per m of clear row width per s [TEAM-WEB]
    w_row: float = 0.55             # clear row width m [TEAM-WEB]
    row_ends_orchestra: int = 190   # 19 × 5 × 2 [ESTIMATE from TEAM-WEB layout]
    row_ends_mezz: int = 112        # 7 × 8 × 2 [ESTIMATE from TEAM-WEB layout]
    row_ends_balcony: int = 18      # 3 × 3 × 2 [ESTIMATE from TEAM-WEB layout]


# Population

@dataclass(frozen=True)
class Population:
    n_total: int = 1329                                   # [PROBLEM]

    orchestra: Tuple[int, int, int] = (132, 263, 132)
    mezzanine: Tuple[int, int, int] = (162, 162, 162)
    balcony: Tuple[int, int, int] = (105, 106, 105)
    wheelchairs: int = 3                                  # included above [PROBLEM]

    def seat_counts(self) -> Dict[str, int]:
        d = {}
        for pre, trip in (("O", self.orchestra), ("M", self.mezzanine), ("B", self.balcony)):
            for i, n in enumerate(trip, start=1):
                d[f"{pre}{i}"] = n
        return d


# Behaviour (logit route choice)
#    U_k = logsumexp_e[ −β1·d(k,e) + β2·1(e=Main) + β4·1(e=plan) ] − β3·wait_k
# d is free-walk time to exit in seconds

@dataclass(frozen=True)
class Behaviour:
    name: str
    beta_dist: float      # penalty per second of walking [PLACEHOLDER]
    beta_habit: float     # bonus for the main entrance (habit) [PLACEHOLDER]
    beta_queue: float     # penalty per second of visible queue [PLACEHOLDER]
    beta_guide: float = 0.0   # bonus for the instructed exit [PLACEHOLDER]
    speed_factor: float = 1.0 # class speed multiplier
    uses_stairs: bool = True  # can use stairs (wheelchair = False)
    delay: float = 0.0


BEHAVIOURS: Dict[str, Behaviour] = {
    # concert-goers: strong main-entrance habit
    "concert_self":    Behaviour("concert_self",    beta_dist=0.25, beta_habit=2.0, beta_queue=0.10),
    # people following crowd managers
    "concert_managed": Behaviour("concert_managed", beta_dist=0.25, beta_habit=2.0, beta_queue=0.10, beta_guide=2.5),
    # people following LED strips
    "concert_led":     Behaviour("concert_led",     beta_dist=0.25, beta_habit=2.0, beta_queue=0.10, beta_guide=4.0),
    # ordinary dismissal
    "normal":          Behaviour("normal",          beta_dist=0.20, beta_habit=3.0, beta_queue=0.05),
    # students: familiar, nearest exit
    "lecture":         Behaviour("lecture",         beta_dist=0.50, beta_habit=0.3, beta_queue=0.30),
    "lecture_led":     Behaviour("lecture_led",     beta_dist=0.50, beta_habit=0.3, beta_queue=0.30, beta_guide=4.0),

    # wheelchair users: manual-wheelchair speed proxy, no stairs (elevators prohibited), guided to a
    # step-free exit [ASSUMED; data source to check: Boyce, Shields & Silcock 1999, Fire Technology 35(1)]
    "wheelchair":      Behaviour("wheelchair",      beta_dist=0.25, beta_habit=2.0, beta_queue=0.10,
                                 beta_guide=10.0,  # escorted → follows the plan
                                 speed_factor=0.69 / 1.34, uses_stairs=False,  # 0.69 m/s at baseline (Boyce et al., 1999)
                                 delay=10.0),
}

# (Lovreglio, Fonzone & dell'Olio 2016, Transportation Research Part A 92:59–75)。

# The structure of the utility is supported by Lovreglio et al. (2016); the values are assumptions
# reported as ranges (uncertainty.RANGES, tornado figure).

# Official accessibility page places two fixed spaces in the rear mezzanine ADA platform.
# Problem permits up to three wheelchair users; user requests all three together.
# Map the rear central platform (DD/EE 400-series companion seats) to coarse zone M2.
# The third co-located user is an explicit assumption, not a verified third fixed space.
WHEELCHAIR_SEATS: Dict[str, int] = {"M2": 3}
WHEELCHAIR_LOCATION_SOURCE = {
    "zone": "M2", "location": "rear central mezzanine ADA platform near rows DD/EE, seats 408-411",
    "verified_spaces": 2, "modeled_people": 3,
    "url": "https://www.cornellconcertseries.com/accessibility",
    "status": "Two fixed spaces verified; third person in the same area is a user-approved assumption.",
}

# wheelchair users to street level


#   Elevators are prohibited, so wheelchair users first reach an outdoor refuge (landing outside the
#   main doors, or the portico), then are helped down the exterior stairs once the stairs are clear.
WHEELCHAIR_ASSIST = {
    "v_stair": 0.31,
                         # individuals 0.07–0.91] / assisted stair speed
    "L_main_stair": 12.0,  # monumental stairs
    "L_side_stair": 10.0,  # side exterior stairs
}


VALIDATION_FLOWS = {
    "theatre_drill": 0.96,
    "lab_bottleneck": 1.9,
}

# Stationary managers at indoor decision nodes.
# 10 m adopted from von Schantz & Ehtamo (2022), Physica A 594, 127011,
# Section 5 (conference building); Section 4 uses 5 m. This is a modeling proxy,
# not a measured voice/visibility radius at Bailey Hall. Same-floor graph distance.
MANAGERS = {
    "law_min": 6,                 # [PROBLEM] 1329 audience members; staff not added to audience counts
    "radius_m": 10.0,             # [LIT-ADAPTED] https://doi.org/10.1016/j.physa.2022.127011
    "p_follow": 0.9,              # [ASSUMED] fixed share susceptible to local instructions
}
MANAGER_RADIUS_RANGE = (5.0, 15.0)  # [ASSUMED SENSITIVITY]; 5 and 10 m used in reference
MANAGER_DEFAULT_NODES = ("M1", "M2", "M3", "B1", "B3", "FOY")  # [DESIGN ASSUMPTION], not an optimum
MANAGER_CANDIDATE_KINDS = ("seat", "access", "door", "corridor", "foyer")
# A seat-zone node means a manager in that zone's aisle, not occupying a chair.
MANAGER_SEARCH = {                # [EXPERIMENT DESIGN], configurable CLI defaults
    "candidates": 200, "screen_reps": 10, "top_k": 10, "retest_reps": 100,
    "placement_seed": 1729, "screen_seed": 20261107, "retest_seed": 20261108,
}


#   κ = Var/Mean of persons served per step. Lanes of followers keep ~1.3 s headways (Hoogendoorn &
#   Daamen 2005) → more regular than Poisson; default κ = 0.5 [ASSUMED from LIT], κ = 0, 2 as checks.
SERVICE_DISPERSION = 0.5


# 5. Initial pre-movement distributions (seconds)
#    REACTION_SPECS is authoritative; windows below are legacy compatibility only.

PREP_WINDOWS: Dict[str, Tuple[float, float]] = {
    "concert_emergency": (2.0, 18.0),    # [LEGACY TEAM-WEB] not used by standard reaction factories
    "normal_dismissal":  (8.0, 50.0),    # [TEAM-WEB]
    "lecture_emergency": (1.0, 11.0),    # [TEAM-WEB]
}

# [LITERATURE-ADAPTED] Frozen on 2026-11-07 by user choice. These are not Bailey measurements.
# Galea et al. (2017), doi:10.1002/fam.2424, Eq.1 (excludes wheelchair users/helpers).
# Benson et al. (2023), doi:10.3934/mbe.2023403, Sec.2.2: a non-emergency
# classroom MODELING assumption, extrapolated here to ordinary concert dismissal.
# Forssberg et al. (2019), doi:10.1007/s10694-019-00881-1, Table6 school:
# total pre-movement, Gamma shape=3, SCALE=24.5 s, not the response-only Table5.
REACTION_SPECS = {
    "concert_emergency": dict(family="lognormal", mu=3.93, sigma=0.44),
    "normal_dismissal": dict(family="truncnorm", mu=35.0, sigma=20.0, low=0.0, high=120.0),
    "lecture_emergency": dict(family="gamma", shape=3.0, scale=24.5),
}
# Compatibility for legacy low-level experiments only; standard factories use REACTION_SPECS.
PREP_LOGNORMAL = {"concert_emergency": (3.93, 0.44)}


def prep_min(key: str) -> float:
    return float(REACTION_SPECS[key].get("low", 0.0))


# compliance shares
COMPLIANCE = {
    "managers": 0.60,
                        # used only for a uniform override; by default derived from MANAGERS
    "led":      0.77,   # dynamic-sign following proxy (Galea, Xie & Lawrence, 2014)

                        # with managers too: 1 − (1 − 0.77)(1 − c_managers) (independent channels)
}

BROADCAST_DELAY = 20.0  # delay before blocked-exit broadcast [TEAM-WEB]


#    Sources checked 2026-11-08 (BibTeX keys in docs/references.bib). Market prices are
#    vendor/retail estimates, not Bailey Hall quotes; quantities (300 m LED, ~60 LED runs,
#    20 posts, 2 staff-hours, events per year) are [ASSUMED].

@dataclass(frozen=True)
class Costs:
    'Cost presets (USD) for the cost–egress-time Pareto analysis; sources per field.'
    # $17.87/h NY mean, ushers/lobby attendants (BLS OEWS May 2023, 39-3031) × 4 h [LIT: BLS2023Ushers]
    crowd_manager_per_event: float = 71.48
    # ICC/NASFM crowd manager course $19.95 (Fire Engineering, 2012) [LIT: FireEngineeringCrowdManager]
    manager_training_per_year: float = 19.95
    # 2025-26 Dallas Morse Coors Concert Series alone had 7 Bailey concerts; total large events [ASSUMED]
    events_per_year: int = 40
    # installed LED tape $8–18/ft (Countbricks 2026), $12–30/ft (Indochina Light 2026); midpoint $19/ft
    # × 984 ft (300 m, [ASSUMED]) [LIT-MARKET]
    led_install: float = 18700.0
    # per run/yr: upkeep <$1–2 (Recolux 2025) + annual test $2–5 + batteries $15–40 per 3–5 yr
    # (FireProtectNearMe 2026) ≈ $7–17; × ~60 runs of 5 m [ASSUMED]; midpoint [LIT-MARKET]
    led_maintenance_per_year: float = 720.0
    # Visiontron heavy-duty stanchion $147.94/post (Crowd Control Store) × 20 posts [ASSUMED count] [LIT-MARKET]
    barrier_purchase: float = 2958.80
    # 2 staff × 1 h [ASSUMED] × $17.87/h (BLS) [LIT: BLS2023Ushers]
    barrier_setup_per_event: float = 35.74
    # LED useful life is L70; typical claim 50,000 h ≈ 5.7 yr continuous (DOE) [ASSUMED, LIT-supported range]
    lifetime_years: int = 10
    discount_rate: float = 0.0


# Cost sensitivity: low–high range of every cost parameter
COST_RANGES: Dict[str, Tuple[float, float]] = {
    "crowd_manager_per_event": (31.35, 108.96),  # national P10–P90 $10.45–18.16/h (BLS May 2023) × 3–6 h
    "manager_training_per_year": (0.0, 19.95),   # free state sessions – ICC/NASFM course price
    "events_per_year": (7.0, 80.0),              # 7 = 2025-26 concert series in Bailey; 80 [ASSUMED]
    "led_install": (7872.0, 29520.0),            # $8–30/ft installed × 984 ft
    "led_maintenance_per_year": (420.0, 1020.0), # $7–17 per run × 60 runs
    "barrier_purchase": (385.0, 2958.80),        # $19.25/post (PRIME, if price is per pair) – $147.94/post, × 20
    "barrier_setup_per_event": (10.45, 72.64),   # 2 staff × 0.5–2 h [ASSUMED] × $10.45–18.16/h (BLS)
    "lifetime_years": (5.0, 15.0),               # 5 ≈ 50,000 h always on (DOE L70 claim); 15 [ASSUMED]
}


# Full configuration

@dataclass(frozen=True)
class Config:
    phys: Physical = field(default_factory=Physical)
    geo: Geometry = field(default_factory=Geometry)
    pop: Population = field(default_factory=Population)
    costs: Costs = field(default_factory=Costs)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


DEFAULT = Config()


PARAM_SOURCES = [
    ("dt", "0.1 s", "USER choice; diagnostic half-step test does not establish convergence"),
    ("q", "0.96 people/(m·s) of clear width", "Imanishi & Sano (2019), theatre door proxy"),
    ("crowd managers", "≥ 6 (1 per 250)", "PROBLEM"),
    ("manager radius / p_follow", "10 m / 0.9", "von Schantz & Ehtamo (2022), adapted; compliance ASSUMED"),
    ("service dispersion κ", "0.5 (Poisson = 1)", "ASSUMED from LIT: Hoogendoorn & Daamen 2005 lane headways ≈1.3 s"),
    ("β structure", "distance, crowding, flow", "LIT: Lovreglio, Fonzone & dell'Olio 2016 (values ASSUMED)"),
    ("wheelchairs", "3 together at rear M2 ADA platform; 0.69 m/s; no elevators", "Cornell Concert Series: 2 spaces verified; third co-located by assumption"),
    ("costs", "see Costs", "BLS 2023 wages; vendor/retail prices; DOE L70; quantities ASSUMED"),
    ("v_emergency", "1.34 m/s", "Weidmann (1993), free flow"), ("v_normal", "1.10 m/s", "TEAM-WEB (midpoint)"),
    ("stair_factor", "0.44/1.34 emergency; 0.58 ordinary", "NIST TN 1839; TEAM-WEB"), ("row_factor", "0.50", "TEAM-WEB"),
    ("k_max", "4q/v0 = 2.87 p/m²", "DERIVED"),
    ("door widths A/D/F", "1.2 / 1.2 / 1.3 m", "TEAM-WEB"),
    ("main exterior doors", "5 × 1.4 = 7.0 m", "TEAM-WEB"),
    ("side steps / balcony stairs", "1.2 / 1.3 m", "TEAM-WEB"),
    ("side doors S", "1.2 m", "PLACEHOLDER"),
    ("stage stairs / stage escape door", "1.2 / 1.0 m", "PLACEHOLDER"),
    ("corridor width", "2.2 m", "TEAM-WEB"),
    ("cell lengths", "see Geometry", "ESTIMATE"),
    ("seat split per zone", "see Population", "ESTIMATE"),
    ("β1..β4 per behaviour", "see BEHAVIOURS", "PLACEHOLDER"),
    ("pre-movement", "concert LN(3.93,.44); ordinary TN(35,20;0,120); lecture Gamma(3,24.5 scale)", "Galea2017 Eq1; Benson2023 Sec2.2 modeling assumption; Forssberg2019 Table6; external transfer"),
    ("compliance LED", "0.77", "Galea, Xie & Lawrence (2014), dynamic-sign proxy"),
    ("broadcast delay", "20 s", "TEAM-WEB"),
]


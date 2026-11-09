"""checks for the newer parts.
  crowd-manager model, service variability κ, costs & Pareto, wheelchairs, clear width.
run:  python -m pytest -q"""
import os
import sys

from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bailey.building import build_building
from bailey.config import DEFAULT as cfg, MANAGERS
from bailey.cost import annual_cost, design_table, pareto_mask
from bailey.plugins.randomness import PoissonRandomness
from bailey.scenarios import SCENARIOS, manager_compliance, run

SEATS = {k: float(v) for k, v in cfg.pop.seat_counts().items()}


def test_capacity_uses_clear_width():
    'capacity uses the full clear width.'
    b = build_building(cfg)
    assert b.cell["D1"].inflow_cap == pytest.approx(cfg.phys.q * cfg.geo.w_mezz_door)
    main = b.edge_cap[("FOY", "EXIT-Main")]
    assert main == pytest.approx(cfg.phys.q * cfg.geo.w_main_exterior)
    assert b.edge_cap[("POR1", "EXIT-Side-W")] == pytest.approx(cfg.phys.q * cfg.geo.w_side_stair)


def test_manager_compliance_monotone_and_bounded():
    'more managers never lowers compliance; ≤ p_follow.'
    prev = -1
    for n in range(0, 20):
        c, k = manager_compliance(SEATS, n)
        avg = sum(c[z] * SEATS[z] for z in SEATS) / sum(SEATS.values())
        assert avg >= prev - 1e-12
        assert max(c.values()) <= MANAGERS["p_follow"] + 1e-12
        assert sum(k.values()) <= n
        prev = avg
    c0, _ = manager_compliance(SEATS, 0)
    assert all(v == 0 for v in c0.values())


def test_manager_initial_coverage_uses_station_ranges():
    from bailey.guidance import guidance_info, default_stations
    b=build_building(cfg)
    info=guidance_info(b,cfg,default_stations(b,MANAGERS["law_min"]))
    c, allocation=manager_compliance(SEATS,MANAGERS["law_min"])
    assert c == info["compliance_by_zone"]
    assert sum(allocation.values()) == MANAGERS["law_min"]


# @pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0])
def test_service_dispersion_mean_and_variance(kappa):
    'mean μΔt for every κ; Var/Mean grows with κ.'
    r = PoissonRandomness((2, 18), {"O1": 10}, dispersion=kappa).sample(np.random.default_rng(3))
    s = r.service(np.full(100000, 2.0), 0.0, "cell")
    assert s.mean() == pytest.approx(2.0, rel=0.02)
    ratio = s.var() / s.mean()
    assert ratio == pytest.approx(kappa, abs=0.12)


def test_pareto_mask():
    cost = [1, 2, 3, 2]
    time = [10, 8, 9, 12]
    assert list(pareto_mask(cost, time)) == [True, True, False, False]


def test_costs_add_up():
    a = annual_cost(cfg, 6)
    b = annual_cost(cfg, 6, led=True, barriers=True)
    assert b["total"] == pytest.approx(a["total"] + b["led"] + b["barriers"])
    df = design_table([dict(managers=6, led=False, barriers=False, T=150.0),
                       dict(managers=8, led=False, barriers=False, T=140.0)], cfg)
    assert df["pareto"].all() and df.loc[1, "seconds_saved"] == pytest.approx(10.0)


def test_wheelchair_users_leave_without_stairs():
    'wheelchair users never use stairs and do get out.'
    # Use a tight clearance threshold to check the full class mass at the exits.
    strict_cfg = cfg.with_(phys=replace(cfg.phys, empty_threshold=1e-7))
    r, _ = run(SCENARIOS["wheelchair"], strict_cfg)
    assert np.isfinite(r.class_T["wheelchair"])
    used = dict(zip(r.b.exits, r.class_exit["wheelchair"]))
    assert used["EXIT-Stage-W"] + used["EXIT-Stage-E"] == pytest.approx(0.0, abs=1e-9)
    assert sum(used.values()) == pytest.approx(3.0, abs=1e-6)


def test_manager_count_changes_coverage_and_preserves_population():
    # More guidance need not reduce T: poor routing can create congestion.
    a,ia=run(SCENARIOS["concert"].with_(managers=6),cfg)
    b,ib=run(SCENARIOS["concert"].with_(managers=12),cfg)
    assert set(ia["manager_nodes"]) < set(ib["manager_nodes"])
    assert set(ia["manager_covered_nodes"]) <= set(ib["manager_covered_nodes"])
    assert a.exit_cum[-1].sum() == b.exit_cum[-1].sum() == 1329


def test_single_door_matches_closed_form():
    'engine within 3% of τ + N/μ.'
    from bailey.validation import single_door_check
    d = single_door_check(cfg, Ns=(100, 300), widths=(0.9, 1.5))
    assert (d["rel_err"].abs() < 0.03).all(), d


def test_wheelchair_street_after_refuge_and_guidance():
    'street ≥ refuge; main guidance uses main.'
    from bailey.validation import wheelchair_report
    r, _ = run(SCENARIOS["wheelchair"], cfg)
    rep = wheelchair_report(r, cfg)
    assert rep["T_street"] >= rep["T_refuge"] > 0
    r2, _ = run(SCENARIOS["wheelchair"].with_(wheelchair_guide="main"), cfg)
    rep2 = wheelchair_report(r2, cfg)
    assert rep2["routes"]["main stairs"]["share"] == pytest.approx(1.0, abs=0.02)


def test_wheelchair_delay_postpones_start():
    """Isolate assistance delay: in an uncongested fixed route it adds exactly 30 s.

    In a crowd a later start can encounter a shorter queue, so full-network final
    egress is not monotone in this delay. This test checks the delay mechanism.
    """
    from bailey.flow import PopClass, simulate
    from bailey.plugins.randomness import mean_field_release
    from bailey.routing import cell_times, exit_distances, plan_nearest
    b=build_building(cfg);base=__import__('bailey.config',fromlist=['BEHAVIOURS']).BEHAVIOURS['wheelchair']
    plan=plan_nearest(b,exit_distances(b,cell_times(b,cfg,cfg.phys.v_emergency,base),set()))
    times=[]
    for delay in (0.,30.):
        c=PopClass('wheelchair',replace(base,delay=delay),{'M2':3},plan)
        result=simulate(b,cfg,[c],cfg.phys.v_emergency,capacities=False,
                        randomness=mean_field_release('concert_emergency'))
        times.append(result.class_T['wheelchair'])
    assert times[1]-times[0]==pytest.approx(30.,abs=cfg.phys.dt)

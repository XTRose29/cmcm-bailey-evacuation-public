"""checks for the 2026-11-07 additions.
  T definitions, seat-row capacity, cost sensitivity, wheelchair MC, T_ground in the parameter MC, new figures.
run:  python -m pytest -q"""
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bailey.building import build_building
from bailey.config import COST_RANGES, DEFAULT as cfg
from bailey.cost import cost_sensitivity, design_label, led_cost_per_second, structurally_dominated
from bailey.montecarlo import make_job, run_batch
from bailey.scenarios import SCENARIOS, run

CFG_ROW = cfg.with_(geo=dataclasses.replace(cfg.geo, seat_row_capacity=True))


def test_T_building_before_street():
    'leaving the building never comes after reaching the street.'
    for n in ("concert", "lecture", "normal"):
        r, _ = run(SCENARIOS[n], cfg)
        assert r.T <= r.T_ground


def test_row_capacity_off_by_default():
    'seat-row capacity is off in the clear-width baseline.'
    b = build_building(cfg)
    assert all(not np.isfinite(c.outflow_cap) for c in b.cells)
    r = run(SCENARIOS["concert"], cfg)[0]
    assert r.remaining[-1] == 0 and r.exit_cum[-1].sum() == 1329


def test_row_capacity_values_and_cumulative_limits():
    'check cumulative service; adaptive routes need not be monotone.'
    b = build_building(CFG_ROW)
    g, seats = CFG_ROW.geo, CFG_ROW.pop.seat_counts()
    o_tot = sum(n for z, n in seats.items() if z[0] == "O")
    assert b.cell["O2"].outflow_cap == pytest.approx(g.row_ends_orchestra * seats["O2"] / o_tot * g.q_row * g.w_row)
    assert sum(b.cell[f"B{k}"].outflow_cap for k in (1, 2, 3)) == pytest.approx(g.row_ends_balcony * g.q_row * g.w_row)
    assert not np.isfinite(b.cell["D1"].outflow_cap)
    for n in ("concert", "lecture"):
        r = run(SCENARIOS[n], CFG_ROW)[0]
        for cell in b.seats():
            cols = [m for m, (i, _) in enumerate(r.edges) if i == cell]
            passed = np.cumsum(r.flows[:, cols].sum(1))
            assert np.all(passed <= np.floor(b.cell[cell].outflow_cap*r.t + 1e-9))
        assert r.remaining[-1] == 0


def _designs():
    rows = []
    for n, T in ((6, 150.7), (7, 149.3), (8, 149.2), (10, 149.2)):
        rows.append(dict(managers=n, led=False, barriers=False, T_mc=T))
        rows.append(dict(managers=n, led=True, barriers=False, T_mc=T + 0.8))
        rows.append(dict(managers=n, led=False, barriers=True, T_mc=T + 27.0))
    return rows


def test_slower_devices_dominated_for_any_cost():
    'slower device designs never on the front.'
    rows = _designs()
    df = pd.DataFrame(rows)
    assert structurally_dominated(df).sum() == 8
    cs = cost_sensitivity(rows, cfg, n_draws=200)
    fr = cs["freq"]
    assert (fr.loc[fr["dominated_for_any_cost"], "share_on_front"] == 0).all()
    assert not cs["oat"]["device_on_front"].any()
    assert set(cs["oat"]["parameter"]) == set(COST_RANGES)
    assert design_label(rows[1]) == "6 mgr + LED"


def test_faster_device_can_join_front():
    rows = [dict(managers=6, led=False, barriers=False, T_mc=150.0),
            dict(managers=6, led=True, barriers=False, T_mc=140.0),
            dict(managers=7, led=False, barriers=False, T_mc=149.0)]
    cs = cost_sensitivity(rows, cfg, n_draws=100)
    assert cs["freq"].set_index("design").loc["6 mgr + LED", "share_on_front"] > 0


def test_led_cost_per_second():
    d = led_cost_per_second(cfg, 2.0, n_draws=200)
    lo = (COST_RANGES["led_install"][0] / COST_RANGES["lifetime_years"][1] + COST_RANGES["led_maintenance_per_year"][0]) / 2
    hi = (COST_RANGES["led_install"][1] / COST_RANGES["lifetime_years"][0] + COST_RANGES["led_maintenance_per_year"][1]) / 2
    assert d["usd_per_second"].between(lo - 1e-6, hi + 1e-6).all()


def test_wheelchair_mc_job_reports_street_time():
    'each replicate reports refuge and street times.'
    sc = SCENARIOS["concert"].with_(name="wc_test", wheelchairs=3, wheelchair_guide="main")
    out = run_batch({"w": make_job(sc, cfg, keep_curves=False, wheel=True)}, K=2, workers=1)["w"]
    assert np.all(np.isfinite(out["T_street"])) and np.all(out["T_street"] >= out["T_refuge"])


def test_param_mc_records_T_ground():
    from bailey.uncertainty import _pmc_job
    r = _pmc_job(("concert", 0, 2026, cfg))
    assert "T_ground" in r and r["T_ground"] >= r["T"]


def test_new_figures_draw(tmp_path):
    'the new figures draw on synthetic data.'
    from bailey import paper as P
    rng = np.random.default_rng(0)
    mcs = {n: dict(T=150 + rng.normal(0, 3, 30), T_ground=230 + rng.normal(0, 5, 30)) for n in ("concert", "lecture")}
    assert os.path.exists(P.fig_T_definition(mcs, ["concert", "lecture"], str(tmp_path)))
    assert os.path.exists(P.fig_cost_sensitivity(cost_sensitivity(_designs(), cfg, n_draws=50), str(tmp_path)))
    wmc = {"a": dict(T=150 + rng.normal(0, 3, 20), T_refuge=140 + rng.normal(0, 3, 20),
                     T_street=200 + rng.normal(0, 5, 20))}
    assert os.path.exists(P.fig_wheelchair_mc(wmc, str(tmp_path)))
    rc = dict(det=pd.DataFrame(dict(scenario=["concert", "lecture"], T_off=[144, 145], T_on=[144, 145.5])),
              mc=pd.DataFrame(dict(scenario=["concert"], K=[10], dT_mean=[0.2], dT_ci_lo=[0.0], dT_ci_hi=[0.4])))
    assert os.path.exists(P.fig_rowcap(rc, str(tmp_path)))


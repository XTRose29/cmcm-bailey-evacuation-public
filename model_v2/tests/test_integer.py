"""Integer conservation, finite people, bottlenecks and exact clearance."""
import json
from dataclasses import replace

import numpy as np
import pytest

from bailey.building import Building, Cell
from bailey.config import DEFAULT, BEHAVIOURS
from bailey.export import export_json
from bailey.flow import PopClass, simulate
from bailey.integer import ENGINE_VERSION, apportion, integer_classes, RouteAllocator, ServiceCredit
from bailey.plugins.randomness import MeanFieldPlaceholder, PoissonRandomness
from bailey.scenarios import SCENARIOS, run


def toy(n=5, rate=0.2):
    cells = [Cell("R", "room", "seat", 0.0),
             Cell("D", "door", "door", 0.0, 1.0, inflow_cap=rate, n_max=2.8),
             Cell("EXIT-Main", "exit", "exit", exit_group="EXIT-Main")]
    b = Building(cells, [("R", "D"), ("D", "EXIT-Main")])
    cls = [PopClass("people", BEHAVIOURS["concert_self"], {"R": n})]
    return b, cls


def assert_integer_result(r, n):
    for arr in (r.remaining, r.remaining_inside, r.occ, r.queue_in, r.flows,
                r.exit_cum, r.edge_queue, r.edge_arr, *r.class_exit.values()):
        assert np.issubdtype(arr.dtype, np.integer)
        if arr is not r.edge_arr:  # rerouting can remove people from a queue
            assert (arr >= 0).all()
    assert np.array_equal(r.remaining + r.exit_cum.sum(1), np.full(len(r.t), n))
    for i, cid in enumerate(r.b.transient):
        assert np.all(r.occ[:, i] <= np.floor(r.b.cell[cid].n_max))
    if np.isfinite(r.T):
        k = np.flatnonzero(r.remaining_inside == 0)[0]
        assert r.T == r.t[k]
        assert k == 0 or r.remaining_inside[k-1] >= 1
    if np.isfinite(r.T_ground):
        assert r.remaining[-1] == 0
        assert r.exit_cum[-1].sum() == n


def test_joint_population_rounding_preserves_zone_and_wheelchairs():
    beh = BEHAVIOURS["concert_self"]
    c = [PopClass("wheelchair", beh, {"R": 1}), PopClass("a", beh, {"R": 3.5}),
         PopClass("b", beh, {"R": 5.5})]
    out = integer_classes(c)
    assert [x.seat_counts["R"] for x in out] == [1, 4, 5]
    with pytest.raises(ValueError, match="non-integer total"):
        integer_classes([PopClass("invalid", beh, {"R": 1.4})])
    with pytest.raises(ValueError):
        integer_classes([PopClass("invalid", beh, {"R": -1})])


def test_blocked_sweep_plot_marks_unfinished_evacuation(tmp_path):
    import pandas as pd
    from bailey.paper import fig_blocked_sweep
    df = pd.DataFrame([
        {"target": "none", "discovery": "—", "T": 221.5},
        {"target": "EXIT-Main", "discovery": "known", "T": 397.5},
        {"target": "EXIT-Main", "discovery": "broadcast", "T": 401.0},
        {"target": "EXIT-Main", "discovery": "arrival", "T": np.inf},
    ])
    fig_blocked_sweep(df, str(tmp_path))
    assert (tmp_path / "F6_blocked_exit_sweep.png").is_file()


def test_single_arrivals_use_both_routes():
    a = RouteAllocator((1, 1, 2))
    counts = sum((a.split(0, np.array([1]), np.array([[.35, .65]]))
                  for _ in range(100)), np.zeros((1, 2), dtype=int))
    assert counts.tolist() == [[35, 65]]


def test_small_service_rate_does_not_round_to_zero():
    b, c = toy(n=5, rate=0.2)
    r = simulate(b, DEFAULT, c, 1.0, t_max=40)
    assert_integer_result(r, 5)
    assert np.isfinite(r.T)
    cum = r.flows[:, 0].cumsum()
    assert np.all(cum <= np.floor(0.2*r.t + 1e-9))


def test_idle_capacity_does_not_accumulate_unlimited_burst():
    s = ServiceCredit((1,))
    for _ in range(1000):
        budget = s.offer(np.array([0.6]))
        s.consume(np.array([0]))
    assert budget[0] <= 1
    assert s.offer(np.array([0.0]), disabled=np.array([True]))[0] == 0


def test_sampled_service_slots_expire_without_carrying_people():
    s = ServiceCredit((1,))
    assert s.offer(np.array([3.0]), np.array([True]))[0] == 3
    s.consume(np.array([0]))
    assert s.offer(np.array([0.0]), np.array([True]))[0] == 0


def test_one_reaction_sample_is_used_once_across_classes():
    b, _ = toy(n=2, rate=10)
    cls = [PopClass("a", BEHAVIOURS["concert_self"], {"R": 1}),
           PopClass("b", BEHAVIOURS["concert_self"], {"R": 1})]
    rnd = PoissonRandomness((0, 100), {"R": 2}, service="hard",
                           _draws={"R": np.array([0.0, 100.0])})
    r = simulate(b, DEFAULT, cls, 1.0, randomness=rnd, t_max=5)
    assert r.exit_cum[-1].sum() == 1
    assert r.remaining[-1] == 1
    assert np.isinf(r.T) and np.isinf(r.mean_time)
    assert_integer_result(r, 2)


@pytest.mark.parametrize("legacy_threshold", [0, 0.1, 0.5, 1, 2])
def test_last_whole_person_cannot_be_discarded(legacy_threshold):
    b, c = toy(n=1, rate=.2)
    r = simulate(b, DEFAULT, c, 1.0, threshold=legacy_threshold, t_max=20)
    assert_integer_result(r, 1)
    assert r.T == 5.0 + DEFAULT.phys.dt


def test_merge_never_overfills_integer_storage():
    b, c = toy(n=5, rate=4)
    b.cells.insert(1, Cell("R2", "room2", "seat", 0.0))
    b = Building(b.cells, b.edges + [("R2", "D")])
    c.append(PopClass("other", c[0].beh, {"R2": 7}))
    r = simulate(b, DEFAULT, c, 1.0)
    assert_integer_result(r, 12)
    assert r.occ[:, b.idx["D"]].max() <= 2


def test_cutoff_keeps_remaining_people():
    b, c = toy(n=5, rate=.2)
    r = simulate(b, DEFAULT, c, 1.0, t_max=2)
    assert_integer_result(r, 5)
    assert np.isinf(r.T) and not r.meta["completed"]


@pytest.mark.parametrize("mode", ["known", "arrival", "broadcast"])
def test_blockage_reroutes_only_whole_people(mode):
    r, _ = run(SCENARIOS["blocked"].with_(discovery=mode), DEFAULT)
    assert_integer_result(r, 1329)
    assert r.exit_cum[-1, r.b.exits.index("EXIT-Main")] == 0


def test_export_has_json_integer_counts(tmp_path):
    b, c = toy()
    r = simulate(b, DEFAULT, c, 1.0)
    path = export_json(r, str(tmp_path / "counts.json"))
    d = json.loads(open(path).read())
    assert all(type(x) is int for values in d["occupancy"].values() for x in values)
    assert d["meta"]["population_model"] == ENGINE_VERSION


def test_integer_service_apportionment_respects_every_demand():
    rng = np.random.default_rng(4)
    for _ in range(200):
        demand = rng.integers(0, 30, size=(3, 4))
        n = int(rng.integers(0, demand.sum()+1))
        got = apportion(n, demand)
        assert got.sum() == n and (got <= demand).all() and (got >= 0).all()

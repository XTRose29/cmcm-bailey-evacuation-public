'run:  python -m pytest -q'
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bailey.building import build_building, egress_view
from bailey.config import BEHAVIOURS, DEFAULT as cfg, PREP_WINDOWS, prep_min
from bailey.flow import PopClass, simulate
from bailey.markov import MarkovClass, absorbing_analysis, bottleneck_lower_bound, transition_matrix
from bailey.routing import Router, cell_times, exit_distances, leave_prob
from bailey.scenarios import SCENARIOS, run

SEATS = {k: float(v) for k, v in cfg.pop.seat_counts().items()}


def _x0(b):
    x = np.zeros(b.nT)
    for s, n in SEATS.items():
        x[b.transient.index(s)] = n
    return x


def test_population_total():
    assert sum(SEATS.values()) == cfg.pop.n_total == 1329


def test_rows_sum_to_one():
    'every row of P sums to 1.'
    b = build_building(cfg)
    for name, beh in BEHAVIOURS.items():
        tau = cell_times(b, cfg, cfg.phys.v_emergency, beh)
        stairs = {c.id for c in b.cells if c.stair} if not beh.uses_stairs else set()
        P = transition_matrix(b, Router(b, beh, tau, stairs), leave_prob(tau, cfg.phys.dt))
        assert np.allclose(P.sum(1), 1.0), name


@pytest.mark.parametrize("walk", ["geometric", "erlang"])
def test_integer_ensemble_matches_markov_expectations(walk):
    """Individual simulations match analytic means statistically, not identically."""
    from bailey.plugins.randomness import PoissonRandomness
    b = build_building(cfg)
    beh = dataclasses.replace(BEHAVIOURS["concert_self"], beta_queue=0.0)
    v = cfg.phys.v_emergency
    tau = cell_times(b, cfg, v, beh)
    ref = absorbing_analysis(b, cfg, MarkovClass("s", Router(b, beh, tau, set()),
                                                leave_prob(tau, cfg.phys.dt), _x0(b)))
    means, exits = [], []
    for seed in range(12):
        rnd = PoissonRandomness((0, 0), SEATS, service="hard").sample(np.random.default_rng(seed))
        sim = simulate(b, cfg, [PopClass("s", beh, SEATS)], v, capacities=False,
                       t_max=3000, walk_model=walk, randomness=rnd)
        assert sim.exit_cum[-1].sum() == 1329
        means.append(sim.meta["mean_time_ground"])
        exits.append(sim.exit_cum[-1])
    se = np.std(means, ddof=1) / np.sqrt(len(means))
    assert abs(np.mean(means)-ref["mean_time"]) < 5*se + 0.02
    expected = ref["exit_people"]*len(exits)
    probability = ref["exit_people"]/1329
    error = np.abs(np.sum(exits, axis=0)-expected)
    assert np.all(error < 5*np.sqrt(1329*len(exits)*probability*(1-probability))+1)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_conservation_and_lower_bound(name):
    'conservation; T ≥ max-flow lower bound.'
    sc = SCENARIOS[name]
    r, _ = run(sc, cfg)
    total = r.remaining + r.exit_cum.sum(1)
    assert np.allclose(total, cfg.pop.n_total, atol=1e-6)
    assert np.isfinite(r.T)
    from test_integer import assert_integer_result
    assert_integer_result(r, 1329)
    b = build_building(cfg, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
    bv = egress_view(b)
    tau = cell_times(bv, cfg, getattr(cfg.phys, sc.v_key), BEHAVIOURS["concert_self"])
    d = exit_distances(bv, tau, set(sc.blocked))
    lb = bottleneck_lower_bound(bv, d, cfg.pop.n_total, prep_min(sc.prep_key), set(sc.blocked))
    assert r.T >= lb["LB"] - 1.0, (r.T, lb["LB"])


def test_blocked_cell_carries_no_flow():
    r, _ = run(SCENARIOS["blocked"], cfg)
    m = r.b.exits.index("EXIT-Main")
    assert r.exit_cum[-1, m] == pytest.approx(0.0, abs=1e-9)


def test_less_capacity_is_never_faster():
    'with fixed routing, narrower doors never help.'
    beh = dataclasses.replace(BEHAVIOURS["concert_self"], beta_queue=0.0)
    Ts = []
    for q in (1.3, 1.0, 0.8):
        c2 = cfg.with_(phys=dataclasses.replace(cfg.phys, q=q))
        b = build_building(c2)
        from bailey.plugins.randomness import MeanFieldPlaceholder
        Ts.append(simulate(b, c2, [PopClass("s", beh, SEATS)], c2.phys.v_emergency,
                           randomness=MeanFieldPlaceholder((2, 18))).T)
    assert Ts[0] <= Ts[1] <= Ts[2]


def test_time_step_converged():
    """User-selected default 0.1 s versus a diagnostic half-step (not the production setting)."""
    T = {}
    for dt in (0.1, 0.05):
        c2 = cfg.with_(phys=dataclasses.replace(cfg.phys, dt=dt))
        T[dt] = run(SCENARIOS["concert"], c2)[0].T
    assert abs(T[0.1] - T[0.05]) < 2.0


# ---------------------------------------------------------------------------
# Section 4 queueing
# ---------------------------------------------------------------------------
def test_queue_curves_consistent():
    'N-curves: Q = A − D, queues empty at the end.'
    from bailey.plugins.queueing import FlowQueueModel
    r, _ = run(SCENARIOS["concert"], cfg)
    qa = FlowQueueModel().analyze(r)
    assert len(qa["table"]) > 10
    for j, s in qa["series"].items():
        assert np.allclose(s["A"] - s["D"], s["Q"], atol=1e-6), j
        assert s["Q"][-1] == 0, j
        assert (s["w"] >= 0).all()


def test_departures_never_exceed_capacity():
    """Whole people respect cumulative service, with at most one carried slot."""
    from bailey.plugins.queueing import FlowQueueModel
    r, _ = run(SCENARIOS["concert"], cfg)
    dt = cfg.phys.dt
    for j, s in FlowQueueModel().analyze(r)["series"].items():
        d = np.diff(np.concatenate([[0.0], s["D"]]))
        assert np.all(s["D"] <= np.floor(s["mu"] * r.t + 1e-9)), j
        assert d.max() <= 1 + s["mu"] * dt + 1e-9, j


# ---------------------------------------------------------------------------
# Section 5 Poisson
# ---------------------------------------------------------------------------
def _poisson(dist="uniform", service="poisson", m=0, sc="concert"):
    from bailey.montecarlo import _streams
    from bailey.plugins.randomness import PoissonRandomness
    proto = PoissonRandomness(PREP_WINDOWS[SCENARIOS[sc].prep_key], SEATS, dist=dist, service=service)
    return proto.sample_streams(*_streams(2026, m))


# @pytest.mark.parametrize("dist", ["uniform", "exponential", "poisson-stream"])
@pytest.mark.parametrize("dist", ["uniform", "exponential"])
def test_everyone_gets_a_reaction_time(dist):
    rnd = _poisson(dist)
    for z, n in SEATS.items():
        assert rnd.draws[z].size == int(round(n))
        assert rnd.release_cdf(z, 1e6) == 1.0
        assert rnd.release_cdf(z, -1.0) == 0.0


def test_common_random_numbers_reproducible():
    'same (seed, m) gives identical runs.'
    a = run(SCENARIOS["concert"], cfg, randomness=_poisson(m=3))[0]
    b = run(SCENARIOS["concert"], cfg, randomness=_poisson(m=3))[0]
    c = run(SCENARIOS["concert"], cfg, randomness=_poisson(m=4))[0]
    assert a.T == b.T and np.allclose(a.remaining, b.remaining)
    n = min(len(a.remaining), len(c.remaining))
    assert not np.allclose(a.remaining[:n], c.remaining[:n])


@pytest.mark.parametrize("name", ["concert", "blocked", "normal"])
def test_poisson_run_conserves_people(name):
    r, _ = run(SCENARIOS[name], cfg, randomness=_poisson(sc=name))
    assert np.allclose(r.remaining + r.exit_cum.sum(1), cfg.pop.n_total, atol=1e-6)
    assert np.isfinite(r.T)
    if name == "blocked":
        assert r.exit_cum[-1, r.b.exits.index("EXIT-Main")] == pytest.approx(0.0, abs=1e-9)


def test_poisson_service_keeps_mean_capacity():
    'Poisson service has mean μΔt.'
    rnd = _poisson()
    exp_ = np.full(20000, 0.78)
    assert rnd.service(exp_, 0.0, "cell").mean() == pytest.approx(0.78, rel=0.03)
    assert np.all(_poisson(service="hard").service(exp_, 0.0, "cell") == exp_)


def test_sampled_reactions_release_all_people_without_fractional_tail():
    from bailey.montecarlo import reaction_sampler, _streams
    rnd = reaction_sampler(SCENARIOS["concert"], cfg, SEATS, service="hard")
    rnd = rnd.sample_streams(*_streams(2026, 1))
    rnd.rng_motion = None  # isolate reaction variability
    r = run(SCENARIOS["concert"], cfg, randomness=rnd)[0]
    assert r.remaining_inside[-1] == 0
    assert r.exit_cum[-1].sum() == 1329
    assert r.T >= max(max(x) for x in rnd.person_draws.values())


def test_concert_uses_literature_release_profile():
    from bailey.montecarlo import reaction_sampler
    from bailey.plugins.randomness import mean_field_release
    sc = SCENARIOS["concert"]
    cdf = mean_field_release(sc.prep_key)
    assert cdf.release_cdf("O1", 0) == 0.0
    assert cdf.release_cdf("O1", np.exp(3.93)) == pytest.approx(0.5)
    proto = reaction_sampler(sc, cfg, SEATS)
    assert proto.dist == "lognormal"
    drawn = proto.sample(np.random.default_rng(2026))
    values = np.concatenate(list(drawn.draws.values()))
    assert np.isfinite(values).all() and values.min() > 0 and values.max() > 18

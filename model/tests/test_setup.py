"""Uniform release, map-aligned wheelchair counts, local managers, held-out search."""
from dataclasses import replace
import json
import numpy as np
import pytest

from bailey.building import Building, Cell, build_building
from bailey.config import DEFAULT, BEHAVIOURS, MANAGERS, PREP_WINDOWS, REACTION_SPECS, WHEELCHAIR_LOCATION_SOURCE
from bailey.guidance import candidate_nodes, coverage, default_stations
from bailey.montecarlo import make_job, reaction_sampler
from bailey.plugins.randomness import mean_field_release
from bailey.routing import Router, cell_times
from bailey.scenarios import SCENARIOS, build_population, run
from bailey.manager_search import draw_placements, ordered, rank_times, search


@pytest.mark.parametrize('name',list(SCENARIOS))
def test_scenario_factories_use_same_literature_distribution(name):
    from bailey.plugins.randomness import reaction_distribution
    sc=SCENARIOS[name]; spec=REACTION_SPECS[sc.prep_key]
    rv=reaction_distribution(spec); release=mean_field_release(sc.prep_key)
    q=release.reaction_times('M2',162)
    assert np.allclose(rv.cdf(q),(np.arange(162)+.5)/162)
    proto=reaction_sampler(sc,DEFAULT,DEFAULT.pop.seat_counts())
    assert proto.dist==spec['family']
    samples=proto.sample(np.random.default_rng(23))
    assert sum(map(len,samples.person_draws.values()))==1329
    assert all(np.isfinite(x).all() and (x>0).all() for x in samples.person_draws.values())
    control=reaction_sampler(sc,DEFAULT,{'M2':162},dist='mean-field')
    assert np.allclose(control.reaction_times('M2',162),q)
    with pytest.raises(ValueError,match='centrally'):
        reaction_sampler(sc,DEFAULT,{'M2':162},dist='exponential')


@pytest.mark.parametrize('key',list(REACTION_SPECS))
def test_parameter_noise_scales_entire_distribution_and_restores(key):
    from bailey.uncertainty import patched
    original={k:dict(v) for k,v in REACTION_SPECS.items()}
    q=mean_field_release(key).reaction_times('M2',162)
    with patched({'prep':2,'radius_m':5}):
        assert np.allclose(mean_field_release(key).reaction_times('M2',162),q*2)
        assert MANAGERS['radius_m']==5
    assert REACTION_SPECS==original and MANAGERS['radius_m']==10


def test_sampled_reactions_match_published_moments_and_bounds():
    from bailey.plugins.randomness import reaction_distribution
    for name in ('concert','lecture','normal'):
        sc=SCENARIOS[name]; spec=REACTION_SPECS[sc.prep_key]
        rv=reaction_distribution(spec)
        proto=reaction_sampler(sc,DEFAULT,{'M2':100000})
        x=proto.sample(np.random.default_rng(817)).person_draws['M2']
        assert abs(x.mean()-rv.mean()) < 5*rv.std()/np.sqrt(len(x))
        assert abs(x.std()-rv.std()) < .02*rv.std()
        if name=='normal': assert np.all((x>0)&(x<120))
    assert reaction_distribution(REACTION_SPECS['lecture_emergency']).mean()==pytest.approx(73.5)


def test_wheelchairs_together_at_rear_M2_and_total_unchanged():
    b=build_building(DEFAULT)
    for number in range(4):
        classes,info=build_population(SCENARIOS['concert'].with_(wheelchairs=number),DEFAULT,b)
        assert sum(sum(c.seat_counts.values()) for c in classes)==1329
        for z,n in DEFAULT.pop.seat_counts().items():
            assert sum(c.seat_counts.get(z,0) for c in classes)==n
        if number:
            wc=next(c for c in classes if c.name=='wheelchair')
            assert wc.seat_counts=={'M2':number}
            assert not wc.beh.uses_stairs
            assert cell_times(b,DEFAULT,DEFAULT.phys.v_emergency,wc.beh)[b.idx['M2']]==pytest.approx(DEFAULT.geo.l_wheelchair_access/.69)
    assert WHEELCHAIR_LOCATION_SOURCE['verified_spaces']==2
    assert WHEELCHAIR_LOCATION_SOURCE['modeled_people']==3
    with pytest.raises(ValueError,match='Wheelchair count'):
        build_population(SCENARIOS['concert'].with_(wheelchairs=4),DEFAULT,b)


def small_graph():
    # A and B are two metres apart along a corridor; C is across a wall;
    # U is upstairs even if a drawing puts it directly over A.
    cells=[Cell('A','A','corridor',2),Cell('B','B','corridor',2),
           Cell('C','C','corridor',2),Cell('U','U','seat',2,level='2F')]
    return Building(cells,[('A','B'),('B','U')])


def test_radius_uses_passages_not_drawing_or_different_floors():
    b=small_graph()
    assert coverage(b,['A'],1)[0]==('A',)
    assert coverage(b,['A'],2)[0]==('A','B')
    assert 'C' not in coverage(b,['A'],100)[0]
    assert 'U' not in coverage(b,['A'],100)[0]
    assert coverage(Building(b.cells,[]),['A'],100)[0]==('A',)
    assert coverage(b,['A','B'],2)[0]==('A','B')


def test_local_beta_applies_only_at_covered_current_node():
    b=build_building(DEFAULT)
    sc=SCENARIOS['concert'].with_(manager_nodes=('M1',),managers=1,manager_radius_m=0)
    classes,info=build_population(sc,DEFAULT,b)
    c=next(c for c in classes if c.name=='manager_receptive')
    assert set(c.node_behaviours)=={'M1'}
    assert c.beh.beta_guide==0 and c.node_behaviours['M1'].beta_guide==2.5
    tau=cell_times(b,DEFAULT,DEFAULT.phys.v_emergency,c.beh)
    base=Router(b,c.beh,tau,set(),c.plan)
    local=Router(b,c.beh,tau,set(),c.plan,c.node_behaviours)
    a,bb=base.split(np.zeros(len(b.ids))),local.split(np.zeros(len(b.ids)))
    idx=b.idx['M1']
    assert not np.allclose(a[idx],bb[idx])
    assert np.array_equal(np.delete(a,idx,axis=0),np.delete(bb,idx,axis=0))
    assert info['manager_radius_m']==0


@pytest.mark.parametrize('nodes,count,radius',[(('M1','M1'),2,10),(('UNKNOWN',),1,10),(('BS1',),1,10),(('M1',),2,10),(('M1',),1,-1)])
def test_invalid_manager_setups_are_rejected(nodes,count,radius):
    sc=SCENARIOS['concert'].with_(manager_nodes=nodes,managers=count,manager_radius_m=radius)
    with pytest.raises(ValueError):build_population(sc,DEFAULT,build_building(DEFAULT))


def test_lecture_has_no_manager_effect_and_no_hidden_placement():
    b=build_building(DEFAULT)
    classes,info=build_population(SCENARIOS['lecture'],DEFAULT,b)
    assert all(not c.node_behaviours for c in classes) and 'manager_nodes' not in info
    with pytest.raises(ValueError):
        build_population(SCENARIOS['lecture'].with_(manager_nodes=('M1',)),DEFAULT,b)


def test_random_layouts_unique_reproducible_and_physically_eligible():
    b=build_building(DEFAULT)
    a=draw_placements(b,6,100,27)
    assert a==draw_placements(b,6,100,27)
    assert len(set(a.values()))==100
    assert all(len(set(v))==6 and set(v)<=set(candidate_nodes(b)) for v in a.values())
    with pytest.raises(ValueError): draw_placements(b,0,2,1)


def test_ranking_penalizes_failure_and_does_not_average_completed_only():
    a=rank_times('a',('M1',),[50,np.inf],1800)
    b=rank_times('b',('M1',),[200,210],1800)
    assert ordered([a,b])[0]['layout']=='b'
    assert a['restricted_mean_T']==925 and a['timeout_rate']==.5


def test_search_retests_only_top_candidates_plus_control_on_independent_seeds(tmp_path,monkeypatch):
    import bailey.manager_search as ms
    seen=[]
    def fake_job(sc,cfg,**kw):return dict(sc=sc,seed=kw['seed'])
    def fake_batch(jobs,K,workers):
        seen.append((tuple(jobs),K,next(iter(jobs.values()))['seed']))
        return {k:{'T':np.full(K,200-int(k.split('_')[-1])*10),'setup':{}} for k in jobs}
    monkeypatch.setattr(ms,'make_job',fake_job);monkeypatch.setattr(ms,'run_batch',fake_batch)
    kw=dict(candidates=5,screen_reps=2,top_k=2,retest_reps=3,placement_seed=11,screen_seed=12,retest_seed=13,progress=lambda *a,**k:None)
    result=search(SCENARIOS['concert'],DEFAULT,tmp_path,**kw)
    assert result['layout']=='layout_0004'
    assert set(seen[0][0])=={f'layout_{i:04d}' for i in range(5)}
    assert set(seen[1][0])=={'layout_0004','layout_0003','layout_0000'}
    assert seen[0][1:]==(2,12) and seen[1][1:]==(3,13)
    search(SCENARIOS['concert'],DEFAULT,tmp_path,**kw)
    assert len(seen)==2  # checkpoint resume doesn't resimulate
    with pytest.raises(ValueError,match='independent'):
        search(SCENARIOS['concert'],DEFAULT,tmp_path,**dict(kw,retest_seed=12))
    assert json.loads((tmp_path/'selected.json').read_text())['manager_nodes']

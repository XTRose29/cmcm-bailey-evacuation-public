"""Random station search with independent re-evaluation.

No exhaustive-optimum claim. Every candidate is evaluated with common replicate
seeds within a phase; retesting uses a disjoint seed namespace. Rank first by the
fraction not cleared by the simulation horizon, then restricted mean completion
time E[min(T,t_max)], then restricted p95. With no failures, this is mean T ranking."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BEHAVIOURS, MANAGERS, PREP_WINDOWS, REACTION_SPECS, SERVICE_DISPERSION,
                     WHEELCHAIR_LOCATION_SOURCE, WHEELCHAIR_SEATS)
from .guidance import candidate_nodes, default_stations, guidance_info
from .building import build_building
from .montecarlo import make_job, run_batch


def draw_placements(b, n_managers, count, seed, baseline=None):
    """Uniform unique subsets; one explicitly labeled reference plus random layouts."""
    candidates = candidate_nodes(b)
    if isinstance(n_managers, bool) or int(n_managers) != n_managers or not 0 <= n_managers <= len(candidates):
        raise ValueError("Invalid manager count")
    if int(count) != count or not 1 <= count <= math.comb(len(candidates), n_managers):
        raise ValueError("Requested more unique layouts than the candidate set permits")
    reference = tuple(sorted(default_stations(b, n_managers) if baseline is None else baseline))
    if len(reference) != n_managers or len(set(reference)) != len(reference) or not set(reference) <= set(candidates):
        raise ValueError("Invalid reference placement")
    rng = np.random.default_rng(seed)
    layouts = [reference]; seen = {reference}
    while len(layouts) < count:
        nodes = tuple(sorted(rng.choice(candidates, n_managers, replace=False).tolist()))
        if nodes not in seen:
            seen.add(nodes); layouts.append(nodes)
    return {f"layout_{i:04d}": nodes for i, nodes in enumerate(layouts)}


def rank_times(key, nodes, T, t_max):
    values = np.asarray(T, dtype=float)
    if not len(values) or np.isnan(values).any() or (values < 0).any():
        raise ValueError("Invalid clearance times")
    capped = np.minimum(values, t_max)
    complete = np.isfinite(values)
    finite = values[complete]
    return dict(layout=key, nodes="|".join(nodes), K=len(values),
                timeout_rate=float(np.mean(~complete)),
                restricted_mean_T=float(capped.mean()),
                restricted_p95_T=float(np.quantile(capped, .95, method="higher")),
                T_mean=float(values.mean()),
                T_mean_se=float(values.std(ddof=1)/np.sqrt(len(values))) if complete.all() and len(values)>1 else None,
                P_under_180=float(np.mean(values < 180)),
                complete_runs=int(complete.sum()),
                completed_only_mean=float(finite.mean()) if len(finite) else None)


def ordered(rows):
    return sorted(rows, key=lambda r: (r["timeout_rate"], r["restricted_mean_T"],
                                       r["restricted_p95_T"], r["layout"]))


def _write_json(path, data):
    # Strict JSON: incomplete T is null + explicit completed=false, never invalid Infinity.
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)): return [clean(v) for v in x]
        if isinstance(x, (float,np.floating)) and not np.isfinite(x): return None
        return x
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def search(sc, cfg, outdir, *, candidates, screen_reps, top_k, retest_reps,
           placement_seed, screen_seed, retest_seed, workers=1, batch_size=10,
           progress=print):
    if sc.population not in ("managers", "led"):
        raise ValueError("Manager search requires a concert scenario with managers")
    if screen_seed == retest_seed:
        raise ValueError("Screening and retesting must use independent seed namespaces")
    for label, value in [('screen_reps',screen_reps),('retest_reps',retest_reps),('top_k',top_k),('batch_size',batch_size)]:
        if isinstance(value,bool) or int(value)!=value or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if top_k > candidates:
        raise ValueError("top_k cannot exceed candidates")
    b = build_building(cfg, closed=sc.closed, barriers=sc.barriers, blocked=sc.blocked)
    n = sc.managers if sc.managers is not None else (len(sc.manager_nodes) if sc.manager_nodes is not None else MANAGERS['law_min'])
    # User can run n=0 as an external control, but optimization obeys problem minimum.
    if n < MANAGERS['law_min']:
        raise ValueError("Concert deployment search requires at least six crowd managers")
    layouts = draw_placements(b, n, candidates, placement_seed, sc.manager_nodes)
    source_hash = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*.py')):
        source_hash.update(str(p.relative_to(Path(__file__).parent)).encode()); source_hash.update(p.read_bytes())
    setup = dict(scenario=asdict(sc), config=asdict(cfg), managers=dict(MANAGERS),
                 behaviours={k:asdict(v) for k,v in BEHAVIOURS.items()}, reaction_specs=REACTION_SPECS,
                 service_dispersion=SERVICE_DISPERSION, wheelchair_seats=WHEELCHAIR_SEATS,
                 wheelchair_source=WHEELCHAIR_LOCATION_SOURCE, code_hash=source_hash.hexdigest(),
                 candidate_nodes=list(candidate_nodes(b)), candidates=candidates, screen_reps=screen_reps,
                 top_k=top_k,retest_reps=retest_reps,placement_seed=placement_seed,
                 screen_seed=screen_seed,retest_seed=retest_seed,reference_layout='layout_0000',
                 objective='timeout rate, restricted mean T, restricted p95 T (ascending)',
                 layouts=layouts)
    signature = hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()
    out = Path(outdir);out.mkdir(parents=True,exist_ok=True)
    manifest = out/'setup.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('signature') != signature:
        raise ValueError("Output directory contains another setup; use a new directory")
    _write_json(manifest, dict(signature=signature, **setup))
    all_raw = []

    def phase(name, selected, K, seed):
        phase_dir=out/name;phase_dir.mkdir(exist_ok=True)
        pending=[key for key in selected if not (phase_dir/f'{key}.json').exists()]
        for first in range(0,len(pending),batch_size):
            keys=pending[first:first+batch_size]
            jobs={key:make_job(sc.with_(name=key,managers=n,manager_nodes=layouts[key]),cfg,
                              seed=seed,keep_curves=False,keep_diagnostics=False) for key in keys}
            result=run_batch(jobs,K,workers)
            for key in keys:
                vals=result[key]['T']
                payload=dict(signature=signature,seed=seed,T=[float(t) if np.isfinite(t) else None for t in vals],
                             summary=rank_times(key,layouts[key],vals,cfg.phys.t_max),setup=result[key]['setup'])
                _write_json(phase_dir/f'{key}.json',payload)
            progress(f'[{name}] {len(selected)-len(pending)+min(first+batch_size,len(pending))}/{len(selected)} layouts; {K} repeats each',flush=True)
        rows=[]
        for key in selected:
            data=json.loads((phase_dir/f'{key}.json').read_text())
            if data.get('signature') != signature or data['seed'] != seed or len(data['T']) != K:
                raise ValueError(f"Invalid checkpoint for {key}")
            T=np.array([math.inf if t is None else t for t in data['T']])
            rows.append(rank_times(key,layouts[key],T,cfg.phys.t_max))
            all_raw.extend(dict(phase=name,layout=key,nodes='|'.join(layouts[key]),seed=seed,replicate=m,
                                T=float(t) if np.isfinite(t) else None,completed=bool(np.isfinite(t))) for m,t in enumerate(T))
        ranking=ordered(rows)
        pd.DataFrame(ranking).to_csv(out/f'{name}_ranking.csv',index=False)
        return ranking

    screening=phase('screen',list(layouts),screen_reps,screen_seed)
    finalists=[r['layout'] for r in screening[:top_k]]
    # Always re-evaluate the reference with the same independent seeds for a paired comparison.
    retesting=phase('retest',list(dict.fromkeys(finalists+['layout_0000'])),retest_reps,retest_seed)
    best=retesting[0]
    winner=best['layout']
    selected=dict(status='best among sampled layouts after independent retesting; not a global optimum',
                  layout=winner,manager_nodes=layouts[winner],managers=n,
                  manager_radius_m=sc.manager_radius_m if sc.manager_radius_m is not None else MANAGERS['radius_m'],
                  manager_p_follow=sc.manager_p_follow if sc.manager_p_follow is not None else MANAGERS['p_follow'],
                  scenario=asdict(sc),retest_summary=best,screen_finalists=finalists,
                  evaluation_warning='Winner selected using retest results; use fresh holdout seeds for final reporting.')
    _write_json(out/'selected.json',selected)
    pd.DataFrame(all_raw).to_csv(out/'replicates.csv',index=False)
    winner_data=json.loads((out/'retest'/f'{winner}.json').read_text())
    baseline_data=json.loads((out/'retest'/'layout_0000.json').read_text())
    paired=[]
    for m,(a,bb) in enumerate(zip(winner_data['T'],baseline_data['T'])):
        paired.append(dict(replicate=m,seed=retest_seed,T_selected=a,T_reference=bb,
                           delta_T=a-bb if a is not None and bb is not None else None))
    pd.DataFrame(paired).to_csv(out/'retest_paired.csv',index=False)
    # Export station coverage independently of schematic coordinates for audit.
    info=guidance_info(b,cfg,layouts[winner],sc.manager_radius_m,sc.manager_p_follow)
    pd.DataFrame([dict(node=i,level=b.cell[i].level,station=info['manager_nearest_station'].get(i),
                       distance_m=info['manager_distance_m'].get(i),covered=i in info['manager_covered_nodes'])
                  for i in b.transient]).to_csv(out/'selected_coverage.csv',index=False)
    return selected

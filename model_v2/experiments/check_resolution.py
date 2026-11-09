"""Diagnose grid effects without modifying production model rules or parameters."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from bailey.config import DEFAULT
from bailey.scenarios import SCENARIOS,run,prepare
from bailey.flow import simulate
from bailey.plugins.randomness import mean_field_release
from bailey.uncertainty import patched


def one(args):
    step,mode=args
    cfg=DEFAULT.with_(phys=replace(DEFAULT.phys,dt=step))
    with patched({'beta_queue':0.} if mode=='no_queue_feedback' else {}):
        sc=SCENARIOS['concert']
        if mode=='no_capacities':
            b,classes,v,_=prepare(sc,cfg)
            r=simulate(b,cfg,classes,v,capacities=False,randomness=mean_field_release(sc.prep_key))
        else:r,_=run(sc,cfg)
    row=dict(dt=step,mode=mode,T=r.T,T90=r.time_to_fraction(.9),T99=r.time_to_fraction(.99),
             mean_time=r.mean_time,remaining_at_180=int(r.remaining_inside[np.searchsorted(r.t,180,side='right')-1]),
             max_queue=int(r.queue_in.max()),**r.exit_groups())
    for m,e in enumerate(r.b.exits):
        increments=np.diff(np.r_[0,r.exit_cum[:,m]])
        hit=np.flatnonzero(increments)
        row['last_'+e]=float(r.t[hit[-1]]) if len(hit) else None
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[(x,'full') for x in (.2,.1,.05,.025,.0125,.0625,.03125)]
    jobs += [(x,m) for m in ('no_queue_feedback','no_capacities') for x in (.1,.05,.025)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(one,jobs):
            rows.append(row);print(json.dumps(row),flush=True);pd.DataFrame(rows).to_csv(out,index=False)


if __name__=='__main__':main()

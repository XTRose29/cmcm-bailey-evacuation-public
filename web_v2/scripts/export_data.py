"""Export final-model deterministic replays and archived paper statistics.

The browser only replays these Python outputs; it does not run a second model.
"""
from pathlib import Path
import sys, json, os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model_v2'))
import numpy as np
import pandas as pd
from bailey.config import DEFAULT, MANAGER_DEFAULT_NODES
from bailey.scenarios import run, SCENARIOS
from bailey.validation import wheelchair_report

TABLES=ROOT/'results/cluster_formal_20261108/out/tables'
DEST=ROOT/'web_v2/data'
def records(path):
    return json.loads(pd.read_csv(path).to_json(orient='records'))
def main():
    DEST.mkdir(exist_ok=True)
    reference=pd.read_csv(TABLES/'deterministic_primary.csv').set_index('case')
    data={'source_commit':'e4201d3239bdb51d21e73e7aefd3ce81a6dc0d1f',
          'experiment_commit':'9d76e442fd31b99a770485548fcb504b56031cc4',
          'dt':DEFAULT.phys.dt,'population':1329,'wheelchairs':3,
          'manager_nodes':list(MANAGER_DEFAULT_NODES),'replays':{},
          'main':records(TABLES/'main_summary.csv'),
          'paired':records(TABLES/'main_paired.csv'),
          'broadcast':records(TABLES/'broadcast_delays_summary.csv'),
          'numerical':records(TABLES/'numerical_resolution_summary.csv'),
          'cost':records(ROOT/'results/paper_cost_update/cost_recomputed_latest.csv')}
    for name in reference.index:
        path=DEST/(name+'.json')
        if path.exists():
            replay=json.loads(path.read_text())
        else:
            print('Exporting',name,flush=True)
            res,_=run(SCENARIOS[name].with_(wheelchairs=3),DEFAULT)
            assert np.all(res.remaining+res.exit_cum.sum(axis=1)==1329)
            assert np.all(res.occ==np.floor(res.occ))
            # Retain exact frames at roughly one-second intervals and the last frame.
            ix=np.unique(np.r_[np.arange(0,len(res.t),10),len(res.t)-1])
            replay={'T':res.T,'T_ground':res.T_ground,
                'wheelchair':{k:float(v) for k,v in wheelchair_report(res,DEFAULT).items()
                             if k in ('T_refuge','T_street')},
                'nodes':[{'id':c.id,'label':c.zh.replace('mezzanine','Mezzanine ',1),'kind':c.kind,'level':c.level,
                          'x':float(c.xy[0]),'y':float(c.xy[1]),'outside':c.outside}
                         for c in res.b.cells],
                'edges':res.b.edges,'time':res.t[ix].round(3).tolist(),
                'occupancy':res.occ[ix].astype(int).tolist(),
                'queue':res.queue_in[ix].astype(int).tolist(),
                'inside':res.remaining_inside[ix].astype(int).tolist(),
                'to_street':res.remaining[ix].astype(int).tolist()}
            path.write_text(json.dumps(replay,separators=(',',':'))+'\n')
        expected=float(reference.loc[name,'T'])
        if abs(replay['T']-expected)>1e-6:
            raise RuntimeError(f'{name}: replay {replay["T"]} does not match archived {expected}')
        data['replays'][name]=replay
        print(name,'verified T =',replay['T'],flush=True)
    (DEST/'paper-data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    print('Export complete: all nine replays match archived deterministic times.')
if __name__=='__main__':main()

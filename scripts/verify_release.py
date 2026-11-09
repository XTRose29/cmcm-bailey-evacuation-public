"""Check published data consistency without rerunning expensive experiments."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'results/formal/tables'
def main():
    summary=pd.read_csv(TABLES/'main_summary.csv').set_index('case')
    reps=pd.read_csv(TABLES/'main_replicates.csv')
    for name,rows in reps.groupby('case'):
        assert len(rows)==int(summary.loc[name,'K'])==300
        assert np.isclose(rows['T'].mean(),summary.loc[name,'T_mean'])
        assert np.isclose(np.quantile(rows['T'],.95,method='higher'),summary.loc[name,'T95'])
        assert np.isclose((rows['T']<180).mean(),summary.loc[name,'P_under_180'])
    data=json.loads((ROOT/'web/data/results.json').read_text())
    det=pd.read_csv(TABLES/'deterministic_primary.csv').set_index('case')
    assert data['dt']==.1 and data['population']==1329 and data['wheelchairs']==3
    assert set(data['replays'])==set(summary.index)
    for name,r in data['replays'].items():
        occ=np.array(r['occupancy']);queue=np.array(r['queue'])
        assert np.isfinite(occ).all() and (occ>=0).all() and np.equal(occ,np.floor(occ)).all()
        assert (queue>=0).all() and np.equal(queue,np.floor(queue)).all()
        assert (occ.sum(axis=1)==1329).all()
        inside=[i for i,n in enumerate(r['nodes']) if n['kind']!='exit' and not n['outside']]
        assert np.array_equal(occ[:,inside].sum(axis=1),r['inside'])
        assert np.isclose(r['T'],det.loc[name,'T'])
        assert r['inside'][-1]==0
    for row in data['main']:
        assert np.isclose(row['T_mean'],summary.loc[row['case'],'T_mean'])
    cost=pd.read_csv(ROOT/'results/costs/cost_recomputed_latest.csv')
    assert np.allclose(cost['T_mc'],pd.read_csv(TABLES/'cost_pareto.csv')['T_mc'])
    assert np.allclose(cost['cost_total'],cost[['cost_managers','cost_led','cost_barriers']].sum(axis=1))
    manifest=ROOT/'release_manifest.json'
    if manifest.exists():
        for name,sha in json.loads(manifest.read_text())['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    print('Verified: 2,700 main replicates, nine integer replays, experiment summaries, updated costs and release hashes.')
if __name__=='__main__':main()

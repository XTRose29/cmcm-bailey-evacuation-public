"""Statistical reporting and checkpoint reproducibility, independent of engine tests."""
import argparse
from pathlib import Path
import numpy as np
import pytest
import run_overnight as N


def test_strict_target_and_censoring():
    r=N.metrics([170.,180.,float('inf')])
    assert r['P_under_180']==pytest.approx(1/3)
    assert r['P_at_most_180']==pytest.approx(2/3)
    assert r['T_mean'] is None and r['timeout_rate']==pytest.approx(1/3)
    assert r['restricted_mean_T']==pytest.approx((170+180+1800)/3)
    assert N.wilson(0,100)[1]>0
    assert N.wilson(100,100)[0]<1


def test_checkpoint_resume_does_not_repeat_mc(tmp_path,monkeypatch):
    a=argparse.Namespace(out=str(tmp_path),deadline=None,smoke=True,batch_size=1,commit='test',workers=1)
    e=N.Experiment(a);calls=[]
    def batch(jobs,K,workers):
        calls.append((jobs['x']['seed'],K))
        x=np.full(K,190.)
        return {'x':dict(T=x,T90=x,T50=x,T_ground=x,mean_time=x,K=K,n=1329,setup={},exits={'main':np.full(K,1329)},queues=[],starts=[],seed=jobs['x']['seed'])}
    monkeypatch.setattr(N,'run_batch',batch)
    first=e.mc('test',{'x':{}},2,123)
    second=e.mc('test',{'x':{}},2,123)
    assert len(calls)==2 and calls[0][0]!=calls[1][0]
    assert np.array_equal(first['x']['T'],second['x']['T'])
    assert (tmp_path/'tables'/'test_replicates.csv').exists()
    with pytest.raises(ValueError):N.Experiment(argparse.Namespace(**{**vars(a),'batch_size':2}))

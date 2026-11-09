"""Checkpointed overnight experiments; orchestration only, never edits model parameters.

Run from a frozen source copy. See ../docs/EXPERIMENTS.md for populations, seeds,
priorities, stopping rules and the distinction between selection and evaluation.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from bailey import config as C
from bailey.config import DEFAULT
from bailey.scenarios import SCENARIOS, BLOCK_TARGETS, run
from bailey.montecarlo import make_job, run_batch
from bailey.manager_search import search
from bailey.plugins.queueing import FlowQueueModel
from bailey.plugins.randomness import reaction_distribution

TZ=ZoneInfo('America/New_York')
HERE=Path(__file__).resolve().parent
MAIN=('normal','concert','blocked','lecture','barriers','led','combo','lecture_led','lecture_barriers')
STAGES=('preflight','deterministic','main','manager_search','holdout','controls','obstructions',
        'manager_robustness','wheelchairs','cost','sweeps','parameters','validation','ablations','paper')
SEEDS=dict(main=26071101,controls=26071102,holdout=26071103,obstructions=26071104,
           manager_robustness=26071105,wheelchairs=26071106,cost=26071107,
           placement=26071108,screen=26071109,retest=26071110,parameters=26071111)


def now(): return dt.datetime.now(TZ).isoformat(timespec='seconds')
def digest():
    h=hashlib.sha256()
    for p in sorted(list((HERE/'bailey').rglob('*.py'))+[HERE/'make_figures.py',Path(__file__).resolve()]):
        h.update(str(p.relative_to(HERE)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def atomic_json(path,data):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def save_pickle(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def load_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def wilson(k,n):
    z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)

def metrics(T,horizon=1800):
    T=np.asarray(T,float);x=np.minimum(T,horizon);n=len(T);ok=np.isfinite(T)
    se=x.std(ddof=1)/np.sqrt(n) if n>1 else math.nan
    lo,hi=wilson(int(np.sum(T<180)),n)
    return dict(K=n,completed=int(ok.sum()),timeout_rate=float((~ok).mean()),
                restricted_mean_T=float(x.mean()),restricted_mean_ci_low=float(x.mean()-1.96*se),
                restricted_mean_ci_high=float(x.mean()+1.96*se),
                T_mean=float(T.mean()) if ok.all() else None,
                T_median=float(np.quantile(T,.5,method='higher')),
                T95=float(np.quantile(T,.95,method='higher')),
                P_under_180=float(np.mean(T<180)),P180_ci_low=lo,P180_ci_high=hi,
                P_at_most_180=float(np.mean(T<=180)))

def combined(parts):
    """Combine blocks without changing samples; preserve arrays for existing plotters."""
    from bailey.montecarlo import summary
    out=dict(parts[0]);out['K']=sum(p['K'] for p in parts)
    for key in ('T','T90','T50','T_ground','mean_time','curves','T_refuge','T_street'):
        if key in out:out[key]=np.concatenate([p[key] for p in parts])
    out['exits']={k:np.concatenate([p['exits'][k] for p in parts]) for k in out['exits']}
    for key in ('queues','starts'):out[key]=[v for p in parts for v in p.get(key,[])][:40]
    out.update(summary(out['T']));return out


class DeadlineReached(Exception):pass


class Experiment:
    def __init__(self,a):
        self.a=a;self.out=Path(a.out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        for folder in ('tables','figures','results','blocks','completed'):
            (self.out/folder).mkdir(exist_ok=True)
        self.hash=digest();self.deadline=dt.datetime.fromisoformat(a.deadline).astimezone(TZ) if a.deadline else None
        self.status={'started':now(),'pid':os.getpid(),'code_hash':self.hash,'stages':{},'state':'running'}
        manifest=dict(code_hash=self.hash,config=dataclasses.asdict(DEFAULT),reaction_specs=C.REACTION_SPECS,
                      behaviours={k:dataclasses.asdict(v) for k,v in C.BEHAVIOURS.items()},
                      managers=C.MANAGERS,wheelchair_seats=C.WHEELCHAIR_SEATS,seeds=SEEDS,
                      profile='smoke' if a.smoke else 'formal',batch_size=a.batch_size,
                      python=sys.version,platform=platform.platform(),commit=a.commit)
        p=self.out/'manifest.json'
        if p.exists():
            old=json.loads(p.read_text())
            if any(old.get(k)!=manifest[k] for k in ('code_hash','profile','batch_size')):
                raise ValueError('Different code/profile/batch size: use a new output directory')
        atomic_json(p,manifest);self.update()

    def update(self):
        self.status['updated']=now();atomic_json(self.out/'status.json',self.status)

    def check(self):
        if digest()!=self.hash:raise RuntimeError('Frozen source changed during experiments')
        if self.deadline and dt.datetime.now(TZ)>=self.deadline:raise DeadlineReached()

    def count(self,k):return min(k,2) if self.a.smoke else k

    def mc(self,label,jobs,K,seed):
        K=self.count(K);folder=self.out/'blocks'/label;folder.mkdir(parents=True,exist_ok=True)
        parts={key:[] for key in jobs};elapsed=0
        for start in range(0,K,self.a.batch_size):
            self.check();size=min(self.a.batch_size,K-start)
            # Same block seed and replicate indices for every comparison in this suite.
            block_seed=int(np.random.SeedSequence([seed,start]).generate_state(1)[0])
            path=folder/f'{start:05d}.pkl'
            if path.exists():data=load_pickle(path)
            else:
                for job in jobs.values():job['seed']=block_seed
                t=time.monotonic();data=run_batch(jobs,size,self.a.workers);elapsed+=time.monotonic()-t
                save_pickle(path,data)
            for key in jobs:parts[key].append(data[key])
            print(f'[{now()}] {label}: {start+size}/{K} replicates x {len(jobs)} cases',flush=True)
            self.status['progress']=dict(suite=label,completed=start+size,target=K,cases=len(jobs));self.update()
            self.export(label,{key:combined(v) for key,v in parts.items()})
        out={key:combined(v) for key,v in parts.items()};save_pickle(self.out/'results'/f'{label}.pkl',out)
        if elapsed:print(f'[{label}] measured {elapsed:.1f} s for new blocks',flush=True)
        return out

    def export(self,label,result):
        rows=[];raw=[]
        for key,r in result.items():
            rows.append(dict(case=key,**metrics(r['T'])))
            for m,t in enumerate(r['T']):
                row=dict(case=key,replicate=m,T=t,completed=bool(np.isfinite(t)),T90=r['T90'][m],
                         T_ground=r['T_ground'][m],mean_time=r['mean_time'][m])
                for k in ('T_refuge','T_street'):
                    if k in r:row[k]=r[k][m]
                raw.append(row)
        tab=self.out/'tables';tab.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(tab/f'{label}_summary.csv',index=False)
        pd.DataFrame(raw).to_csv(tab/f'{label}_replicates.csv',index=False)
        pd.DataFrame([dict(case=k,**r['setup']) for k,r in result.items()]).to_json(tab/f'{label}_setups.json',orient='records',indent=2)
        pairs=[]
        for a,b in [('barriers','concert'),('led','concert'),('combo','concert'),('lecture_led','lecture'),('lecture_barriers','lecture')]:
            if a in result and b in result:pairs.append((a,b))
        pairs += [(k,k.replace('selected|','reference|',1)) for k in result if k.startswith('selected|') and k.replace('selected|','reference|',1) in result]
        out=[]
        for a,b in pairs:
            d=np.minimum(result[a]['T'],DEFAULT.phys.t_max)-np.minimum(result[b]['T'],DEFAULT.phys.t_max)
            se=d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else math.nan
            out.append(dict(design=a,reference=b,K=len(d),restricted_delta_mean=d.mean(),
                            ci_low=d.mean()-1.96*se,ci_high=d.mean()+1.96*se,
                            delta_P180=np.mean(result[a]['T']<180)-np.mean(result[b]['T']<180)))
        if out:pd.DataFrame(out).to_csv(tab/f'{label}_paired.csv',index=False)
        self.report()

    def jobs(self,names,wheelchairs=3,**kw):
        return {n:make_job(SCENARIOS[n].with_(wheelchairs=wheelchairs),DEFAULT,keep_curves=True,
                          keep_diagnostics=True,wheel=wheelchairs>0,**kw) for n in names}

    def stage(self,name):
        self.check();marker=self.out/'completed'/f'{name}.json'
        if marker.exists():
            self.status['stages'][name]=json.loads(marker.read_text());self.update();return
        self.status['stages'][name]={'state':'running','start':now()};self.update()
        t=time.monotonic();getattr(self,name)()
        info={'state':'complete','seconds':round(time.monotonic()-t,1),'finished':now()}
        atomic_json(marker,info);self.status['stages'][name]=info;self.update();self.report()

    def preflight(self):
        rows=[]
        for k,spec in C.REACTION_SPECS.items():
            rv=reaction_distribution(spec);f=float(rv.cdf(180))
            rows.append(dict(event=k,mean=rv.mean(),sd=rv.std(),p95=rv.ppf(.95),
                             probability_all_reacted_by_180=f**1329,
                             meaning='IID pre-movement-only upper bound; ignores travel and queues'))
        (self.out/'tables').mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(self.out/'tables'/'reaction_diagnostics.csv',index=False)
        t=time.monotonic()
        self.mc('pilot',self.jobs(('concert','blocked','lecture','combo')),4,26070000)
        atomic_json(self.out/'pilot_timing.json',dict(seconds=time.monotonic()-t,workers=self.a.workers,
                    note='Pilot includes setup; later throughput also depends on scenario and cache.'))

    def deterministic(self):
        from bailey import paper as P
        from bailey.validation import wheelchair_report,single_door_check
        from bailey.export import export_json
        fig=str(self.out/'figures');res={};qa={};rows=[];checks=[]
        for n in MAIN:
            self.check();r,info=run(SCENARIOS[n].with_(wheelchairs=3),DEFAULT,queue_model=FlowQueueModel())
            if not np.isfinite(r.T) or r.remaining_inside[-1]!=0:raise RuntimeError(f'Deterministic clearance failed: {n}')
            if not np.all(r.remaining+r.exit_cum.sum(axis=1)==1329):raise RuntimeError('Population conservation failed')
            res[n]=r;qa[n]=FlowQueueModel().analyze(r)
            rows.append(dict(case=n,T=r.T,T_ground=r.T_ground,**{k:v for k,v in wheelchair_report(r,DEFAULT).items() if k in ('T_refuge','T_street')}))
            qa[n]['table'].to_csv(self.out/'tables'/f'queue_{n}.csv',index=False)
        pd.DataFrame(rows).to_csv(self.out/'tables'/'deterministic_primary.csv',index=False)
        for n in ('concert','lecture','blocked'):
            for step in (.2,.1,.05):
                self.check();cfg=DEFAULT.with_(phys=dataclasses.replace(DEFAULT.phys,dt=step))
                rr,_=run(SCENARIOS[n].with_(wheelchairs=3),cfg)
                checks.append(dict(case=n,dt=step,T=rr.T))
        pd.DataFrame(checks).to_csv(self.out/'tables'/'time_step_check.csv',index=False)
        single_door_check(DEFAULT).to_csv(self.out/'tables'/'single_door_check.csv',index=False)
        save_pickle(self.out/'results'/'deterministic.pkl',dict(res=res,qa=qa))
        P.fig_evac_curves(res,('normal','concert','blocked','lecture'),fig,title='Primary scenarios: 1329 people including 3 wheelchair users')
        P.fig_evac_curves(res,('concert','barriers','led','combo'),fig,fname='primary_design_curves',title='Concert design comparison')
        P.fig_bottleneck_ranking(qa,('normal','concert','blocked','lecture'),fig)
        numerical_jobs={}
        for n in ('concert','lecture','blocked'):
            for step in (.1,.05):
                cfg=DEFAULT.with_(phys=dataclasses.replace(DEFAULT.phys,dt=step))
                numerical_jobs[f'{n}|dt{step}']=make_job(SCENARIOS[n].with_(wheelchairs=3),cfg,
                    keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('numerical_resolution',numerical_jobs,30,26071120)

    def main(self):self.mc('main',self.jobs(MAIN),300,SEEDS['main'])
    def controls(self):self.mc('controls',self.jobs(MAIN,wheelchairs=0),100,SEEDS['controls'])

    def manager_search(self):
        def progress(*args,**kw):
            print(*args,**kw);self.check()
        search(SCENARIOS['concert'].with_(wheelchairs=3),DEFAULT,self.out/'manager_search',
               candidates=4 if self.a.smoke else 200,screen_reps=self.count(3),
               top_k=2 if self.a.smoke else 10,retest_reps=self.count(30),
               placement_seed=SEEDS['placement'],screen_seed=SEEDS['screen'],retest_seed=SEEDS['retest'],
               workers=self.a.workers,batch_size=2 if self.a.smoke else 5,progress=progress)

    def selected(self):return tuple(json.loads((self.out/'manager_search'/'selected.json').read_text())['manager_nodes'])

    def holdout(self):
        nodes=self.selected();jobs={}
        for n in ('concert','blocked','led','combo'):
            for tag,loc in (('selected',nodes),('reference',None)):
                sc=SCENARIOS[n].with_(wheelchairs=3,manager_nodes=loc,managers=6)
                jobs[f'{tag}|{n}']=make_job(sc,DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('holdout',jobs,200,SEEDS['holdout'])

    def obstructions(self):
        jobs={}
        for base in ('concert','lecture'):
            for target in BLOCK_TARGETS:
                for mode in ('known','arrival','broadcast'):
                    key=f'{base}|{target}|{mode}'
                    sc=SCENARIOS[base].with_(name=key,wheelchairs=3,blocked=(target,),discovery=mode)
                    jobs[key]=make_job(sc,DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('obstructions',jobs,20,SEEDS['obstructions'])
        jobs={}
        for d in (0,10,20,30,45,60,90):
            sc=SCENARIOS['blocked'].with_(wheelchairs=3,broadcast_delay=float(d))
            jobs[str(d)]=make_job(sc,DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('broadcast_delays',jobs,50,SEEDS['obstructions']+100)

    def manager_robustness(self):
        jobs={};nodes=self.selected()
        for r in (5.,10.,15.):
            for p in (.75,.9,.95):
                for tag,loc in (('selected',nodes),('reference',None)):
                    key=f'{tag}|r{r}|p{p}'
                    sc=SCENARIOS['concert'].with_(wheelchairs=3,manager_nodes=loc,manager_radius_m=r,manager_p_follow=p)
                    jobs[key]=make_job(sc,DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        # One unavailable manager is an execution-failure stress test, not a deployable <6-manager plan.
        for i in range(len(nodes)):
            loc=nodes[:i]+nodes[i+1:]
            jobs[f'outage|{nodes[i]}']=make_job(SCENARIOS['concert'].with_(wheelchairs=3,manager_nodes=loc,managers=5),DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('manager_robustness',jobs,20,SEEDS['manager_robustness'])
        jobs={}
        for n in ('led','combo','lecture_led'):
            for p in (0.,.5,.77,1.):
                jobs[f'{n}|follow{p}']=make_job(SCENARIOS[n].with_(wheelchairs=3,compliance=p),DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('led_compliance',jobs,20,SEEDS['manager_robustness']+100)

    def wheelchairs(self):
        from make_figures import _wheel_patch
        jobs={}
        for n in ('concert','blocked','lecture'):
            for guide in ('nearest','main'):
                for delay in (0.,10.,20.,30.):
                    # main guidance with a blocked main exit is deliberately a stress case.
                    with _wheel_patch({'M2':3},delay):
                        jobs[f'{n}|{guide}|delay{delay}']=make_job(SCENARIOS[n].with_(wheelchairs=3,wheelchair_guide=guide),DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        self.mc('wheelchairs',jobs,20,SEEDS['wheelchairs'])

    def cost(self):
        from bailey.cost import design_table,cost_sensitivity
        jobs={};designs={}
        for count in (6,7,8,10,12,15):
            for base,led,bar in (('concert',False,False),('led',True,False),('barriers',False,True),('combo',True,True)):
                key=f'{count}|{base}';designs[key]=dict(managers=count,led=led,barriers=bar)
                jobs[key]=make_job(SCENARIOS[base].with_(wheelchairs=3,managers=count),DEFAULT,keep_curves=False,keep_diagnostics=False,wheel=True)
        r=self.mc('cost',jobs,30,SEEDS['cost'])
        rows=[dict(**designs[k],T_mc=float(np.minimum(v['T'],DEFAULT.phys.t_max).mean())) for k,v in r.items()]
        design_table(rows,DEFAULT,obj='T_mc').to_csv(self.out/'tables'/'cost_pareto.csv',index=False)
        for k,df in cost_sensitivity(rows,DEFAULT,n_draws=2000).items():df.to_csv(self.out/'tables'/f'cost_{k}.csv',index=False)

    def cached_stage(self,key,fn):
        self.check();p=self.out/'results'/f'{key}.pkl'
        if p.exists():return load_pickle(p)
        value=fn();save_pickle(p,value);return value

    def sweeps(self):
        import make_figures as F
        r=self.cached_stage('sweeps',lambda:F.stage_sweeps(self.a.workers))
        for k,v in r.items():
            if isinstance(v,pd.DataFrame):v.to_csv(self.out/'tables'/f'sweep_{k}_no_wheelchairs.csv',index=False)

    def parameters(self):
        from bailey import uncertainty as U
        # Preserve the established parameter experiment (no wheelchair users), explicitly labeled.
        self.cached_stage('tornado',lambda:U.tornado(DEFAULT,scenarios=('normal','concert','blocked','lecture'),workers=self.a.workers)).to_csv(self.out/'tables'/'parameter_tornado_no_wheelchairs.csv',index=False)
        rows=[]
        for start in range(0,self.count(100),self.a.batch_size):
            self.check();K=min(self.a.batch_size,self.count(100)-start)
            seed=int(np.random.SeedSequence([SEEDS['parameters'],start]).generate_state(1)[0])
            p=self.out/'blocks'/'parameters'/f'{start:05d}.pkl'
            if p.exists():r=load_pickle(p)
            else:
                r=U.param_mc(DEFAULT,K=K,seed=seed,workers=self.a.workers);r['block_seed']=seed;r['replicate']=r['m']+start;save_pickle(p,r)
            rows.append(r);pd.concat(rows).to_csv(self.out/'tables'/'parameter_mc_no_wheelchairs.csv',index=False)
        save_pickle(self.out/'results'/'pmc.pkl',pd.concat(rows))

    def validation(self):
        import make_figures as F
        r=self.cached_stage('valid',lambda:F.stage_validation(self.count(20),self.a.workers))
        r['bracket_mc'].to_csv(self.out/'tables'/'capacity_mc_no_wheelchairs.csv',index=False)
        r['bracket_det'].to_csv(self.out/'tables'/'capacity_det_no_wheelchairs.csv',index=False)

    def ablations(self):
        import make_figures as F
        for key,fn in [('var',F.stage_mc_variants),('disp',F.stage_dispersion),('rowcap',F.stage_rowcap)]:
            self.cached_stage(key,lambda fn=fn:fn(self.count(20),self.a.workers))

    def paper(self):
        # Existing paper atlas is an explicitly no-wheelchair control, separate from primary outputs.
        import make_figures as F
        root=self.out/'paper_no_wheelchairs'
        F.FIG=str(root/'figures');F.TAB=str(root/'tables');Path(F.TAB).mkdir(parents=True,exist_ok=True)
        det=self.cached_stage('paper_det',F.stage_det);mk=self.cached_stage('markov',F.stage_markov)
        def get(k):
            p=self.out/'results'/f'{k}.pkl';return load_pickle(p) if p.exists() else None
        mc=get('controls');sw=get('sweeps');var=get('var');torn=get('tornado');pmc=get('pmc')
        xt=dict(disp=get('disp'),rowcap=get('rowcap'),valid=get('valid'))
        F.write_tables(det,mk,mc,var,None,torn,pmc,xt)
        F.plot_all(det,mk,sw,mc,var,None,torn,pmc,xt)

    def report(self):
        lines=['# Overnight experiment report','',f'Updated: {now()}',f'Code SHA256: `{self.hash}`','',
               'Primary results include 3 wheelchair users at M2 within 1329 total. Controls use 0 wheelchair users.',
               'Strict target: T < 180 s. Unfinished runs remain failures; restricted means cap T at 1800 s.',
               'Results under literature-adapted reaction distributions are not a measured Bailey Hall safety certification.',
               '**Known numerical limitation:** the fixed 0.1 s deterministic concert result differs from the diagnostic 0.05 s run by about 14 s (0-wheelchair check). The original <2 s convergence test fails. See numerical_resolution and time_step_check; do not treat small design differences as established.','']
        for label in ('main','holdout','controls'):
            p=self.out/'tables'/f'{label}_summary.csv'
            if p.exists():
                d=pd.read_csv(p);lines += [f'## {label}','', '| Case | K | Mean T (s) | T95 (s) | P(T<180) | Timeouts |','|---|---:|---:|---:|---:|---:|']
                for r in d.to_dict('records'):lines.append(f"| {r['case']} | {r['K']} | {r['T_mean']:.2f} | {r['T95']:.2f} | {r['P_under_180']:.3f} | {r['timeout_rate']:.3f} |")
                lines.append('')
        p=self.out/'manager_search'/'selected.json'
        if p.exists():lines += ['## Selected manager nodes','',', '.join(json.loads(p.read_text())['manager_nodes']),
                               'Selected from sampled layouts; final claims use holdout results, not screening/retest.','']
        lines += ['## Completion status','',json.dumps(self.status['stages'],indent=2),
                  '', 'Raw replicates, paired comparisons and Wilson intervals: tables/. Frozen setup: manifest.json.',
                  'Interrupted batches resume with the same command. Never pool pilot, screening, retest or holdout samples.']
        p=self.out/'REPORT.md';tmp=p.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(p)

    def final_figures(self):
        import matplotlib.pyplot as plt
        figdir=self.out/'figures';figdir.mkdir(exist_ok=True)
        for label in ('main','holdout','controls'):
            p=self.out/'results'/f'{label}.pkl'
            if not p.exists():continue
            r=load_pickle(p);fig,ax=plt.subplots(figsize=(9,4))
            for k,v in r.items():
                x=np.sort(v['T']);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=k)
            ax.axvline(180,color='black',ls='--',lw=1);ax.set(xlabel='Time until last person leaves building (s)',ylabel='Empirical CDF',title=label)
            ax.legend(fontsize=7);fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(figdir/f'{label}_ecdf.{ext}',dpi=180)
            plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=25)
    ap.add_argument('--deadline',default=None,help='ISO datetime with timezone; checkpoint then stop starting work')
    ap.add_argument('--stages',default=','.join(STAGES));ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--commit',default='unpublished')
    a=ap.parse_args()
    if a.workers<1 or a.batch_size<1:ap.error('workers and batch size must be positive')
    requested=a.stages.split(',')
    if any(n not in STAGES for n in requested):ap.error('unknown stage')
    e=Experiment(a);failed=False
    try:
        for name in requested:
            try:e.stage(name)
            except DeadlineReached:
                e.status['state']='checkpointed_at_deadline';break
            except Exception:
                failed=True;e.status['stages'][name]={'state':'failed','traceback':traceback.format_exc()}
                e.update();print(traceback.format_exc(),flush=True)
                # Dependent selection evaluation must not silently use a missing or stale winner.
                if name in ('preflight','deterministic','manager_search'):break
        else:e.status['state']='completed_with_errors' if failed else 'complete'
        if failed:e.status['state']='completed_with_errors'
    finally:
        e.final_figures();e.update();e.report()
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

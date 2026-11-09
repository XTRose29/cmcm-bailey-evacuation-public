"""Freeze source and start a durable, sleep-inhibited local experiment process."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--deadline',default=None,help='Optional hard stop; omit to finish every stage')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--commit',required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parent;out=Path(a.out).resolve()
    import xml.etree.ElementTree as ET
    test_report=root/'outputs'/'frozen-tests.xml'
    if not test_report.exists():raise SystemExit('Run python -m pytest -q --junitxml=outputs/frozen-tests.xml first')
    cases=ET.parse(test_report).getroot().findall('.//testcase')
    failures=[c.attrib['name'] for c in cases if c.find('failure') is not None or c.find('error') is not None]
    if any(n!='test_time_step_converged' for n in failures):raise SystemExit(f'Unexpected test failures: {failures}')
    if (out/'launch.json').exists():raise SystemExit('Run already exists; inspect status/PID before resuming frozen command.')
    code=out/'source';code.mkdir(parents=True,exist_ok=True)
    for name in ('bailey','tests','docs'):
        shutil.copytree(root/name,code/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pdf'))
    for name in ('run_overnight.py','make_figures.py','search_managers.py','run.py','README.md','requirements.txt'):
        if (root/name).exists():shutil.copy2(root/name,code/name)
    shutil.copy2(test_report,out/'validation-tests.xml')
    (out/'validation.json').write_text(json.dumps(dict(tests=len(cases),failures=failures,known_limitation='0.1 s user-selected production step; original convergence tolerance retained'),indent=2)+'\n')
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    env['MPLBACKEND']='Agg';env['BAILEY_CACHE']=str(out/'plan_cache')
    # Warm starts are parameter/version keyed. Copy only current compatible plans;
    # cache keys include engine version, reaction spec, geometry and all design inputs.
    cache=root/'outputs'/'cache'
    if cache.exists():shutil.copytree(cache,out/'plan_cache',dirs_exist_ok=True)
    command=[sys.executable,'-u',str(code/'run_overnight.py'),'--out',str(out),'--workers',str(a.workers),
             '--commit',a.commit]
    if a.deadline:command += ['--deadline',a.deadline]
    caffeinate=shutil.which('caffeinate')
    if caffeinate:command=[caffeinate,'-i','-s']+command
    with (out/'run.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    data=dict(pid=process.pid,command=command,cwd=str(code),started=dt.datetime.now(ZoneInfo('America/New_York')).isoformat(),
              deadline=a.deadline,commit=a.commit,log=str(out/'run.log'),report=str(out/'REPORT.md'))
    (out/'launch.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()

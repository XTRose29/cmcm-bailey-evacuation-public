"""Random node placement screening followed by independent repeated evaluation.

Example smoke: python search_managers.py --candidates 4 --screen-reps 2 --top-k 2 --retest-reps 3 --out outputs/manager-smoke
Run without small-count flags only when ready for the full configured search.
"""
import argparse
from bailey.config import DEFAULT, MANAGERS, MANAGER_SEARCH
from bailey.scenarios import SCENARIOS
from bailey.manager_search import search


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario',choices=['concert','blocked','barriers','led','combo','wheelchair'],default='concert')
    ap.add_argument('--managers',type=int,default=MANAGERS['law_min'])
    ap.add_argument('--radius',type=float,default=MANAGERS['radius_m'])
    ap.add_argument('--p-follow',type=float,default=MANAGERS['p_follow'])
    ap.add_argument('--wheelchairs',type=int,default=None)
    for key in ('candidates','screen_reps','top_k','retest_reps','placement_seed','screen_seed','retest_seed'):
        ap.add_argument('--'+key.replace('_','-'),type=int,default=MANAGER_SEARCH[key])
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=10)
    ap.add_argument('--out',default='outputs/manager-search')
    a=ap.parse_args()
    sc=SCENARIOS[a.scenario].with_(managers=a.managers,manager_radius_m=a.radius,manager_p_follow=a.p_follow)
    if a.wheelchairs is not None: sc=sc.with_(wheelchairs=a.wheelchairs)
    result=search(sc,DEFAULT,a.out,**{k:getattr(a,k) for k in MANAGER_SEARCH},workers=a.workers,batch_size=a.batch_size)
    print('Selected:',result['manager_nodes'])
    print('Independent retest:',result['retest_summary'])
    print('Saved:',a.out)


if __name__=='__main__': main()

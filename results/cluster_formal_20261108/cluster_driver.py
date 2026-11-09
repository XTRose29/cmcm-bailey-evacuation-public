'Cluster driver for the frozen Bailey Hall overnight experiments.\n\nModel code and run_overnight.py are untouched:\n    All missing blocks of a stage share one big process pool. Block seeds are unchanged and every\n    replicate depends only on (seed, m), so results equal the sequential block-by-block run.\n    Manager search sends more layouts per pool call (per-layout seeds unchanged, same results).\n    LED plan caches are filled in parallel before a stage (originally built one by one).\n    status.json / REPORT.md / manifest.json are left to the final run_overnight.py pass.\nRun with the frozen source directory as the working directory.'
import argparse
import datetime as dt
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.getcwd())
import run_overnight as RO          # sets BLAS threads to 1 first
import numpy as np
from bailey import montecarlo as MC

# stages whose plans are pre-built in parallel
WARM = ('main', 'controls', 'holdout', 'obstructions', 'manager_robustness', 'cost')


def log(msg):
    print(f'[{dt.datetime.now(RO.TZ).isoformat(timespec="seconds")}] {msg}', flush=True)


def _warm_one(call):
    'build one scenario in a worker to fill the disk cache.'
    args, kw = call
    t = time.monotonic()
    MC.make_job(*args, **kw)
    return time.monotonic() - t


class Driver(RO.Experiment):
    def __init__(self, a):
        #
        self.a = a
        self.out = Path(a.out).resolve()
        for folder in ('tables', 'figures', 'results', 'blocks', 'completed'):
            (self.out / folder).mkdir(parents=True, exist_ok=True)
        self.hash = RO.digest()
        self.deadline = None
        self.status = {'stages': {}, 'state': 'running'}
        self.scanning = False

    def update(self):
        pass

    def report(self):
        pass

    def mc(self, label, jobs, K, seed):
        if self.scanning:
            return {}
        K = self.count(K)
        folder = self.out / 'blocks' / label
        folder.mkdir(parents=True, exist_ok=True)
        per_block = max(1, len(jobs) * self.a.batch_size)
        group = max(1, self.a.tasks_per_pool // per_block)
        missing = {}
        for start in range(0, K, self.a.batch_size):
            if not (folder / f'{start:05d}.pkl').exists():
                missing.setdefault(min(self.a.batch_size, K - start), []).append(start)
        for size, starts in missing.items():
            for i in range(0, len(starts), group):
                chunk = starts[i:i + group]
                self.check()
                merged = {}
                for start in chunk:
                    # same block seed as Experiment.mc
                    block_seed = int(np.random.SeedSequence([seed, start]).generate_state(1)[0])
                    for key, job in jobs.items():
                        merged[f'{start:05d}#{key}'] = dict(job, seed=block_seed)
                t = time.monotonic()
                res = MC.run_batch(merged, size, self.a.workers)
                for start in chunk:
                    RO.save_pickle(folder / f'{start:05d}.pkl', {key: res[f'{start:05d}#{key}'] for key in jobs})
                log(f'{label}: blocks {chunk} ({len(jobs)} cases x {size} reps each) done in {time.monotonic() - t:.0f} s')
        # all blocks on disk; original code loads and exports
        return RO.Experiment.mc(self, label, jobs, K, seed)

    def manager_search(self):
        # same arguments, larger batch_size
        def progress(*args, **kw):
            print(*args, **kw)
            self.check()
        RO.search(RO.SCENARIOS['concert'].with_(wheelchairs=3), RO.DEFAULT, self.out / 'manager_search',
                  candidates=200, screen_reps=self.count(3), top_k=10, retest_reps=self.count(30),
                  placement_seed=RO.SEEDS['placement'], screen_seed=RO.SEEDS['screen'],
                  retest_seed=RO.SEEDS['retest'], workers=self.a.workers,
                  batch_size=self.a.search_batch, progress=progress)

    def warm(self, name):
        'dry-scan the stage for make_job calls, then build them in parallel.'
        calls, seen = [], set()

        def record(*args, **kw):
            k = repr((args, sorted(kw.items())))
            if k not in seen:
                seen.add(k)
                calls.append((args, kw))
            return {'setup': {}}

        real = RO.make_job
        RO.make_job = record
        self.scanning = True
        try:
            getattr(self, name)()
        except Exception as e:  # post-processing fails on empty results
            log(f'{name}: scan ended early ({type(e).__name__}); {len(calls)} scenarios recorded')
        finally:
            RO.make_job = real
            self.scanning = False
        if not calls:
            return
        t = time.monotonic()
        with ProcessPoolExecutor(max_workers=min(len(calls), self.a.workers)) as ex:
            secs = list(ex.map(_warm_one, calls))
        log(f'{name}: built {len(calls)} scenario plans in {time.monotonic() - t:.0f} s (slowest {max(secs):.0f} s)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--stages', required=True)
    ap.add_argument('--batch-size', type=int, default=25)   # must match the formal run
    ap.add_argument('--tasks-per-pool', type=int, default=0, help='default 4 x workers')
    ap.add_argument('--search-batch', type=int, default=100)
    a = ap.parse_args()
    a.smoke = False
    a.tasks_per_pool = a.tasks_per_pool or 4 * a.workers
    stages = a.stages.split(',')
    if any(s not in RO.STAGES for s in stages):
        ap.error('unknown stage')
    d = Driver(a)
    log(f'host {platform.node()} python {sys.version.split()[0]} numpy {np.__version__} '
        f'workers {a.workers} code_hash {d.hash} cache {os.environ.get("BAILEY_CACHE")}')
    for name in stages:
        if (d.out / 'completed' / f'{name}.json').exists():
            log(f'{name}: already complete, skipped')
            continue
        t = time.monotonic()
        log(f'{name}: start')
        if name in WARM:
            d.warm(name)
        d.stage(name)
        log(f'{name}: complete in {time.monotonic() - t:.0f} s')


if __name__ == '__main__':
    main()

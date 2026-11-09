"""Export model results as JSON and CSV, keyed by building-cell ID."""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from .flow import SimResult


def _r(a, nd=3):
    return [round(float(x), nd) for x in a]


def export_json(res: SimResult, path: str, every: int = 1) -> str:
    b = res.b
    sl = slice(None, None, every)
    data = {
        "meta": {**{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                    for k, v in res.meta.items()},
                 "T": res.T if np.isfinite(res.T) else None,
                 "T_ground": res.T_ground if np.isfinite(res.T_ground) else None,
                 "mean_time": res.mean_time,
                 "exit_groups": res.exit_groups(),
                 "schema": "bailey-integer-v2"},
        "states": [dict(id=c.id, zh=c.zh, kind=c.kind, level=c.level, x=round(c.xy[0], 3),
                        y=round(c.xy[1], 3), outside=c.outside, exit_group=c.exit_group,
                        mu_in=None if not np.isfinite(c.inflow_cap) else c.inflow_cap,
                        n_max=None if not np.isfinite(c.n_max) else round(c.n_max, 2))
                   for c in b.cells],
        "edges": [{"from": i, "to": k,
                   "cap": b.edge_cap.get((i, k)),
                   "fallback": (i, k) in b.reverse} for i, k in b.edges],
        "time": _r(res.t[sl], 3),
        "occupancy": {c: res.occ[sl, b.idx[c]].tolist() for c in b.ids},
        "queue_in": {c: res.queue_in[sl, b.idx[c]].tolist() for c in b.ids},
        "edge_flow": [res.flows[sl, m].tolist() for m in range(len(res.edges))],
        "remaining_inside": res.remaining_inside[sl].tolist(),
        "remaining_to_street": res.remaining[sl].tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def export_timeseries_csv(res: SimResult, path: str) -> str:
    'long table for spreadsheets.'
    b = res.b
    rows = []
    for c in b.ids:
        k = b.idx[c]
        rows.append(pd.DataFrame({"time": res.t, "state": c, "people": res.occ[:, k],
                                  "queue_in": res.queue_in[:, k]}))
    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


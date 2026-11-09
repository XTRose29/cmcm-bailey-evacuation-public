# Revised paper costs

These tables apply the sourced cost inputs in `model_v2/bailey/config.py` to the unchanged 30-run design-comparison evacuation times from the formal archive.

- `cost_recomputed_latest.csv`: annual costs and cost–time frontier.
- `cost_recomputed_oat.csv`: one-at-a-time cost sensitivity.
- `cost_recomputed_freq.csv`: frequency on the frontier across 2,000 sampled cost sets.

Regenerate all three with `python scripts/recompute_costs.py`. These tables supersede the historical cost columns, not the archived evacuation times.

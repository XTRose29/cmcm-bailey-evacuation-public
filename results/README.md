# Results

| Directory | Contents |
|---|---|
| `formal/tables/` | Main outcomes, paired comparisons, controls, obstruction/guidance experiments, sensitivity and numerical checks |
| `formal/manager_search/` | All layout evaluations in `replicates.csv`, rankings, selected placement and setup metadata |
| `formal/no_wheelchairs/tables/` | Additional diagnostics without wheelchair users |
| `costs/` | Revised annual costs and cost-only uncertainty analysis |
| `verification/` | Source-comparison audit and publication check records |

`formal/manifest.json` records the historical simulation parameters and seeds; `formal/status.json` records completion of all fifteen stages. Its cost inputs predate the revised cost analysis. Use `costs/` for the updated estimates.

The archive stores numeric outcomes rather than large binary caches or generated figures. Per-layout JSON duplicates have been consolidated into the complete manager-search CSV and setup records. See `../docs/REPRODUCIBILITY.md` for source versions and limitations.

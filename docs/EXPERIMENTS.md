# Formal experiment plan

Use `model_v2/run_overnight.py` as the executable specification. The archived run uses 0.1 s steps, a 1,800 s horizon, batch size 25, and three wheelchair users unless a stage explicitly specifies otherwise. Population is always 1,329. Read completed stage records in the archived `status.json`.

| Stage | Design and purpose |
|---|---|
| Preflight | Reaction-distribution diagnostics and a four-replicate pilot |
| Deterministic | Nine primary situations; population conservation, queues, wheelchair times, one-door and time-step checks; 30-run Monte Carlo time-step comparisons |
| Main | 300 runs for each of nine situations: normal, concert, blocked, lecture, barriers, LEDs, combined, lecture LEDs, lecture barriers |
| Manager search | 200 layouts including the reference; three screening runs; top ten plus reference retested with 30 runs |
| Holdout | Independent 200-run evaluation of selected and reference manager layouts |
| Controls | 100 runs per primary situation without wheelchair users |
| Obstructions | Blocked-route and discovery alternatives, 20 runs each; broadcast-delay comparison, 50 runs each |
| Manager robustness | Placement, radius and following-probability variants, 20 runs each; LED compliance variants, 20 runs each |
| Wheelchairs | Wheelchair number, placement/guidance and assistance cases, 20 runs each |
| Cost | Manager counts, LEDs and barriers, 30 runs each; annual cost comparison |
| Sweeps | Deterministic route-choice, barrier, blockage and compliance sweeps without wheelchair users |
| Parameters | Deterministic parameter sensitivity and sampled uncertain parameters without wheelchair users |
| Validation | One-door and passage-flow/capacity comparisons, including uncertainty checks |
| Ablations | Reaction/service randomness and seat-row-capacity checks |
| Paper | Assemble figures and reports from completed outputs |

The complete parameter grids and replicate counts for subsidiary checks are defined in the corresponding `Experiment` methods. Do not substitute main-experiment results for the separate 30-run cost comparison.

## Commands

From the repository root:

```bash
python model_v2/run_overnight.py --out outputs/reproduction --workers 8 --batch-size 25 \
  --commit "$(git rev-parse HEAD)"
```

Use a fresh output directory. Completed stages are checkpointed and the same command resumes unfinished work. A `--smoke` run is for setup checks, not paper results. Changing worker count does not change the prescribed seed mapping. Changing batch size changes the block seed allocation and therefore the sample realizations.

The optional cluster driver is preserved at `results/cluster_formal_20261108/cluster_driver.py`. Run it with `model_v2` as the working directory; see `--help`. It changes scheduling and cache warm-up, not model equations. Site-specific Slurm account settings are intentionally not included.

## Seeds and reporting

Suite seeds are in `run_overnight.SEEDS` and the archived manifest: main 26071101, controls 26071102, holdout 26071103, obstructions 26071104, manager robustness 26071105, wheelchairs 26071106, cost 26071107, placement 26071108, screening 26071109, retest 26071110, and parameters 26071111. Subsidiary stages use the explicit offsets in the runner.

Report mean, median, 95th percentile, completion/timeout rate, probability of finishing strictly before 180 s, and its Wilson interval. Guidance comparisons use paired differences. Manager selection and evaluation use different seeds. Preserve incomplete runs rather than discarding them.

## Cost update

The archived formal manifest contains the historical cost assumptions. Updated cost inputs are in `model_v2/bailey/config.py`; `scripts/recompute_costs.py` applies them to the same archived evacuation times. The resulting three CSV files in `results/paper_cost_update/` are the revised paper cost analysis. They include 2,000 cost-only uncertainty draws with seed 2026; these are not extra evacuation simulations.

# Bailey Hall evacuation: model, experiments and visualization

Public research companion for the 2026 Cornell Mathematical Modeling paper on Bailey Hall. The model follows **1,329 whole people**, including **three wheelchair users in M2**, through a network of seating areas, corridors, doors and stairs.

**[Open the interactive visualization](https://xtrose29.github.io/bailey-evacuation/index.html)** · [Experiment tables](results/cluster_formal_20261108/out/tables) · [Paper figures](paper/figures) · [Appendix LaTeX](paper/appendix_implementation.tex)

## What is included

| Directory | Contents |
|---|---|
| `model_v2/bailey/` | Final Python model, parameters, integer movement, guidance, queues and Monte Carlo sampling |
| `model_v2/run_overnight.py` | Checkpointed formal experiment runner and reporting definitions |
| `model_v2/tests/` | Model checks, including the retained time-step convergence test |
| `results/cluster_formal_20261108/out/` | Archived formal tables, individual run summaries, manager-search candidates, seeds and run manifest |
| `results/paper_cost_update/` | Updated costs using sourced inputs; evacuation times are unchanged |
| `paper/figures/` | Seven PDF figures used in the paper |
| `paper/source_data/` | Exact CSV inputs for those figures |
| `web_v2/` | English, self-contained results explorer and final-model replay data |
| `docs/` | Experiment plan, implementation notes and provenance |

## Main findings

The main experiments use 300 Monte Carlo simulations per situation, a 0.1 s time step, and three wheelchair users within the 1,329-person total.

| Situation | Mean time to leave the building (s) |
|---|---:|
| Normal concert dismissal | 413.21 |
| Concert emergency | 258.59 |
| Lecture emergency | 331.83 |
| Concert emergency with the main exit blocked | 416.14 |
| Concert with corridor barriers | 258.51 |
| Concert with LED guidance | 257.10 |
| Concert with barriers and LEDs | 256.69 |

Reaction times and passage capacity strongly affect the predictions. The results suggest prioritizing earlier responses, available exit routes and timely information about blocked routes. LED guidance offers a smaller additional benefit under the baseline assumptions. None of the main simulations finished in under 180 s; this is a model result, not a measurement of Bailey Hall.

## Install and run

Python 3.10 or newer is required. The archived formal run used Python 3.10.20; use the pinned numerical environment for the closest reproduction.

```bash
git clone https://github.com/XTRose29/cmcm-bailey-evacuation-public.git
cd cmcm-bailey-evacuation-public
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-paper.txt
python model_v2/run.py params
python -m pytest model_v2/tests -q
```

**Known numerical limitation:** `test_time_step_converged` retains its original two-second tolerance and can fail. Archived deterministic concert times are 221.3 s at 0.1 s and 213.65 s at 0.05 s. Do not loosen the test to conceal this difference. See [validation notes](docs/VALIDATION.md).

For the paper's nine situations, including three wheelchair users:

```bash
python model_v2/run_overnight.py --out outputs/reproduction --workers 8 --batch-size 25 \
  --stages deterministic,main --commit "$(git rev-parse HEAD)"
```

For the complete experiment plan, omit `--stages`. This is a substantial computation, not a quick preview. Do not point a new run at the archived results folder. The simpler `run.py concert` command uses the scenario default of zero wheelchair users; use the formal runner or explicitly set `wheelchairs=3` in Python for the paper population.

To use the existing results without rerunning simulations:

```bash
python scripts/verify_release.py
python scripts/recompute_costs.py
python paper/make_figures.py
python web_v2/scripts/build.py
```

Open `web_v2/index.html` directly in a browser. To regenerate deterministic replay data from the Python model, run `python web_v2/scripts/export_data.py` after removing only the replay JSON files you intend to regenerate. The exporter checks every replay's clearance time against the archived deterministic table. It reuses existing replay files by default.

## Versions and interpretation

- The archived formal simulations used source commit `9d76e442fd31b99a770485548fcb504b56031cc4`.
- The publication source was copied from `e4201d3239bdb51d21e73e7aefd3ce81a6dc0d1f`, which updates cost inputs without changing movement equations.
- This public edition translates comments, documentation and display labels into English. Model calculations and numerical parameters are preserved. Its file hashes therefore differ from the historical run hash.
- The historical manifest and historical cost tables remain in the archive. **Use `results/paper_cost_update/` for the paper's revised costs.**
- The website replays outputs from the final Python engine. It replaces the older JavaScript simulator, whose parameters did not match the paper. Replay finish times are deterministic results and must not be confused with Monte Carlo averages.
- Historical binary caches, large pickle files, incomplete runs, draft manuscripts and private working notes are not required for this public package. Archived CSVs can be inspected without loading pickle files; missing simulation caches are regenerated when rerunning experiments.

See [provenance](docs/PROVENANCE.md), the [full experiment plan](docs/EXPERIMENTS.md), and [implementation details](docs/IMPLEMENTATION.md).

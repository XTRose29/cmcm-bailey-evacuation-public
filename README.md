# Bailey Hall Evacuation

A network model of 1,329 people leaving Bailey Hall under normal and emergency conditions. The Python simulation combines integer populations, route choices, walking stages, capacity-limited queues and Monte Carlo sampling.

**[Interactive visualization](https://xtrose29.github.io/bailey-evacuation/index.html)** · [Model](docs/MODEL.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Repository structure

```text
model/       Python model, experiment runners and tests
results/     Formal experiment tables and revised cost analysis
web/         Self-contained visualization and replay data
scripts/     Data verification, cost updates and deployment
docs/        Model, experiment and reproducibility notes
```

The repository contains code and research data. Manuscript files, LaTeX appendices and pre-rendered paper figures are maintained separately.

## Quick start

Use Python 3.10 or newer. Dependencies are pinned to the numerical libraries used for the formal Linux experiments.

```bash
git clone https://github.com/XTRose29/cmcm-bailey-evacuation-public.git
cd cmcm-bailey-evacuation-public
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model/run.py params
python scripts/verify_release.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run experiments

The primary experiments use 0.1 s time steps and three wheelchair users in M2, included in the total population of 1,329.

```bash
python model/run_overnight.py --out outputs/reproduction --workers 8 \
  --batch-size 25 --stages deterministic,main
```

Omit `--stages` to run the full experiment plan. Use a new output directory; completed stages are checkpointed. Keep the seeds and batch size unchanged when reproducing the archived sampling scheme. The simpler `model/run.py` interface uses zero-wheelchair defaults for most individual scenarios.

## Results

| Situation | Mean clearance time (s), 300 runs |
|---|---:|
| Normal concert dismissal | 413.21 |
| Concert emergency | 258.59 |
| Lecture emergency | 331.83 |
| Main exit blocked | 416.14 |
| Concert with barriers | 258.51 |
| Concert with LEDs | 257.10 |
| Concert with barriers and LEDs | 256.69 |

Full summaries and individual outcomes are in [`results/formal/tables/`](results/formal/tables). Manager search records are in [`results/formal/manager_search/`](results/formal/manager_search). Updated annual costs are in [`results/costs/`](results/costs); these supersede the historical cost columns in the formal archive.

## Visualization

Open `web/index.html` locally or use the [live page](https://xtrose29.github.io/bailey-evacuation/index.html). The page displays nine deterministic Python-model replays and separate archived Monte Carlo statistics.

```bash
python web/scripts/build.py
python web/scripts/build.py --check
```

To regenerate replay data from the model, run `python web/scripts/export_data.py --refresh`. To update costs without rerunning evacuations, run `python scripts/recompute_costs.py`.

## Checks and limitations

```bash
python -m pytest model/tests -q
python scripts/verify_release.py
```

The original time-step convergence test is retained and can fail; the publication check recorded 96 passes and this one known failure. The model has not been calibrated against a measured full evacuation of Bailey Hall. See [reproducibility and validation notes](docs/REPRODUCIBILITY.md) for the exact source versions, environment and numerical limitations.

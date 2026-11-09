# Bailey Hall Evacuation

Python model and results for 1,329 people leaving Bailey Hall.

**[Interactive visualization](https://xtrose29.github.io/bailey-evacuation/index.html)**

- `model/`: model implementation and experiment runners.
- `results/formal/`: archived simulation outcomes, scenario settings and manager-placement results.
- `results/costs/`: updated cost estimates.
- `web/`: English visualization, source and recorded model data.

## Run

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python model/run_overnight.py --out outputs/reproduction --workers 8 --batch-size 25 --stages deterministic,main
```

Omit `--stages` to run all experiments. The formal runner uses 0.1 s steps and three wheelchair users within the total population of 1,329. Individual scenarios run through `model/run.py` generally default to zero wheelchair users.

Update costs using the archived simulation times:

```bash
python model/recompute_costs.py
```

Open `web/index.html` locally, or rebuild it with `python web/scripts/build.py`. Use `python web/scripts/export_data.py --refresh` to regenerate the deterministic replays. Replay times are distinct from the archived Monte Carlo averages.

The visualization retains the seat-row layout and individual moving dots. `web/scripts/person_tracks.py` reconstructs display identities from the final model's integer transfers, checking population counts at every step. `web/src/layout.js` provides drawing geometry; `web/src/people.js` places the dots along schematic paths. These display paths do not add a second movement model or change the saved results.

The formal simulation source was `9d76e44`; final cost inputs came from `e4201d3`. The historical manifest retains the earlier cost assumptions; use `results/costs/` for updated costs. The model has unresolved time-step sensitivity, documented by the numerical results in `results/formal/tables/`, and has not been calibrated against a full measured evacuation.

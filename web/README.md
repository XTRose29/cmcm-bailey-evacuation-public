# Visualization

Live page: https://xtrose29.github.io/bailey-evacuation/index.html

- `src/`: editable HTML template and interaction code.
- `data/results.json`: nine deterministic replays and archived experiment summaries.
- `scripts/export_data.py`: export or refresh final Python-model replays.
- `scripts/build.py`: build the self-contained `index.html`.

Run from the repository root:

```bash
python web/scripts/export_data.py           # reuse verified replays
python web/scripts/export_data.py --refresh # recompute all nine replays
python web/scripts/build.py
```

The browser replays Python output; it does not implement another movement solver. Scenario, floor, population/queue, speed and time controls affect the display only. Deterministic finish times are distinct from the Monte Carlo averages.

The live page is served by `XTRose29/xtrose29.github.io`, branch `master`, at `bailey-evacuation/index.html`. With an authenticated GitHub CLI, `python scripts/publish_visualization.py --publish` updates only that HTML file. Without `--publish`, the script performs a dry run.

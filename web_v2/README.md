# Final-model results explorer

Live URL: https://xtrose29.github.io/bailey-evacuation/index.html

This English HTML page combines nine deterministic Python-model replays, archived 300-run scenario statistics, paired guidance effects and updated costs. It replaces the early JavaScript simulator. It does not claim that deterministic replay times equal Monte Carlo means.

From the repository root:

```bash
python web_v2/scripts/export_data.py  # reuses existing replay JSONs; computes missing ones
python web_v2/scripts/build.py
python web_v2/scripts/build.py --check
```

Open `web_v2/index.html` in a browser. It is self-contained and works offline. Scenario, floor, queue/population view, playback speed and time controls operate on recorded final-model output. No model parameter changes occur in the browser.

The stable live URL is served by `XTRose29/xtrose29.github.io`, branch `master`, file `bailey-evacuation/index.html`. The source is maintained in this public research repository. To synchronize an intentional rebuild using an authenticated GitHub CLI, run `python scripts/publish_visualization.py --publish`; the default is a dry run. This updates only that HTML file.

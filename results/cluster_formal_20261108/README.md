# Archived formal experiments

`out/` contains the numeric tables, individual replicate summaries, manager-placement search records, historical run manifest and completion status from the final cluster experiment. All fifteen stages completed. The source version and environment are documented in `../../docs/PROVENANCE.md`.

The historical cost tables and `manifest.json` predate the final cost-input update. Use `../paper_cost_update/` for the revised costs in the paper. Evacuation outcomes are unchanged.

The public copy omits large binary simulation caches, redundant images and cluster scheduler logs. Main run-level outcomes, paired comparisons, setup JSONs, and sensitivity/validation tables are included. Regenerate the seven paper figures with `python paper/make_figures.py` from the repository root.

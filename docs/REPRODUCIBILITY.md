# Source and result provenance

The formal results were produced on Cornell's Unicorn cluster with Python 3.10.20 on Linux, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, Matplotlib 3.10.8 and NetworkX 3.4.2.

- Formal simulation source: `9d76e442fd31b99a770485548fcb504b56031cc4`.
- Historical source digest (LF content, POSIX paths): `aa1b1a84a425539fa40ae6e840cd5d185cf9d6affbb045e623fa1c60f6d985c5`.
- Publication source snapshot: `e4201d3239bdb51d21e73e7aefd3ce81a6dc0d1f`. Its last update supplies sourced cost inputs; movement results are reused.

These identifiers refer to the original development history and are recorded for provenance; that private history is not a dependency for downloading or running this public package. This repository contains the current source and archived numeric tables directly.

Comments, docstrings, human-readable scenario labels and building labels were translated into English. Historical label fields named `zh` retain their API names for compatibility but now contain English text. Numeric values, computational expressions, parameter values, seeds and test assertions were not changed. `results/verification/source_translation_audit.json` records the syntax-tree comparison against the original snapshot.

The archived manifest retains the historical source hash and cost inputs. It is not a hash of this English edition. `release_manifest.json` instead records hashes of the published source, tables and replay data. It intentionally excludes itself and build caches. For exact citation, use the public commit associated with the release in addition to this manifest.

Archived CSV tables and manager-search records are provided without the large pickle caches. The partial Windows run and early prototypes are excluded. The historical cost analysis remains under the formal archive; the revised paper costs are clearly separated in `results/costs/`.

The schematic geometry originates from the building diagrams used in the CMCM problem and the model's diagram code. They are illustrative, not surveyed measurements. No claim of ownership over third-party building diagrams or research references is made.

## Validation and remaining numerical limitations

This model has implementation checks and sensitivity experiments, but has not been calibrated against a measured full evacuation of Bailey Hall.

The source tests cover integer populations, conservation, passage/storage limits, reaction distributions, local guidance, wheelchair setup, paired random streams and the experiment runner. Numerical convergence remains an open issue: `test_time_step_converged` compares deterministic time steps against a two-second tolerance and is retained unchanged.

Archived deterministic concert clearance is 221.3 s at dt=0.1 and 213.65 s at dt=0.05. The separate 30-run Monte Carlo means are 258.35 s and 258.195 s. Neither agreement in sample averages nor an integer population eliminates the need for further numerical checks.

`scripts/verify_release.py` checks that archived main summaries agree with the replicate CSV, replay counts remain whole and conserve the population, replay clearance times match the archived deterministic table, costs match the updated calculation, and the published source/data manifest is intact. These are publication consistency checks, not validation against real building observations.

The webpage intentionally has no independent JavaScript movement solver. Its playback frames come from the final Python engine at dt=0.1, sampled about once per second for display. Its Monte Carlo tables are read from the archived formal experiment.

Publication checks on the available local environment passed 96 model tests and retained one failure. That test uses the zero-wheelchair scenario default: its times were 227.7 s and 213.65 s. These are distinct from the paper's three-wheelchair deterministic times above. See `../results/verification/publication_checks.json` for environment details.

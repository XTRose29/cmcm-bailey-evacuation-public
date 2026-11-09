# Source and result provenance

The formal results were produced on Cornell's Unicorn cluster with Python 3.10.20 on Linux, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, Matplotlib 3.10.8 and NetworkX 3.4.2.

- Formal simulation source: `9d76e442fd31b99a770485548fcb504b56031cc4`.
- Historical source digest (LF content, POSIX paths): `aa1b1a84a425539fa40ae6e840cd5d185cf9d6affbb045e623fa1c60f6d985c5`.
- Publication source snapshot: `e4201d3239bdb51d21e73e7aefd3ce81a6dc0d1f`. Its last update supplies sourced cost inputs; movement results are reused.

These identifiers refer to the original development history and are recorded for provenance; that private history is not a dependency for downloading or running this public package. This repository contains the current source and archived numeric tables directly.

Comments, docstrings, human-readable scenario labels and building labels were translated into English. Historical label fields named `zh` retain their API names for compatibility but now contain English text. Numeric values, computational expressions, parameter values, seeds and test assertions were not changed. `docs/source_translation_audit.json` records the syntax-tree comparison against the original snapshot.

The archived manifest retains the historical source hash and cost inputs. It is not a hash of this English edition. `release_manifest.json` instead records hashes of the published source, tables, replay data and figures. It intentionally excludes itself and build caches. For exact citation, use the public commit associated with the release in addition to this manifest.

Archived CSV tables and manager-search records are provided without the large pickle caches. The seven final paper figures and their exact source CSVs are included. The partial Windows run and early prototypes are excluded. The historical cost analysis remains under the formal archive; the revised paper costs are clearly separated in `results/paper_cost_update/`.

The floor-plan assets and schematic geometry originate from the building diagrams used in the CMCM problem and the model's diagram code. They are illustrative, not surveyed measurements. No claim of ownership over third-party building diagrams or research references is made.

# Validation and remaining numerical limitations

This model has implementation checks and sensitivity experiments, but has not been calibrated against a measured full evacuation of Bailey Hall.

The source tests cover integer populations, conservation, passage/storage limits, reaction distributions, local guidance, wheelchair setup, paired random streams and the experiment runner. Numerical convergence remains an open issue: `test_time_step_converged` compares deterministic time steps against a two-second tolerance and is retained unchanged.

Archived deterministic concert clearance is 221.3 s at dt=0.1 and 213.65 s at dt=0.05. The separate 30-run Monte Carlo means are 258.35 s and 258.195 s. Neither agreement in sample averages nor an integer population eliminates the need for further numerical checks.

`scripts/verify_release.py` checks that archived main summaries agree with the replicate CSV, replay counts remain whole and conserve the population, replay clearance times match the archived deterministic table, costs match the updated calculation, and the published source/data manifest is intact. These are publication consistency checks, not validation against real building observations.

The webpage intentionally has no independent JavaScript movement solver. Its playback frames come from the final Python engine at dt=0.1, sampled about once per second for display. Its Monte Carlo tables are read from the archived formal experiment.

Publication checks on the available local environment passed 96 model tests and retained one failure. That test uses the zero-wheelchair scenario default: its times were 227.7 s and 213.65 s. These are distinct from the paper's three-wheelchair deterministic times above. See `publication_checks.json` for environment details.

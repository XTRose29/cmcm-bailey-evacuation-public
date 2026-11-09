# Final Python model

See the root README for installation, the formal runner, result versions and known numerical limitations. Parameter definitions are in `bailey/config.py`.

The paper population is 1,329 people including three wheelchair users in M2. Use `run_overnight.py` for paper experiments. The single-scenario `run.py` interface keeps its original scenario defaults; most of those defaults have no wheelchair users.

Run tests from the repository root with `python -m pytest model_v2/tests -q`. The time-step convergence assertion is intentionally retained; consult `docs/VALIDATION.md` before interpreting failures.

# Key implementation details

The authoritative computation is Python. The website displays exported Python replays and archived experiment summaries.

| File | Responsibility |
|---|---|
| `model/bailey/config.py` | Physical, geometric, reaction-time, behavior, accessibility and cost inputs, with source notes |
| `building.py` | Directed building network, storage limits, closures and outside areas |
| `integer.py` | Whole-person class allocation, route allocation, capacity credits and storage limiting |
| `routing.py` | Distance, familiar-route, queue and guidance scores; stable logit probabilities |
| `guidance.py` | Crowd Manager coverage using same-floor passage distances |
| `flow.py` | Walking stages, queues, capacity-limited integer movements and clearance times |
| `plugins/randomness.py` | Reaction distributions, stochastic walking and variable passage service |
| `montecarlo.py` | Independent random streams and parallel replicates |
| `manager_search.py` | Random placement screening and independent retesting |
| `validation.py` | Analytic checks and wheelchair refuge/street-time reporting |
| `run_overnight.py` | Formal experiment definitions, seeds, batches, confidence intervals and completion records |

Paths without a prefix in the table are relative to `model/bailey/`.

## Whole-person updates

Populations, queues and movements are integers. Fractional route probabilities and passage-service credits are bookkeeping quantities, never fractions of a person. Largest-remainder allocation preserves totals when splitting groups; deterministic routes carry deficits between arrivals, while stochastic routes use multinomial draws. Simultaneous transfers are limited by available people, passage service and receiving storage. Clearance requires exactly zero people inside.

## Reaction times and random streams

Concert emergencies use lognormal reaction times with log-scale parameters 3.93 and 0.44. Normal dismissal uses a Gaussian with mean parameter 35 s and standard deviation 20 s, truncated to [0,120] s. Lecture emergencies use Gamma(shape=3, scale=24.5 s). Deterministic runs use finite population quantiles, while stochastic runs sample individual times. The same population-order seed keeps assignments consistent.

Reaction, passage-service and motion streams are derived separately from each replicate seed. Formal blocks use `SeedSequence([suite_seed, start])`; each block contains at most 25 replicates. Keep the seed scheme and batch size when reproducing the archived experiment. Matching seeds supports paired comparisons; it does not make trajectories identical after routes diverge.

The legacy class name `PoissonRandomness` does not mean the baseline uses Poisson service. Baseline dispersion is 0.5 and service is sampled with a binomial distribution; the Poisson case at dispersion 1 is disabled. See `plugins/randomness.py` for supported alternatives.

## Guidance and wheelchair users

Reference Crowd Managers stand at M1, M2, M3, B1, B3 and FOY. Coverage follows passage distances on the same floor, with a 10 m radius and 0.9 following probability. Covered decisions use a guidance coefficient of 2.5. LEDs use coefficient 4.0 and compliance 0.77. These are modeling assumptions; geometric proximity alone does not establish visibility or audibility in the actual hall.

Three wheelchair users replace three walking occupants in M2. The third position is an assumed arrangement in the same area. Routes exclude stairs until an outdoor refuge is reached. Additional assistance time to street level is reported separately. The broad main exterior steps are not a walking-population bottleneck in the model, while side exterior stairs remain capacity-limited.

## Metrics

Building clearance time ends when the last person leaves the building; porticos are outside. The formal success criterion is strictly `T < 180`. Formal summaries in `run_overnight.metrics` use Wilson intervals for that proportion and the higher-sample definition of percentiles. Incomplete runs count as failures and are retained in restricted means capped at 1,800 s. Some older convenience functions use `<= 180`; they are not the paper's reporting definition.

The absorbing Markov chain is an uncongested analytical reference. The final clearance predictions come from the integer queue simulation.

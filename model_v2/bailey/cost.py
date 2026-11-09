"""Q5: costs and the Pareto front.

A design = (number of crowd managers n, LED strips yes/no, barriers yes/no).

                        n · (cost per event · events per year + yearly training)
lifetime + maintenance per year
lifetime + setup · events

Pareto-optimal = no other design is both cheaper and faster.
The law requires n ≥ 6; LED strips and barriers help managers, they do not replace them."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config


def annual_cost(cfg: Config, n_managers: int, led: bool = False, barriers: bool = False) -> Dict[str, float]:
    'See docs/IMPLEMENTATION.md for implementation details.'
    c = cfg.costs
    life = max(c.lifetime_years, 1)
    out = {
        "managers": n_managers * (c.crowd_manager_per_event * c.events_per_year + c.manager_training_per_year),
        "led": (c.led_install / life + c.led_maintenance_per_year) if led else 0.0,
        "barriers": (c.barrier_purchase / life + c.barrier_setup_per_event * c.events_per_year) if barriers else 0.0,
    }
    out["total"] = sum(out.values())
    return out


def annual_costs(cfg: Config, extra_managers: int = 1) -> Dict[str, float]:
    'legacy summary used by run.py.'
    base = annual_cost(cfg, 0)
    return {"extra_managers": annual_cost(cfg, extra_managers)["managers"] - base["managers"],
            "led": annual_cost(cfg, 0, led=True)["led"],
            "barriers": annual_cost(cfg, 0, barriers=True)["barriers"]}


def cost_per_second_saved(annual: float, T_base: float, T_new: float) -> Optional[float]:
    '$ per year per second saved (None if nothing saved).'
    saved = T_base - T_new
    return annual / saved if saved > 0 else None


def pareto_mask(cost: Sequence[float], time: Sequence[float]) -> np.ndarray:
    'both minimised; True = on the front.'
    cost, time = np.asarray(cost, float), np.asarray(time, float)
    keep = np.ones(len(cost), bool)
    for i in range(len(cost)):
        dominated = (cost <= cost[i]) & (time <= time[i]) & ((cost < cost[i]) | (time < time[i]))
        keep[i] = not dominated.any()
    return keep


def design_table(rows: Sequence[Dict], cfg: Config, obj: str = "T") -> pd.DataFrame:
    'Adds yearly cost columns and the Pareto flag to the per-design results.'
    df = pd.DataFrame(rows)
    costs = [annual_cost(cfg, int(r.managers), bool(r.led), bool(r.barriers)) for r in df.itertuples()]
    for k in ("managers", "led", "barriers", "total"):
        df[f"cost_{k}"] = [c[k] for c in costs]
    df["pareto"] = pareto_mask(df["cost_total"], df[obj])
    base = df[(df["managers"] == df["managers"].min()) & (~df["led"]) & (~df["barriers"])]
    if len(base):
        b = base.iloc[0]
        df["extra_cost"] = df["cost_total"] - b["cost_total"]
        df["seconds_saved"] = b[obj] - df[obj]
        df["usd_per_second_saved"] = [cost_per_second_saved(x, b[obj], t) if x > 0 else None
                                      for x, t in zip(df["extra_cost"], df[obj])]
    return df


# ---------------------------------------------------------------------------
# cost sensitivity
#
#   No new simulation: designs keep their T; only the yearly costs and the Pareto front change.
# ---------------------------------------------------------------------------
_INT_COSTS = ("events_per_year", "lifetime_years")


def _with_costs(cfg: Config, **vals) -> Config:
    from dataclasses import replace
    vals = {k: (int(round(v)) if k in _INT_COSTS else float(v)) for k, v in vals.items()}
    return cfg.with_(costs=replace(cfg.costs, **vals))


def design_label(r) -> str:
    "design label, e.g. '6 mgr' or '7 mgr + LED + barriers'."
    s = f"{int(r['managers'])} mgr"
    if bool(r["led"]):
        s += " + LED"
    if bool(r["barriers"]):
        s += " + barriers"
    return s


def structurally_dominated(df: pd.DataFrame, obj: str = "T_mc") -> pd.Series:
    """A device design that is not faster than the same manager count without devices is dominated for ANY
    positive device cost: same staff cost, extra device cost, no time gain."""
    plain = {int(r["managers"]): r[obj] for _, r in df.iterrows() if not r["led"] and not r["barriers"]}
    return pd.Series([bool(r["led"] or r["barriers"]) and int(r["managers"]) in plain
                      and r[obj] >= plain[int(r["managers"])] for _, r in df.iterrows()], index=df.index)


def cost_sensitivity(rows: Sequence[Dict], cfg: Config, ranges: Optional[Dict] = None, obj: str = "T_mc",
                     n_draws: int = 2000, seed: int = 2026) -> Dict[str, pd.DataFrame]:
    """How the Pareto front moves when the cost parameters vary over config.COST_RANGES.
high end
joint random draws: share of draws on the front"""
    from .config import COST_RANGES
    ranges = ranges or COST_RANGES
    df = pd.DataFrame(rows).reset_index(drop=True)
    labels = [design_label(r) for _, r in df.iterrows()]
    dom = structurally_dominated(df, obj)

    def front(c: Config):
        t = design_table(df.to_dict("records"), c, obj=obj)
        return t["pareto"].to_numpy(), t["cost_total"].to_numpy()

    base_mask, base_cost = front(cfg)
    oat = []
    for key, (lo, hi) in ranges.items():
        for level, val in (("low", lo), ("high", hi)):
            m, _ = front(_with_costs(cfg, **{key: val}))
            oat.append(dict(parameter=key, level=level, value=val,
                            pareto_designs="; ".join(l for l, x in zip(labels, m) if x),
                            same_front_as_default=bool((m == base_mask).all()),
                            device_on_front=bool(any(x and (r["led"] or r["barriers"]) for x, (_, r) in zip(m, df.iterrows())))))
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(df))
    for _ in range(n_draws):
        vals = {k: rng.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
        m, _ = front(_with_costs(cfg, **vals))
        hits += m
    freq = pd.DataFrame(dict(design=labels, managers=df["managers"], led=df["led"], barriers=df["barriers"],
                             T=df[obj], cost_default=base_cost, pareto_default=base_mask,
                             share_on_front=hits / n_draws, dominated_for_any_cost=dom.to_numpy()))
    return dict(oat=pd.DataFrame(oat), freq=freq.sort_values(["share_on_front", "T"], ascending=[False, True]))


def led_cost_per_second(cfg: Config, seconds_saved: float, ranges: Optional[Dict] = None,
                        n_draws: int = 2000, seed: int = 2026) -> pd.DataFrame:
    'Yearly LED cost per second saved (e.g. lectures, where LED saves 1.87 s) over the cost ranges.'
    from .config import COST_RANGES
    ranges = ranges or COST_RANGES
    keys = ("led_install", "led_maintenance_per_year", "lifetime_years")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        vals = {k: rng.uniform(*ranges[k]) for k in keys}
        led = annual_cost(_with_costs(cfg, **vals), 0, led=True)["led"]
        out.append(dict(**vals, led_per_year=led,
                        usd_per_second=led / seconds_saved if seconds_saved > 0 else np.nan))
    return pd.DataFrame(out)

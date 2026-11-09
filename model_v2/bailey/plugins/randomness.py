"""randomness).

The engine asks this plugin for four things:
                              fraction of a seat zone that has started moving by time t
  3. service(expected, t, kind):
                              realised service this step; `expected` = μ_j·Δt (array).
                              draw one Monte Carlo realisation; returns a new instance

Two implementations
------------------------------
    Mean-field release for r ~ U[a, b]; service = μΔt exactly (hard capacity).

      Per-person reaction time R_p (doc 5.3, scheme B); release = empirical CDF.
teammate's windows）
                            doc 5.3 form, same mean as the uniform version
                            doc 5.3 scheme A: hall-wide Poisson stream (disabled).
the doc's Poisson service (disabled)
hard capacity
      Service with mean μ_jΔt and dispersion κ = Var/Mean (default from config, see sources there).

Common random numbers (doc 5.7): reaction times and service draws use two independent
streams derived from the same (seed, m), so every scenario sees the same people in replicate m."""
from __future__ import annotations

from math import erfc, log, sqrt
import hashlib
import json
from typing import Dict, Optional, Tuple

import numpy as np


class RandomnessModel:
    name = "base"
    is_placeholder = False

    def release_cdf(self, seat: str, t: float) -> float:
        raise NotImplementedError

    rng_motion = None
    stochastic_service = False

    def reaction_times(self, seat: str, count: int) -> np.ndarray:
        'Finite-person quantiles for a custom CDF.'
        out = []
        for u in (np.arange(count) + 0.5) / max(count, 1):
            lo, hi = 0.0, 1.0
            while self.release_cdf(seat, hi) < u and hi < 1e9:
                hi *= 2
            if self.release_cdf(seat, hi) < u:
                raise ValueError("Reaction CDF does not reach all person quantiles")
            for _ in range(50):
                mid = (lo + hi) / 2
                if self.release_cdf(seat, mid) < u:
                    lo = mid
                else:
                    hi = mid
            out.append(hi)
        return np.asarray(out)

    def capacity_multiplier(self, t: float) -> float:
        return 1.0

    def service(self, expected: np.ndarray, t: float, kind: str) -> np.ndarray:
        'default: hard capacity, exactly μΔt.'
        return expected

    def sample(self, rng: np.random.Generator) -> "RandomnessModel":
        raise NotImplementedError


class MeanFieldPlaceholder(RandomnessModel):
    'Deterministic: expected (mean-field) release for r ~ Uniform[a, b].'
    name = "mean-field uniform release"
    is_placeholder = True

    def __init__(self, window: Tuple[float, float]):
        self.a, self.b = window

    def release_cdf(self, seat, t):
        if self.b <= self.a:
            return 1.0 if t >= self.a else 0.0
        return float(np.clip((t - self.a) / (self.b - self.a), 0.0, 1.0))

    def reaction_times(self, seat, count):
        return self.a + (self.b - self.a) * (np.arange(count) + 0.5) / max(count, 1)

    def sample(self, rng):
        return self


class LognormalRelease(RandomnessModel):
    """Deterministic population release from a lognormal reaction-time CDF."""

    name = "mean-field lognormal release"

    def __init__(self, log_params: Tuple[float, float]):
        self.mu, self.sigma = log_params

    def release_cdf(self, seat, t):
        return 0.0 if t <= 0 else 0.5 * erfc((self.mu - log(t)) / (self.sigma * sqrt(2.0)))

    def reaction_times(self, seat, count):
        from scipy.special import ndtri
        return np.exp(self.mu + self.sigma * ndtri((np.arange(count) + 0.5) / max(count, 1)))


def reaction_distribution(spec):
    """One distribution definition for deterministic, stochastic and plotting paths."""
    from scipy.stats import gamma, lognorm, truncnorm
    family = spec["family"]
    if family == "lognormal":
        return lognorm(s=spec["sigma"], scale=np.exp(spec["mu"]))
    if family == "gamma":
        return gamma(a=spec["shape"], scale=spec["scale"])
    if family == "truncnorm":
        mu, sd = spec["mu"], spec["sigma"]
        return truncnorm((spec["low"]-mu)/sd, (spec["high"]-mu)/sd, loc=mu, scale=sd)
    raise ValueError(f"Unknown reaction family: {family}")


class DistributionRelease(RandomnessModel):
    """Finite midpoint quantiles of the configured pre-movement distribution."""
    def __init__(self, spec):
        self.spec = dict(spec)
        self.rv = reaction_distribution(self.spec)
        self.name = "deterministic quantiles: " + self.spec["family"]

    def release_cdf(self, seat, t):
        return float(self.rv.cdf(t))

    def reaction_times(self, seat, count):
        return self.rv.ppf((np.arange(count)+0.5)/max(count, 1))

    def sample(self, rng):
        return self


def mean_field_release(prep_key: str) -> RandomnessModel:
    from ..config import REACTION_SPECS
    return DistributionRelease(REACTION_SPECS[prep_key])


class PoissonRandomness(RandomnessModel):
    """the stochastic version of Section 5.

reaction-time window in s
    service     : "poisson" (legacy selector for variable service) | "hard"
    dt          : retained for caller compatibility (used by the disabled Poisson stream)"""
    is_placeholder = False

    def __init__(self, window: Tuple[float, float], seat_counts: Dict[str, float],
                 dist: str = "uniform", service: str = "poisson", dt: float = 0.5,
                 dispersion: Optional[float] = None,
                 _draws: Optional[Dict[str, np.ndarray]] = None,
                 _rng_service: Optional[np.random.Generator] = None,
                 log_params: Optional[Tuple[float, float]] = None,
                 _rng_motion: Optional[np.random.Generator] = None,
                 reaction_spec: Optional[Dict] = None):
        if dist == "poisson-stream":
            raise ValueError("Poisson arrival stream is disabled")
        assert dist in ("uniform", "exponential", "lognormal", "truncnorm", "gamma", "mean-field"), dist
        assert service in ("poisson", "hard"), service
        self.a, self.b = window
        self.seat_counts = {k: int(round(v)) for k, v in seat_counts.items()}
        self.dist, self.service_kind, self.dt = dist, service, dt
        if dispersion is None:
            from ..config import SERVICE_DISPERSION
            dispersion = SERVICE_DISPERSION
        self.kappa = 0.0 if service == "hard" else float(dispersion)
        if abs(self.kappa - 1.0) < 1e-9:
            raise ValueError("Poisson service (κ = 1) is disabled")
        self.person_draws = _draws
        self.draws = None if _draws is None else {z: np.sort(r) for z, r in _draws.items()}
        self.rng_motion = _rng_motion
        self.stochastic_service = self.kappa > 0
        # sorted reaction times per zone
        self.rng_s = _rng_service
        self.name = f"stochastic (reaction={dist}, service κ={self.kappa:g})"
        if dist == "lognormal" and log_params is None and reaction_spec is None:
            raise ValueError("Lognormal reaction times require log-scale parameters")
        self.log_params = log_params
        self.reaction_spec = dict(reaction_spec) if reaction_spec is not None else None
        self._mf = (DistributionRelease(self.reaction_spec) if self.reaction_spec is not None else
                    LognormalRelease(log_params) if log_params is not None else MeanFieldPlaceholder(window))

    # sampling ----------
    def sample(self, rng: np.random.Generator) -> "PoissonRandomness":
        'rng is only used to spawn two child streams.'
        ss = np.random.SeedSequence(int(rng.integers(0, 2**63 - 1)))
        r_react, r_serv = (np.random.default_rng(s) for s in ss.spawn(2))
        return self.sample_streams(r_react, r_serv)

    def sample_streams(self, r_react: np.random.Generator,
                       r_serv: np.random.Generator) -> "PoissonRandomness":
        # Separate reproducible motion stream; do not consume reaction/service draws.
        payload = json.dumps(r_serv.bit_generator.state, sort_keys=True).encode()
        motion_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")
        r_motion = np.random.default_rng(motion_seed)
        draws = None
        if self.dist != "mean-field":
            draws = self._draw_reactions(r_react)
        return PoissonRandomness((self.a, self.b), self.seat_counts, self.dist, self.service_kind,
                                 self.dt, dispersion=self.kappa, _draws=draws, _rng_service=r_serv,
                                 log_params=self.log_params, _rng_motion=r_motion, reaction_spec=self.reaction_spec)

    def _draw_reactions(self, rng) -> Dict[str, np.ndarray]:
        zones = sorted(self.seat_counts)          # fixed order for CRN
        out = {}
        if self.reaction_spec is not None:
            rv = reaction_distribution(self.reaction_spec)
            # Inverse CDF shares the same per-person uniforms even across event families.
            for z in zones:
                u = np.clip(rng.random(self.seat_counts[z]), np.finfo(float).eps, 1-np.finfo(float).eps)
                out[z] = rv.ppf(u)
            return out
        if self.dist == "uniform":
            for z in zones:
                out[z] = rng.uniform(self.a, self.b, self.seat_counts[z])
        elif self.dist == "exponential":
            mean = np.exp(self.log_params[0] + self.log_params[1]**2/2) if self.log_params else 0.5 * (self.a + self.b)
            for z in zones:
                out[z] = rng.exponential(mean, self.seat_counts[z])
        elif self.dist == "lognormal":
            for z in zones:
                out[z] = rng.lognormal(*self.log_params, size=self.seat_counts[z])
        # elif self.dist == "poisson-stream":
            # scheme A
            # n = np.array([self.seat_counts[z] for z in zones])
            # lam0 = n.sum() / max(self.b - self.a, 1e-9)
            # left = n.copy()
            # times = {z: [] for z in zones}
            # t = self.a
            # while left.sum() > 0:
                # t += self.dt
                # k = min(int(rng.poisson(lam0 * self.dt)), int(left.sum()))
                # if k:
                    # take = rng.multivariate_hypergeometric(left, k)
                    # for zi, z in enumerate(zones):
                        # times[z] += [t] * int(take[zi])
                    # left -= take
            # out = {z: np.array(times[z], float) for z in zones}
        return out

    # called by the engine ----------
    def reaction_times(self, seat, count):
        if self.person_draws is None:
            return self._mf.reaction_times(seat, count)
        r = self.person_draws.get(seat)
        if r is None or len(r) != count:
            raise ValueError(f"Reaction samples for {seat} must match its integer population")
        return r.copy()

    def release_cdf(self, seat, t):
        if self.draws is None:                    # mean-field
            return self._mf.release_cdf(seat, t)
        r = self.draws.get(seat)
        if r is None or r.size == 0:
            return self._mf.release_cdf(seat, t)
        return float(np.searchsorted(r, t, side="right")) / r.size

    def service(self, expected, t, kind):
        if self.kappa <= 0 or self.rng_s is None:
            return expected
        ok = np.isfinite(expected) & (expected > 0)
        m = np.where(ok, expected, 0.0)
        k = self.kappa
        rng = self.rng_s
        # Poisson
            # s = rng.poisson(m).astype(float)
        if k < 1.0:                                 # binomial, Var ≈ κ·mean
            n = np.maximum(np.ceil(m / (1.0 - k)), 1.0)
            s = rng.binomial(n.astype(np.int64), np.clip(m / n, 0.0, 1.0)).astype(float)
        else:                                         # negative binomial
            r = np.maximum(m / (k - 1.0), 1e-9)
            s = rng.negative_binomial(r, 1.0 / k).astype(float)
        return np.where(ok, s, expected)


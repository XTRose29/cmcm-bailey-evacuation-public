"""Conservative allocation of indivisible people.

Rates, probabilities, and service credit may be real; population counts may not."""
from dataclasses import replace

import numpy as np


ENGINE_VERSION = "integer-v3-literature-reactions-local-guidance"


def apportion(total, weights):
    """Largest remainders, with stable index order for ties.

    When total <= sum(integer demands), no demand can be exceeded."""
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Allocation weights must be finite and nonnegative")
    if total < 0 or int(total) != total:
        raise ValueError("Population total must be a nonnegative integer")
    total = int(total)
    if total == 0:
        return np.zeros(w.shape, dtype=np.int64)
    if w.sum() <= 0:
        raise ValueError("Cannot allocate people to zero demand")
    quotas = total * w / w.sum()
    out = np.floor(quotas).astype(np.int64)
    order = np.argsort(-(quotas - out).ravel(), kind="stable")
    out.flat[order[:total - int(out.sum())]] += 1
    return out


def integer_classes(classes):
    """Round class shares jointly in each seat zone.

    An integer-valued class (e.g. one wheelchair user) is preserved exactly.
    Fractional zone totals are rejected, never silently rounded."""
    counts = [{} for _ in classes]
    for z in sorted({z for c in classes for z in c.seat_counts}):
        weights = np.array([c.seat_counts.get(z, 0) for c in classes], float)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"Invalid population in {z}")
        n = weights.sum()
        if abs(n - round(n)) > 1e-7:
            raise ValueError(f"Seat zone {z} has non-integer total {n}")
        # Preserve exact integer components; apportion only the residual shares.
        fixed = weights == np.floor(weights)
        allocated = np.zeros(len(classes), dtype=np.int64)
        allocated[fixed] = weights[fixed].astype(np.int64)
        if (~fixed).any():
            allocated[~fixed] = apportion(round(n) - allocated.sum(), weights[~fixed])
        for j, x in enumerate(allocated):
            if x:
                counts[j][z] = int(x)
    return [replace(c, seat_counts=cnt) for c, cnt in zip(classes, counts)]


def limit_counts(demand, budget):
    'Integer proportional service within a budget.'
    total = int(demand.sum())
    if not np.isfinite(budget) or total <= budget:
        return demand.copy()
    return apportion(max(0, int(np.floor(budget + 1e-10))), demand)


class RouteAllocator:
    """Multinomial routes, or deterministic cumulative deficits.

    Deficits carry across arrivals, so batches of one person do not always pick
    the same door. They are allocation credits, never fractional people."""
    def __init__(self, shape, rng=None):
        self.debt = np.zeros(shape, dtype=float)
        self.rng = rng

    def split(self, variant, counts, probabilities):
        out = np.zeros(probabilities.shape, dtype=np.int64)
        for i in np.flatnonzero(counts):
            p = probabilities[i]
            if p.sum() <= 0:
                raise ValueError("Cannot route people from a dead end")
            if self.rng is not None:
                out[i] = self.rng.multinomial(int(counts[i]), p / p.sum())
            else:
                d = self.debt[variant, i]
                d[p == 0] = 0.0
                for _ in range(int(counts[i])):
                    d += p
                    k = int(np.argmax(np.where(p > 0, d, -np.inf)))
                    out[i, k] += 1
                    d[k] -= 1.0
        return out


class ServiceCredit:
    """Fractional service work, not fractional people.

    Deterministic servers retain at most one unused person-slot before adding
    the next interval's work. Idle periods cannot bank an unlimited burst.
    Stochastic service slots expire each step; only fractional work from a
    scaled shared gate can carry over. Blocked resources must be reset."""
    def __init__(self, shape):
        self.balance = np.zeros(shape, float)

    def offer(self, amounts, stochastic=None, disabled=None):
        amounts = np.asarray(amounts, float)
        if (amounts < 0).any() or np.isnan(amounts).any():
            raise ValueError("Service amounts must be nonnegative")
        self.balance = np.minimum(self.balance, 1.0)
        if stochastic is not None:
            self.balance[stochastic] %= 1.0
        finite = np.isfinite(amounts)
        self.balance[finite] += amounts[finite]
        self.balance[~finite] = 0.0
        if disabled is not None:
            self.balance[disabled] = 0.0
        offered = np.where(finite, np.floor(self.balance + 1e-10), np.inf)
        if disabled is not None:
            offered[disabled] = 0
        return offered

    def consume(self, counts):
        # Unlimited resources have zero balance and are ignored here.
        self.balance = np.maximum(0.0, self.balance - counts)

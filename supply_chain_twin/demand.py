"""Demand-generating processes.

Phase 1 used a stationary Poisson+noise process — fine for testing inventory
logic, but unrealistic: it has no pattern, so nothing is gained by forecasting
it (the sample mean already is the best forecast). Phase 2 introduces a
seasonal process with a weekly pattern and a trend, which is what makes
forecasting worth doing at all, and lets an (s, S) policy tuned on
historical averages be measurably beaten by a policy that tracks the pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class DemandProcess(Protocol):
    """A process that generates one day of demand at a time, given the day
    index. Stateless with respect to the simulation — all state (trend
    progression, etc.) is a pure function of `day`, so the same process
    instance can generate historical data and then continue seamlessly into
    the simulation's future."""

    def sample(self, day: int, rng: np.random.Generator) -> float: ...


@dataclass
class StationaryPoissonProcess:
    """Phase 1's demand model: Poisson baseline plus Gaussian noise, no
    day-dependence. Kept for backward compatibility and as a forecasting
    baseline case where no model should be able to beat the sample mean."""

    mean: float
    std: float = 0.0

    def sample(self, day: int, rng: np.random.Generator) -> float:
        base = rng.poisson(self.mean)
        noise = rng.normal(0.0, self.std) if self.std > 0 else 0.0
        return max(0.0, float(base + noise))


@dataclass
class SeasonalDemandProcess:
    """Weekly-seasonal demand with a linear trend, Poisson-ish overdispersion,
    and Gaussian noise on top. `weekday_multipliers` has 7 entries (Mon..Sun)
    describing relative demand strength per day of week — the mean multiplier
    should be close to 1.0 so `base_level` stays interpretable as the average.
    """

    base_level: float
    weekday_multipliers: Sequence[float] = field(
        default_factory=lambda: (1.1, 1.0, 0.95, 1.0, 1.15, 1.3, 0.9)
    )
    trend_per_day: float = 0.0
    noise_std: float = 3.0

    def __post_init__(self) -> None:
        if len(self.weekday_multipliers) != 7:
            raise ValueError("weekday_multipliers must have exactly 7 entries")

    def expected_value(self, day: int) -> float:
        """The noise-free expected demand for a given day — used by
        forecasters as ground truth to evaluate against, and by the engine
        for diagnostics."""
        level = self.base_level + self.trend_per_day * day
        multiplier = self.weekday_multipliers[day % 7]
        return max(0.0, level * multiplier)

    def sample(self, day: int, rng: np.random.Generator) -> float:
        expected = self.expected_value(day)
        base = rng.poisson(max(expected, 0.01))
        noise = rng.normal(0.0, self.noise_std)
        return max(0.0, float(base + noise))


def generate_history(
    process: DemandProcess,
    days: int,
    rng: np.random.Generator,
    start_day: int = 0,
) -> np.ndarray:
    """Generate `days` of demand starting at `start_day`. Used to produce the
    pre-twin historical data a forecaster trains on, and can continue from
    any offset so history and simulation share one continuous timeline."""
    return np.array(
        [process.sample(start_day + d, rng) for d in range(days)],
        dtype=float,
    )

"""Tests for demand-generating processes."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_twin.demand import (
    SeasonalDemandProcess,
    StationaryPoissonProcess,
    generate_history,
)


class TestStationaryPoissonProcess:
    def test_non_negative_and_reproducible(self):
        process = StationaryPoissonProcess(mean=20.0, std=5.0)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        samples_a = [process.sample(d, rng_a) for d in range(50)]
        samples_b = [process.sample(d, rng_b) for d in range(50)]
        assert samples_a == samples_b
        assert all(s >= 0 for s in samples_a)

    def test_day_independent(self):
        # Same rng state -> same draw regardless of day index, since this
        # process ignores `day` entirely.
        process = StationaryPoissonProcess(mean=10.0, std=0.0)
        rng_a = np.random.default_rng(5)
        rng_b = np.random.default_rng(5)
        assert process.sample(0, rng_a) == process.sample(999, rng_b)


class TestSeasonalDemandProcess:
    def test_rejects_wrong_length_multipliers(self):
        with pytest.raises(ValueError):
            SeasonalDemandProcess(base_level=10.0, weekday_multipliers=(1.0, 1.0))

    def test_weekday_pattern_applied(self):
        process = SeasonalDemandProcess(
            base_level=100.0,
            weekday_multipliers=(1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5),
            trend_per_day=0.0,
        )
        assert process.expected_value(5) == pytest.approx(200.0)
        assert process.expected_value(6) == pytest.approx(50.0)
        assert process.expected_value(5 + 7) == pytest.approx(200.0)  # repeats weekly

    def test_trend_increases_expected_value(self):
        process = SeasonalDemandProcess(
            base_level=100.0,
            weekday_multipliers=(1, 1, 1, 1, 1, 1, 1),
            trend_per_day=1.0,
        )
        assert process.expected_value(50) > process.expected_value(0)

    def test_samples_are_non_negative(self):
        process = SeasonalDemandProcess(base_level=5.0, trend_per_day=-1.0)
        rng = np.random.default_rng(0)
        samples = [process.sample(d, rng) for d in range(200)]
        assert all(s >= 0 for s in samples)


class TestGenerateHistory:
    def test_length_and_offset(self):
        process = StationaryPoissonProcess(mean=15.0, std=2.0)
        rng = np.random.default_rng(0)
        history = generate_history(process, days=90, rng=rng, start_day=30)
        assert len(history) == 90
        assert history.dtype == float

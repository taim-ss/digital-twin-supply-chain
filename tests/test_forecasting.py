"""Tests for forecasters and the walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_twin.demand import SeasonalDemandProcess, generate_history
from supply_chain_twin.forecasting import (
    ExponentialSmoothingForecaster,
    GradientBoostingForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    backtest,
)


@pytest.fixture(scope="module")
def seasonal_history() -> np.ndarray:
    process = SeasonalDemandProcess(base_level=50.0, trend_per_day=0.05, noise_std=2.0)
    rng = np.random.default_rng(42)
    return generate_history(process, days=150, rng=rng)


class TestNaiveForecaster:
    def test_repeats_last_value(self):
        f = NaiveForecaster()
        f.fit(np.array([1.0, 2.0, 3.0, 42.0]))
        assert list(f.predict(4)) == [42.0, 42.0, 42.0, 42.0]

    def test_rejects_empty_history(self):
        with pytest.raises(ValueError):
            NaiveForecaster().fit(np.array([]))


class TestSeasonalNaiveForecaster:
    def test_tiles_last_season(self):
        f = SeasonalNaiveForecaster(season_length=7)
        history = np.arange(21, dtype=float)  # last 7: [14..20]
        f.fit(history)
        pred = f.predict(10)
        assert list(pred[:7]) == list(range(14, 21))
        assert list(pred[7:10]) == list(range(14, 17))

    def test_rejects_short_history(self):
        with pytest.raises(ValueError):
            SeasonalNaiveForecaster(season_length=7).fit(np.array([1.0, 2.0]))


class TestExponentialSmoothingForecaster:
    def test_rejects_short_history(self):
        with pytest.raises(ValueError):
            ExponentialSmoothingForecaster(seasonal_periods=7).fit(np.arange(10, dtype=float))

    def test_fits_and_predicts_non_negative(self, seasonal_history):
        f = ExponentialSmoothingForecaster(seasonal_periods=7)
        f.fit(seasonal_history)
        pred = f.predict(14)
        assert len(pred) == 14
        assert (pred >= 0).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ExponentialSmoothingForecaster().predict(7)


class TestGradientBoostingForecaster:
    def test_rejects_short_history(self):
        with pytest.raises(ValueError):
            GradientBoostingForecaster().fit(np.arange(10, dtype=float))

    def test_fits_and_predicts_non_negative(self, seasonal_history):
        f = GradientBoostingForecaster(n_estimators=20)  # small, just for speed
        f.fit(seasonal_history)
        pred = f.predict(7)
        assert len(pred) == 7
        assert (pred >= 0).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            GradientBoostingForecaster().predict(7)


class TestBacktest:
    def test_naive_backtest_reports_sane_metrics(self, seasonal_history):
        result = backtest(
            make_forecaster=NaiveForecaster,
            name="naive",
            history=seasonal_history,
            horizon=7,
            n_folds=4,
            min_train=60,
        )
        assert result.name == "naive"
        assert result.mae >= 0
        assert result.rmse >= result.mae  # RMSE never below MAE
        assert result.residual_std >= 0

    def test_seasonal_naive_beats_naive_on_seasonal_data(self, seasonal_history):
        naive = backtest(NaiveForecaster, "naive", seasonal_history, horizon=7, n_folds=4, min_train=60)
        seasonal = backtest(
            lambda: SeasonalNaiveForecaster(season_length=7),
            "seasonal_naive",
            seasonal_history,
            horizon=7,
            n_folds=4,
            min_train=60,
        )
        # Seasonal naive should exploit the weekly pattern naive can't see.
        assert seasonal.mae < naive.mae

    def test_raises_when_history_too_short(self):
        with pytest.raises(ValueError):
            backtest(NaiveForecaster, "naive", np.arange(10, dtype=float), horizon=7, n_folds=4, min_train=60)

"""Demand forecasters and walk-forward backtesting.

Four forecasters, deliberately spanning "no model" to "a real ML model," so
the gradient boosting model has to earn its complexity against honest
baselines rather than being compared to nothing:

- NaiveForecaster: repeats the last observed value. The bar any model has
  to clear.
- SeasonalNaiveForecaster: repeats the same weekday from last week. Already
  strong on a weekly-seasonal series, and cheap to compute.
- ExponentialSmoothingForecaster: Holt-Winters, a classical statistical
  method built for exactly this trend + seasonality shape.
- GradientBoostingForecaster: lag/rolling/day-of-week features feeding a
  scikit-learn gradient boosting regressor, forecasting recursively
  (each prediction becomes a lag feature for the next step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Forecaster(Protocol):
    """Every forecaster fits on a demand history and produces a point
    forecast for the next `horizon` days. `start_day` tells the forecaster
    which absolute day index `history[0]` corresponds to, so day-of-week
    features stay aligned between training and prediction."""

    name: str

    def fit(self, history: np.ndarray, start_day: int = 0) -> None: ...

    def predict(self, horizon: int) -> np.ndarray: ...


@dataclass
class NaiveForecaster:
    name: str = "naive"
    _last_value: float = field(default=0.0, init=False, repr=False)

    def fit(self, history: np.ndarray, start_day: int = 0) -> None:
        if len(history) == 0:
            raise ValueError("history must be non-empty")
        self._last_value = float(history[-1])

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last_value, dtype=float)


@dataclass
class SeasonalNaiveForecaster:
    season_length: int = 7
    name: str = "seasonal_naive"
    _last_season: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False, repr=False)

    def fit(self, history: np.ndarray, start_day: int = 0) -> None:
        if len(history) < self.season_length:
            raise ValueError(f"need at least {self.season_length} points of history")
        self._last_season = history[-self.season_length:].copy()

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / self.season_length))
        return np.tile(self._last_season, reps)[:horizon]


@dataclass
class ExponentialSmoothingForecaster:
    seasonal_periods: int = 7
    name: str = "holt_winters"
    _fitted: object = field(default=None, init=False, repr=False)

    def fit(self, history: np.ndarray, start_day: int = 0) -> None:
        min_needed = 2 * self.seasonal_periods
        if len(history) < min_needed:
            raise ValueError(f"Holt-Winters needs at least {min_needed} points, got {len(history)}")
        model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._fitted = model.fit()

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        forecast = self._fitted.forecast(horizon)
        return np.clip(np.asarray(forecast, dtype=float), 0.0, None)


@dataclass
class GradientBoostingForecaster:
    """Recursive multi-step forecasting: predict one day, append it to the
    series as if observed, then predict the next. Simpler than a direct
    multi-horizon model and adequate for the short (lead-time-scale)
    horizons this twin forecasts over."""

    lags: Sequence[int] = (1, 2, 7, 14)
    rolling_windows: Sequence[int] = (7, 14)
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 0
    name: str = "gradient_boosting"
    _model: object = field(default=None, init=False, repr=False)
    _history: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False, repr=False)
    _start_day: int = field(default=0, init=False, repr=False)

    @property
    def _min_history(self) -> int:
        return max(max(self.lags), max(self.rolling_windows)) + 1

    def _features(self, series: np.ndarray, day_index: int) -> np.ndarray:
        feats = [series[-lag] for lag in self.lags]
        feats += [float(np.mean(series[-window:])) for window in self.rolling_windows]
        feats.append(day_index % 7)
        return np.array(feats, dtype=float)

    def fit(self, history: np.ndarray, start_day: int = 0) -> None:
        min_hist = self._min_history
        min_needed = min_hist + 20  # leave enough rows to actually train on
        if len(history) < min_needed:
            raise ValueError(f"gradient boosting needs at least {min_needed} points, got {len(history)}")

        self._history = history.copy()
        self._start_day = start_day

        X, y = [], []
        for i in range(min_hist, len(history)):
            X.append(self._features(history[:i], start_day + i))
            y.append(history[i])

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(np.array(X), np.array(y))

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        series = self._history.copy()
        day_index = self._start_day + len(self._history)
        preds = []
        for _ in range(horizon):
            feats = self._features(series, day_index).reshape(1, -1)
            pred = max(0.0, float(self._model.predict(feats)[0]))
            preds.append(pred)
            series = np.append(series, pred)
            day_index += 1
        return np.array(preds, dtype=float)


@dataclass
class BacktestResult:
    """Error metrics from walk-forward validation, plus the residual std
    used downstream to size safety stock (bigger forecast error -> more
    safety stock needed for the same target service level)."""

    name: str
    mae: float
    rmse: float
    mape: float
    residual_std: float

    def row(self) -> str:
        return f"{self.name:<18} {self.mae:>8.2f} {self.rmse:>8.2f} {self.mape:>9.1f}% {self.residual_std:>10.2f}"

    @staticmethod
    def header() -> str:
        return f"{'model':<18} {'MAE':>8} {'RMSE':>8} {'MAPE':>10} {'resid std':>10}"


def backtest(
    make_forecaster: Callable[[], Forecaster],
    name: str,
    history: np.ndarray,
    horizon: int,
    n_folds: int = 6,
    min_train: int = 60,
) -> BacktestResult:
    """Walk-forward validation: at `n_folds` evenly-spaced points in
    `history`, fit fresh on everything up to that point and forecast the
    next `horizon` days. Every fold's train slice starts at absolute day 0,
    so day-of-week features stay aligned with the un-truncated timeline."""

    n = len(history)
    usable = n - min_train - horizon
    if usable <= 0:
        raise ValueError("not enough history for the requested min_train/horizon")
    fold_starts = np.linspace(min_train, min_train + usable, n_folds, dtype=int)

    actuals: list[float] = []
    forecasts: list[float] = []

    for start in fold_starts:
        train = history[:start]
        actual = history[start:start + horizon]
        forecaster = make_forecaster()
        forecaster.fit(train, start_day=0)
        forecasts.extend(forecaster.predict(horizon).tolist())
        actuals.extend(actual.tolist())

    actual_arr = np.array(actuals)
    forecast_arr = np.array(forecasts)
    errors = actual_arr - forecast_arr

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nonzero = actual_arr > 1e-6
    mape = float(np.mean(np.abs(errors[nonzero] / actual_arr[nonzero])) * 100) if nonzero.any() else float("nan")
    residual_std = float(np.std(errors))

    return BacktestResult(name=name, mae=mae, rmse=rmse, mape=mape, residual_std=residual_std)

"""Inventory policies that decide when and how much to reorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .forecasting import Forecaster


class ReorderPolicy(Protocol):
    """Interface every replenishment policy implements — lets Phase 2/3 swap
    in forecast-driven or optimization-driven policies without touching the engine."""

    fixed_order_cost: float

    def order_quantity(self, inventory_position: float) -> float: ...

    def refresh(self, demand_history: Sequence[float], day: int) -> None:
        """Called once per simulated day before the reorder decision, so a
        policy can periodically re-fit itself against the demand seen so
        far. Static policies leave this a no-op."""
        ...


@dataclass
class ReorderUpToPolicy:
    """(s, S) policy: order up to S whenever inventory position drops to s or below."""

    reorder_point: float
    order_up_to_level: float
    fixed_order_cost: float = 50.0

    def __post_init__(self) -> None:
        if self.order_up_to_level <= self.reorder_point:
            raise ValueError("order_up_to_level must exceed reorder_point")

    def order_quantity(self, inventory_position: float) -> float:
        """Return 0 above the reorder point, otherwise the gap up to S."""
        if inventory_position > self.reorder_point:
            return 0.0
        return self.order_up_to_level - inventory_position

    def refresh(self, demand_history: Sequence[float], day: int) -> None:
        pass  # static policy — reorder_point/order_up_to_level never change


@dataclass
class ForecastDrivenPolicy:
    """(s, S) policy whose s and S are recomputed periodically from a demand
    forecast, instead of fixed by hand:

    - reorder point  = forecasted demand over the lead time + safety stock
    - safety stock    = z * (forecast residual std) * sqrt(lead time)
    - order-up-to     = reorder point + `cycle_days` of forecasted average
                        demand (the cycle stock between reorder events)

    `residual_std` should come from backtesting the forecaster (see
    `forecasting.backtest`) — it is the forecaster's own historical error,
    which is what safety stock is meant to buffer against.
    """

    forecaster: Forecaster
    lead_time_days: int
    residual_std: float
    service_z: float = 1.65  # ~95% service level under a normal error assumption
    cycle_days: int = 7
    review_period_days: int = 7
    fixed_order_cost: float = 50.0

    reorder_point: float = field(default=0.0, init=False)
    order_up_to_level: float = field(default=1.0, init=False)
    _last_refresh_day: int = field(default=-1, init=False, repr=False)

    def refresh(self, demand_history: Sequence[float], day: int) -> None:
        never_refreshed = self._last_refresh_day == -1
        due = never_refreshed or (day - self._last_refresh_day) >= self.review_period_days
        if not due or len(demand_history) < 1:
            return

        self.forecaster.fit(np.asarray(demand_history, dtype=float), start_day=0)
        forecast = self.forecaster.predict(self.lead_time_days)

        forecast_lead_time_demand = float(np.sum(forecast))
        safety_stock = self.service_z * self.residual_std * (self.lead_time_days ** 0.5)
        mean_forecast = float(np.mean(forecast))

        self.reorder_point = forecast_lead_time_demand + safety_stock
        self.order_up_to_level = self.reorder_point + self.cycle_days * mean_forecast
        self._last_refresh_day = day

    def order_quantity(self, inventory_position: float) -> float:
        if inventory_position > self.reorder_point:
            return 0.0
        return max(0.0, self.order_up_to_level - inventory_position)

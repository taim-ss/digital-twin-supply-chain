"""Tests for ForecastDrivenPolicy, including its engine integration."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_twin.demand import SeasonalDemandProcess, generate_history
from supply_chain_twin.engine import SimulationEngine
from supply_chain_twin.entities import Node, NodeType
from supply_chain_twin.forecasting import NaiveForecaster, SeasonalNaiveForecaster
from supply_chain_twin.policies import ForecastDrivenPolicy, ReorderUpToPolicy


class CountingForecaster:
    """A stub that records how many times fit() was called, to test the
    policy's review-period gating in isolation from a real model."""

    name = "counting"

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, history: np.ndarray, start_day: int = 0) -> None:
        self.fit_calls += 1

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, 10.0)


class TestForecastDrivenPolicyRefreshGating:
    def test_refits_on_first_call_regardless_of_absolute_day(self):
        forecaster = CountingForecaster()
        policy = ForecastDrivenPolicy(
            forecaster=forecaster, lead_time_days=3, residual_std=2.0, review_period_days=7,
        )
        policy.refresh([10.0] * 100, day=180)
        assert forecaster.fit_calls == 1
        assert policy.reorder_point > 0

    def test_does_not_refit_before_review_period_elapses(self):
        forecaster = CountingForecaster()
        policy = ForecastDrivenPolicy(
            forecaster=forecaster, lead_time_days=3, residual_std=2.0, review_period_days=7,
        )
        policy.refresh([10.0] * 10, day=0)
        for day in range(1, 7):
            policy.refresh([10.0] * (10 + day), day=day)
        assert forecaster.fit_calls == 1  # gated until day 7

    def test_refits_once_review_period_elapses(self):
        forecaster = CountingForecaster()
        policy = ForecastDrivenPolicy(
            forecaster=forecaster, lead_time_days=3, residual_std=2.0, review_period_days=7,
        )
        policy.refresh([10.0] * 10, day=0)
        policy.refresh([10.0] * 17, day=7)
        assert forecaster.fit_calls == 2

    def test_reorder_point_reflects_forecast_and_safety_stock(self):
        forecaster = CountingForecaster()  # always forecasts 10.0/day
        policy = ForecastDrivenPolicy(
            forecaster=forecaster,
            lead_time_days=4,
            residual_std=2.0,
            service_z=1.65,
            review_period_days=7,
        )
        policy.refresh([1.0], day=0)
        expected_lead_time_demand = 10.0 * 4
        expected_safety_stock = 1.65 * 2.0 * (4 ** 0.5)
        assert policy.reorder_point == pytest.approx(expected_lead_time_demand + expected_safety_stock)


class TestForecastDrivenPolicyEndToEnd:
    def test_runs_cleanly_with_seasonal_demand_and_seasonal_naive_forecaster(self):
        process = SeasonalDemandProcess(base_level=30.0, trend_per_day=0.0, noise_std=2.0)
        rng = np.random.default_rng(1)
        history = generate_history(process, days=60, rng=rng)

        supplier = Node(name="S", node_type=NodeType.SUPPLIER)
        warehouse = Node(name="W", node_type=NodeType.WAREHOUSE, lead_time_days=3)
        warehouse.inventory.on_hand = 100.0

        policy = ForecastDrivenPolicy(
            forecaster=SeasonalNaiveForecaster(season_length=7),
            lead_time_days=3,
            residual_std=5.0,
            review_period_days=7,
        )

        engine = SimulationEngine(
            warehouse=warehouse,
            supplier=supplier,
            policy=policy,
            horizon=90,
            demand_process=process,
            initial_demand_history=history,
            seed=2,
        )
        kpis = engine.run()

        assert 0.0 <= kpis.service_level <= 1.0
        assert 0.0 <= kpis.fill_rate <= 1.0
        assert policy.reorder_point > 0
        assert policy.order_up_to_level > policy.reorder_point

    def test_forecast_driven_and_static_policy_both_run_on_same_seasonal_process(self):
        # Not a claim that one beats the other on every seed — just proves
        # both policy types are drop-in compatible with the same engine
        # and demand process, which is the actual Phase 2 architecture goal.
        process = SeasonalDemandProcess(base_level=25.0, noise_std=2.0)
        rng = np.random.default_rng(3)
        history = generate_history(process, days=60, rng=rng)

        def build(policy) -> SimulationEngine:
            supplier = Node(name="S", node_type=NodeType.SUPPLIER)
            warehouse = Node(name="W", node_type=NodeType.WAREHOUSE, lead_time_days=3)
            warehouse.inventory.on_hand = 80.0
            return SimulationEngine(
                warehouse=warehouse,
                supplier=supplier,
                policy=policy,
                horizon=60,
                demand_process=process,
                initial_demand_history=list(history),
                seed=9,
            )

        static_kpis = build(ReorderUpToPolicy(reorder_point=60, order_up_to_level=150)).run()
        forecast_kpis = build(
            ForecastDrivenPolicy(
                forecaster=NaiveForecaster(), lead_time_days=3, residual_std=5.0, review_period_days=7,
            )
        ).run()

        for kpis in (static_kpis, forecast_kpis):
            assert 0.0 <= kpis.service_level <= 1.0

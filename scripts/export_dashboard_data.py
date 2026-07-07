"""Generate the JSON payload behind docs/index.html.

The twin itself is defined declaratively in scenarios/baseline.json — this
script only decides what to compute over it (the forecaster backtests and
the lead-time x service-level policy grid). Every scenario cell uses the
same methodology: 365-day horizon, weekly re-forecast, N_SEEDS paired
replications. The dashboard headline is the (lead_time=3, 95%) cell of the
grid, so the hero numbers and the explorer's default view always agree.

Run after changing any model: python scripts/export_dashboard_data.py
(takes ~10 minutes — Holt-Winters refits weekly inside every run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain_twin.engine import SimulationEngine, run_replications
from supply_chain_twin.forecasting import (
    ExponentialSmoothingForecaster,
    GradientBoostingForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    backtest,
)
from supply_chain_twin.policies import ForecastDrivenPolicy, ReorderUpToPolicy
from supply_chain_twin.scenario import Scenario

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data.json"

SIM_HORIZON = 365
N_SEEDS = 10
LEAD_TIMES = [3, 5, 7]
SERVICE_LEVELS = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
HEADLINE_CELL = (3, "95%")

FORECASTERS = {
    "naive": NaiveForecaster,
    "seasonal_naive": lambda: SeasonalNaiveForecaster(season_length=7),
    "holt_winters": lambda: ExponentialSmoothingForecaster(seasonal_periods=7),
    "gradient_boosting": lambda: GradientBoostingForecaster(n_estimators=150),
}
WINNING_MODEL = "holt_winters"


def static_policy(scenario: Scenario, history: np.ndarray, lead_time: int, z: float) -> ReorderUpToPolicy:
    """The no-forecasting baseline: sized once from historical averages."""
    mean_demand = float(np.mean(history))
    std_demand = float(np.std(history))
    safety_stock = z * std_demand * (lead_time ** 0.5)
    reorder_point = mean_demand * lead_time + safety_stock
    order_up_to = reorder_point + scenario.policy_defaults["cycle_days"] * mean_demand
    return ReorderUpToPolicy(
        reorder_point=reorder_point,
        order_up_to_level=order_up_to,
        fixed_order_cost=scenario.costs["fixed_order_cost"],
    )


def make_engine_factory(scenario: Scenario, history: np.ndarray, policy_factory, lead_time: int):
    process = scenario.build_process()

    def factory(seed: int) -> SimulationEngine:
        supplier, warehouse = scenario.build_nodes(lead_time_days=lead_time)
        warehouse.inventory.on_hand = float(np.mean(history)) * lead_time
        return SimulationEngine(
            warehouse=warehouse, supplier=supplier, policy=policy_factory(),
            horizon=SIM_HORIZON, demand_process=process,
            holding_cost_per_unit=scenario.costs["holding_cost_per_unit_day"],
            initial_demand_history=list(history), seed=seed,
        )

    return factory


def kpi_block(kpis) -> dict:
    return {
        "service_level": round(kpis.means.service_level, 4),
        "stockout_days": round(kpis.means.stockout_days, 1),
        "total_cost": round(kpis.means.total_cost, 0),
    }


def main() -> None:
    scenario = Scenario.load("baseline")
    history = scenario.build_history()

    forecaster_results = []
    for name, factory in FORECASTERS.items():
        r = backtest(factory, name, history, horizon=3, n_folds=6, min_train=90)
        forecaster_results.append({
            "name": name, "mae": round(r.mae, 2), "rmse": round(r.rmse, 2), "mape": round(r.mape, 1),
        })

    train_len = scenario.history_days - 30
    train, actual = history[:train_len], history[train_len:]
    demo = FORECASTERS[WINNING_MODEL]()
    demo.fit(train, start_day=0)
    predicted = demo.predict(len(actual))
    forecast_vs_actual = {
        "days": list(range(train_len, scenario.history_days)),
        "actual": [round(float(x), 1) for x in actual],
        "forecast": [round(float(x), 1) for x in predicted],
    }

    matrix = []
    for lead_time in LEAD_TIMES:
        bt = backtest(FORECASTERS[WINNING_MODEL], WINNING_MODEL, history,
                      horizon=lead_time, n_folds=6, min_train=90)
        for label, z in SERVICE_LEVELS.items():
            static_kpis = run_replications(
                make_engine_factory(scenario, history,
                                    lambda: static_policy(scenario, history, lead_time, z), lead_time),
                range(N_SEEDS),
            )
            forecast_kpis = run_replications(
                make_engine_factory(
                    scenario, history,
                    lambda: ForecastDrivenPolicy(
                        forecaster=FORECASTERS[WINNING_MODEL](), lead_time_days=lead_time,
                        residual_std=bt.residual_std, service_z=z,
                        cycle_days=scenario.policy_defaults["cycle_days"],
                        review_period_days=scenario.policy_defaults["review_period_days"],
                    ),
                    lead_time,
                ),
                range(N_SEEDS),
            )
            matrix.append({
                "lead_time": lead_time,
                "service_level_label": label,
                "static": kpi_block(static_kpis),
                "forecast": kpi_block(forecast_kpis),
            })
            print(f"lead_time={lead_time} service={label}: done", flush=True)

    headline_row = next(
        row for row in matrix
        if row["lead_time"] == HEADLINE_CELL[0] and row["service_level_label"] == HEADLINE_CELL[1]
    )
    payload = {
        "scenario": scenario.name,
        "methodology": {
            "horizon_days": SIM_HORIZON,
            "replications": N_SEEDS,
            "review_period_days": scenario.policy_defaults["review_period_days"],
        },
        "headline": {
            "lead_time": HEADLINE_CELL[0],
            "service_level_label": HEADLINE_CELL[1],
            "static": headline_row["static"],
            "forecast": headline_row["forecast"],
        },
        "forecaster_comparison": forecaster_results,
        "winning_model": WINNING_MODEL,
        "forecast_vs_actual": forecast_vs_actual,
        "scenario_matrix": matrix,
        "lead_times": LEAD_TIMES,
        "service_levels": list(SERVICE_LEVELS.keys()),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

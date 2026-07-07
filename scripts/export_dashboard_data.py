"""Generate the JSON payload embedded in docs/index.html.

Run after changing any model in supply_chain_twin/ to refresh the live
dashboard's data: python scripts/export_dashboard_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain_twin.demand import SeasonalDemandProcess, generate_history
from supply_chain_twin.engine import SimulationEngine, run_replications
from supply_chain_twin.entities import Node, NodeType
from supply_chain_twin.forecasting import (
    ExponentialSmoothingForecaster,
    GradientBoostingForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    backtest,
)
from supply_chain_twin.policies import ForecastDrivenPolicy, ReorderUpToPolicy

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "data.json"

HISTORY_DAYS = 180
SIM_HORIZON = 180
REVIEW_PERIOD_DAYS = 14
N_SEEDS = 5
CYCLE_DAYS = 7
LEAD_TIMES = [3, 5, 7]
SERVICE_LEVELS = {"90%": 1.28, "95%": 1.65, "99%": 2.33}

FORECASTERS = {
    "naive": NaiveForecaster,
    "seasonal_naive": lambda: SeasonalNaiveForecaster(season_length=7),
    "holt_winters": lambda: ExponentialSmoothingForecaster(seasonal_periods=7),
    "gradient_boosting": lambda: GradientBoostingForecaster(n_estimators=150),
}
WINNING_MODEL = "holt_winters"


def build_history():
    process = SeasonalDemandProcess(base_level=40.0, trend_per_day=0.03, noise_std=3.0)
    rng = np.random.default_rng(123)
    history = generate_history(process, days=HISTORY_DAYS, rng=rng)
    return process, history


def static_policy(history: np.ndarray, lead_time: int, z: float) -> ReorderUpToPolicy:
    mean_demand = float(np.mean(history))
    std_demand = float(np.std(history))
    safety_stock = z * std_demand * (lead_time ** 0.5)
    reorder_point = mean_demand * lead_time + safety_stock
    order_up_to = reorder_point + CYCLE_DAYS * mean_demand
    return ReorderUpToPolicy(reorder_point=reorder_point, order_up_to_level=order_up_to)


def make_engine(policy_factory, process, history, lead_time: int):
    def factory(seed: int) -> SimulationEngine:
        supplier = Node(name="Supplier", node_type=NodeType.SUPPLIER)
        warehouse = Node(name="Central Warehouse", node_type=NodeType.WAREHOUSE, lead_time_days=lead_time)
        warehouse.inventory.on_hand = float(np.mean(history)) * lead_time
        return SimulationEngine(
            warehouse=warehouse, supplier=supplier, policy=policy_factory(),
            horizon=SIM_HORIZON, demand_process=process,
            initial_demand_history=list(history), seed=seed,
        )
    return factory


def main() -> None:
    process, history = build_history()

    # --- Forecaster comparison (fixed at lead time 3, for the headline chart) ---
    forecaster_results = []
    for name, factory in FORECASTERS.items():
        r = backtest(factory, name, history, horizon=3, n_folds=6, min_train=90)
        forecaster_results.append({"name": name, "mae": round(r.mae, 2), "rmse": round(r.rmse, 2), "mape": round(r.mape, 1)})

    # --- Forecast vs actual (held-out last 30 of 180 days, winning model) ---
    train_len = HISTORY_DAYS - 30
    train, actual = history[:train_len], history[train_len:]
    demo = FORECASTERS[WINNING_MODEL]()
    demo.fit(train, start_day=0)
    predicted = demo.predict(len(actual))
    forecast_vs_actual = {
        "days": list(range(train_len, HISTORY_DAYS)),
        "actual": [round(float(x), 1) for x in actual],
        "forecast": [round(float(x), 1) for x in predicted],
    }

    # --- Scenario matrix: lead time x service level, static vs forecast-driven ---
    matrix = []
    for lead_time in LEAD_TIMES:
        bt = backtest(FORECASTERS[WINNING_MODEL], WINNING_MODEL, history, horizon=lead_time, n_folds=6, min_train=90)
        for label, z in SERVICE_LEVELS.items():
            static_kpis = run_replications(
                make_engine(lambda: static_policy(history, lead_time, z), process, history, lead_time),
                range(N_SEEDS),
            )
            forecast_kpis = run_replications(
                make_engine(
                    lambda: ForecastDrivenPolicy(
                        forecaster=FORECASTERS[WINNING_MODEL](), lead_time_days=lead_time,
                        residual_std=bt.residual_std, service_z=z, cycle_days=CYCLE_DAYS,
                        review_period_days=REVIEW_PERIOD_DAYS,
                    ),
                    process, history, lead_time,
                ),
                range(N_SEEDS),
            )
            matrix.append({
                "lead_time": lead_time,
                "service_level_label": label,
                "static": {
                    "service_level": round(static_kpis.means.service_level, 4),
                    "stockout_days": round(static_kpis.means.stockout_days, 1),
                    "total_cost": round(static_kpis.means.total_cost, 0),
                },
                "forecast": {
                    "service_level": round(forecast_kpis.means.service_level, 4),
                    "stockout_days": round(forecast_kpis.means.stockout_days, 1),
                    "total_cost": round(forecast_kpis.means.total_cost, 0),
                },
            })
            print(f"lead_time={lead_time} service={label}: done")

    payload = {
        "forecaster_comparison": forecaster_results,
        "winning_model": WINNING_MODEL,
        "forecast_vs_actual": forecast_vs_actual,
        "scenario_matrix": matrix,
        "lead_times": LEAD_TIMES,
        "service_levels": list(SERVICE_LEVELS.keys()),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Phase 2 entry point.

1. Generate 180 days of pre-twin historical demand from a seasonal process
   (weekly pattern + mild trend) — the static Phase 1 process had no pattern
   to forecast, so this is the first thing that makes forecasting worth doing.
2. Backtest four forecasters against that history and print a comparison.
3. Build a ForecastDrivenPolicy from the winning forecaster and run it
   forward, alongside a Phase 1-style static policy sized from the same
   historical data, on the *same* seasonal demand process — isolating what
   forecasting actually buys over a naive average-based reorder policy.
4. Save a forecast-vs-actual chart, a forecaster comparison chart, and a
   policy comparison chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .demand import SeasonalDemandProcess, generate_history
from .engine import SimulationEngine, run_replications
from .entities import Node, NodeType
from .forecasting import (
    BacktestResult,
    ExponentialSmoothingForecaster,
    Forecaster,
    GradientBoostingForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    backtest,
)
from .policies import ForecastDrivenPolicy, ReorderPolicy, ReorderUpToPolicy

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

HISTORY_DAYS = 180
BACKTEST_HORIZON = 7  # matches the lead time used downstream
LEAD_TIME_DAYS = 3
SIM_HORIZON = 365
N_SEEDS = 20
SERVICE_Z = 1.65
CYCLE_DAYS = 7

FORECASTER_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": lambda: SeasonalNaiveForecaster(season_length=7),
    "holt_winters": lambda: ExponentialSmoothingForecaster(seasonal_periods=7),
    "gradient_boosting": lambda: GradientBoostingForecaster(n_estimators=150),
}


def build_history() -> tuple[SeasonalDemandProcess, np.ndarray]:
    process = SeasonalDemandProcess(base_level=40.0, trend_per_day=0.03, noise_std=3.0)
    rng = np.random.default_rng(123)
    history = generate_history(process, days=HISTORY_DAYS, rng=rng)
    return process, history


def run_backtests(history: np.ndarray) -> list[BacktestResult]:
    return [
        backtest(factory, name, history, horizon=BACKTEST_HORIZON, n_folds=6, min_train=90)
        for name, factory in FORECASTER_FACTORIES.items()
    ]


def make_static_baseline_policy(history: np.ndarray) -> ReorderUpToPolicy:
    """What a planner would set by hand from historical averages, with no
    forecasting: safety stock sized off the raw historical demand std
    rather than a model's forecast error."""
    mean_demand = float(np.mean(history))
    std_demand = float(np.std(history))
    safety_stock = SERVICE_Z * std_demand * (LEAD_TIME_DAYS ** 0.5)
    reorder_point = mean_demand * LEAD_TIME_DAYS + safety_stock
    order_up_to = reorder_point + CYCLE_DAYS * mean_demand
    return ReorderUpToPolicy(reorder_point=reorder_point, order_up_to_level=order_up_to)


def make_forecast_policy(best: BacktestResult) -> ForecastDrivenPolicy:
    return ForecastDrivenPolicy(
        forecaster=FORECASTER_FACTORIES[best.name](),
        lead_time_days=LEAD_TIME_DAYS,
        residual_std=best.residual_std,
        service_z=SERVICE_Z,
        cycle_days=CYCLE_DAYS,
        review_period_days=7,
    )


def make_engine(
    policy_factory: Callable[[], ReorderPolicy],
    process: SeasonalDemandProcess,
    history: np.ndarray,
) -> Callable[[int], SimulationEngine]:
    """Return a seed -> SimulationEngine factory for run_replications, all
    sharing the same demand process and historical seed data."""

    def factory(seed: int) -> SimulationEngine:
        supplier = Node(name="Supplier", node_type=NodeType.SUPPLIER)
        warehouse = Node(name="Central Warehouse", node_type=NodeType.WAREHOUSE, lead_time_days=LEAD_TIME_DAYS)
        warehouse.inventory.on_hand = float(np.mean(history)) * LEAD_TIME_DAYS
        return SimulationEngine(
            warehouse=warehouse,
            supplier=supplier,
            policy=policy_factory(),
            horizon=SIM_HORIZON,
            demand_process=process,
            initial_demand_history=list(history),
            seed=seed,
        )

    return factory


def plot_forecast_vs_actual(history: np.ndarray, name: str) -> None:
    train_len = HISTORY_DAYS - 30
    train, actual = history[:train_len], history[train_len:]
    forecaster = FORECASTER_FACTORIES[name]()
    forecaster.fit(train, start_day=0)
    forecast = forecaster.predict(len(actual))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    days = range(train_len, HISTORY_DAYS)
    ax.plot(days, actual, label="Actual", linewidth=1.4)
    ax.plot(days, forecast, label=f"Forecast ({name})", linewidth=1.4, linestyle="--")
    ax.set_title(f"Held-out forecast vs actual demand - last 30 of {HISTORY_DAYS} historical days")
    ax.set_xlabel("Day")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_mae_comparison(results: list[BacktestResult]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r.name for r in results]
    maes = [r.mae for r in results]
    colors = ["#888888"] * len(results)
    colors[int(np.argmin(maes))] = "#2a7a3f"
    ax.bar(names, maes, color=colors)
    ax.set_title("Backtest MAE by forecaster (lower is better)")
    ax.set_ylabel("MAE (units/day)")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "forecaster_comparison.png", dpi=150)
    plt.close(fig)


def plot_policy_comparison(static_kpis, forecast_kpis) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    labels = ["Static\n(Phase 1)", "Forecast-driven\n(Phase 2)"]
    colors = ["#888888", "#2a7a3f"]

    axes[0].bar(labels, [static_kpis.means.stockout_days, forecast_kpis.means.stockout_days],
                yerr=[static_kpis.stds.stockout_days, forecast_kpis.stds.stockout_days],
                color=colors, capsize=5)
    axes[0].set_title("Stockout days / year")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, [static_kpis.means.service_level, forecast_kpis.means.service_level],
                yerr=[static_kpis.stds.service_level, forecast_kpis.stds.service_level],
                color=colors, capsize=5)
    axes[1].set_title("Service level")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(labels, [static_kpis.means.total_cost, forecast_kpis.means.total_cost],
                yerr=[static_kpis.stds.total_cost, forecast_kpis.stds.total_cost],
                color=colors, capsize=5)
    axes[2].set_title("Total cost / year")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "policy_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    process, history = build_history()

    print("=" * 56)
    print("  Forecaster backtest (6-fold walk-forward, 7-day horizon)")
    print("=" * 56)
    print(BacktestResult.header())
    results = run_backtests(history)
    for r in results:
        print(r.row())

    best = min(results, key=lambda r: r.mae)
    print(f"\nBest by MAE: {best.name}")

    plot_mae_comparison(results)
    plot_forecast_vs_actual(history, best.name)

    print("\n" + "=" * 56)
    print(f"  Policy comparison: static vs forecast-driven ({best.name})")
    print(f"  {N_SEEDS} seeds, {SIM_HORIZON}-day horizon, same seasonal demand")
    print("=" * 56)

    static_engine_factory = make_engine(lambda: make_static_baseline_policy(history), process, history)
    forecast_engine_factory = make_engine(lambda: make_forecast_policy(best), process, history)

    static_kpis = run_replications(static_engine_factory, range(N_SEEDS))
    forecast_kpis = run_replications(forecast_engine_factory, range(N_SEEDS))

    print("\n-- Static (Phase 1) --")
    print(static_kpis.report())
    print("\n-- Forecast-driven (Phase 2) --")
    print(forecast_kpis.report())

    cost_change = (
        (forecast_kpis.means.total_cost - static_kpis.means.total_cost)
        / static_kpis.means.total_cost * 100
    )
    print(f"\nTotal cost change vs static: {cost_change:+.1f}%")
    print(f"Service level: {static_kpis.means.service_level:.1%} (static) vs "
          f"{forecast_kpis.means.service_level:.1%} (forecast-driven)")

    plot_policy_comparison(static_kpis, forecast_kpis)
    print(f"\nCharts saved to: {DOCS_DIR}")


if __name__ == "__main__":
    main()

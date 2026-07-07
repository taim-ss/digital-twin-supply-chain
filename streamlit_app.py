"""Interactive control panel over the digital twin.

Adjust lead time, target service level, and the forecasting model, then
watch the static (historically-tuned) and forecast-driven policies run
head to head on the same seasonal demand — live, not precomputed.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

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

st.set_page_config(page_title="Supply Chain Digital Twin", layout="wide")

FORECASTERS = {
    "Holt-Winters (seasonal + trend)": lambda: ExponentialSmoothingForecaster(seasonal_periods=7),
    "Gradient boosting (lag features)": lambda: GradientBoostingForecaster(n_estimators=120),
    "Seasonal naive (last week repeats)": lambda: SeasonalNaiveForecaster(season_length=7),
    "Naive (last value repeats)": NaiveForecaster,
}
SERVICE_LEVELS = {"90%": 1.28, "95%": 1.65, "99%": 2.33}

st.title("Supply Chain Digital Twin")
st.caption(
    "Phase 1 simulation core + Phase 2 forecasting, wired into one live control panel. "
    "Every run below executes the real simulation engine, not a lookup table."
)

with st.sidebar:
    st.header("Scenario")
    base_level = st.slider("Average daily demand", 10, 100, 40, step=5)
    trend_per_day = st.slider("Demand trend (units/day growth)", -0.2, 0.2, 0.03, step=0.01)
    lead_time_days = st.slider("Supplier lead time (days)", 1, 10, 3)
    service_label = st.select_slider("Target service level", options=list(SERVICE_LEVELS.keys()), value="95%")
    cycle_days = st.slider("Reorder cycle (days of cover)", 3, 21, 7)

    st.header("Forecaster")
    forecaster_label = st.selectbox("Model", list(FORECASTERS.keys()))

    st.header("Simulation")
    horizon = st.slider("Horizon (days)", 90, 365, 365, step=15)
    n_seeds = st.slider("Replications", 5, 30, 12)

    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

if not run_clicked:
    st.info("Set your scenario in the sidebar, then click **Run simulation**.")
    st.stop()

service_z = SERVICE_LEVELS[service_label]
forecaster_factory = FORECASTERS[forecaster_label]

process = SeasonalDemandProcess(base_level=base_level, trend_per_day=trend_per_day, noise_std=base_level * 0.075)
history_rng = np.random.default_rng(123)
history = generate_history(process, days=180, rng=history_rng)

with st.spinner("Backtesting the forecaster..."):
    result = backtest(forecaster_factory, forecaster_label, history, horizon=lead_time_days, n_folds=6, min_train=90)

mean_demand = float(np.mean(history))
std_demand = float(np.std(history))
static_safety_stock = service_z * std_demand * (lead_time_days ** 0.5)
static_reorder_point = mean_demand * lead_time_days + static_safety_stock
static_order_up_to = static_reorder_point + cycle_days * mean_demand


def make_engine_factory(policy_factory):
    def factory(seed: int) -> SimulationEngine:
        supplier = Node(name="Supplier", node_type=NodeType.SUPPLIER)
        warehouse = Node(name="Central Warehouse", node_type=NodeType.WAREHOUSE, lead_time_days=lead_time_days)
        warehouse.inventory.on_hand = mean_demand * lead_time_days
        return SimulationEngine(
            warehouse=warehouse,
            supplier=supplier,
            policy=policy_factory(),
            horizon=horizon,
            demand_process=process,
            initial_demand_history=list(history),
            seed=seed,
        )

    return factory


with st.spinner(f"Running {n_seeds} replications for both policies..."):
    static_factory = make_engine_factory(
        lambda: ReorderUpToPolicy(reorder_point=static_reorder_point, order_up_to_level=static_order_up_to)
    )
    forecast_factory = make_engine_factory(
        lambda: ForecastDrivenPolicy(
            forecaster=forecaster_factory(),
            lead_time_days=lead_time_days,
            residual_std=result.residual_std,
            service_z=service_z,
            cycle_days=cycle_days,
            review_period_days=7,
        )
    )
    static_kpis = run_replications(static_factory, range(n_seeds))
    forecast_kpis = run_replications(forecast_factory, range(n_seeds))

st.subheader("Result")
cols = st.columns(3)
cols[0].metric(
    "Stockout days / year",
    f"{forecast_kpis.means.stockout_days:.1f}",
    delta=f"{forecast_kpis.means.stockout_days - static_kpis.means.stockout_days:+.1f} vs static",
    delta_color="inverse",
)
cols[1].metric(
    "Service level",
    f"{forecast_kpis.means.service_level:.1%}",
    delta=f"{(forecast_kpis.means.service_level - static_kpis.means.service_level) * 100:+.1f} pts vs static",
)
cost_delta_pct = (forecast_kpis.means.total_cost - static_kpis.means.total_cost) / static_kpis.means.total_cost * 100
cols[2].metric(
    "Total cost / year",
    f"{forecast_kpis.means.total_cost:,.0f}",
    delta=f"{cost_delta_pct:+.1f}% vs static",
    delta_color="inverse",
)

st.caption(
    f"Forecaster backtest: MAE {result.mae:.2f}, RMSE {result.rmse:.2f}, "
    f"MAPE {result.mape:.1f}%, residual std {result.residual_std:.2f} "
    f"(this is what sizes the forecast-driven policy's safety stock)."
)

left, right = st.columns(2)
with left:
    st.markdown("**Static (Phase 1)**")
    st.code(static_kpis.report(), language=None)
with right:
    st.markdown("**Forecast-driven (Phase 2)**")
    st.code(forecast_kpis.report(), language=None)

st.subheader("Held-out forecast vs actual demand")
train_len = 150
train_hist, actual_hist = history[:train_len], history[train_len:]
demo_forecaster = forecaster_factory()
demo_forecaster.fit(train_hist, start_day=0)
predicted = demo_forecaster.predict(len(actual_hist))
chart_data = {
    "day": list(range(train_len, 180)),
    "actual": actual_hist.tolist(),
    "forecast": predicted.tolist(),
}
st.line_chart(
    {"actual": chart_data["actual"], "forecast": chart_data["forecast"]},
    x_label="last 30 historical days", y_label="units",
)

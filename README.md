# Supply Chain Digital Twin

**[→ Live results dashboard](https://taim-ss.github.io/digital-twin-supply-chain/)** — explore the tradeoff across lead times and service levels interactively.

A warehouse inventory policy tuned from historical averages, replaced by one driven by a live demand forecast. Run head to head on identical seasonal, trending demand for a full simulated year:

| | Static (historically-tuned) | Forecast-driven |
|---|---|---|
| Stockout days / year | 7.5 | **1.8** (−76%) |
| Service level | 97.9% | **99.5%** |
| Total cost / year | 35,209 | 41,162 (+16.9%) |

![Policy comparison](docs/policy_comparison.png)

Not "AI wins on every metric" — a real tradeoff, and the reason for it is the finding: the static policy is sized once from 180 days of history and never updates, so as real demand grows past that stale average, it quietly under-stocks. The forecast-driven policy re-forecasts every 7 days and holds more inventory to match *actual current* demand — trading 16.9% more holding cost for 76% fewer stockouts. Whether that trade is worth it depends on your real stockout cost, which this simulation doesn't model (only holding and ordering cost do) — explore how it shifts across lead times and service targets on the [live dashboard](https://taim-ss.github.io/digital-twin-supply-chain/).

## The forecaster behind it

Four models, backtested with 6-fold walk-forward validation:

| model | MAE | RMSE | MAPE |
|---|---|---|---|
| naive | 10.59 | 13.66 | 25.1% |
| seasonal_naive | 8.83 | 11.37 | 19.8% |
| **holt_winters** | **6.35** | **8.53** | **13.7%** |
| gradient_boosting | 8.55 | 11.43 | 18.8% |

Holt-Winters wins because the demand process has the exact trend + weekly seasonality shape it's built to model directly — gradient boosting still clearly beats the naive baseline, just not the classical method built for this pattern.

![Forecaster comparison](docs/forecaster_comparison.png)
![Forecast vs actual](docs/forecast_vs_actual.png)

## Try it yourself

An interactive control panel runs the real engine live — pick lead time, target service level, and forecaster, and watch both policies run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Architecture

```
supply_chain_twin/
├── entities.py       # Node, Inventory, Shipment
├── demand.py          # DemandProcess: stationary (Phase 1) / seasonal+trend (Phase 2)
├── forecasting.py     # 4 forecasters + walk-forward backtest
├── policies.py        # ReorderPolicy: static (s,S) and forecast-driven
├── engine.py           # SimulationEngine, KPIs, replication runner
├── run.py              # Phase 1 entry point
└── run_phase2.py       # Phase 2 entry point: backtest -> policy -> comparison
streamlit_app.py         # Live interactive control panel
docs/index.html          # The live dashboard (static, GitHub Pages)
tests/                   # 38 tests, all passing
```

`ReorderPolicy` and `DemandProcess` are both small protocols — the forecast-driven policy and the seasonal demand model plug into `SimulationEngine` without it knowing or caring which kind it's running. That's what let Phase 2 add a full forecasting pipeline without touching Phase 1's engine or breaking any of its tests.

Modeling assumptions: single echelon (one warehouse, one unconstrained supplier), lost sales (unmet demand is lost, not backordered), and the lead-time convention that an order placed on day *t* with lead time *L* arrives at the start of day *t + L*.

## Roadmap

- ~~**Phase 1** — simulation core.~~ Done.
- ~~**Phase 2** — forecasting model driving the reorder policy.~~ Done.
- **Phase 3** — a routing optimizer whose plan feeds back into the twin's state, closing the loop between forecast, simulation, and network decisions.

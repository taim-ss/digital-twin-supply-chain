"""Phase 3 entry point.

Run the multi-echelon network twin (two warehouses, four retails) under
both allocation strategies on identical demand, paired seed by seed:

1. NearestWarehouseAllocator — every retail served only by its home
   warehouse; the plan any regional org chart produces by default.
2. TransportationLPAllocator — a min-cost transportation LP over every
   warehouse-retail lane whose plan feeds back into the twin's state.

The scenario (scenarios/network.json) is built so the two diverge:
southern demand roughly doubles over the year while the South warehouse's
inbound capacity stays fixed, so only cross-shipping northern slack can
keep southern shelves stocked. Prints a replicated KPI comparison and
saves the routing story chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .network import (
    NetworkSimulationEngine,
    aggregate_network_kpis,
    load_network,
)
from .routing import NearestWarehouseAllocator, TransportationLPAllocator

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

SIM_HORIZON = 365
N_SEEDS = 10


def make_engines(cfg: dict, seed: int) -> tuple[NetworkSimulationEngine, NetworkSimulationEngine]:
    home = [r["home_warehouse"] for r in cfg["retails"]]
    naive = NetworkSimulationEngine(
        allocator=NearestWarehouseAllocator(home_warehouse=home),
        horizon=SIM_HORIZON,
        seed=seed,
        scenario=cfg,
    )
    optimized = NetworkSimulationEngine(
        allocator=TransportationLPAllocator(
            shortage_penalty=cfg["costs"]["shortage_penalty_per_unit"]
        ),
        horizon=SIM_HORIZON,
        seed=seed,
        scenario=cfg,
    )
    return naive, optimized


def plot_routing_story(
    cfg: dict,
    naive_engine: NetworkSimulationEngine,
    opt_engine: NetworkSimulationEngine,
) -> None:
    """Two panels: the squeeze (southern demand vs fixed inbound capacity),
    and what it costs each allocator (cumulative stockout days)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    south_daily_capacity = cfg["warehouses"][1]["weekly_inbound_capacity"] / 7.0
    demand = np.asarray(naive_engine.south_demand_history)
    rolling = np.convolve(demand, np.ones(7) / 7, mode="valid")
    axes[0].plot(range(6, len(demand)), rolling, color="#2a5a8f", linewidth=1.6,
                 label="Southern demand (7-day avg)")
    axes[0].axhline(south_daily_capacity, color="#b04a4a", linestyle="--", linewidth=1.4,
                    label=f"South WH inbound capacity ({south_daily_capacity:.0f}/day)")
    axes[0].set_title("The squeeze: southern demand outgrows its warehouse")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Units / day")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(naive_engine.cumulative_stockouts, color="#888888", linewidth=1.6,
                 label="Nearest-warehouse (naive)")
    axes[1].plot(opt_engine.cumulative_stockouts, color="#2a7a3f", linewidth=1.6,
                 label="Transportation LP")
    axes[1].set_title("What it costs: cumulative retail stockout days")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Stockout days (all retails)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "routing_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    cfg = load_network()

    print("=" * 56)
    print("  Network twin: naive vs LP allocation")
    print(f"  {N_SEEDS} paired seeds, {SIM_HORIZON}-day horizon")
    print("=" * 56)

    naive_results, opt_results = [], []
    story_engines = None
    for seed in range(N_SEEDS):
        naive_engine, opt_engine = make_engines(cfg, seed)
        naive_results.append(naive_engine.run())
        opt_results.append(opt_engine.run())
        if seed == 0:
            story_engines = (naive_engine, opt_engine)

    naive_kpis = aggregate_network_kpis(naive_results)
    opt_kpis = aggregate_network_kpis(opt_results)

    print("\n-- Nearest-warehouse (naive baseline) --")
    print(naive_kpis.report())
    print("\n-- Transportation LP (optimizer) --")
    print(opt_kpis.report())

    stockout_change = (
        (opt_kpis.means.stockout_days - naive_kpis.means.stockout_days)
        / naive_kpis.means.stockout_days * 100
    )
    cost_change = (
        (opt_kpis.means.total_cost - naive_kpis.means.total_cost)
        / naive_kpis.means.total_cost * 100
    )
    print(f"\nStockout days vs naive:  {stockout_change:+.1f}%")
    print(f"Total cost vs naive:     {cost_change:+.1f}%")
    print(f"Fill rate: {naive_kpis.means.fill_rate:.1%} (naive) vs "
          f"{opt_kpis.means.fill_rate:.1%} (LP)")

    plot_routing_story(cfg, *story_engines)
    print(f"\nChart saved to: {DOCS_DIR / 'routing_comparison.png'}")


if __name__ == "__main__":
    main()

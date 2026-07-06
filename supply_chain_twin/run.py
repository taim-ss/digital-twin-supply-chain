"""Entry point: run one detailed scenario plus a replication study, print KPIs,
and save a two-panel chart of inventory and demand over time."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .engine import SimulationEngine, run_replications
from .entities import Node, NodeType
from .policies import ReorderUpToPolicy


def make_engine(seed: int) -> SimulationEngine:
    """Build a fresh engine for the baseline scenario. Engines are stateful,
    so every run (and every replication) needs its own."""
    supplier = Node(name="Supplier", node_type=NodeType.SUPPLIER)
    warehouse = Node(
        name="Central Warehouse",
        node_type=NodeType.WAREHOUSE,
        lead_time_days=3,
    )
    warehouse.inventory.on_hand = 80.0

    policy = ReorderUpToPolicy(
        reorder_point=60.0,
        order_up_to_level=150.0,
        fixed_order_cost=75.0,
    )

    return SimulationEngine(
        warehouse=warehouse,
        supplier=supplier,
        policy=policy,
        horizon=365,
        demand_mean=20.0,
        demand_std=5.0,
        holding_cost_per_unit=0.5,
        seed=seed,
    )


def main() -> None:
    # Single detailed run (seed fixed for a reproducible chart)
    engine = make_engine(seed=42)
    kpis = engine.run()

    print("=" * 48)
    print("  SUPPLY CHAIN TWIN - Phase 1 (single run)")
    print("=" * 48)
    print(kpis.report())

    # Replication study: same scenario, 30 independent demand streams
    replicated = run_replications(make_engine, seeds=range(30))
    print()
    print("=" * 48)
    print("  Replication study (30 seeds)")
    print("=" * 48)
    print(replicated.report())

    # Two-panel chart from the detailed run
    policy = engine.policy
    days = range(len(engine.inventory_history))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(days, engine.inventory_history, label="On-hand inventory", linewidth=1.2)
    ax1.axhline(policy.reorder_point, color="orange", linestyle="--",
                label=f"Reorder point ({policy.reorder_point:.0f})")
    ax1.axhline(policy.order_up_to_level, color="green", linestyle=":",
                label=f"Order-up-to ({policy.order_up_to_level:.0f})")
    ax1.set_title("Inventory Over Time - Central Warehouse")
    ax1.set_ylabel("Units")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(days, engine.demand_history, label="Demand", linewidth=0.9, alpha=0.8)
    ax2.plot(days, engine.fulfilled_history, label="Fulfilled", linewidth=0.9, alpha=0.8)
    ax2.set_title("Daily Demand vs Fulfilled")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Units")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "simulation_chart.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nChart saved to: {out}")


if __name__ == "__main__":
    main()

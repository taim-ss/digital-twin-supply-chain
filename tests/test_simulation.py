"""Deterministic tests for the Phase 1 simulation core."""

from __future__ import annotations

import pytest

from supply_chain_twin import (
    KPIs,
    Node,
    NodeType,
    ReorderUpToPolicy,
    SimulationEngine,
    run_replications,
)


def make_engine(
    seed: int = 0,
    *,
    demand_mean: float = 20.0,
    demand_std: float = 5.0,
    horizon: int = 365,
    starting_stock: float = 80.0,
    lead_time_days: int = 3,
    reorder_point: float = 60.0,
    order_up_to: float = 150.0,
) -> SimulationEngine:
    supplier = Node(name="S", node_type=NodeType.SUPPLIER)
    warehouse = Node(name="W", node_type=NodeType.WAREHOUSE, lead_time_days=lead_time_days)
    warehouse.inventory.on_hand = starting_stock
    policy = ReorderUpToPolicy(reorder_point=reorder_point, order_up_to_level=order_up_to)
    return SimulationEngine(
        warehouse=warehouse,
        supplier=supplier,
        policy=policy,
        horizon=horizon,
        demand_mean=demand_mean,
        demand_std=demand_std,
        seed=seed,
    )


class TestPolicy:
    def test_no_order_above_reorder_point(self):
        policy = ReorderUpToPolicy(reorder_point=60, order_up_to_level=150)
        assert policy.order_quantity(61) == 0.0

    def test_orders_gap_to_s_at_or_below_reorder_point(self):
        policy = ReorderUpToPolicy(reorder_point=60, order_up_to_level=150)
        assert policy.order_quantity(60) == 90.0
        assert policy.order_quantity(10) == 140.0

    def test_rejects_inverted_levels(self):
        with pytest.raises(ValueError):
            ReorderUpToPolicy(reorder_point=150, order_up_to_level=60)


class TestEngine:
    def test_zero_demand_means_no_stockouts_and_flat_inventory(self):
        engine = make_engine(demand_mean=0.0, demand_std=0.0, horizon=50)
        kpis = engine.run()
        assert kpis.stockout_days == 0
        assert kpis.service_level == 1.0
        assert kpis.fill_rate == 1.0  # vacuous demand counts as fully served
        assert engine.inventory_history == [80.0] * 50

    def test_order_arrives_after_lead_time(self):
        # Start below reorder point with zero demand: order placed day 0,
        # lead time 3 means arrival at the start of day 3.
        engine = make_engine(
            demand_mean=0.0, demand_std=0.0, horizon=5,
            starting_stock=10.0, lead_time_days=3,
        )
        engine.run()
        # Days 0-2 end with the original 10 on hand; day 3 onward includes the arrival.
        assert engine.inventory_history[:3] == [10.0, 10.0, 10.0]
        assert engine.inventory_history[3] == 150.0

    def test_inventory_position_prevents_duplicate_orders(self):
        engine = make_engine(
            demand_mean=0.0, demand_std=0.0, horizon=5,
            starting_stock=10.0, lead_time_days=3,
        )
        kpis = engine.run()
        # Only one order should ever be placed: position stays at S once ordered.
        assert kpis.total_ordering_cost == engine.policy.fixed_order_cost

    def test_seed_reproducibility(self):
        a = make_engine(seed=7).run()
        b = make_engine(seed=7).run()
        assert a == b

    def test_heavy_demand_causes_stockouts(self):
        engine = make_engine(demand_mean=500.0, horizon=30)
        kpis = engine.run()
        assert kpis.stockout_days > 0
        assert kpis.fill_rate < 1.0

    def test_rejects_wrong_node_types(self):
        supplier = Node(name="S", node_type=NodeType.SUPPLIER)
        with pytest.raises(ValueError):
            SimulationEngine(
                warehouse=supplier,  # wrong type on purpose
                supplier=supplier,
                policy=ReorderUpToPolicy(reorder_point=1, order_up_to_level=2),
            )


class TestReplications:
    def test_aggregates_across_seeds(self):
        result = run_replications(make_engine, seeds=range(5))
        assert result.replications == 5
        assert 0.0 <= result.means.service_level <= 1.0
        assert result.stds.total_holding_cost >= 0.0

    def test_requires_at_least_two_seeds(self):
        with pytest.raises(ValueError):
            run_replications(make_engine, seeds=[1])


class TestKPIs:
    def test_total_cost_is_sum_of_components(self):
        kpis = KPIs(total_holding_cost=100.0, total_ordering_cost=50.0)
        assert kpis.total_cost == 150.0

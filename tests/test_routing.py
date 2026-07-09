"""Tests for the allocation strategies and the network engine."""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain_twin.network import (
    NetworkSimulationEngine,
    aggregate_network_kpis,
    load_network,
)
from supply_chain_twin.routing import NearestWarehouseAllocator, TransportationLPAllocator


COST = [[0.5, 2.0], [2.0, 0.5]]  # wh0 near retail0, wh1 near retail1


class TestNearestWarehouseAllocator:
    def test_serves_home_retail_fully_when_stocked(self):
        alloc = NearestWarehouseAllocator(home_warehouse=[0, 1])
        plan = alloc.allocate(needs=[10, 20], available=[100, 100], cost_per_unit=COST)
        assert plan[0][0] == 10 and plan[1][1] == 20
        assert plan[0][1] == 0 and plan[1][0] == 0  # never crosses

    def test_scales_pro_rata_under_scarcity(self):
        alloc = NearestWarehouseAllocator(home_warehouse=[0, 0])
        plan = alloc.allocate(needs=[30, 10], available=[20, 0], cost_per_unit=COST)
        assert plan[0][0] == pytest.approx(15)  # 30 * 20/40
        assert plan[0][1] == pytest.approx(5)   # 10 * 20/40

    def test_never_crosses_even_when_other_warehouse_has_slack(self):
        alloc = NearestWarehouseAllocator(home_warehouse=[0, 1])
        plan = alloc.allocate(needs=[50, 0], available=[10, 500], cost_per_unit=COST)
        assert plan[1][0] == 0  # the defining limitation of the baseline
        assert plan[0][0] == 10


class TestTransportationLPAllocator:
    def test_prefers_cheap_home_lane(self):
        alloc = TransportationLPAllocator(shortage_penalty=10)
        plan = alloc.allocate(needs=[10, 10], available=[50, 50], cost_per_unit=COST)
        assert plan[0][0] == pytest.approx(10, abs=1e-6)
        assert plan[1][1] == pytest.approx(10, abs=1e-6)
        assert plan[0][1] == pytest.approx(0, abs=1e-6)

    def test_cross_ships_when_home_warehouse_is_short(self):
        alloc = TransportationLPAllocator(shortage_penalty=10)
        plan = alloc.allocate(needs=[0, 40], available=[100, 10], cost_per_unit=COST)
        assert plan[1][1] == pytest.approx(10, abs=1e-6)  # home stock first
        assert plan[0][1] == pytest.approx(30, abs=1e-6)  # slack re-routed

    def test_leaves_demand_short_when_penalty_below_lane_cost(self):
        expensive = [[0.5, 50.0], [50.0, 0.5]]  # crossing costs more than the penalty
        alloc = TransportationLPAllocator(shortage_penalty=10)
        plan = alloc.allocate(needs=[0, 40], available=[100, 0], cost_per_unit=expensive)
        assert plan[0][1] == pytest.approx(0, abs=1e-6)  # economically rational shortage

    def test_respects_supply_limits(self):
        alloc = TransportationLPAllocator(shortage_penalty=10)
        plan = alloc.allocate(needs=[100, 100], available=[30, 30], cost_per_unit=COST)
        assert plan.sum(axis=1)[0] <= 30 + 1e-6
        assert plan.sum(axis=1)[1] <= 30 + 1e-6


class TestNetworkEngine:
    def test_runs_and_produces_sane_kpis(self):
        cfg = load_network()
        engine = NetworkSimulationEngine(
            allocator=TransportationLPAllocator(shortage_penalty=cfg["costs"]["shortage_penalty_per_unit"]),
            horizon=60, seed=1,
        )
        kpis = engine.run()
        assert 0.0 <= kpis.fill_rate <= 1.0
        assert kpis.transport_cost > 0
        assert kpis.total_cost == pytest.approx(
            kpis.transport_cost + kpis.holding_cost + kpis.ordering_cost
        )

    def test_allocation_plan_mutates_twin_state(self):
        # The loop-closing property: after a review day, retail pipelines
        # hold shipments created by the optimizer's plan.
        engine = NetworkSimulationEngine(
            allocator=TransportationLPAllocator(shortage_penalty=10), horizon=1, seed=0,
        )
        engine.run()
        pipeline_units = sum(
            s.quantity for inv in engine.retail_inv for s in inv.pending_shipments
        )
        assert pipeline_units > 0

    def test_seed_reproducibility(self):
        def run(seed):
            return NetworkSimulationEngine(
                allocator=NearestWarehouseAllocator(home_warehouse=[0, 0, 1, 1]),
                horizon=45, seed=seed,
            ).run()
        assert run(5) == run(5)

    def test_aggregation_requires_two_results(self):
        with pytest.raises(ValueError):
            aggregate_network_kpis([NetworkSimulationEngine(
                allocator=NearestWarehouseAllocator(home_warehouse=[0, 0, 1, 1]),
                horizon=10, seed=0,
            ).run()])

    def test_optimizer_beats_naive_on_stockouts_in_the_squeeze(self):
        # The scenario's design claim: with southern demand outgrowing the
        # South WH's capacity, the LP must produce fewer stockouts.
        cfg = load_network()
        naive = NetworkSimulationEngine(
            allocator=NearestWarehouseAllocator(home_warehouse=[r["home_warehouse"] for r in cfg["retails"]]),
            horizon=365, seed=3,
        ).run()
        optimized = NetworkSimulationEngine(
            allocator=TransportationLPAllocator(shortage_penalty=cfg["costs"]["shortage_penalty_per_unit"]),
            horizon=365, seed=3,
        ).run()
        assert optimized.stockout_days < naive.stockout_days
        assert optimized.fill_rate > naive.fill_rate

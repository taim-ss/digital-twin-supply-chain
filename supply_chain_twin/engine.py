"""Day-stepped simulation engine with stochastic demand and KPI tracking.

Model assumptions (Phase 1):
- Single echelon: one warehouse replenished by an unconstrained supplier.
- Lost sales: demand not met from on-hand stock is lost, not backordered.
- Lead time convention: an order placed on day t with lead time L arrives
  at the start of day t + L.

Phase 2 adds:
- A pluggable `DemandProcess` (default: the Phase 1 stationary Poisson model,
  unchanged) so a seasonal/trending process can replace it without touching
  the engine.
- `initial_demand_history`: pre-twin historical demand the policy can
  forecast from starting on day 0, and which anchors the absolute day
  index so day-of-week features stay aligned across history + simulation.
- A per-day call to `policy.refresh(...)`, letting a forecast-driven policy
  periodically re-fit itself against demand observed so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional, Sequence

import numpy as np

from .demand import DemandProcess, StationaryPoissonProcess
from .entities import Node, NodeType, Shipment
from .policies import ReorderPolicy


@dataclass
class KPIs:
    """Aggregated performance metrics for a completed simulation run."""

    service_level: float = 0.0  # fraction of days with no stockout
    fill_rate: float = 0.0  # fraction of demanded units actually fulfilled
    stockout_days: int = 0
    average_inventory: float = 0.0
    total_holding_cost: float = 0.0
    total_ordering_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.total_holding_cost + self.total_ordering_cost

    def report(self) -> str:
        lines = [
            f"{'Service level':<22} {self.service_level:>8.1%}",
            f"{'Fill rate':<22} {self.fill_rate:>8.1%}",
            f"{'Stockout days':<22} {self.stockout_days:>8.1f}",
            f"{'Average inventory':<22} {self.average_inventory:>8.1f}",
            f"{'Holding cost':<22} {self.total_holding_cost:>8.2f}",
            f"{'Ordering cost':<22} {self.total_ordering_cost:>8.2f}",
            f"{'Total cost':<22} {self.total_cost:>8.2f}",
        ]
        return "\n".join(lines)


@dataclass
class ReplicatedKPIs:
    """Mean and standard deviation of each KPI across independent replications."""

    means: KPIs
    stds: KPIs
    replications: int

    def report(self) -> str:
        rows = [
            ("Service level", self.means.service_level, self.stds.service_level, "{:>7.1%}"),
            ("Fill rate", self.means.fill_rate, self.stds.fill_rate, "{:>7.1%}"),
            ("Stockout days", self.means.stockout_days, self.stds.stockout_days, "{:>7.1f}"),
            ("Average inventory", self.means.average_inventory, self.stds.average_inventory, "{:>7.1f}"),
            ("Holding cost", self.means.total_holding_cost, self.stds.total_holding_cost, "{:>7.0f}"),
            ("Ordering cost", self.means.total_ordering_cost, self.stds.total_ordering_cost, "{:>7.0f}"),
            ("Total cost", self.means.total_cost, self.stds.total_cost, "{:>7.0f}"),
        ]
        lines = [f"{'KPI':<22} {'mean':>7}   {'std':>7}   (n={self.replications})"]
        for label, m, s, fmt in rows:
            lines.append(f"{label:<22} {fmt.format(m)}   {fmt.format(s)}")
        return "\n".join(lines)


class SimulationEngine:
    """Simulates one warehouse, replenished from an unconstrained supplier,
    facing stochastic daily demand over a fixed horizon.

    Day order of operations: receive due shipments -> demand arrives and is
    fulfilled from on-hand -> policy refreshed and a replenishment order
    placed if triggered -> costs and snapshot recorded.
    """

    def __init__(
        self,
        warehouse: Node,
        supplier: Node,
        policy: ReorderPolicy,
        horizon: int = 365,
        demand_mean: float = 20.0,
        demand_std: float = 5.0,
        holding_cost_per_unit: float = 0.5,
        seed: Optional[int] = None,
        demand_process: Optional[DemandProcess] = None,
        initial_demand_history: Optional[Sequence[float]] = None,
    ) -> None:
        if warehouse.node_type not in (NodeType.WAREHOUSE, NodeType.RETAIL):
            raise ValueError("warehouse must be a WAREHOUSE or RETAIL node")
        if supplier.node_type != NodeType.SUPPLIER:
            raise ValueError("supplier must be a SUPPLIER node")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        warehouse.upstream = supplier
        self.warehouse = warehouse
        self.policy = policy
        self.horizon = horizon
        self.holding_cost_per_unit = holding_cost_per_unit
        self.rng = np.random.default_rng(seed)

        # Phase 1 behavior is preserved exactly when demand_process is
        # omitted: same rng call sequence as the original inline formula.
        self.demand_process: DemandProcess = demand_process or StationaryPoissonProcess(
            mean=demand_mean, std=demand_std
        )

        seed_history = list(initial_demand_history) if initial_demand_history is not None else []
        self.history_offset = len(seed_history)
        self.demand_history: list[float] = seed_history
        self.fulfilled_history: list[float] = []

    @property
    def inventory_history(self) -> list[float]:
        return self.warehouse.inventory.history

    def _generate_demand(self, absolute_day: int) -> float:
        return self.demand_process.sample(absolute_day, self.rng)

    def _receive_shipments(self) -> None:
        """Age pending shipments by one day; deliver those whose lead time elapsed."""
        inv = self.warehouse.inventory
        still_pending = []
        for shipment in inv.pending_shipments:
            shipment.remaining_days -= 1
            if shipment.remaining_days <= 0:
                inv.on_hand += shipment.quantity
                inv.on_order -= shipment.quantity
            else:
                still_pending.append(shipment)
        inv.pending_shipments.clear()
        inv.pending_shipments.extend(still_pending)

    def _place_order_if_needed(self) -> float:
        """Evaluate the reorder policy and place a replenishment order if triggered.
        Returns the fixed ordering cost incurred (0 if no order was placed)."""
        inv = self.warehouse.inventory
        qty = self.policy.order_quantity(inv.position)
        if qty <= 0:
            return 0.0
        inv.on_order += qty
        inv.pending_shipments.append(
            Shipment(quantity=qty, remaining_days=self.warehouse.lead_time_days)
        )
        return self.policy.fixed_order_cost

    def run(self) -> KPIs:
        """Execute the simulation and return aggregated KPIs."""
        inv = self.warehouse.inventory
        total_demand = 0.0
        total_fulfilled = 0.0
        stockout_days = 0
        total_holding_cost = 0.0
        total_ordering_cost = 0.0

        for step in range(self.horizon):
            absolute_day = self.history_offset + step
            self._receive_shipments()

            demand = self._generate_demand(absolute_day)
            fulfilled = min(demand, inv.on_hand)
            inv.on_hand -= fulfilled
            total_demand += demand
            total_fulfilled += fulfilled
            if fulfilled < demand:
                stockout_days += 1

            self.demand_history.append(demand)
            self.policy.refresh(self.demand_history, absolute_day)
            total_ordering_cost += self._place_order_if_needed()
            total_holding_cost += inv.on_hand * self.holding_cost_per_unit

            inv.snapshot()
            self.fulfilled_history.append(fulfilled)

        average_inventory = mean(inv.history) if inv.history else 0.0

        return KPIs(
            service_level=(self.horizon - stockout_days) / self.horizon,
            fill_rate=total_fulfilled / total_demand if total_demand > 0 else 1.0,
            stockout_days=stockout_days,
            average_inventory=average_inventory,
            total_holding_cost=total_holding_cost,
            total_ordering_cost=total_ordering_cost,
        )


def run_replications(make_engine, seeds: Sequence[int]) -> ReplicatedKPIs:
    """Run the same scenario across independent seeds and aggregate KPIs.

    `make_engine` is a factory taking one seed argument and returning a
    fresh SimulationEngine — a factory is required because engines and their
    nodes are stateful and cannot be reused across runs.
    """
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds for a std estimate")

    results = [make_engine(seed).run() for seed in seeds]

    def agg(attr: str, fn) -> float:
        return fn([getattr(r, attr) for r in results])

    fields = (
        "service_level",
        "fill_rate",
        "stockout_days",
        "average_inventory",
        "total_holding_cost",
        "total_ordering_cost",
    )
    means = KPIs(**{f: agg(f, mean) for f in fields})
    stds = KPIs(**{f: agg(f, stdev) for f in fields})
    return ReplicatedKPIs(means=means, stds=stds, replications=len(seeds))

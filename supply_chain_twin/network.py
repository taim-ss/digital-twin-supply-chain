"""Phase 3: the multi-echelon network twin.

Two regional warehouses serve four retail stores. Weekly, the twin
forecasts each retail's demand, replenishes each warehouse from the
supplier (up to a fixed inbound capacity), and asks an `Allocator` how to
distribute warehouse stock to retails. The allocator's plan creates real
shipments that mutate the twin's state — the feedback loop that closes
the forecast -> optimize -> simulate cycle.

The network itself is defined declaratively in scenarios/network.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Sequence

import numpy as np

from .demand import SeasonalDemandProcess, generate_history
from .entities import Inventory, Shipment
from .forecasting import SeasonalNaiveForecaster
from .routing import Allocator

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def load_network(name: str = "network") -> dict:
    return json.loads((SCENARIOS_DIR / f"{name}.json").read_text())


@dataclass
class NetworkKPIs:
    """Network-wide performance for a completed run."""

    fill_rate: float = 0.0
    stockout_days: float = 0.0  # summed across retails
    transport_cost: float = 0.0
    holding_cost: float = 0.0
    ordering_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.transport_cost + self.holding_cost + self.ordering_cost

    def report(self) -> str:
        lines = [
            f"{'Fill rate':<22} {self.fill_rate:>10.1%}",
            f"{'Stockout days':<22} {self.stockout_days:>10.1f}",
            f"{'Transport cost':<22} {self.transport_cost:>10.0f}",
            f"{'Holding cost':<22} {self.holding_cost:>10.0f}",
            f"{'Ordering cost':<22} {self.ordering_cost:>10.0f}",
            f"{'Total cost':<22} {self.total_cost:>10.0f}",
        ]
        return "\n".join(lines)


@dataclass
class AggregatedNetworkKPIs:
    means: NetworkKPIs
    stds: NetworkKPIs
    replications: int

    def report(self) -> str:
        rows = [
            ("Fill rate", "{:>9.1%}", lambda k: k.fill_rate),
            ("Stockout days", "{:>9.1f}", lambda k: k.stockout_days),
            ("Transport cost", "{:>9.0f}", lambda k: k.transport_cost),
            ("Holding cost", "{:>9.0f}", lambda k: k.holding_cost),
            ("Ordering cost", "{:>9.0f}", lambda k: k.ordering_cost),
            ("Total cost", "{:>9.0f}", lambda k: k.total_cost),
        ]
        lines = [f"{'KPI':<22} {'mean':>9}   {'std':>9}   (n={self.replications})"]
        for label, fmt, get in rows:
            lines.append(f"{label:<22} {fmt.format(get(self.means))}   {fmt.format(get(self.stds))}")
        return "\n".join(lines)


def aggregate_network_kpis(results: Sequence[NetworkKPIs]) -> AggregatedNetworkKPIs:
    if len(results) < 2:
        raise ValueError("need at least 2 replications for a std estimate")
    names = [f.name for f in dataclass_fields(NetworkKPIs)]
    means = NetworkKPIs(**{n: mean(getattr(r, n) for r in results) for n in names})
    stds = NetworkKPIs(**{n: stdev(getattr(r, n) for r in results) for n in names})
    return AggregatedNetworkKPIs(means=means, stds=stds, replications=len(results))


class NetworkSimulationEngine:
    """Day-stepped simulation of the warehouse-retail network.

    Weekly cycle: forecast each retail (seasonal-naive — cheap and it
    tracks the trend with a one-week lag), replenish each warehouse from
    the unconstrained supplier up to its regional target but never above
    its inbound capacity, then hand retail needs + warehouse stock to the
    allocator and execute its plan as real shipments.
    """

    SERVICE_Z = 1.65

    def __init__(
        self,
        allocator: Allocator,
        horizon: int = 365,
        seed: Optional[int] = None,
        scenario: Optional[dict] = None,
    ) -> None:
        self.cfg = scenario or load_network()
        self.allocator = allocator
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

        self.processes = [
            SeasonalDemandProcess(
                base_level=r["demand"]["base_level"],
                trend_per_day=r["demand"]["trend_per_day"],
                noise_std=r["demand"]["noise_std"],
            )
            for r in self.cfg["retails"]
        ]
        history_days = self.cfg["history_days"]
        hist_rng = np.random.default_rng(self.cfg["history_seed"])
        self.histories = [
            list(generate_history(p, days=history_days, rng=hist_rng))
            for p in self.processes
        ]
        self.history_offset = history_days

        mean_demands = [float(np.mean(h)) for h in self.histories]
        self.retail_inv = [Inventory(on_hand=7 * m) for m in mean_demands]
        self.wh_inv = []
        for i, wh in enumerate(self.cfg["warehouses"]):
            regional = sum(m for j, m in enumerate(mean_demands)
                           if self.cfg["retails"][j]["home_warehouse"] == i)
            self.wh_inv.append(Inventory(on_hand=wh["initial_days_of_stock"] * regional))

        self.home = [r["home_warehouse"] for r in self.cfg["retails"]]
        self.lane_cost = np.asarray(self.cfg["lanes"]["cost_per_unit"], dtype=float)
        self.lane_lead = np.asarray(self.cfg["lanes"]["lead_days"], dtype=int)

        # time series for the story chart
        self.south_demand_history: list[float] = []
        self.cumulative_stockouts: list[float] = []

    def _receive(self, inv: Inventory) -> None:
        still = []
        for s in inv.pending_shipments:
            s.remaining_days -= 1
            if s.remaining_days <= 0:
                inv.on_hand += s.quantity
                inv.on_order -= s.quantity
            else:
                still.append(s)
        inv.pending_shipments.clear()
        inv.pending_shipments.extend(still)

    def _forecasts(self, days_ahead: int) -> list[np.ndarray]:
        out = []
        for hist in self.histories:
            f = SeasonalNaiveForecaster(season_length=7)
            f.fit(np.asarray(hist, dtype=float))
            out.append(f.predict(days_ahead))
        return out

    def run(self) -> NetworkKPIs:
        cfg = self.cfg
        review = cfg["review_period_days"]
        supplier_lead = cfg["supplier_lead_days"]
        holding_rate = cfg["costs"]["holding_cost_per_unit_day"]
        order_cost = cfg["costs"]["fixed_order_cost"]

        total_demand = 0.0
        total_fulfilled = 0.0
        stockout_days = 0.0
        transport_cost = 0.0
        holding_cost = 0.0
        ordering_cost = 0.0
        cum_stockouts = 0.0

        south_idx = [j for j, r in enumerate(cfg["retails"]) if r["home_warehouse"] == 1]

        for step in range(self.horizon):
            abs_day = self.history_offset + step

            for inv in self.wh_inv + self.retail_inv:
                self._receive(inv)

            day_south_demand = 0.0
            for j, inv in enumerate(self.retail_inv):
                demand = self.processes[j].sample(abs_day, self.rng)
                fulfilled = min(demand, inv.on_hand)
                inv.on_hand -= fulfilled
                total_demand += demand
                total_fulfilled += fulfilled
                if fulfilled < demand:
                    stockout_days += 1
                    cum_stockouts += 1
                self.histories[j].append(demand)
                if j in south_idx:
                    day_south_demand += demand
            self.south_demand_history.append(day_south_demand)
            self.cumulative_stockouts.append(cum_stockouts)

            if step % review == 0:
                # Stock ordered now arrives after supplier_lead days but can
                # only leave the warehouse at a review, so it covers demand up
                # to a full cycle later — the target must span that whole
                # exposure window or every allocation is supply-capped.
                horizon_days = 2 * review + supplier_lead
                forecasts = self._forecasts(horizon_days)
                stds = [float(np.std(h[-28:])) for h in self.histories]

                # Warehouse replenishment: regional order-up-to target,
                # capacity-capped — but network-aware: when a warehouse's
                # regional need exceeds its inbound capacity, the excess
                # spills to peers with spare capacity. Stock gets
                # pre-positioned where the network can still bring it in;
                # whether it ever reaches the short region is then purely
                # the allocator's decision (identical rule for both).
                targets = [
                    sum(
                        float(np.sum(forecasts[j])) + self.SERVICE_Z * stds[j] * (horizon_days ** 0.5)
                        for j in range(len(forecasts)) if self.home[j] == i
                    )
                    for i in range(len(cfg["warehouses"]))
                ]
                caps = [wh["weekly_inbound_capacity"] for wh in cfg["warehouses"]]
                positions = [inv.position for inv in self.wh_inv]
                desired = [max(t - p, 0.0) for t, p in zip(targets, positions)]
                # Echelon accounting: total ordering is capped by the
                # network-level deficit, so overflow already pre-positioned
                # at a peer warehouse counts and ordering can't run away.
                remaining = max(sum(targets) - sum(positions), 0.0)
                orders = []
                for d, c in zip(desired, caps):
                    qty = min(d, c, remaining)
                    orders.append(qty)
                    remaining -= qty
                for i in range(len(orders)):
                    if remaining <= 0:
                        break
                    spill = min(remaining, caps[i] - orders[i])
                    orders[i] += spill
                    remaining -= spill
                for i, qty in enumerate(orders):
                    if qty > 0:
                        inv = self.wh_inv[i]
                        inv.on_order += qty
                        inv.pending_shipments.append(Shipment(quantity=qty, remaining_days=supplier_lead))
                        ordering_cost += order_cost

                # Allocation: retail needs vs warehouse stock -> allocator plan
                alloc_days = review + int(self.lane_lead.max())
                needs = [
                    float(np.sum(forecasts[j][:alloc_days]))
                    + self.SERVICE_Z * stds[j] * (alloc_days ** 0.5)
                    - self.retail_inv[j].position
                    for j in range(len(forecasts))
                ]
                available = [inv.on_hand for inv in self.wh_inv]
                plan = self.allocator.allocate(needs, available, self.lane_cost)
                for i in range(plan.shape[0]):
                    for j in range(plan.shape[1]):
                        qty = plan[i][j]
                        if qty <= 1e-9:
                            continue
                        self.wh_inv[i].on_hand -= qty
                        self.retail_inv[j].on_order += qty
                        self.retail_inv[j].pending_shipments.append(
                            Shipment(quantity=qty, remaining_days=int(self.lane_lead[i][j]))
                        )
                        transport_cost += qty * float(self.lane_cost[i][j])

            day_on_hand = sum(inv.on_hand for inv in self.wh_inv + self.retail_inv)
            holding_cost += day_on_hand * holding_rate
            for inv in self.retail_inv + self.wh_inv:
                inv.snapshot()

        return NetworkKPIs(
            fill_rate=total_fulfilled / total_demand if total_demand > 0 else 1.0,
            stockout_days=stockout_days,
            transport_cost=transport_cost,
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
        )

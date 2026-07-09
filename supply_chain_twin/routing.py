"""Allocation strategies: how warehouse stock gets distributed to retails.

Two implementations of the same protocol, mirroring Phase 2's structure —
a naive baseline that any planner would default to, and the optimizer that
has to beat it on the same demand:

- NearestWarehouseAllocator: every retail is served only by its home
  warehouse; when the warehouse is short, its retails get scaled down
  pro-rata. No pooling — one region's slack can't help another.
- TransportationLPAllocator: a transportation linear program (scipy HiGHS)
  over every warehouse-retail lane, minimizing shipping cost plus a
  shortage penalty. When one region's demand outgrows its warehouse,
  the LP re-routes another region's slack — at the cross-lane price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import linprog


class Allocator(Protocol):
    """Given each retail's need and each warehouse's available stock,
    return a shipment plan: matrix[warehouse][retail] = units to ship."""

    name: str

    def allocate(
        self,
        needs: Sequence[float],
        available: Sequence[float],
        cost_per_unit: Sequence[Sequence[float]],
    ) -> np.ndarray: ...


@dataclass
class NearestWarehouseAllocator:
    """Home-warehouse-only allocation, scaled pro-rata under scarcity."""

    home_warehouse: Sequence[int]  # retail index -> warehouse index
    name: str = "nearest_warehouse"

    def allocate(self, needs, available, cost_per_unit) -> np.ndarray:
        n_wh = len(available)
        n_retail = len(needs)
        plan = np.zeros((n_wh, n_retail))
        for wh in range(n_wh):
            mine = [j for j in range(n_retail) if self.home_warehouse[j] == wh and needs[j] > 0]
            total_need = sum(needs[j] for j in mine)
            if total_need <= 0:
                continue
            scale = min(1.0, available[wh] / total_need)
            for j in mine:
                plan[wh][j] = needs[j] * scale
        return plan


@dataclass
class TransportationLPAllocator:
    """Min-cost transportation LP with soft demand (shortage penalty).

    Variables: x[i][j] = units warehouse i ships to retail j, plus a
    shortage slack s[j] per retail. Minimize sum(cost * x) + penalty * s
    subject to: shipments out of each warehouse <= its available stock,
    and shipments into each retail + its shortage >= its need. The
    penalty prices a unit of unmet demand, so the LP only uses an
    expensive cross-lane when the shortage it prevents costs more.
    """

    shortage_penalty: float = 10.0
    name: str = "transportation_lp"

    def allocate(self, needs, available, cost_per_unit) -> np.ndarray:
        needs = np.maximum(np.asarray(needs, dtype=float), 0.0)
        available = np.asarray(available, dtype=float)
        cost = np.asarray(cost_per_unit, dtype=float)
        n_wh, n_retail = cost.shape
        if needs.sum() <= 0 or available.sum() <= 0:
            return np.zeros((n_wh, n_retail))

        n_x = n_wh * n_retail
        c = np.concatenate([cost.ravel(), np.full(n_retail, self.shortage_penalty)])

        # Supply rows: sum_j x[i][j] <= available[i]
        a_supply = np.zeros((n_wh, n_x + n_retail))
        for i in range(n_wh):
            a_supply[i, i * n_retail:(i + 1) * n_retail] = 1.0
        # Demand rows: -(sum_i x[i][j] + s[j]) <= -needs[j]
        a_demand = np.zeros((n_retail, n_x + n_retail))
        for j in range(n_retail):
            a_demand[j, j::n_retail][:n_wh] = -1.0
            a_demand[j, n_x + j] = -1.0

        result = linprog(
            c,
            A_ub=np.vstack([a_supply, a_demand]),
            b_ub=np.concatenate([available, -needs]),
            bounds=(0, None),
            method="highs",
        )
        if not result.success:  # infeasibility shouldn't occur with soft demand
            raise RuntimeError(f"allocation LP failed: {result.message}")
        return result.x[:n_x].reshape(n_wh, n_retail)

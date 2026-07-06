"""Inventory policies that decide when and how much to reorder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ReorderPolicy(Protocol):
    """Interface every replenishment policy implements — lets Phase 2/3 swap
    in forecast-driven or optimization-driven policies without touching the engine."""

    fixed_order_cost: float

    def order_quantity(self, inventory_position: float) -> float: ...


@dataclass
class ReorderUpToPolicy:
    """(s, S) policy: order up to S whenever inventory position drops to s or below."""

    reorder_point: float
    order_up_to_level: float
    fixed_order_cost: float = 50.0

    def __post_init__(self) -> None:
        if self.order_up_to_level <= self.reorder_point:
            raise ValueError("order_up_to_level must exceed reorder_point")

    def order_quantity(self, inventory_position: float) -> float:
        """Return 0 above the reorder point, otherwise the gap up to S."""
        if inventory_position > self.reorder_point:
            return 0.0
        return self.order_up_to_level - inventory_position

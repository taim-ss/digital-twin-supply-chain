"""Domain entities: supply chain nodes, inventory, and in-transit shipments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional


class NodeType(Enum):
    SUPPLIER = "supplier"
    WAREHOUSE = "warehouse"
    RETAIL = "retail"


@dataclass
class Shipment:
    """A replenishment order in transit."""

    quantity: float
    remaining_days: int


@dataclass
class Inventory:
    """On-hand stock, in-transit pipeline, and a running snapshot history."""

    on_hand: float = 0.0
    on_order: float = 0.0
    history: list[float] = field(default_factory=list)
    pending_shipments: Deque[Shipment] = field(default_factory=deque)

    @property
    def position(self) -> float:
        """On-hand + on-order — what a reorder policy actually evaluates."""
        return self.on_hand + self.on_order

    def snapshot(self) -> None:
        """Record the current on-hand level; called once per simulated day."""
        self.history.append(self.on_hand)


@dataclass
class Node:
    """A physical location in the supply chain."""

    name: str
    node_type: NodeType
    lead_time_days: int = 0
    inventory: Inventory = field(default_factory=Inventory)
    upstream: Optional["Node"] = None

"""Declarative twin definitions (DTDL-inspired).

A scenario document describes the twin — its demand process, network,
costs, and policy defaults — as data rather than code, so every consumer
(the dashboard exporter, notebooks, future phases) runs the *same* twin
instead of each hardcoding its own copy of the parameters. See
`scenarios/baseline.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .demand import SeasonalDemandProcess, generate_history
from .entities import Node, NodeType

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


@dataclass
class Scenario:
    """A parsed twin definition."""

    name: str
    description: str
    demand: dict
    history_days: int
    history_seed: int
    network: dict
    costs: dict
    policy_defaults: dict

    @classmethod
    def load(cls, name: str = "baseline") -> "Scenario":
        path = SCENARIOS_DIR / f"{name}.json"
        raw = json.loads(path.read_text())
        return cls(**raw)

    def build_process(self) -> SeasonalDemandProcess:
        d = self.demand
        return SeasonalDemandProcess(
            base_level=d["base_level"],
            trend_per_day=d["trend_per_day"],
            noise_std=d["noise_std"],
            weekday_multipliers=tuple(d["weekday_multipliers"]),
        )

    def build_history(self) -> np.ndarray:
        rng = np.random.default_rng(self.history_seed)
        return generate_history(self.build_process(), days=self.history_days, rng=rng)

    def build_nodes(self, lead_time_days: int | None = None) -> tuple[Node, Node]:
        """Return (supplier, warehouse). `lead_time_days` overrides the
        document's default — the explorer varies it as a what-if input."""
        supplier = Node(name=self.network["supplier"]["name"], node_type=NodeType.SUPPLIER)
        warehouse = Node(
            name=self.network["warehouse"]["name"],
            node_type=NodeType.WAREHOUSE,
            lead_time_days=lead_time_days if lead_time_days is not None
            else self.network["warehouse"]["lead_time_days"],
        )
        return supplier, warehouse

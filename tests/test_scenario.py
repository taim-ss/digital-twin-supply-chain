"""Tests for the declarative scenario loader."""

from __future__ import annotations

import numpy as np

from supply_chain_twin.entities import NodeType
from supply_chain_twin.scenario import Scenario


class TestScenarioLoader:
    def test_loads_baseline(self):
        s = Scenario.load("baseline")
        assert s.name == "baseline"
        assert s.demand["base_level"] == 40.0
        assert s.network["warehouse"]["lead_time_days"] == 3

    def test_history_is_deterministic(self):
        a = Scenario.load("baseline").build_history()
        b = Scenario.load("baseline").build_history()
        assert len(a) == 180
        assert np.array_equal(a, b)

    def test_nodes_and_lead_time_override(self):
        s = Scenario.load("baseline")
        supplier, warehouse = s.build_nodes()
        assert supplier.node_type == NodeType.SUPPLIER
        assert warehouse.lead_time_days == 3
        _, overridden = s.build_nodes(lead_time_days=7)
        assert overridden.lead_time_days == 7

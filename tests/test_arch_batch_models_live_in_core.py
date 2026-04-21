from __future__ import annotations

import importlib.resources
import pytest

pytestmark = pytest.mark.unit



def test_qt_free_batch_models_live_in_core() -> None:
    widget_source = (
        importlib.resources.files("kindred.gui.widgets")
        .joinpath("batch_initial_conditions_table.py")
        .read_text(encoding="utf-8")
    )
    assert "from kindred.core.batch_initial_conditions import BatchInitialConditionsStore" in widget_source

    sim_controller_source = (
        importlib.resources.files("kindred.gui.controllers")
        .joinpath("simulation_controller.py")
        .read_text(encoding="utf-8")
    )
    assert "from kindred.core.batch_simulation_cache import BatchSimulationCache" in sim_controller_source

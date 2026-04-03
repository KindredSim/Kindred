from __future__ import annotations

import importlib.resources


def test_lazy_import_pressure_is_reduced_to_single_entrypoints() -> None:
    simulation_preparation_source = (
        importlib.resources.files("kindred.core")
        .joinpath("simulation_preparation.py")
        .read_text(encoding="utf-8")
    )
    assert (
        "from kindred.core.simulator.solvers import (\n"
        "    DEFAULT_SOLVER_NAME,\n"
        "    SimulationRequest,\n"
        "    normalize_solver_name,\n"
        "    solve_ode,\n"
        ")"
    ) in simulation_preparation_source
    assert "from kindred.core.simulator.solvers import SimulationRequest" not in simulation_preparation_source
    assert "from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name" not in simulation_preparation_source

    project_controller_source = (
        importlib.resources.files("kindred.gui.controllers")
        .joinpath("project_controller.py")
        .read_text(encoding="utf-8")
    )
    assert "from kindred.gui.widgets.export_dialog import ExportDialog" in project_controller_source
    assert "if self._export_dialog is None:\n            from kindred.gui.widgets.export_dialog import ExportDialog" not in project_controller_source

    worker_source = (
        importlib.resources.files("kindred.gui.fitting")
        .joinpath("worker.py")
        .read_text(encoding="utf-8")
    )
    assert "from kindred.core.simulator.solvers import normalize_solver_name" in worker_source
    assert "from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER" in worker_source
    assert "        from kindred.core.simulator.solvers import normalize_solver_name" not in worker_source

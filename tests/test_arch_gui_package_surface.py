from __future__ import annotations

import importlib.resources
from pathlib import Path


def test_gui_package_keeps_compat_indirection_out_of_gui_namespace() -> None:
    gui_source = importlib.resources.files("kindred.gui").joinpath("__init__.py").read_text(encoding="utf-8")

    assert "from .compat.shims import" not in gui_source
    assert "kindred.compat" not in gui_source

    ports_source = (
        importlib.resources.files("kindred.gui")
        .joinpath("ports.py")
        .read_text(encoding="utf-8")
    )
    assert "from kindred.compat.shims import" not in ports_source
    assert "from kindred.gui.compat.shims import" not in ports_source


def test_evidenced_gui_import_sites_point_to_real_owner_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (
            "kindred/gui/controllers/dataset_manager.py",
            ["from kindred.compat.shims import parse_dsl_to_mechanism, apply_parameter_algebra_to_mechanism"],
            [
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism",
            ],
        ),
        (
            "kindred/gui/main_window.py",
            [
                "from kindred.compat.shims import parse_temperature_schedule",
                "from kindred.compat.shims import parse_dsl_to_mechanism, UnitsModel",
                "from kindred.compat.shims import integrate_ctc as _integrate_ctc",
                "from kindred.compat.shims import (",
                "from kindred.compat.shims import build_sparse_jacobian",
            ],
            [
                "from kindred.core.temperature_dsl import parse_temperature_schedule",
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.units import UnitsModel",
                "from kindred.core.results import integrate_ctc as _integrate_ctc",
                "from kindred.core.ode_builder import build_ode_rhs_from_mechanism",
                "from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism",
                "from kindred.core.simulator.solvers import SimulationRequest, solve_ode",
                "from kindred.core.sparse_jacobian import build_sparse_jacobian",
            ],
        ),
        (
            "kindred/gui/mixins/fitting_mixin.py",
            ["from kindred.compat.shims import parse_dsl_to_mechanism, UnitsModel"],
            [
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.units import UnitsModel",
            ],
        ),
        (
            "kindred/gui/controllers/simulation_controller.py",
            [
                "from kindred.compat.shims import parse_dsl_to_mechanism, UnitsModel",
                "from kindred.compat.shims import parse_dsl_to_mechanism, UnitsModel",
            ],
            [
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.units import UnitsModel",
                "from kindred.core.units import UnitsModel",
            ],
        ),
        (
            "kindred/gui/simulation_worker.py",
            [
                "from kindred.compat.shims import evaluate_algebra_series_for_simulation_with_errors",
                "from kindred.compat.shims import solve_ode",
                "Import lazily through compatibility shims to avoid circular dependencies.",
            ],
            [
                "from kindred.core.algebra.simulation_series import (",
                "from kindred.core.simulator.solvers import solve_ode",
            ],
        ),
        (
            "kindred/gui/main_window_variable_runtime.py",
            ["from kindred.compat.shims import ("],
            [
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.simulator.parameter_algebra import (",
                "from kindred.core.units import UnitsModel",
            ],
        ),
        (
            "kindred/gui/widgets/mechanism_editor.py",
            ["from kindred.compat.shims import parse_dsl_to_mechanism"],
            ["from kindred.core.simulator.dsl import parse_dsl_to_mechanism"],
        ),
        (
            "kindred/gui/widgets/computational_mode_dialog.py",
            ["from kindred.compat.shims import UnitsModel, parse_dsl_to_mechanism"],
            [
                "from kindred.core.simulator.dsl import parse_dsl_to_mechanism",
                "from kindred.core.units import UnitsModel",
            ],
        ),
    ]

    for rel_path, forbidden_snippets, required_snippets in cases:
        source = (repo_root / rel_path).read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in source
        for snippet in required_snippets:
            assert snippet in source

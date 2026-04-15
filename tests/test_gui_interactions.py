import pytest

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.project_schema import PROJECT_DEFAULTS
from kindred.gui.widgets.solver_settings import SolverSettingsDialog


pytestmark = pytest.mark.gui


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_simulation_panel_solver_dropdown_lists_stiff_solvers_only(main_window):
    items = _combo_items(main_window._solver_method_combo)
    assert "ROS3" not in items
    assert "ROS4" not in items
    assert items == ["Radau", "BDF"]


def test_solver_settings_dialog_solver_dropdown_excludes_ros_solvers(qt_app):
    dialog = SolverSettingsDialog()
    items = _combo_items(dialog._combo_solver)
    assert "ROS3" not in items
    assert "ROS4" not in items
    assert items == ["Radau", "BDF"]


def test_invalid_solver_values_normalize_to_bdf(main_window):
    main_window._mechanism_editor.set_slider_solver_value("unknown_solver_name")
    assert main_window._mechanism_editor.slider_solver_value() == "BDF"
    main_window._mechanism_editor.set_slider_solver_value("bdf")
    assert main_window._mechanism_editor.slider_solver_value() == "BDF"


def test_solver_settings_dialog_normalizes_invalid_solver_values(qt_app):
    dialog = SolverSettingsDialog()
    dialog._combo_solver.setCurrentText("Radau")
    dialog._combo_slider_preview_solver.setCurrentText("Radau")

    dialog.set_settings({"solver": "unknown_solver_name", "slider_preview_solver": " bdf "})

    assert dialog._combo_solver.currentText() == "BDF"
    assert dialog._combo_slider_preview_solver.currentText() == "BDF"


def test_solver_settings_dialog_invalid_tolerances_reset_to_defaults(qt_app):
    dialog = SolverSettingsDialog()
    dialog._spin_rtol.setValue(1e-4)
    dialog._spin_atol.setValue(1e-9)

    dialog.set_settings({"solver": "unknown_solver_name", "rtol": "bad", "atol": "bad"})

    assert dialog._combo_solver.currentText() == "BDF"
    assert dialog._spin_rtol.value() == pytest.approx(1e-6)
    assert dialog._spin_atol.value() == pytest.approx(1e-12)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_solver_settings_dialog_nonfinite_and_nonpositive_tolerances_reset_to_defaults(qt_app, bad_value):
    dialog = SolverSettingsDialog()
    dialog._spin_rtol.setValue(1e-4)
    dialog._spin_atol.setValue(1e-9)

    dialog.set_settings({"rtol": bad_value, "atol": bad_value})

    assert dialog._spin_rtol.value() == pytest.approx(1e-6)
    assert dialog._spin_atol.value() == pytest.approx(1e-12)


def test_solver_settings_dialog_string_boolean_payloads_are_parsed_explicitly(qt_app):
    dialog = SolverSettingsDialog()

    dialog.set_settings(
        {
            "use_sparse_jacobian": "False",
            "wegscheider_cyclicity_enabled": "",
            "limit_blas_threads_per_worker": "0",
        }
    )

    assert dialog._sparse_checkbox.isChecked() is False
    assert dialog._wegscheider_checkbox.isChecked() is False
    assert dialog._limit_blas_checkbox.isChecked() is False


def test_solver_settings_dialog_unknown_boolean_strings_stay_disabled(qt_app):
    dialog = SolverSettingsDialog()

    dialog.set_settings(
        {
            "use_sparse_jacobian": "garbage",
            "wegscheider_cyclicity_enabled": "garbage",
            "limit_blas_threads_per_worker": "garbage",
        }
    )

    assert dialog._sparse_checkbox.isChecked() is False
    assert dialog._wegscheider_checkbox.isChecked() is False
    assert dialog._limit_blas_checkbox.isChecked() is False


def test_solver_settings_dialog_factory_defaults_match_schema_and_cache_defaults(qt_app):
    from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

    dialog = SolverSettingsDialog()

    assert dialog._sparse_checkbox.isChecked() is bool(PROJECT_DEFAULTS["use_sparse_jacobian"])
    assert dialog._wegscheider_checkbox.isChecked() is bool(PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"])
    assert dialog._max_parallel_workers_spin.value() == int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
    assert dialog._max_parallel_workers_spin.maximum() == int(MAX_PARALLEL_WORKERS_CEILING)
    assert dialog._limit_blas_checkbox.isChecked() is bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"])
    assert dialog._spin_result_cache_cap.value() == int(BatchSimulationCache.result_cache_cap)
    assert dialog._spin_preview_cache_cap.value() == int(BatchSimulationCache.preview_cache_cap)


def test_solver_settings_dialog_caps_worker_count_at_shared_ceiling(qt_app):
    from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
    from kindred.gui.widgets import solver_settings

    dialog = SolverSettingsDialog()

    dialog.set_settings({"max_parallel_batch_workers": 400})

    assert solver_settings._MAX_PARALLEL_WORKERS_SPIN_MAX == int(MAX_PARALLEL_WORKERS_CEILING)
    assert dialog._max_parallel_workers_spin.maximum() == int(MAX_PARALLEL_WORKERS_CEILING)
    assert dialog._max_parallel_workers_spin.value() == int(MAX_PARALLEL_WORKERS_CEILING)

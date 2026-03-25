import pytest

from kindred.gui.widgets.solver_settings import SolverSettingsDialog


pytestmark = pytest.mark.gui


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_simulation_panel_solver_dropdown_lists_stiff_solvers_only(main_window):
    items = _combo_items(main_window._solver_method_combo)
    assert "ROS3" not in items
    assert "ROS4" not in items
    assert items == ["LSODA", "Radau", "BDF"]


def test_solver_settings_dialog_solver_dropdown_excludes_ros_solvers(qt_app):
    dialog = SolverSettingsDialog()
    items = _combo_items(dialog._combo_solver)
    assert "ROS3" not in items
    assert "ROS4" not in items
    assert items == ["LSODA", "Radau", "BDF"]

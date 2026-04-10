from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

pytestmark = [pytest.mark.gui]


def _make_parameters_tab(*, integration_defaults=("LSODA", 1e-6, 1e-12)):
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable

    entries = [{"id": "ds1", "label": "DS 1"}]
    species = ["A", "B"]
    species_table = UnifiedSpeciesTable(
        dataset_entries=list(entries),
        mechanism_species=list(species),
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: [str(e["id"]) for e in entries],
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda _ds_id: 1.0,
        persist_dataset_weight_callback=lambda _ds_id, _weight: None,
        dataset_manager_getter=lambda: None,
        worker_running_getter=lambda: False,
    )
    tab = ParametersIcsTab(
        parameter_state=[],
        initial_parameter_snapshot=[],
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={},
        mechanism_species=list(species),
        dataset_entries=list(entries),
        prepared_param_names=[],
        selected_dataset_ids_getter=lambda: [str(e["id"]) for e in entries],
        dataset_entries_getter=lambda: list(entries),
        worker_running_getter=lambda: False,
        dataset_manager_getter=lambda: None,
        reactions_text_getter=lambda: "",
        integration_defaults=integration_defaults,
        config_defaults={},
        ic_panel=species_table,
    )
    return tab, species_table


def test_grid_plot_view_legend_only_names_model_curves(qtbot):
    from kindred.gui.widgets.grid_plot_view import GridPlotView

    view = GridPlotView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()

    t = np.linspace(0.0, 1.0, 5)
    view.set_datasets(
        [
            {
                "name": "DS1",
                "data_x": t,
                "data_y": np.linspace(1.0, 0.4, t.size),
                "model_x": t,
                "model_y": np.linspace(1.0, 0.35, t.size),
                "model_series": {
                    "A": np.linspace(1.0, 0.35, t.size),
                    "B": np.linspace(0.7, 0.15, t.size),
                },
                "all_species": {
                    "A": np.linspace(1.0, 0.4, t.size),
                    "B": np.linspace(0.8, 0.2, t.size),
                },
                "current_species": "A",
            }
        ]
    )
    QtWidgets.QApplication.processEvents()

    species_list = view._species_list
    assert species_list.count() == 2
    for row in range(species_list.count()):
        species_list.item(row).setSelected(True)
    QtWidgets.QApplication.processEvents()

    assert len(view._plot_series_items) == 1
    series_map = view._plot_series_items[0]
    assert series_map["A::data"].opts.get("name") is None
    assert series_map["B::data"].opts.get("name") is None
    assert series_map["A::model"].opts.get("name") == "A"
    assert series_map["B::model"].opts.get("name") == "B"


def test_parameters_integration_controls_are_visible_without_expanding_section(qtbot):
    tab, species_table = _make_parameters_tab()
    qtbot.addWidget(tab)
    tab.resize(900, 700)
    tab.show()
    QtWidgets.QApplication.processEvents()

    try:
        solver_combo = tab.findChild(QtWidgets.QComboBox, "global_fit_integration_solver")
        rtol_edit = tab.findChild(QtWidgets.QLineEdit, "global_fit_integration_rtol")
        atol_edit = tab.findChild(QtWidgets.QLineEdit, "global_fit_integration_atol")

        assert solver_combo is not None
        assert rtol_edit is not None
        assert atol_edit is not None
        assert solver_combo.isVisible()
        assert rtol_edit.isVisible()
        assert atol_edit.isVisible()
    finally:
        species_table.close()
        tab.close()
        QtWidgets.QApplication.processEvents()


def test_fitting_default_solver_constant_and_parameters_tab_default(qtbot):
    from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER

    assert FITTING_DEFAULT_SOLVER == "LSODA"

    tab, species_table = _make_parameters_tab(
        integration_defaults=(FITTING_DEFAULT_SOLVER, 1e-6, 1e-12)
    )
    qtbot.addWidget(tab)
    tab.show()
    QtWidgets.QApplication.processEvents()

    try:
        assert tab._integration_solver_combo.currentText() == FITTING_DEFAULT_SOLVER
        collected = tab.collect_integration_settings()
        assert collected == (FITTING_DEFAULT_SOLVER, 1e-6, 1e-12)
    finally:
        species_table.close()
        tab.close()
        QtWidgets.QApplication.processEvents()


def test_global_fit_worker_defaults_to_fitting_solver(qt_app):
    from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER
    from kindred.gui.fitting.worker import GlobalFitWorker

    t = np.array([0.0, 1.0], dtype=float)
    worker = GlobalFitWorker(
        [{"id": "ds1", "t": t.copy(), "y": np.array([1.0, 0.5], dtype=float), "species": "A"}],
        {"k1": 1.0},
        fit_evaluator=lambda _params: {"t": t.copy(), "species": {"A": np.array([1.0, 0.5], dtype=float)}},
    )

    try:
        assert worker._solver == FITTING_DEFAULT_SOLVER
    finally:
        worker.deleteLater()
        qt_app.processEvents()


def test_fitting_window_solver_combo_defaults_to_lsoda_even_with_radau_prepared_meta(qt_app):
    """Combo must show LSODA regardless of what the main simulation was prepared with."""
    from dataclasses import dataclass
    from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER
    from kindred.gui.fitting.window import FittingWindow

    @dataclass
    class _FakeMeta:
        version: int = 1
        mechanism_text_sha256: str = ""
        mechanism_text_len: int = 0
        param_names: list = None
        t_end: float = 10.0
        num_points: int = 100
        temperature_K: float = 298.15
        solver_requested: str = "Radau"
        solver_normalized: str = "Radau"
        solver_warning: str = ""
        rtol: float = 1e-6
        atol: float = 1e-12
        use_sparse_jacobian: bool = False
        wegscheider_cyclicity_enabled: bool = False
        initial_prefix: str = "init_"

        def __post_init__(self):
            if self.param_names is None:
                self.param_names = []

    t = np.arange(0, 10, dtype=float)

    def sim_func(_params):
        return {"t": t.copy(), "species": {"A": t.copy()}}

    sim_func._kindred_prepared_simulation_meta = _FakeMeta()

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.5, "min": 0.01, "max": 1.0}],
        dataset_entries=[{
            "id": "ds1", "label": "Dataset 1",
            "t": t.copy(), "species_data": {"A": t.copy()},
            "selected_species": ["A"], "weight": 1.0, "include": True,
        }],
        simulation_func=sim_func,
        mechanism_species=["A"],
    )
    try:
        qt_app.processEvents()
        combo = window._params_ics_tab._integration_solver_combo
        assert combo.currentText() == FITTING_DEFAULT_SOLVER, (
            f"Expected {FITTING_DEFAULT_SOLVER}, got {combo.currentText()}"
        )
    finally:
        window.close()
        qt_app.processEvents()

import numpy as np
import pytest

from PySide6 import QtWidgets

from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def _make_window() -> FittingWindow:
    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, t.size)
    model = np.linspace(1.0, 0.4, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": model.copy()}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([y.copy()]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )


def _make_two_dataset_window() -> FittingWindow:
    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
            {
                "id": "ds2",
                "label": "ds2",
                "t": t.copy(),
                "species_data": {"B": y_b.copy()},
                "selected_species": ["B"],
                "weight": 0.5,
                "include": True,
            },
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}},
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]},
            {"id": "ds2", "t": t.copy(), "y": np.vstack([y_b.copy()]), "species": ["B"]},
        ],
        dataset_weights={"ds1": 1.0, "ds2": 0.5},
    )


def _tab_index(tab_bar: QtWidgets.QTabBar, title: str) -> int:
    titles = [tab_bar.tabText(i) for i in range(tab_bar.count())]
    return titles.index(title)


def test_global_fit_right_panel_tabs_follow_workflow_and_rehome_surfaces(qt_app):
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        top_tabs = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        content_stack = window.findChild(QtWidgets.QStackedWidget, "global_fit_current_tab_stack")
        results_subtabs = window.findChild(QtWidgets.QTabBar, "global_fit_results_subtabs")
        footer = window.findChild(QtWidgets.QWidget, "global_fit_footer")

        assert top_tabs is not None
        assert content_stack is not None
        assert results_subtabs is not None
        assert footer is not None
        assert window.findChild(QtWidgets.QSplitter, "global_fit_shell_splitter") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_right_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_context_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_detail_tabs") is None

        assert window._tabs is top_tabs

        titles = [top_tabs.tabText(i) for i in range(top_tabs.count())]
        assert titles == ["Data and Targets", "Parameters", "Results"]

        data_targets_widget = content_stack.widget(_tab_index(top_tabs, "Data and Targets"))
        params_widget = content_stack.widget(_tab_index(top_tabs, "Parameters"))

        sampling_panel = window.findChild(QtWidgets.QWidget, "global_fit_sampling_panel")
        species_table_widget = window.findChild(QtWidgets.QTableWidget, "global_fit_unified_species_table")
        weight_mode = window.findChild(QtWidgets.QComboBox, "global_fit_weight_mode_combo")
        weight_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_dataset_weight_edit")
        results_summary_button = window.findChild(QtWidgets.QPushButton, "global_fit_results_summary_footer_button")
        run_block = window.findChild(QtWidgets.QLabel, "global_fit_run_block_reason_label")

        assert sampling_panel is not None
        assert species_table_widget is not None
        assert weight_mode is not None
        assert weight_edit is not None
        assert results_summary_button is not None
        assert run_block is not None

        assert window._data_tab._dataset_table.columnCount() == 3
        assert [window._data_tab._dataset_table.horizontalHeaderItem(i).text() for i in range(window._data_tab._dataset_table.columnCount())] == [
            "Use",
            "Dataset",
            "Species",
        ]

        # Data and Targets tab contains unified list, sampling panel, species table
        assert data_targets_widget.isAncestorOf(window._data_targets_tab.unified_list)
        assert data_targets_widget.isAncestorOf(sampling_panel)
        assert data_targets_widget.isAncestorOf(species_table_widget)
        assert data_targets_widget.isAncestorOf(window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group"))
        assert data_targets_widget.isAncestorOf(weight_mode)
        assert data_targets_widget.isAncestorOf(weight_edit)
        assert window._data_targets_tab.unified_list._list.count() == 1

        # Parameters tab contains param table and method combo, but not species table
        assert params_widget.isAncestorOf(window._params_ics_tab._param_table)
        assert params_widget.isAncestorOf(window._params_ics_tab._method_combo)
        assert not params_widget.isAncestorOf(species_table_widget)

        assert footer.isAncestorOf(results_summary_button)
        assert footer.isAncestorOf(run_block)

        # Results subtab host is populated eagerly even before the Results page is active.
        assert [results_subtabs.tabText(i) for i in range(results_subtabs.count())] == ["ds1", "All Datasets"]

        # Switch to Results and verify the current page only; no shell splitter remains.
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert content_stack.currentWidget() is window._run_results_tab
    finally:
        window.close()


def test_unified_list_drives_targets_dataset_selection(qt_app):
    window = _make_two_dataset_window()
    try:
        unified_list = window._data_targets_tab.unified_list
        unified_list.select_dataset("ds2")
        qt_app.processEvents()
        assert window._species_table._current_dataset_id == "ds2"

        unified_list.select_dataset("ds1")
        qt_app.processEvents()
        assert window._species_table._current_dataset_id == "ds1"

        unified_list.select_dataset("ds2")
        qt_app.processEvents()
        assert window._species_table._current_dataset_id == "ds2"
    finally:
        window.close()


def test_targets_dataset_switch_flushes_visible_weight_edit(qt_app):
    window = _make_two_dataset_window()
    try:
        # Targets panel is always visible in unified layout — no subtab switch needed.
        qt_app.processEvents()

        unified_list = window._data_targets_tab.unified_list
        weight_mode = window.findChild(QtWidgets.QComboBox, "global_fit_weight_mode_combo")
        weight_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_dataset_weight_edit")
        assert weight_mode is not None
        assert weight_edit is not None

        unified_list.select_dataset("ds1")
        qt_app.processEvents()
        weight_mode.setCurrentIndex(1)
        weight_edit.setText("0.75")
        qt_app.processEvents()

        unified_list.select_dataset("ds2")
        qt_app.processEvents()

        assert window._dataset_entries[0]["weight"] == pytest.approx(0.75)

        unified_list.select_dataset("ds1")
        qt_app.processEvents()
        assert weight_edit.text() == "0.75"
    finally:
        window.close()


def test_shell_uses_single_top_navigation_host_and_survives_narrow_resize(qt_app):
    window = _make_window()
    try:
        window.show()
        window.resize(980, 700)
        qt_app.processEvents()

        top_tabs = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        content_stack = window.findChild(QtWidgets.QStackedWidget, "global_fit_current_tab_stack")

        assert top_tabs is not None
        assert content_stack is not None
        assert window.findChild(QtWidgets.QSplitter, "global_fit_shell_splitter") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_right_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_context_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_detail_tabs") is None

        assert top_tabs.count() == 3
        assert content_stack.width() >= 260
        assert content_stack.parentWidget() is window

        # Switch to Results: stack still occupies the shell directly.
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert content_stack.width() >= 260
    finally:
        window.close()


def test_results_page_switches_without_subset_panel_logic(qt_app):
    """Regression: Results page is stack-driven and no subset shell remains."""
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        top_tabs = window._tabs

        assert not hasattr(window, "_subset_widget")

        # Switch to each page — the stack updates without any left-panel show/hide behavior.
        for title in ("Data and Targets", "Parameters"):
            top_tabs.setCurrentIndex(_tab_index(top_tabs, title))
            qt_app.processEvents()
            assert window._current_tab_stack.currentIndex() == _tab_index(top_tabs, title)

        # Switch to Results — the Results page becomes current.
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._current_tab_stack.currentWidget() is window._run_results_tab
    finally:
        window.close()


def test_unified_layout_structure_and_subset_visibility(qt_app):
    """Regression: 3 top-level tabs, DataTargetsTab has unified master/detail layout, subset visibility correct."""
    from kindred.gui.fitting.data_tab import DataTab
    from kindred.gui.fitting.data_targets_tab import DataTargetsTab
    from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable

    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        top_tabs = window._tabs

        # Exactly 3 top-level tabs with correct names
        assert top_tabs.count() == 3
        titles = [top_tabs.tabText(i) for i in range(top_tabs.count())]
        assert titles == ["Data and Targets", "Parameters", "Results"]

        # Page 0 is DataTargetsTab
        dt_tab = window._data_targets_tab
        assert isinstance(dt_tab, DataTargetsTab)

        # DataTargetsTab contains a QSplitter, NOT a QTabWidget for subtabs.
        assert dt_tab.findChild(QtWidgets.QTabWidget) is None
        assert dt_tab.findChild(QtWidgets.QSplitter) is not None

        # UnifiedDatasetList is present
        assert isinstance(dt_tab.unified_list, UnifiedDatasetList)
        assert dt_tab.isAncestorOf(dt_tab.unified_list)

        # Species table and sampling panel are children of the DataTargetsTab.
        # DataTab itself is hidden (signals/state owner only, not a layout child).
        assert isinstance(dt_tab.data_tab, DataTab)
        assert dt_tab.isAncestorOf(dt_tab.data_tab._sampling_panel_widget)
        assert dt_tab.isAncestorOf(dt_tab.species_table)
        assert isinstance(dt_tab.species_table, UnifiedSpeciesTable)

        # Unified list is populated with 1 dataset
        assert dt_tab.unified_list.selected_dataset_id() == "ds1"

        # Results page is a normal stack page; no separate subset widget remains.
        assert not hasattr(window, "_subset_widget")
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Parameters"))
        qt_app.processEvents()
        assert window._current_tab_stack.currentWidget() is window._params_ics_tab

        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._current_tab_stack.currentWidget() is window._run_results_tab

        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Data and Targets"))
        qt_app.processEvents()
        assert window._current_tab_stack.currentWidget() is window._data_targets_tab

        # Species table is loaded for a dataset (proves on_tab_activated ran during construction)
        assert window._species_table._current_dataset_id is not None
    finally:
        window.close()

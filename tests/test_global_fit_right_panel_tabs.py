import numpy as np
import pytest

from PySide6 import QtCore, QtWidgets

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


def _targets_dataset_ids(dataset_list: QtWidgets.QListWidget) -> list[str]:
    ids: list[str] = []
    for row in range(dataset_list.count()):
        item = dataset_list.item(row)
        assert item is not None
        ids.append(str(item.data(QtCore.Qt.UserRole) or ""))
    return ids


def _tab_index(tab_bar: QtWidgets.QTabBar, title: str) -> int:
    titles = [tab_bar.tabText(i) for i in range(tab_bar.count())]
    return titles.index(title)


def test_global_fit_right_panel_tabs_follow_workflow_and_rehome_surfaces(qt_app):
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        top_tabs = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        shell_splitter = window.findChild(QtWidgets.QSplitter, "global_fit_shell_splitter")
        content_stack = window.findChild(QtWidgets.QStackedWidget, "global_fit_current_tab_stack")
        footer = window.findChild(QtWidgets.QWidget, "global_fit_footer")

        assert top_tabs is not None
        assert shell_splitter is not None
        assert content_stack is not None
        assert footer is not None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_right_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_context_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_detail_tabs") is None

        assert window._tabs is top_tabs

        titles = [top_tabs.tabText(i) for i in range(top_tabs.count())]
        assert titles == ["Data and Targets", "Parameters", "Results"]

        data_targets_widget = content_stack.widget(_tab_index(top_tabs, "Data and Targets"))
        params_widget = content_stack.widget(_tab_index(top_tabs, "Parameters"))

        sampling_panel = window.findChild(QtWidgets.QWidget, "global_fit_sampling_panel")
        targets_list = window.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        weight_mode = window.findChild(QtWidgets.QComboBox, "global_fit_weight_mode_combo")
        weight_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_dataset_weight_edit")
        ic_table = window.findChild(QtWidgets.QTableWidget, "global_fit_initial_conditions_table")
        results_summary_button = window.findChild(QtWidgets.QPushButton, "global_fit_results_summary_footer_button")
        run_block = window.findChild(QtWidgets.QLabel, "global_fit_run_block_reason_label")

        assert sampling_panel is not None
        assert targets_list is not None
        assert weight_mode is not None
        assert weight_edit is not None
        assert ic_table is not None
        assert results_summary_button is not None
        assert run_block is not None

        assert window._data_tab._dataset_table.columnCount() == 3
        assert [window._data_tab._dataset_table.horizontalHeaderItem(i).text() for i in range(window._data_tab._dataset_table.columnCount())] == [
            "Use",
            "Dataset",
            "Species",
        ]

        # Data and Targets tab contains Data, Targets, and IC panels
        assert data_targets_widget.isAncestorOf(window._data_tab._dataset_table)
        assert data_targets_widget.isAncestorOf(sampling_panel)
        assert data_targets_widget.isAncestorOf(targets_list)
        assert data_targets_widget.isAncestorOf(window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel"))
        assert data_targets_widget.isAncestorOf(weight_mode)
        assert data_targets_widget.isAncestorOf(weight_edit)
        assert data_targets_widget.isAncestorOf(ic_table)
        assert _targets_dataset_ids(targets_list) == ["ds1"]

        # Parameters tab contains param table and method combo, but not IC
        assert params_widget.isAncestorOf(window._params_ics_tab._param_table)
        assert params_widget.isAncestorOf(window._params_ics_tab._method_combo)
        assert not params_widget.isAncestorOf(ic_table)

        assert footer.isAncestorOf(results_summary_button)
        assert footer.isAncestorOf(run_block)

        # On Data and Targets tab (config): subset_widget is hidden
        assert not window._subset_widget.isVisible()

        # Switch to Results to verify splitter sizing
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._subset_widget.isVisible()
        shell_sizes = shell_splitter.sizes()
        assert len(shell_sizes) == 2
        assert shell_sizes[0] > shell_sizes[1]
    finally:
        window.close()


def test_unified_list_drives_targets_dataset_selection(qt_app):
    window = _make_two_dataset_window()
    try:
        # Unified list drives all panels — select ds2 in the unified list.
        unified_list = window._data_targets_tab.unified_list
        unified_list.select_dataset("ds2")
        qt_app.processEvents()

        dataset_list = window.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        assert dataset_list is not None
        current = dataset_list.currentItem()
        assert current is not None
        assert str(current.data(QtCore.Qt.UserRole) or "") == "ds2"

        # Select ds1 via unified list.
        unified_list.select_dataset("ds1")
        qt_app.processEvents()
        current = dataset_list.currentItem()
        assert current is not None
        assert str(current.data(QtCore.Qt.UserRole) or "") == "ds1"

        # Switch back to ds2 via unified list — targets follows.
        unified_list.select_dataset("ds2")
        qt_app.processEvents()
        current = dataset_list.currentItem()
        assert current is not None
        assert str(current.data(QtCore.Qt.UserRole) or "") == "ds2"
    finally:
        window.close()


def test_targets_dataset_switch_flushes_visible_weight_edit(qt_app):
    window = _make_two_dataset_window()
    try:
        # Targets panel is always visible in unified layout — no subtab switch needed.
        qt_app.processEvents()

        dataset_list = window.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        weight_mode = window.findChild(QtWidgets.QComboBox, "global_fit_weight_mode_combo")
        weight_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_dataset_weight_edit")
        assert dataset_list is not None
        assert weight_mode is not None
        assert weight_edit is not None

        dataset_list.setCurrentRow(0)
        qt_app.processEvents()
        weight_mode.setCurrentIndex(1)
        weight_edit.setText("0.75")
        qt_app.processEvents()

        dataset_list.setCurrentRow(1)
        qt_app.processEvents()

        assert window._dataset_entries[0]["weight"] == pytest.approx(0.75)

        dataset_list.setCurrentRow(0)
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
        shell_splitter = window.findChild(QtWidgets.QSplitter, "global_fit_shell_splitter")
        content_stack = window.findChild(QtWidgets.QStackedWidget, "global_fit_current_tab_stack")

        assert top_tabs is not None
        assert shell_splitter is not None
        assert content_stack is not None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_right_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_context_tabs") is None
        assert window.findChild(QtWidgets.QTabWidget, "global_fit_detail_tabs") is None

        assert top_tabs.count() == 3
        # On Data and Targets tab (default): subset hidden, content_stack gets full width
        assert not window._subset_widget.isVisible()
        assert content_stack.width() >= 260
        assert window._subset_widget.parentWidget() is shell_splitter
        assert content_stack.parentWidget() is shell_splitter

        # Switch to Results: subset visible with expected width
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._subset_widget.isVisible()
        assert window._subset_widget.width() >= 300
        assert content_stack.width() >= 260
    finally:
        window.close()


def test_subset_widget_visible_only_on_results_tab(qt_app):
    """Regression: left plot panel hidden on config tabs, visible on Results."""
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        top_tabs = window._tabs

        # On construction (Data and Targets tab, index 0): subset_widget hidden
        assert not window._subset_widget.isVisible()

        # Switch to each config tab — still hidden
        for title in ("Data and Targets", "Parameters"):
            top_tabs.setCurrentIndex(_tab_index(top_tabs, title))
            qt_app.processEvents()
            assert not window._subset_widget.isVisible(), f"subset visible on {title}"

        # Switch to Results — visible
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._subset_widget.isVisible()

        # Switch back to Data and Targets — hidden again
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Data and Targets"))
        qt_app.processEvents()
        assert not window._subset_widget.isVisible()
    finally:
        window.close()


def test_unified_layout_structure_and_subset_visibility(qt_app):
    """Regression: 3 top-level tabs, DataTargetsTab has unified master/detail layout, subset visibility correct."""
    from kindred.gui.fitting.data_tab import DataTab
    from kindred.gui.fitting.data_targets_tab import DataTargetsTab
    from kindred.gui.fitting.initial_conditions_panel import InitialConditionsPanel
    from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList

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

        # All three panels are children of the DataTargetsTab
        assert isinstance(dt_tab.data_tab, DataTab)
        assert dt_tab.isAncestorOf(dt_tab.data_tab)
        assert dt_tab.isAncestorOf(dt_tab.targets_weights_tab)
        assert dt_tab.isAncestorOf(dt_tab.ic_panel)
        assert isinstance(dt_tab.ic_panel, InitialConditionsPanel)

        # Unified list is populated with 1 dataset
        assert dt_tab.unified_list.selected_dataset_id() == "ds1"

        # Subset widget hidden on Data and Targets
        assert not window._subset_widget.isVisible()

        # Subset widget hidden on Parameters
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Parameters"))
        qt_app.processEvents()
        assert not window._subset_widget.isVisible()

        # Subset widget visible on Results
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Results"))
        qt_app.processEvents()
        assert window._subset_widget.isVisible()

        # Back to Data and Targets — hidden again
        top_tabs.setCurrentIndex(_tab_index(top_tabs, "Data and Targets"))
        qt_app.processEvents()
        assert not window._subset_widget.isVisible()

        # Targets dataset list is populated (proves on_tab_activated ran during construction)
        dataset_list = window.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        assert dataset_list is not None
        assert dataset_list.count() >= 1
    finally:
        window.close()

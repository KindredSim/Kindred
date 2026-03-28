from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.plot_config import is_pyqtgraph_available
from kindred.gui.widgets.axis_toolbar import AxisToolbar
from kindred.gui.widgets.dataset_plot_panel import DatasetPlotPanel
from kindred.gui.widgets.grid_plot_view import GridPlotView
from kindred.gui.widgets.plot_tabs import PlotTabsWidget

pytestmark = [pytest.mark.gui]


def test_grid_plot_view_species_list_has_minimum_vertical_policy_and_no_hardcoded_max_height(qt_app):
    view = GridPlotView()
    try:
        species_list = view._species_list
        policy = species_list.sizePolicy()
        assert policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
        assert policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Minimum

        # Qt's unconstrained maximum height (QWIDGETSIZE_MAX) is 16,777,215.
        assert species_list.maximumHeight() == 16_777_215
    finally:
        view.close()
        QtWidgets.QApplication.processEvents()


def test_plot_tabs_widget_tabs_expands_both_directions(qt_app):
    widget = PlotTabsWidget()
    try:
        policy = widget._tabs.sizePolicy()
        assert policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
        assert policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    finally:
        widget.close()
        QtWidgets.QApplication.processEvents()


def test_horizontal_axis_toolbar_defaults_to_compact_popup_y_selector(qt_app):
    toolbar = AxisToolbar(orientation="horizontal")
    try:
        toolbar.set_x_candidates(["t", "A"], default="t")
        toolbar.set_y_candidates([("A", True), ("B", False)])
        toolbar.show()
        qt_app.processEvents()

        manual_range_row = getattr(toolbar, "_manual_range_row", None)
        assert manual_range_row is not None
        assert not toolbar._y_list.isVisible()
        assert not manual_range_row.isVisible()
        assert toolbar.sizeHint().height() <= 72
    finally:
        toolbar.close()
        QtWidgets.QApplication.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_main_plot_workspace_prefers_horizontal_controls_and_tabbed_secondary_surfaces(qt_app):
    widget = PlotTabsWidget()
    try:
        widget.resize(1400, 900)
        widget.show()
        qt_app.processEvents()

        plot = widget._main_plot
        assert getattr(plot._toolbar, "_orientation", None) == "horizontal"

        details_tabs = getattr(plot, "_details_tabs", None)
        assert isinstance(details_tabs, QtWidgets.QTabWidget)
        assert [details_tabs.tabText(i) for i in range(details_tabs.count())] == [
            "Statistics",
            "Parameters",
        ]

        splitter = plot._main_splitter
        assert splitter.orientation() == QtCore.Qt.Orientation.Vertical
        sizes = splitter.sizes()
        assert len(sizes) == 2
        assert sizes[0] > sizes[1]
        total = sum(sizes)
        assert plot._control_strip.height() <= 96
        assert sizes[1] <= max(160, int(total * 0.2))
    finally:
        widget.close()
        QtWidgets.QApplication.processEvents()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_dataset_plot_panel_uses_compact_top_species_strip_instead_of_sidebar(qt_app):
    panel = DatasetPlotPanel(dataset_name="demo")
    try:
        t = [0.0, 1.0, 2.0]
        a = [1.0, 0.5, 0.2]
        b = [0.4, 0.3, 0.1]
        panel.set_data(t, a, xlabel="Time", ylabel="A", all_species={"A": a, "B": b})
        panel.resize(1200, 800)
        panel.show()
        qt_app.processEvents()

        species_strip = getattr(panel, "_species_strip", None)
        species_scroll = getattr(panel, "_species_scroll", None)
        assert species_strip is not None
        assert species_scroll is not None
        assert isinstance(panel._species_checkboxes_layout, QtWidgets.QHBoxLayout)
        assert panel.layout().indexOf(species_strip) < panel.layout().indexOf(panel._plot_panel)
        assert species_scroll.maximumHeight() <= 96
    finally:
        panel.close()
        QtWidgets.QApplication.processEvents()

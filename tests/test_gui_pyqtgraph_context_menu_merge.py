import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.plot_tabs import PlotTabsWidget
from kindred.gui.widgets.pyqtgraph_plot_panel_impl import (
    PYQTGRAPH_AVAILABLE,
    PyQtGraphPlotPanel,
)


pytestmark = pytest.mark.gui


class _DummyClipboard:
    def __init__(self) -> None:
        self.last_text = ""

    def setText(self, text: str, *_args, **_kwargs) -> None:  # noqa: N802 - Qt-style
        self.last_text = str(text)


def _capture_context_menu(monkeypatch):
    captured_menus = []

    def _fake_exec(self, *_args, **_kwargs):
        captured_menus.append(self)
        return None

    monkeypatch.setattr(QtWidgets.QMenu, "exec_", _fake_exec)
    return captured_menus


def _find_action(actions, text: str):
    for action in actions:
        if action.text() == text:
            return action
    raise AssertionError(f"Missing action {text!r}")


def _split_tsv(text: str) -> list[list[str]]:
    lines = [line for line in str(text).splitlines() if line]
    return [line.split("\t") for line in lines]


def _find_header_index(header: list[str], *, prefix: str, contains: list[str]) -> int:
    for idx, cell in enumerate(header):
        if not cell.startswith(prefix):
            continue
        if all(token in cell for token in contains):
            return idx
    raise AssertionError(f"Missing header with prefix={prefix!r} contains={contains!r}")


def _numeric_column(rows: list[list[str]], idx: int) -> np.ndarray:
    values = [row[idx] for row in rows if idx < len(row) and row[idx] != ""]
    return np.asarray([float(value) for value in values], dtype=float)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_pyqtgraph_native_menus_disabled_and_custom_actions_present(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    assert panel._plot_item.menuEnabled() is False
    assert panel._plot_item.getViewBox().menuEnabled() is False

    scene = panel._plot_item.scene()
    export_calls = {"n": 0}

    def _fake_show_export_dialog(_self):
        export_calls["n"] += 1

    monkeypatch.setattr(type(scene), "showExportDialog", _fake_show_export_dialog)

    captured_menus = _capture_context_menu(monkeypatch)

    panel._show_context_menu(QtCore.QPoint(0, 0))

    assert captured_menus, "Expected _show_context_menu to call QMenu.exec_"
    menu = captured_menus[0]

    def _action_index(text: str) -> int:
        for idx, action in enumerate(menu.actions()):
            if action.text() == text:
                return idx
        raise AssertionError(f"Missing action {text!r}")

    export_idx = _action_index("Export Plot...")
    mouse_idx = _action_index("Mouse Mode")
    reset_idx = _action_index("Reset View")

    assert export_idx < reset_idx
    assert mouse_idx < reset_idx
    assert menu.actions()[reset_idx - 1].isSeparator()

    mouse_action = menu.actions()[mouse_idx]
    mouse_menu = mouse_action.menu()
    assert mouse_menu is not None

    pan_action = next(a for a in mouse_menu.actions() if a.text() == "Pan (3-Button)")
    rect_action = next(a for a in mouse_menu.actions() if a.text() == "Rect Zoom (1-Button)")

    import pyqtgraph as pg

    viewbox = panel._plot_item.getViewBox()
    pan_action.trigger()
    assert viewbox.state.get("mouseMode") == pg.ViewBox.PanMode
    rect_action.trigger()
    assert viewbox.state.get("mouseMode") == pg.ViewBox.RectMode

    menu.actions()[export_idx].trigger()
    assert export_calls["n"] == 1


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_plot_context_menu_toggle_hides_and_restores_canonical_reference_lines(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_canonical_ghost_toggle_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    captured_menus = _capture_context_menu(monkeypatch)

    panel._show_context_menu(QtCore.QPoint(0, 0))
    menu = captured_menus.pop()
    toggle_action = next(
        action for action in menu.actions() if action.text() == "Show Canonical Reference Lines"
    )
    assert toggle_action.isCheckable()
    assert toggle_action.isChecked() is True
    assert toggle_action.isEnabled() is False

    t = np.array([0.0, 1.0, 2.0], dtype=float)
    panel.set_data(
        t,
        {"A": np.array([1.0, 0.5, 0.25], dtype=float)},
        label="set1",
        overlays=[
            {
                "label": "set2",
                "set_id": "set2",
                "t": t,
                "series": {"A": np.array([0.75, 0.4, 0.2], dtype=float)},
            },
            {
                "label": "set2",
                "set_id": "set2",
                "curve_role": "canonical_ghost",
                "t": t,
                "series": {"A": np.array([0.9, 0.6, 0.3], dtype=float)},
            },
        ],
    )
    QtWidgets.QApplication.processEvents()

    non_ghost_key = panel._format_species_set_label("A", "set2")
    ghost_key = f"{non_ghost_key} [canonical]"
    overlays = list(getattr(panel, "_simulation_overlays", []) or [])
    assert any(str(entry.get("curve_role") or "") == "canonical_ghost" for entry in overlays)
    assert non_ghost_key in panel._plot_items
    assert ghost_key in panel._plot_items

    panel._show_context_menu(QtCore.QPoint(0, 0))
    menu = captured_menus.pop()
    toggle_action = next(
        action for action in menu.actions() if action.text() == "Show Canonical Reference Lines"
    )
    assert toggle_action.isEnabled() is True
    toggle_action.trigger()
    QtWidgets.QApplication.processEvents()

    overlays = list(getattr(panel, "_simulation_overlays", []) or [])
    assert any(str(entry.get("curve_role") or "") == "canonical_ghost" for entry in overlays)
    assert non_ghost_key in panel._plot_items
    assert ghost_key not in panel._plot_items

    panel._show_context_menu(QtCore.QPoint(0, 0))
    menu = captured_menus.pop()
    toggle_action = next(
        action for action in menu.actions() if action.text() == "Show Canonical Reference Lines"
    )
    assert toggle_action.isChecked() is False
    toggle_action.trigger()
    QtWidgets.QApplication.processEvents()

    overlays = list(getattr(panel, "_simulation_overlays", []) or [])
    assert any(str(entry.get("curve_role") or "") == "canonical_ghost" for entry in overlays)
    assert non_ghost_key in panel._plot_items
    assert ghost_key in panel._plot_items


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_plot_axis_inversion_actions_are_scoped_to_simulation_plot(qtbot, monkeypatch):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    dataset_panel = widget.add_dataset_tab("dataset-1")
    QtWidgets.QApplication.processEvents()

    captured_menus = _capture_context_menu(monkeypatch)

    widget._main_plot._show_context_menu(QtCore.QPoint(0, 0))
    main_menu = captured_menus.pop()
    axis_direction_action = _find_action(main_menu.actions(), "Axis Direction")
    axis_direction_menu = axis_direction_action.menu()
    assert axis_direction_menu is not None
    invert_x_action = _find_action(axis_direction_menu.actions(), "Invert X-Axis")
    invert_y_action = _find_action(axis_direction_menu.actions(), "Invert Y-Axis")
    assert invert_x_action.isCheckable()
    assert invert_y_action.isCheckable()

    dataset_panel._plot_panel._show_context_menu(QtCore.QPoint(0, 0))
    dataset_menu = captured_menus.pop()
    assert all(action.text() != "Axis Direction" for action in dataset_menu.actions())


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_plot_context_menu_includes_copy_visible_data_and_dataset_plot_does_not(qtbot, monkeypatch):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    dataset_panel = widget.add_dataset_tab("dataset-1")
    QtWidgets.QApplication.processEvents()

    captured_menus = _capture_context_menu(monkeypatch)

    widget._main_plot._show_context_menu(QtCore.QPoint(0, 0))
    main_menu = captured_menus.pop()
    _find_action(main_menu.actions(), "Copy Visible Data")

    dataset_panel._plot_panel._show_context_menu(QtCore.QPoint(0, 0))
    dataset_menu = captured_menus.pop()
    assert all(action.text() != "Copy Visible Data" for action in dataset_menu.actions())


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_writes_structural_tsv_for_visible_primary_overlays_and_dataset_markers(
    qtbot, monkeypatch
):
    panel = PyQtGraphPlotPanel(
        enable_copy_visible_data_action=True,
        enable_canonical_ghost_toggle_action=True,
    )
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("C", True))

    t_primary = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    panel.set_data(
        t_primary,
        {
            "A": np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            "B": np.array([0.10, 0.20, 0.30, 0.40], dtype=float),
            "C": np.array([5.0, 6.0, 7.0, 8.0], dtype=float),
            "D": np.array([1.0, 1.5, 2.0, 2.5], dtype=float),
        },
        label="set1",
        overlays=[
            {
                "label": "set2",
                "set_id": "set2",
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "series": {
                    "A": np.array([101.0, 102.0, 103.0], dtype=float),
                    "B": np.array([0.15, 0.25, 0.35], dtype=float),
                    "C": np.array([201.0, 202.0, 203.0], dtype=float),
                },
            },
            {
                "label": "set2",
                "set_id": "set2",
                "curve_role": "canonical_ghost",
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "series": {
                    "A": np.array([901.0, 902.0, 903.0], dtype=float),
                    "B": np.array([0.16, 0.26, 0.36], dtype=float),
                    "C": np.array([951.0, 952.0, 953.0], dtype=float),
                },
            },
        ],
    )
    panel.set_selected_series(["A", "C"])
    panel._on_x_axis_changed("B")
    panel._add_secondary_y_axis()
    panel.set_selected_series(["A"])

    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0], dtype=float),
                "species": {
                    "A": np.array([1001.0, 1002.0], dtype=float),
                    "B": np.array([0.12, 0.22], dtype=float),
                    "C": np.array([1101.0, 1102.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A"}
    panel._update_plot()
    QtWidgets.QApplication.processEvents()

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows, "Expected clipboard TSV output"
    header = rows[0]
    body = rows[1:]
    assert len(body) == 4

    primary_cols = [idx for idx, cell in enumerate(header) if cell.startswith("set1::")]
    overlay_cols = [idx for idx, cell in enumerate(header) if cell.startswith("set2::")]
    dataset_cols = [idx for idx, cell in enumerate(header) if cell.startswith("ds1::")]
    assert primary_cols
    assert overlay_cols
    assert dataset_cols
    assert header[0] != ""
    assert header[-1] != ""
    assert max(primary_cols) + 2 == min(overlay_cols)
    assert header[max(primary_cols) + 1] == ""
    assert max(overlay_cols) + 2 == min(dataset_cols)
    assert header[max(overlay_cols) + 1] == ""

    primary_headers = [header[idx] for idx in primary_cols]
    assert any("Time" in cell for cell in primary_headers)
    assert any("[B]" in cell for cell in primary_headers)
    assert any("C" in cell and "[right axis]" in cell for cell in primary_headers)
    assert any("A" in cell for cell in primary_headers)

    overlay_headers = [header[idx] for idx in overlay_cols]
    assert any("A" in cell for cell in overlay_headers)
    assert all("C" not in cell for cell in overlay_headers)

    dataset_headers = [header[idx] for idx in dataset_cols]
    assert any("A" in cell for cell in dataset_headers)
    assert all("C" not in cell for cell in dataset_headers)

    assert "901.0" not in clipboard.last_text
    assert "902.0" not in clipboard.last_text
    assert all("canonical" not in cell.lower() for cell in header)
    assert all("[ref]" not in cell for cell in header)

    overlay_x_idx = next(idx for idx in overlay_cols if "[B]" in header[idx])
    overlay_y_idx = next(idx for idx in overlay_cols if "A" in header[idx] and "[B]" not in header[idx])
    dataset_x_idx = next(idx for idx in dataset_cols if "[B]" in header[idx])
    dataset_y_idx = next(idx for idx in dataset_cols if "A" in header[idx] and "[B]" not in header[idx])

    assert body[3][overlay_x_idx] == ""
    assert body[3][overlay_y_idx] == ""
    assert body[2][dataset_x_idx] == ""
    assert body[2][dataset_y_idx] == ""


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_includes_secondary_axis_only_primary_trace_without_warning(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    warning_calls = []
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("C", True),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append((args, kwargs)),
    )

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "C": np.array([5.0, 6.0, 7.0], dtype=float),
        },
        label="set1",
    )
    panel._add_secondary_y_axis()
    panel.set_selected_series([])
    QtWidgets.QApplication.processEvents()

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    primary_headers = [cell for cell in rows[0] if cell.startswith("set1::")]
    assert any("Time" in cell for cell in primary_headers)
    assert any("C" in cell and "[right axis]" in cell for cell in primary_headers)
    assert all("A" not in cell for cell in primary_headers)

@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_uses_synchronized_secondary_axis_trace_after_x_axis_change(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("C", True))

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([10.0, 20.0, 30.0], dtype=float),
            "C": np.array([5.0, 6.0, 7.0], dtype=float),
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    panel._add_secondary_y_axis()
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    secondary_item = panel._secondary_y_items["C"]
    plotted_x, plotted_y = secondary_item.getData()
    np.testing.assert_allclose(np.asarray(plotted_x, dtype=float), np.array([10.0, 20.0, 30.0], dtype=float))
    np.testing.assert_allclose(np.asarray(plotted_y, dtype=float), np.array([5.0, 6.0, 7.0], dtype=float))

    panel._series["C"] = np.array([105.0, 106.0, 107.0], dtype=float)

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    body = rows[1:]
    x_idx = _find_header_index(header, prefix="set1::", contains=["[B]"])
    y_idx = _find_header_index(header, prefix="set1::", contains=["C", "[right axis]"])

    np.testing.assert_allclose(_numeric_column(body, x_idx), np.asarray(plotted_x, dtype=float))
    np.testing.assert_allclose(_numeric_column(body, y_idx), np.asarray(plotted_y, dtype=float))
    assert "105.0" not in clipboard.last_text


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_uses_synchronized_secondary_axis_trace_after_set_data_refresh(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("C", True))

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([10.0, 20.0, 30.0], dtype=float),
            "C": np.array([5.0, 6.0, 7.0], dtype=float),
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    panel._add_secondary_y_axis()

    panel.set_data(
        np.array([4.0, 5.0, 6.0], dtype=float),
        {
            "A": np.array([11.0, 12.0, 13.0], dtype=float),
            "B": np.array([40.0, 50.0, 60.0], dtype=float),
            "C": np.array([15.0, 16.0, 17.0], dtype=float),
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    QtWidgets.QApplication.processEvents()

    secondary_item = panel._secondary_y_items["C"]
    plotted_x, plotted_y = secondary_item.getData()
    np.testing.assert_allclose(np.asarray(plotted_x, dtype=float), np.array([4.0, 5.0, 6.0], dtype=float))
    np.testing.assert_allclose(np.asarray(plotted_y, dtype=float), np.array([15.0, 16.0, 17.0], dtype=float))

    panel._series["C"] = np.array([215.0, 216.0, 217.0], dtype=float)

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    body = rows[1:]
    time_idx = _find_header_index(header, prefix="set1::", contains=["Time"])
    y_idx = _find_header_index(header, prefix="set1::", contains=["C", "[right axis]"])

    np.testing.assert_allclose(_numeric_column(body, time_idx), np.asarray(plotted_x, dtype=float))
    np.testing.assert_allclose(_numeric_column(body, y_idx), np.asarray(plotted_y, dtype=float))
    assert "215.0" not in clipboard.last_text


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_respects_coarse_sampling_for_secondary_axis_after_x_axis_change(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("C", True))

    t = np.linspace(0.0, 10.0, 2000)
    x_species = np.linspace(100.0, 300.0, 2000)
    secondary_series = np.linspace(5.0, 15.0, 2000)
    panel.set_data(
        t,
        {
            "A": np.sin(t),
            "B": x_species,
            "C": secondary_series,
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    panel._on_toolbar_option_requested("sampling", "coarse")
    panel._add_secondary_y_axis()
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    expected_idx = np.unique(np.linspace(0, t.shape[0] - 1, num=panel._sampling_target, dtype=int))
    secondary_item = panel._secondary_y_items["C"]
    plotted_x, plotted_y = secondary_item.getData()
    np.testing.assert_allclose(np.asarray(plotted_x, dtype=float), x_species[expected_idx])
    np.testing.assert_allclose(np.asarray(plotted_y, dtype=float), secondary_series[expected_idx])
    assert len(plotted_x) <= panel._sampling_target

    panel._series["C"] = secondary_series + 1000.0

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    body = rows[1:]
    time_idx = _find_header_index(header, prefix="set1::", contains=["Time"])
    x_idx = _find_header_index(header, prefix="set1::", contains=["[B]"])
    y_idx = _find_header_index(header, prefix="set1::", contains=["C", "[right axis]"])

    assert len(body) == len(expected_idx)
    np.testing.assert_allclose(_numeric_column(body, time_idx), t[expected_idx])
    np.testing.assert_allclose(_numeric_column(body, x_idx), np.asarray(plotted_x, dtype=float))
    np.testing.assert_allclose(_numeric_column(body, y_idx), np.asarray(plotted_y, dtype=float))
    assert "1005.0" not in clipboard.last_text


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_respects_coarse_sampling_for_secondary_axis_after_set_data_refresh(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("C", True))

    t_initial = np.linspace(0.0, 10.0, 2000)
    panel.set_data(
        t_initial,
        {
            "A": np.sin(t_initial),
            "C": np.linspace(5.0, 15.0, 2000),
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    panel._on_toolbar_option_requested("sampling", "coarse")
    panel._add_secondary_y_axis()

    t_refresh = np.linspace(20.0, 30.0, 2000)
    refreshed_secondary = np.linspace(25.0, 35.0, 2000)
    panel.set_data(
        t_refresh,
        {
            "A": np.cos(t_refresh),
            "C": refreshed_secondary,
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    QtWidgets.QApplication.processEvents()

    expected_idx = np.unique(np.linspace(0, t_refresh.shape[0] - 1, num=panel._sampling_target, dtype=int))
    secondary_item = panel._secondary_y_items["C"]
    plotted_x, plotted_y = secondary_item.getData()
    np.testing.assert_allclose(np.asarray(plotted_x, dtype=float), t_refresh[expected_idx])
    np.testing.assert_allclose(np.asarray(plotted_y, dtype=float), refreshed_secondary[expected_idx])
    assert len(plotted_x) <= panel._sampling_target

    panel._series["C"] = refreshed_secondary + 2000.0

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    body = rows[1:]
    time_idx = _find_header_index(header, prefix="set1::", contains=["Time"])
    y_idx = _find_header_index(header, prefix="set1::", contains=["C", "[right axis]"])

    assert len(body) == len(expected_idx)
    np.testing.assert_allclose(_numeric_column(body, time_idx), np.asarray(plotted_x, dtype=float))
    np.testing.assert_allclose(_numeric_column(body, y_idx), np.asarray(plotted_y, dtype=float))
    assert "2025.0" not in clipboard.last_text


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_respects_coarse_sampling_for_overlay_blocks(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    t_primary = np.linspace(0.0, 10.0, 2000)
    t_overlay = np.linspace(0.0, 5.0, 1500)
    t_dataset = np.linspace(0.0, 3.0, 1600)
    panel.set_data(
        t_primary,
        {
            "A": np.sin(t_primary),
            "B": np.cos(t_primary),
        },
        label="set1",
        overlays=[
            {
                "label": "set2",
                "t": t_overlay,
                "series": {
                    "A": np.linspace(10.0, 20.0, 1500),
                    "B": np.linspace(30.0, 40.0, 1500),
                },
            }
        ],
    )
    panel.set_selected_series(["A"])
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": t_dataset,
                "species": {
                    "A": np.linspace(100.0, 200.0, 1600),
                    "B": np.linspace(300.0, 400.0, 1600),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A"}
    panel._on_toolbar_option_requested("sampling", "coarse")
    panel._update_plot()
    QtWidgets.QApplication.processEvents()

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    body = rows[1:]
    overlay_x_idx = _find_header_index(header, prefix="set2::", contains=["Time"])
    overlay_y_idx = _find_header_index(header, prefix="set2::", contains=["A"])
    dataset_x_idx = _find_header_index(header, prefix="ds1::", contains=["Time"])
    dataset_y_idx = _find_header_index(header, prefix="ds1::", contains=["A"])

    overlay_expected_idx = np.unique(np.linspace(0, t_overlay.shape[0] - 1, num=panel._sampling_target, dtype=int))
    dataset_expected_idx = np.unique(np.linspace(0, t_dataset.shape[0] - 1, num=panel._sampling_target, dtype=int))

    assert len(body) == panel._sampling_target
    np.testing.assert_allclose(_numeric_column(body, overlay_x_idx), t_overlay[overlay_expected_idx])
    np.testing.assert_allclose(
        _numeric_column(body, overlay_y_idx),
        np.linspace(10.0, 20.0, 1500)[overlay_expected_idx],
    )
    np.testing.assert_allclose(_numeric_column(body, dataset_x_idx), t_dataset[dataset_expected_idx])
    np.testing.assert_allclose(
        _numeric_column(body, dataset_y_idx),
        np.linspace(100.0, 200.0, 1600)[dataset_expected_idx],
    )


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_visible_data_keeps_species_x_plot_usable_when_time_mismatches(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    warning_calls = []
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append((args, kwargs)),
    )

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0], dtype=float),
        },
        label="set1",
    )
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    primary_key = panel._format_species_set_label("A", "set1")
    assert primary_key in panel._plot_items

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    primary_headers = [cell for cell in rows[0] if cell.startswith("set1::")]
    assert all("Time" not in cell for cell in primary_headers)
    assert any("[B]" in cell for cell in primary_headers)
    assert any(cell.endswith("::A") or cell == "set1::A" for cell in primary_headers)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_species_x_partial_render_state_filters_incompatible_y_series_across_render_export_hover_and_copy(
    qtbot, monkeypatch
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    warning_calls = []
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warning_calls.append((args, kwargs)),
    )

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0], dtype=float),
            "C": np.array([7.0, 8.0, 9.0, 10.0, 11.0], dtype=float),
        },
        label="set1",
    )
    panel.set_selected_series(["A", "C"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    rendered_keys = set(panel._plot_items.keys())
    assert panel._format_species_set_label("A", "set1") in rendered_keys
    assert panel._format_species_set_label("C", "set1") not in rendered_keys

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert any(header == "A" for header in axis_header)
    assert "C" not in axis_header

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header == "A" for header in all_header)
    assert "C" not in all_header

    x_data = np.asarray(panel._get_x_data()[0], dtype=float)
    hit = panel._find_nearest_data_point(2.0, 20.0, x_data)
    assert hit is not None
    assert hit.label == "A"

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    primary_headers = [cell for cell in rows[0] if cell.startswith("set1::")]
    assert any(cell.endswith("::A") or cell == "set1::A" for cell in primary_headers)
    assert all(not cell.endswith("::C") and "C [right axis]" not in cell for cell in primary_headers)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_plot_axis_inversion_toggles_render_direction_and_restores(qt_app):
    panel = PyQtGraphPlotPanel(enable_axis_inversion_actions=True)
    try:
        panel.show()
        panel.resize(400, 300)
        panel.set_data(
            np.array([2.0, 6.0, 10.0], dtype=float),
            {"A": np.array([1.0, 5.0, 9.0], dtype=float)},
        )
        panel._toolbar.set_auto_range(False)
        panel._plot_item.setRange(xRange=(2.0, 10.0), yRange=(1.0, 9.0), padding=0.0)
        qt_app.processEvents()

        def _scene_positions():
            viewbox = panel._plot_item.getViewBox()
            x_low = viewbox.mapViewToScene(QtCore.QPointF(2.0, 5.0)).x()
            x_high = viewbox.mapViewToScene(QtCore.QPointF(10.0, 5.0)).x()
            y_low = viewbox.mapViewToScene(QtCore.QPointF(6.0, 1.0)).y()
            y_high = viewbox.mapViewToScene(QtCore.QPointF(6.0, 9.0)).y()
            return x_low, x_high, y_low, y_high

        viewbox = panel._plot_item.getViewBox()
        initial_x_low, initial_x_high, initial_y_low, initial_y_high = _scene_positions()
        assert viewbox.state.get("xInverted") is False
        assert viewbox.state.get("yInverted") is False
        assert initial_x_low < initial_x_high
        assert initial_y_low > initial_y_high
        x_data, y_data = panel._plot_items["A"].getData()
        np.testing.assert_allclose(np.asarray(x_data, dtype=float), np.array([2.0, 6.0, 10.0], dtype=float))
        np.testing.assert_allclose(np.asarray(y_data, dtype=float), np.array([1.0, 5.0, 9.0], dtype=float))

        panel._toggle_invert_x()
        qt_app.processEvents()
        assert viewbox.state.get("xInverted") is True
        x_low, x_high, y_low, y_high = _scene_positions()
        assert x_low > x_high
        assert y_low > y_high
        x_data, y_data = panel._plot_items["A"].getData()
        np.testing.assert_allclose(np.asarray(x_data, dtype=float), np.array([2.0, 6.0, 10.0], dtype=float))
        np.testing.assert_allclose(np.asarray(y_data, dtype=float), np.array([1.0, 5.0, 9.0], dtype=float))

        panel._toggle_invert_y()
        qt_app.processEvents()
        assert viewbox.state.get("xInverted") is True
        assert viewbox.state.get("yInverted") is True
        x_low, x_high, y_low, y_high = _scene_positions()
        assert x_low > x_high
        assert y_low < y_high

        panel._toggle_invert_x()
        panel._toggle_invert_y()
        qt_app.processEvents()
        assert viewbox.state.get("xInverted") is False
        assert viewbox.state.get("yInverted") is False
        restored_x_low, restored_x_high, restored_y_low, restored_y_high = _scene_positions()
        assert restored_x_low < restored_x_high
        assert restored_y_low > restored_y_high
    finally:
        panel.deleteLater()
        qt_app.processEvents()


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_secondary_axis_inherits_inversion_across_remove_readd_lifecycle(qt_app, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_axis_inversion_actions=True)
    try:
        panel.show()
        panel.resize(400, 300)
        panel.set_data(
            np.array([2.0, 6.0, 10.0], dtype=float),
            {
                "A": np.array([1.0, 5.0, 9.0], dtype=float),
                "B": np.array([10.0, 50.0, 90.0], dtype=float),
            },
        )
        panel._toolbar.set_auto_range(False)
        panel._plot_item.setRange(xRange=(2.0, 10.0), yRange=(1.0, 9.0), padding=0.0)
        qt_app.processEvents()

        def _scene_positions(viewbox, *, y_low: float, y_high: float):
            x_low = viewbox.mapViewToScene(QtCore.QPointF(2.0, 5.0)).x()
            x_high = viewbox.mapViewToScene(QtCore.QPointF(10.0, 5.0)).x()
            y_scene_low = viewbox.mapViewToScene(QtCore.QPointF(6.0, y_low)).y()
            y_scene_high = viewbox.mapViewToScene(QtCore.QPointF(6.0, y_high)).y()
            return x_low, x_high, y_scene_low, y_scene_high

        monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *args, **kwargs: ("B", True))

        panel._toggle_invert_x()
        panel._toggle_invert_y()
        qt_app.processEvents()

        panel._add_secondary_y_axis()
        qt_app.processEvents()

        primary_viewbox = panel._plot_item.getViewBox()
        secondary_viewbox = panel._secondary_y_axis
        assert secondary_viewbox is not None
        secondary_viewbox.setRange(xRange=(2.0, 10.0), yRange=(10.0, 90.0), padding=0.0)
        qt_app.processEvents()

        assert primary_viewbox.state.get("xInverted") is True
        assert primary_viewbox.state.get("yInverted") is True
        assert secondary_viewbox.state.get("xInverted") is True
        assert secondary_viewbox.state.get("yInverted") is True
        primary_x_low, primary_x_high, primary_y_low, primary_y_high = _scene_positions(
            primary_viewbox,
            y_low=1.0,
            y_high=9.0,
        )
        secondary_x_low, secondary_x_high, secondary_y_low, secondary_y_high = _scene_positions(
            secondary_viewbox,
            y_low=10.0,
            y_high=90.0,
        )
        assert primary_x_low > primary_x_high
        assert primary_y_low < primary_y_high
        assert secondary_x_low > secondary_x_high
        assert secondary_y_low < secondary_y_high

        panel._remove_secondary_y_axis()
        qt_app.processEvents()
        assert panel._secondary_y_axis is None
        assert panel._secondary_y_items == {}

        panel.resize(420, 300)
        qt_app.processEvents()

        panel._add_secondary_y_axis()
        qt_app.processEvents()
        secondary_viewbox = panel._secondary_y_axis
        assert secondary_viewbox is not None
        secondary_viewbox.setRange(xRange=(2.0, 10.0), yRange=(10.0, 90.0), padding=0.0)
        qt_app.processEvents()

        assert secondary_viewbox.state.get("xInverted") is True
        assert secondary_viewbox.state.get("yInverted") is True
        secondary_x_low, secondary_x_high, secondary_y_low, secondary_y_high = _scene_positions(
            secondary_viewbox,
            y_low=10.0,
            y_high=90.0,
        )
        assert secondary_x_low > secondary_x_high
        assert secondary_y_low < secondary_y_high

        panel._toggle_invert_x()
        panel._toggle_invert_y()
        qt_app.processEvents()

        assert primary_viewbox.state.get("xInverted") is False
        assert primary_viewbox.state.get("yInverted") is False
        assert secondary_viewbox.state.get("xInverted") is False
        assert secondary_viewbox.state.get("yInverted") is False
        primary_x_low, primary_x_high, primary_y_low, primary_y_high = _scene_positions(
            primary_viewbox,
            y_low=1.0,
            y_high=9.0,
        )
        secondary_x_low, secondary_x_high, secondary_y_low, secondary_y_high = _scene_positions(
            secondary_viewbox,
            y_low=10.0,
            y_high=90.0,
        )
        assert primary_x_low < primary_x_high
        assert primary_y_low > primary_y_high
        assert secondary_x_low < secondary_x_high
        assert secondary_y_low > secondary_y_high
    finally:
        panel.deleteLater()
        qt_app.processEvents()

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

import kindred.gui.widgets.pyqtgraph_plot_panel_impl as plot_panel_impl
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


def _axis_label_text(panel: PyQtGraphPlotPanel, axis_name: str) -> str:
    axis = panel._plot_item.getAxis(axis_name)
    return str(getattr(axis, "labelText", "") or "")


def _legend_texts(panel: PyQtGraphPlotPanel) -> list[str]:
    legend = getattr(panel, "_legend", None)
    if legend is None:
        return []
    return [str(label.text) for _sample, label in list(legend.items)]


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
def test_clear_resets_simulation_metadata_to_init_values(qtbot):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.asarray([0.0, 1.0], dtype=float),
        {"A": np.asarray([1.0, 2.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "dup",
                "popup_label": "dup (row 2)",
                "set_id": "set2",
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([3.0, 4.0], dtype=float)},
            }
        ],
    )
    panel._simulation_set_popup_label = "dup (row 1)"

    assert panel._simulation_set_label == "dup"
    assert panel._simulation_set_popup_label == "dup (row 1)"
    assert list(panel._simulation_overlays or [])

    panel.clear()

    assert panel._simulation_set_label is None
    assert panel._simulation_set_popup_label is None
    assert panel._simulation_overlays == []


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
def test_main_plot_context_menu_includes_copy_actions_and_dataset_plot_does_not(qtbot, monkeypatch):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    dataset_panel = widget.add_dataset_tab("dataset-1")
    QtWidgets.QApplication.processEvents()
    widget._main_plot.set_copy_all_export_plan_provider(lambda: None)

    captured_menus = _capture_context_menu(monkeypatch)

    widget._main_plot._show_context_menu(QtCore.QPoint(0, 0))
    main_menu = captured_menus.pop()
    _find_action(main_menu.actions(), "Copy All")
    _find_action(main_menu.actions(), "Copy Visible Data")

    dataset_panel._plot_panel._show_context_menu(QtCore.QPoint(0, 0))
    dataset_menu = captured_menus.pop()
    assert all(action.text() != "Copy All" for action in dataset_menu.actions())
    assert all(action.text() != "Copy Visible Data" for action in dataset_menu.actions())


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_and_dataset_plot_context_menus_do_not_expose_hover_toggle(qtbot, monkeypatch):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    dataset_panel = widget.add_dataset_tab("dataset-1")
    QtWidgets.QApplication.processEvents()

    captured_menus = _capture_context_menu(monkeypatch)

    widget._main_plot._show_context_menu(QtCore.QPoint(0, 0))
    main_menu = captured_menus.pop()
    assert all(action.text() != "Enable Hover/Crosshair" for action in main_menu.actions())

    dataset_panel._plot_panel._show_context_menu(QtCore.QPoint(0, 0))
    dataset_menu = captured_menus.pop()
    assert all(action.text() != "Enable Hover/Crosshair" for action in dataset_menu.actions())


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_main_and_dataset_plot_context_menus_do_not_expose_secondary_axis(qtbot, monkeypatch):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    dataset_panel = widget.add_dataset_tab("dataset-1")
    QtWidgets.QApplication.processEvents()

    captured_menus = _capture_context_menu(monkeypatch)

    widget._main_plot._show_context_menu(QtCore.QPoint(0, 0))
    main_menu = captured_menus.pop()
    assert all(action.text() != "Secondary Y-Axis" for action in main_menu.actions())

    dataset_panel._plot_panel._show_context_menu(QtCore.QPoint(0, 0))
    dataset_menu = captured_menus.pop()
    assert all(action.text() != "Secondary Y-Axis" for action in dataset_menu.actions())


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_plot_panel_backend_does_not_carry_secondary_axis_or_hover_helpers_or_state(qtbot):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    assert not hasattr(panel, "_add_secondary_y_axis")
    assert not hasattr(panel, "_remove_secondary_y_axis")
    assert not hasattr(panel, "_secondary_y_axis")
    assert not hasattr(panel, "_secondary_y_species")
    assert not hasattr(panel, "_secondary_y_items")
    assert not hasattr(panel, "_secondary_y_overlay_items")
    assert not hasattr(panel, "_secondary_y_dataset_overlay_items")
    assert not hasattr(panel, "_enable_hover_crosshair_toggle_action")
    assert not hasattr(panel, "_hover_crosshair_enabled")
    assert not hasattr(panel, "_crosshair_v")
    assert not hasattr(panel, "_crosshair_h")
    assert not hasattr(panel, "_tooltip_text")
    assert not hasattr(panel, "_clear_hover_state")
    assert not hasattr(panel, "_ensure_hover_visual_items")
    assert not hasattr(panel, "_set_hover_crosshair_enabled")
    assert not hasattr(panel, "_find_nearest_hover_hit")
    assert not hasattr(panel, "_find_nearest_data_point")
    assert not hasattr(panel, "_on_mouse_moved")


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
    assert any("A" in cell for cell in primary_headers)
    assert all("C" not in cell for cell in primary_headers)

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
def test_visible_export_and_copy_visible_data_use_popup_labels_for_duplicate_simulation_names(
    qtbot, monkeypatch
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {"A": np.array([1.0, 2.0, 3.0], dtype=float)},
        label="dup",
        overlays=[
            {
                "label": "dup",
                "popup_label": "dup (row 2)",
                "set_id": "set2",
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "series": {"A": np.array([10.0, 11.0, 12.0], dtype=float)},
            }
        ],
    )
    panel._simulation_set_popup_label = "dup (row 1)"
    panel.set_selected_series(["A"])
    QtWidgets.QApplication.processEvents()

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert "Time (s)" in axis_header
    assert "A" in axis_header
    assert any(cell.startswith("dup (row 2)::") for cell in axis_header)
    assert all(not cell.startswith("dup::") for cell in axis_header)

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert "Time (s)" in all_header
    assert "A" in all_header
    assert any(cell.startswith("dup (row 2)::") for cell in all_header)
    assert all(not cell.startswith("dup::") for cell in all_header)

    panel._copy_visible_data()

    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    assert any(cell.startswith("dup (row 1)::") for cell in header)
    assert any(cell.startswith("dup (row 2)::") for cell in header)
    assert all(not cell.startswith("dup::") for cell in header)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_all_writes_structural_tsv_for_shown_blocks_deduped_overlays_and_dataset_markers(
    qtbot, monkeypatch
):
    panel = PyQtGraphPlotPanel(enable_canonical_ghost_toggle_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    panel._sampling_target = 3

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([0.1, 0.2, 0.3], dtype=float),
            "C": np.array([5.0, 6.0, 7.0], dtype=float),
        },
        label="focused",
        overlays=[
            {
                "label": "set2",
                "set_id": "set2",
                "t": np.linspace(0.0, 9.0, 10),
                "series": {
                    "A": np.linspace(50.0, 59.0, 10),
                    "B": np.linspace(0.5, 1.4, 10),
                    "C": np.linspace(150.0, 159.0, 10),
                },
            },
            {
                "label": "set3",
                "set_id": "set3",
                "t": np.linspace(0.0, 9.0, 10),
                "series": {
                    "A": np.linspace(250.0, 259.0, 10),
                    "B": np.linspace(2.5, 3.4, 10),
                    "C": np.linspace(350.0, 359.0, 10),
                },
            },
            {
                "label": "set2",
                "set_id": "set2",
                "curve_role": "canonical_ghost",
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "series": {
                    "A": np.array([901.0, 902.0, 903.0], dtype=float),
                    "B": np.array([9.1, 9.2, 9.3], dtype=float),
                    "C": np.array([951.0, 952.0, 953.0], dtype=float),
                },
            },
        ],
    )
    panel.set_selected_series(["A", "C"])
    panel._on_x_axis_changed("B")
    panel.set_selected_series(["A"])
    panel._on_toolbar_option_requested("sampling", "coarse")

    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.linspace(0.0, 5.0, 6),
                "species": {
                    "A": np.linspace(1001.0, 1006.0, 6),
                    "B": np.linspace(10.1, 10.6, 6),
                    "C": np.linspace(1101.0, 1106.0, 6),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A"}
    panel._update_plot()
    QtWidgets.QApplication.processEvents()

    plot_contract = plot_panel_impl
    panel.set_copy_all_export_plan_provider(
        lambda: plot_contract.CopyAllExportPlan(
            shown_blocks=[
                plot_contract.CopyAllShownBlock(
                    set_id="set1",
                    label="set1",
                    t=np.linspace(0.0, 5.0, 6),
                    series={
                        "A": np.linspace(11.0, 16.0, 6),
                        "B": np.linspace(0.11, 0.66, 6),
                        "C": np.linspace(21.0, 26.0, 6),
                    },
                ),
                plot_contract.CopyAllShownBlock(
                    set_id="set2",
                    label="set2",
                    t=np.linspace(10.0, 15.0, 6),
                    series={
                        "A": np.linspace(31.0, 36.0, 6),
                        "B": np.linspace(1.11, 1.66, 6),
                        "C": np.linspace(41.0, 46.0, 6),
                    },
                ),
            ],
            missing_items=[],
        )
    )

    panel._copy_all()

    rows = _split_tsv(clipboard.last_text)
    assert rows, "Expected clipboard TSV output"
    header = rows[0]
    body = rows[1:]
    assert len(body) == 6

    set1_cols = [idx for idx, cell in enumerate(header) if cell.startswith("set1::")]
    set2_cols = [idx for idx, cell in enumerate(header) if cell.startswith("set2::")]
    set3_cols = [idx for idx, cell in enumerate(header) if cell.startswith("set3::")]
    dataset_cols = [idx for idx, cell in enumerate(header) if cell.startswith("ds1::")]
    assert set1_cols
    assert set2_cols
    assert set3_cols
    assert dataset_cols

    assert sum(1 for cell in header if cell == "set2::Time (s)") == 1
    assert sum(1 for cell in header if cell.startswith("set2::") and "[B]" in cell) == 1
    assert "set3::Time (s)" not in header
    assert "901.0" not in clipboard.last_text
    assert "902.0" not in clipboard.last_text

    set1_time_idx = _find_header_index(header, prefix="set1::", contains=["Time"])
    set1_x_idx = _find_header_index(header, prefix="set1::", contains=["B", "[B]"])
    set1_a_idx = _find_header_index(header, prefix="set1::", contains=["A"])
    set3_x_idx = _find_header_index(header, prefix="set3::", contains=["B", "[B]"])
    set3_a_idx = _find_header_index(header, prefix="set3::", contains=["A"])
    dataset_x_idx = _find_header_index(header, prefix="ds1::", contains=["B", "[B]"])
    dataset_y_idx = _find_header_index(header, prefix="ds1::", contains=["A"])

    np.testing.assert_allclose(_numeric_column(body, set1_time_idx), np.linspace(0.0, 5.0, 6))
    np.testing.assert_allclose(_numeric_column(body, set1_x_idx), np.linspace(0.11, 0.66, 6))
    np.testing.assert_allclose(_numeric_column(body, set1_a_idx), np.linspace(11.0, 16.0, 6))
    assert all("C" not in cell for cell in header if cell.startswith("set1::"))
    assert all("C" not in cell for cell in header if cell.startswith("set2::"))
    assert all("C" not in cell for cell in header if cell.startswith("set3::"))
    assert all("C" not in cell for cell in header if cell.startswith("ds1::"))

    assert _numeric_column(body, set3_x_idx).shape[0] == 3
    assert _numeric_column(body, set3_a_idx).shape[0] == 3
    assert _numeric_column(body, dataset_x_idx).shape[0] == 3
    assert _numeric_column(body, dataset_y_idx).shape[0] == 3


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_all_soft_fail_yes_copies_available_blocks_only(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    clipboard.last_text = "unchanged"
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([0.1, 0.2, 0.3], dtype=float),
        },
        label="focused",
    )
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("B")

    captured_missing = []

    def _confirm(missing_items):
        captured_missing.extend(missing_items)
        return True

    monkeypatch.setattr(panel, "_confirm_copy_all_missing_items", _confirm, raising=False)
    plot_contract = plot_panel_impl
    panel.set_copy_all_export_plan_provider(
        lambda: plot_contract.CopyAllExportPlan(
            shown_blocks=[
                plot_contract.CopyAllShownBlock(
                    set_id="set1",
                    label="set1",
                    t=np.array([0.0, 1.0, 2.0], dtype=float),
                    series={
                        "A": np.array([10.0, 11.0, 12.0], dtype=float),
                        "B": np.array([0.4, 0.5, 0.6], dtype=float),
                    },
                )
            ],
            missing_items=[
                plot_contract.CopyAllMissingItem(
                    set_id="set2",
                    label="dup",
                    popup_label="dup (row 2)",
                    reason="no_cached_results",
                )
            ],
        )
    )

    panel._copy_all()

    assert [item.popup_label for item in captured_missing] == ["dup (row 2)"]
    assert [item.reason for item in captured_missing] == ["no_cached_results"]
    rows = _split_tsv(clipboard.last_text)
    assert rows
    assert any(cell.startswith("set1::") for cell in rows[0])
    assert all(not cell.startswith("set2::") for cell in rows[0])


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_all_does_not_recover_missing_shown_set_from_local_overlay_fallback(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([0.1, 0.2, 0.3], dtype=float),
        },
        label="dup (row 1)",
        overlays=[
            {
                "label": "dup",
                "set_id": "set2",
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "series": {
                    "A": np.array([10.0, 11.0, 12.0], dtype=float),
                    "B": np.array([0.4, 0.5, 0.6], dtype=float),
                },
            }
        ],
    )
    panel.set_selected_series(["A"])

    captured_missing = []

    def _confirm(missing_items):
        captured_missing.extend(missing_items)
        return True

    monkeypatch.setattr(panel, "_confirm_copy_all_missing_items", _confirm, raising=False)
    plot_contract = plot_panel_impl
    panel.set_copy_all_export_plan_provider(
        lambda: plot_contract.CopyAllExportPlan(
            shown_blocks=[
                plot_contract.CopyAllShownBlock(
                    set_id="set1",
                    label="dup (row 1)",
                    t=np.array([0.0, 1.0, 2.0], dtype=float),
                    series={
                        "A": np.array([1.0, 2.0, 3.0], dtype=float),
                        "B": np.array([0.1, 0.2, 0.3], dtype=float),
                    },
                )
            ],
            missing_items=[
                plot_contract.CopyAllMissingItem(
                    set_id="set2",
                    label="dup",
                    popup_label="dup (row 2)",
                    reason="no_cached_results",
                )
            ],
        )
    )

    panel._copy_all()

    assert [item.popup_label for item in captured_missing] == ["dup (row 2)"]
    rows = _split_tsv(clipboard.last_text)
    assert rows
    header = rows[0]
    assert any(cell.startswith("dup (row 1)::") for cell in header)
    assert all(not cell.startswith("dup::") for cell in header)
    assert all(not cell.startswith("dup (row 2)::") for cell in header)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_copy_all_soft_fail_no_leaves_clipboard_unchanged(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    clipboard = _DummyClipboard()
    clipboard.last_text = "unchanged"
    monkeypatch.setattr(panel, "_get_clipboard", lambda: clipboard)

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([1.0, 2.0, 3.0], dtype=float),
            "B": np.array([0.1, 0.2, 0.3], dtype=float),
        },
        label="focused",
    )
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("B")

    monkeypatch.setattr(panel, "_confirm_copy_all_missing_items", lambda _missing_items: False, raising=False)
    plot_contract = plot_panel_impl
    panel.set_copy_all_export_plan_provider(
        lambda: plot_contract.CopyAllExportPlan(
            shown_blocks=[
                plot_contract.CopyAllShownBlock(
                    set_id="set1",
                    label="set1",
                    t=np.array([0.0, 1.0, 2.0], dtype=float),
                    series={
                        "A": np.array([10.0, 11.0, 12.0], dtype=float),
                        "B": np.array([0.4, 0.5, 0.6], dtype=float),
                    },
                )
            ],
            missing_items=[
                plot_contract.CopyAllMissingItem(
                    set_id="set2",
                    label="set2",
                    popup_label="set2",
                    reason="preview_pending",
                )
            ],
        )
    )

    panel._copy_all()

    assert clipboard.last_text == "unchanged"


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
def test_species_x_partial_render_state_filters_incompatible_y_series_across_render_export_and_copy(
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

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    primary_headers = [cell for cell in rows[0] if cell.startswith("set1::")]
    assert any(cell.endswith("::A") or cell == "set1::A" for cell in primary_headers)
    assert all(not cell.endswith("::C") and "C [right axis]" not in cell for cell in primary_headers)

@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_species_x_mixed_primary_and_overlay_lengths_keep_overlay_local_c_series_visible_and_exported(
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
        overlays=[
            {
                "label": "set2",
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "series": {
                    "A": np.array([101.0, 102.0, 103.0, 104.0, 105.0], dtype=float),
                    "B": np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=float),
                    "C": np.array([70.0, 80.0, 90.0, 100.0, 110.0], dtype=float),
                },
            }
        ],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "species": {
                    "A": np.array([201.0, 202.0, 203.0, 204.0, 205.0], dtype=float),
                    "B": np.array([21.0, 22.0, 23.0, 24.0, 25.0], dtype=float),
                    "C": np.array([170.0, 180.0, 190.0, 200.0, 210.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A", "C"}
    panel.set_selected_series(["A", "C"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    assert panel._format_species_set_label("A", "set1") in panel._plot_items
    assert panel._format_species_set_label("C", "set1") not in panel._plot_items
    assert panel._format_species_set_label("C", "set2") in panel._plot_items
    assert any(entry.dataset == "ds1" and entry.species == "C" for entry in panel._active_overlay_series)

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert any(header == "A" for header in axis_header)
    assert any(header.startswith("set2::") and header.endswith("::C") for header in axis_header)
    assert any(header.startswith("ds1::") and header.endswith("::C") for header in axis_header)

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header.startswith("set2::") and header.endswith("::C") for header in all_header)
    assert any(header.startswith("ds1::") and header.endswith("::C") for header in all_header)

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    primary_headers = [cell for cell in rows[0] if cell.startswith("set1::")]
    assert any(cell.endswith("::A") or cell == "set1::A" for cell in primary_headers)
    assert all(not cell.endswith("::C") and "C [right axis]" not in cell for cell in primary_headers)
    assert any(cell.startswith("set2::") and cell.endswith("::C") for cell in rows[0])
    assert any(cell.startswith("ds1::") and cell.endswith("::C") for cell in rows[0])


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_species_x_overlay_only_visible_state_exports_and_copies_overlay_blocks(qtbot, monkeypatch):
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
        overlays=[
            {
                "label": "set2",
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "series": {
                    "B": np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=float),
                    "C": np.array([70.0, 80.0, 90.0, 100.0, 110.0], dtype=float),
                },
            }
        ],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "species": {
                    "B": np.array([21.0, 22.0, 23.0, 24.0, 25.0], dtype=float),
                    "C": np.array([170.0, 180.0, 190.0, 200.0, 210.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"C"}
    panel.set_selected_series(["C"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    assert panel._format_species_set_label("C", "set1") not in panel._plot_items
    assert panel._format_species_set_label("C", "set2") in panel._plot_items
    assert any(entry.dataset == "ds1" and entry.species == "C" for entry in panel._active_overlay_series)

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert all(header != "C" for header in axis_header)
    assert any(header.startswith("set2::") and header.endswith("::C") for header in axis_header)
    assert any(header.startswith("ds1::") and header.endswith("::C") for header in axis_header)

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    assert all(not cell.startswith("set1::") for cell in rows[0])
    assert any(cell.startswith("set2::") and cell.endswith("::C") for cell in rows[0])
    assert any(cell.startswith("ds1::") and cell.endswith("::C") for cell in rows[0])


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_species_x_dataset_overlay_enabled_subset_skips_simulation_only_species_for_status_and_export(
    qtbot,
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0], dtype=float),
            "C": np.array([7.0, 8.0, 9.0, 10.0, 11.0], dtype=float),
        },
        label="set1",
        overlays=[
            {
                "label": "set2",
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "series": {
                    "A": np.array([101.0, 102.0, 103.0, 104.0, 105.0], dtype=float),
                    "B": np.array([11.0, 12.0, 13.0, 14.0, 15.0], dtype=float),
                    "C": np.array([70.0, 80.0, 90.0, 100.0, 110.0], dtype=float),
                },
            }
        ],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                "species": {
                    "A_conc": np.array([201.0, 202.0, 203.0, 204.0, 205.0], dtype=float),
                    "B": np.array([21.0, 22.0, 23.0, 24.0, 25.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
    panel.set_selected_series(["A", "C"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    assert panel._format_species_set_label("A", "set1") in panel._plot_items
    assert panel._format_species_set_label("C", "set1") not in panel._plot_items
    assert panel._format_species_set_label("C", "set2") in panel._plot_items
    assert any(entry.dataset == "ds1" and entry.species == "A" for entry in panel._active_overlay_series)
    assert all(not (entry.dataset == "ds1" and entry.species == "C") for entry in panel._active_overlay_series)

    status_text = panel._overlay_panel._status_label.text()
    assert "species 'C'" not in status_text

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert any(header.startswith("set2::") and header.endswith("::C") for header in axis_header)
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in axis_header)
    assert all(not (header.startswith("ds1::") and header.endswith("::C")) for header in axis_header)

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header.startswith("set2::") and header.endswith("::C") for header in all_header)
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in all_header)
    assert all(not (header.startswith("ds1::") and header.endswith("::C")) for header in all_header)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_dataset_overlay_cache_refreshes_on_enabled_species_change_and_later_consumers_do_not_reresolve(
    qtbot,
    monkeypatch,
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

    alias_values = np.array([201.0, 202.0, 203.0], dtype=float)
    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0], dtype=float),
        },
        label="set1",
        owned_species=["A", "A_conc", "B"],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "species": {
                    "A": np.array([101.0, 102.0, 103.0], dtype=float),
                    "A_conc": alias_values,
                    "B": np.array([1.0, 2.0, 3.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A"}
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    initial_overlay = panel._active_overlay_series[0]
    assert initial_overlay.resolved_x_column == "B"
    assert initial_overlay.resolved_y_column == "A"

    panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
    panel._on_overlay_selection_changed(["ds1"])
    QtWidgets.QApplication.processEvents()

    refreshed_overlay = panel._active_overlay_series[0]
    assert refreshed_overlay.resolved_x_column == "B"
    assert refreshed_overlay.resolved_y_column == "A_conc"
    np.testing.assert_allclose(refreshed_overlay.x, np.array([1.0, 2.0, 3.0], dtype=float))
    np.testing.assert_allclose(refreshed_overlay.y, alias_values)

    initial_all_header, initial_all_rows = panel.build_visible_export("all")
    assert initial_all_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in initial_all_header)

    def _fail_resolve(*_args, **_kwargs):
        raise AssertionError("overlay consumer re-resolved dataset provenance after build")

    monkeypatch.setattr(plot_panel_impl, "_resolve_dataset_species", _fail_resolve)

    panel.refresh_overlay_presentation_for_current_roster()

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in axis_header)

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in all_header)

    panel._copy_visible_data()

    assert warning_calls == []
    rows = _split_tsv(clipboard.last_text)
    assert rows
    assert any(cell.startswith("ds1::") and cell.endswith("::A") for cell in rows[0])


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_export_all_overlay_cache_is_built_lazily(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.array([0.0, 1.0, 2.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0], dtype=float),
        },
        label="set1",
        owned_species=["A", "B"],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0], dtype=float),
                "species": {
                    "A": np.array([101.0, 102.0, 103.0], dtype=float),
                    "B": np.array([11.0, 12.0, 13.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A", "B"}
    panel.set_selected_series(["A"])
    QtWidgets.QApplication.processEvents()

    build_calls: list[tuple[str, ...]] = []
    original_build = panel._build_overlay_series

    def _spy_build(selected_series):
        build_calls.append(tuple(selected_series))
        return original_build(selected_series)

    monkeypatch.setattr(panel, "_build_overlay_series", _spy_build)

    panel._update_plot()
    QtWidgets.QApplication.processEvents()

    assert build_calls == [("A",)]
    assert panel._export_all_overlay_series == []
    assert panel._export_all_overlay_warnings == []

    build_calls.clear()
    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in all_header)
    assert build_calls == [("A", "B")]

    build_calls.clear()
    cached_header, cached_rows = panel.build_visible_export("all")
    assert cached_rows == all_rows
    assert cached_header == all_header
    assert build_calls == []


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_time_axis_dataset_overlay_enabled_alias_remains_visible_and_exported(
    qtbot,
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        },
        label="set1",
        owned_species=["A", "A_conc"],
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
                "species": {
                    "A": np.array([101.0, 102.0, 103.0, 104.0], dtype=float),
                    "A_conc": np.array([201.0, 202.0, 203.0, 204.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("t")
    QtWidgets.QApplication.processEvents()

    assert len(panel._active_overlay_series) == 1
    overlay = panel._active_overlay_series[0]
    assert overlay.dataset == "ds1"
    assert overlay.species == "A"
    assert overlay.resolved_y_column == "A_conc"
    assert panel._overlay_panel._status_label.text() == ""

    axis_header, axis_rows = panel.build_visible_export("axis")
    assert axis_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in axis_header)

    all_header, all_rows = panel.build_visible_export("all")
    assert all_rows
    assert any(header.startswith("ds1::") and header.endswith("::A") for header in all_header)


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_time_axis_dataset_overlay_filtered_subset_still_warns_and_blocks_export_for_missing_series(
    qtbot,
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        },
        label="set1",
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
                "species": {
                    "A_conc": np.array([201.0, 202.0, 203.0, 204.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
    panel.set_selected_series(["B"])
    panel._on_x_axis_changed("t")
    QtWidgets.QApplication.processEvents()

    assert all(entry.dataset != "ds1" for entry in panel._active_overlay_series)

    status_text = panel._overlay_panel._status_label.text()
    assert "species 'B'" in status_text

    with pytest.raises(ValueError, match="Cannot export overlay datasets until issues are resolved"):
        panel.build_visible_export("axis")

    with pytest.raises(ValueError, match="Cannot export overlay datasets until issues are resolved"):
        panel.build_visible_export("all")


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_species_x_dataset_overlay_enabled_column_still_warns_and_blocks_export_for_real_mismatch(
    qtbot,
):
    panel = PyQtGraphPlotPanel(enable_copy_visible_data_action=True)
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    panel.set_data(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        {
            "A": np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            "B": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        },
        label="set1",
    )
    panel.set_overlay_catalog(
        {
            "ds1": {
                "t": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
                "species": {
                    "A_conc": np.array([201.0, 202.0, 203.0], dtype=float),
                    "B": np.array([21.0, 22.0, 23.0, 24.0], dtype=float),
                },
            }
        }
    )
    panel._overlay_panel._selected["ds1"] = True
    panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
    panel.set_selected_series(["A"])
    panel._on_x_axis_changed("B")
    QtWidgets.QApplication.processEvents()

    status_text = panel._overlay_panel._status_label.text()
    assert "A_conc" in status_text
    assert "length" in status_text

    with pytest.raises(ValueError, match="Cannot export overlay datasets until issues are resolved"):
        panel.build_visible_export("axis")


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

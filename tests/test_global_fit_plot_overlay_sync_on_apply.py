from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def _make_window(*, selected_species: list[str]):
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 6)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            # Provide full observed series; selected_species is the applied fit-target snapshot.
            "species_data": {"A": y_a.copy(), "B": y_b.copy()},
            "selected_species": list(selected_species),
            "weight": 1.0,
            "include": True,
        }
    ]
    selected_rows = [np.asarray(dataset_entries[0]["species_data"][name]) for name in selected_species]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": t.copy(),
            "y": np.vstack(selected_rows),
            "species": list(selected_species),
        }
    ]
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )


def _set_fit_targets_dataset(widget, *, dataset_id: str) -> None:
    from PySide6 import QtCore
    window = widget if hasattr(widget, '_data_targets_tab') else widget.window()
    ulist = window._data_targets_tab.unified_list._list
    for i in range(ulist.count()):
        item = ulist.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            ulist.setCurrentRow(i)
            return
    raise AssertionError(f"Dataset id not in list: {dataset_id!r}")


def test_apply_fit_targets_refreshes_overlay_and_plot_without_manual_toggle(qt_app):
    from PySide6 import QtCore, QtWidgets

    window = _make_window(selected_species=["A"])
    try:
        subset = window._subset_widget
        selector = subset._selector

        assert selector.selected_dataset_species() == {"ds1": {"A"}}
        assert {ds["name"] for ds in getattr(subset._grid, "_datasets", [])} == {"ds1"}

        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        assert apply_btn is not None

        _set_fit_targets_dataset(panel, dataset_id="ds1")

        from kindred.gui.fitting.unified_species_table import _Col
        table = window._species_table._table
        for row in range(table.rowCount()):
            name = table.item(row, _Col.SPECIES).text()
            if name == "A":
                table.item(row, _Col.INCLUDE).setCheckState(QtCore.Qt.Unchecked)
            elif name == "B":
                table.item(row, _Col.INCLUDE).setCheckState(QtCore.Qt.Checked)
        qt_app.processEvents()

        apply_btn.click()
        qt_app.processEvents()

        assert selector.selected_dataset_species() == {"ds1": {"B"}}
        assert {ds["name"] for ds in getattr(subset._grid, "_datasets", [])} == {"ds1"}
    finally:
        window.close()
        qt_app.processEvents()

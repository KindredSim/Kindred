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


def _set_fit_targets_dataset(panel, *, dataset_id: str) -> None:
    from PySide6 import QtCore, QtWidgets

    dataset_list = panel.window().findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
    assert dataset_list is not None
    for i in range(dataset_list.count()):
        item = dataset_list.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            dataset_list.setCurrentRow(i)
            return
    raise AssertionError(f"Dataset id not in list: {dataset_id!r}")


def test_apply_fit_targets_refreshes_overlay_and_plot_without_manual_toggle(qt_app):
    from PySide6 import QtWidgets

    window = _make_window(selected_species=["A"])
    try:
        subset = window._subset_widget
        selector = subset._selector

        assert selector.selected_dataset_species() == {"ds1": {"A"}}
        assert {ds["name"] for ds in getattr(subset._grid, "_datasets", [])} == {"ds1"}

        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        _set_fit_targets_dataset(panel, dataset_id="ds1")

        checkbox_a = next(cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() == "A")
        checkbox_b = next(cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() == "B")
        checkbox_a.setChecked(False)
        checkbox_b.setChecked(True)
        qt_app.processEvents()

        apply_btn.click()
        qt_app.processEvents()

        assert selector.selected_dataset_species() == {"ds1": {"B"}}
        assert {ds["name"] for ds in getattr(subset._grid, "_datasets", [])} == {"ds1"}
    finally:
        window.close()
        qt_app.processEvents()

from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def _make_window(*, selected_species: list[str]):
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 6)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)
    y_c = np.linspace(0.3, 0.1, t.size)

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": y_a.copy(), "B": y_b.copy(), "C": y_c.copy()},
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
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy(), "C": y_c.copy()}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )


def _set_fit_targets_dataset(panel, *, dataset_id: str) -> None:
    from PySide6 import QtWidgets

    combo = panel.findChild(QtWidgets.QComboBox, "global_fit_fit_targets_dataset_combo")
    assert combo is not None
    for i in range(combo.count()):
        if str(combo.itemData(i)) == str(dataset_id):
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"Dataset id not in combo: {dataset_id!r}")


def test_fit_targets_bulk_buttons_update_pending_only_and_require_apply(qt_app):
    from PySide6 import QtWidgets

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None

        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None
        bulk_all = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_bulk_all")
        bulk_none = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_bulk_none")
        bulk_invert = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_bulk_invert")
        assert bulk_all is not None
        assert bulk_none is not None
        assert bulk_invert is not None

        _set_fit_targets_dataset(panel, dataset_id="ds1")

        available = set(window._fit_targets_available_by_dataset["ds1"])
        assert window._fit_targets_selection_applied["ds1"] == ["A"]
        assert set(window._fit_targets_selection_pending["ds1"]) == {"A"}
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        bulk_none.click()
        qt_app.processEvents()
        assert set(window._fit_targets_selection_pending["ds1"]) == set()
        assert window._fit_targets_selection_applied["ds1"] == ["A"]
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]
        assert apply_btn.isEnabled()

        bulk_invert.click()
        qt_app.processEvents()
        assert set(window._fit_targets_selection_pending["ds1"]) == available
        assert window._fit_targets_selection_applied["ds1"] == ["A"]
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        bulk_invert.click()
        qt_app.processEvents()
        assert set(window._fit_targets_selection_pending["ds1"]) == set()

        bulk_all.click()
        qt_app.processEvents()
        assert set(window._fit_targets_selection_pending["ds1"]) == available
        assert window._fit_targets_selection_applied["ds1"] == ["A"]
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        apply_btn.click()
        qt_app.processEvents()
        assert set(window._fit_targets_selection_applied["ds1"]) == available
        assert set(window._global_payload_lookup["ds1"]["species"]) == available
    finally:
        window.close()
        qt_app.processEvents()

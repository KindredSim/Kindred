from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets

pytestmark = [pytest.mark.gui]


def _combo_items(combo: QtWidgets.QComboBox) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def test_species_statistics_selector_populates_and_switches(main_window, qtbot):
    qtbot.addWidget(main_window)
    if hasattr(main_window, "set_simulation_cache_caps"):
        main_window.set_simulation_cache_caps(result_cap=10, preview_cap=10)

    cache_key = "stats-selector-test"
    row_a = main_window._batch_store.ensure_set("Set A")
    row_b = main_window._batch_store.ensure_set("Set B")
    set_id_a = main_window._batch_store.set_id_for_row(row_a)
    set_id_b = main_window._batch_store.set_id_for_row(row_b)

    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    series_a = {"A": np.asarray([0.0, 0.5, 1.0], dtype=float)}
    series_b = {"A": np.asarray([0.0, 1.0, 2.0], dtype=float)}

    cache = main_window.simulation_controller.batch_cache.result_cache
    assert cache is not None
    if hasattr(cache, "clear"):
        cache.clear()
    cache[f"{cache_key}::{set_id_a}"] = {"t": t, "series": series_a, "algebra_scalars": {}}
    cache[f"{cache_key}::{set_id_b}"] = {"t": t, "series": series_b, "algebra_scalars": {}}
    main_window.simulation_controller.batch_cache.active_cache_key = cache_key

    ok = main_window.display_cached_batch_selection(
        cache_key=cache_key,
        selected_sets=[set_id_a, set_id_b],
        prefer_set=set_id_a,
    )
    assert ok is True

    plot = main_window._plot_tabs._main_plot
    selector = getattr(plot, "_stats_result_selector", None)
    assert isinstance(selector, QtWidgets.QComboBox)
    assert _combo_items(selector) == ["Set A", "Set B"]

    table = plot.stats_table()
    QtWidgets.QApplication.processEvents()
    policy = table.sizePolicy()
    assert policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert table.maximumHeight() == 16_777_215

    assert table.item(0, 0).text() == "A"
    assert table.item(0, 1).text() == "1"

    selector.setCurrentText("Set B")
    QtWidgets.QApplication.processEvents()
    assert table.item(0, 1).text() == "2"

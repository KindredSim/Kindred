from __future__ import annotations

import types

import numpy as np
import pytest

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


def _assert_ranges_close(before, after) -> None:
    assert len(before) == 2
    assert len(after) == 2
    for axis in range(2):
        assert len(before[axis]) == 2
        assert len(after[axis]) == 2
        for bound in range(2):
            b = float(before[axis][bound])
            a = float(after[axis][bound])
            tol = 1e-6 * max(1.0, abs(b))
            assert abs(a - b) <= tol


def test_global_fit_running_state_disables_subset_grid_autorange(qt_app):
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        grid = window._subset_widget._grid
        assert grid._plot_items, "Expected at least one plot item in the subset grid."
        vb = grid._plot_items[0].getViewBox()

        window._set_running_state(True)
        qt_app.processEvents()

        assert tuple(vb.autoRangeEnabled()) == (False, False)
    finally:
        window.close()


def test_global_fit_best_updates_do_not_change_view_range_while_running(qt_app):
    window = _make_window()
    try:
        window.show()
        qt_app.processEvents()

        grid = window._subset_widget._grid
        assert grid._plot_items, "Expected at least one plot item in the subset grid."

        window._set_running_state(True)
        qt_app.processEvents()

        vb_before = grid._plot_items[0].getViewBox()
        range_before = vb_before.viewRange()

        window._worker = types.SimpleNamespace(isRunning=lambda: True)
        payload = {
            "cost": 1.0,
            "shared_params": {"k1": 1.23},
            "dataset_params": {"ds1": {}},
            "model_series": {"ds1": {"A": np.asarray([1e6, 1e6, 1e6, 1e6, 1e6], dtype=float)}},
            "dataset_stats": {"ds1": {"chi_squared": 1.0, "r_squared": 0.0}},
        }
        window._handle_global_best_update(payload)
        window._apply_pending_best_update()
        qt_app.processEvents()

        vb_after = grid._plot_items[0].getViewBox()
        range_after = vb_after.viewRange()
        _assert_ranges_close(range_before, range_after)
    finally:
        window.close()

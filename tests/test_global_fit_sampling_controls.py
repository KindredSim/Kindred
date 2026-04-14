from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def _make_window(*, t1: np.ndarray, t2: np.ndarray):
    from kindred.gui.fitting.window import FittingWindow

    y1 = t1.copy()
    y2 = (t2 * 3.0).copy()
    x1 = (t1 * 0.5).copy()
    x2 = (t2 * 0.25).copy()

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t1.copy(),
            "species_data": {"A": y1.copy(), "X": x1.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "ds2",
            "t": t2.copy(),
            "species_data": {"A": y2.copy(), "X": x2.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
    ]

    sim_t = np.arange(0, max(int(t1.max(initial=0)), int(t2.max(initial=0))) + 1, dtype=float)
    sim_y = np.zeros_like(sim_t)

    def simulation_func(_params):
        return {"t": sim_t.copy(), "species": {"A": sim_y.copy()}}

    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_species=["A", "X"],
    )


def _select_dataset_row(window, *, row: int, qt_app) -> None:
    table = getattr(window._data_tab, "_dataset_table", None)
    assert table is not None
    table.selectRow(int(row))
    qt_app.processEvents()


def _sampling_widgets(window):
    from PySide6 import QtWidgets

    panel = window.findChild(QtWidgets.QWidget, "global_fit_sampling_panel")
    assert panel is not None
    x_axis = panel.findChild(QtWidgets.QComboBox, "global_fit_sampling_x_axis")
    x_mode = panel.findChild(QtWidgets.QComboBox, "global_fit_sampling_x_mode")
    t_min = panel.findChild(QtWidgets.QDoubleSpinBox, "global_fit_sampling_t_min")
    t_max = panel.findChild(QtWidgets.QDoubleSpinBox, "global_fit_sampling_t_max")
    n_points = panel.findChild(QtWidgets.QSpinBox, "global_fit_sampling_n_points")
    apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_sampling_apply")
    assert x_axis is not None
    assert x_mode is not None
    assert t_min is not None
    assert t_max is not None
    assert n_points is not None
    assert apply_btn is not None
    return panel, x_axis, x_mode, t_min, t_max, n_points, apply_btn


def test_advanced_integration_settings_widgets_present(qt_app):
    from PySide6 import QtWidgets

    t1 = np.arange(0, 20, dtype=float)
    t2 = np.arange(0, 10, dtype=float)
    window = _make_window(t1=t1, t2=t2)
    try:
        qt_app.processEvents()
        solver_combo = window.findChild(QtWidgets.QComboBox, "global_fit_integration_solver")
        rtol_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_integration_rtol")
        atol_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_integration_atol")
        assert solver_combo is not None
        assert rtol_edit is not None
        assert atol_edit is not None

        items = [solver_combo.itemText(i) for i in range(solver_combo.count())]
        assert items == ["Radau", "BDF"]
        assert rtol_edit.text() == "1e-6"
        assert atol_edit.text() == "1e-12"
    finally:
        window.close()
        qt_app.processEvents()


def test_sampling_changes_do_not_affect_payload_until_apply(qt_app):
    t1 = np.arange(0, 200, dtype=float)
    t2 = np.arange(0, 10, dtype=float)

    window = _make_window(t1=t1, t2=t2)
    try:
        qt_app.processEvents()

        original = window._global_payload_lookup["ds1"]["t"].copy()
        assert original.size == 200
        assert window._global_payload_lookup["ds1"].get("x_name") in (None, "t")

        _select_dataset_row(window, row=0, qt_app=qt_app)
        _panel, _x_axis, _x_mode, t_min, t_max, n_points, apply_btn = _sampling_widgets(window)

        t_min.setValue(50.0)
        t_max.setValue(149.0)
        n_points.setValue(10)
        qt_app.processEvents()

        # Pending edits do not affect payload until Apply.
        pending_payload = window._global_payload_lookup["ds1"]["t"]
        assert pending_payload.size == 200

        apply_btn.click()
        qt_app.processEvents()

        sampled = window._global_payload_lookup["ds1"]["t"]
        assert sampled.size == 10
        assert sampled[0] == 50.0
        assert sampled[-1] == 149.0
        assert np.all((sampled >= 50.0) & (sampled <= 149.0))
        assert np.all(np.diff(sampled) > 0)
        assert list(sampled) == [50.0, 61.0, 72.0, 83.0, 94.0, 105.0, 116.0, 127.0, 138.0, 149.0]

        y = window._global_payload_lookup["ds1"]["y"]
        assert y.shape == (1, 10)
        assert list(y.reshape(-1)) == list(sampled)
    finally:
        window.close()
        qt_app.processEvents()


def test_x_axis_selection_is_apply_only_and_populates_payload(qt_app):
    t1 = np.arange(0, 20, dtype=float)
    t2 = np.arange(0, 10, dtype=float)

    window = _make_window(t1=t1, t2=t2)
    try:
        qt_app.processEvents()

        # Default is time.
        assert window._global_payload_lookup["ds1"].get("x_name") in (None, "t")
        assert "x_obs" not in window._global_payload_lookup["ds1"]

        _select_dataset_row(window, row=0, qt_app=qt_app)
        _panel, x_axis, _x_mode, _t_min, _t_max, _n_points, apply_btn = _sampling_widgets(window)

        # Pending selection does not affect payload until Apply.
        x_axis.setCurrentIndex(max(0, x_axis.findData("X")))
        qt_app.processEvents()
        assert window._global_payload_lookup["ds1"].get("x_name") in (None, "t")
        assert "x_obs" not in window._global_payload_lookup["ds1"]

        apply_btn.click()
        qt_app.processEvents()

        payload = dict(window._global_payload_lookup["ds1"])
        assert payload.get("x_name") == "X"
        x_obs = np.asarray(payload.get("x_obs"), dtype=float).reshape(-1)
        t_obs = np.asarray(payload.get("t"), dtype=float).reshape(-1)
        assert x_obs.size == t_obs.size
        assert np.allclose(x_obs, 0.5 * t_obs)

        # Per-dataset setting: ds2 remains time until applied.
        assert window._global_payload_lookup["ds2"].get("x_name") in (None, "t")
    finally:
        window.close()
        qt_app.processEvents()


def test_implicit_weights_use_post_sampling_point_counts(qt_app, monkeypatch):
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t1 = np.arange(0, 200, dtype=float)
    t2 = np.arange(0, 100, dtype=float)
    window = _make_window(t1=t1, t2=t2)

    def _fake_least_squares(fun, x0, **_kwargs):
        residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)

        class _Result:
            pass

        result = _Result()
        result.x = np.asarray(x0, dtype=float)
        result.success = True
        result.message = "fake"
        result.nfev = 1
        result.fun = residuals
        result.jac = np.zeros((residuals.size, result.x.size), dtype=float)
        return result

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (_fake_least_squares, lambda *_a, **_k: None),
    )

    try:
        qt_app.processEvents()
        _select_dataset_row(window, row=0, qt_app=qt_app)
        _panel, _x_axis, _x_mode, t_min, t_max, n_points, apply_btn = _sampling_widgets(window)
        t_min.setValue(50.0)
        t_max.setValue(149.0)
        n_points.setValue(10)
        qt_app.processEvents()
        apply_btn.click()
        qt_app.processEvents()

        ds1 = dict(window._global_payload_lookup["ds1"])
        ds2 = dict(window._global_payload_lookup["ds2"])
        assert int(np.asarray(ds1["t"]).size) == 10
        assert int(np.asarray(ds2["t"]).size) == 100

        result = global_fitting.fit_global(
            lambda _params: {"t": np.arange(0, 200, dtype=float), "species": {"A": np.zeros(200, dtype=float)}},
            datasets=[ds1, ds2],
            shared_params={"k1": 0.2},
            weights=None,
            method="trf",
            max_nfev=1,
        )
        weights_by_id = {info.dataset_id: float(info.weight) for info in result.dataset_info}
        assert set(weights_by_id) == {"ds1", "ds2"}
        assert weights_by_id["ds1"] / weights_by_id["ds2"] == pytest.approx(10.0)
    finally:
        window.close()
        qt_app.processEvents()


def test_x_mapping_mode_is_apply_only_and_plumbed(qt_app):
    t1 = np.arange(0, 20, dtype=float)
    t2 = np.arange(0, 10, dtype=float)

    window = _make_window(t1=t1, t2=t2)
    try:
        qt_app.processEvents()

        assert "x_mapping_mode" not in window._global_payload_lookup["ds1"]

        _select_dataset_row(window, row=0, qt_app=qt_app)
        _panel, x_axis, x_mode, _t_min, _t_max, _n_points, apply_btn = _sampling_widgets(window)

        assert x_axis.currentData() == "t"
        assert x_mode.isEnabled() is False
        assert x_mode.isHidden() is True

        x_axis.setCurrentIndex(max(0, x_axis.findData("X")))
        qt_app.processEvents()
        assert x_mode.isEnabled() is True
        assert x_mode.isHidden() is False

        x_mode.setCurrentIndex(max(0, x_mode.findData("time_guided")))
        qt_app.processEvents()

        # Pending edits do not affect payload until Apply.
        assert "x_mapping_mode" not in window._global_payload_lookup["ds1"]

        apply_btn.click()
        qt_app.processEvents()

        payload = dict(window._global_payload_lookup["ds1"])
        assert payload.get("x_name") == "X"
        assert payload.get("x_mapping_mode") == "time_guided"

        x_mode.setCurrentIndex(max(0, x_mode.findData("monotone")))
        qt_app.processEvents()
        assert dict(window._global_payload_lookup["ds1"]).get("x_mapping_mode") == "time_guided"

        apply_btn.click()
        qt_app.processEvents()
        assert dict(window._global_payload_lookup["ds1"]).get("x_mapping_mode") == "monotone"
    finally:
        window.close()
        qt_app.processEvents()

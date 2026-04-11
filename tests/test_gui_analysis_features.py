import os

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

pytestmark = pytest.mark.gui

pytest.importorskip("scipy", reason="requires scipy for GUI analysis feature tests")

from kindred.gui.main_window import MainWindow  # noqa: E402  # reason: guarded by pytest.importorskip("scipy")
from kindred.core.analysis.global_fitting import (  # noqa: E402  # reason: guarded by pytest.importorskip("scipy")
    DatasetFitInfo,
    GlobalFitResult,
)


@pytest.fixture(scope="session")
def qt_app():
    """Ensure a QApplication exists for GUI-driven tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtCore.QStandardPaths.setTestModeEnabled(True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def analysis_window(qt_app, monkeypatch, tmp_path):
    """Provide a MainWindow configured with a simple mechanism."""
    def _fake_templates_dir(_self):
        target = tmp_path / "templates"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    window = MainWindow()
    dsl = "\n".join([
        "reaction: A -> B; k=0.2",
        "initial: A=1.0",
        "initial: B=0.0",
    ])
    window._mechanism_editor._reactions_text.setPlainText(dsl)
    yield window
    window.close()


def test_global_fit_handler_invokes_backend(monkeypatch, analysis_window):
    t = np.linspace(0, 5, 50)
    y = np.exp(-0.2 * t)
    dataset = {"t": t, "species": {"A": y}}
    analysis_window._on_dataset_loaded("exp1.csv", dataset)
    analysis_window._on_dataset_loaded("exp2.csv", dataset)
    panel = analysis_window._right_panel._data_manager
    panel._datasets["exp1.csv"] = dataset
    panel._datasets["exp2.csv"] = dataset

    _ = GlobalFitResult(
        success=True,
        shared_params={"k": 0.21},
        dataset_params={},
        uncertainties=None,
        global_chi_squared=0.01,
        global_r_squared=0.99,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="exp1.csv",
                r_squared=0.99,
                chi_squared=0.01,
                rmse=0.001,
                mae=0.001,
                residuals=np.zeros_like(t),
                n_points=len(t),
                weight=1.0,
            )
        ],
        nfev=12,
        message="ok",
        covariance=None,
    )

    created = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["kwargs"] = kwargs

        def setWindowTitle(self, title):
            created["title"] = title

        def show(self):
            created["show"] = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)
    monkeypatch.setattr(
        analysis_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda mech: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    monkeypatch.setattr(
        MainWindow,
        "_extract_mechanism_initials",
        lambda self, mechanism: {"A": 1.0},
        raising=False,
    )

    monkeypatch.setattr(
        MainWindow,
        "_simulate_mechanism",
        lambda self, mechanism, t_end, num_points: {
            "t": t,
            "species": {"A": y},
        },
    )
    analysis_window._run_global_fit()
    assert created, "Global fitting window should open"
    kwargs = created["kwargs"]
    assert kwargs["mode"] == "global"
    assert len(kwargs["dataset_entries"]) == 2
    assert kwargs["parameter_defs"][0]["name"] == "k1"


def test_fitting_window_smoke(qapp):
    """Basic instantiation of FittingWindow should not raise."""
    from kindred.gui.fitting.window import FittingWindow

    parameter_defs = [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]
    t_axis = np.linspace(0, 2, 8)
    dataset_entries = [{
        "id": "demo",
        "label": "demo",
        "t": t_axis,
        "species_data": {"A": np.exp(-0.2 * t_axis)},
        "selected_species": ["A"],
        "weight": 1.0,
        "include": True,
    }]

    window = FittingWindow(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        apply_callback=lambda params: None,
    )
    assert window._params_ics_tab._param_table.rowCount() == 1
    assert window._data_tab._dataset_table.rowCount() == 1
    assert window._params_ics_tab._param_table.item(0, 2).text() == "k1"
    assert pytest.approx(float(window._params_ics_tab._param_table.item(0, 3).text()), rel=1e-9) == 0.2
    assert pytest.approx(float(window._params_ics_tab._param_table.item(0, 4).text()), rel=1e-9) == 0.01
    assert pytest.approx(float(window._params_ics_tab._param_table.item(0, 5).text()), rel=1e-9) == 1.0
    window.close()


def test_fitting_window_small_multiples_grid(qapp):
    pytest.importorskip("pyqtgraph", reason="pyqtgraph is required for the fitting window small-multiples grid.")
    from kindred.gui.fitting.window import FittingWindow

    parameter_defs = [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]
    t_axis = np.linspace(0, 2, 8)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t_axis,
            "species_data": {"A": np.exp(-0.2 * t_axis)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "ds2",
            "t": t_axis,
            "species_data": {"A": np.exp(-0.3 * t_axis)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds3",
            "label": "ds3",
            "t": t_axis,
            "species_data": {"A": np.exp(-0.4 * t_axis)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        apply_callback=lambda params: None,
    )

    assert set(window._run_results_tab._dataset_plot_views.keys()) == {"ds1", "ds2", "ds3"}
    assert {ds["name"] for ds in getattr(window._run_results_tab._all_datasets_plot_view, "_datasets", [])} == {"ds1", "ds2", "ds3"}

    model_series = {
        "ds1": {"A": np.exp(-0.25 * t_axis)},
        "ds2": {"A": np.exp(-0.35 * t_axis)},
        "ds3": {"A": np.exp(-0.45 * t_axis)},
    }
    window._run_results_tab.push_live_update({"model_series": model_series, "dataset_stats": {}}, refresh_all=True)
    ds1_payload = next(ds for ds in window._run_results_tab._all_datasets_plot_view._datasets if ds.get("name") == "ds1")
    assert ds1_payload.get("model_y") is not None
    assert np.allclose(ds1_payload["model_y"], model_series["ds1"]["A"])

    window.close()


def test_fitting_window_uses_pending_dataset_weight_on_immediate_run(qapp, monkeypatch):
    """Global fitting window must flush the visible dataset weight before starting a fit."""
    from kindred.core.analysis.dataset_parameter_overrides import split_fit_dataset_parameter_overrides
    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0, 1, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t_axis,
            "species_data": {"A": np.ones_like(t_axis)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "ds2",
            "t": t_axis,
            "species_data": {"A": np.full_like(t_axis, 2.0)},
            "selected_species": ["A"],
            "weight": 0.5,
            "include": True,
        },
    ]
    dataset_payloads = [
        {"id": "ds1", "t": t_axis, "y": np.vstack([np.ones_like(t_axis)]), "species": ["A"]},
        {"id": "ds2", "t": t_axis, "y": np.vstack([np.full_like(t_axis, 2.0)]), "species": ["A"]},
    ]
    dataset_params = {
        "ds1": {"init:A": 1.0},
        "ds2": {"init:A": 2.0},
    }

    captured = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            datasets,
            shared_params,
            *,
            dataset_overrides=None,
            dataset_params=None,
            dataset_variable_params=None,
            weights=None,
            **kwargs,
        ):
            super().__init__()
            captured["datasets"] = datasets
            captured["shared_params"] = shared_params
            captured["dataset_overrides"] = dataset_overrides
            captured["dataset_params"] = dataset_params
            captured["dataset_variable_params"] = dataset_variable_params
            captured["weights"] = weights

        def start(self):
            pass

        def isRunning(self):
            return False

        def cancel(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda params: {"t": t_axis, "species": {"A": np.exp(-params["k"] * t_axis)}},
        dataset_params=dataset_params,
        dataset_variable_params={},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0, "ds2": 0.5},
        apply_callback=lambda params: None,
    )
    top_tabs = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
    assert top_tabs is not None
    assert window._tabs is top_tabs
    dt_idx = [top_tabs.tabText(i) for i in range(top_tabs.count())].index("Data and Targets")
    window._tabs.setCurrentIndex(dt_idx)
    qapp.processEvents()

    ulist = window._data_targets_tab.unified_list._list
    weight_mode = window.findChild(QtWidgets.QComboBox, "global_fit_weight_mode_combo")
    weight_edit = window.findChild(QtWidgets.QLineEdit, "global_fit_dataset_weight_edit")
    assert weight_mode is not None
    assert weight_edit is not None

    for i in range(ulist.count()):
        item = ulist.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == "ds2":
            ulist.setCurrentRow(i)
            break
    else:
        raise AssertionError("ds2 not present in unified dataset list")
    qapp.processEvents()

    weight_mode.setCurrentIndex(1)
    weight_edit.setText("0.75")
    qapp.processEvents()

    window._start_fit()
    assert captured["dataset_params"] is None
    assert captured["dataset_variable_params"] is None
    dataset_params_forwarded, _variable_params_forwarded = split_fit_dataset_parameter_overrides(captured["dataset_overrides"])
    assert dataset_params_forwarded["ds1"]["init:A"] == pytest.approx(1.0)
    assert dataset_params_forwarded["ds2"]["init:A"] == pytest.approx(2.0)
    assert captured["weights"]["ds2"] == pytest.approx(0.75)
    window._set_running_state(False)
    window.close()


def test_fitting_window_applies_dataset_initial_updates(qapp):
    """After a fit completes, dataset-specific initials are staged until applied."""
    from kindred.gui.fitting.window import FittingWindow
    from kindred.core.analysis.global_fitting import GlobalFitResult, DatasetFitInfo

    t_axis = np.linspace(0, 1, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t_axis,
            "species_data": {"A": np.ones_like(t_axis)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "ds2",
            "t": t_axis,
            "species_data": {"A": np.full_like(t_axis, 2.0)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
    ]
    dataset_payloads = [
        {"id": "ds1", "t": t_axis, "y": np.vstack([np.ones_like(t_axis)]), "species": ["A"]},
        {"id": "ds2", "t": t_axis, "y": np.vstack([np.full_like(t_axis, 2.0)]), "species": ["A"]},
    ]
    parameter_defs = [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]
    dataset_params = {
        "ds1": {"init:A": 0.8},
        "ds2": {"init:A": 1.2},
    }
    dataset_variable_params = {
        "ds1": {"init:A": {"initial": 0.8, "min": 0.1, "max": 2.0}},
        "ds2": {"init:A": {"initial": 1.2, "min": 0.1, "max": 2.5}},
    }
    applied = []

    def _updater(ds_id, values):
        applied.append((ds_id, dict(values)))

    window = FittingWindow(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        simulation_func=lambda params: {"t": t_axis, "species": {"A": np.exp(-params["k"] * t_axis)}},
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0, "ds2": 1.0},
        apply_callback=lambda params: None,
        dataset_settings_updater=_updater,
    )

    result = GlobalFitResult(
        success=True,
        shared_params={"k": 0.25},
        dataset_params={"ds1": {"init:A": 0.6}, "ds2": {"init:A": 1.6}},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.9,
        dataset_info=[
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=np.zeros(5),
                n_points=5,
                weight=1.0,
            ),
            DatasetFitInfo(
                dataset_id="ds2",
                r_squared=0.9,
                chi_squared=0.1,
                rmse=0.1,
                mae=0.1,
                residuals=np.zeros(5),
                n_points=5,
                weight=1.0,
            ),
        ],
        nfev=10,
        message="ok",
        covariance=None,
        objective_residuals=np.zeros(10),
        model_series={
            "ds1": {"A": np.ones(5)},
            "ds2": {"A": np.ones(5)},
        },
        residual_series={
            "ds1": {"A": np.zeros(5)},
            "ds2": {"A": np.zeros(5)},
        },
    )

    window._handle_global_fit_complete({"result": result})
    assert window._params_ics_tab.get_staged_dataset_params()["ds1"]["init:A"] == pytest.approx(0.6)
    assert window._params_ics_tab.get_staged_dataset_params()["ds2"]["init:A"] == pytest.approx(1.6)
    assert applied == []
    window.close()


def test_global_fit_parameter_toggles_persist_across_best_updates_and_affect_config(qapp):
    """
    Regression: per-parameter Fit/Log10 toggles must not reset on best-updates and
    must be reflected in the next collected config.
    """
    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)
    window = FittingWindow(
        mode="global",
        parameter_defs=[
            {"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0},
            {"name": "k2", "value": 0.3, "min": 0.01, "max": 1.0},
        ],
        dataset_entries=[
            {
                "id": "demo",
                "label": "demo",
                "t": t_axis,
                "species_data": {"A": np.exp(-0.2 * t_axis)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda params: {"t": t_axis, "species": {"A": np.exp(-params["k1"] * t_axis)}},
        dataset_payloads=[{"id": "demo", "t": t_axis, "y": np.vstack([np.exp(-0.2 * t_axis)]), "species": ["A"]}],
        dataset_weights={"demo": 1.0},
        apply_callback=lambda params: None,
    )
    try:
        # Toggle: k1 uses log10; k2 excluded from fitting.
        window._params_ics_tab._param_table.item(0, 1).setCheckState(QtCore.Qt.Checked)   # Log10
        window._params_ics_tab._param_table.item(1, 0).setCheckState(QtCore.Qt.Unchecked)  # Fit

        # Simulate a best update which repopulates the parameter table.
        window._handle_global_best_update({"cost": 1.0, "shared_params": {"k1": 0.25, "k2": 0.35}})

        assert window._params_ics_tab._param_table.item(0, 1).checkState() == QtCore.Qt.Checked
        assert window._params_ics_tab._param_table.item(1, 0).checkState() == QtCore.Qt.Unchecked

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        assert set(config["parameters"].keys()) == {"k1"}
        assert config["log10_params"] == {"k1": True}
        assert "k2" in config["fixed_params"]
    finally:
        window.close()


def test_global_fit_fixed_params_are_passed_to_simulation_even_when_not_fitted(qt_app, monkeypatch):
    """Unchecked Fit params remain fixed at the table value via the simulation wrapper."""
    from PySide6 import QtCore as _QtCore

    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)
    seen = {}

    def _base_sim(params):
        seen.clear()
        seen.update(dict(params))
        return {"t": t_axis, "species": {"A": np.exp(-float(params.get("k1", 0.2)) * t_axis)}}

    captured = {}

    class _FakeWorker(_QtCore.QObject):
        progress = _QtCore.Signal(int, str)
        bestUpdated = _QtCore.Signal(dict)
        finished = _QtCore.Signal(dict)
        error = _QtCore.Signal(str)

        def __init__(self, datasets, shared_params, *, fit_evaluator=None, log10_params=None, **kwargs):
            super().__init__()
            captured["shared_params"] = dict(shared_params)
            captured["log10_params"] = dict(log10_params or {})
            captured["fit_evaluator"] = fit_evaluator

        def start(self):
            return None

        def isRunning(self):
            return False

        def cancel(self):
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[
            {"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0},
            {"name": "k2", "value": 0.3, "min": 0.01, "max": 1.0},
        ],
        dataset_entries=[
            {
                "id": "demo",
                "label": "demo",
                "t": t_axis,
                "species_data": {"A": np.exp(-0.2 * t_axis)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=_base_sim,
        dataset_payloads=[{"id": "demo", "t": t_axis, "y": np.vstack([np.exp(-0.2 * t_axis)]), "species": ["A"]}],
        dataset_weights={"demo": 1.0},
        apply_callback=lambda params: None,
    )
    try:
        # Exclude k2 from fitting but set its fixed value.
        window._params_ics_tab._param_table.item(1, 0).setCheckState(QtCore.Qt.Unchecked)
        window._params_ics_tab._param_table.item(1, 3).setText("9.87")

        config = window._params_ics_tab._collect_parameter_config()
        dataset_selection = window._collect_dataset_selection()
        window._start_global_fit(config, dataset_selection)

        assert captured["shared_params"] == {"k1": pytest.approx(0.2)}

        sim = captured["fit_evaluator"]
        assert callable(sim)
        sim({"k1": 0.2, "init:A": 1.0})
        assert float(seen.get("k2")) == pytest.approx(9.87)
    finally:
        window._set_running_state(False)
        window.close()


def test_global_fit_fixed_params_accept_evaluate_series_only_evaluator(qt_app, monkeypatch):
    from PySide6 import QtCore as _QtCore

    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)
    seen = {}

    class _EvaluateOnly:
        def evaluate_series(self, params):
            seen.clear()
            seen.update(dict(params))
            return {
                "t": t_axis,
                "species": {"A": np.exp(-float(params.get("k1", 0.2)) * t_axis)},
            }

    captured = {}

    class _FakeWorker(_QtCore.QObject):
        progress = _QtCore.Signal(int, str)
        bestUpdated = _QtCore.Signal(dict)
        finished = _QtCore.Signal(dict)
        error = _QtCore.Signal(str)

        def __init__(self, datasets, shared_params, *, fit_evaluator=None, log10_params=None, **kwargs):
            super().__init__()
            captured["shared_params"] = dict(shared_params)
            captured["log10_params"] = dict(log10_params or {})
            captured["fit_evaluator"] = fit_evaluator

        def start(self):
            return None

        def isRunning(self):
            return False

        def cancel(self):
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[
            {"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0},
            {"name": "k2", "value": 0.3, "min": 0.01, "max": 1.0},
        ],
        dataset_entries=[
            {
                "id": "demo",
                "label": "demo",
                "t": t_axis,
                "species_data": {"A": np.exp(-0.2 * t_axis)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=_EvaluateOnly(),
        dataset_payloads=[{"id": "demo", "t": t_axis, "y": np.vstack([np.exp(-0.2 * t_axis)]), "species": ["A"]}],
        dataset_weights={"demo": 1.0},
        apply_callback=lambda params: None,
    )
    try:
        window._params_ics_tab._param_table.item(1, 0).setCheckState(QtCore.Qt.Unchecked)
        window._params_ics_tab._param_table.item(1, 3).setText("9.87")

        config = window._params_ics_tab._collect_parameter_config()
        dataset_selection = window._collect_dataset_selection()
        window._start_global_fit(config, dataset_selection)

        sim = captured["fit_evaluator"]
        assert callable(sim)
        assert hasattr(sim, "evaluate_series")
        sim.evaluate_series({"k1": 0.2, "init:A": 1.0})
        assert float(seen.get("k2")) == pytest.approx(9.87)
    finally:
        window._set_running_state(False)
        window.close()


def test_global_fit_fixed_params_preserve_with_fixed_params_branch(qt_app):
    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)
    fixed_capture = {}
    seen = {}

    class _EvaluatorWithFixedParams:
        def evaluate_series(self, params):
            return {
                "t": t_axis,
                "species": {"A": np.exp(-float(params.get("k1", 0.2)) * t_axis)},
            }

        def with_fixed_params(self, fixed_params):
            fixed_capture.clear()
            fixed_capture.update(dict(fixed_params))

            class _EvaluateOnly:
                def evaluate_series(self_inner, params):
                    seen.clear()
                    seen.update(dict(params))
                    return {
                        "t": t_axis,
                        "species": {"A": np.exp(-float(params.get("k1", 0.2)) * t_axis)},
                    }

            return _EvaluateOnly()

    wrapped = FittingWindow._simulation_with_fixed_params(_EvaluatorWithFixedParams(), {"k2": 9.87})

    assert callable(wrapped)
    assert hasattr(wrapped, "evaluate_series")
    wrapped.evaluate_series({"k1": 0.2, "init:A": 1.0})
    assert fixed_capture == {"k2": 9.87}
    assert float(seen.get("k1")) == pytest.approx(0.2)


def test_global_fit_fixed_param_wrapper_fatals_configured_nonfinite_forwarded_key(qt_app):
    from kindred.core.exceptions import FitSimulationError
    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)

    def _simulate(_params):
        return {"t": t_axis, "species": {"A": np.ones_like(t_axis)}}

    wrapped = FittingWindow._simulation_with_fixed_params(_simulate, {"unused_fixed": float("nan")})

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        wrapped.evaluate_series({"k1": 0.2, "init:A": 1.0})

    assert getattr(exc_info.value, "details", {}).get("fatal") is True


def test_global_fit_run_prep_prunes_stale_and_unknown_dataset_params(qt_app):
    from kindred.gui.fitting.window import FittingWindow
    from kindred.core.simulation_preparation import PreparedSimulationMetadata

    t_axis = np.linspace(0.0, 1.0, 5)

    def _simulate(params):
        return {"t": t_axis, "species": {"A": np.ones_like(t_axis)}}

    _simulate._kindred_prepared_simulation_meta = PreparedSimulationMetadata(  # type: ignore[attr-defined]
        version=1,
        mechanism_text_sha256="abc",
        mechanism_text_len=3,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        temperature_K=298.15,
        solver_requested="LSODA",
        solver_normalized="LSODA",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t_axis,
                "species_data": {"A": np.ones_like(t_axis)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_params={
            "ds1": {
                "k1": 9.0,
                "init:A": 1.0,
                "init:Removed": float("nan"),
                "unknown_extra": float("inf"),
            }
        },
        dataset_variable_params={
            "ds1": {
                "init:Removed": {"initial": 1.0, "min": 0.0, "max": 2.0},
                "unknown_extra": {"initial": 1.0, "min": 0.0, "max": 2.0},
            }
        },
        simulation_func=_simulate,
        mechanism_species=["A"],
        dataset_payloads=[{"id": "ds1", "t": t_axis, "y": np.vstack([np.ones_like(t_axis)]), "species": ["A"]}],
        apply_callback=lambda params: None,
    )
    try:
        fixed = window._dataset_params_for_run(["ds1"], {"k1"}, {})
        variable = window._variable_params_for_run(["ds1"], {"k1"}, {})

        assert fixed == {"ds1": {"init:A": pytest.approx(1.0)}}
        assert variable == {}
    finally:
        window.close()


def test_global_fit_run_prep_uses_prepared_metadata_for_dataset_param_pruning(qt_app):
    from kindred.core.simulation_preparation import PreparedSimulationMetadata
    from kindred.gui.fitting.window import FittingWindow

    t_axis = np.linspace(0.0, 1.0, 5)

    def _simulate(params):
        return {"t": t_axis, "species": {"A": np.ones_like(t_axis)}}

    _simulate._kindred_prepared_simulation_meta = PreparedSimulationMetadata(  # type: ignore[attr-defined]
        version=1,
        mechanism_text_sha256="abc",
        mechanism_text_len=3,
        param_names=["prepared_only"],
        t_end=1.0,
        num_points=5,
        temperature_K=298.15,
        solver_requested="LSODA",
        solver_normalized="LSODA",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "table_only", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t_axis,
                "species_data": {"A": np.ones_like(t_axis)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=_simulate,
        mechanism_species=["A"],
        dataset_params={
            "ds1": {
                "prepared_only": 2.0,
                "table_only": 3.0,
                "init:A": 1.0,
                "init:Removed": float("nan"),
            }
        },
        dataset_variable_params={
            "ds1": {
                "prepared_only": {"initial": 2.0, "min": 0.0, "max": 10.0},
                "table_only": {"initial": 3.0, "min": 0.0, "max": 10.0},
                "init:Removed": {"initial": 1.0, "min": 0.0, "max": 2.0},
            }
        },
        dataset_payloads=[{"id": "ds1", "t": t_axis, "y": np.vstack([np.ones_like(t_axis)]), "species": ["A"]}],
        apply_callback=lambda params: None,
    )
    try:
        fixed = window._dataset_params_for_run(["ds1"], {"shared_only"}, {})
        variable = window._variable_params_for_run(["ds1"], {"shared_only"}, {})
        stripped = window._dataset_params_for_run(["ds1"], {"prepared_only"}, {})

        assert fixed == {"ds1": {"prepared_only": pytest.approx(2.0), "init:A": pytest.approx(1.0)}}
        assert set(variable["ds1"]) == {"prepared_only"}
        assert variable["ds1"]["prepared_only"]["initial"] == pytest.approx(2.0)
        assert stripped == {"ds1": {"init:A": pytest.approx(1.0)}}
    finally:
        window.close()

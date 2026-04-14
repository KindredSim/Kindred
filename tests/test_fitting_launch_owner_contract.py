from __future__ import annotations

from dataclasses import replace
import importlib.resources
from types import SimpleNamespace
import numpy as np
import pytest


def _seed_one_dataset(main_window) -> None:
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    t = np.linspace(0.0, 1.0, 6)
    data_panel._datasets["ds1"] = {
        "t": t.copy(),
        "species": {
            "A": np.linspace(1.0, 0.5, t.size),
            "B": np.linspace(0.0, 0.4, t.size),
        },
    }


def _seed_two_datasets(main_window) -> None:
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    t = np.linspace(0.0, 1.0, 6)
    data_panel._datasets["ds1"] = {
        "t": t.copy(),
        "species": {
            "A": np.linspace(1.0, 0.5, t.size),
            "B": np.linspace(0.0, 0.4, t.size),
        },
    }
    data_panel._datasets["ds2"] = {
        "t": t.copy(),
        "species": {
            "A": np.linspace(0.8, 0.3, t.size),
            "B": np.linspace(0.1, 0.5, t.size),
        },
    }


def _seed_simple_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )


@pytest.mark.gui
def test_fitting_mixin_run_global_fit_delegates_to_launch_owner(main_window, monkeypatch):
    from kindred.gui.fitting import GlobalFitLaunchContext

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    def fake_launch(context):
        captured["context"] = context
        return None

    monkeypatch.setattr("kindred.gui.mixins.fitting_mixin.launch_global_fit_session", fake_launch, raising=False)

    try:
        main_window._run_global_fit()
    finally:
        for window in list(getattr(main_window, "_active_fit_windows", []) or []):
            window.close()

    context = captured.get("context")
    assert isinstance(context, GlobalFitLaunchContext)
    assert context.parent is main_window
    assert context.dataset_manager is main_window._dataset_manager
    assert callable(context.reactions_text_getter)
    assert callable(context.reactions_text_setter)


@pytest.mark.gui
def test_fitting_package_launch_owner_builds_window_payloads(main_window, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.gui.fitting import launch_global_fit_session

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    context = replace(main_window._build_global_fit_launch_context(), window_factory=_FakeWindow)
    window = launch_global_fit_session(context)
    assert isinstance(window, QtWidgets.QDialog)
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert "simulation_func" in kwargs
    assert isinstance(kwargs["simulation_func"], SerialFittingEvaluator)
    prepared_payload = kwargs["simulation_func"].context.execution_request.to_payload()["prepared_payload"]
    assert isinstance(prepared_payload, dict)
    assert "rhs" not in prepared_payload
    assert kwargs.get("dataset_payloads")
    assert kwargs["dataset_payloads"][0]["id"] == "ds1"
    assert callable(kwargs.get("project_apply_callback"))


@pytest.mark.gui
def test_fitting_package_launch_owner_uses_serial_evaluator(
    qt_app,
    main_window,
    monkeypatch,
):
    from PySide6 import QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.gui.fitting import launch_global_fit_session

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    context = replace(main_window._build_global_fit_launch_context(), window_factory=_FakeWindow)
    window = launch_global_fit_session(context)
    try:
        assert isinstance(window, QtWidgets.QDialog)
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)

        simulation_func = kwargs.get("simulation_func")
        assert isinstance(simulation_func, SerialFittingEvaluator)
        assert type(simulation_func) is SerialFittingEvaluator

        fixed = simulation_func.with_fixed_params({"k_fixed": 1.23})
        assert isinstance(fixed, SerialFittingEvaluator)
        assert type(fixed) is SerialFittingEvaluator

        simulation_builder = kwargs.get("simulation_builder")
        assert callable(simulation_builder)
        rebuilt = simulation_builder(
            "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
            ["k1"],
            solver="BDF",
            rtol=1e-6,
            atol=1e-12,
        )
        assert isinstance(rebuilt, SerialFittingEvaluator)
        assert type(rebuilt) is SerialFittingEvaluator
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_package_launch_owner_preserves_serial_evaluator_through_worker_handoff(
    qt_app,
    main_window,
    monkeypatch,
):
    from PySide6 import QtCore, QtWidgets

    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.gui.fitting import launch_global_fit_session
    from kindred.gui.fitting.window import FittingWindow

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}
    launch_kwargs: dict[str, object] = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)
        bestUpdated = QtCore.Signal(dict)

        def __init__(self, datasets, shared_params, *, fit_evaluator=None, **kwargs):
            super().__init__()
            captured["datasets"] = list(datasets)
            captured["shared_params"] = dict(shared_params)
            captured["fit_evaluator"] = fit_evaluator

        def start(self):
            return

        def isRunning(self):
            return False

        def cancel(self):
            return

    class _CaptureWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            launch_kwargs.update(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = launch_global_fit_session(replace(main_window._build_global_fit_launch_context(), window_factory=_CaptureWindow))
    assert isinstance(window, QtWidgets.QDialog)
    eager_window = FittingWindow(**launch_kwargs)
    try:
        config = eager_window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = eager_window._collect_dataset_selection()
        assert selection["ids"] == ["ds1"]
        fixed_params = eager_window._fixed_params_for_run(config)
        fit_evaluator = eager_window._simulation_with_fixed_params(eager_window._simulation_func, fixed_params)
        assert isinstance(fit_evaluator, SerialFittingEvaluator)
        assert type(fit_evaluator) is SerialFittingEvaluator

        t = np.linspace(0.0, 1.0, 6)
        eager_window._start_global_fit_worker(
            datasets=[
                FitDatasetSpec(
                    dataset_id="ds1",
                    t_exp=t,
                    species_list=["A"],
                    y_matrix=np.vstack([np.ones_like(t)]),
                    point_count=int(t.size),
                    x_name="t",
                    x_obs=None,
                    x_mode="auto",
                )
            ],
            config=config,
            dataset_overrides=[],
            weights=eager_window._weights_for_run(selection),
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=fit_evaluator,
            stamp={},
            stamp_hash="stamp-hash",
            stamp_short="stamp",
        )

        captured_fit_evaluator = captured.get("fit_evaluator")
        assert isinstance(captured_fit_evaluator, SerialFittingEvaluator)
        assert type(captured_fit_evaluator) is SerialFittingEvaluator
    finally:
        eager_window.close()
        eager_window.deleteLater()
        window.close()
        window.deleteLater()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_package_launch_owner_routes_multi_dataset_gui_fit_to_process_pool(
    qtbot,
    qt_app,
    main_window,
    monkeypatch,
):
    from types import SimpleNamespace

    from PySide6 import QtWidgets

    from kindred.core.analysis import global_fitting
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.core.api.fitting import fit_global
    import kindred.core.fitting_optimization as fitting_optimization
    from kindred.gui.fitting import launch_global_fit_session
    from kindred.gui.fitting.window import FittingWindow

    _seed_two_datasets(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    launch_kwargs: dict[str, object] = {}
    captured: dict[str, object] = {}
    state = {"objective_calls": 0}
    original_assemble = global_fitting._assemble_global_fit_result

    class _CaptureWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            launch_kwargs.update(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    def fake_least_squares(fun, x0, **kwargs):
        first_residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        state["objective_calls"] += 1
        second_residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        state["objective_calls"] += 1
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            success=True,
            message="ok",
            nfev=2,
            fun=second_residuals,
            jac=np.eye(len(first_residuals), len(np.asarray(x0, dtype=float))),
        )

    def fake_de(*_args, **_kwargs):
        raise AssertionError("DE should not be used for this test")

    def counting_assemble(*args, **kwargs):
        process_pool = kwargs["process_pool"]
        captured["process_pool_workers"] = None if process_pool is None else int(process_pool.max_workers)
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", lambda: (fake_least_squares, fake_de))
    monkeypatch.setattr(global_fitting, "_assemble_global_fit_result", counting_assemble)

    window = launch_global_fit_session(replace(main_window._build_global_fit_launch_context(), window_factory=_CaptureWindow))
    assert isinstance(window, QtWidgets.QDialog)
    eager_window = FittingWindow(**launch_kwargs)
    try:
        config = eager_window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = eager_window._collect_dataset_selection()
        assert selection["ids"] == ["ds1", "ds2"]

        fixed_params = eager_window._fixed_params_for_run(config)
        fit_evaluator = eager_window._simulation_with_fixed_params(eager_window._simulation_func, fixed_params)
        eager_window._fit_func = fit_global
        eager_window._start_global_fit_worker(
            datasets=coerce_fit_dataset_specs(list(launch_kwargs["dataset_payloads"])),
            config=config,
            dataset_overrides=[],
            weights=eager_window._weights_for_run(selection),
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=fit_evaluator,
            stamp={},
            stamp_hash="stamp-hash",
            stamp_short="stamp",
        )
        assert eager_window._worker is not None
        captured["worker_fit_evaluator"] = eager_window._worker._fit_evaluator
        qtbot.waitUntil(lambda: "process_pool_workers" in captured, timeout=5000)
        qtbot.waitUntil(
            lambda: eager_window._worker is None or not eager_window._worker.isRunning(),
            timeout=5000,
        )
        qtbot.waitUntil(lambda: eager_window._last_result is not None, timeout=5000)
        result = eager_window._last_result

        assert result.success is True
        assert state["objective_calls"] == 2
        assert len(selection["ids"]) == 2
        assert captured["worker_fit_evaluator"] is fit_evaluator
        assert captured["process_pool_workers"] == 2
    finally:
        eager_window.close()
        eager_window.deleteLater()
        window.close()
        window.deleteLater()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_package_launch_owner_preserves_invalid_payload_results(main_window, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.gui.fitting import launch_global_fit_session

    _seed_one_dataset(main_window)
    _seed_simple_mechanism(main_window)
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    captured: dict[str, object] = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    monkeypatch.setattr(
        "kindred.gui.fitting.launch._coerce_dataset_payload",
        lambda **_kwargs: SimpleNamespace(
            state="invalid",
            payload=None,
            error="Dataset 'ds1' has invalid x_obs for X='X'.",
        ),
    )

    context = replace(main_window._build_global_fit_launch_context(), window_factory=_FakeWindow)
    window = launch_global_fit_session(context)
    assert isinstance(window, QtWidgets.QDialog)
    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("dataset_payloads") == []
    payload_results = kwargs.get("dataset_payload_results")
    assert isinstance(payload_results, dict)
    assert payload_results["ds1"].state == "invalid"
    assert "invalid x_obs" in str(payload_results["ds1"].error)


def test_fitting_launch_owner_module_does_not_import_fit_dialog() -> None:
    source = importlib.resources.files("kindred.gui.fitting").joinpath("launch.py").read_text(encoding="utf-8")

    assert "from kindred.gui.fit_dialog import" not in source
    assert "import kindred.gui.fit_dialog" not in source

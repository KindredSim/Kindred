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
    from kindred.core.simulation_plan import SimulationAlgebraPolicy
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
    assert kwargs.get("simulation_func") is None
    simulation_builder = kwargs.get("simulation_builder")
    assert callable(simulation_builder)
    evaluator = simulation_builder(
        "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        ["k1"],
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
    )
    assert isinstance(evaluator, SerialFittingEvaluator)
    assert evaluator.context.simulation_plan.execution_mode == "fitting"
    assert evaluator.context.simulation_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
    prepared_payload = evaluator.context.execution_request.to_payload()["prepared_payload"]
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
    from kindred.core.simulation_plan import SimulationAlgebraPolicy
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
        assert simulation_func is None

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
        assert rebuilt.context.simulation_plan.execution_mode == "fitting"
        assert rebuilt.context.simulation_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
        fixed = rebuilt.with_fixed_params({"k_fixed": 1.23})
        assert isinstance(fixed, SerialFittingEvaluator)
        assert type(fixed) is SerialFittingEvaluator
        assert fixed.context.simulation_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()


@pytest.mark.gui
def test_fitting_launch_deferred_builder_uses_frozen_runtime_settings(main_window, monkeypatch):
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

    allow_live_solver_getter = {"value": True}
    solver_getter_calls: list[str] = []
    captured: dict[str, object] = {}

    def _solver_settings():
        solver_getter_calls.append("called")
        if not allow_live_solver_getter["value"]:
            raise AssertionError("deferred fitting builder must not read live solver settings")
        return {
            "solver": "BDF",
            "rtol": 1e-6,
            "atol": 1e-12,
            "use_sparse_jacobian": False,
            "wegscheider_cyclicity_enabled": False,
        }

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = dict(kwargs)

        def setWindowTitle(self, *_args):
            return None

    context = replace(
        main_window._build_global_fit_launch_context(),
        get_solver_settings=_solver_settings,
        window_factory=_FakeWindow,
    )
    window = launch_global_fit_session(context)
    try:
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        runtime_settings_getter = kwargs.get("runtime_settings_getter")
        assert callable(runtime_settings_getter)
        runtime_settings = runtime_settings_getter()
        simulation_builder = kwargs.get("simulation_builder")
        assert callable(simulation_builder)

        allow_live_solver_getter["value"] = False
        rebuilt = simulation_builder(
            "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
            ["k1"],
            solver="BDF",
            rtol=1e-6,
            atol=1e-12,
            temperature_K=float(runtime_settings["temperature_K"]),
            use_sparse_jacobian=bool(runtime_settings["use_sparse_jacobian"]),
            wegscheider_cyclicity_enabled=bool(runtime_settings["wegscheider_cyclicity_enabled"]),
        )

        assert isinstance(rebuilt, SerialFittingEvaluator)
        assert solver_getter_calls
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.gui
def test_fitting_package_launch_owner_preserves_serial_evaluator_through_worker_handoff(
    qt_app,
    qtbot,
    main_window,
    monkeypatch,
):
    from PySide6 import QtCore, QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.core.simulation_plan import SimulationAlgebraPolicy
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

    class _RuntimeSession:
        def __init__(self) -> None:
            self.ready = False

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            self.ready = True

        def close(self, *, kill: bool = False) -> None:
            return None

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
    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        lambda _evaluator, *, max_lanes, ledger=None: _RuntimeSession(),
    )

    window = launch_global_fit_session(replace(main_window._build_global_fit_launch_context(), window_factory=_CaptureWindow))
    assert isinstance(window, QtWidgets.QDialog)
    eager_window = FittingWindow(**launch_kwargs)
    try:
        config = eager_window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = eager_window._collect_dataset_selection()
        assert selection["ids"] == ["ds1"]
        assert eager_window._simulation_func is None

        eager_window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        eager_window._on_targets_applied()
        qtbot.waitUntil(lambda: eager_window._run_button.isEnabled(), timeout=3000)
        eager_window._start_fit()

        captured_fit_evaluator = captured.get("fit_evaluator")
        assert isinstance(captured_fit_evaluator, SerialFittingEvaluator)
        assert type(captured_fit_evaluator) is SerialFittingEvaluator
        assert (
            captured_fit_evaluator.context.execution_request
            is captured_fit_evaluator.context.simulation_plan.execution_request
        )
        assert (
            captured_fit_evaluator.context.simulation_plan.algebra_policy
            is SimulationAlgebraPolicy.FITTING_STRICT
        )
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


@pytest.mark.unit
def test_fitting_launch_owner_module_does_not_import_fit_dialog() -> None:
    source = importlib.resources.files("kindred.gui.fitting").joinpath("launch.py").read_text(encoding="utf-8")

    assert "from kindred.gui.fit_dialog import" not in source
    assert "import kindred.gui.fit_dialog" not in source

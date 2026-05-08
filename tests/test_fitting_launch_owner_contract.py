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
def test_fitting_window_base_evaluator_state_is_owned_outside_simulation_func_field(qt_app):
    from kindred.gui.fitting.window import FittingWindow
    from kindred.gui.fitting.launch import FittingLaunchIdentityOwner
    from kindred.gui.fitting.run_command import FittingRunCommandOwner
    from kindred.gui.fitting.run_state import FittingRunStateOwner
    from kindred.gui.fitting.runtime_preparation import FittingRuntimePreparationOwner
    from kindred.gui.fitting.worker_launch import FittingAcceptedLaunchWorkerOwner

    t = np.asarray([0.0, 1.0], dtype=float)

    def _base_evaluator(_params):
        return {"t": t.copy(), "species": {"A": np.asarray([1.0, 0.8], dtype=float)}}

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {"A": np.asarray([1.0, 0.8], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"}
        ],
        mechanism_species=["A"],
        simulation_func=_base_evaluator,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        assert not hasattr(type(window), "_build_current_fit_runtime_identity")
        assert not hasattr(type(window), "build_current_fit_runtime_identity")
        assert not hasattr(type(window), "_start_fit")
        assert not hasattr(type(window), "_start_accepted_fit_worker")
        assert not hasattr(type(window), "_start_accepted_fit_runtime_launch")
        assert not hasattr(type(window), "start_accepted_fit_worker")
        assert callable(getattr(window, "run_fit", None))
        assert getattr(window, "fit_launch_identity_owner", None) is not None
        assert isinstance(window.fit_launch_identity_owner, FittingLaunchIdentityOwner)
        assert getattr(window, "fit_run_command_owner", None) is not None
        assert isinstance(window.fit_run_command_owner, FittingRunCommandOwner)
        assert getattr(window, "fit_worker_launch_owner", None) is not None
        assert isinstance(window.fit_worker_launch_owner, FittingAcceptedLaunchWorkerOwner)
        assert getattr(window, "fit_runtime_readiness", None) is not None
        assert not hasattr(type(window), "_prepare_fit_runtime_for_current_state")
        assert not hasattr(type(window), "_poll_fit_runtime_preparation")
        assert not hasattr(window, "_fit_runtime_prepare_refresh_pending")
        assert not hasattr(window, "_close_after_fit_runtime_prepare")
        assert getattr(window, "fit_runtime_preparation_owner", None) is not None
        assert isinstance(window.fit_runtime_preparation_owner, FittingRuntimePreparationOwner)
        assert isinstance(window.fit_runtime_preparation_owner.refresh_pending, bool)
        assert window.fit_runtime_preparation_owner.close_after_prepare is False
        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert "_simulation_func" not in window.__dict__
        assert "_last_fit_config" not in window.__dict__
        assert "_active_fit_dataset_ids" not in window.__dict__
        assert "_active_fit_run_stamp_hash" not in window.__dict__
        assert "_active_fit_run_superseded" not in window.__dict__
        assert isinstance(window.fit_run_state_owner, FittingRunStateOwner)
        assert identity is not None
        assert identity.base_evaluator is _base_evaluator
    finally:
        window.close()


def test_fitting_sidecar_owners_use_public_owner_dependencies():
    from pathlib import Path

    package_root = Path("kindred/gui/fitting")
    for relative in ("run_command.py", "runtime_preparation.py"):
        source = (package_root / relative).read_text(encoding="utf-8")
        assert "._fit_runtime_readiness" not in source
        assert "._fit_launch_identity_owner" not in source
        assert "._fit_worker_launch_owner" not in source
    run_command_source = (package_root / "run_command.py").read_text(encoding="utf-8")
    assert "._collect_dataset_selection" not in run_command_source
    launch_source = (package_root / "launch.py").read_text(encoding="utf-8")
    assert "._collect_dataset_selection" not in launch_source
    assert "show_dataset_messages" not in launch_source


def test_fitting_workflow_tests_use_public_runtime_readiness_owner():
    from pathlib import Path

    offenders: list[str] = []
    for path in Path("tests").glob("test_global_fit*.py"):
        source = path.read_text(encoding="utf-8")
        if "._fit_runtime_readiness" in source:
            offenders.append(str(path))

    assert offenders == []


def test_run_fit_revalidates_explicit_launch_even_when_runtime_is_ready():
    from kindred.gui.fitting.launch import FittingLaunchPurpose, FittingLaunchResult
    from kindred.gui.fitting.run_command import FittingRunCommandOwner
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeLaunchDecisionState,
        FittingRuntimeReadinessState,
    )

    stale_identity = SimpleNamespace(name="stale")
    explicit_identity = SimpleNamespace(name="explicit")
    accepted_launch = SimpleNamespace(identity=explicit_identity)
    calls: list[object] = []

    class _Worker:
        def isRunning(self) -> bool:
            return False

    class _SpeciesTable:
        def flush_visible_weight_edits(self) -> None:
            calls.append("flush_visible")

        def flush_dataset_weight_editor(self) -> None:
            calls.append("flush_dataset")

    class _Readiness:
        def snapshot(self):
            return SimpleNamespace(
                state=FittingRuntimeReadinessState.READY,
                identity=stale_identity,
            )

        def is_ready_for(self, identity):
            calls.append(("ready_for", identity))
            return identity is stale_identity or identity is explicit_identity

        def set_blocked(self, *_args, **_kwargs) -> None:
            calls.append("blocked")

        def prepare_or_accept_launch(self, identity):
            calls.append(("accept", identity))
            return SimpleNamespace(
                state=FittingRuntimeLaunchDecisionState.ACCEPTED,
                accepted_launch=accepted_launch,
                snapshot=self.snapshot(),
            )

    class _LaunchOwner:
        def build_current_launch_result(self, *, purpose, **_kwargs):
            calls.append(("build", purpose))
            return FittingLaunchResult(identity=explicit_identity)

        def render_launch_rejection(self, *_args, **_kwargs) -> None:
            calls.append("reject")

    class _WorkerLaunchOwner:
        def start_runtime_launch(self, launch):
            calls.append(("start", launch))

    window = SimpleNamespace(
        _worker=_Worker(),
        _species_table=_SpeciesTable(),
        fit_runtime_readiness=_Readiness(),
        fit_launch_identity_owner=_LaunchOwner(),
        fit_worker_launch_owner=_WorkerLaunchOwner(),
        _reset_fit_run_cached_state=lambda: calls.append("reset"),
        _fit_runtime_identity_matches_current_noncollecting_inputs=lambda identity: identity is stale_identity,
        _capture_failed_fit_restore_baseline=lambda: calls.append("capture_baseline"),
        _clear_failed_run_visual_state=lambda: calls.append("clear_failed"),
        _refresh_run_button_enabled_state=lambda: calls.append("refresh_button"),
    )

    FittingRunCommandOwner(window).run_fit()

    assert ("build", FittingLaunchPurpose.EXPLICIT_RUN) in calls
    assert ("accept", explicit_identity) in calls
    assert ("start", accepted_launch) in calls
    assert ("accept", stale_identity) not in calls


@pytest.mark.gui
def test_fitting_window_routes_passive_and_explicit_launch_identity_through_owner(qt_app, monkeypatch):
    from kindred.gui.fitting.window import FittingWindow
    from kindred.gui.fitting.launch import FittingLaunchPurpose, FittingLaunchResult

    t = np.asarray([0.0, 1.0], dtype=float)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {"A": np.asarray([1.0, 0.8], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"}
        ],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.asarray([1.0, 0.8])}},
    )
    try:
        calls: list[FittingLaunchPurpose] = []

        def _capture_result(*, purpose, **_kwargs):
            calls.append(purpose)
            return FittingLaunchResult(identity=None)

        owner = window.fit_launch_identity_owner
        monkeypatch.setattr(owner, "build_current_launch_result", _capture_result)

        window.fit_runtime_preparation_owner.prepare_current_state()
        window.run_fit()

        assert calls == [FittingLaunchPurpose.PASSIVE_READINESS, FittingLaunchPurpose.EXPLICIT_RUN]
    finally:
        window.close()


@pytest.mark.gui
def test_fitting_launch_validation_result_owns_payload_errors_without_window_split(qt_app, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.core.analysis.fit_dataset_payload import FitDatasetPayloadResult
    from kindred.gui.fitting.launch import FittingLaunchPurpose
    from kindred.gui.fitting.window import FittingWindow

    t = np.asarray([0.0, 1.0], dtype=float)
    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, text, *args, **kwargs: captured.append((str(title), str(text)))
        or QtWidgets.QMessageBox.StandardButton.Ok,
    )
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {"A": np.asarray([1.0, 0.8], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"}
        ],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.asarray([1.0, 0.8])}},
    )
    try:
        assert not hasattr(FittingWindow, "_datasets_payloads_for_run")
        assert not hasattr(FittingWindow, "_datasets_payloads_for_readiness")
        window._global_payload_results["ds1"] = FitDatasetPayloadResult.invalid("invalid payload")

        passive = window.fit_launch_identity_owner.build_current_launch_result(
            purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            refresh_current_mechanism=False,
        )
        explicit = window.fit_launch_identity_owner.build_current_launch_result(
            purpose=FittingLaunchPurpose.EXPLICIT_RUN,
            refresh_current_mechanism=False,
        )

        assert passive.identity is None
        assert explicit.identity is None
        assert passive.rejection is not None
        assert explicit.rejection == passive.rejection
        assert captured == []

        window.fit_launch_identity_owner.render_launch_rejection(
            explicit,
            purpose=FittingLaunchPurpose.EXPLICIT_RUN,
        )
        assert captured
        assert captured[-1][0] == "Global Fit"
        assert "invalid payload" in captured[-1][1]
    finally:
        window.close()


@pytest.mark.gui
def test_fitting_mixin_run_global_fit_launches_window_through_fitting_owner(main_window, monkeypatch):
    from PySide6 import QtWidgets

    from kindred.core.fitting_evaluation import SerialFittingEvaluator

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

        def setWindowTitle(self, title):
            captured["title"] = str(title)

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)

    try:
        window = main_window._run_global_fit()
        assert isinstance(window, _FakeWindow)
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        assert kwargs.get("parent") is main_window
        assert kwargs.get("simulation_func") is None
        assert kwargs.get("dataset_manager") is main_window._dataset_manager
        assert kwargs.get("dataset_payloads")
        assert kwargs["dataset_payloads"][0]["id"] == "ds1"
        assert callable(kwargs.get("mechanism_text_getter"))
        assert callable(kwargs.get("reactions_text_getter"))
        assert callable(kwargs.get("reactions_text_setter"))
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
        assert str(captured.get("title") or "").startswith("Global Fit")
    finally:
        for window in list(getattr(main_window, "_active_fit_windows", []) or []):
            window.close()


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
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
            ]
        )
    )
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
            captured["worker_kwargs"] = dict(kwargs)

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

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FakeWorker)
    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        lambda _evaluator, *, max_lanes, ledger=None: _RuntimeSession(),
    )

    window = launch_global_fit_session(replace(main_window._build_global_fit_launch_context(), window_factory=_CaptureWindow))
    assert isinstance(window, QtWidgets.QDialog)
    eager_window = FittingWindow(**launch_kwargs)
    try:
        config = eager_window._params_ics_tab.collect_parameter_config()
        assert config is not None
        selection = eager_window.fit_launch_identity_owner.collect_dataset_selection()
        assert list(selection.ids) == ["ds1"]
        assert "_simulation_func" not in eager_window.__dict__
        assert eager_window._fit_evaluator_state.current_base_evaluator() is None

        eager_window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        eager_window._on_targets_applied()
        qtbot.waitUntil(lambda: eager_window._run_button.isEnabled(), timeout=3000)
        collect_calls = []
        original_collect = eager_window._params_ics_tab.collect_parameter_config_bundle

        def _collect_once_for_launch(*, show_errors: bool):
            collect_calls.append(bool(show_errors))
            return original_collect(show_errors=show_errors)

        monkeypatch.setattr(eager_window._params_ics_tab, "collect_parameter_config_bundle", _collect_once_for_launch)
        eager_window.run_fit()

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
        schedule = captured_fit_evaluator.context.execution_request.intervention_schedule
        assert schedule is not None
        assert schedule.to_payload()["instant_events"][0]["value"] == pytest.approx(2.0)
        worker_kwargs = captured.get("worker_kwargs")
        assert isinstance(worker_kwargs, dict)
        run_stamp = worker_kwargs["run_stamp"]
        assert run_stamp["prepared_simulation"]["intervention_schedule_fingerprint"] == schedule.fingerprint
        assert collect_calls == [True]
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

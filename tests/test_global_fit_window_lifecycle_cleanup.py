from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import threading
from unittest import mock

import pytest
import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.analysis.fit_dataset_payload import FitDatasetPayloadResult
from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.core.fitting_completion import FitDetailSection, FitDiagnostic, GlobalFitCompletion
from kindred.core.simulation_failure import build_simulation_failure
from kindred.gui.controllers.dataset_manager import DatasetManager
from kindred.gui.fitting.launch import FittingLaunchDatasetSelection
from kindred.gui.fitting.window import FittingWindow, _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS

pytestmark = pytest.mark.gui


def _process_deferred_deletes(iterations: int = 5) -> None:
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()
    with suppress(RuntimeError, TypeError):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()


def _build_completion(
    *,
    status: str,
    optimizer_converged: bool | None = None,
    nonfinite_metrics: bool = False,
    optimizer_diagnostic: FitDiagnostic | None = None,
    dataset_failures: dict[str, FitDiagnostic] | None = None,
    dataset_warnings: dict[str, str] | None = None,
    detail_sections: list[FitDetailSection] | None = None,
) -> GlobalFitCompletion:
    if optimizer_converged is None:
        optimizer_converged = status == "ok"
    return GlobalFitCompletion(
        status=status,
        optimizer_converged=optimizer_converged,
        nonfinite_metrics=nonfinite_metrics,
        optimizer_diagnostic=optimizer_diagnostic,
        dataset_failures=dataset_failures or {},
        dataset_warnings=dataset_warnings or {},
        detail_sections=detail_sections or [],
    )


def _make_diagnostic(
    *,
    phase: str,
    dataset_id: str | None = None,
    message: str,
    stack_trace: str | None = None,
    remediation: str | None = None,
    failure_kind: str = "simulation_error",
    details: dict[str, object] | None = None,
) -> FitDiagnostic:
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=build_simulation_failure(
            kind=failure_kind,
            message=message,
            context={"stack_trace": stack_trace} if stack_trace is not None else None,
            details=details,
        ),
        remediation=remediation,
    )


def _make_detail_section(*, dataset_id: str | None = None, message: str, stack_trace: str) -> FitDetailSection:
    return FitDetailSection(
        dataset_id=dataset_id,
        failure=build_simulation_failure(
            kind="simulation_error",
            message=message,
            context={"stack_trace": stack_trace},
        ),
    )


def _build_success_result(*, dataset_id: str = "ds1", param_name: str = "k", value: float = 1.0) -> GlobalFitResult:
    model = np.asarray([1.0, 0.8, 0.6], dtype=float)
    return GlobalFitResult(
        shared_params={str(param_name): float(value)},
        dataset_params={str(dataset_id): {}},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.0,
        dataset_info=[
            DatasetFitInfo(
                dataset_id=str(dataset_id),
                r_squared=0.0,
                chi_squared=1.0,
                rmse=1.0,
                mae=1.0,
                residuals=np.asarray([0.0], dtype=float),
                n_points=1,
                weight=1.0,
            )
        ],
        nfev=1,
        message="ok",
        completion=_build_completion(status="ok", optimizer_converged=True),
        covariance=None,
        objective_residuals=np.asarray([0.0], dtype=float),
        model_series={str(dataset_id): {"A": model}},
        residual_series={str(dataset_id): {"A": np.asarray([0.0, 0.0, 0.0], dtype=float)}},
    )


def _build_completion_result(
    *,
    status: str,
    optimizer_converged: bool | None = None,
    dataset_id: str = "ds1",
    param_name: str = "k",
    value: float = 1.0,
    message: str | None = None,
) -> GlobalFitResult:
    result = _build_success_result(dataset_id=dataset_id, param_name=param_name, value=value)
    result.message = str(message or ("ok" if status == "ok" else "failed"))
    completion_kwargs: dict[str, object] = {
        "status": status,
        "optimizer_converged": optimizer_converged,
    }
    if status == "fail":
        completion_kwargs["optimizer_diagnostic"] = _make_diagnostic(
            phase="fatal",
            message=str(message or "failed"),
            remediation="generic_retry",
        )
    result.completion = _build_completion(**completion_kwargs)
    return result


def _build_window(
    *,
    dataset_entries: list[dict[str, object]] | None = None,
    dataset_payloads: list[dict[str, object]] | None = None,
    simulation_func=Ellipsis,
) -> FittingWindow:
    if dataset_entries is None:
        dataset_entries = [
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ]
    if dataset_payloads is None:
        dataset_payloads = [
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ]
    if simulation_func is Ellipsis:
        def simulation_func(_params):
            return {"t": np.asarray([0.0, 1.0, 2.0]), "species": {"A": np.asarray([1.0, 0.8, 0.6])}}
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=list(dataset_entries),
        dataset_payloads=list(dataset_payloads),
        mechanism_species=["A"],
        simulation_func=simulation_func,
    )


def _build_dataset_variable_window(*, simulation_func=None) -> FittingWindow:
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_variable_params={
            "ds1": {
                "init:A": {
                    "initial": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "log10": False,
                }
            }
        },
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        simulation_func=simulation_func or _basic_serial_fitting_evaluator(),
    )


def _basic_mechanism_text() -> str:
    return "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )


def _basic_serial_fitting_evaluator(*, solver: str = "BDF", rtol: float = 1e-6, atol: float = 1e-12):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    context = prepare_fitting_execution_context(
        mechanism_text=_basic_mechanism_text(),
        param_names=["k"],
        t_end=2.0,
        num_points=3,
        solver=str(solver),
        rtol=float(rtol),
        atol=float(atol),
        initial_prefix="initial:",
    )
    return SerialFittingEvaluator(context)


class _DatasetPanelSignal:
    def connect(self, _callback) -> None:
        return None


class _DatasetPanel:
    def __init__(self) -> None:
        self.simulateRequested = _DatasetPanelSignal()


class _DatasetPlotTabs:
    def __init__(self) -> None:
        self.grid_payload = []

    def sync_dataset_tab(self, _name: str, **_dataset_payload):
        return _DatasetPanel()

    def sync_dataset_grid(self, dataset_entries) -> None:
        self.grid_payload = list(dataset_entries)

    def remove_dataset_tab(self, _name: str) -> None:
        return None


class _SignalWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    bestUpdated = QtCore.Signal(dict)
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(object)

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self._running = True
        self.cancel_called = False

    def start(self) -> None:
        return None

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_called = True


def _start_worker_from_accepted_launch(
    window,
    *,
    datasets,
    config,
    dataset_overrides=None,
    weights=None,
    requested_solver="BDF",
    requested_rtol=1e-6,
    requested_atol=1e-12,
    fit_evaluator=None,
    stamp=None,
    stamp_hash="test-run",
    stamp_short=None,
    runtime_session=None,
):
    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.core.analysis.dataset_parameter_overrides import coerce_fit_dataset_parameter_overrides
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, coerce_fitting_series_evaluator
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeAcceptedLaunch, FittingRuntimeIdentity

    raw_datasets = list(datasets or [])
    if not raw_datasets:
        raw_datasets = [dict(value) for value in getattr(window, "_global_payload_lookup", {}).values()]
    dataset_specs = tuple(coerce_fit_dataset_specs(raw_datasets))
    overrides = tuple(
        coerce_fit_dataset_parameter_overrides(
            dataset_ids=[spec.dataset_id for spec in dataset_specs],
            dataset_overrides=list(dataset_overrides or []),
        )
    )
    evaluator = fit_evaluator if fit_evaluator is not None else (lambda _params: {})
    try:
        readiness_required = type(coerce_fitting_series_evaluator(evaluator)) is SerialFittingEvaluator
    except Exception:
        readiness_required = False
    identity = FittingRuntimeIdentity(
        datasets=dataset_specs,
        config=dict(config or {}),
        dataset_overrides=overrides,
        weights=dict(weights) if isinstance(weights, dict) else None,
        requested_solver=str(requested_solver),
        requested_rtol=float(requested_rtol),
        requested_atol=float(requested_atol),
        fit_evaluator=evaluator,
        stamp=dict(stamp or {}),
        stamp_hash=str(stamp_hash or ""),
        stamp_short=str(stamp_short or stamp_hash or "")[:12],
        lane_count=window._fit_runtime_lane_budget(len(dataset_specs)),
        readiness_required=bool(readiness_required),
    )
    window.fit_worker_launch_owner.start_worker(FittingRuntimeAcceptedLaunch(identity=identity, session=runtime_session))


def test_passive_fit_runtime_preparation_builds_deferred_evaluator_before_run(
    qt_app,
    qtbot,
    monkeypatch,
):
    class _ReadyRuntimeSession:
        def __init__(self) -> None:
            self.warm_calls: list[int | None] = []

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            self.warm_calls.append(lane_count)

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.warm_calls)

        def close(self, *, kill: bool = False) -> None:
            return None

    sessions: list[_ReadyRuntimeSession] = []
    builder_calls: list[tuple[str, tuple[str, ...], str, float, float]] = []

    def _session_factory(_evaluator, *, max_lanes, ledger=None):
        session = _ReadyRuntimeSession()
        sessions.append(session)
        return session

    def _simulation_builder(mechanism_text, param_names, *, solver=None, rtol=None, atol=None):
        builder_calls.append(
            (
                str(mechanism_text),
                tuple(str(name) for name in param_names),
                str(solver),
                float(rtol),
                float(atol),
            )
        )
        return _basic_serial_fitting_evaluator(solver=str(solver), rtol=float(rtol), atol=float(atol))

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        _session_factory,
    )
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=lambda: "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=_simulation_builder,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._on_targets_applied()
        window.show()
        qtbot.waitUntil(lambda: bool(sessions) and sessions[-1].is_ready(), timeout=3000)

        assert builder_calls
        assert window._run_button.isEnabled() is True

        warm_count = len(sessions[-1].warm_calls)
        window.run_fit()

        assert len(sessions[-1].warm_calls) == warm_count
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_runtime_readiness_accepts_infinite_bounds_for_non_de_methods(qt_app):
    window = _build_window()
    try:
        table = window._params_ics_tab._param_table
        table.item(0, 4).setText("-inf")
        table.item(0, 5).setText("inf")
        window._params_ics_tab._method_combo.setCurrentText("trf")

        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert identity is not None
        assert identity.config["bounds"]["k"] == (float("-inf"), float("inf"))
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_runtime_readiness_still_blocks_de_infinite_bounds(qt_app):
    window = _build_window()
    try:
        table = window._params_ics_tab._param_table
        table.item(0, 4).setText("-inf")
        table.item(0, 5).setText("inf")
        window._params_ics_tab._method_combo.setCurrentText("differential_evolution")

        assert window.fit_launch_identity_owner.build_current_fit_runtime_identity() is None
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_runtime_readiness_rejects_nonfinite_initial_value(qt_app):
    window = _build_window()
    try:
        table = window._params_ics_tab._param_table
        table.item(0, 3).setText("inf")
        table.item(0, 4).setText("-inf")
        table.item(0, 5).setText("inf")
        window._params_ics_tab._method_combo.setCurrentText("trf")

        assert window.fit_launch_identity_owner.build_current_fit_runtime_identity() is None
    finally:
        window.close()
        qt_app.processEvents()


def test_explicit_parameter_collection_rejects_nonfinite_initial_value(qt_app, monkeypatch):
    window = _build_window()
    warnings: list[tuple[str, str]] = []

    def _capture_warning(_parent, title, message, *_args, **_kwargs):
        warnings.append((str(title), str(message)))
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_capture_warning))
    try:
        table = window._params_ics_tab._param_table
        table.item(0, 3).setText("inf")
        table.item(0, 4).setText("-inf")
        table.item(0, 5).setText("inf")

        assert window._params_ics_tab.collect_parameter_config() is None
        assert warnings == [("Invalid Parameter", "Parameter 'k' initial value must be finite.")]
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_runtime_readiness_still_blocks_de_infinite_dataset_bounds(qt_app):
    window = _build_dataset_variable_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        table = window._params_ics_tab._param_table
        table.item(0, 0).setCheckState(QtCore.Qt.Unchecked)
        dataset_row = 1
        table.item(dataset_row, 4).setText("-inf")
        table.item(dataset_row, 5).setText("inf")
        window._params_ics_tab._method_combo.setCurrentText("differential_evolution")

        assert window.fit_launch_identity_owner.build_current_fit_runtime_identity() is None
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_runtime_readiness_ignores_de_infinite_dataset_bounds_for_excluded_dataset(qt_app):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "Dataset 2",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([1.0, 0.7, 0.4], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": False,
        },
    ]
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_variable_params={
            "ds1": {
                "init:A": {"initial": 1.0, "min": 0.1, "max": 10.0, "log10": False},
            },
            "ds2": {
                "init:A": {"initial": 1.0, "min": float("-inf"), "max": float("inf"), "log10": False},
            },
        },
        dataset_entries=dataset_entries,
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.7, 0.4]), "species": "A"},
        ],
        mechanism_species=["A"],
        mechanism_text_getter=_basic_mechanism_text,
        simulation_func=_basic_serial_fitting_evaluator(),
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._species_table._fit_targets_selection_applied["ds2"] = ["A"]
        table = window._params_ics_tab._param_table
        table.item(0, 0).setCheckState(QtCore.Qt.Unchecked)
        window._params_ics_tab._method_combo.setCurrentText("differential_evolution")

        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert identity is not None
        assert [override.dataset_id for override in identity.dataset_overrides] == ["ds1"]
        assert "ds2" not in identity.stamp["dataset_variable_params"]
    finally:
        window.close()
        qt_app.processEvents()


def test_passive_mechanism_refresh_does_not_reenter_identity_build(qt_app, monkeypatch):
    refreshed_mechanism = "\n".join(
        [
            "reaction: A -> C; k=0.4",
            "initial: A=1.0",
            "initial: C=0.0",
        ]
    )

    class _DatasetManagerStub:
        @staticmethod
        def scan_mechanism_parameters(mechanism_text: str) -> list[dict[str, object]]:
            if "A -> C" in str(mechanism_text):
                return [{"name": "k2", "value": 0.4, "min": 0.0, "max": 1.0}]
            return [{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}]

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        dataset_manager=_DatasetManagerStub(),
        simulation_func=_basic_serial_fitting_evaluator(),
        simulation_builder=lambda *_args, **_kwargs: _basic_serial_fitting_evaluator(),
        mechanism_text_getter=lambda: refreshed_mechanism,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        rebuild_calls = 0
        original_rebuild = window._params_ics_tab.rebuild_for_mechanism

        def _rebuild_for_mechanism(*args, **kwargs):
            nonlocal rebuild_calls
            rebuild_calls += 1
            if rebuild_calls > 1:
                raise AssertionError("mechanism refresh re-entered")
            result = original_rebuild(*args, **kwargs)
            window._refresh_run_button_enabled_state()
            return result

        monkeypatch.setattr(window._params_ics_tab, "rebuild_for_mechanism", _rebuild_for_mechanism)

        assert window.fit_launch_identity_owner.build_current_fit_runtime_identity() is not None
        assert rebuild_calls == 1
    finally:
        window.close()
        qt_app.processEvents()


def test_passive_fit_runtime_preparation_builds_evaluator_off_gui_thread(
    qt_app,
    qtbot,
    monkeypatch,
):
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context, SerialFittingEvaluator

    gui_thread_id = threading.get_ident()
    builder_thread_ids: list[int] = []
    runtime_getter_thread_ids: list[int] = []
    runtime_settings_seen: list[dict[str, object]] = []

    class _ReadyRuntimeSession:
        def __init__(self) -> None:
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False) -> None:
            return None

    def _session_factory(_evaluator, *, max_lanes, ledger=None):
        return _ReadyRuntimeSession()

    def _runtime_settings_getter():
        runtime_getter_thread_ids.append(threading.get_ident())
        return {
            "temperature_K": 310.0,
            "use_sparse_jacobian": False,
            "wegscheider_cyclicity_enabled": False,
        }

    def _simulation_builder(
        mechanism_text,
        param_names,
        *,
        solver=None,
        rtol=None,
        atol=None,
        temperature_K=None,
        use_sparse_jacobian=None,
        wegscheider_cyclicity_enabled=None,
    ):
        builder_thread_ids.append(threading.get_ident())
        runtime_settings_seen.append(
            {
                "temperature_K": float(temperature_K),
                "use_sparse_jacobian": bool(use_sparse_jacobian),
                "wegscheider_cyclicity_enabled": bool(wegscheider_cyclicity_enabled),
            }
        )
        context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text),
            param_names=list(param_names),
            t_end=2.0,
            num_points=3,
            temperature_K=float(temperature_K),
            solver=str(solver),
            rtol=float(rtol),
            atol=float(atol),
            use_sparse_jacobian=bool(use_sparse_jacobian),
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
            initial_prefix="initial:",
        )
        return SerialFittingEvaluator(context)

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        _session_factory,
    )
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=_simulation_builder,
        runtime_settings_getter=_runtime_settings_getter,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()

        qtbot.waitUntil(lambda: window.fit_runtime_readiness.snapshot().state.name == "READY", timeout=3000)
        assert builder_thread_ids
        assert gui_thread_id not in builder_thread_ids
        assert runtime_getter_thread_ids
        assert set(runtime_getter_thread_ids) == {gui_thread_id}
        assert runtime_settings_seen == [
            {
                "temperature_K": 310.0,
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
            }
        ]
        assert window._run_button.isEnabled() is True
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_fit_runtime_preparation_stays_failed_without_auto_retry(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    completed = threading.Event()
    attempts = []

    def _session_factory(_fit_evaluator, lane_count):
        attempts.append(int(lane_count))
        raise RuntimeError("factory boom")

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="factory-fail",
        stamp_short="factory-fail",
        lane_count=1,
    )

    controller.set_desired_identity(identity)
    assert completed.wait(1.0)
    controller.handle_worker_finished()

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.FAILED
    assert "factory boom" in str(snapshot.error)
    assert snapshot.desired_hash == ""
    assert attempts == [1]
    assert snapshot.ledger.preparation_starts == 1


def test_required_fit_runtime_preparation_without_session_fails(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    completed = threading.Event()

    def _session_factory(_fit_evaluator, lane_count):
        return None

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="missing-required-session",
        stamp_short="missing-required-session",
        lane_count=1,
        readiness_required=True,
    )

    controller.set_desired_identity(identity)
    assert completed.wait(1.0)
    controller.handle_worker_finished()

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.FAILED
    assert "required fitting runtime session" in str(snapshot.error)
    assert controller.is_ready_for(identity) is False
    assert snapshot.ready_hash == ""


def test_fit_runtime_readiness_controller_owns_launch_accept_or_prepare_decision(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeLaunchDecisionState,
    )

    controller = FittingRuntimeReadinessController(
        session_factory=lambda _fit_evaluator, _lane_count: None,
        finished_callback=lambda: None,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=lambda _params: {"t": [0.0], "series": {"A": [1.0]}},
        stamp={},
        stamp_hash="generic-ready",
        stamp_short="generic-ready",
        lane_count=1,
        readiness_required=False,
    )

    decision = controller.prepare_or_accept_launch(identity)

    assert decision.state is FittingRuntimeLaunchDecisionState.ACCEPTED
    assert decision.accepted_launch is not None
    assert decision.accepted_launch.identity.stamp_hash == "generic-ready"


def test_fit_runtime_poll_retries_completion_when_worker_is_still_unwinding(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessState,
    )

    class _RuntimeSession:
        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False) -> None:
            return None

    class _UnwindingWorker:
        status = "prepared"
        session_created = True

        def __init__(self, identity: FittingRuntimeIdentity, session: _RuntimeSession) -> None:
            self.prepared_identity = identity
            self.prepared_session = session
            self.running_checks = 0

        def isRunning(self) -> bool:
            self.running_checks += 1
            return self.running_checks == 1

        def cancel(self) -> None:
            return None

    window = _build_window()
    try:
        identity = FittingRuntimeIdentity(
            datasets=(),
            config={},
            dataset_overrides=(),
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=_basic_serial_fitting_evaluator(),
            stamp={},
            stamp_hash="unwinding-worker",
            stamp_short="unwinding-worker",
            lane_count=1,
            readiness_required=True,
        )
        session = _RuntimeSession()
        worker = _UnwindingWorker(identity, session)
        readiness = window.fit_runtime_readiness
        readiness._desired_identity = identity
        readiness._active_identity = identity
        readiness._worker = worker
        readiness._state = FittingRuntimeReadinessState.PREPARING

        window.fit_runtime_preparation_owner.poll_preparation()

        assert readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING
        qtbot.waitUntil(
            lambda: readiness.snapshot().state is FittingRuntimeReadinessState.READY,
            timeout=1000,
        )
        assert readiness.snapshot().session is session
        assert worker.running_checks >= 2
    finally:
        window.close()


def test_close_retries_fit_runtime_completion_when_worker_is_still_unwinding(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessState,
    )

    class _RuntimeSession:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    class _UnwindingWorker:
        status = "prepared"
        session_created = True

        def __init__(self, identity: FittingRuntimeIdentity, session: _RuntimeSession) -> None:
            self.prepared_identity = identity
            self.prepared_session = session
            self.running_checks = 0
            self.cancelled = False

        def isRunning(self) -> bool:
            self.running_checks += 1
            return self.running_checks <= 2

        def cancel(self) -> None:
            self.cancelled = True

    window = _build_window()
    try:
        identity = FittingRuntimeIdentity(
            datasets=(),
            config={},
            dataset_overrides=(),
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=_basic_serial_fitting_evaluator(),
            stamp={},
            stamp_hash="closing-unwinding-worker",
            stamp_short="closing-unwinding-worker",
            lane_count=1,
            readiness_required=True,
        )
        session = _RuntimeSession()
        worker = _UnwindingWorker(identity, session)
        readiness = window.fit_runtime_readiness
        readiness._desired_identity = identity
        readiness._active_identity = identity
        readiness._worker = worker
        readiness._state = FittingRuntimeReadinessState.PREPARING

        window.close()
        qt_app.processEvents()

        assert shiboken6.isValid(window)
        assert window._closing is True
        assert window.fit_runtime_preparation_owner.close_after_prepare is True
        assert worker.cancelled is True

        window.fit_runtime_preparation_owner.poll_preparation()

        assert readiness.snapshot().state is FittingRuntimeReadinessState.CLOSING
        qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=1000)
        assert worker.running_checks >= 3
        assert session.close_calls == [True]
    finally:
        if shiboken6.isValid(window):
            window.close()


def test_fit_runtime_identity_defensively_copies_mutable_inputs():
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeIdentity

    config = {"parameters": {"k": 1.0}}
    weights = {"ds1": 1.0}
    stamp = {"algorithm": {"method": "trf"}}

    identity = FittingRuntimeIdentity(
        datasets=(),
        config=config,
        dataset_overrides=(),
        weights=weights,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp=stamp,
        stamp_hash="immutable-inputs",
        stamp_short="immutable-inputs",
        lane_count=1,
    )

    config["parameters"]["k"] = 2.0
    weights["ds1"] = 3.0
    stamp["algorithm"]["method"] = "lm"

    assert identity.config["parameters"]["k"] == 1.0
    assert identity.weights == {"ds1": 1.0}
    assert identity.stamp["algorithm"]["method"] == "trf"


def test_superseded_same_hash_different_lane_preparation_does_not_publish_ready(qt_app):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
    )

    completed = threading.Event()
    sessions: list[object] = []

    class _RuntimeSession:
        def __init__(self, lane_count: int) -> None:
            self.lane_count = int(lane_count)
            self.closed: list[bool] = []

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            return None

        def is_ready(self, *, lane_count=None) -> bool:
            return int(lane_count or 0) == self.lane_count

        def cancel_run(self) -> None:
            return None

        def close(self, *, kill: bool = False) -> None:
            self.closed.append(bool(kill))

    def _session_factory(_fit_evaluator, lane_count):
        session = _RuntimeSession(int(lane_count))
        sessions.append(session)
        return session

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    first = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="same-stamp",
        stamp_short="same-stamp",
        lane_count=1,
    )
    second = replace(first, lane_count=2)

    controller.set_desired_identity(first)
    assert completed.wait(1.0)
    completed.clear()

    controller.set_desired_identity(second)

    snapshot = controller.snapshot()
    assert snapshot.identity is None or snapshot.identity.lane_count != 1
    assert controller.accepted_launch_for(second) is None
    assert sessions[0].closed == [False]


def test_fit_runtime_preparation_failure_is_visible_in_window(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    def _simulation_builder(*_args, **_kwargs):
        raise RuntimeError("builder boom")

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=_simulation_builder,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()

        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.worker is None
            or not window.fit_runtime_readiness.worker.isRunning(),
            timeout=3000,
        )
        window.fit_runtime_preparation_owner.poll_preparation()

        assert window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.FAILED
        assert "Fitting runtime preparation failed: builder boom" in window._status_label.text()
        assert window._run_button.isEnabled() is False
        assert window._stop_button.isEnabled() is False
        assert window.fit_runtime_preparation_owner.refresh_pending is False
    finally:
        window.close()
        qt_app.processEvents()


def test_worker_rejects_missing_required_fit_runtime_session(qt_app):
    from kindred.gui.fitting.worker import GlobalFitWorker

    window = _build_window()
    try:
        dataset = dict(window._global_payload_lookup["ds1"])
    finally:
        window.close()
    worker = GlobalFitWorker(
        [dataset],
        {"k": 1.0},
        fit_evaluator=_basic_serial_fitting_evaluator(),
        fit_runtime_session=None,
        fit_runtime_max_lanes=1,
    )
    with pytest.raises(RuntimeError, match="required fitting runtime session"):
        worker._execute()


def test_cancelled_fit_runtime_preparation_returns_readiness_to_empty(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    release_warm = threading.Event()
    completed = threading.Event()
    sessions: list[object] = []

    class _RuntimeSession:
        def __init__(self) -> None:
            self.closed: list[bool] = []

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    return

        def cancel_run(self) -> None:
            release_warm.set()

        def is_ready(self, *, lane_count=None) -> bool:
            return False

        def close(self, *, kill: bool = False) -> None:
            self.closed.append(bool(kill))

    def _session_factory(_fit_evaluator, lane_count):
        session = _RuntimeSession()
        sessions.append(session)
        return session

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="cancel-prep",
        stamp_short="cancel-prep",
        lane_count=1,
    )

    controller.set_desired_identity(identity)
    assert controller.snapshot().state is FittingRuntimeReadinessState.PREPARING

    assert controller.cancel(kill=True) is False
    assert completed.wait(1.0)
    controller.handle_worker_finished()

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.EMPTY
    assert snapshot.desired_hash == ""
    assert snapshot.active_hash == ""
    assert snapshot.ready_hash == ""
    assert sessions
    assert sessions[-1].closed == [True]


def test_close_during_pre_session_fit_runtime_construction_defers_without_false_session_kill(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    factory_entered = threading.Event()
    release_factory = threading.Event()
    completed = threading.Event()

    def _session_factory(_fit_evaluator, lane_count):
        factory_entered.set()
        release_factory.wait(1.0)
        return None

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="close-before-session",
        stamp_short="close-before-session",
        lane_count=1,
        readiness_required=True,
    )

    controller.set_desired_identity(identity)
    assert factory_entered.wait(1.0)

    assert controller.close(kill=True) is False
    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.CLOSING
    assert snapshot.ledger.session_creations == 0
    assert snapshot.ledger.session_closes == 0

    release_factory.set()
    assert completed.wait(1.0)
    controller.handle_worker_finished()

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.CLOSED
    assert snapshot.ledger.session_creations == 0
    assert snapshot.ledger.session_closes == 0
    assert snapshot.ledger.close_completed == 1


def test_close_during_failed_fit_runtime_preparation_completes_closed(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    completed = threading.Event()

    def _session_factory(_fit_evaluator, lane_count):
        raise RuntimeError("factory failed after close")

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="close-failed-prep",
        stamp_short="close-failed-prep",
        lane_count=1,
        readiness_required=True,
    )

    controller.set_desired_identity(identity)
    assert completed.wait(1.0)
    assert controller.snapshot().state is FittingRuntimeReadinessState.PREPARING

    assert controller.close(kill=True) is True

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.CLOSED
    assert snapshot.error is None
    assert snapshot.ledger.close_completed == 1


def test_close_kills_prepared_fit_runtime_session_after_deferred_drain(qt_app, qtbot):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
        FittingRuntimeReadinessController,
        FittingRuntimeReadinessState,
    )

    completed = threading.Event()
    close_calls: list[bool] = []

    class _RuntimeSession:
        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            return None

        def cancel_run(self) -> None:
            return None

        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False) -> None:
            close_calls.append(bool(kill))

    def _session_factory(_fit_evaluator, lane_count):
        return _RuntimeSession()

    controller = FittingRuntimeReadinessController(
        session_factory=_session_factory,
        finished_callback=completed.set,
    )
    identity = FittingRuntimeIdentity(
        datasets=(),
        config={},
        dataset_overrides=(),
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=_basic_serial_fitting_evaluator(),
        stamp={},
        stamp_hash="prepared-before-close",
        stamp_short="prepared-before-close",
        lane_count=1,
    )

    controller.set_desired_identity(identity)
    assert completed.wait(1.0)
    assert controller.snapshot().state is FittingRuntimeReadinessState.PREPARING

    assert controller.close(kill=True) is True

    snapshot = controller.snapshot()
    assert snapshot.state is FittingRuntimeReadinessState.CLOSED
    assert close_calls == [True]


def test_global_fit_window_close_deletes_dialog(qt_app, qtbot):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0]),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6])},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=[],
        dataset_entries=dataset_entries,
        dataset_payloads=[],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0]), "species": {"A": np.asarray([0.0])}},
    )
    destroyed = {"fired": False}
    window.destroyed.connect(lambda *_args: destroyed.__setitem__("fired", True))

    window.show()
    qtbot.wait(10)
    window.close()
    _process_deferred_deletes()

    qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
    assert destroyed["fired"] is True


def test_global_fit_window_deletes_worker_after_run(qt_app, qtbot):
    t_axis = np.asarray([0.0, 1.0, 2.0], dtype=float)
    y_axis = np.asarray([1.0, 0.8, 0.6], dtype=float)

    def fake_fit_global(_simulate, datasets, shared_params, **kwargs):
        progress = kwargs.get("progress_callback")
        assert callable(progress)
        progress(1, 1.0, dict(shared_params))
        return GlobalFitResult(
            shared_params=dict(shared_params),
            dataset_params={str(datasets[0]["id"]): {}},
            uncertainties=None,
            global_chi_squared=1.0,
            global_r_squared=0.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id=str(datasets[0]["id"]),
                    r_squared=0.0,
                    chi_squared=1.0,
                    rmse=1.0,
                    mae=1.0,
                    residuals=np.asarray([0.0], dtype=float),
                    n_points=1,
                    weight=1.0,
                )
            ],
            nfev=1,
            message="ok",
            completion=_build_completion(status="ok", optimizer_converged=True),
            covariance=None,
            objective_residuals=np.asarray([0.0], dtype=float),
            model_series={str(datasets[0]["id"]): {"A": y_axis}},
            residual_series={str(datasets[0]["id"]): {"A": np.asarray([0.0, 0.0, 0.0], dtype=float)}},
        )

    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t_axis,
            "species_data": {"A": y_axis},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": t_axis, "y": y_axis, "species": "A"}]

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=dataset_entries,
        dataset_payloads=dataset_payloads,
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": t_axis, "species": {"A": y_axis}},
        fit_func=fake_fit_global,
    )
    qtbot.addWidget(window)

    window.run_fit()
    worker = window._worker
    assert worker is not None
    qtbot.waitUntil(lambda: (not shiboken6.isValid(worker)) or worker.isFinished(), timeout=5000)
    _process_deferred_deletes(iterations=10)
    qtbot.waitUntil(lambda: not shiboken6.isValid(worker), timeout=2000)


def test_global_fit_window_close_hard_terminates_stuck_worker(qt_app, qtbot):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0]),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6])},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=[],
        dataset_entries=dataset_entries,
        dataset_payloads=[],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0]), "species": {"A": np.asarray([0.0])}},
    )
    qtbot.addWidget(window)

    class _StuckWorker:
        def __init__(self):
            self.cancel_called = False
            self.terminate_called = False
            self.wait_calls = []
            self.deleted = False
            self._running = True
            self.finished = mock.MagicMock()

        def isRunning(self):
            return bool(self._running)

        def cancel(self):
            self.cancel_called = True

        def wait(self, msecs: int | None = None):
            self.wait_calls.append(msecs)
            return not self._running

        def terminate(self):
            self.terminate_called = True

        def deleteLater(self):
            self.deleted = True

    stuck = _StuckWorker()
    window._worker = stuck  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    window.closeEvent(event)

    assert stuck.cancel_called is True
    assert stuck.wait_calls and int(stuck.wait_calls[0]) == 2000
    assert stuck.terminate_called is True
    assert stuck.deleted is False
    assert window._worker_registry.contains_thread(window._worker)
    assert event.isAccepted() is True
    window._worker_registry.release_thread(window._worker)
    window._worker = None


def test_stale_finished_from_older_fit_worker_does_not_clear_newer_worker(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="old",
            stamp_short="old",
        )
        old_worker = workers[-1]
        old_worker._running = False

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="new",
            stamp_short="new",
        )
        new_worker = workers[-1]
        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True

        old_worker.finished.emit({"result": _build_success_result()})

        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True
    finally:
        window.close()


def test_serial_fit_worker_start_requires_accepted_launch_even_with_ready_snapshot(qt_app, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import (
        FittingRuntimeIdentity,
    )

    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    class _RuntimeSession:
        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)

    window = _build_window()
    try:
        evaluator = _basic_serial_fitting_evaluator()
        identity = FittingRuntimeIdentity(
            datasets=(),
            config={"parameters": {"k": 1.0}},
            dataset_overrides=(),
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=evaluator,
            stamp={},
            stamp_hash="ready-but-not-accepted",
            stamp_short="ready-but-not-accepted",
            lane_count=1,
            readiness_required=True,
        )
        assert window.fit_runtime_readiness.accepted_launch_for(identity) is None

        assert workers == []
        assert window._worker is None
    finally:
        window.close()


def test_stale_error_and_best_update_from_older_fit_worker_do_not_clobber_newer_run(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    warning_calls = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda *_args, **_kwargs: warning_calls.append(True),
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="old",
            stamp_short="old",
        )
        old_worker = workers[-1]
        old_worker._running = False

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="new",
            stamp_short="new",
        )
        new_worker = workers[-1]
        window._params_ics_tab.set_last_fit_params({"k": 77.0})
        window._best_cost = 123.0
        window._status_label.setText("Running newer fit")

        old_worker.bestUpdated.emit(
            {
                "cost": 9.0,
                "shared_params": {"k": 0.1},
                "dataset_params": {"ds1": {}},
                "model_series": {},
                "dataset_stats": {},
            }
        )
        window._apply_pending_best_update()

        assert window._worker is new_worker
        assert window._params_ics_tab.get_last_fit_params() == {"k": 77.0}
        assert window._best_cost == 123.0
        assert window._status_label.text() == "Running newer fit"

        old_worker.error.emit({"kind": "fitting_error", "message": "stale boom"})

        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True
        assert window._status_label.text() == "Running newer fit"
        assert warning_calls == []
    finally:
        window.close()


def test_stale_terminal_payload_after_newer_completion_is_rejected_by_run_stamp(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="old",
            stamp_short="old",
        )
        old_worker = workers[-1]
        old_worker._running = False

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="new",
            stamp_short="new",
        )
        new_worker = workers[-1]
        new_worker._running = False

        new_worker.finished.emit(
            {
                "result": _build_success_result(value=2.0),
                "run_stamp_hash": "new",
            }
        )
        assert window._worker is None
        assert window._last_result.shared_params == {"k": 2.0}

        old_worker.finished.emit(
            {
                "result": _build_success_result(value=9.0),
                "run_stamp_hash": "old",
            }
        )

        assert window._last_result.shared_params == {"k": 2.0}
    finally:
        window.close()


def test_runtime_input_change_supersedes_active_fit_worker_outputs(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _CancelableWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            workers.append(self)

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def terminate(self) -> None:
            self._running = False

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _CancelableWorker)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        prior_result = _build_success_result(value=1.0)
        window._last_result = prior_result
        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="active",
            stamp_short="active",
        )
        worker = workers[-1]

        window.handle_external_runtime_inputs_changed()

        assert worker.cancel_called is True
        assert worker.wait_calls == [2000]
        assert window._worker is None
        assert window.fit_run_state_owner.active_run_stamp_hash == ""
        assert window._run_button.isEnabled() is False

        worker.bestUpdated.emit({"cost": 99.0, "shared_params": {"k": 99.0}, "dataset_params": {"ds1": {}}})
        worker.finished.emit({"result": _build_success_result(value=99.0), "run_stamp_hash": "active"})
        QtCore.QCoreApplication.processEvents()

        assert window._last_result is prior_result
        assert window._best_cost is None
    finally:
        window.close()


def test_runtime_input_change_supersedes_stopped_worker_pending_terminal_signal(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        prior_result = _build_success_result(value=1.0)
        window._last_result = prior_result
        window._best_cost = None
        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="active",
            stamp_short="active",
        )
        worker = workers[-1]
        worker._running = False

        window.handle_external_runtime_inputs_changed()

        assert window._worker is None
        assert window.fit_run_state_owner.active_run_superseded is True
        assert window.fit_run_state_owner.active_run_stamp_hash == ""

        worker.bestUpdated.emit({"cost": 99.0, "shared_params": {"k": 99.0}, "dataset_params": {"ds1": {}}})
        worker.finished.emit({"result": _build_success_result(value=99.0), "run_stamp_hash": "active"})
        QtCore.QCoreApplication.processEvents()

        assert window._last_result is prior_result
        assert window._best_cost is None
    finally:
        window.close()


def test_passive_fit_runtime_preparation_keeps_inputs_editable(qt_app, qtbot, monkeypatch):
    release_warm = threading.Event()
    events: list[str] = []

    class _RuntimeSession:
        def __init__(self) -> None:
            self.ready = False

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def warm(self, *, cancellation_check=None, lane_count=None):
            events.append("runtime:warm")
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    return
            self.ready = True

        def close(self, *, kill: bool = False):
            release_warm.set()

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        return _RuntimeSession()

    class _FactoryWorker(_SignalWorker):
        def start(self) -> None:
            events.append("worker:start")

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)
    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._on_targets_applied()
        qtbot.waitUntil(lambda: "runtime:warm" in events, timeout=2000)

        assert window._run_button.isEnabled() is False
        assert window._stop_button.isEnabled() is True
        assert window._params_ics_tab._param_table.isEnabled() is True
        assert window._species_table._table.isEnabled() is True

        release_warm.set()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state.name == "READY",
            timeout=2000,
        )
        assert "worker:start" not in events
        assert window._params_ics_tab._param_table.isEnabled() is True
        assert window._species_table._table.isEnabled() is True
    finally:
        release_warm.set()
        window.close()


def test_integration_setting_change_invalidates_ready_fit_runtime(qt_app, qtbot, monkeypatch):
    class _ReadySession:
        def __init__(self) -> None:
            self.closed: list[bool] = []

        def warm(self, *, cancellation_check=None, lane_count=None):
            return None

        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False):
            self.closed.append(bool(kill))

    sessions: list[_ReadySession] = []

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        session = _ReadySession()
        sessions.append(session)
        return session

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator(solver="BDF"))
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._on_targets_applied()
        qtbot.waitUntil(lambda: bool(sessions) and window._run_button.isEnabled(), timeout=2000)
        session = sessions[-1]
        assert window._run_button.isEnabled() is True

        window._params_ics_tab._integration_rtol_edit.setText("1e-5")
        qt_app.processEvents()

        assert session.closed == [False]
        assert window._run_button.isEnabled() is False
    finally:
        window.close()


def test_accepted_fixed_param_launch_does_not_poison_base_evaluator(qt_app, qtbot, monkeypatch):
    class _ReadySession:
        def warm(self, *, cancellation_check=None, lane_count=None):
            return None

        def is_ready(self, *, lane_count=None) -> bool:
            return True

        def close(self, *, kill: bool = False):
            return None

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        lambda _evaluator, *, max_lanes, ledger=None: _ReadySession(),
    )

    window = _build_dataset_variable_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        table = window._params_ics_tab._param_table
        shared_fit_item = table.item(0, 0)
        shared_value_item = table.item(0, 3)
        assert shared_fit_item is not None
        assert shared_value_item is not None
        shared_value_item.setText("1.23")
        shared_fit_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()

        fixed_identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()
        assert fixed_identity is not None
        assert getattr(fixed_identity.fit_evaluator, "_fixed_params", {}).get("k") == pytest.approx(1.23)
        window.fit_runtime_readiness.set_desired_identity(fixed_identity)
        qtbot.waitUntil(lambda: window.fit_runtime_readiness.is_ready_for(fixed_identity), timeout=2000)
        monkeypatch.setattr(window.fit_worker_launch_owner, "start_worker", lambda _accepted_launch: None)

        window.run_fit()

        assert getattr(window._fit_evaluator_state.current_base_evaluator(), "_fixed_params", {}) == {}

        shared_fit_item.setCheckState(QtCore.Qt.Checked)
        qt_app.processEvents()
        next_identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert next_identity is not None
        assert getattr(next_identity.fit_evaluator, "_fixed_params", {}) == {}
    finally:
        window.close()


def test_dataset_scoped_readiness_uses_live_checked_fit_flag(qt_app):
    window = _build_dataset_variable_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        shared_fit_item = window._params_ics_tab._param_table.item(0, 0)
        assert shared_fit_item is not None
        shared_fit_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()
        dataset_row = 1
        table = window._params_ics_tab._param_table
        assert table.item(dataset_row, 0).checkState() == QtCore.Qt.Checked
        window._params_ics_tab._parameter_state[dataset_row]["fit"] = False

        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert identity is not None
        assert "init:A" in identity.dataset_overrides[0].variable_params
    finally:
        window.close()


def test_dataset_scoped_readiness_rejects_live_unchecked_fit_flag(qt_app):
    window = _build_dataset_variable_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        shared_fit_item = window._params_ics_tab._param_table.item(0, 0)
        assert shared_fit_item is not None
        shared_fit_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()
        dataset_row = 1
        table = window._params_ics_tab._param_table
        fit_item = table.item(dataset_row, 0)
        assert fit_item is not None
        table.blockSignals(True)
        try:
            fit_item.setCheckState(QtCore.Qt.Unchecked)
        finally:
            table.blockSignals(False)
        window._params_ics_tab._parameter_state[dataset_row]["fit"] = True

        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert identity is None
    finally:
        window.close()


def test_passive_readiness_and_run_fit_use_single_launch_config_collector(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        bundle = window._params_ics_tab.collect_parameter_config_snapshot_for_readiness()
        assert bundle is not None
        base_config, base_dataset_params, base_variable_params = bundle
        collect_calls: list[bool] = []
        started: list[object] = []

        def _copy_bundle():
            return (
                dict(base_config),
                {str(ds_id): dict(values) for ds_id, values in base_dataset_params.items()},
                {
                    str(ds_id): {str(param): dict(spec) for param, spec in params.items()}
                    for ds_id, params in base_variable_params.items()
                },
            )

        def _shared_launch_collector(*, show_errors: bool):
            collect_calls.append(bool(show_errors))
            return _copy_bundle()

        def _old_collector_called():
            raise AssertionError("legacy fitting parameter collector was used")

        monkeypatch.setattr(
            window._params_ics_tab,
            "collect_parameter_config_bundle",
            _shared_launch_collector,
            raising=False,
        )
        monkeypatch.setattr(window._params_ics_tab, "collect_parameter_config", _old_collector_called)
        monkeypatch.setattr(
            window._params_ics_tab,
            "collect_parameter_config_snapshot_for_readiness",
            _old_collector_called,
        )
        monkeypatch.setattr(
            window.fit_worker_launch_owner,
            "start_worker",
            lambda accepted_launch: started.append(accepted_launch),
        )

        window.fit_runtime_preparation_owner.prepare_current_state()
        window.run_fit()

        assert collect_calls == [False, False, True]
        assert started
    finally:
        window.close()


def test_fit_runtime_readiness_refresh_does_not_open_modal_for_partial_tolerance(qt_app, qtbot, monkeypatch):
    window = _build_window()
    dialogs: list[str] = []

    def _capture_warning(*_args, **_kwargs):
        dialogs.append("warning")
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    def _capture_information(*_args, **_kwargs):
        dialogs.append("information")
        return int(QtWidgets.QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_capture_warning))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(_capture_information))
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        rtol_edit = window._params_ics_tab._integration_rtol_edit

        rtol_edit.setText("1e-")
        qtbot.waitUntil(
            lambda: not window.fit_runtime_preparation_owner.refresh_pending,
            timeout=1000,
        )

        assert dialogs == []
        assert window._run_button.isEnabled() is False
    finally:
        window.close()


def test_start_fit_reports_invalid_dataset_payload_on_launch(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        warnings: list[tuple[str, str]] = []

        def _capture_warning(_parent, title, message, *_args, **_kwargs):
            warnings.append((str(title), str(message)))
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_capture_warning))
        monkeypatch.setattr(window, "_refresh_fit_window_state_for_current_mechanism", lambda **_kwargs: True)
        window._global_payload_results["ds1"] = FitDatasetPayloadResult.invalid("payload exploded")

        window.run_fit()

        assert warnings
        assert warnings[-1][0] == "Global Fit"
        assert "Dataset 'ds1' has invalid payload:" in warnings[-1][1]
        assert "payload exploded" in warnings[-1][1]
    finally:
        window.close()
        qt_app.processEvents()


def test_run_fit_button_rejects_stale_ready_identity_with_current_lane_budget(qt_app, qtbot, monkeypatch):
    created: list[tuple[str, int]] = []
    budget = {"value": 2}

    class _ReadySession:
        def __init__(self, *, lane_count: int) -> None:
            self.lane_count = int(lane_count)
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready) and int(lane_count or 1) <= self.lane_count

        def close(self, *, kill: bool = False):
            return None

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        created.append(("create", int(max_lanes)))
        return _ReadySession(lane_count=int(max_lanes))

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        monkeypatch.setattr(window, "_fit_runtime_lane_budget", lambda _dataset_count: int(budget["value"]))
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._on_targets_applied()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)

        assert created == [("create", 2)]
        assert window._run_button.isEnabled() is True

        budget["value"] = 4
        window._refresh_run_button_enabled_state()

        assert window._run_button.isEnabled() is False
        assert created == [("create", 2)]
    finally:
        window.close()


def test_run_fit_button_rejects_stale_ready_identity_after_runtime_setting_change(qt_app, qtbot, monkeypatch):
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context, SerialFittingEvaluator

    created_temperatures: list[float] = []
    runtime_settings = {
        "temperature_K": 298.15,
        "use_sparse_jacobian": False,
        "wegscheider_cyclicity_enabled": False,
    }

    class _ReadySession:
        def __init__(self) -> None:
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            return None

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        return _ReadySession()

    def _runtime_settings_getter():
        return dict(runtime_settings)

    def _simulation_builder(
        mechanism_text,
        param_names,
        *,
        solver,
        rtol,
        atol,
        temperature_K=None,
        use_sparse_jacobian=None,
        wegscheider_cyclicity_enabled=None,
    ):
        created_temperatures.append(float(temperature_K))
        context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text),
            param_names=list(param_names),
            t_end=2.0,
            num_points=3,
            temperature_K=float(temperature_K),
            solver=str(solver),
            rtol=float(rtol),
            atol=float(atol),
            use_sparse_jacobian=bool(use_sparse_jacobian),
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
            initial_prefix="initial:",
        )
        return SerialFittingEvaluator(context)

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=_simulation_builder,
        runtime_settings_getter=_runtime_settings_getter,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)
        assert created_temperatures == [298.15]

        runtime_settings["temperature_K"] = 310.0
        window._refresh_run_button_enabled_state()

        assert window._run_button.isEnabled() is False
        assert created_temperatures == [298.15]
    finally:
        window.close()


def test_runtime_settings_getter_failure_blocks_fitting_readiness(qt_app, qtbot, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    def _runtime_settings_getter():
        raise RuntimeError("settings unavailable")

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=lambda *_args, **_kwargs: _basic_serial_fitting_evaluator(),
        runtime_settings_getter=_runtime_settings_getter,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]

        window.fit_runtime_preparation_owner.prepare_current_state()

        snapshot = window.fit_runtime_readiness.snapshot()
        assert snapshot.state is FittingRuntimeReadinessState.BLOCKED
        assert "settings unavailable" in str(snapshot.error.__cause__)
        assert window._run_button.isEnabled() is False
        assert "Fitting runtime not ready" in window._status_label.text()
    finally:
        window.close()


@pytest.mark.parametrize("change", ["dataset_weight", "weight_mode"])
def test_run_fit_button_rejects_stale_ready_identity_after_weight_change(qt_app, qtbot, monkeypatch, change):
    created: list[int] = []

    class _ReadySession:
        def __init__(self) -> None:
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            return None

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        created.append(int(max_lanes))
        return _ReadySession()

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        if change == "dataset_weight":
            window._species_table._weight_mode_combo.blockSignals(True)
            window._species_table._weight_mode_combo.setCurrentIndex(1)
            window._species_table._weight_mode_combo.blockSignals(False)
        window._on_targets_applied()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)

        assert created == [1]
        assert window._run_button.isEnabled() is True

        if change == "dataset_weight":
            created.clear()
            window._persist_dataset_weight("ds1", 2.0)
        else:
            window._species_table._weight_mode_combo.setCurrentIndex(1)

        assert window._run_button.isEnabled() is False
        assert created == ([] if change == "dataset_weight" else [1])
    finally:
        window.close()


def test_dataset_removal_reschedules_fit_runtime_preparation(qt_app, qtbot, monkeypatch):
    created: list[int] = []

    class _ReadySession:
        def __init__(self) -> None:
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            return None

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        created.append(int(max_lanes))
        return _ReadySession()

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
            {
                "id": "ds2",
                "label": "Dataset 2",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([0.9, 0.7, 0.5], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([0.9, 0.7, 0.5]), "species": "A"},
        ],
        simulation_func=_basic_serial_fitting_evaluator(),
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._species_table._fit_targets_selection_applied["ds2"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._on_targets_applied()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)
        assert created == [2]

        window._remove_datasets_from_session(["ds2"])

        assert window._run_button.isEnabled() is False
        qtbot.waitUntil(lambda: len(created) >= 2, timeout=2000)
        assert created[-1] == 1
    finally:
        window.close()


def test_deferred_fit_runtime_identity_remains_ready_after_acceptance(qt_app, qtbot, monkeypatch):
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context, SerialFittingEvaluator

    created: list[int] = []
    started: list[dict[str, object]] = []

    class _ReadySession:
        def __init__(self) -> None:
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            return None

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        return _ReadySession()

    def _simulation_builder(
        mechanism_text,
        param_names,
        *,
        solver,
        rtol,
        atol,
        temperature_K=None,
        use_sparse_jacobian=None,
        wegscheider_cyclicity_enabled=None,
    ):
        created.append(1)
        context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text),
            param_names=list(param_names),
            t_end=2.0,
            num_points=3,
            temperature_K=float(temperature_K),
            solver=str(solver),
            rtol=float(rtol),
            atol=float(atol),
            use_sparse_jacobian=bool(use_sparse_jacobian),
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
            initial_prefix="initial:",
        )
        return SerialFittingEvaluator(context)

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        mechanism_text_getter=_basic_mechanism_text,
        reactions_text_getter=lambda: "reaction: A -> B; k=0.2",
        simulation_func=None,
        simulation_builder=_simulation_builder,
        runtime_settings_getter=lambda: {
            "temperature_K": 298.15,
            "use_sparse_jacobian": False,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)
        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()
        assert identity is not None
        assert window.fit_runtime_readiness.is_ready_for(identity)
        monkeypatch.setattr(window.fit_worker_launch_owner, "start_worker", lambda accepted_launch: started.append(accepted_launch))

        window.run_fit()
        window._refresh_run_button_enabled_state()
        current_identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()

        assert started
        assert current_identity is not None
        assert window.fit_runtime_readiness.is_ready_for(current_identity)
        assert window._run_button.isEnabled() is True
        assert len(created) == 1
    finally:
        window.close()


def test_fit_runtime_input_change_restarts_preparation_after_old_worker_exits(qt_app, qtbot, monkeypatch):
    window = _build_window()
    try:
        release_warm = threading.Event()
        warm_hashes: list[str] = []

        class _RuntimeSession:
            def __init__(self) -> None:
                self.ready = False

            def warm(self, *, cancellation_check=None, lane_count=None):
                warm_hashes.append(window.fit_runtime_readiness.snapshot().active_hash)
                while not release_warm.wait(0.01):
                    if cancellation_check is not None and cancellation_check():
                        return
                self.ready = True

            def is_ready(self, *, lane_count=None) -> bool:
                return bool(self.ready)

            def cancel_run(self):
                return None

            def close(self, *, kill: bool = False):
                return None

        monkeypatch.setattr(
            "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
            lambda _evaluator, *, max_lanes, ledger=None: _RuntimeSession(),
        )
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        def builder(_mechanism_text, _param_names, *, solver, rtol, atol, **_runtime_settings):
            return _basic_serial_fitting_evaluator(
                solver=str(solver),
                rtol=float(rtol),
                atol=float(atol),
            )

        window._fit_evaluator_state.set_base_evaluator(_basic_serial_fitting_evaluator(solver="BDF"))
        window._simulation_builder = builder
        window._fit_evaluator_state.set_simulation_builder(builder)
        window._on_targets_applied()
        qtbot.waitUntil(lambda: bool(warm_hashes), timeout=2000)
        old_hash = warm_hashes[-1]

        window._params_ics_tab._integration_rtol_edit.setText("1e-5")
        window._on_fit_runtime_inputs_changed()

        assert window.fit_runtime_preparation_owner.refresh_pending is True

        release_warm.set()
        qtbot.waitUntil(lambda: len(warm_hashes) >= 2, timeout=2000)
        assert warm_hashes[-1] != old_hash
    finally:
        release_warm.set()
        window.close()


def test_ic_apply_schedules_fit_runtime_preparation_refresh(qt_app):
    window = _build_window()
    try:
        window.fit_runtime_preparation_owner.refresh_pending = False
        assert window.fit_runtime_preparation_owner.refresh_pending is False

        window._species_table.icApplied.emit(
            "ds1",
            {"A": {"initial": 2.0, "min": 0.0, "max": 10.0, "log10": False}},
            {"A": True},
        )

        assert window.fit_runtime_preparation_owner.refresh_pending is True
    finally:
        window.close()


def test_added_observable_scalar_parameter_schedules_fit_runtime_preparation_refresh(qt_app):
    window = _build_window()
    try:
        window.fit_runtime_preparation_owner.refresh_pending = False
        window._params_ics_tab._shared_param_definitions["scale"] = {
            "value": 1.0,
            "min": 0.0,
            "max": 10.0,
            "source": "scalar parameter",
        }
        assert window.fit_runtime_preparation_owner.refresh_pending is False

        window._params_ics_tab.add_missing_scalars_as_parameters(
            ["scale"],
            ["ds1"],
            "shared",
        )

        assert window.fit_runtime_preparation_owner.refresh_pending is True
    finally:
        window.close()


def test_close_waits_for_active_fit_runtime_preparation_before_deleting(qt_app, qtbot, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        release_warm = threading.Event()
        warm_started = threading.Event()

        class _RuntimeSession:
            def __init__(self) -> None:
                self.cancelled = False

            def warm(self, *, cancellation_check=None, lane_count=None):
                warm_started.set()
                while not release_warm.wait(0.01):
                    if cancellation_check is not None and cancellation_check():
                        return

            def cancel_run(self):
                self.cancelled = True

            def is_ready(self, *, lane_count=None) -> bool:
                return False

            def close(self, *, kill: bool = False):
                close_calls.append(bool(kill))

        sessions: list[_RuntimeSession] = []
        close_calls: list[bool] = []
        monkeypatch.setattr(
            "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
            lambda _evaluator, *, max_lanes, ledger=None: sessions.append(_RuntimeSession()) or sessions[-1],
        )
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._fit_evaluator_state.set_base_evaluator(_basic_serial_fitting_evaluator())
        window._on_targets_applied()
        qtbot.waitUntil(lambda: warm_started.is_set(), timeout=2000)

        window.close()
        qt_app.processEvents()

        assert shiboken6.isValid(window)
        assert window.fit_runtime_preparation_owner.close_after_prepare is True
        assert sessions[-1].cancelled is True

        release_warm.set()

        qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
        assert close_calls == [True]
    finally:
        release_warm.set()
        if shiboken6.isValid(window):
            window.close()


def test_close_waits_for_failed_fit_runtime_preparation_before_deleting(qt_app, qtbot, monkeypatch):
    window = _build_window()
    release_warm = threading.Event()
    try:
        window.show()
        qt_app.processEvents()
        warm_started = threading.Event()
        close_calls: list[bool] = []

        class _RuntimeSession:
            def __init__(self) -> None:
                self.cancelled = False

            def warm(self, *, cancellation_check=None, lane_count=None):
                warm_started.set()
                while not release_warm.wait(0.01):
                    if cancellation_check is not None and cancellation_check():
                        break
                raise RuntimeError("warm failed after close")

            def cancel_run(self):
                self.cancelled = True

            def is_ready(self, *, lane_count=None) -> bool:
                return False

            def close(self, *, kill: bool = False):
                close_calls.append(bool(kill))

        sessions: list[_RuntimeSession] = []
        monkeypatch.setattr(
            "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
            lambda _evaluator, *, max_lanes, ledger=None: sessions.append(_RuntimeSession()) or sessions[-1],
        )
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._fit_evaluator_state.set_base_evaluator(_basic_serial_fitting_evaluator())
        window._on_targets_applied()
        qtbot.waitUntil(lambda: warm_started.is_set(), timeout=2000)

        window.close()
        qt_app.processEvents()

        assert shiboken6.isValid(window)
        assert window.fit_runtime_preparation_owner.close_after_prepare is True
        assert sessions[-1].cancelled is True

        release_warm.set()

        qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
        assert close_calls == [True]
    finally:
        release_warm.set()
        if shiboken6.isValid(window):
            window.close()


def test_detached_fit_worker_registry_releases_without_custom_finished_payload(qt_app, qtbot):
    window = _build_window()

    class _StuckWorker:
        def __init__(self):
            self.cancel_called = False
            self.terminate_called = False
            self.wait_calls = []
            self.deleted = False
            self._running = True
            self.finished = mock.MagicMock()

        def isRunning(self):
            return bool(self._running)

        def cancel(self):
            self.cancel_called = True

        def wait(self, msecs: int | None = None):
            self.wait_calls.append(msecs)
            return not self._running

        def terminate(self):
            self.terminate_called = True

        def deleteLater(self):
            self.deleted = True

        def setParent(self, _parent):
            return None

    stuck = _StuckWorker()
    window._worker = stuck  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert window._worker_registry.contains_thread(stuck)

    stuck._running = False
    qtbot.waitUntil(lambda: not window._worker_registry.contains_thread(stuck), timeout=2000)
    qtbot.waitUntil(lambda: stuck.deleted is True, timeout=2000)
    assert stuck.deleted is True


def test_detached_fit_worker_late_emissions_do_not_reenter_deleted_dialog(qt_app, qtbot, monkeypatch):
    workers: list[_SignalWorker] = []

    class _LateSignalWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            self.terminate_called = False
            workers.append(self)

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            return False

        def terminate(self) -> None:
            self.terminate_called = True

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _LateSignalWorker)

    window = _build_window()
    callbacks: list[tuple[str, object]] = []
    window._on_worker_progress = lambda percent, message, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("progress", worker)
    )
    window._handle_global_best_update = lambda payload, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("best", worker)
    )
    window._handle_global_fit_complete = lambda payload, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("finished", worker)
    )
    window._on_worker_error = lambda error, *, worker=None: callbacks.append(("error", worker))  # type: ignore[method-assign]
    window._schedule_worker_cleanup = lambda worker: callbacks.append(("cleanup", worker))  # type: ignore[method-assign]

    config = {
        "parameters": {"k": 1.0},
        "bounds": {"k": (0.0, 2.0)},
        "fixed_params": {},
        "method": "trf",
        "max_nfev": 2,
        "seed": None,
        "log10_params": {},
    }

    window._set_running_state(True)
    _start_worker_from_accepted_launch(
        window,
        datasets=[],
        config=config,
        dataset_overrides=[],
        weights=None,
        requested_solver="BDF",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=lambda _params: {},
        stamp={},
        stamp_hash="late",
        stamp_short="late",
    )
    worker = workers[-1]

    window.close()
    _process_deferred_deletes(iterations=10)
    qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
    assert window._worker_registry.contains_thread(worker)

    worker.progress.emit(17, "late progress")
    worker.bestUpdated.emit({"cost": 3.14})
    worker.finished.emit({"result": _build_success_result()})
    worker.error.emit({"kind": "fitting_error", "message": "late error"})
    QtCore.QCoreApplication.processEvents()
    QtCore.QCoreApplication.processEvents()

    assert callbacks == []


def test_consecutive_fit_dispatch_cycles_leave_clean_state(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.quit_called = False
            self.wait_calls: list[int] = []
            workers.append(self)

        def quit(self) -> None:
            self.quit_called = True

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        for run_index in range(3):
            window._set_running_state(True)
            _start_worker_from_accepted_launch(
                window,
                datasets=[],
                config=config,
                dataset_overrides=[],
                weights=None,
                requested_solver="BDF",
                requested_rtol=1e-6,
                requested_atol=1e-12,
                fit_evaluator=lambda _params: {},
                stamp={},
                stamp_hash=f"run-{run_index}",
                stamp_short=f"run-{run_index}",
            )
            worker = workers[-1]
            window._pending_best_payload = {"iteration": run_index + 1}
            window._pending_best_worker = worker
            window._pending_best_timer.start()

            worker._running = False
            worker.finished.emit({"result": _build_success_result(value=float(run_index + 1))})
            QtCore.QCoreApplication.processEvents()

            assert window._worker is None
            assert not window._pending_best_timer.isActive()
            assert window._pending_best_payload is None
            assert window._pending_best_worker is None
            assert worker.quit_called is True
            assert worker.wait_calls == [2000]
    finally:
        window.close()


def test_cancelled_fit_hard_teardown_returns_dialog_to_rerunnable_idle_state(qt_app, monkeypatch):
    worker_ref: dict[str, object] = {}
    cancel_events: list[str] = []

    class _CancelableWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            self.terminate_called = False
            self.deleted = False
            worker_ref["worker"] = self

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def cancel(self) -> None:
            cancel_events.append("worker:cancel")
            super().cancel()

        def terminate(self) -> None:
            self.terminate_called = True

        def deleteLater(self) -> None:
            self.deleted = True

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _CancelableWorker)

    window = _build_window()
    try:
        runtime_close_calls: list[bool] = []

        class _BadCloseRuntimeSession:
            def warm(self, *, cancellation_check=None, lane_count=None):
                return None

            def is_ready(self, *, lane_count=None) -> bool:
                return True

            def close(self, *, kill: bool = False):
                cancel_events.append(f"runtime:close:{kill}")
                runtime_close_calls.append(bool(kill))
                raise RuntimeError("runtime close failed")

        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        from kindred.gui.fitting.runtime_readiness import FittingRuntimeIdentity
        from kindred.gui.fitting.window import FittingRuntimeSession

        identity = FittingRuntimeIdentity(
            datasets=(),
            config=config,
            dataset_overrides=(),
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=_basic_serial_fitting_evaluator(),
            stamp={},
            stamp_hash="cancel-run",
            stamp_short="cancel-run",
            lane_count=1,
            readiness_required=True,
        )
        original_session_factory = FittingRuntimeSession.from_serial_evaluator
        FittingRuntimeSession.from_serial_evaluator = staticmethod(
            lambda _fit_evaluator, *, max_lanes, ledger=None: _BadCloseRuntimeSession()
        )
        try:
            window.fit_runtime_readiness.set_desired_identity(identity)
            worker = window.fit_runtime_readiness.worker
            if worker is not None:
                while worker.isRunning():
                    QtCore.QCoreApplication.processEvents()
                window.fit_runtime_readiness.handle_worker_finished()
        finally:
            FittingRuntimeSession.from_serial_evaluator = original_session_factory
        accepted_launch = window.fit_runtime_readiness.accepted_launch_for(identity)
        assert accepted_launch is not None

        window._set_running_state(True)
        window.fit_worker_launch_owner.start_worker(accepted_launch)
        worker = worker_ref["worker"]

        window._cancel_fit()

        assert runtime_close_calls == [True]
        assert cancel_events == ["worker:cancel", "runtime:close:True"]
        assert worker.cancel_called is True
        assert worker.wait_calls == [2000]
        assert worker.terminate_called is False
        assert window._worker is None
        assert window._run_button.isEnabled() is False
        assert window._stop_button.isEnabled() is False
        assert window._status_label.text() == "Fit cancelled"
        worker.bestUpdated.emit({"cost": 99.0})
        worker.error.emit({"kind": "cancelled", "message": "late cancelled"})
        QtCore.QCoreApplication.processEvents()
        assert "late cancelled" not in window._status_label.text()
    finally:
        window.close()


def test_stop_fit_schedules_runtime_preparation_refresh(qt_app, monkeypatch):
    worker_ref: dict[str, object] = {}
    scheduled: list[str] = []

    class _CancelableWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            worker_ref["worker"] = self

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def terminate(self) -> None:
            self._running = False

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _CancelableWorker)

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="cancel-run",
            stamp_short="cancel-run",
        )
        monkeypatch.setattr(window.fit_runtime_preparation_owner, "schedule_refresh", lambda: scheduled.append("schedule"))

        window._cancel_fit()

        assert worker_ref["worker"].cancel_called is True
        assert scheduled == ["schedule"]
        assert window._status_label.text() == "Fit cancelled"
    finally:
        window.close()


def test_stop_fit_runtime_preparation_does_not_schedule_refresh(qt_app, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState
    from kindred.gui.fitting.window import FittingRuntimeSession

    scheduled: list[str] = []
    release_warm = threading.Event()

    class _BlockingRuntimeSession:
        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    return

        def is_ready(self, *, lane_count=None) -> bool:
            return False

        def cancel_run(self) -> None:
            release_warm.set()

        def close(self, *, kill: bool = False) -> None:
            release_warm.set()

    original_session_factory = FittingRuntimeSession.from_serial_evaluator
    FittingRuntimeSession.from_serial_evaluator = staticmethod(
        lambda _fit_evaluator, *, max_lanes, ledger=None: _BlockingRuntimeSession()
    )
    window = _build_dataset_variable_window()
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()
        assert window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING
        window.fit_runtime_preparation_owner.schedule_refresh()
        assert window.fit_runtime_preparation_owner.refresh_pending is True
        monkeypatch.setattr(window.fit_runtime_preparation_owner, "schedule_refresh", lambda: scheduled.append("schedule"))

        window._cancel_fit()

        assert scheduled == []
        assert window.fit_runtime_preparation_owner.refresh_pending is False
        assert window._status_label.text() == "Fitting runtime preparation cancelled"
    finally:
        release_warm.set()
        FittingRuntimeSession.from_serial_evaluator = original_session_factory
        window.close()


def test_visible_stop_button_cancels_run_initiated_fit_runtime_preparation(qt_app, qtbot, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    release_warm = threading.Event()
    sessions: list[object] = []

    class _BlockingRuntimeSession:
        def __init__(self) -> None:
            self.cancelled = False
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    self.cancelled = True
                    return
            self.ready = True

        def cancel_run(self) -> None:
            self.cancelled = True
            release_warm.set()

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False) -> None:
            release_warm.set()

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        session = _BlockingRuntimeSession()
        sessions.append(session)
        return session

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text

        window.run_fit()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING,
            timeout=2000,
        )

        assert window._stop_button.isEnabled() is True
        window._stop_button.click()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state is not FittingRuntimeReadinessState.PREPARING,
            timeout=2000,
        )

        assert sessions
        assert sessions[-1].cancelled is True
        assert window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.EMPTY
        assert window._status_label.text() == "Fitting runtime preparation cancelled"
        session_count = len(sessions)
        qtbot.wait(100)
        window.fit_runtime_preparation_owner.poll_preparation()
        qt_app.processEvents()
        assert len(sessions) == session_count
        assert window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.EMPTY
    finally:
        release_warm.set()
        window.close()


def test_run_fit_starts_accepted_launch_after_runtime_ready(qt_app, qtbot, monkeypatch):
    from kindred.core.fitting_evaluation import SerialFittingEvaluator
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    release_warm = threading.Event()
    started: list[dict[str, object]] = []

    class _ReadyAfterReleaseRuntimeSession:
        def __init__(self) -> None:
            self.ready = False

        @property
        def ledger(self):
            return None

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    return
            self.ready = True

        def cancel_run(self) -> None:
            release_warm.set()

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False) -> None:
            release_warm.set()

    class _CaptureWorker(_SignalWorker):
        def __init__(self, datasets, shared_params, *, fit_evaluator=None, fit_runtime_session=None, **kwargs):
            super().__init__()
            started.append(
                {
                    "datasets": list(datasets),
                    "shared_params": dict(shared_params),
                    "fit_evaluator": fit_evaluator,
                    "fit_runtime_session": fit_runtime_session,
                    "run_stamp_hash": str(kwargs.get("run_stamp_hash") or ""),
                }
            )
            self._running = False

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        return _ReadyAfterReleaseRuntimeSession()

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)
    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _CaptureWorker)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text

        window.run_fit()
        assert started == []
        release_warm.set()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.READY,
            timeout=2000,
        )
        assert started == []
        window.run_fit()
        qtbot.waitUntil(lambda: bool(started), timeout=2000)

        assert isinstance(started[-1]["fit_evaluator"], SerialFittingEvaluator)
        assert started[-1]["fit_runtime_session"] is not None
        assert started[-1]["run_stamp_hash"]
    finally:
        release_warm.set()
        window.close()


def test_deferred_fit_launch_captures_failed_restore_baseline(qt_app, qtbot, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    release_warm = threading.Event()
    workers: list[_SignalWorker] = []

    class _ReadyAfterReleaseRuntimeSession:
        def __init__(self) -> None:
            self.ready = False

        @property
        def ledger(self):
            return None

        def warm(self, *, cancellation_check=None, lane_count=None) -> None:
            while not release_warm.wait(0.01):
                if cancellation_check is not None and cancellation_check():
                    return
            self.ready = True

        def cancel_run(self) -> None:
            release_warm.set()

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False) -> None:
            release_warm.set()

    class _CaptureWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        lambda *_args, **_kwargs: _ReadyAfterReleaseRuntimeSession(),
    )
    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _CaptureWorker)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text

        window.run_fit()
        assert workers == []
        release_warm.set()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.READY,
            timeout=2000,
        )
        assert workers == []
        window.run_fit()
        qtbot.waitUntil(lambda: bool(workers), timeout=2000)
        worker = workers[-1]

        window._params_ics_tab.push_best_update({"k": 0.9}, {})
        assert window._params_ics_tab._param_table.item(0, 3).text() == "0.9"

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        worker.finished.emit({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._params_ics_tab._param_table.item(0, 3).text() == "1"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
    finally:
        release_warm.set()
        window.close()


def test_deferred_generic_evaluator_prepares_without_runtime_session_requirement(qt_app, qtbot, monkeypatch):
    from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

    def _generic_builder(*_args, **_kwargs):
        def _evaluate(_params):
            return {"t": np.asarray([0.0, 1.0]), "species": {"A": np.asarray([1.0, 1.0])}}

        return _evaluate

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generic evaluator must not create runtime session")),
    )

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 1.0], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0]), "y": np.asarray([1.0, 1.0]), "species": "A"}
        ],
        mechanism_species=["A"],
        mechanism_text_getter=lambda: "reaction: A -> B; k=0\ninitial: A=1.0",
        reactions_text_getter=lambda: "reaction: A -> B; k=0",
        simulation_func=None,
        simulation_builder=_generic_builder,
    )
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window.fit_runtime_preparation_owner.prepare_current_state()
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state is not FittingRuntimeReadinessState.PREPARING,
            timeout=2000,
        )

        snapshot = window.fit_runtime_readiness.snapshot()
        assert snapshot.state is FittingRuntimeReadinessState.READY
        assert snapshot.identity is not None
        assert snapshot.identity.readiness_required is False
        assert snapshot.session is None
        assert window._run_button.isEnabled() is True
    finally:
        window.close()


def test_successful_fit_result_parameter_mutation_reprepares_next_run_identity(qt_app, qtbot, monkeypatch):
    created: list[object] = []

    class _ReadySession:
        def __init__(self) -> None:
            self.closed: list[bool] = []
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            self.closed.append(bool(kill))

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        session = _ReadySession()
        created.append(session)
        return session

    monkeypatch.setattr("kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator", _fake_from_serial)

    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        window._on_targets_applied()
        qtbot.waitUntil(lambda: len(created) == 1 and window._run_button.isEnabled(), timeout=2000)
        old_session = created[-1]

        window._handle_global_fit_complete({"result": _build_success_result(param_name="k", value=1.5)})

        qtbot.waitUntil(lambda: len(created) >= 2, timeout=2000)
        assert old_session.closed == [False]
        assert window._params_ics_tab.get_parameter_state()[0]["value"] == pytest.approx(1.5)
        assert window._run_button.isEnabled() is True
    finally:
        window.close()


def test_fit_runtime_session_cache_invalidates_when_lane_budget_changes(qt_app, qtbot, monkeypatch):
    created: list[tuple[str, int]] = []

    class _FakeSession:
        def __init__(self, *, budget: int):
            self.budget = int(budget)
            self.closed: list[bool] = []
            self.ready = False

        def warm(self, *, cancellation_check=None, lane_count=None):
            self.ready = True

        def is_ready(self, *, lane_count=None) -> bool:
            return bool(self.ready)

        def close(self, *, kill: bool = False):
            self.closed.append(bool(kill))

    def _fake_from_serial(_evaluator, *, max_lanes, ledger=None):
        created.append(("create", int(max_lanes)))
        return _FakeSession(budget=int(max_lanes))

    monkeypatch.setattr(
        "kindred.gui.fitting.window.FittingRuntimeSession.from_serial_evaluator",
        _fake_from_serial,
    )
    window = _build_window(simulation_func=_basic_serial_fitting_evaluator())
    try:
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]
        window._mechanism_text_getter = _basic_mechanism_text
        identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity()
        assert identity is not None
        first_identity = replace(identity, lane_count=2)
        second_identity = replace(identity, lane_count=2)
        third_identity = replace(identity, lane_count=4)

        window.fit_runtime_readiness.set_desired_identity(first_identity)
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state.name == "READY",
            timeout=2000,
        )
        first = window.fit_runtime_readiness.snapshot().session
        window.fit_runtime_readiness.set_desired_identity(second_identity)
        second = window.fit_runtime_readiness.snapshot().session
        window.fit_runtime_readiness.set_desired_identity(third_identity)
        qtbot.waitUntil(
            lambda: window.fit_runtime_readiness.snapshot().state.name == "READY"
            and window.fit_runtime_readiness.snapshot().session is not first,
            timeout=2000,
        )
        third = window.fit_runtime_readiness.snapshot().session

        assert first is second
        assert third is not first
        assert created == [("create", 2), ("create", 4)]
        assert first.closed == [False]
    finally:
        window.close()


def test_old_worker_best_update_is_disconnected_after_completion(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            workers.append(self)

        def quit(self) -> None:
            return None

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        callbacks: list[dict] = []
        monkeypatch.setattr(
            window,
            "_dispatch_fit_worker_best_update",
            lambda payload: callbacks.append(dict(payload)),
        )

        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="done",
            stamp_short="done",
        )
        worker = workers[-1]
        worker._running = False
        worker.finished.emit({"result": _build_success_result()})
        QtCore.QCoreApplication.processEvents()

        worker.bestUpdated.emit({"cost": 99.0})
        QtCore.QCoreApplication.processEvents()

        assert callbacks == []
    finally:
        window.close()


@pytest.mark.parametrize(
    ("event", "expected_details"),
    [
        ("complete", ""),
        ("error", ""),
    ],
)
def test_completion_stops_pending_best_timer_before_dialog(qt_app, monkeypatch, event, expected_details):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        window._pending_best_payload = {"cost": 1.0}
        window._pending_best_worker = worker
        window._pending_best_timer.start()

        states: list[tuple[bool, object, object]] = []
        captured = {"details": None}

        def _capture_exec(self):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            captured["details"] = self.detailedText()
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        def _capture_warning(*_args, **_kwargs):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            captured["details"] = None
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_capture_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        if event == "complete":
            window._handle_global_fit_complete({"result": _build_success_result()}, worker=worker)
        else:
            window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        assert states == [(False, None, None)]
        assert captured["details"] == expected_details
    finally:
        window.close()


def test_worker_error_logs_traceback_and_populates_dialog_details(qt_app, monkeypatch, caplog):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        dialogs: list[dict[str, object]] = []

        def _unexpected_warning(*_args, **_kwargs):
            raise AssertionError("worker errors should use an instance-based warning dialog")

        def _capture_exec(self):
            dialogs.append(
                {
                    "text": self.text(),
                    "details": self.detailedText(),
                    "title": self.windowTitle(),
                    "icon": self.icon(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_unexpected_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        stack_trace = "Traceback line 1\nTraceback line 2"
        with caplog.at_level("WARNING", logger="kindred.gui.fitting.window"):
            window._on_worker_error(
                {
                    "kind": "fitting_error",
                    "message": "boom",
                    "context": {"stack_trace": stack_trace},
                },
                worker=worker,
            )

        assert dialogs == [
            {
                "text": "boom",
                "details": stack_trace,
                "title": "Fitting",
                "icon": QtWidgets.QMessageBox.Icon.Warning,
            }
        ]
        messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.fitting.window"]
        assert any(message.startswith("Fitting worker reported error:") for message in messages)
        assert stack_trace in messages
    finally:
        window.close()


def test_completion_dialog_uses_instance_message_box_with_error_diagnostics_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()
        dialogs: list[dict[str, object]] = []

        def _unexpected_information(*_args, **_kwargs):
            raise AssertionError("completion dialog should use an instance-based message box")

        def _unexpected_warning(*_args, **_kwargs):
            raise AssertionError("completion dialog should use an instance-based message box")

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                    "icon": self.icon(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(_unexpected_information))
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_unexpected_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete",
                "text": mock.ANY,
                "details": "",
                "icon": QtWidgets.QMessageBox.Icon.Information,
            }
        ]
    finally:
        window.close()


def test_completion_dialog_hides_detail_pane_when_no_detail_text_exists(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": _build_success_result()})

        assert dialogs == [""]
    finally:
        window.close()


def test_completion_dialog_aggregates_per_dataset_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds2",
                "label": "Dataset 2",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            dataset_failures={
                "ds1": _make_diagnostic(phase="final_replay", dataset_id="ds1", message="first", stack_trace="trace one"),
                "ds2": _make_diagnostic(phase="final_replay", dataset_id="ds2", message="second", stack_trace="trace two"),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds1", message="first", stack_trace="trace one"),
                _make_detail_section(dataset_id="ds2", message="second", stack_trace="trace two"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs
        detail_text = dialogs[0]
        assert "Dataset 1" in detail_text
        assert "trace one" in detail_text
        assert "Dataset 2" in detail_text
        assert "trace two" in detail_text
        assert "suppressed top trace" not in detail_text
        assert "\n\n---\n\n" in detail_text
    finally:
        window.close()


def test_completion_dialog_does_not_log_suppressed_top_level_detail_text_on_success(qt_app, monkeypatch, caplog):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        with caplog.at_level("WARNING", logger="kindred.gui.fitting.window"):
            window._handle_global_fit_complete({"result": result})

        messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.fitting.window"]
        assert not messages
    finally:
        window.close()


def test_successful_completion_suppresses_top_level_error_diagnostics_logging_and_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()

        warning_mock = mock.Mock()
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr("kindred.gui.fitting.window.logger.warning", warning_mock)
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [""]
        warning_mock.assert_not_called()
    finally:
        window.close()


def test_warning_completion_labels_top_level_error_diagnostics_when_dataset_error_entry_missing(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds_x",
                "label": "Dataset X",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds_x",
                message="orphan failure",
                stack_trace="orphan trace",
            ),
            detail_sections=[
                _make_detail_section(dataset_id="ds_x", message="orphan failure", stack_trace="orphan trace"),
            ],
        )
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append({"title": self.windowTitle(), "details": self.detailedText()})
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "details": "Dataset X\norphan trace",
            }
        ]
    finally:
        window.close()


def test_warning_completion_surfaces_optimizer_diagnostic_message_in_dialog_body(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds1",
                message="optimizer step rejected",
            ),
        )
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "text": (
                    "Final Chi-Squared (\u03c7\u00b2): 1\n\n"
                    "Fitted Parameters:\n"
                    "  k = 1\n\n"
                    "Warnings:\n"
                    "- Optimizer did not report convergence; results may be suboptimal.\n"
                    "- optimizer step rejected"
                ),
                "details": "",
            }
        ]
    finally:
        window.close()


def test_failed_completion_preparation_fatal_uses_preparation_remediation_not_x_axis(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="undefined symbol k_total",
                remediation="preparation",
                failure_kind="preparation_error",
                details={"stage": "parameter_algebra"},
            ),
        )

        severity, title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert title == "Global Fit Failed"
        assert "parameter algebra failed" in text.lower()
        assert "fix the preparation or parameter algebra error" in text.lower()
        assert "adjust t_min/t_max" not in text.lower()
    finally:
        window.close()


def test_failed_completion_alignment_failure_keeps_x_axis_remediation(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="Dataset 'ds1': observed X values fall outside model range. Adjust t_min/t_max.",
                    remediation="x_axis_mapping",
                ),
            },
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "adjust t_min/t_max" in text.lower()
        assert "fix x axis / mapping" in text.lower()
    finally:
        window.close()


def test_failed_completion_nonfinite_metrics_shows_nonfinite_message_not_x_axis(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            nonfinite_metrics=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="Final χ² is non-finite; results are invalid.",
                remediation="nonfinite_metrics",
            ),
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "final χ² is non-finite; results are invalid." in text.lower()
        assert "inspect the fit objective and inputs for non-finite values" in text.lower()
        assert "fix x axis / mapping" not in text.lower()
    finally:
        window.close()


def test_failed_completion_dataset_failures_own_fail_body_summary_when_top_level_exists(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="top-level process-pool failure",
                remediation="generic_retry",
                stack_trace="top trace",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="dataset replay failed",
                    remediation="generic_retry",
                ),
            },
            detail_sections=[
                _make_detail_section(message="top-level process-pool failure", stack_trace="top trace"),
            ],
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "- Dataset 1: dataset replay failed" in text
        assert "top-level process-pool failure" not in text
    finally:
        window.close()


def test_failed_completion_does_not_leave_project_apply_scopes_enabled(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._available_project_apply_scopes() == set()
        assert not window._apply_to_project_button.isEnabled()
    finally:
        window.close()


def test_failed_completion_keeps_user_edited_value_cells_after_clearing_fit_authority(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._params_ics_tab.set_last_fit_params({"k": 0.5})
        parameter_state = window._params_ics_tab.get_parameter_state()
        parameter_state[0]["value"] = 1.7
        parameter_state[0]["last_fit"] = 0.5
        window._params_ics_tab.set_parameter_state(parameter_state)
        window._params_ics_tab._populate_parameter_table()

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._params_ics_tab._param_table.item(0, 3).text() == "1.7"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_restores_pre_run_value_cells_after_live_best_update(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._pre_run_parameter_state = window._params_ics_tab.get_parameter_state()
        window._params_ics_tab.push_best_update({"k": 0.9}, {})
        assert window._params_ics_tab._param_table.item(0, 3).text() == "0.9"

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._params_ics_tab._param_table.item(0, 3).text() == "1"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_restores_pre_run_staged_dataset_params(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._params_ics_tab.set_staged_dataset_params({"ds1": {"init:A": 2.5}})
        window._pre_run_parameter_state = window._params_ics_tab.get_parameter_state()
        window._pre_run_staged_dataset_params = window._params_ics_tab.get_staged_dataset_params()
        window._params_ics_tab.push_best_update({"k": 0.9}, {"ds1": {"init:A": 0.4}})

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_staged_dataset_params() == {"ds1": {"init:A": 2.5}}
        assert window._staged_initial_condition_parameters() == {"ds1": {"init:A": 2.5}}
        assert _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS in window._available_project_apply_scopes()
    finally:
        window.close()
        qt_app.processEvents()


def test_start_fit_launch_failure_after_mechanism_refresh_preserves_refreshed_parameter_rows(qt_app, monkeypatch):
    old_mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    refreshed_mechanism = "\n".join(
        [
            "reaction: A -> C; k=0.4",
            "initial: A=1.0",
            "initial: C=0.0",
        ]
    )

    class _PreparedEvaluator:
        def __init__(self, mechanism_text: str, param_names: list[str], *, solver: str, rtol: float, atol: float) -> None:
            self.prepared_metadata = {
                "version": 1,
                "mechanism_text_sha256": FittingWindow._mechanism_text_sha256(mechanism_text),
                "mechanism_text_len": len(mechanism_text),
                "param_names": list(param_names),
                "t_end": 2.0,
                "num_points": 3,
                "temperature_K": 298.15,
                "solver_requested": solver,
                "solver_normalized": solver,
                "solver_warning": None,
                "rtol": float(rtol),
                "atol": float(atol),
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
                "initial_prefix": "init:",
            }

        def __call__(self, _params):
            return {
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            }

        def with_fixed_params(self, _fixed_params):
            return self

    class _DatasetManagerStub:
        @staticmethod
        def scan_mechanism_parameters(mechanism_text: str) -> list[dict[str, object]]:
            if "A -> C" in str(mechanism_text):
                return [{"name": "k2", "value": 0.4, "min": 0.0, "max": 1.0}]
            return [{"name": "k1", "value": 0.2, "min": 0.0, "max": 1.0}]

        @staticmethod
        def sync_fit_result_views(_model_series, *, dataset_stats=None, dataset_ids=None) -> None:
            return None

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.0, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        dataset_manager=_DatasetManagerStub(),
        simulation_func=_PreparedEvaluator(old_mechanism, ["k1"], solver="BDF", rtol=1e-6, atol=1e-12),
        simulation_builder=lambda mechanism_text, param_names, *, solver, rtol, atol: _PreparedEvaluator(
            mechanism_text,
            list(param_names),
            solver=solver,
            rtol=rtol,
            atol=atol,
        ),
        mechanism_text_getter=lambda: refreshed_mechanism,
        reactions_text_getter=lambda: refreshed_mechanism,
    )
    try:
        window.show()
        qt_app.processEvents()
        assert [str(entry.get("param_name") or "") for entry in window._params_ics_tab.get_parameter_state()] == ["k2"]

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        monkeypatch.setattr(
            window.fit_worker_launch_owner,
            "start_worker",
            lambda _accepted_launch: (_ for _ in ()).throw(RuntimeError("launch boom")),
        )

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None
        assert set(config["parameters"]) == {"k2"}

        window._params_ics_tab._integration_solver_combo.setCurrentText("BDF")
        window._params_ics_tab._integration_rtol_edit.setText("1e-6")
        window._params_ics_tab._integration_atol_edit.setText("1e-12")
        window.run_fit()

        parameter_names = [str(entry.get("param_name") or "") for entry in window._params_ics_tab.get_parameter_state()]
        assert parameter_names == ["k2"]
        assert window._params_ics_tab._param_table.item(0, 2).text() == "k2"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
        assert window._params_ics_tab.get_mechanism_species() == ["A", "C"]
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_dataset_manager_fit_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": _build_success_result()})
        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is not None
        assert ds_view["chi_squared"] == pytest.approx(1.0)

        window.fit_run_state_owner.set_active_dataset_ids(["ds1"])
        failed = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed})

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["model_x"] is None
        assert ds_view["model_y"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_results_summary_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._run_results_tab.set_run_stamp({"solver": "BDF"}, "hash123", "hash123")
        window._results_summary_button.setEnabled(True)
        assert window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {"solver": "BDF"}

        failed = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed})

        assert not window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {}
        assert window._run_results_tab._last_run_stamp_hash == ""
        assert window._run_results_tab._last_run_stamp_short == ""
    finally:
        window.close()
        qt_app.processEvents()


def test_target_change_during_active_fit_cancels_stale_run_and_uses_current_targets(qt_app, monkeypatch):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {
                "A": np.asarray([1.0, 0.8, 0.6], dtype=float),
                "B": np.asarray([0.2, 0.3, 0.4], dtype=float),
            },
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "y": np.asarray(
                [
                    [1.0, 0.8, 0.6],
                    [0.2, 0.3, 0.4],
                ],
                dtype=float,
            ),
            "species": ["A", "B"],
        }
    ]
    window = _build_window(dataset_entries=dataset_entries, dataset_payloads=dataset_payloads)
    try:
        window.show()
        qt_app.processEvents()
        assert window._run_results_tab._fit_targets_by_dataset["ds1"] == ["A"]

        class _CancelableWorker(_SignalWorker):
            def wait(self, msecs: int | None = None) -> bool:
                self._running = False
                return True

            def terminate(self) -> None:
                self._running = False

            def deleteLater(self) -> None:
                return None

        worker = _CancelableWorker()
        window._worker = worker
        worker.progress.connect(window._dispatch_fit_worker_progress)
        worker.bestUpdated.connect(window._dispatch_fit_worker_best_update)
        worker.finished.connect(window._dispatch_fit_worker_finished)
        worker.error.connect(window._dispatch_fit_worker_error)
        window._species_table._fit_targets_selection_applied["ds1"] = ["B"]
        window._on_targets_applied()
        qt_app.processEvents()
        assert worker.cancel_called is True
        assert window._worker is None
        assert window._results_rebuild_pending is False
        assert window._run_results_tab._fit_targets_by_dataset["ds1"] == ["B"]

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window.hide()
        failed = _build_completion_result(
            status="fail",
            dataset_id="ds1",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed, "run_stamp_hash": "stale"}, worker=worker)

        assert window._run_results_tab._fit_targets_by_dataset["ds1"] == ["B"]
        payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert payload["current_species"] == "B"
        assert sorted(payload["all_species"].keys()) == ["B"]
    finally:
        window.close()
        qt_app.processEvents()


def test_start_fit_failure_clears_prior_fit_state_before_worker_launch(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        window._handle_global_fit_complete({"result": _build_success_result()})

        monkeypatch.setattr(
            window._params_ics_tab,
            "collect_parameter_config",
            lambda: {
                "parameters": {"k": 1.0},
                "bounds": {"k": (0.0, 2.0)},
                "fixed_params": {},
                "method": "trf",
                "max_nfev": 2,
                "seed": None,
                "log10_params": {},
            },
        )
        monkeypatch.setattr(
            window.fit_launch_identity_owner,
            "collect_dataset_selection",
            lambda: FittingLaunchDatasetSelection(
                rows=({"id": "ds1", "label": "Dataset 1", "species": "A", "include": True, "weight": 1.0},),
                ids=("ds1",),
            ),
        )
        monkeypatch.setattr(window._params_ics_tab, "collect_integration_settings", lambda: ("BDF", 1e-6, 1e-12))
        monkeypatch.setattr(window, "_datasets_payloads_for_readiness", lambda _ids: None)
        monkeypatch.setattr(window, "_datasets_payloads_for_run", lambda _ids: None)

        window.run_fit()

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["chi_squared"] is None
        assert window._available_project_apply_scopes() == set()
    finally:
        window.close()
        qt_app.processEvents()


def test_run_fit_unavailable_evaluator_clears_prior_dataset_manager_fit_state(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": _build_success_result()})
        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is not None
        assert ds_view["chi_squared"] == pytest.approx(1.0)

        window._fit_evaluator_state.set_base_evaluator(None)
        window._simulation_builder = None
        window._fit_evaluator_state.set_simulation_builder(None)

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None

        window.run_fit()

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["model_x"] is None
        assert ds_view["model_y"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_run_fit_unavailable_evaluator_clears_open_results_summary_state(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._run_results_tab.set_run_stamp({"solver": "BDF"}, "hash123", "hash123")
        window._run_results_tab.update_statistics({"Datasets": 1})
        window._results_summary_button.setEnabled(True)
        window._run_results_tab.open_results_summary_dialog()
        qt_app.processEvents()
        assert window._run_results_tab._stamp_dialog is not None
        assert window._run_results_tab._stamp_dialog.isVisible()

        refresh_calls: list[tuple] = []
        original_refresh = window._run_results_tab._stamp_dialog.refresh

        def tracking_refresh(*args, **kwargs):
            refresh_calls.append((args, kwargs))
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(window._run_results_tab._stamp_dialog, "refresh", tracking_refresh)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._fit_evaluator_state.set_base_evaluator(None)
        window._simulation_builder = None
        window._fit_evaluator_state.set_simulation_builder(None)

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None

        window.run_fit()

        assert not window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {}
        assert refresh_calls
        args, kwargs = refresh_calls[-1]
        assert args[0] == {}
        assert args[1] == ""
        assert args[2] == ""
        assert args[3] is None
        assert kwargs["fitted_params"] is None
        assert kwargs["dataset_fitted_params"] is None
    finally:
        if getattr(window._run_results_tab, "_stamp_dialog", None) is not None:
            window._run_results_tab._stamp_dialog.close()
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_only_active_run_dataset_fit_state_even_with_partial_failed_result(
    qt_app, monkeypatch
):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "Dataset 2",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([0.9, 0.7, 0.5], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": False,
        },
    ]
    dataset_payloads = [
        {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"},
        {"id": "ds2", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([0.9, 0.7, 0.5]), "species": "A"},
    ]
    window = _build_window(dataset_entries=dataset_entries, dataset_payloads=dataset_payloads)
    try:
        window.show()
        qt_app.processEvents()
        datasets = {
            "ds1": {"t": np.asarray([0.0, 1.0, 2.0], dtype=float), "species": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)}},
            "ds2": {"t": np.asarray([0.0, 1.0, 2.0], dtype=float), "species": {"A": np.asarray([0.9, 0.7, 0.5], dtype=float)}},
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: datasets.get(str(name)),
        )
        window._dataset_manager.sync_fit_result_views(
            {
                "ds1": {"A": np.asarray([0.95, 0.75, 0.55], dtype=float)},
                "ds2": {"A": np.asarray([0.85, 0.65, 0.45], dtype=float)},
            },
            dataset_stats={
                "ds1": {"chi_squared": 1.0, "r_squared": 0.9},
                "ds2": {"chi_squared": 2.0, "r_squared": 0.8},
            },
            dataset_ids=["ds1", "ds2"],
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window.fit_run_state_owner.set_active_dataset_ids(["ds1"])
        failed = _build_completion_result(
            status="fail",
            dataset_id="ds1",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {"ds1": {"A": np.asarray([0.5, 0.4, 0.3], dtype=float)}}
        failed.dataset_info = [
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.12,
                chi_squared=9.9,
                rmse=1.0,
                mae=1.0,
                residuals=np.asarray([1.0], dtype=float),
                n_points=1,
                weight=1.0,
            )
        ]

        window._handle_global_fit_complete({"result": failed})

        ds1_view = window._dataset_manager._dataset_views["ds1"]
        assert ds1_view["model_series"] is None
        assert ds1_view["chi_squared"] is None
        assert ds1_view["r_squared"] is None
        assert window._run_results_tab._latest_model_series_by_dataset == {}
        assert window._run_results_tab._last_stats == {}

        ds2_view = window._dataset_manager._dataset_views["ds2"]
        assert ds2_view["model_series"] is not None
        assert ds2_view["chi_squared"] == pytest.approx(2.0)
        assert ds2_view["r_squared"] == pytest.approx(0.8)
    finally:
        window.close()
        qt_app.processEvents()


def test_worker_error_clears_active_run_fit_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        window._handle_global_fit_complete({"result": _build_success_result()})
        window._params_ics_tab.set_last_fit_params({"k": 0.5})
        window.fit_run_state_owner.set_active_dataset_ids(["ds1"])

        worker = _SignalWorker()
        worker._running = False
        window._worker = worker

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._available_project_apply_scopes() == set()
        assert window._run_results_tab._latest_model_series_by_dataset == {}
        assert window._run_results_tab._last_stats == {}
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_routes_through_failed_run_closeout_helper(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        calls: list[object] = []

        monkeypatch.setattr(
            window,
            "_clear_failed_run_visual_state",
            lambda current_result=None: calls.append(current_result),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": result})

        assert calls == [result]
    finally:
        window.close()
        qt_app.processEvents()


def test_terminal_completion_without_result_exits_visible_running_state(qt_app):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()

        window._set_running_state(True)
        assert window._stop_button.isEnabled()
        assert not window._params_ics_tab._add_param_button.isEnabled()

        window._handle_global_fit_complete({})

        assert window._status_label.text() == "Global fit failed"
        assert not window._stop_button.isEnabled()
        assert window._params_ics_tab._add_param_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()


def test_worker_error_routes_through_failed_run_closeout_helper(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        calls: list[object] = []

        monkeypatch.setattr(
            window,
            "_clear_failed_run_visual_state",
            lambda current_result=None: calls.append(current_result),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        window.fit_run_state_owner.set_active_dataset_ids(["ds1"])

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        assert calls == [None]
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_keeps_top_level_message_in_details_without_stack_trace(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="top-level process-pool failure",
                remediation="generic_retry",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="dataset replay failed",
                    remediation="generic_retry",
                ),
            },
            detail_sections=[
                FitDetailSection(
                    dataset_id=None,
                    failure=build_simulation_failure(
                        kind="simulation_error",
                        message="top-level process-pool failure",
                    ),
                )
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["top-level process-pool failure"]
    finally:
        window.close()


def test_failed_completion_keeps_top_level_trace_when_matching_dataset_error_has_no_detail(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds1",
                message="top-level failure",
                stack_trace="top trace",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(phase="final_replay", dataset_id="ds1", message="dataset message only"),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds1", message="top-level failure", stack_trace="top trace"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["Dataset 1\ntop trace"]
    finally:
        window.close()


def test_failed_completion_keeps_distinct_top_level_trace_for_same_dataset(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ghost_ds",
                message="optimizer-time failure",
                stack_trace="TOP TRACE",
            ),
            dataset_failures={
                "ghost_ds": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ghost_ds",
                    message="per-dataset failure",
                    stack_trace="DATASET TRACE",
                ),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ghost_ds", message="optimizer-time failure", stack_trace="TOP TRACE"),
                _make_detail_section(dataset_id="ghost_ds", message="per-dataset failure", stack_trace="DATASET TRACE"),
            ],
        )
        dialogs: list[dict[str, object]] = []

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Global Fit Failed",
                "text": mock.ANY,
                "details": "ghost_ds\nTOP TRACE\n\n---\n\nghost_ds\nDATASET TRACE",
            }
        ]
    finally:
        window.close()


def test_failed_completion_suppresses_duplicate_top_level_dataset_trace(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds_x",
                "label": "Dataset X",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds_x",
                message="duplicate failure",
                stack_trace="duplicate trace",
            ),
            dataset_failures={
                "ds_x": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds_x",
                    message="duplicate failure",
                    stack_trace="duplicate trace",
                ),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds_x", message="duplicate failure", stack_trace="duplicate trace"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["Dataset X\nduplicate trace"]
    finally:
        window.close()


def test_warning_completion_keeps_unlabeled_top_level_trace_without_dataset_tag(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="fatal failure",
                stack_trace="fatal trace",
            ),
            detail_sections=[
                _make_detail_section(message="fatal failure", stack_trace="fatal trace"),
            ],
        )
        warning_mock = mock.Mock()
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append({"title": self.windowTitle(), "details": self.detailedText()})
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr("kindred.gui.fitting.window.logger.warning", warning_mock)
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "details": "fatal trace",
            }
        ]
        assert warning_mock.call_args_list == [
            mock.call("Global fit completed with warnings: %s", "failed"),
            mock.call("%s", "fatal trace"),
        ]
    finally:
        window.close()


def test_close_teardown_disconnects_worker_signals(qt_app, monkeypatch):
    """_hard_teardown_worker must disconnect signals so queued emissions
    after closeEvent cannot reach handlers on a partially-destroyed window."""
    workers: list[_SignalWorker] = []

    class _TeardownWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            workers.append(self)

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def quit(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _TeardownWorker)

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        window._set_running_state(True)
        _start_worker_from_accepted_launch(
            window,
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="teardown",
            stamp_short="teardown",
        )
        worker = workers[-1]
        assert window._worker is worker

        # Track calls through downstream handlers (not dispatch slots) so that
        # _disconnect_fit_worker_signals sees the original slot methods.
        best_handler_calls: list[dict] = []
        progress_handler_calls: list[tuple] = []
        window._handle_global_best_update = lambda payload, *, worker=None: best_handler_calls.append(dict(payload))
        window._on_worker_progress = lambda p, m, *, worker=None: progress_handler_calls.append((p, m))

        window._hard_teardown_worker(reason="test teardown", disable_ui=False)

        worker.bestUpdated.emit({"cost": 42.0})
        worker.progress.emit(50, "should not arrive")
        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.processEvents()

        assert best_handler_calls == [], "bestUpdated signal should be disconnected after hard teardown"
        assert progress_handler_calls == [], "progress signal should be disconnected after hard teardown"
    finally:
        window.close()


def test_on_worker_progress_returns_early_when_closing(qt_app):
    """_on_worker_progress must not touch widgets when _closing is True."""
    window = _build_window()
    try:
        set_value_calls: list[int] = []
        set_text_calls: list[str] = []
        window._progress_bar.setValue = lambda v: set_value_calls.append(v)
        window._status_label.setText = lambda t: set_text_calls.append(t)

        window._closing = True
        window._on_worker_progress(50, "should be ignored")

        assert set_value_calls == [], "_progress_bar.setValue called despite _closing=True"
        assert set_text_calls == [], "_status_label.setText called despite _closing=True"
    finally:
        window._closing = False
        window.close()


def test_start_fit_clears_cached_state_before_launch(monkeypatch):
    window = _build_window()
    try:
        window._latest_model_series = {"stale": {"A": np.asarray([1.0], dtype=float)}}
        window._latest_dataset_stats = {"stale": {"chi_squared": 1.0}}
        window._latest_plot_model_series = {"stale": {"A": np.asarray([1.0], dtype=float)}}
        window._latest_plot_model_x = {"stale": np.asarray([0.0], dtype=float)}
        window._best_cost = 9.0
        window._best_effort_failures.add("stale.best_effort")
        window._teardown_disable_failures.add("stale.disable")

        monkeypatch.setattr(
            window._params_ics_tab,
            "collect_parameter_config",
            lambda: {
                "parameters": {"k": 1.0},
                "bounds": {"k": (0.0, 2.0)},
                "fixed_params": {},
                "method": "trf",
                "max_nfev": 2,
                "seed": None,
                "log10_params": {},
            },
        )
        monkeypatch.setattr(
            window.fit_launch_identity_owner,
            "collect_dataset_selection",
            lambda: FittingLaunchDatasetSelection(
                rows=({"id": "ds1", "label": "Dataset 1", "species": "A", "include": True, "weight": 1.0},),
                ids=("ds1",),
            ),
        )
        window.run_fit()

        assert window._latest_model_series == {}
        assert window._latest_dataset_stats == {}
        assert window._latest_plot_model_series == {}
        assert window._latest_plot_model_x == {}
        assert window._best_cost is None
        assert window._best_effort_failures == set()
        assert window._teardown_disable_failures == set()
    finally:
        window.close()

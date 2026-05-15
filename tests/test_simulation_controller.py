from __future__ import annotations

import inspect
import threading
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
import warnings
from typing import Any, Callable, Mapping, Optional
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_containment import (
    BatchCompletionRecord,
    BatchLaneOutcome,
    BatchPolledCompletion,
    BatchRequestMetadata,
)
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulation_identity import SimulationIdentity, SimulationScopeIdentity
from kindred.gui.controllers.simulation_controller import (
    SimulationController,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.parallel_batch_executor import default_batch_lane_pool_factory
from kindred.gui.controllers.parallel_batch_outcome import resolve_parallel_batch_outcome
from kindred.gui.controllers.simulation_completion_publication import CompletionCallbackState
from kindred.gui.controllers.simulation_run_state import PreviewOwnershipState
from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers
from kindred.gui.ports import SliderReplayIntent, SimulationUiPorts
from kindred.gui.simulation_run_ui_owner import SimulationRunUiOwner
from kindred.gui.simulation_worker import SimulationWorker
from tests.worker_stubs import make_stubborn_worker
from tests.batch_context_test_helpers import seed_batch_context


class _RecordingLanePool:
    def __init__(self, submitted: list[dict[str, object]]) -> None:
        self.submitted = submitted
        self.close_calls: list[bool] = []
        self.ready_lane_count = 999

    def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
        _ = wait
        self.ready_lane_count = max(1, int(max_lanes))

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        self.submitted.append(dict(task))
        return BatchLaneOutcome(
            lane_id="test-lane",
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=str(set_id),
            owner_epoch=1,
            success=True,
            payload={"success": True, "set_id": str(set_id), "run_id": int(run_id)},
        )

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))


class _ProtocolLanePool:
    ready_lane_count = 999

    def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
        _ = wait
        self.ready_lane_count = max(1, int(max_lanes))

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        _ = task, active_timeout_s
        return _lane_outcome(str(set_id), run_id=int(run_id), request_id=int(request_id))

    def close(self, *, kill: bool = False) -> None:
        _ = kill


@pytest.mark.unit
def test_simulation_run_ui_owner_clears_algebra_status_tooltip(qt_app) -> None:
    owner = SimulationRunUiOwner(
        schedule_runtime_availability_refresh=lambda **_kwargs: None,
        results_table_getter=lambda: None,
    )
    algebra_label = QtWidgets.QLabel()
    owner.bind_widgets(
        run_button=QtWidgets.QPushButton(),
        stop_button=QtWidgets.QPushButton(),
        progress=QtWidgets.QProgressBar(),
        status_label=QtWidgets.QLabel(),
        algebra_status_label=algebra_label,
    )

    owner.set_algebra_status_text("Algebra: 1 ok, 1 error", details="bad algebra")
    owner.set_algebra_status_text("")

    assert algebra_label.text() == ""
    assert algebra_label.toolTip() == ""


def _join_active_batch_requests(controller: SimulationController) -> None:
    controller.parallel_batch.join_active_requests(timeout_s=2.0)


def _test_simulation_plan_payload(
    *,
    set_id: str = "id1",
    set_name: str = "set1",
    mechanism_text: str = "reaction: A -> B; k=1",
    initials: dict[str, float] | None = None,
    fast_mode: bool = False,
    cache_key: str = "cache",
    simulation_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    identity = dict(simulation_identity or {"schema_id": "schema", "param_fingerprint": f"fingerprint-{set_id}"})
    return SimulationPlan.from_execution_request(
        {
            "prepared_payload": None,
            "initials": dict(initials or {"A": 1.0}),
            "t_span": (0.0, 10.0),
            "solver_config": {"solver": "BDF"},
            "mechanism_text": str(mechanism_text),
            "simulation_identity": identity,
        },
        execution_mode="preview" if bool(fast_mode) else "explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={
            "cache_key": str(cache_key),
            "simulation_identity": identity,
        },
        cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": [str(set_id)]},
        metadata={"set_id": str(set_id), "set_name": str(set_name), "fast_mode": bool(fast_mode)},
    ).to_payload()


def _batch_policy_context(controller: SimulationController):
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    return policy_context


def _test_callback_context(
    *,
    run_id: int,
    request_id: int,
    cache_key: str,
    fast_mode: bool = False,
    parallel: bool = False,
    queue_ids: tuple[str, ...] = (),
    queue_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "active": True,
        "run_id": int(run_id),
        "request_id": int(request_id),
        "cache_key": str(cache_key),
        "fast_mode": bool(fast_mode),
        "parallel": bool(parallel),
    }
    if queue_ids:
        context["queue_ids"] = tuple(str(item) for item in queue_ids)
        context["queue_names"] = tuple(str(item) for item in queue_names)
        context["pos"] = 0
        context["total"] = len(queue_ids)
    return context


def _test_launch_provenance() -> dict[str, Any]:
    return {
        "temperature_K": 298.15,
        "temperature_source": "ui",
        "simulation_time": 1.0,
        "num_points_requested": 3,
        "mechanism_text": "reaction: A -> B ; k=0.1",
    }


def _capture_callback_identity(
    controller: SimulationController,
    *,
    run_id: int = 1,
    fast_mode: bool = False,
    request_id: int = 1,
    owner_epoch: Optional[int] = None,
    batch_set: Optional[str] = None,
    batch_set_id: Optional[str] = None,
    cache_key: str = "cache-key",
    callback_context: Mapping[str, Any] | None = None,
    simulation_identity: Optional[dict[str, Any]] = None,
    preview_batch_cache_token: Optional[str] = None,
    launch_provenance: Mapping[str, Any] | None = None,
) -> SimulationCallbackIdentity:
    resolved_simulation_identity = simulation_identity if simulation_identity is not None else {}
    resolved_preview_batch_cache_token = preview_batch_cache_token if preview_batch_cache_token is not None else ""
    resolved_context = (
        callback_context
        if callback_context is not None
        else _test_callback_context(
            run_id=int(run_id),
            request_id=int(request_id),
            cache_key=str(cache_key),
            fast_mode=bool(fast_mode),
        )
    )
    return controller._capture_simulation_callback_identity(
        run_id=int(run_id),
        fast_mode=bool(fast_mode),
        request_id=int(request_id),
        owner_epoch=owner_epoch,
        batch_set=batch_set,
        batch_set_id=batch_set_id,
        cache_key=str(cache_key),
        callback_context=resolved_context,
        simulation_identity=resolved_simulation_identity,
        preview_batch_cache_token=resolved_preview_batch_cache_token,
        launch_provenance=launch_provenance if launch_provenance is not None else _test_launch_provenance(),
    )


def _malformed_callback_identity_without_context(
    *,
    run_id: Optional[int],
    fast_mode: bool,
    request_id: Optional[int],
    owner_epoch: Optional[int],
    batch_set: Optional[str],
    batch_set_id: Optional[str],
    cache_key: Optional[str],
):
    return SimpleNamespace(
        run_id=run_id,
        fast_mode=fast_mode,
        request_id=request_id,
        owner_epoch=owner_epoch,
        batch_set=batch_set,
        batch_set_id=batch_set_id,
        cache_key=cache_key,
        callback_context=None,
        simulation_identity={},
        preview_batch_cache_token=None,
        launch_provenance=None,
    )


def _connect_worker_application_signals(
    controller: SimulationController,
    worker: object,
    *,
    run_id: int,
    fast_mode: bool,
    request_id: int,
    owner_epoch: Optional[int] = None,
    set_name: str,
    set_id: str,
    cache_key: str,
    simulation_identity: Optional[dict[str, Any]] = None,
    preview_batch_cache_token: Optional[str] = None,
) -> SimulationCallbackIdentity:
    identity = _capture_callback_identity(
        controller,
        run_id=run_id,
        fast_mode=fast_mode,
        request_id=request_id,
        owner_epoch=owner_epoch,
        batch_set=set_name,
        batch_set_id=set_id,
        cache_key=cache_key,
        simulation_identity=simulation_identity,
        preview_batch_cache_token=preview_batch_cache_token,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._connect_simulation_worker_application_signals(
        worker,
        callback_identity=identity,
    )
    return identity


def _complete_with_callback_identity(
    controller: SimulationController,
    result: dict[str, Any],
    *,
    run_id: int = 1,
    fast_mode: bool = False,
    request_id: int = 1,
    owner_epoch: Optional[int] = None,
    batch_set: Optional[str] = None,
    batch_set_id: Optional[str] = None,
    cache_key: str = "cache-key",
    callback_context: Mapping[str, Any] | None = None,
    simulation_identity: Optional[dict[str, Any]] = None,
    preview_batch_cache_token: Optional[str] = None,
    callback_identity: SimulationCallbackIdentity | None = None,
) -> None:
    identity = callback_identity or _capture_callback_identity(
        controller,
        run_id=run_id,
        fast_mode=fast_mode,
        request_id=request_id,
        owner_epoch=owner_epoch,
        batch_set=batch_set,
        batch_set_id=batch_set_id,
        cache_key=cache_key,
        callback_context=callback_context,
        simulation_identity=simulation_identity,
        preview_batch_cache_token=preview_batch_cache_token,
    )
    controller._on_simulation_complete(result, callback_identity=identity)


def _error_with_callback_identity(
    controller: SimulationController,
    error_msg: object,
    *,
    run_id: int = 1,
    fast_mode: bool = False,
    request_id: int = 1,
    owner_epoch: Optional[int] = None,
    batch_set: Optional[str] = None,
    batch_set_id: Optional[str] = None,
    cache_key: str = "cache-key",
    callback_context: Mapping[str, Any] | None = None,
    callback_identity: SimulationCallbackIdentity | None = None,
) -> None:
    identity = callback_identity or _capture_callback_identity(
        controller,
        run_id=run_id,
        fast_mode=fast_mode,
        request_id=request_id,
        owner_epoch=owner_epoch,
        batch_set=batch_set,
        batch_set_id=batch_set_id,
        cache_key=cache_key,
        callback_context=callback_context,
    )
    controller._on_simulation_error(error_msg, callback_identity=identity)


def _install_ready_batch_lane_pool(
    controller: SimulationController,
    pool: object,
    *,
    max_lanes: int,
) -> object:
    if not hasattr(pool, "ready_lane_count"):
        try:
            setattr(pool, "ready_lane_count", max(1, int(max_lanes)))
        except Exception:
            pass
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    return controller.parallel_batch.ensure_lane_pool(max_lanes=max(1, int(max_lanes)))


def _lane_outcome(
    set_id: str,
    payload: dict[str, Any] | None = None,
    *,
    run_id: int = 1,
    request_id: int = 2,
    success: bool | None = None,
    owner_epoch: int = 1,
    failure: dict[str, Any] | None = None,
) -> BatchLaneOutcome:
    payload_dict = dict(payload or {})
    if success is None:
        success = payload_dict.get("success") is not False
    return BatchLaneOutcome(
        lane_id="test-lane",
        run_id=int(run_id),
        request_id=int(request_id),
        set_id=str(set_id),
        owner_epoch=int(owner_epoch),
        success=bool(success),
        payload=payload_dict,
        failure=failure,
    )


def _timeout_failure_outcome(set_id: str, seconds: float = 0.2) -> BatchLaneOutcome:
    return _lane_outcome(
        set_id,
        {
            "success": False,
            "error": build_simulation_failure(
                "timeout",
                f"Simulation timed out after {seconds:.1f} seconds.",
                details={"walltime_s": seconds},
            ),
        },
    )


_WEGSCHEIDER_GUI_UNRESOLVED = "\n".join(
    [
        "equilibrium: PBMproduct <-> Methidequinone + Amine ; kf=1 ; K=2",
        "equilibrium: Methidequinone <-> Methidequinone_CIS ; kf=1 ; K=3",
        "equilibrium: Methidequinone_CIS + Amine <-> PBMproduct ; kf=1 ; K=7",
        "init: PBMproduct=1, Methidequinone=0, Methidequinone_CIS=0, Amine=0",
    ]
)


class _EditableText:
    def __init__(self, text: str = "") -> None:
        self._text = str(text)

    def toPlainText(self) -> str:
        return self._text

    def setPlainText(self, text: str) -> None:
        self._text = str(text)


class _MinimalMechanismEditor:
    def __init__(self, text: str) -> None:
        self._reactions_text = _EditableText(text)

    def slider_points_value(self) -> int:
        return 100

    def slider_solver_value(self) -> str:
        return "BDF"


def _install_active_lane_outcomes(
    controller: SimulationController,
    outcomes: dict[str, BatchLaneOutcome],
    *,
    set_names: dict[str, str] | None = None,
    owner_epoch: int = 1,
    callback_identities: dict[str, object] | None = None,
    with_callback_identity: bool = True,
    fast_mode: bool = False,
) -> None:
    class _StaticLanePool:
        def __init__(self, lane_outcomes: dict[str, BatchLaneOutcome]) -> None:
            self.lane_outcomes = dict(lane_outcomes)
            self.ready_lane_count = max(1, len(lane_outcomes))

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            _ = wait
            self.ready_lane_count = max(1, int(max_lanes))

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            _ = task, run_id, request_id, active_timeout_s
            return self.lane_outcomes[str(set_id)]

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            return None

    pool = _StaticLanePool(outcomes)
    controller._batch_parallel.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda _message, _exc: None,
    )
    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller._batch_parallel.ensure_lane_pool(max_lanes=max(1, len(outcomes)))
    controller._batch_parallel.begin_run(
        run_id=int(next(iter(outcomes.values())).run_id) if outcomes else 1,
        request_id=int(next(iter(outcomes.values())).request_id) if outcomes else 1,
        fast_mode=bool(fast_mode),
        queue_ids=[str(sid) for sid in outcomes],
        queue_names=[(set_names or {}).get(sid, sid) for sid in outcomes],
        keep_lane_pool_alive=False,
        preview_owner_epoch=int(owner_epoch) if bool(fast_mode) else None,
        active_timeout_s=1.0,
    )
    identity_by_set_id = callback_identities
    if identity_by_set_id is None and bool(with_callback_identity):
        first = next(iter(outcomes.values())) if outcomes else None
        callback_context = _test_callback_context(
            run_id=int(first.run_id) if first is not None else 1,
            request_id=int(first.request_id) if first is not None else 1,
            cache_key="ck",
            fast_mode=bool(fast_mode),
            parallel=True,
            queue_ids=tuple(str(sid) for sid in outcomes),
            queue_names=tuple((set_names or {}).get(sid, sid) for sid in outcomes),
        )
        identity_by_set_id = {
            str(sid): SimulationCallbackIdentity.capture(
                run_id=int(outcome.run_id),
                fast_mode=bool(fast_mode),
                request_id=int(outcome.request_id),
                owner_epoch=owner_epoch,
                batch_set=(set_names or {}).get(sid, sid),
                batch_set_id=str(sid),
                cache_key=str(callback_context["cache_key"]),
                callback_context=callback_context,
                simulation_identity=controller.batch_context_owner.simulation_identity_for_set(str(sid)),
                preview_batch_cache_token=controller.batch_context_owner.preview_batch_cache_token_for_set(str(sid)),
            )
            for sid, outcome in outcomes.items()
        }
    elif identity_by_set_id is None:
        identity_by_set_id = {
            str(sid): _malformed_callback_identity_without_context(
                run_id=int(outcome.run_id),
                fast_mode=bool(fast_mode),
                request_id=int(outcome.request_id),
                owner_epoch=owner_epoch,
                batch_set=(set_names or {}).get(sid, sid),
                batch_set_id=str(sid),
                cache_key="ck",
            )
            for sid, outcome in outcomes.items()
        }
    for sid, outcome in outcomes.items():
        controller._batch_parallel.submit_task(
            {},
            set_id=str(sid),
            set_name=(set_names or {}).get(sid, sid),
            expected_owner_epoch=owner_epoch,
            callback_identity=(identity_by_set_id or {}).get(str(sid)),
        )
    controller.parallel_batch.join_active_requests(timeout_s=2.0)


class _CompletedBatchHandle:
    def __init__(self, outcome: BatchLaneOutcome) -> None:
        self.outcome = outcome

    def is_done(self) -> bool:
        return True


def _consume_parallel_batch_outcome_for_test(
    controller: SimulationController,
    *,
    set_id: str,
    outcome: BatchLaneOutcome,
    run_id: int = 1,
    request_id: int = 2,
    fast_mode: bool = False,
    source: str = "test",
    completed_ts: float = 1.0,
) -> bool:
    _ = fast_mode
    request_metadata = controller._batch_parallel.active_request_metadata(str(set_id))
    completion_record = BatchCompletionRecord(
        metadata=BatchRequestMetadata(
            set_id=str(set_id),
            set_name=str(request_metadata.get("set_name") or set_id),
            run_id=int(run_id),
            request_id=int(request_id),
            generation=int(request_metadata.get("generation") or 0),
            preview_owner_epoch=request_metadata.get("preview_owner_epoch"),
            expected_owner_epoch=request_metadata.get("owner_epoch"),
        ),
        outcome=outcome,
        completed_ts=float(completed_ts),
        request_metadata=request_metadata,
    )
    return controller._consume_parallel_batch_outcome(
        set_id=set_id,
        outcome=outcome,
        source=source,
        completed_ts=completed_ts,
        completion_record=completion_record,
    )


@dataclass
class _FakeButton:
    enabled: bool = True

    def isEnabled(self) -> bool:
        return bool(self.enabled)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

@dataclass
class _FakeLabel:
    text: str = ""

    def setText(self, text: str) -> None:
        self.text = str(text)

    def repaint(self) -> None:
        return

@dataclass
class _FakeProgress:
    value: int = 0

    def setValue(self, value: int) -> None:
        self.value = int(value)

    def repaint(self) -> None:
        return

@dataclass
class _FakeSpinBox:
    _value: float

    def value(self) -> float:
        return float(self._value)

    def setValue(self, value: float) -> None:
        self._value = float(value)

class _FakeSignal:
    def __init__(self, *, disconnect_raises_typeerror: bool = False) -> None:
        self._handlers: list[Callable[..., Any]] = []
        self._disconnect_raises_typeerror = bool(disconnect_raises_typeerror)

    def connect(self, handler: Callable[..., Any]) -> None:
        self._handlers.append(handler)

    def disconnect(self) -> None:
        if self._disconnect_raises_typeerror:
            raise TypeError("not connected")
        self._handlers.clear()

class _FakeWorker(QtCore.QObject):
    def __init__(
        self,
        *,
        running: bool = False,
        wait_returns: bool = True,
        signal_disconnect_typeerror: bool = False,
    ) -> None:
        super().__init__()
        self._running = bool(running)
        self._wait_returns = bool(wait_returns)
        self._cancelled = False
        self._terminated = False

        self.progress = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)
        self.finished = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)
        self.error = _FakeSignal(disconnect_raises_typeerror=signal_disconnect_typeerror)

    def isRunning(self) -> bool:  # Qt-ish API
        return bool(self._running)

    def cancel(self) -> None:
        self._cancelled = True
        self._running = False

    def wait(self, _ms: Optional[int] = None) -> bool:
        return bool(self._wait_returns)

    def terminate(self) -> None:
        self._terminated = True
        self._running = False

    def start(self) -> None:
        self._running = True


class _FakeContainedOwner:
    def __init__(self) -> None:
        self.close_calls: list[bool] = []

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(bool(kill))


def _install_recording_contained_worker(
    monkeypatch,
    created: dict[str, object],
    controller: SimulationController | None = None,
) -> None:
    from kindred.core.simulation_plan import SimulationPlan

    class _RecordingContainedWorker:
        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ):
            self.owner = owner
            self.parent = parent
            self.simulation_plan_payload = dict(simulation_plan_payload)
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()
            plan = SimulationPlan.from_payload(self.simulation_plan_payload)
            request = plan.to_execution_request().to_payload()
            created["mechanism_text"] = str(request.get("mechanism_text") or "")
            created["initials"] = dict(request.get("initials") or {})
            created["t_span"] = tuple(request.get("t_span") or ())
            created["solver_config"] = dict(request.get("solver_config") or {})
            prepared = request.get("prepared_payload")
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = bool(include_mechanism_in_result_payload)

        def start(self) -> None:
            created["started"] = True

        def isRunning(self) -> bool:
            return False

        def cancel(self) -> None:
            created["cancelled"] = True

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _RecordingContainedWorker)
    if controller is not None:
        monkeypatch.setattr(
            controller._runtime_application,
            "acquire_ready_owner",
            lambda **_kwargs: "ready-contained-owner",
        )


@pytest.mark.unit
def test_unready_runtime_abort_does_not_persist_requested_run_disable(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    mw.set_runtime_backed_run_controls_ready(True)
    mw.set_run_button_enabled(False)
    mw._stop_btn.setEnabled(True)
    mw._sim_progress.setValue(47)
    seed_batch_context(controller.batch_context_owner, active=True)

    controller._abort_for_unready_interactive_runtime(
        fast_mode=False,
        context={"active": True, "rows": [0], "queue_ids": ["set1"]},
    )

    assert mw._run_button_requested_enabled is True
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 0
    assert mw._status_label.text == "Simulation runtime is not ready."
    assert mw._runtime_availability_refresh_requests == 1

    mw.set_runtime_backed_run_controls_ready(True)

    assert mw._run_btn.isEnabled() is True


class _QtSignalWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, str)
    result_ready = QtCore.Signal(dict)
    error = QtCore.Signal(object)

    def __init__(self, *, running: bool = True) -> None:
        super().__init__()
        self._running = bool(running)
        self.cancel_calls: list[None] = []
        self.wait_calls: list[int] = []
        self.progress.connect(lambda *_args: None)
        self.result_ready.connect(lambda *_args: None)

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_calls.append(None)

    def wait(self, ms: Optional[int] = None) -> bool:
        self.wait_calls.append(int(ms or 0))
        return False

def _successful_result_payload() -> dict[str, Any]:
    return {
        "t": np.linspace(0.0, 1.0, 3),
        "Y": np.asarray([[1.0, 0.5, 0.1], [0.0, 0.5, 0.9]], dtype=float),
        "species_names": ["A", "B"],
        "mechanism": None,
        "mechanism_text": "reaction: A -> B ; k=0.1",
        "solver_config": {"solver": "Radau", "temperature_K": 298.15},
        "algebra_scalars": {},
        "algebra_errors": [],
        "fallback_occurred": False,
        "fallback_message": None,
    }


def _pending_slider_preview_launch(controller: SimulationController):
    return controller.run_state.pending_slider_preview_launch

class _FakeMainWindow(QtCore.QObject):
    def settings_set_value(self, key: str, value: object) -> None:
        self._settings.setValue(str(key), value)

    def settings_sync(self) -> None:
        self._settings.sync()

    def run_button_is_enabled(self) -> bool:
        return bool(self._run_btn.isEnabled())

    def set_run_button_enabled(self, enabled: bool) -> None:
        self._run_button_requested_enabled = bool(enabled)
        self._run_btn.setEnabled(
            bool(self._run_button_requested_enabled)
            and bool(getattr(self, "_simulation_runtime_run_ready", True))
        )

    def set_runtime_backed_run_controls_ready(self, ready: bool) -> None:
        self._simulation_runtime_run_ready = bool(ready)
        self._run_btn.setEnabled(
            bool(getattr(self, "_run_button_requested_enabled", True))
            and bool(self._simulation_runtime_run_ready)
        )

    def schedule_runtime_availability_refresh(self) -> None:
        self._runtime_availability_refresh_requests = int(
            getattr(self, "_runtime_availability_refresh_requests", 0) or 0
        ) + 1

    def set_stop_button_enabled(self, enabled: bool) -> None:
        self._stop_btn.setEnabled(bool(enabled))

    def set_status_text(self, text: str) -> None:
        self._status_label.setText(str(text))

    def set_sim_progress_value(self, value: int) -> None:
        self._sim_progress.setValue(int(value))

    def repaint_simulation_widgets(self) -> None:
        self._sim_progress.repaint()
        self._status_label.repaint()

    def set_algebra_status_text(self, text: str, *, details: Optional[str] = None) -> None:
        self._algebra_status_label.setText(str(text))
        if hasattr(self._algebra_status_label, "setToolTip"):
            self._algebra_status_label.setToolTip(str(details or ""))

    def message_box_warning(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(None, str(title), str(message))

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None:
        full_message = str(message)
        if details:
            full_message = f"{full_message}\n\nDetails:\n{details}"
        QtWidgets.QMessageBox.critical(None, str(title), full_message)

    def message_box_question(self, title: str, message: str, *, accept_label: str = "Apply") -> bool:
        self._message_box_questions.append(
            {
                "title": str(title),
                "message": str(message),
                "accept_label": str(accept_label),
            }
        )
        return bool(self._message_box_question_response)

    def choose_wegscheider_resolution(
        self,
        title: str,
        message: str,
        choices: dict[str, list[dict[str, str]]],
    ) -> dict[str, str] | None:
        self._wegscheider_resolution_choice_prompts.append(
            {
                "title": str(title),
                "message": str(message),
                "choices": {
                    str(cycle_id): [dict(item) for item in options]
                    for cycle_id, options in dict(choices or {}).items()
                },
            }
        )
        if self._wegscheider_resolution_choice_response is None:
            return None
        return dict(self._wegscheider_resolution_choice_response)

    def mechanism_reactions_text_raw(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is not None and hasattr(editor, "_reactions_text"):
            return str(editor._reactions_text.toPlainText())
        return str(self._get_mechanism_text() or "")

    def mechanism_state_network_dsl_raw(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        state_editor = getattr(editor, "_state_network_editor", None)
        if state_editor is not None and hasattr(state_editor, "get_state_network_dsl"):
            return str(state_editor.get_state_network_dsl() or "")
        return ""

    def mechanism_slider_points_value(self) -> Optional[int]:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "slider_points_value"):
            return None
        try:
            return int(editor.slider_points_value())
        except Exception:
            return None

    def mechanism_slider_solver_value(self) -> Optional[str]:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "slider_solver_value"):
            return None
        try:
            value = editor.slider_solver_value()
        except Exception:
            return None
        return str(value) if value is not None else None

    def auto_lock_for_run(self) -> bool:
        self._auto_lock_for_run_calls = int(getattr(self, "_auto_lock_for_run_calls", 0)) + 1
        return bool(getattr(self, "_auto_lock_for_run_result", True))

    def is_mechanism_ready_for_run(self) -> bool:
        self._is_mechanism_ready_for_run_calls = int(getattr(self, "_is_mechanism_ready_for_run_calls", 0)) + 1
        return bool(getattr(self, "_is_mechanism_ready_for_run_result", True))

    def temperature_spinbox_value(self) -> float:
        return float(self._temperature_spinbox.value())

    def num_points_spinbox_value(self) -> int:
        return int(self._num_points_spinbox.value())

    def sim_time_spinbox_text(self) -> str:
        return str(self._sim_time_spinbox.text())

    def parse_sim_time_seconds(self) -> float:
        return float(self._parse_sim_time_seconds())

    def use_sparse_jacobian(self) -> bool:
        return bool(self._use_sparse_jacobian)

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_cyclicity_enabled)

    def main_plot(self) -> object:
        return self._plot_tabs._main_plot

    def set_results_table(self, table: object) -> None:
        self._results_table = table

    def sync_main_plot_copy_labels(self, primary_set_id: str, selected_set_ids) -> None:
        pass

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._temperature_mode_indicator.setText(str(text))

    def set_mechanism_reactions_text_with_optional_undo(
        self,
        new_text: str,
        description: str,
        *,
        record_undo: bool,
    ) -> None:
        self._set_text_with_optional_undo(
            self._mechanism_editor._reactions_text,
            str(new_text),
            str(description),
            bool(record_undo),
        )

    def apply_wegscheider_resolution_source_rewrite(self, reactions_text: str) -> None:
        self._wegscheider_resolution_rewrites.append(str(reactions_text))
        self.set_mechanism_reactions_text_with_optional_undo(
            str(reactions_text),
            "Resolve Wegscheider cyclicity",
            record_undo=True,
        )

    def stop_slider_release_commit_timer(self) -> None:
        timer = self._slider_release_commit_timer
        if timer is not None and timer.isActive():
            timer.stop()

    def has_pending_slider_values(self) -> bool:
        return bool(self._pending_slider_values)

    def finalize_slider_release_commit(self) -> None:
        self._finalize_slider_release_commit()

    def stop_variable_update_timer(self) -> None:
        timer = self._variable_update_timer
        if timer is not None:
            timer.stop()

    def stop_species_slider_update_timer(self) -> None:
        timer = getattr(self, "_species_slider_update_timer", None)
        if timer is not None:
            timer.stop()

    def set_slider_triggered_simulation(self, value: bool) -> None:
        self._slider_triggered_simulation = bool(value)

    def slider_triggered_simulation(self) -> bool:
        return bool(self._slider_triggered_simulation)

    def last_slider_change_name(self) -> str:
        return str(self._last_slider_change_name or "")

    def slider_drag_active(self) -> bool:
        return bool(self._slider_drag_active)

    def suppress_slider_refresh(self) -> bool:
        return bool(self._suppress_slider_refresh)

    def slider_gesture_target_set_ids_snapshot(self) -> list[str]:
        return [str(set_id) for set_id in getattr(self, "_slider_gesture_target_set_ids_snapshot", [])]

    def preview_initials_for_row(self, row: int, baseline: dict[str, float]) -> dict[str, float]:
        _ = row
        return {str(key): float(value) for key, value in dict(baseline or {}).items()}

    def preview_batch_cache_token(self, rows: list[int]) -> str:
        _ = rows
        return ""

    def show_preview_unavailable_for_dirty_state(self, message: str) -> None:
        self._preview_unavailable_messages.append(str(message))

    def _remember_last_mechanism(self, mechanism: object, dsl_text: str, solver_config: dict[str, Any]) -> None:
        self._mechanism_helpers.remember_last_mechanism(mechanism, dsl_text, solver_config)

    def _clear_last_mechanism(self) -> None:
        self._mechanism_helpers.clear_last_mechanism()

    def last_mechanism(self) -> Optional[object]:
        return self._mechanism_helpers.last_mechanism()

    def last_mechanism_context(self) -> dict[str, Any]:
        return self._mechanism_helpers.last_mechanism_context()

    def batch_rows_for_scope(self, scope: str) -> list[int]:
        return [int(row) for row in (self._batch_rows_for_scope(str(scope)) or [])]

    def batch_set_ids_for_scope(self, scope: str) -> list[str]:
        return [str(set_id) for set_id in (self._batch_set_ids_for_scope(str(scope)) or [])]

    def shown_batch_set_ids(self) -> list[str]:
        return [str(set_id) for set_id in (self._shown_batch_set_ids() or [])]

    def batch_current_row(self) -> Optional[int]:
        row = self._batch_current_row()
        return int(row) if row is not None else None

    def batch_set_id_for_row(self, row: int) -> Optional[str]:
        value = self._batch_set_id_for_row(int(row))
        return str(value) if value is not None else None

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]:
        value = self._batch_set_name_for_id(str(set_id))
        return str(value) if value is not None else None

    def batch_set_id_for_name(self, name: str) -> Optional[str]:
        value = self._batch_set_id_for_name(str(name))
        return str(value) if value is not None else None

    def batch_preferred_primary_set_id(self, rows: list[int]) -> Optional[str]:
        value = self._batch_preferred_primary_set_id(list(rows))
        return str(value) if value is not None else None

    def set_active_batch_selection(self, set_id: str, set_name: str, selected_ids: list[str]) -> None:
        _ = (set_id, set_name, selected_ids)

    def clear_display_selection_state(self) -> None:
        return None

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: dict[str, Any] | None = None,
        t_end: float = 0.0,
    ) -> str:
        if scope_identity is not None and hasattr(scope_identity, "cache_key"):
            return str(scope_identity.cache_key())
        return str(
            self._batch_cache_key(
                mechanism_text=str(mechanism_text),
                solver_config=dict(solver_config or {}),
                t_end=float(t_end),
            )
        )

    def batch_store_row_count(self) -> int:
        return int(self._batch_store.row_count())

    def batch_store_set_names(self) -> list[str]:
        return [str(name) for name in (self._batch_store.set_names() or [])]

    def batch_store_visible_species(self) -> list[str]:
        return [str(species) for species in (self._batch_store.visible_species() or [])]

    def batch_model_validate_rows(self, rows: list[int]) -> set[tuple[int, str]]:
        invalid = self._batch_model.validate_rows(list(rows))
        if not invalid:
            return set()
        return {(int(row), str(species)) for row, species in invalid}

    def batch_initials_for_row(self, row: int) -> dict[str, float]:
        initials = self._batch_initials_for_row(int(row))
        if not initials:
            return {}
        return {str(key): float(value) for key, value in dict(initials).items()}

    def display_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: list[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[object] = None,
        valid_set_ids: Optional[list[str] | tuple[str, ...]] = None,
        allow_fallback: bool = True,
    ) -> bool:
        return bool(
            self._display_cached_batch_selection(
                cache_key=str(cache_key),
                selected_sets=list(selected_sets),
                prefer_set=str(prefer_set) if prefer_set is not None else None,
                cache_store=cache_store,
                valid_set_ids=(
                    tuple(str(set_id) for set_id in valid_set_ids)
                    if valid_set_ids is not None
                    else None
                ),
                allow_fallback=bool(allow_fallback),
            )
        )

    def publish_simulation_completion_result(
        self,
        *,
        t: object,
        series: dict[str, object],
        cache_key: Optional[str],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        selected_sets: list[str],
        prefer_set: Optional[str],
        redraw_valid_set_ids: Optional[list[str] | tuple[str, ...]],
        has_redraw_subset: bool,
        slider_triggered: bool,
        explicit_batch_coalescing: bool,
        algebra_scalars: Optional[dict[str, object]],
        owned_species: Optional[list[str]] = None,
    ) -> bool:
        displayed = False
        if cache_key:
            displayed = self.display_cached_batch_selection(
                cache_key=str(cache_key),
                selected_sets=list(selected_sets),
                prefer_set=prefer_set,
                cache_store=None,
                valid_set_ids=(redraw_valid_set_ids if has_redraw_subset else None),
                allow_fallback=False,
            )
        if displayed:
            return True
        if str(batch_set_id or "").strip():
            self.set_active_batch_selection(str(batch_set_id), str(batch_set or ""), [str(batch_set_id)])
        else:
            self.clear_display_selection_state()
        self.set_data(
            t,
            series,
            label=(str(batch_set) if batch_set else None),
            overlays=[],
            owned_species=owned_species,
        )
        self.sync_main_plot_copy_labels(
            str(batch_set_id or ""),
            [str(batch_set_id)] if str(batch_set_id or "").strip() else [],
        )
        plot = self.main_plot()
        if hasattr(plot, "set_scalar_values"):
            plot.set_scalar_values(dict(algebra_scalars or {}))
        display_label = str(batch_set) if batch_set else (str(batch_set_id) if batch_set_id else "Results")
        if hasattr(plot, "set_statistics_results"):
            plot.set_statistics_results({display_label: {"t": t, "series": series}}, prefer=display_label)
        elif hasattr(plot, "update_statistics"):
            plot.update_statistics(t, series)
        self.set_results_table(plot.stats_table())
        return True

    def publish_completion_intervention_annotations(
        self,
        solver_provenance: Optional[dict[str, Any]],
    ) -> None:
        plot = self.main_plot()
        setter = getattr(plot, "set_intervention_annotations_from_provenance", None)
        if callable(setter):
            setter(solver_provenance if isinstance(solver_provenance, dict) else None)

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(self, species_names: list[str], *, preserve_active_cache: bool = False) -> None:
        self._sync_batch_species_columns(list(species_names), preserve_active_cache=bool(preserve_active_cache))

    def has_slider_overrides(self) -> bool:
        return bool(self._slider_overrides)

    def variable_slider_values(self) -> dict[str, float]:
        return {str(key): float(value) for key, value in dict(self._variable_slider_values or {}).items()}

    def simulation_schema_id(self) -> str:
        return str(getattr(self, "_simulation_schema_id", "schema-default"))

    def simulation_param_fingerprint(self, set_id: Optional[str] = None) -> str:
        mapping = getattr(self, "_simulation_param_fingerprints", None)
        if isinstance(mapping, dict):
            set_id_s = str(set_id or "")
            if set_id_s in mapping:
                return str(mapping[set_id_s])
            if "" in mapping:
                return str(mapping[""])
        return str(getattr(self, "_simulation_param_fingerprint", "params-default"))

    def slider_overrides(self, set_id: Optional[str] = None) -> dict[str, float]:
        _ = set_id
        overrides: dict[str, float] = {}
        for key, value in (self._slider_overrides or {}).items():
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(parsed):
                continue
            overrides[str(key)] = float(parsed)
        return overrides

    def apply_overrides_to_text(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        _ = set_id
        return str(self._apply_overrides_to_text(str(base_text)))

    def apply_overrides_to_state_network_dsl(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        _ = set_id
        return str(self._apply_overrides_to_state_network_dsl(str(base_text)))

    def get_mechanism_text(self) -> str:
        return str(self._get_mechanism_text() or "")

    def initial_solver_name(self) -> Optional[str]:
        solver = getattr(self, "_initial_solver", None)
        return str(solver) if solver is not None else None

    def initial_rtol(self) -> Optional[float]:
        value = getattr(self, "_initial_rtol", None)
        return float(value) if value is not None else None

    def initial_atol(self) -> Optional[float]:
        value = getattr(self, "_initial_atol", None)
        return float(value) if value is not None else None

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]:
        value = self._dsl_global_temperature_K(str(dsl_text))
        return float(value) if value is not None else None

    def prepare_slider_runtime(
        self,
        param_names: Optional[list[str]] = None,
        *,
        set_id: Optional[str] = None,
    ) -> Optional[object]:
        return self._prepare_slider_runtime(param_names=param_names, set_id=set_id)

    def apply_slider_overrides_to_bindings(self, runtime: object, *, set_id: Optional[str] = None) -> bool:
        return bool(self._apply_slider_overrides_to_bindings(runtime, set_id=set_id))

    def set_slider_runtime_dirty(self, value: bool) -> None:
        self._slider_runtime_dirty = bool(value)

    def snapshot_datasets(self) -> dict[str, Any]:
        return dict(self._snapshot_datasets() or {})

    def last_fit_metadata(self) -> Optional[dict[str, Any]]:
        value = self._last_fit_metadata
        return dict(value) if isinstance(value, dict) else None

    def set_last_simulation_provenance(self, provenance: dict[str, Any]) -> None:
        self._last_simulation_provenance = dict(provenance)

    def set_last_simulation_ctc(self, ctc: dict[str, float]) -> None:
        self._last_simulation_ctc = {str(key): float(value) for key, value in (ctc or {}).items()}

    def integrate_ctc(
        self,
        t: object,
        y: object,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> tuple[float, str, bool, float, str]:
        _ = (t, y, uniformity_eps, tail_strategy)
        return (1.0, "trapz", True, float(uniformity_eps), str(tail_strategy))

    def publish_simulation_completion_provenance(
        self,
        *,
        mechanism_text: str,
        solver_method: str,
        solver_label: str,
        solver_warning: Optional[str],
        solver_config: dict[str, Any],
        temperature_K: float,
        temperature_source: str,
        energy_unit: Optional[str],
        energy_mode: bool,
        simulation_time: float | str,
        num_points_requested: int,
        species_names: list[str],
        t: object,
        series: dict[str, object],
        algebra_scalars: Optional[dict[str, object]] = None,
        dataset_overlays: object = None,
        solver_provenance: Optional[dict[str, object]] = None,
        warnings: Optional[list[dict[str, object]]] = None,
    ) -> dict[str, Any]:
        ctc_values = {str(name): 1.0 for name in series}
        self.set_last_simulation_ctc(ctc_values)
        provenance = {
            "mechanism_dsl": str(mechanism_text),
            "solver": str(solver_method),
            "solver_label": str(solver_label),
            "solver_warning": str(solver_warning) if solver_warning else None,
            "rtol": solver_config.get("rtol", 1e-6),
            "atol": solver_config.get("atol", 1e-12),
            "temperature_K": float(temperature_K),
            "temperature_source": str(temperature_source),
            "energy_unit": energy_unit,
            "energy_mode": bool(energy_mode),
            "simulation_time": simulation_time,
            "num_points_requested": int(num_points_requested),
            "num_species": len(species_names),
            "num_points": len(t),
            "species_names": list(species_names),
            "datasets": self.snapshot_datasets(),
            "ctc": {
                "integration_method": "trapz",
                "uniform_grid_detected": True,
                "uniformity_eps": 1e-6,
                "tail_strategy": "38",
            },
        }
        if algebra_scalars:
            provenance["algebra_scalars"] = dict(algebra_scalars)
        if dataset_overlays is not None:
            provenance["dataset_overlays"] = dataset_overlays
        if solver_provenance:
            provenance["solver_provenance"] = dict(solver_provenance)
            symbolic_identity = solver_provenance.get("symbolic_jacobian_identity")
            if isinstance(symbolic_identity, dict):
                provenance["symbolic_jacobian_identity"] = dict(symbolic_identity)
        if warnings:
            provenance["warnings"] = [dict(item) for item in warnings]
        fit_meta = self.last_fit_metadata()
        if fit_meta:
            provenance["fit"] = fit_meta
        self.set_last_simulation_provenance(provenance)
        return provenance

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: dict[str, Any]) -> None:
        self._remember_last_mechanism(mechanism, str(mechanism_text), dict(solver_config))

    def is_energy_mode_mechanism(self, mechanism: object) -> bool:
        return bool(self._is_energy_mode_mechanism(mechanism))

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool:
        return bool(self._dsl_has_computational_mode_generated_block(str(mechanism_text)))

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        self._sync_energy_mode_temperature_from_mechanism(mechanism)

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self._populate_energy_mode_variables_from_mechanism(
            mechanism,
            refresh_sliders=bool(refresh_sliders),
            preserve_visibility=bool(preserve_visibility),
        )

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None:
        self._extract_and_populate_variables(preserve_visibility=bool(preserve_visibility))

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        self._sync_mechanism_controls_to_focused_batch_set(use_workspace=bool(use_workspace))

    def apply_pending_init_migration(self, *, seed_sets: dict[str, dict[str, float]], rewrite: str) -> bool:
        if not seed_sets or not rewrite:
            return False
        for set_name, seed in dict(seed_sets).items():
            row_idx = self._batch_store.ensure_set(str(set_name))
            for species, value in dict(seed).items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not np.isfinite(parsed):
                    continue
                self._batch_store.set_value(row_idx, str(species), f"{float(parsed):.6g}")
        self.set_mechanism_reactions_text_with_optional_undo(
            str(rewrite),
            "Migrate initial concentrations to batch table",
            record_undo=True,
        )
        return True

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        self._invalidate_pending_init_preserved_results_after_failed_run()

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None:
        self._arm_pending_init_result_invalidation_guard(rewrite=rewrite)

@pytest.fixture
def mw(qt_app) -> _FakeMainWindow:
    _ = qt_app
    window = _FakeMainWindow()
    window._settings = MagicMock()

    window._run_btn = _FakeButton(True)
    window._run_button_requested_enabled = True
    window._simulation_runtime_run_ready = True
    window._runtime_availability_refresh_requests = 0
    window._stop_btn = _FakeButton(False)
    window._status_label = _FakeLabel()
    window._algebra_status_label = _FakeLabel()
    window._sim_progress = _FakeProgress()
    window._temperature_spinbox = _FakeSpinBox(298.15)
    window._num_points_spinbox = _FakeSpinBox(100.0)
    window._initial_solver = "BDF"
    window._initial_rtol = 1e-6
    window._initial_atol = 1e-12
    window._last_fit_metadata = None
    window._last_simulation_provenance = {}
    window._last_simulation_ctc = {}

    window._slider_triggered_simulation = False
    window._pending_slider_values = {}
    window._slider_overrides = {}
    window._variable_slider_values = {}
    window._slider_drag_active = False
    window._slider_gesture_target_set_ids_snapshot = []
    window._last_slider_change_name = ""
    window._slider_runtime_dirty = False
    window._preview_unavailable_messages = []
    window._use_sparse_jacobian = False
    window._wegscheider_cyclicity_enabled = False
    window._message_box_question_response = True
    window._message_box_questions = []
    window._wegscheider_resolution_rewrites = []
    window._wegscheider_resolution_choice_response = None
    window._wegscheider_resolution_choice_prompts = []

    window._batch_store = MagicMock()
    window._batch_store.row_count.return_value = 0
    window._batch_store.set_names.return_value = []
    window._batch_store.visible_species.return_value = []

    window._batch_set_ids_for_scope = MagicMock(return_value=[])
    window._shown_batch_set_ids = MagicMock(return_value=[])
    window._batch_rows_for_scope = MagicMock(return_value=[])
    window._batch_current_row = MagicMock(return_value=None)
    window._batch_set_id_for_row = MagicMock(return_value=None)
    window._batch_set_id_for_name = MagicMock(return_value=None)
    window._batch_set_name_for_id = MagicMock(return_value=None)
    window._batch_preferred_primary_set_id = MagicMock(return_value=None)
    window._batch_cache_key = MagicMock(return_value="cache-key")
    window._simulation_schema_id = "schema-default"
    window._simulation_param_fingerprints = {"": "params-default"}
    window._auto_lock_for_run_result = True
    window._auto_lock_for_run_calls = 0
    window._is_mechanism_ready_for_run_result = True
    window._is_mechanism_ready_for_run_calls = 0

    window._display_cached_batch_selection = MagicMock(return_value=False)
    window._flush_slider_plot_updates = MagicMock(return_value=False)
    window.set_data = MagicMock()
    window._plot_tabs = MagicMock()
    window._plot_tabs._main_plot = MagicMock()
    window._plot_tabs._main_plot.stats_table.return_value = MagicMock()
    window._results_table = MagicMock()
    window._results_table.viewport.return_value = MagicMock()

    window._parse_sim_time_seconds = MagicMock(return_value=10.0)
    window._dsl_global_temperature_K = MagicMock(return_value=None)
    window._sync_batch_species_columns = MagicMock()
    window._sim_time_spinbox = MagicMock()
    window._sim_time_spinbox.text.return_value = "10.0"
    window._snapshot_datasets = MagicMock(return_value={})

    window._prepare_slider_runtime = MagicMock(return_value=None)
    window._apply_slider_overrides_to_bindings = MagicMock(return_value=False)
    window._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    window._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text: str(text))
    window.reset_mechanism_workspaces = MagicMock(return_value=False)
    window.discard_concentration_overlays_for_set_ids = MagicMock(return_value=False)
    window.discard_concentration_overlays_for_rows = MagicMock(return_value=False)
    window._dirty_state_generations = {}
    window.has_dirty_state_for_set = MagicMock(
        side_effect=lambda set_id: int(window._dirty_state_generations.get(str(set_id or ""), 1) or 0) > 0
    )
    window.dirty_state_generation = MagicMock(
        side_effect=lambda set_id: int(window._dirty_state_generations.get(str(set_id or ""), 1) or 0)
    )
    window._run_simulation_internal = MagicMock()

    window._get_mechanism_text = MagicMock(return_value="")
    window._is_energy_mode_mechanism = MagicMock(return_value=False)
    window._dsl_has_computational_mode_generated_block = MagicMock(return_value=False)
    window._sync_energy_mode_temperature_from_mechanism = MagicMock()
    window._set_temperature_override_state = MagicMock()
    window._update_temperature_mode_indicator = MagicMock()
    window._temperature_mode_indicator = _FakeLabel()

    window._remember_last_mechanism = MagicMock()
    window._populate_energy_mode_variables_from_mechanism = MagicMock()
    window._extract_and_populate_variables = MagicMock()
    window._sync_mechanism_controls_to_focused_batch_set = MagicMock()
    window._update_batch_row_controls_state = MagicMock()
    window._batch_model = MagicMock()
    window._batch_model.columnCount.return_value = 1
    window._batch_model.index.side_effect = lambda *_args, **_kwargs: object()
    window._batch_model.dataChanged = MagicMock()
    window._batch_model.dataChanged.emit = MagicMock()
    window._batch_model.validate_rows = MagicMock(return_value=set())

    window._set_text_with_optional_undo = MagicMock()
    window._invalidate_pending_init_preserved_results_after_failed_run = MagicMock()
    window._arm_pending_init_result_invalidation_guard = MagicMock()
    window._suppress_slider_runtime_invalidation = False
    window._suppress_slider_refresh = False

    window._variable_update_timer = MagicMock()
    window._variable_update_timer.isActive.return_value = False
    window._species_slider_update_timer = MagicMock()
    window._species_slider_update_timer.isActive.return_value = False
    window._slider_release_commit_timer = MagicMock()
    window._slider_release_commit_timer.isActive.return_value = False
    window._finalize_slider_release_commit = MagicMock()

    window._batch_initials_for_row = MagicMock(return_value={})
    window._variable_runtime = window
    window._mechanism_helpers = MainWindowMechanismHelpers(window)

    return window

@pytest.fixture
def controller(mw: _FakeMainWindow) -> SimulationController:
    ui = SimulationUiPorts(
        dialogs=mw,
        settings=mw,
        run_ui=mw,
        slider=mw,
        batch=mw,
        mechanism=mw,
        solver=mw,
        runtime=mw,
        results=mw,
        provenance=mw,
        mechanism_helpers=mw._mechanism_helpers,
    )
    c = SimulationController(ui, parent=mw)
    try:
        yield c
    finally:
        with suppress(RuntimeError, TypeError):
            c._shutdown_batch_lane_pool(force_terminate=True)
        with suppress(RuntimeError, TypeError):
            c._close_contained_simulation_owner(kill=True)
        timer = getattr(c, "_slider_plot_coalesce_timer", None)
        with suppress(RuntimeError, TypeError):
            if timer is not None and timer.isActive():
                timer.stop()
        timer = getattr(c, "_batch_completion_poll_timer", None)
        with suppress(RuntimeError, TypeError):
            if timer is not None and timer.isActive():
                timer.stop()


def _configure_single_selected_set(mw: _FakeMainWindow) -> None:
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_store.visible_species.return_value = ["PBMproduct"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_ids_for_scope.return_value = ["set1"]
    mw._shown_batch_set_ids.return_value = ["set1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "set1"
    mw._batch_set_name_for_id.return_value = "set1"
    mw._batch_preferred_primary_set_id.return_value = "set1"


def _install_mechanism_editor_text(mw: _FakeMainWindow, text: str) -> None:
    editor = _MinimalMechanismEditor(str(text))
    mw._mechanism_editor = editor
    mw._get_mechanism_text = MagicMock(return_value=str(text))

    def _set_text_with_optional_undo(widget: object, new_text: str, _description: str, _record_undo: bool) -> None:
        widget.setPlainText(str(new_text))
        mw._get_mechanism_text = MagicMock(return_value=str(new_text))

    mw._set_text_with_optional_undo = MagicMock(side_effect=_set_text_with_optional_undo)


@pytest.mark.unit
def test_run_accepts_wegscheider_resolution_and_rewrites_source_before_dispatch(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    _configure_single_selected_set(mw)
    _install_mechanism_editor_text(mw, _WEGSCHEIDER_GUI_UNRESOLVED)
    mw._wegscheider_cyclicity_enabled = True
    mw._wegscheider_resolution_choice_response = {"cycle_1": "Keq3"}
    controller._selected_run_runtime_snapshot = MagicMock(
        return_value=SimpleNamespace(required=False, ready=True, message="")
    )
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._wegscheider_resolution_choice_prompts
    assert mw._wegscheider_resolution_choice_prompts[0]["title"] == "Resolve Wegscheider Cyclicity"
    assert len(mw._wegscheider_resolution_rewrites) == 1
    rewritten = mw.mechanism_reactions_text_raw()
    assert "param Keq3 = 1 / (Keq1 * Keq2)" in rewritten
    assert "Running simulation..." == mw._status_label.text
    controller.run_simulation_internal.assert_called_once()


@pytest.mark.unit
def test_run_cancelled_wegscheider_resolution_preserves_source_and_blocks_dispatch(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    _configure_single_selected_set(mw)
    _install_mechanism_editor_text(mw, _WEGSCHEIDER_GUI_UNRESOLVED)
    mw._wegscheider_cyclicity_enabled = True
    mw._wegscheider_resolution_choice_response = None
    controller._selected_run_runtime_snapshot = MagicMock(
        return_value=SimpleNamespace(required=False, ready=True, message="")
    )
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._wegscheider_resolution_choice_prompts
    assert mw._wegscheider_resolution_rewrites == []
    assert mw.mechanism_reactions_text_raw() == _WEGSCHEIDER_GUI_UNRESOLVED
    assert "Run cancelled: unresolved Wegscheider cyclicity." == mw._status_label.text
    controller.run_simulation_internal.assert_not_called()


@pytest.mark.unit
def test_run_uses_user_selected_wegscheider_dependent_parameter(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    _configure_single_selected_set(mw)
    _install_mechanism_editor_text(mw, _WEGSCHEIDER_GUI_UNRESOLVED)
    mw._wegscheider_cyclicity_enabled = True
    mw._wegscheider_resolution_choice_response = {"cycle_1": "Keq2"}
    controller._selected_run_runtime_snapshot = MagicMock(
        return_value=SimpleNamespace(required=False, ready=True, message="")
    )
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    prompt = mw._wegscheider_resolution_choice_prompts[0]
    offered_parameters = {
        item["parameter_name"]
        for item in prompt["choices"]["cycle_1"]
    }
    assert offered_parameters == {"Keq1", "Keq2", "Keq3"}
    rewritten = mw.mechanism_reactions_text_raw()
    assert "param Keq2 =" in rewritten
    assert "param Keq3 =" not in rewritten
    controller.run_simulation_internal.assert_called_once()


@pytest.mark.unit
def test_simulation_identity_for_supported_symbolic_jacobian_uses_actual_artifact(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    from kindred.core.simulation_preparation import prepare_simulation_worker_run

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "# Algebra",
            "param k1 = 0.5",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    solver_config = {
        "solver": "BDF",
        "grid": {"N": 12},
        "temperature_K": 298.15,
        "use_sparse_jacobian": True,
    }
    _install_mechanism_editor_text(mw, mechanism_text)

    identity = controller._simulation_identity_for_set(
        set_id="set1",
        solver_config=solver_config,
        t_end=5.0,
        fast_mode=False,
    )
    request_mechanism_text = controller._request_mechanism_text_for_set(
        set_id="set1",
        has_slider_overrides=False,
    )
    prepared = prepare_simulation_worker_run(
        mechanism_text=request_mechanism_text,
        initials={},
        t_span=(0.0, 5.0),
        solver_config=solver_config,
    )

    symbolic = identity.to_payload()["symbolic_jacobian_identity"]
    expected = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity")
    assert symbolic["kind"] == "jacobian"
    assert symbolic["backend_name"] == "sympy"
    assert symbolic["artifact_fingerprint"]
    assert symbolic["source_fingerprint"]
    assert symbolic == expected


@pytest.mark.unit
def test_symbolic_jacobian_identity_respects_wegscheider_cyclicity_mode(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    _install_mechanism_editor_text(mw, _WEGSCHEIDER_GUI_UNRESOLVED)
    base_solver_config = {
        "solver": "BDF",
        "grid": {"N": 12},
        "temperature_K": 298.15,
        "use_sparse_jacobian": True,
    }

    cyclicity_off_identity = controller._symbolic_jacobian_identity_for_set(
        set_id="set1",
        solver_config={**base_solver_config, "wegscheider_cyclicity_enabled": False},
        fast_mode=False,
    )
    cyclicity_on_identity = controller._symbolic_jacobian_identity_for_set(
        set_id="set1",
        solver_config={**base_solver_config, "wegscheider_cyclicity_enabled": True},
        fast_mode=False,
    )

    assert cyclicity_off_identity["kind"] == "jacobian"
    assert cyclicity_on_identity == {}


@pytest.mark.unit
def test_fast_preview_identity_includes_symbolic_snapshot_for_slider_parameter_overrides(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    _install_mechanism_editor_text(mw, mechanism_text)
    mw._slider_overrides = {"k1": 0.4}
    mw._simulation_param_fingerprints = {"set1": "slider-k1"}

    identity = controller._simulation_identity_for_set(
        set_id="set1",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 12},
            "temperature_K": 298.15,
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
        t_end=5.0,
        fast_mode=True,
    )

    symbolic_identity = identity.to_payload()["symbolic_jacobian_identity"]
    assert symbolic_identity["kind"] == "jacobian"
    assert symbolic_identity["parameter_symbols"] == ["k1"]
    assert symbolic_identity["evaluation_snapshot_fingerprint"]

    mw._slider_overrides = {"k1": 0.8}
    mw._simulation_param_fingerprints = {"set1": "slider-k1-updated"}
    updated_identity = controller._simulation_identity_for_set(
        set_id="set1",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 12},
            "temperature_K": 298.15,
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
        t_end=5.0,
        fast_mode=True,
    )
    updated_symbolic_identity = updated_identity.to_payload()["symbolic_jacobian_identity"]
    assert symbolic_identity["structure_fingerprint"] == updated_symbolic_identity["structure_fingerprint"]
    assert (
        symbolic_identity["evaluation_snapshot_fingerprint"]
        != updated_symbolic_identity["evaluation_snapshot_fingerprint"]
    )


@pytest.mark.unit
def test_symbolic_jacobian_identity_normalizes_indexed_k_direct_spelling_for_irreversible_parameter(
    controller: SimulationController,
    mw: _FakeMainWindow,
) -> None:
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    _install_mechanism_editor_text(mw, mechanism_text)
    mw._slider_overrides = {"K1": 0.4}

    identity = controller._symbolic_jacobian_identity_for_set(
        set_id="set1",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 12},
            "temperature_K": 298.15,
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
        fast_mode=True,
    )

    assert identity["parameter_symbols"] == ["k1"]
    assert identity["evaluation_snapshot_fingerprint"]


@pytest.mark.unit
def test_batch_preview_identity_includes_symbolic_snapshot_for_slider_parameter_overrides() -> None:
    from kindred.gui.simulation_batch_owner import SimulationBatchOwner

    mechanism_owner = MagicMock()
    mechanism_owner.has_slider_overrides.return_value = True
    mechanism_owner.slider_overrides.return_value = {"k1": 0.4}
    owner = SimulationBatchOwner(
        batch_rows_for_scope=lambda _scope: [],
        batch_set_ids_for_scope=lambda _scope: [],
        shown_batch_set_ids=lambda: [],
        slider_edit_target_set_ids=lambda: [],
        focused_batch_set_id=lambda: None,
        batch_current_row=lambda: None,
        batch_set_id_for_row=lambda _row: None,
        batch_set_name_for_id=lambda _set_id: None,
        batch_set_id_for_name=lambda _name: None,
        batch_preferred_primary_set_id=lambda _rows: None,
        batch_cache_key=lambda *args, **kwargs: "cache",
        batch_cache_getter=lambda: None,
        batch_store=MagicMock(),
        batch_model=MagicMock(),
        batch_initials_for_row=lambda _row: {},
        preview_session=MagicMock(),
        mechanism_owner=mechanism_owner,
        solver_owner=MagicMock(),
        results_controller_getter=lambda: None,
        set_status_text=lambda _text: None,
        update_batch_row_controls_state=lambda: None,
        sync_batch_species_columns=lambda *args, **kwargs: None,
    )

    payload = owner._symbolic_jacobian_identity_for_preview(
        set_id="set1",
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 12},
            "temperature_K": 298.15,
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    mechanism_owner.slider_overrides.return_value = {"k1": 0.8}
    updated_payload = owner._symbolic_jacobian_identity_for_preview(
        set_id="set1",
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 12},
            "temperature_K": 298.15,
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )

    assert payload["backend_name"] == "sympy"
    assert payload["parameter_symbols"] == ["k1"]
    assert payload["structure_fingerprint"]
    assert payload["evaluation_snapshot_fingerprint"]
    assert payload["structure_fingerprint"] == updated_payload["structure_fingerprint"]
    assert payload["evaluation_snapshot_fingerprint"] != updated_payload["evaluation_snapshot_fingerprint"]


@pytest.mark.unit
def test_default_batch_lane_pool_factory_creates_warm_lane_pool():
    from kindred.core.batch_containment import BatchLanePool

    pool = default_batch_lane_pool_factory(3, True)
    try:
        assert isinstance(pool, BatchLanePool)
        assert pool.retained_lane_count == 0
    finally:
        pool.close(kill=True)


@pytest.mark.unit
def test_batch_run_context_storage_is_owned_outside_controller_dict(controller: SimulationController):
    assert "batch_run_context" not in controller.__dict__
    assert not hasattr(type(controller), "_batch_run_context")
    assert not hasattr(type(controller), "batch_run_context")
    assert controller.batch_context_owner.active_batch_state() is None

    seed_batch_context(controller.batch_context_owner, active=True, request_id=12)

    assert "batch_run_context" not in controller.__dict__
    active_state = controller.batch_context_owner.active_batch_state()
    assert active_state is not None
    assert active_state.active is True
    assert active_state.request_id == 12

    seed_batch_context(controller.batch_context_owner, active=False, request_id=13)

    assert "batch_run_context" not in controller.__dict__
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is False
    assert policy_context.request_id == 13


@pytest.mark.unit
def test_parallel_batch_runtime_readiness_state_is_owned_outside_controller(controller: SimulationController):
    from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
        ParallelBatchRuntimeReadinessOwner,
    )

    assert "pool_eagerly_created" not in controller.__dict__
    assert "_pool_eagerly_created" not in controller.__dict__
    assert "_pool_eager_creation_thread" not in controller.__dict__
    assert "_pool_eager_creation_lock" not in controller.__dict__
    assert not hasattr(type(controller), "_pool_eagerly_created")
    assert not hasattr(type(controller), "_pool_eager_creation_thread")
    assert isinstance(
        controller.parallel_batch_runtime_readiness_owner,
        ParallelBatchRuntimeReadinessOwner,
    )


@pytest.mark.unit
def test_simulation_controller_tests_do_not_lock_batch_runtime_readiness_compat_fields():
    from pathlib import Path

    lines = [
        line
        for line in Path("tests/test_simulation_controller.py").read_text(encoding="utf-8").splitlines()
        if "not in source" not in line
    ]
    source = "\n".join(lines)

    assert "._pool_eagerly_created" not in source
    assert "._pool_eager_creation_thread" not in source


@pytest.mark.unit
def test_set_simulation_cache_caps_clamps_and_persists(mw: _FakeMainWindow, controller: SimulationController):
    result = controller.set_simulation_cache_caps(result_cap=-5, preview_cap="7", persist=True)
    assert result.ok is True
    assert result.operation == "set_caps"
    assert controller.batch_cache.result_cache_max_entries() == 0
    assert controller.batch_cache.preview_cache_max_entries() == 7
    mw._settings.setValue.assert_any_call("simulation/result_cache_cap", 0)
    mw._settings.setValue.assert_any_call("simulation/preview_cache_cap", 7)

@pytest.mark.unit
def test_new_contained_owner_uses_blas_thread_limit_setting(controller: SimulationController):
    from kindred.core.runtime_defaults import contained_child_blas_thread_env

    controller.parallel_batch.limit_blas_threads_per_worker = True
    limited_owner = controller._new_contained_simulation_owner(
        fast_mode=False,
        simulation_plan_payload={},
    )
    try:
        assert (
            limited_owner._runtime_owner._kernel_owner._handler_spec.env
            == contained_child_blas_thread_env(enabled=True)
        )
    finally:
        limited_owner.close(kill=True)

    controller.parallel_batch.limit_blas_threads_per_worker = False
    unlimited_owner = controller._new_contained_simulation_owner(
        fast_mode=False,
        simulation_plan_payload={},
    )
    try:
        assert unlimited_owner._runtime_owner._kernel_owner._handler_spec.env == {}
    finally:
        unlimited_owner.close(kill=True)


@pytest.mark.unit
def test_contained_owner_identity_tracks_blas_thread_limit_setting(controller: SimulationController):
    solver_config = {"solver": "BDF", "grid": {"N": 100}, "use_sparse_jacobian": True}

    controller.parallel_batch.limit_blas_threads_per_worker = True
    limited_ordinary = controller._ordinary_contained_owner_identity(
        owner_mechanism_text="A -> B ; k=1",
        solver_config=solver_config,
        t_end=10.0,
        set_id="set-1",
    )
    limited_preview = controller._preview_contained_owner_identity(
        owner_mechanism_text="A -> B ; k=1",
        solver_config=solver_config,
        t_end=10.0,
        set_id="set-1",
        parameter_names=[],
    )

    controller.parallel_batch.limit_blas_threads_per_worker = False
    unlimited_ordinary = controller._ordinary_contained_owner_identity(
        owner_mechanism_text="A -> B ; k=1",
        solver_config=solver_config,
        t_end=10.0,
        set_id="set-1",
    )
    unlimited_preview = controller._preview_contained_owner_identity(
        owner_mechanism_text="A -> B ; k=1",
        solver_config=solver_config,
        t_end=10.0,
        set_id="set-1",
        parameter_names=[],
    )

    assert limited_ordinary["contained_child_blas_threads_limited"] is True
    assert limited_preview["contained_child_blas_threads_limited"] is True
    assert unlimited_ordinary["contained_child_blas_threads_limited"] is False
    assert unlimited_preview["contained_child_blas_threads_limited"] is False
    assert limited_ordinary != unlimited_ordinary
    assert limited_preview != unlimited_preview


@pytest.mark.unit
def test_simulation_cache_stats_surfaces_failures(controller: SimulationController):
    controller.batch_cache.stats_best_effort = MagicMock(side_effect=RuntimeError("boom"))
    result = controller.simulation_cache_stats()
    assert result.ok is False
    assert result.operation == "stats"
    assert result.stats is None
    assert "Failed to read simulation cache status" in result.message

@pytest.mark.unit
def test_purge_simulation_result_cache_surfaces_failures(controller: SimulationController):
    controller.batch_cache.purge_result_cache = MagicMock(side_effect=RuntimeError("purge boom"))

    result = controller.purge_simulation_result_cache()

    assert result.ok is False
    assert result.operation == "purge_result_cache"
    assert "Failed to clear simulation result cache" in result.message

@pytest.mark.unit
def test_cleanup_worker_safely_does_not_force_terminate(controller: SimulationController):
    worker = _FakeWorker(running=True, wait_returns=False, signal_disconnect_typeerror=True)
    controller._cleanup_worker_safely(worker, "test worker")
    assert worker._cancelled is True
    assert worker._terminated is False

@pytest.mark.unit
def test_cleanup_worker_safely_defers_qthread_deletion_until_finished(controller: SimulationController, monkeypatch):
    send_called = {"n": 0}

    def _boom_send_posted_events(*_args, **_kwargs) -> None:
        send_called["n"] += 1
        raise AssertionError("sendPostedEvents must not run while worker thread is still running")

    monkeypatch.setattr(QtCore.QCoreApplication, "sendPostedEvents", _boom_send_posted_events)

    worker = make_stubborn_worker(_FakeWorker)
    controller._cleanup_worker_safely(worker, "test worker")
    assert worker._cancelled is True
    assert worker._delete_later_called is False
    assert worker.deleteLater in worker.finished._handlers
    assert send_called["n"] == 0

@pytest.mark.unit
def test_release_current_simulation_worker_skips_unregistered_qt_signal_disconnect_warning(
    controller: SimulationController,
):
    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._release_current_simulation_worker()

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]
    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

@pytest.mark.unit
def test_cleanup_worker_safely_disconnects_registered_qt_signal_handlers_without_warning(
    controller: SimulationController,
    monkeypatch,
):
    worker = _QtSignalWorker(running=False)
    progress = MagicMock()
    complete = MagicMock()
    error = MagicMock()
    controller.on_simulation_progress = progress
    controller._on_simulation_complete = complete
    controller._on_simulation_error = error
    monkeypatch.setattr(controller, "_delete_worker_if_stopped", MagicMock())

    identity = _connect_worker_application_signals(
        controller,
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    worker.progress.emit(10, "running")
    worker.result_ready.emit({"payload": True})
    worker.error.emit({"kind": "failure"})

    assert progress.call_count == 1
    assert complete.call_count == 1
    assert error.call_count == 1
    assert complete.call_args.kwargs["callback_identity"] is identity
    assert error.call_args.kwargs["callback_identity"] is identity

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._cleanup_worker_safely(worker, "simulation worker")

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]
    assert getattr(worker, "_kindred_controller_worker_signal_handlers", ()) == ()
    controller._delete_worker_if_stopped.assert_called_once_with(worker, "simulation worker")

    worker.progress.emit(20, "after")
    worker.result_ready.emit({"payload": False})
    worker.error.emit({"kind": "ignored"})

    assert progress.call_count == 1
    assert complete.call_count == 1
    assert error.call_count == 1

@pytest.mark.unit
def test_disconnect_simulation_worker_application_signals_preserves_failed_runtime_disconnects(
    controller: SimulationController,
):
    class _Signal:
        def __init__(self, *, raise_runtimeerror: bool = False) -> None:
            self.raise_runtimeerror = bool(raise_runtimeerror)
            self.handlers: list[Callable[..., Any]] = []

        def connect(self, handler: Callable[..., Any]) -> None:
            self.handlers.append(handler)

        def disconnect(self, handler: Callable[..., Any]) -> None:
            if self.raise_runtimeerror:
                raise RuntimeError("disconnect failed")
            self.handlers.remove(handler)

        def emit(self, *args: Any) -> None:
            for handler in tuple(self.handlers):
                handler(*args)

    class _Worker:
        def __init__(self) -> None:
            self.progress = _Signal(raise_runtimeerror=True)
            self.result_ready = _Signal()
            self.error = _Signal()

    worker = _Worker()
    progress = MagicMock()
    complete = MagicMock()
    error = MagicMock()
    controller.on_simulation_progress = progress
    controller.on_simulation_complete = complete
    controller.on_simulation_error = error
    controller._record_nonfatal_exception = MagicMock()

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        controller._disconnect_simulation_worker_application_signals(worker)

    remaining = getattr(worker, "_kindred_controller_worker_signal_handlers", ())
    assert len(remaining) == 1
    assert remaining[0][0] == "progress"
    controller._record_nonfatal_exception.assert_called_once()

    warning_messages = [str(item.message) for item in recorded]
    assert not [message for message in warning_messages if "Failed to disconnect" in message]

    worker.progress.emit(20, "done")
    worker.result_ready.emit({"payload": False})
    worker.error.emit({"kind": "ignored"})

    assert progress.call_count == 1
    assert complete.call_count == 0
    assert error.call_count == 0

@pytest.mark.unit
def test_connect_simulation_worker_application_signals_preserves_tracked_disconnect_failures_on_reconnect(
    controller: SimulationController,
):
    class _Signal:
        def __init__(self, *, raise_runtimeerror: bool = False) -> None:
            self.raise_runtimeerror = bool(raise_runtimeerror)
            self.handlers: list[Callable[..., Any]] = []

        def connect(self, handler: Callable[..., Any]) -> None:
            self.handlers.append(handler)

        def disconnect(self, handler: Callable[..., Any]) -> None:
            if self.raise_runtimeerror:
                raise RuntimeError("disconnect failed")
            self.handlers.remove(handler)

    class _Worker:
        def __init__(self) -> None:
            self.progress = _Signal(raise_runtimeerror=True)
            self.result_ready = _Signal()
            self.error = _Signal()

    worker = _Worker()
    controller._record_nonfatal_exception = MagicMock()

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=7,
        fast_mode=False,
        request_id=11,
        set_name="set1",
        set_id="id1",
        cache_key="ck",
    )

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=8,
        fast_mode=False,
        request_id=12,
        set_name="set2",
        set_id="id2",
        cache_key="ck2",
    )

    connections = getattr(worker, "_kindred_controller_worker_signal_handlers", ())
    names = [signal_name for signal_name, _handler in connections]
    assert names.count("progress") == 2
    assert names.count("result_ready") == 1
    assert names.count("error") == 1
    controller._record_nonfatal_exception.assert_called_once()

@pytest.mark.unit
def test_prepare_simulation_shutdown_for_close_keeps_window_recoverable_when_worker_errors_after_deferred_close(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(42)

    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    worker.error.connect(
        lambda msg: controller.on_simulation_error(msg, callback_identity=identity)
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._status_label.text == "Simulation cancelled by user"
    assert mw._sim_progress.value == 0

@pytest.mark.unit
def test_prepare_simulation_shutdown_for_close_ignores_deleted_retained_worker(controller: SimulationController):
    worker = QtCore.QThread(parent=controller)
    controller._retained_simulation_workers.append(worker)
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)

    close_ready = controller.prepare_simulation_shutdown_for_close()

    assert close_ready is True
    assert controller._shutdown_requested_for_close is False
    assert controller._retained_simulation_workers == []

@pytest.mark.unit
def test_release_current_simulation_worker_ignores_deleted_worker(controller: SimulationController):
    worker = QtCore.QThread(parent=controller)
    controller._simulation_worker = worker
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)

    controller.release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == []

@pytest.mark.unit
def test_run_simulation_from_slider_ignores_deleted_current_worker(
    mw: _FakeMainWindow, controller: SimulationController
):
    calls: list[dict[str, Any]] = []

    def _record_run(**kwargs) -> None:
        calls.append(dict(kwargs))

    controller.run_simulation_internal = _record_run
    worker = QtCore.QThread(parent=controller)
    controller._simulation_worker = worker
    worker.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(worker, QtCore.QEvent.DeferredDelete)
    controller._latest_sim_request_id = 1
    controller._pending_slider_sim_request_id = 1
    controller._pending_slider_target_set_ids = ("id1",)
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"

    controller._run_simulation_from_slider()

    assert controller._simulation_worker is None
    assert calls and calls[0]["fast_mode"] is True

@pytest.mark.unit
def test_deferred_close_successful_completion_does_not_schedule_next_serial_batch_run(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller._simulation_running = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_ids=["id1", "id2"], queue_names=["set1", "set2"], total=2)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    identity = _capture_callback_identity(
        controller,
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    worker.result_ready.connect(
        lambda payload: controller.on_simulation_complete(payload, callback_identity=identity)
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.result_ready.emit(_successful_result_payload())

    assert scheduled == []
    assert _batch_policy_context(controller).active is False
    assert controller._simulation_running is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers

@pytest.mark.unit
def test_deferred_close_successful_completion_does_not_schedule_pending_slider_rerun_and_still_recovers_ui(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 9
    controller._latest_sim_request_id = 15
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._pending_slider_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(42)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    identity = _capture_callback_identity(
        controller,
        run_id=9,
        fast_mode=False,
        request_id=15,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    worker.result_ready.connect(
        lambda payload: controller.on_simulation_complete(payload, callback_identity=identity)
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.result_ready.emit(_successful_result_payload())

    assert scheduled == []
    assert controller._pending_slider_simulation is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert worker not in controller._retained_simulation_workers
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 100
    assert mw._status_label.text == "Simulation complete: 2 species, 3 points"

@pytest.mark.unit
def test_deferred_close_error_recovery_restores_later_serial_batch_continuation(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True

    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="old-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    worker.error.connect(
        lambda msg: controller.on_simulation_error(msg, callback_identity=identity)
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert controller._shutdown_requested_for_close is True

    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._shutdown_requested_for_close is True
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.finished.emit()

    assert controller._shutdown_requested_for_close is False
    assert worker not in controller._retained_simulation_workers

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._active_run_id = 13
    controller._latest_sim_request_id = 12
    controller._simulation_running = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_ids=["id1", "id2"], queue_names=["set1", "set2"], total=2)

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=13,
        fast_mode=False,
        request_id=12,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="new-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._start_next_batch_simulation]
    assert _batch_policy_context(controller).active is True
    assert _batch_policy_context(controller).pos == 1

@pytest.mark.unit
def test_deferred_close_error_recovery_restores_later_pending_slider_rerun(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._active_run_id = 7
    controller._latest_sim_request_id = 9
    controller._simulation_running = True

    identity = _capture_callback_identity(
        controller,
        run_id=7,
        fast_mode=False,
        request_id=9,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="old-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    worker.error.connect(
        lambda msg: controller.on_simulation_error(msg, callback_identity=identity)
    )

    close_ready = controller.prepare_simulation_shutdown_for_close()
    assert close_ready is False
    assert controller._shutdown_requested_for_close is True

    worker.error.emit({"kind": "cancelled", "message": "Simulation cancelled by user"})

    assert controller._shutdown_requested_for_close is True
    assert worker in controller._retained_simulation_workers

    worker._running = False
    worker.finished.emit()

    assert controller._shutdown_requested_for_close is False
    assert worker not in controller._retained_simulation_workers

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._active_run_id = 15
    controller._latest_sim_request_id = 16
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._pending_slider_simulation = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_ids=["id1"], queue_names=["set1"], total=1)

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=15,
        fast_mode=False,
        request_id=16,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="new-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True

@pytest.mark.unit
def test_simulation_worker_does_not_shadow_qthread_finished_signal():
    assert "finished" not in SimulationWorker.__dict__

@pytest.mark.unit
def test_shutdown_batch_lane_pool_kills_warm_lane_pool(
    controller: SimulationController,
):
    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False):
            self.close_calls.append(bool(kill))

    pool = _FakeLanePool()
    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller._batch_parallel.ensure_lane_pool(max_lanes=1)
    controller._shutdown_batch_lane_pool(force_terminate=True)
    assert pool.close_calls == [True]

@pytest.mark.unit
def test_slider_request_during_parallel_full_run_defers_without_force_terminate(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._run_simulation_internal = MagicMock()
    mw._run_btn = _FakeButton(False)

    worker = _FakeWorker(running=True, wait_returns=False, signal_disconnect_typeerror=True)
    worker._fast_mode = False  # type: ignore[attr-defined]

    def _boom_thread_terminate() -> None:
        raise AssertionError("QThread.terminate must not be called from slider deferral path")

    worker.terminate = _boom_thread_terminate  # type: ignore[assignment]
    controller._simulation_worker = worker

    seed_batch_context(controller.batch_context_owner, active=True, parallel=True)
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is True
    controller._run_simulation_internal.assert_not_called()

@pytest.mark.unit
def test_slider_request_while_fast_worker_running_is_latest_only_and_does_not_cancel(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._run_simulation_internal = MagicMock()
    mw._run_btn = _FakeButton(True)

    worker = _FakeWorker(running=True, wait_returns=True)
    worker._fast_mode = True  # type: ignore[attr-defined]
    worker.cancel = MagicMock(side_effect=AssertionError("Fast worker must not be cancelled for latest-only scheduling"))
    controller._simulation_worker = worker

    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is True
    controller._run_simulation_internal.assert_not_called()

@pytest.mark.unit
def test_stale_fast_completion_schedules_pending_slider_run(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True

    controller._active_run_id = 5
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=1)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=5,
        fast_mode=True,
        request_id=int(rid_old),
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_second_stale_fast_completion_does_not_queue_duplicate_handoff_before_first_callback_runs(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 8
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=8,
        epoch=2,
        target_set_ids=("id1",),
    )
    controller.run_state.pending_slider_preview_launch = controller.run_state.pending_slider_preview_launch.__class__(
        active=True,
        request_id=7,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=False, parallel=True, fast_mode=True, request_id=7, preview_owner_epoch=1, keep_lane_pool_alive=True)

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert getattr(controller.run_state.pending_slider_preview_launch, "handoff_queued") is True


@pytest.mark.unit
def test_stale_fast_completion_treats_same_request_as_superseded_after_owner_epoch_changes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("captured-id",),
    )
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_worker_signal_completion_preserves_owner_epoch_for_stale_fast_replay(
    monkeypatch, controller: SimulationController
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="preview-cache",
    )

    worker.result_ready.emit(_successful_result_payload())

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_worker_signal_completion_uses_captured_owner_epoch_after_context_turnover(
    monkeypatch, controller: SimulationController
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="preview-cache",
    )
    policy_context = _batch_policy_context(controller).evolve(preview_owner_epoch=4)
    controller.batch_context_owner.serialize_completion_policy_context(policy_context)

    worker.result_ready.emit(_successful_result_payload())

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_completion_missing_callback_context_does_not_use_current_preview_owner_epoch(
    monkeypatch, controller: SimulationController
):
    controller._latest_sim_request_id = 8
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        run_id=5,
        preview_owner_epoch=4,
        cache_key="preview-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
    )
    controller._completion_callback_owner._deps.freshness.assess_callback = MagicMock(
        side_effect=AssertionError("freshness must not be consulted without callback context")
    )
    controller._completion_publication_owner.publish_success = MagicMock()
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_complete(
            _successful_result_payload(),
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=True,
                request_id=7,
                owner_epoch=None,
                batch_set="set2",
                batch_set_id="id2",
                cache_key="preview-cache",
            ),
        ),

    controller._completion_callback_owner._deps.freshness.assess_callback.assert_not_called()
    controller._completion_publication_owner.publish_success.assert_not_called()


@pytest.mark.unit
def test_completion_missing_callback_context_does_not_use_current_queue_or_stale_context(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=0,
        target_set_ids=("id2",),
    )
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )
    controller._completion_callback_owner._deps.freshness.assess_callback = MagicMock(
        side_effect=AssertionError("freshness must not be consulted without callback context")
    )
    controller._completion_publication_owner.publish_success = MagicMock()
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_complete(
            _successful_result_payload(),
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="current-cache",
            ),
        ),

    controller._completion_callback_owner._deps.freshness.assess_callback.assert_not_called()
    controller._completion_publication_owner.publish_success.assert_not_called()


@pytest.mark.unit
def test_completion_missing_callback_context_does_not_use_current_publication_context(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-primary", "callback-id"],
        queue_names=["current-primary-set", "callback-set"],
        primary_set_id="current-primary",
        total=2,
        pos=0,
    )
    controller._queue_slider_plot_update = MagicMock()
    controller.ui.batch._batch_set_ids_for_scope = MagicMock(return_value=["current-primary"])
    controller.ui.batch._batch_current_row = MagicMock(return_value=0)
    controller.ui.batch._batch_set_id_for_row = MagicMock(return_value="current-primary")
    controller.ui.results.publish_simulation_completion_result = MagicMock()
    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller._result_materialization_owner.refresh_primary_result_controls = MagicMock()
    controller.ui.provenance.publish_simulation_completion_provenance = MagicMock()
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_complete(
            _successful_result_payload(),
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set="callback-set",
                batch_set_id="callback-id",
                cache_key="current-cache",
            ),
        ),

    controller._queue_slider_plot_update.assert_not_called()
    controller.ui.batch._batch_set_ids_for_scope.assert_not_called()
    controller.ui.batch._batch_current_row.assert_not_called()
    controller.ui.batch._batch_set_id_for_row.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()
    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller._result_materialization_owner.refresh_primary_result_controls.assert_not_called()
    controller.ui.provenance.publish_simulation_completion_provenance.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert policy_context.completed_set_ids == ()
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_completion_missing_callback_context_does_not_deactivate_mismatched_current_run(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-primary", "callback-id"],
        queue_names=["current-primary-set", "callback-set"],
        primary_set_id="current-primary",
        total=2,
        pos=0,
    )
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_complete(
            _successful_result_payload(),
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set="callback-set",
                batch_set_id="callback-id",
                cache_key="captured-cache",
            ),
        ),

    controller.ui.dialogs.message_box_critical.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_completion_malformed_callback_context_does_not_deactivate_current_run(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-primary", "callback-id"],
        queue_names=["current-primary-set", "callback-set"],
        primary_set_id="current-primary",
        total=2,
        pos=0,
    )
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_complete(
            _successful_result_payload(),
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="current-cache",
            ),
        ),

    controller.ui.dialogs.message_box_critical.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_scalar_completion_without_callback_identity_is_rejected_without_current_context_capture(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-primary", "callback-id"],
        queue_names=["current-primary-set", "callback-set"],
        primary_set_id="current-primary",
        total=2,
        pos=0,
    )
    controller.ui.batch._batch_set_ids_for_scope = MagicMock(return_value=["current-primary"])
    controller.ui.batch._batch_current_row = MagicMock(return_value=0)
    controller.ui.batch._batch_set_id_for_row = MagicMock(return_value="current-primary")
    controller.ui.results.publish_simulation_completion_result = MagicMock()
    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller._result_materialization_owner.refresh_primary_result_controls = MagicMock()
    controller.ui.provenance.publish_simulation_completion_provenance = MagicMock()
    controller.ui.dialogs.message_box_critical = MagicMock()

    with pytest.raises(TypeError, match="callback_identity"):
        controller._on_simulation_complete(
            _successful_result_payload(),
        )

    controller.ui.batch._batch_set_ids_for_scope.assert_not_called()
    controller.ui.batch._batch_current_row.assert_not_called()
    controller.ui.batch._batch_set_id_for_row.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()
    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller._result_materialization_owner.refresh_primary_result_controls.assert_not_called()
    controller.ui.provenance.publish_simulation_completion_provenance.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert policy_context.completed_set_ids == ()
    assert controller._simulation_running is True
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_worker_signal_completion_uses_captured_cache_key_after_context_turnover(
    controller: SimulationController,
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("id2",),
    )
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("id2",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=3,
        cache_key="initial-current-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
    )
    controller.batch_cache.active_preview_cache_key = "active-cache-after-turnover"
    controller._queue_slider_plot_update = MagicMock()

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="captured-preview-cache",
        simulation_identity={"schema_id": "captured-schema"},
        preview_batch_cache_token="captured-token",
    )
    controller.batch_context_owner.record_cache_key("current-context-cache-after-turnover")

    worker.result_ready.emit(_successful_result_payload())

    controller._queue_slider_plot_update.assert_called_once()
    assert controller._queue_slider_plot_update.call_args.kwargs["cache_key"] == "captured-preview-cache"


@pytest.mark.unit
def test_worker_signal_completion_uses_captured_context_identity_after_context_turnover(
    controller: SimulationController,
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("id2",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=3,
        cache_key="captured-preview-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
        pos=0,
        total=1,
        primary_set_id="id2",
        simulation_identity_by_set_id={"id2": {"schema_id": "captured-schema"}},
        preview_batch_cache_token_by_set_id={"id2": "captured-token"},
    )
    published: list[dict[str, object]] = []
    controller._cache_admin.publish_completion_cache = lambda **kwargs: published.append(dict(kwargs))

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="captured-preview-cache",
        simulation_identity={"schema_id": "captured-schema"},
        preview_batch_cache_token="captured-token",
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=3,
        cache_key="current-preview-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
        pos=0,
        total=1,
        primary_set_id="id2",
        simulation_identity_by_set_id={"id2": {"schema_id": "current-schema"}},
        preview_batch_cache_token_by_set_id={"id2": "current-token"},
    )

    worker.result_ready.emit(_successful_result_payload())

    assert published
    assert published[0]["cache_key"] == "captured-preview-cache"
    assert published[0]["simulation_identity"] == {"schema_id": "captured-schema"}
    assert published[0]["preview_batch_cache_token"] == "captured-token"


@pytest.mark.unit
def test_worker_signal_completion_preserves_warning_payloads_in_completion_cache(
    controller: SimulationController,
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=0,
        target_set_ids=("id2",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=0,
        cache_key="captured-preview-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
        pos=0,
        total=1,
        primary_set_id="id2",
        simulation_identity_by_set_id={"id2": {"schema_id": "captured-schema"}},
        preview_batch_cache_token_by_set_id={"id2": "captured-token"},
    )
    published: list[dict[str, object]] = []
    controller._cache_admin.publish_completion_cache = lambda **kwargs: published.append(dict(kwargs))

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=0,
        set_name="set2",
        set_id="id2",
        cache_key="captured-preview-cache",
    )
    payload = _successful_result_payload()
    payload["warnings"] = [{"kind": "preparation_warning", "message": "symbolic disabled"}]

    worker.result_ready.emit(payload)

    assert published
    assert published[0]["warnings"] == [{"kind": "preparation_warning", "message": "symbolic disabled"}]


@pytest.mark.unit
def test_completion_missing_set_identity_does_not_fallback_to_captured_or_current_context(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("captured-id",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=3,
        cache_key="captured-preview-cache",
        queue_ids=["captured-id"],
        queue_names=["captured-set"],
        pos=0,
        total=1,
        primary_set_id="captured-id",
        simulation_identity_by_set_id={"captured-id": {"schema_id": "captured-schema"}},
        preview_batch_cache_token_by_set_id={"captured-id": "captured-token"},
    )
    identity = controller._capture_simulation_callback_identity(
        run_id=5,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        batch_set=None,
        batch_set_id=None,
        cache_key="captured-preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity={"schema_id": "captured-schema"},
        preview_batch_cache_token="captured-token",
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        preview_owner_epoch=3,
        cache_key="current-preview-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
        total=1,
        primary_set_id="current-id",
        simulation_identity_by_set_id={"current-id": {"schema_id": "current-schema"}},
        preview_batch_cache_token_by_set_id={"current-id": "current-token"},
    )
    published: list[dict[str, object]] = []
    controller._cache_admin.publish_completion_cache = lambda **kwargs: published.append(dict(kwargs))

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=identity,
    )

    assert published == []


@pytest.mark.unit
def test_callback_identity_plan_identity_wins_after_same_run_context_turnover(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    solver_config = {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15}
    captured_identity = SimulationIdentity.build(
        schema_id="captured-schema",
        param_fingerprint="captured-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()
    current_identity = SimulationIdentity.build(
        schema_id="current-schema",
        param_fingerprint="current-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()

    def _plan_payload(identity: dict[str, object]) -> dict[str, object]:
        return SimulationPlan.from_execution_request(
            {
                "prepared_payload": None,
                "initials": {"A": 1.0},
                "t_span": (0.0, 1.0),
                "solver_config": solver_config,
                "mechanism_text": "reaction: A -> B; k=1",
                "simulation_identity": identity,
            },
            execution_mode="explicit",
            algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
            cache_identity_payload={"cache_key": "same-cache", "simulation_identity": identity},
            cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": ["id1"]},
        ).to_payload()

    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="same-cache",
        queue_ids=["id1"],
        queue_names=["set1"],
        pos=0,
        total=1,
        primary_set_id="id1",
        simulation_plan_by_set_id={"id1": _plan_payload(captured_identity)},
        simulation_identity_by_set_id={"id1": {"schema_id": "stale-captured-context"}},
    )
    identity = controller._capture_simulation_callback_identity(
        run_id=5,
        fast_mode=False,
        request_id=7,
        owner_epoch=None,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="same-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity=captured_identity,
        preview_batch_cache_token="",
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="same-cache",
        queue_ids=["id1"],
        queue_names=["set1"],
        pos=0,
        total=1,
        primary_set_id="id1",
        simulation_plan_by_set_id={"id1": _plan_payload(current_identity)},
        simulation_identity_by_set_id={"id1": {"schema_id": "current-context"}},
    )
    published: list[dict[str, object]] = []
    controller._cache_admin.publish_completion_cache = lambda **kwargs: published.append(dict(kwargs))

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=identity,
    )

    assert published
    assert published[0]["simulation_identity"] == captured_identity


@pytest.mark.unit
def test_superseded_multiset_preview_completion_still_displays_current_result_before_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True
    controller._active_run_id = 7
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid_old),
        epoch=1,
        target_set_ids=("id1", "id2"),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=int(rid_old), fast_mode=True, cache_key="preview-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, completed_set_ids=["id1"], preview_scope_set_ids=("id1", "id2"), preview_batch_cache_token_by_set_id={"id1": "", "id2": ""})
    mw._batch_set_ids_for_scope.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.side_effect = lambda row: ("id1", "id2")[int(row)]
    mw._display_cached_batch_selection.return_value = True

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=True,
        request_id=int(rid_old),
        batch_set="set2",
        batch_set_id="id2",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert mw._display_cached_batch_selection.call_count == 1
    assert scheduled == [controller._run_simulation_from_slider]

@pytest.mark.unit
def test_superseded_multiset_preview_partial_completion_keeps_parallel_batch_active_until_full_batch_finishes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True
    controller._active_run_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid_old),
        epoch=1,
        target_set_ids=("id1", "id2"),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=8, request_id=int(rid_old), fast_mode=True, cache_key="preview-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, completed_set_ids=[], preview_scope_set_ids=("id1", "id2"), preview_batch_cache_token_by_set_id={"id1": "", "id2": ""})
    mw._batch_set_ids_for_scope.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.side_effect = lambda row: ("id1", "id2")[int(row)]
    mw._display_cached_batch_selection.return_value = True

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=8,
        fast_mode=True,
        request_id=int(rid_old),
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is True
    completion_state = controller.batch_context_owner.completion_state()
    assert completion_state is not None
    assert completion_state.completed_set_ids == ("id1",)
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert scheduled == []


@pytest.mark.unit
def test_fast_completion_displays_current_owner_even_when_latest_request_id_has_advanced(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 6
    controller._active_run_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=8,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, keep_lane_pool_alive=False, run_id=8, request_id=5, fast_mode=True, cache_key="preview-cache", queue_ids=["id1"], queue_names=["set1"], total=1, preview_scope_set_ids=("id1",), preview_owner_epoch=8, preview_batch_cache_token_by_set_id={"id1": ""})
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = True

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=8,
        fast_mode=True,
        request_id=5,
        owner_epoch=8,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw._display_cached_batch_selection.assert_called_once()

@pytest.mark.unit
def test_stale_fast_completion_without_pending_still_cleans_up_active_run(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 2
    controller._pending_slider_sim_request_id = 2
    controller._pending_slider_simulation = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=2,
        epoch=5,
        target_set_ids=("id1",),
    )

    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=1)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(0)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert scheduled == [controller._run_simulation_from_slider]

@pytest.mark.unit
def test_on_simulation_complete_uses_base_species_count_for_algebra_status_without_mechanism(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=11,
        epoch=1,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, keep_lane_pool_alive=False, run_id=7, request_id=11, fast_mode=True, cache_key="preview-cache", queue_ids=["id1"], queue_names=["set1"], total=1, preview_scope_set_ids=("id1",), preview_batch_cache_token_by_set_id={"id1": ""})
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    result = _successful_result_payload()
    result["species_names"] = ["A", "Alg"]
    result["algebra_errors"] = [{"kind": "algebra_error", "message": "bad algebra"}]
    result["base_species_count"] = 1

    _complete_with_callback_identity(
        controller,
        result,
        run_id=7,
        fast_mode=True,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert mw._algebra_status_label.text == "Algebra: 1 ok, 1 error - bad algebra"

@pytest.mark.unit
def test_on_simulation_complete_prefers_payload_base_species_count_over_mechanism_for_algebra_status(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=11,
        epoch=1,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, keep_lane_pool_alive=False, run_id=7, request_id=11, fast_mode=True, cache_key="preview-cache", queue_ids=["id1"], queue_names=["set1"], total=1, preview_scope_set_ids=("id1",), preview_batch_cache_token_by_set_id={"id1": ""})
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    mechanism = MagicMock()
    mechanism.species_names.return_value = ["A", "Alg"]

    result = _successful_result_payload()
    result["species_names"] = ["A", "Alg"]
    result["mechanism"] = mechanism
    result["algebra_errors"] = [{"kind": "algebra_error", "message": "bad algebra"}]
    result["base_species_count"] = 1

    _complete_with_callback_identity(
        controller,
        result,
        run_id=7,
        fast_mode=True,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert mw._algebra_status_label.text == "Algebra: 1 ok, 1 error - bad algebra"

@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_ui_active_when_full_run_still_in_flight(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._simulation_running = True
    controller._slider_simulation_active = False
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=False)

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    controller.invalidate_slider_preview_work()

    assert controller._pending_slider_simulation is False
    assert _batch_policy_context(controller).active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57

@pytest.mark.unit
def test_invalidate_slider_preview_work_supersedes_active_fast_parallel_batch(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=True, request_id=int(rid))

    def _fake_supersede() -> None:
        controller.batch_context_owner.deactivate()

    controller._supersede_parallel_batch_run_soft = MagicMock(side_effect=_fake_supersede)

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)

    controller.invalidate_slider_preview_work()

    controller._supersede_parallel_batch_run_soft.assert_called_once_with()
    assert _batch_policy_context(controller).active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0

@pytest.mark.unit
def test_invalidate_slider_preview_work_suppresses_stale_completion_ui_after_discard(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=int(rid), run_id=11)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
        batch_set=None,
        batch_set_id=None,
        cache_key="preview-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    controller._pending_slider_plot_cache_key = "preview-ck"
    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(73)
    mw.message_box_critical = MagicMock()
    mw.message_box_warning = MagicMock()

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller.invalidate_slider_preview_work()

    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    assert scheduled == []
    assert controller._pending_slider_plot_cache_key is None
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_active_after_stale_completion(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=int(rid), run_id=11)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
        batch_set=None,
        batch_set_id=None,
        cache_key="preview-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=99)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 99  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller.invalidate_slider_preview_work()
    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    policy_context = _batch_policy_context(controller)
    assert policy_context.active is True
    assert policy_context.fast_mode is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57

@pytest.mark.unit
def test_nonowning_stale_fast_completion_does_not_reset_explicit_run_status_progress(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 3
    controller._pending_slider_sim_request_id = None
    controller._pending_slider_simulation = False
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=1, run_id=11)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=1,
        batch_set=None,
        batch_set_id=None,
        cache_key="preview-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=3)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 3  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    policy_context = _batch_policy_context(controller)
    assert policy_context.active is True
    assert policy_context.fast_mode is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 57


@pytest.mark.unit
def test_preview_request_can_display_rejects_stale_request_when_stopped_fast_worker_has_newer_request_id(
    controller: SimulationController,
):
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=9,
        epoch=4,
        target_set_ids=("id1",),
    )

    assert controller._preview_request_can_display(4) is False


@pytest.mark.unit
def test_preview_request_can_display_accepts_matching_stopped_fast_worker_while_preview_active(
    controller: SimulationController,
):
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=9,
        epoch=4,
        target_set_ids=("id1",),
    )

    assert controller._preview_request_can_display(9) is True


@pytest.mark.unit
def test_stale_fast_completion_with_pending_newer_preview_replays_without_displaying_stopped_old_worker(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid_new),
        epoch=3,
        target_set_ids=("id1",),
    )
    worker = _FakeWorker(running=False, wait_returns=True)
    worker._request_id = int(rid_old)  # type: ignore[attr-defined]
    worker._fast_mode = True  # type: ignore[attr-defined]
    controller._simulation_worker = worker

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)
    mw.set_data.reset_mock()

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=int(rid_old),
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert mw._status_label.text == "Updating simulation..."
    assert mw._sim_progress.value == 41
    mw.set_data.assert_not_called()


@pytest.mark.unit
def test_stale_fast_completion_preserves_current_owner_request_for_pending_replay(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid_old = controller._next_sim_request_id()
    rid_owner = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_owner)
    controller._pending_slider_sim_request_id = int(rid_owner)
    controller._pending_slider_simulation = True
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid_owner),
        epoch=5,
        target_set_ids=("id1",),
    )
    worker = _FakeWorker(running=False, wait_returns=True)
    worker._request_id = int(rid_old)  # type: ignore[attr-defined]
    worker._fast_mode = True  # type: ignore[attr-defined]
    controller._simulation_worker = worker

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    monkeypatch.setattr(
        controller,
        "_next_slider_preview_request_id",
        MagicMock(side_effect=AssertionError("fresh request allocation is not allowed here")),
    )

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=True,
        request_id=int(rid_old),
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert controller._pending_slider_sim_request_id == int(rid_owner)
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_invalidate_slider_preview_work_suppresses_stale_error_ui_after_discard(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=int(rid), run_id=11)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
        batch_set=None,
        batch_set_id=None,
        cache_key="preview-ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller.invalidate_slider_preview_work()

    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    controller._on_simulation_error("boom", callback_identity=callback_identity)

    assert scheduled == []
    assert mw._status_label.text == "Ready"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    message_box.assert_not_called()


@pytest.mark.unit
def test_stale_contained_timeout_error_from_old_run_does_not_clobber_newer_run(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 22
    controller._latest_sim_request_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = False
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=8)
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(44)
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "timeout",
            "Simulation timed out after 0.2 seconds.",
            details={"walltime_s": 0.2},
        ),
        run_id=21,
        fast_mode=False,
        request_id=7,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert controller._simulation_running is True
    assert _batch_policy_context(controller).active is True
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 44
    message_box.assert_not_called()


@pytest.mark.unit
def test_current_preview_timeout_marks_dirty_preview_unavailable_without_modal(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 31
    controller._latest_sim_request_id = 9
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=9,
        epoch=4,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, run_id=31, request_id=9, preview_owner_epoch=4, queue_ids=["id1"], queue_names=["Set 1"])
    mw._dirty_state_generations = {"id1": 7}
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(44)
    mw.message_box_critical = MagicMock()
    mw.show_preview_unavailable_for_dirty_state = MagicMock()

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "active_timeout",
            "Preview simulation timed out after 0.2 seconds.",
            details={"active_solve_timeout_s": 0.2},
        ),
        run_id=31,
        fast_mode=True,
        request_id=9,
        owner_epoch=4,
        batch_set="Set 1",
        batch_set_id="id1",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.message_box_critical.assert_not_called()
    mw.show_preview_unavailable_for_dirty_state.assert_called_once()
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert _batch_policy_context(controller).active is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 0
    assert mw._status_label.text.startswith("Preview unavailable")
    assert mw.has_dirty_state_for_set("id1") is True
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()


@pytest.mark.unit
def test_stale_preview_timeout_does_not_mark_current_dirty_preview_unavailable(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._active_run_id = 41
    controller._latest_sim_request_id = 12
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=12,
        epoch=6,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, run_id=41, request_id=12, preview_owner_epoch=6, queue_ids=["id1"], queue_names=["Set 1"])
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Updating current preview...")
    mw._sim_progress.setValue(55)
    mw.message_box_critical = MagicMock()
    mw.show_preview_unavailable_for_dirty_state = MagicMock()

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "timeout",
            "Old preview timed out.",
            details={"active_solve_timeout_s": 0.2},
        ),
        run_id=41,
        fast_mode=True,
        request_id=11,
        owner_epoch=5,
        batch_set="Set 1",
        batch_set_id="id1",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.message_box_critical.assert_not_called()
    mw.show_preview_unavailable_for_dirty_state.assert_not_called()
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert _batch_policy_context(controller).active is True
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Updating current preview..."
    assert mw._sim_progress.value == 55
    assert scheduled == []


@pytest.mark.unit
def test_current_preview_non_timeout_error_closes_preview_owner_not_ordinary_owner(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 51
    controller._latest_sim_request_id = 14
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=14,
        epoch=8,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, run_id=51, request_id=14, preview_owner_epoch=8, queue_ids=["id1"], queue_names=["Set 1"])
    ordinary_owner = _FakeContainedOwner()
    preview_owner = _FakeContainedOwner()
    controller._ordinary_simulation_owner = ordinary_owner
    controller._preview_simulation_owner = preview_owner
    mw.message_box_critical = MagicMock()

    _error_with_callback_identity(
        controller,
        build_simulation_failure("simulation_error", "Preview child failed."),
        run_id=51,
        fast_mode=True,
        request_id=14,
        owner_epoch=8,
        batch_set="Set 1",
        batch_set_id="id1",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert preview_owner.close_calls == [True]
    assert ordinary_owner.close_calls == []
    assert controller._preview_simulation_owner is None
    assert controller._ordinary_simulation_owner is ordinary_owner


@pytest.mark.unit
def test_current_contained_preview_child_failure_is_status_only_dirty_no_preview(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 52
    controller._latest_sim_request_id = 15
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=15,
        epoch=9,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, run_id=52, request_id=15, preview_owner_epoch=9, queue_ids=["id1"], queue_names=["Set 1"])
    preview_owner = _FakeContainedOwner()
    controller._preview_simulation_owner = preview_owner
    mw.message_box_critical = MagicMock()

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "simulation_error",
            "Preview child failed.",
            details={"source": "simulation_containment"},
        ),
        run_id=52,
        fast_mode=True,
        request_id=15,
        owner_epoch=9,
        batch_set="Set 1",
        batch_set_id="id1",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.message_box_critical.assert_not_called()
    assert preview_owner.close_calls == [True]
    assert mw._preview_unavailable_messages == ["Preview unavailable. Adjust sliders or run again."]
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False


@pytest.mark.unit
def test_current_preview_wegscheider_cyclicity_failure_is_status_only_dirty_no_preview(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 53
    controller._latest_sim_request_id = 16
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=16,
        epoch=10,
        target_set_ids=("id1",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        run_id=53,
        request_id=16,
        preview_owner_epoch=10,
        queue_ids=["id1"],
        queue_names=["Set 1"],
    )
    mw.message_box_critical = MagicMock()

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "preparation_error",
            "Unresolved Wegscheider cyclicity.",
            details={"stage": "wegscheider_cyclicity"},
        ),
        run_id=53,
        fast_mode=True,
        request_id=16,
        owner_epoch=10,
        batch_set="Set 1",
        batch_set_id="id1",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.message_box_critical.assert_not_called()
    assert mw._preview_unavailable_messages == ["Unresolved Wegscheider cyclicity."]
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False


@pytest.mark.unit
def test_invalidate_slider_preview_work_keeps_explicit_run_active_after_stale_error(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._active_run_id = 11
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        run_id=11,
        request_id=int(rid),
        preview_owner_epoch=2,
    )
    stale_preview_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=int(rid),
        owner_epoch=2,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=101)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 101  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller.invalidate_slider_preview_work()
    controller._on_simulation_error("boom", callback_identity=stale_preview_identity)

    policy_context = _batch_policy_context(controller)
    assert policy_context.active is True
    assert policy_context.fast_mode is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._slider_triggered_simulation is False
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 41
    message_box.assert_not_called()

@pytest.mark.unit
def test_invalidate_active_explicit_simulation_for_authoritative_change_cancels_run_and_ignores_old_completion(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._run_sequence_id = 11
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=101)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 101  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(57)
    mw.set_data.reset_mock()

    controller.invalidate_active_explicit_simulation_for_authoritative_change()

    assert controller._active_run_id == 12
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert explicit_worker._cancelled is True
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=False,
        request_id=101,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.set_data.assert_not_called()

@pytest.mark.unit
def test_supersede_active_work_for_authoritative_mechanism_transition_rejects_old_explicit_completion(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._run_sequence_id = 11
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=101)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 101  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker
    mw.set_data.reset_mock()

    controller.supersede_active_work_for_authoritative_mechanism_transition(epoch=4)

    assert controller._active_run_id == 12
    assert explicit_worker._cancelled is True
    assert controller._simulation_running is False
    assert mw._stop_btn.isEnabled() is False

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=11,
        fast_mode=False,
        request_id=101,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.set_data.assert_not_called()


@pytest.mark.unit
def test_runtime_input_supersede_preserves_preview_owner_when_runtime_identity_is_unchanged(
    controller: SimulationController,
):
    class _FakeOwner:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    preview_owner = _FakeOwner()
    controller._preview_simulation_owner = preview_owner
    controller._preview_ownership = PreviewOwnershipState(
        request_id=4,
        epoch=2,
        target_set_ids=("id2",),
    )
    controller._latest_sim_request_id = 4
    controller._simulation_running = True
    controller._slider_simulation_active = True

    controller.supersede_active_work_for_authoritative_mechanism_transition(
        epoch=8,
        affected_set_ids=("id1",),
        close_preview_runtime_owner=False,
    )

    assert controller._authoritative_runtime_input_epoch == 8
    assert controller._authoritative_runtime_input_set_epoch_by_set_id == {"id1": 8}
    assert controller._preview_simulation_owner is preview_owner
    assert preview_owner.close_calls == []
    assert controller.run_state.preview_ownership.request_id == 4


@pytest.mark.unit
def test_scoped_runtime_input_supersede_accepts_unaffected_preview_completion(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 4
    controller._active_run_id = 10
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=4,
        epoch=2,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=4, run_id=10, cache_key="preview-cache", queue_ids=["id2"], queue_names=["set2"], total=1, pos=0, preview_scope_set_ids=("id2",), runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id2": 0})
    mw._batch_set_ids_for_scope.return_value = ["id2"]
    mw._shown_batch_set_ids.return_value = ["id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id2"
    mw._batch_set_name_for_id.return_value = "set2"
    mw._display_cached_batch_selection.return_value = True

    controller.supersede_active_work_for_authoritative_mechanism_transition(
        epoch=8,
        affected_set_ids=("id1",),
        close_preview_runtime_owner=False,
    )

    assert controller.run_state.preview_ownership.request_id == 4

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=10,
        fast_mode=True,
        request_id=4,
        owner_epoch=2,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert mw._display_cached_batch_selection.call_count == 1
    assert _batch_policy_context(controller).active is False


@pytest.mark.unit
def test_scoped_runtime_input_supersede_rejects_affected_preview_completion(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 4
    controller._active_run_id = 10
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=4,
        epoch=2,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=4, run_id=10, cache_key="preview-cache", queue_ids=["id1"], queue_names=["set1"], total=1, pos=0, preview_scope_set_ids=("id1",), runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0})
    callback_identity = _capture_callback_identity(
        controller,
        run_id=10,
        fast_mode=True,
        request_id=4,
        owner_epoch=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw._display_cached_batch_selection.return_value = True

    controller.supersede_active_work_for_authoritative_mechanism_transition(
        epoch=8,
        affected_set_ids=("id1",),
        close_preview_runtime_owner=False,
    )

    assert controller.run_state.preview_ownership.request_id is None

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    mw._display_cached_batch_selection.assert_not_called()
    mw.set_data.assert_not_called()


@pytest.mark.unit
def test_scoped_runtime_input_supersede_rejects_affected_preview_error(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 4
    controller._active_run_id = 10
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._preview_ownership = PreviewOwnershipState(
        request_id=4,
        epoch=2,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=4, run_id=10, cache_key="preview-cache", queue_ids=["id1"], queue_names=["set1"], total=1, pos=0, preview_scope_set_ids=("id1",), runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0})
    callback_identity = _capture_callback_identity(
        controller,
        run_id=10,
        fast_mode=True,
        request_id=4,
        owner_epoch=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="preview-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.message_box_critical = MagicMock()
    mw._status_label.setText("Running preview")

    controller.supersede_active_work_for_authoritative_mechanism_transition(
        epoch=8,
        affected_set_ids=("id1",),
        close_preview_runtime_owner=False,
    )

    controller._on_simulation_error(
        {"kind": "simulation_error", "message": "stale preview failed"},
        callback_identity=callback_identity,
    )

    mw.message_box_critical.assert_not_called()
    assert mw._status_label.text == "Ready"


@pytest.mark.unit
def test_nonowning_stale_fast_error_does_not_reset_explicit_run_status_progress(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 3
    controller._pending_slider_sim_request_id = None
    controller._pending_slider_simulation = False
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, run_id=11, request_id=1)
    stale_preview_identity = _capture_callback_identity(
        controller,
        run_id=11,
        fast_mode=True,
        request_id=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=3)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    explicit_worker = _FakeWorker(running=True, wait_returns=False)
    explicit_worker._request_id = 3  # type: ignore[attr-defined]
    explicit_worker._fast_mode = False  # type: ignore[attr-defined]
    controller._simulation_worker = explicit_worker

    mw._slider_triggered_simulation = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._status_label.setText("Running simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller._on_simulation_error("boom", callback_identity=stale_preview_identity)

    policy_context = _batch_policy_context(controller)
    assert policy_context.active is True
    assert policy_context.fast_mode is False
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is explicit_worker
    assert controller._retained_simulation_workers == []
    assert scheduled == []
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True
    assert mw._status_label.text == "Running simulation..."
    assert mw._sim_progress.value == 41
    message_box.assert_not_called()


@pytest.mark.unit
def test_stale_fast_error_with_pending_newer_preview_replays_without_showing_error_for_stopped_old_worker(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid_new)
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_simulation = True
    controller._discarded_slider_preview_generation_id = None
    controller._active_run_id = 11
    seed_batch_context(controller.batch_context_owner)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid_new),
        epoch=3,
        target_set_ids=("id1",),
    )
    worker = _FakeWorker(running=False, wait_returns=True)
    worker._request_id = int(rid_old)  # type: ignore[attr-defined]
    worker._fast_mode = True  # type: ignore[attr-defined]
    controller._simulation_worker = worker

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(41)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=11,
        fast_mode=True,
        request_id=int(rid_old),
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert mw._status_label.text == "Updating simulation..."
    assert mw._sim_progress.value == 41
    message_box.assert_not_called()


@pytest.mark.unit
def test_stale_fast_error_without_pending_still_cleans_up_active_run(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._latest_sim_request_id = 2
    controller._pending_slider_sim_request_id = 2
    controller._pending_slider_simulation = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=2,
        epoch=5,
        target_set_ids=("id1",),
    )

    controller._active_run_id = 7
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=1)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    mw._status_label.setText("Updating simulation...")
    mw._sim_progress.setValue(0)

    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=7,
        fast_mode=True,
        request_id=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller._simulation_worker is None
    assert scheduled == [controller._run_simulation_from_slider]
    message_box.assert_not_called()


@pytest.mark.unit
def test_stale_fast_error_treats_same_request_as_superseded_after_owner_epoch_changes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]
    message_box.assert_not_called()


@pytest.mark.unit
def test_fast_error_surfaces_current_owner_even_when_latest_request_id_has_advanced(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 6
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=8,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=5, preview_owner_epoch=8)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=11,
        fast_mode=True,
        request_id=5,
        owner_epoch=8,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    critical.assert_called_once()
    assert scheduled == []
    assert mw._status_label.text == "Simulation failed"


@pytest.mark.unit
def test_fast_error_treats_missing_owner_epoch_for_current_owner_as_superseded(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 6
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=8,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=5)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=11,
        fast_mode=True,
        request_id=5,
        owner_epoch=None,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    critical.assert_not_called()
    assert scheduled == []
    assert mw._status_label.text == "Ready"


@pytest.mark.unit
def test_worker_signal_error_preserves_owner_epoch_for_stale_fast_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="preview-cache",
    )

    worker.error.emit("boom")

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]
    message_box.assert_not_called()


@pytest.mark.unit
def test_worker_signal_error_uses_captured_owner_epoch_after_context_turnover(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    worker = _QtSignalWorker(running=False)
    controller._latest_sim_request_id = 7
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_simulation = True
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=7, preview_owner_epoch=3)

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    _connect_worker_application_signals(
        controller,
        worker,
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=3,
        set_name="set2",
        set_id="id2",
        cache_key="preview-cache",
    )
    policy_context = _batch_policy_context(controller).evolve(preview_owner_epoch=4)
    controller.batch_context_owner.serialize_completion_policy_context(policy_context)

    worker.error.emit("boom")

    assert _batch_policy_context(controller).active is False
    assert scheduled == [controller._run_simulation_from_slider]
    message_box.assert_not_called()


@pytest.mark.unit
def test_second_stale_fast_error_preserves_queued_replay_snapshot_before_timer_fires(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    message_box = MagicMock()
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", message_box)

    controller._latest_sim_request_id = 8
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=8,
        epoch=2,
        target_set_ids=("id1",),
    )
    controller.run_state.pending_slider_preview_launch = controller.run_state.pending_slider_preview_launch.__class__(
        active=False,
        request_id=7,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=False, parallel=True, fast_mode=True, request_id=7, preview_owner_epoch=1, keep_lane_pool_alive=True)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=11,
        fast_mode=True,
        request_id=7,
        owner_epoch=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("id2",)
    message_box.assert_not_called()


@pytest.mark.unit
def test_public_simulation_callbacks_do_not_expose_owner_epoch_parameter(controller: SimulationController):
    assert "owner_epoch" not in inspect.signature(controller.on_simulation_complete).parameters
    assert "owner_epoch" not in inspect.signature(controller.on_simulation_error).parameters


@pytest.mark.unit
def test_slider_run_deferral_does_not_set_updating_status_when_no_new_run_starts(
    mw: _FakeMainWindow, controller: SimulationController
):
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = rid

    seed_batch_context(controller.batch_context_owner, active=True, fast_mode=True)
    controller._simulation_worker = None
    controller._simulation_running = False

    mw._status_label.setText("Ready")

    controller._run_simulation_from_slider()

    assert controller._pending_slider_simulation is True
    assert controller._simulation_running is False
    assert mw._status_label.text == "Ready"

@pytest.mark.unit
def test_slider_run_supersedes_active_fast_parallel_preview_for_newer_request(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid_old = controller._next_sim_request_id()
    rid_new = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid_new)
    controller._pending_slider_target_set_ids = ("id1",)
    controller._latest_sim_request_id = int(rid_new)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=True, request_id=int(rid_old))

    controller._run_simulation_from_slider()

    assert _batch_policy_context(controller).active is False
    controller.run_simulation_internal.assert_called_once()
    assert controller.run_simulation_internal.call_args.kwargs["request_id"] == int(rid_new)
    assert controller.run_simulation_internal.call_args.kwargs["reuse_parallel_lane_pool"] is True

@pytest.mark.unit
def test_slider_run_blocks_launch_while_retained_worker_is_still_running(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    worker = make_stubborn_worker(_FakeWorker)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    controller._simulation_running = False
    mw._status_label.setText("Ready")
    mw._run_btn.setEnabled(True)
    mw._stop_btn.setEnabled(False)
    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id1",),
        request_id=7,
    )

    controller.launch_pending_slider_preview_replay()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert controller.has_running_owned_simulation_workers() is True
    assert controller._retained_simulation_workers == [worker]
    assert mw._status_label.text == "Cancelling previous simulation..."
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

@pytest.mark.unit
def test_retained_worker_finish_replays_latest_pending_slider_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._pending_slider_target_set_ids = ("id1",)
    controller._simulation_running = False
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(rid),
        epoch=2,
        target_set_ids=("id1",),
    )

    mw._status_label.setText("Ready")
    mw._run_btn.setEnabled(True)
    mw._stop_btn.setEnabled(False)
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == int(rid)
    assert scheduled == []

    worker._running = False
    worker.finished.emit()

    assert controller._retained_simulation_workers == []
    assert controller.has_running_owned_simulation_workers() is False
    assert scheduled == [controller._run_simulation_from_slider]

    scheduled[0]()

    assert controller.run_simulation_internal.call_count == 1
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True


@pytest.mark.unit
def test_retained_worker_finish_replays_current_owner_even_when_latest_request_id_has_advanced(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    controller._latest_sim_request_id = 6
    controller._pending_slider_sim_request_id = 5
    controller._pending_slider_simulation = True
    controller._pending_slider_target_set_ids = ("id1",)
    controller._simulation_running = False
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=8,
        target_set_ids=("id1",),
    )

    worker._running = False
    worker.finished.emit()

    assert controller._retained_simulation_workers == []
    assert controller.has_running_owned_simulation_workers() is False
    assert controller._pending_slider_sim_request_id == 5
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_retained_worker_finish_preserves_reserved_future_slider_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    assert controller._simulation_worker is None
    assert controller._retained_simulation_workers == [worker]

    controller._latest_sim_request_id = 5
    controller._pending_slider_sim_request_id = 6
    controller._pending_slider_simulation = True
    controller._pending_slider_target_set_ids = ("id1",)
    controller._simulation_running = False
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=6,
        epoch=3,
        target_set_ids=("id1",),
    )

    worker._running = False
    worker.finished.emit()

    assert controller._retained_simulation_workers == []
    assert controller.has_running_owned_simulation_workers() is False
    assert controller._pending_slider_sim_request_id == 6
    assert scheduled == [controller._run_simulation_from_slider]

@pytest.mark.unit
def test_retained_worker_finish_cancels_species_timer_before_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    species_timer = _ActiveTimer()
    mw._species_slider_update_timer = species_timer

    worker = _QtSignalWorker(running=True)
    controller._simulation_worker = worker
    controller._release_current_simulation_worker()

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_simulation = True
    controller._pending_slider_target_set_ids = ("id1",)
    controller._simulation_running = False
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"

    def _fire_species_timeout_if_still_active() -> None:
        if species_timer.isActive():
            controller.launch_pending_slider_preview_replay()

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._pending_slider_simulation is True
    assert scheduled == []

    worker._running = False
    worker.finished.emit()

    assert species_timer.stop_calls == 1
    assert species_timer.isActive() is False
    assert scheduled == [controller._run_simulation_from_slider]

    scheduled[0]()
    assert controller.run_simulation_internal.call_count == 1

    _fire_species_timeout_if_still_active()
    assert controller.run_simulation_internal.call_count == 1

@pytest.mark.unit
def test_supersede_parallel_batch_run_soft_invalidates_lane_requests_and_stops_timer(controller: SimulationController):
    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_completion_poll_timer = timer

    release = threading.Event()

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            _ = task, active_timeout_s
            release.wait(timeout=2.0)
            return _lane_outcome(str(set_id), run_id=int(run_id), request_id=int(request_id))

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            release.set()

    pool = _FakeLanePool()
    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller._batch_parallel.ensure_lane_pool(max_lanes=2)
    controller._batch_parallel.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=["a", "b"],
        queue_names=["A", "B"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=1.0,
    )
    for sid in ("a", "b"):
        controller._batch_parallel.submit_task(
            {},
            set_id=sid,
            set_name=sid.upper(),
            callback_identity=_capture_callback_identity(
                controller,
                run_id=1,
                request_id=2,
                fast_mode=True,
                batch_set=sid.upper(),
                batch_set_id=sid,
                cache_key="",
                callback_context=controller.batch_context_owner.callback_context_snapshot(),
            ),
        )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True)

    controller._supersede_parallel_batch_run_soft()
    release.set()
    controller.parallel_batch.join_active_requests(timeout_s=2.0)
    assert not controller._batch_parallel.has_active_requests()
    assert controller._batch_parallel.has_lane_pool()
    assert pool.close_calls == []
    timer.stop.assert_called()

@pytest.mark.unit
def test_superseded_parallel_batch_lane_outcome_error_is_drained_deterministically(controller: SimulationController):
    release = threading.Event()

    class _FakeLanePool(_ProtocolLanePool):
        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            _ = task, active_timeout_s
            release.wait(timeout=2.0)
            return _lane_outcome(str(set_id), run_id=int(run_id), request_id=int(request_id))

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            release.set()

    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: _FakeLanePool()
    controller._batch_parallel.ensure_lane_pool(max_lanes=1)
    controller._batch_parallel.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=["sid"],
        queue_names=["set1"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=1.0,
    )
    controller._batch_parallel.submit_task(
        {},
        set_id="sid",
        set_name="set1",
        callback_identity=_capture_callback_identity(
            controller,
            run_id=1,
            request_id=2,
            fast_mode=True,
            batch_set="set1",
            batch_set_id="sid",
            cache_key="",
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
        ),
    )
    seed_batch_context(controller.batch_context_owner, active=False, parallel=False)
    controller._record_nonfatal_exception = MagicMock()

    controller._supersede_parallel_batch_run_soft()
    release.set()
    controller.parallel_batch.join_active_requests(timeout_s=2.0)
    controller._poll_parallel_batch_completions()

    assert not controller._batch_parallel.has_active_requests()
    controller._record_nonfatal_exception.assert_not_called()

@pytest.mark.unit
def test_stale_lane_request_bookkeeping_does_not_abort_active_run(controller: SimulationController):
    submitted: list[tuple[str, dict[str, object]]] = []
    release = threading.Event()

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []
            self.ready_lane_count = 999

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            _ = active_timeout_s
            submitted.append((self.label, dict(task)))
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id=self.label,
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=str(set_id),
                owner_epoch=1,
                success=True,
                payload={"success": True, "set_id": str(set_id)},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            release.set()

    first = _FakeLanePool("initial")
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: first
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    controller.parallel_batch.begin_run(
        run_id=11,
        request_id=22,
        fast_mode=False,
        queue_ids=["current"],
        queue_names=["current-set"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=1.0,
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=11,
        request_id=22,
        fast_mode=False,
        cache_key="current-cache",
        queue_ids=["current"],
        queue_names=["current-set"],
        total=1,
    )
    controller.parallel_batch.submit_task(
        {"set_id": "current"},
        set_id="current",
        set_name="current-set",
        callback_identity=_capture_callback_identity(
            controller,
            run_id=11,
            fast_mode=False,
            request_id=22,
            batch_set="current-set",
            batch_set_id="current",
            cache_key="current-cache",
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
        ),
    )
    controller._error_handling_owner.handle_error = MagicMock()
    controller._completion_callback_owner.handle_completion = MagicMock()
    controller._record_nonfatal_exception = MagicMock()

    controller._poll_parallel_batch_completions()

    controller._error_handling_owner.handle_error.assert_not_called()
    controller._record_nonfatal_exception.assert_not_called()
    assert _batch_policy_context(controller).active is True
    assert first.close_calls == []
    assert controller.parallel_batch.has_lane_pool()
    assert controller.parallel_batch.has_active_requests()

    release.set()
    controller.parallel_batch.join_active_requests(timeout_s=2.0)
    controller._poll_parallel_batch_completions()
    controller._completion_callback_owner.handle_completion.assert_called_once()

    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["fresh"], queue_names=["fresh-set"], run_id=12, request_id=23, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={"fresh": _test_simulation_plan_payload(set_id="fresh", set_name="fresh-set", initials={"A": 1.0}, cache_key="current-cache")})
    controller.ui.batch.batch_initials_for_row = MagicMock(return_value={"A": 1.0})

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert controller.parallel_batch.has_lane_pool()
    assert [label for label, task in submitted if task.get("set_id") == "fresh"] == [first.label]

@pytest.mark.unit
def test_parallel_keep_lane_pool_alive_completion_stops_polling_without_retained_superseded_requests(
    mw: _FakeMainWindow, controller: SimulationController
):
    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_completion_poll_timer = timer

    controller._record_nonfatal_exception = MagicMock()
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=11, fast_mode=False, cache_key="ck", queue_ids=["sid"], queue_names=["set1"], total=1)

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="sid",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert _batch_policy_context(controller).active is False
    assert not controller._batch_parallel.has_active_requests()
    controller._record_nonfatal_exception.assert_not_called()
    timer.stop.assert_called_once()

@pytest.mark.unit
def test_superseded_parallel_batch_lane_outcome_error_payload_keeps_healthy_pool_alive(controller: SimulationController):
    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    pool = _FakeLanePool()
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner, active=False, parallel=False)
    controller._record_nonfatal_exception = MagicMock()

    controller._poll_parallel_batch_completions()

    assert controller.parallel_batch.has_lane_pool()
    assert pool.close_calls == []
    controller._record_nonfatal_exception.assert_not_called()

@pytest.mark.unit
def test_primary_explicit_completion_preserves_fresh_cache_during_post_run_species_sync(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _Mechanism:
        def species_names(self) -> list[str]:
            return ["A", "C"]

    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, keep_lane_pool_alive=False, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids=["id1"], queue_names=["set1"], total=1)
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    mw._sync_batch_species_columns = MagicMock()

    result = _successful_result_payload()
    result["mechanism"] = _Mechanism()
    result["mechanism_text"] = "reaction: A -> C ; k=0.1"
    result["species_names"] = ["A", "C"]

    _complete_with_callback_identity(
        controller,
        result,
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw._sync_batch_species_columns.assert_called_once_with(["A", "C"], preserve_active_cache=True)
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)


@pytest.mark.unit
def test_callback_identity_requires_cache_key_instead_of_recovering_current_cache(
    controller: SimulationController,
):
    with pytest.raises(ValueError, match="cache_key"):
        SimulationCallbackIdentity.capture(
            run_id=7,
            fast_mode=False,
            request_id=11,
            owner_epoch=None,
            batch_set="set1",
            batch_set_id="id1",
            cache_key=None,  # type: ignore[arg-type]
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
            simulation_identity={},
        )


@pytest.mark.unit
def test_on_simulation_complete_later_completion_does_not_widen_narrowed_valid_subset(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, explicit_cache_preview_token="narrow-preview-token", explicit_cache_preview_scope_set_ids=("id1",), explicit_cache_valid_set_ids=("id1",))
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_preview_token = "narrow-preview-token"
    controller.batch_cache.active_cache_preview_scope_set_ids = ("id1",)
    controller.batch_cache.active_cache_valid_set_ids = ("id1",)

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert controller.batch_cache.active_cache_preview_token == "narrow-preview-token"
    assert controller.batch_cache.active_cache_preview_scope_set_ids == ("id1",)
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)

@pytest.mark.unit
def test_on_simulation_complete_redraw_falls_back_to_current_result_when_constrained_subset_draw_returns_false(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, keep_lane_pool_alive=False, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, explicit_cache_valid_set_ids=("id2",))
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id2",)
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False
    mw.set_data.reset_mock()

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["valid_set_ids"] == ("id2",)
    assert kwargs["allow_fallback"] is False
    mw.set_data.assert_called_once()
    assert mw.set_data.call_args.kwargs["label"] == "set2"

@pytest.mark.unit
def test_on_simulation_complete_coalesced_flush_uses_valid_subset_without_fallback(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, explicit_cache_valid_set_ids=("id2",))
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id2",)
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = False

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert mw._display_cached_batch_selection.call_count == 0

    controller._flush_slider_plot_updates()

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["valid_set_ids"] == ("id2",)
    assert kwargs["allow_fallback"] is False

@pytest.mark.unit
def test_on_simulation_complete_coalesced_flush_keeps_valid_subset_after_dirty_reset(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", total=2, completed_set_ids=["id1"], explicit_cache_valid_set_ids=("id1", "id2"), pending_workspace_reset_set_ids=["id2"], pending_dirty_reset_generation_by_set_id={"id2": 1})
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id1", "id2")
    mw._dirty_state_generations = {"id2": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = False
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._shown_batch_set_ids.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = True

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["selected_sets"] == ["id1", "id2"]
    assert kwargs["valid_set_ids"] == ("id1", "id2")
    assert kwargs["allow_fallback"] is False


@pytest.mark.unit
def test_on_simulation_complete_normalizes_scalar_context_ids_before_dirty_reset(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller._active_run_id = 7
    controller._latest_sim_request_id = 11
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, run_id=7, request_id=11, fast_mode=False, cache_key="fresh-current-cache", queue_ids="id1", queue_names="set1", primary_set_id="id1", total=2, completed_set_ids="id1", explicit_cache_valid_set_ids=("id1", "id2"), pending_workspace_reset_set_ids="id2", pending_dirty_reset_generation_by_set_id={"id2": 1})
    controller.batch_cache.active_cache_key = "fresh-current-cache"
    controller.batch_cache.active_cache_valid_set_ids = ("id1", "id2")
    mw._dirty_state_generations = {"id2": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._batch_set_ids_for_scope.return_value = ["id1"]
    mw._shown_batch_set_ids.return_value = ["id1", "id2"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "id1"
    mw._display_cached_batch_selection.return_value = True

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=7,
        fast_mode=False,
        request_id=11,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="fresh-current-cache",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id2"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id2"])


@pytest.mark.unit
def test_queue_slider_plot_update_gates_by_request_and_run_ids(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 2
    controller._active_run_id = 10

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=1,
        run_id=10,
        slider_triggered=True,
    )
    assert controller._pending_slider_plot_set_ids == set()

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=2,
        run_id=11,
        slider_triggered=True,
    )
    assert controller._pending_slider_plot_set_ids == set()

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="ck",
        request_id=2,
        run_id=10,
        slider_triggered=False,
    )
    assert controller._pending_slider_plot_set_ids == {"s1"}
    assert controller._pending_slider_plot_cache_kind == "result"

    assert controller._slider_plot_coalesce_timer.isActive()
    assert int(controller._slider_plot_coalesce_timer.interval()) >= 1

@pytest.mark.unit
def test_flush_slider_plot_updates_merges_pending_sets_and_calls_display(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = {"b"}
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("a", "b"),
    )
    controller._pending_slider_plot_owner_request_id = 1
    controller._pending_slider_plot_owner_epoch = 1

    mw._shown_batch_set_ids.return_value = ["a", "b"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "a"
    mw._display_cached_batch_selection.return_value = True

    ok = controller._flush_slider_plot_updates()
    assert ok is True
    args, kwargs = mw._display_cached_batch_selection.call_args
    assert kwargs["cache_key"] == "cache-key"
    assert kwargs["prefer_set"] == "a"
    assert kwargs["selected_sets"] == ["a", "b"]

@pytest.mark.unit
def test_flush_slider_plot_updates_force_uses_cache_keys_when_no_selection(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = set()
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("x", "y"),
    )
    controller._pending_slider_plot_owner_request_id = 1
    controller._pending_slider_plot_owner_epoch = 1

    mw._shown_batch_set_ids.return_value = []
    mw._display_cached_batch_selection.return_value = True

    controller.batch_cache.put_completion_entry(
        cache_key="cache-key",
        set_id="x",
        is_preview=True,
        t=[],
        series={},
    )
    controller.batch_cache.put_completion_entry(
        cache_key="cache-key",
        set_id="y",
        is_preview=True,
        t=[],
        series={},
    )

    ok = controller._flush_slider_plot_updates(force=True)
    assert ok is True
    _args, kwargs = mw._display_cached_batch_selection.call_args
    assert sorted(kwargs["selected_sets"]) == ["x", "y"]

@pytest.mark.unit
def test_flush_slider_plot_updates_uses_shown_sets_not_highlighted_selection(mw: _FakeMainWindow, controller: SimulationController):
    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_plot_set_ids = {"dirty"}
    controller._pending_slider_plot_cache_key = "cache-key"
    controller._pending_slider_plot_cache_kind = "preview"
    controller._pending_slider_plot_request_id = 1
    controller._pending_slider_plot_run_id = 2
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("shown-a", "dirty"),
    )
    controller._pending_slider_plot_owner_request_id = 1
    controller._pending_slider_plot_owner_epoch = 1

    mw._batch_set_ids_for_scope.return_value = ["selected-only"]
    mw._shown_batch_set_ids.return_value = ["shown-a", "dirty"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "shown-a"
    mw._display_cached_batch_selection.return_value = True

    ok = controller._flush_slider_plot_updates()
    assert ok is True
    _args, kwargs = mw._display_cached_batch_selection.call_args
    assert kwargs["selected_sets"] == ["shown-a", "dirty"]


@pytest.mark.unit
def test_flush_slider_plot_updates_rejects_pending_preview_after_same_request_owner_epoch_changes(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 2
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("dirty",),
    )

    controller._queue_slider_plot_update(
        set_id="dirty",
        cache_key="cache-key",
        request_id=1,
        run_id=2,
        slider_triggered=True,
    )

    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=2,
        target_set_ids=("other",),
    )
    mw._shown_batch_set_ids.return_value = ["dirty"]

    assert controller._flush_slider_plot_updates() is False
    mw._display_cached_batch_selection.assert_not_called()


@pytest.mark.unit
def test_flush_slider_plot_updates_uses_durable_preview_owner_not_transient_worker_state(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 2
    controller._active_run_id = 10
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._simulation_worker._request_id = 1  # type: ignore[attr-defined]
    controller._simulation_worker._fast_mode = True  # type: ignore[attr-defined]
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=3,
        target_set_ids=("s1",),
    )

    controller._queue_slider_plot_update(
        set_id="s1",
        cache_key="cache-key",
        request_id=1,
        run_id=10,
        slider_triggered=True,
    )

    controller._simulation_worker = None
    controller._slider_simulation_active = False
    mw._shown_batch_set_ids.return_value = ["s1"]
    mw._batch_current_row.return_value = 0
    mw._batch_set_id_for_row.return_value = "s1"
    mw._display_cached_batch_selection.return_value = True

    assert controller._flush_slider_plot_updates() is True
    mw._display_cached_batch_selection.assert_called_once()


@pytest.mark.unit
def test_queue_slider_plot_update_resets_pending_preview_batch_when_owner_changes(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._active_run_id = 4
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("set1",),
    )

    controller._queue_slider_plot_update(
        set_id="set1",
        cache_key="preview-1",
        request_id=1,
        run_id=4,
        slider_triggered=True,
    )

    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=2,
        epoch=2,
        target_set_ids=("set2",),
    )
    controller._queue_slider_plot_update(
        set_id="set2",
        cache_key="preview-2",
        request_id=2,
        run_id=4,
        slider_triggered=True,
    )

    pending = controller.plot_coalescer.pending
    assert pending.set_ids == {"set2"}
    assert pending.cache_key == "preview-2"
    assert pending.request_id == 2
    assert pending.accepted_owner_request_id == 2
    assert pending.accepted_owner_epoch == 2


@pytest.mark.unit
def test_consume_parallel_batch_outcome_success_calls_on_complete_and_clears_maps(mw: _FakeMainWindow, controller: SimulationController):
    outcome = _lane_outcome("sid", {"payload": 123})
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"})

    controller._completion_callback_owner.handle_completion = MagicMock()
    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
    )
    assert ok is True
    assert not controller._batch_parallel.has_active_requests()
    controller._completion_callback_owner.handle_completion.assert_called_once()


@pytest.mark.unit
def test_consume_parallel_batch_outcome_uses_captured_context_for_runtime_stale_after_progress_turnover(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 0, "id2": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 1},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=[],
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    outcome = _lane_outcome("id2", _successful_result_payload(), run_id=3, request_id=5)
    _install_active_lane_outcomes(
        controller,
        {"id2": outcome},
        set_names={"id2": "set2"},
        callback_identities={"id2": identity},
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 2},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=["id1"],
        primary_set_id="id1",
    )
    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller.ui.results.publish_simulation_completion_result = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id2",
        outcome=outcome,
        run_id=3,
        request_id=5,
        source="callback",
    )

    assert ok is True
    policy_context = _batch_policy_context(controller)
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert policy_context.active is False
    assert policy_context.completed_set_ids == ("id1", "id2")
    assert callback_context.stale_runtime_input_set_ids == ("id2",)
    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()


@pytest.mark.unit
def test_consume_parallel_batch_outcome_error_uses_captured_context_for_runtime_stale_after_progress_turnover(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 0, "id2": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 1},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=[],
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    outcome = _lane_outcome(
        "id2",
        success=False,
        run_id=3,
        request_id=5,
        failure={"kind": "internal_error", "message": "stale boom"},
    )
    _install_active_lane_outcomes(
        controller,
        {"id2": outcome},
        set_names={"id2": "set2"},
        callback_identities={"id2": identity},
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 2},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=["id1"],
        primary_set_id="id1",
    )

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id2",
        outcome=outcome,
        run_id=3,
        request_id=5,
        source="callback",
    )

    assert ok is True
    policy_context = _batch_policy_context(controller)
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert policy_context.active is False
    assert policy_context.completed_set_ids == ("id1", "id2")
    assert callback_context.stale_runtime_input_set_ids == ("id2",)
    critical.assert_not_called()
    assert mw._status_label.text != "Simulation failed"


@pytest.mark.unit
def test_consume_parallel_batch_outcome_on_complete_exception_reports_error_and_shutdown(
    mw: _FakeMainWindow, controller: SimulationController
):
    outcome = _lane_outcome("sid", {"payload": 123})
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"})

    controller._completion_callback_owner.handle_completion = MagicMock(side_effect=RuntimeError("ui boom"))
    controller._error_handling_owner.handle_error = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
    )
    assert ok is False
    controller._error_handling_owner.handle_error.assert_called_once()
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)

@pytest.mark.unit
def test_consume_parallel_batch_outcome_error_calls_on_error_and_shutdown(mw: _FakeMainWindow, controller: SimulationController):
    outcome = _lane_outcome(
        "sid",
        success=False,
        failure={"kind": "internal_error", "message": "boom"},
    )
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"})

    controller._error_handling_owner.handle_error = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
        fast_mode=True,
        source="callback",
    )
    assert ok is False
    controller._error_handling_owner.handle_error.assert_called_once()
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_consume_parallel_batch_outcome_error_is_set_scoped_when_batch_sets_remain(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _lane_outcome(
        "bad",
        {
            "success": False,
            "error": build_simulation_failure(
                "timeout",
                "Simulation timed out after 0.2 seconds.",
                details={"walltime_s": 0.2},
            ),
        },
    )
    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome, "ok": _lane_outcome("ok")},
        set_names={"bad": "Bad Set", "ok": "OK Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["bad", "ok"], queue_names=["Bad Set", "OK Set"], total=2)

    controller._error_handling_owner.handle_error = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        source="callback",
    )

    assert ok is True
    controller._error_handling_owner.handle_error.assert_not_called()
    controller._shutdown_batch_lane_pool.assert_not_called()
    assert _batch_policy_context(controller).active is True
    summary = controller.batch_context_owner.completion_summary()
    completion_state = controller.batch_context_owner.completion_state()
    assert summary.failed_set_ids == ("bad",)
    assert completion_state is not None
    assert "bad" in completion_state.completed_set_ids


@pytest.mark.unit
def test_parallel_batch_outcome_missing_identity_metadata_is_rejected_not_self_validated():
    outcome = _lane_outcome("sid", {"payload": 123}, run_id=1, request_id=2)

    resolution = resolve_parallel_batch_outcome(
        set_id="sid",
        outcome=outcome,
        metadata={},
    )

    assert resolution.stale is True
    assert resolution.error_payload is not None
    assert resolution.error_payload["details"]["missing_identity_metadata"] == (
        "run_id",
        "request_id",
    )


@pytest.mark.unit
def test_stale_batch_outcome_by_request_id_clears_last_active_request(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    outcome = _lane_outcome("sid", {"payload": 123}, request_id=99)
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"})
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck")
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
        run_id=1,
        request_id=2,
    )

    assert ok is False
    assert _batch_policy_context(controller).active is False
    assert not controller._batch_parallel.has_active_requests()
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_stale_batch_outcome_by_owner_epoch_is_rejected(controller: SimulationController):
    outcome = _lane_outcome("sid", {"payload": 123}, owner_epoch=3)
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"}, owner_epoch=4)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck")

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
        run_id=1,
        request_id=2,
    )

    assert ok is False
    assert _batch_policy_context(controller).active is False
    assert not controller._batch_parallel.has_active_requests()


@pytest.mark.unit
def test_stale_batch_outcome_rejects_without_mutating_multiset_failure_state(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    stale = _lane_outcome("bad", {"payload": 123}, request_id=99)
    healthy_payload = _successful_result_payload()
    healthy_payload.update({"success": True, "set_id": "ok", "set_name": "OK Set"})
    healthy = _lane_outcome("ok", healthy_payload, request_id=2)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["bad", "ok"], queue_names=["Bad Set", "OK Set"], total=2, pending_workspace_reset_set_ids=["bad", "ok"], pending_dirty_reset_generation_by_set_id={"bad": 1, "ok": 1}, explicit_cache_valid_set_ids=("bad", "ok"))
    failed_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="Bad Set",
        batch_set_id="bad",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    success_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    _install_active_lane_outcomes(
        controller,
        {"bad": stale, "ok": healthy},
        set_names={"bad": "Bad Set", "ok": "OK Set"},
        callback_identities={"bad": failed_identity, "ok": success_identity},
    )
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("bad", "ok")
    mw._dirty_state_generations = {"bad": 1, "ok": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw.message_box_critical = MagicMock()

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=stale,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
    ) is False
    assert _batch_policy_context(controller).active is False
    summary = controller.batch_context_owner.completion_summary()
    completion_state = controller.batch_context_owner.completion_state()
    assert summary.failed_set_ids == ()
    assert completion_state is None or "bad" not in completion_state.completed_set_ids
    assert not controller._batch_parallel.has_active_requests()
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_parallel_batch_outcome_missing_callback_context_does_not_use_current_runtime_stale_context(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    payload = _successful_result_payload()
    payload.update({"success": True, "set_id": "id1", "set_name": "set1"})
    outcome = _lane_outcome("id1", payload, request_id=2)
    _install_active_lane_outcomes(
        controller,
        {"id1": outcome},
        set_names={"id1": "set1"},
        with_callback_identity=False,
    )
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller._simulation_running = True
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 4}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        total=1,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0},
    )
    controller._completion_callback_owner.handle_completion = MagicMock()
    mw.message_box_critical = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id1",
        outcome=outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="test",
    )

    assert ok is False
    controller._completion_callback_owner.handle_completion.assert_not_called()
    state = controller.batch_context_owner.completion_state()
    assert state is None or state.active is False
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_parallel_batch_outcome_stale_run_decision_does_not_mutate_scoped_failure_state(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed = _timeout_failure_outcome("bad")
    healthy = _lane_outcome("ok", _successful_result_payload())
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        explicit_cache_valid_set_ids=("bad", "ok"),
    )
    failed_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="Bad Set",
        batch_set_id="bad",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    success_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    _install_active_lane_outcomes(
        controller,
        {"bad": failed, "ok": healthy},
        set_names={"bad": "Bad Set", "ok": "OK Set"},
        callback_identities={"bad": failed_identity, "ok": success_identity},
    )
    controller._active_run_id = 99
    controller._simulation_running = True
    controller._completion_callback_owner.handle_completion = MagicMock()
    controller._error_handling_owner.handle_error = MagicMock()
    controller._parallel_batch_outcome_owner.handle_scoped_failure = MagicMock(return_value=True)
    mw.message_box_critical = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
    )

    assert ok is True
    controller._parallel_batch_outcome_owner.handle_scoped_failure.assert_not_called()
    controller._completion_callback_owner.handle_completion.assert_not_called()
    controller._error_handling_owner.handle_error.assert_not_called()
    state = controller.batch_context_owner.completion_state()
    summary = controller.batch_context_owner.completion_summary()
    assert state is not None
    assert state.active is True
    assert summary.failed_set_ids == ()
    assert state.completed_set_ids == ()
    assert controller._batch_parallel.active_request_metadata("bad")
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_poll_parallel_batch_outcome_rejects_owner_epoch_mismatch_after_pop(
    mw: _FakeMainWindow,
    controller: SimulationController,
    caplog,
):
    outcome = _lane_outcome("sid", {"payload": 123}, run_id=1, request_id=2, owner_epoch=3)
    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"}, owner_epoch=4)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck")
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._completion_callback_owner.handle_completion = MagicMock()

    controller._poll_parallel_batch_completions()

    controller._completion_callback_owner.handle_completion.assert_not_called()
    assert "Rejected stale batch lane outcome" in caplog.text


@pytest.mark.unit
def test_poll_parallel_batch_runtime_session_stale_record_does_not_reset_current_run(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
    caplog,
):
    outcome = _lane_outcome("sid", {"payload": 123}, run_id=1, request_id=99, owner_epoch=1)
    record = BatchCompletionRecord(
        metadata=BatchRequestMetadata(
            set_id="sid",
            set_name="set1",
            run_id=1,
            request_id=99,
            generation=1,
            preview_owner_epoch=None,
            expected_owner_epoch=None,
        ),
        outcome=outcome,
        completed_ts=1.0,
        request_metadata={
            "set_name": "set1",
            "runtime_session_stale": {
                "expected_run_id": 1,
                "expected_request_id": 2,
                "expected_owner_epoch": None,
                "actual_run_id": 1,
                "actual_request_id": 99,
                "actual_owner_epoch": None,
            },
        },
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["sid"],
        queue_names=["set1"],
    )
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    monkeypatch.setattr(
        controller,
        "_batch_parallel_adapter",
        SimpleNamespace(
            runtime_snapshot=lambda: SimpleNamespace(active=True),
            has_active_requests=lambda: True,
            poll_completed_records=lambda: [
                BatchPolledCompletion(set_id="sid", record=record, source="scan", completed_ts=1.0)
            ],
            is_pool_stale=False,
            active_request_count=lambda: 0,
            shutdown=lambda **_kwargs: None,
        ),
    )

    controller._poll_parallel_batch_completions()

    assert "Rejected stale batch lane outcome" in caplog.text
    policy_context = _batch_policy_context(controller)
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is True
    assert mw._run_btn.isEnabled() is False
    assert mw._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_poll_parallel_batch_fast_owner_epoch_stale_consumes_active_request(
    mw: _FakeMainWindow,
    controller: SimulationController,
    caplog,
):
    outcome = _lane_outcome("sid", {"payload": 123}, run_id=1, request_id=2, owner_epoch=3)
    _install_active_lane_outcomes(
        controller,
        {"sid": outcome},
        set_names={"sid": "set1"},
        owner_epoch=4,
        fast_mode=True,
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=True,
        cache_key="ck",
        preview_owner_epoch=4,
    )
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller._completion_callback_owner.handle_completion = MagicMock()

    controller._poll_parallel_batch_completions()

    controller._completion_callback_owner.handle_completion.assert_not_called()
    assert "Rejected stale batch lane outcome" in caplog.text
    assert not controller._batch_parallel.has_active_requests()


@pytest.mark.unit
def test_parallel_batch_final_success_preserves_prior_scoped_failure_status(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")
    success_result = _successful_result_payload()
    success_result.update(
        {
            "success": True,
            "run_id": 1,
            "set_id": "ok",
            "set_name": "OK Set",
        }
    )
    success_lane_outcome = _lane_outcome("ok", success_result)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["bad", "ok"], queue_names=["Bad Set", "OK Set"], total=2, pending_workspace_reset_set_ids=["bad", "ok"], pending_dirty_reset_generation_by_set_id={"bad": 1, "ok": 1}, explicit_cache_valid_set_ids=("bad", "ok"))
    failed_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="Bad Set",
        batch_set_id="bad",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    success_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome, "ok": success_lane_outcome},
        set_names={"bad": "Bad Set", "ok": "OK Set"},
        callback_identities={"bad": failed_identity, "ok": success_identity},
    )
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("bad", "ok")
    mw._dirty_state_generations = {"bad": 1, "ok": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw.message_box_critical = MagicMock()
    publish_truth_spy = MagicMock(wraps=controller._cache_admin.publish_completion_cache_truth)
    controller._cache_admin.publish_completion_cache_truth = publish_truth_spy

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=1.0,
    ) is True
    failure_cache_state = controller.batch_context_owner.scoped_failure_cache_state()
    assert failure_cache_state.explicit_cache_valid_set_ids == ("ok",)
    assert failure_cache_state.explicit_cache_invalidated_set_ids == ("bad",)
    assert controller.batch_cache.active_cache_valid_set_ids == ("ok",)
    assert controller.batch_cache.active_cache_invalidated_set_ids == ("bad",)

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="ok",
        outcome=success_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    assert mw._status_label.text == "Batch completed with 1 failed set(s)"
    assert "Simulation complete:" not in mw._status_label.text
    publish_truth_spy.assert_called()
    assert publish_truth_spy.call_args.kwargs["active_cache_valid_set_ids"] == ("ok",)
    assert publish_truth_spy.call_args.kwargs["active_cache_invalidated_set_ids"] == ("bad",)
    assert controller.batch_cache.active_cache_valid_set_ids == ("ok",)
    assert controller.batch_cache.active_cache_invalidated_set_ids == ("bad",)
    assert mw._display_cached_batch_selection.call_args.kwargs["valid_set_ids"] == ("ok",)
    assert mw._display_cached_batch_selection.call_args.kwargs["allow_fallback"] is False
    mw.reset_mechanism_workspaces.assert_called_once_with(["ok"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["ok"])
    mw.message_box_critical.assert_called_once()


@pytest.mark.unit
def test_parallel_batch_same_poll_failure_then_success_preserves_failure_cache_truth(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")
    success_result = _successful_result_payload()
    success_result.update(
        {
            "success": True,
            "run_id": 1,
            "set_id": "ok",
            "set_name": "OK Set",
        }
    )
    success_lane_outcome = _lane_outcome("ok", success_result)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        pending_workspace_reset_set_ids=["bad", "ok"],
        pending_dirty_reset_generation_by_set_id={"bad": 1, "ok": 1},
        explicit_cache_valid_set_ids=("bad", "ok"),
    )
    failed_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="Bad Set",
        batch_set_id="bad",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    success_identity = _capture_callback_identity(
        controller,
        run_id=1,
        fast_mode=False,
        request_id=2,
        owner_epoch=None,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome, "ok": success_lane_outcome},
        set_names={"bad": "Bad Set", "ok": "OK Set"},
        callback_identities={"bad": failed_identity, "ok": success_identity},
    )
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("bad", "ok")
    mw._dirty_state_generations = {"bad": 1, "ok": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw.message_box_critical = MagicMock()
    publish_truth_spy = MagicMock(wraps=controller._cache_admin.publish_completion_cache_truth)
    controller._cache_admin.publish_completion_cache_truth = publish_truth_spy

    controller._poll_parallel_batch_completions()

    publish_truth_spy.assert_called()
    assert publish_truth_spy.call_args.kwargs["active_cache_valid_set_ids"] == ("ok",)
    assert publish_truth_spy.call_args.kwargs["active_cache_invalidated_set_ids"] == ("bad",)
    assert controller.batch_cache.active_cache_valid_set_ids == ("ok",)
    assert controller.batch_cache.active_cache_invalidated_set_ids == ("bad",)
    assert mw._display_cached_batch_selection.call_args.kwargs["valid_set_ids"] == ("ok",)
    assert mw._display_cached_batch_selection.call_args.kwargs["allow_fallback"] is False


@pytest.mark.unit
def test_completion_publication_cache_truth_uses_callback_owned_publication_context(
    controller: SimulationController,
):
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        explicit_cache_valid_set_ids=("bad", "ok"),
    )
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        completed_set_ids=["bad"],
        explicit_cache_valid_set_ids=("ok",),
        explicit_cache_invalidated_set_ids=("bad",),
    )
    cache_truth_context = controller.batch_context_owner.completion_policy_context()
    assert cache_truth_context is not None
    callback_context = controller.batch_context_owner.callback_context_with_cache_truth(
        callback_context,
        cache_truth_context,
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        completed_set_ids=[],
        explicit_cache_valid_set_ids=("bad", "ok"),
        explicit_cache_invalidated_set_ids=(),
    )
    publish_truth_spy = MagicMock(wraps=controller._cache_admin.publish_completion_cache_truth)
    controller._cache_admin.publish_completion_cache_truth = publish_truth_spy
    state = CompletionCallbackState(
        run_id=1,
        request_id=2,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        policy_context=controller.batch_context_owner.completion_policy_context(callback_context),
        ctx=callback_context,
        shutdown_requested=False,
        is_preview=False,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )

    controller._completion_publication_owner.publish_cache_truth(state)

    assert publish_truth_spy.call_args.kwargs["active_cache_valid_set_ids"] == ("ok",)
    assert publish_truth_spy.call_args.kwargs["active_cache_invalidated_set_ids"] == ("bad",)


@pytest.mark.unit
def test_completion_publication_preserves_scoped_failure_cache_truth_from_stale_callback(
    controller: SimulationController,
):
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        fast_mode=False,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        queue_names=["Bad Set", "OK Set"],
        total=2,
        explicit_cache_valid_set_ids=("bad", "ok"),
    )
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    captured_policy_context = controller.batch_context_owner.completion_policy_context(callback_context)
    assert captured_policy_context is not None
    controller.batch_context_owner.record_scoped_failure(
        set_id="bad",
        failure={"kind": "simulation_error", "message": "bad failed"},
    )
    publish_truth_spy = MagicMock(wraps=controller._cache_admin.publish_completion_cache_truth)
    controller._cache_admin.publish_completion_cache_truth = publish_truth_spy
    state = CompletionCallbackState(
        run_id=1,
        request_id=2,
        batch_set="OK Set",
        batch_set_id="ok",
        cache_key="ck",
        policy_context=captured_policy_context,
        ctx=callback_context,
        shutdown_requested=False,
        is_preview=False,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )

    controller._completion_publication_owner.publish_cache_truth(state)

    assert publish_truth_spy.call_args.kwargs["active_cache_valid_set_ids"] == ("ok",)
    assert publish_truth_spy.call_args.kwargs["active_cache_invalidated_set_ids"] == ("bad",)


@pytest.mark.unit
def test_completion_publication_missing_cache_key_without_callback_context_does_not_use_current_context(
    controller: SimulationController,
):
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        cache_key="current-cache",
        queue_ids=["id1"],
        queue_names=["set1"],
        explicit_cache_valid_set_ids=("id1",),
    )
    publish_truth_spy = MagicMock(wraps=controller._cache_admin.publish_completion_cache_truth)
    controller._cache_admin.publish_completion_cache_truth = publish_truth_spy
    state = CompletionCallbackState(
        run_id=1,
        request_id=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key=None,
        policy_context=None,
        ctx=None,
        shutdown_requested=False,
        is_preview=False,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )

    controller._completion_publication_owner.publish_cache_truth(state)

    assert state.cache_key is None
    publish_truth_spy.assert_not_called()


@pytest.mark.unit
def test_parallel_batch_final_scoped_failure_finalizes_prior_success(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")

    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome},
        set_names={"bad": "Bad Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["ok", "bad"], queue_names=["OK Set", "Bad Set"], total=2, completed_set_ids=["ok"], pending_workspace_reset_set_ids=["ok", "bad"], pending_dirty_reset_generation_by_set_id={"ok": 1, "bad": 1}, explicit_cache_valid_set_ids=("ok", "bad"))
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("ok", "bad")
    mw._dirty_state_generations = {"ok": 1, "bad": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._shown_batch_set_ids.return_value = ["ok"]
    mw._display_cached_batch_selection.return_value = True
    mw.message_box_critical = MagicMock()

    controller._queue_slider_plot_update(
        set_id="ok",
        cache_key="ck",
        request_id=2,
        run_id=1,
        slider_triggered=False,
        valid_set_ids=("ok",),
        allow_fallback=False,
    )

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    mw.reset_mechanism_workspaces.assert_called_once_with(["ok"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["ok"])
    kwargs = mw._display_cached_batch_selection.call_args.kwargs
    assert kwargs["selected_sets"] == ["ok"]
    assert kwargs["valid_set_ids"] == ("ok",)
    assert kwargs["allow_fallback"] is False
    assert controller.batch_cache.active_cache_valid_set_ids == ("ok",)
    assert controller.batch_cache.active_cache_invalidated_set_ids == ("bad",)
    assert mw._status_label.text == "Batch completed with 1 failed set(s)"


@pytest.mark.unit
def test_parallel_batch_final_scoped_failure_prunes_reset_sets_from_pending_replay(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome},
        set_names={"bad": "Bad Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["ok", "bad"], queue_names=["OK Set", "Bad Set"], total=2, completed_set_ids=["ok"], pending_workspace_reset_set_ids=["ok", "bad"], pending_dirty_reset_generation_by_set_id={"ok": 1, "bad": 1}, explicit_cache_valid_set_ids=("ok", "bad"))
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("ok", "bad")
    mw._dirty_state_generations = {"ok": 1, "bad": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw.message_box_critical = MagicMock()
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("ok", "bad")

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    mw.reset_mechanism_workspaces.assert_called_once_with(["ok"])
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("bad",)
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_parallel_batch_final_scoped_failure_keeps_replay_when_reset_clear_fails(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome},
        set_names={"bad": "Bad Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["ok", "bad"], queue_names=["OK Set", "Bad Set"], total=2, completed_set_ids=["ok"], pending_workspace_reset_set_ids=["ok", "bad"], pending_dirty_reset_generation_by_set_id={"ok": 1, "bad": 1}, explicit_cache_valid_set_ids=("ok", "bad"))
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("ok", "bad")
    mw._dirty_state_generations = {"ok": 1, "bad": 1}
    mw.reset_mechanism_workspaces.return_value = False
    mw.discard_concentration_overlays_for_set_ids.return_value = False
    mw.message_box_critical = MagicMock()
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("ok", "bad")

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    mw.reset_mechanism_workspaces.assert_called_once_with(["ok"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["ok"])
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_target_set_ids == ("ok", "bad")
    assert scheduled == [controller._run_simulation_from_slider]


@pytest.mark.unit
def test_parallel_batch_final_scoped_failure_refreshes_batch_columns_after_overlay_clear(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Mechanism:
        def species_names(self) -> list[str]:
            return ["A", "B"]

    failed_lane_outcome = _timeout_failure_outcome("bad")

    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome},
        set_names={"bad": "Bad Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["ok", "bad"], queue_names=["OK Set", "Bad Set"], total=2, completed_set_ids=["ok"], pending_workspace_reset_set_ids=["ok", "bad"], pending_dirty_reset_generation_by_set_id={"ok": 1, "bad": 1}, explicit_cache_valid_set_ids=("ok", "bad"))
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"
    controller.batch_cache.active_cache_valid_set_ids = ("ok", "bad")
    mw._dirty_state_generations = {"ok": 1, "bad": 1}
    mw.reset_mechanism_workspaces.return_value = False
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._mechanism_helpers.remember_last_mechanism(_Mechanism(), "reaction: A -> B ; k=1", {})
    mw.message_box_critical = MagicMock()

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    mw._sync_batch_species_columns.assert_called_once_with(["A", "B"], preserve_active_cache=True)


@pytest.mark.unit
def test_parallel_batch_scoped_failure_reinvalidates_pending_init_results(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_lane_outcome = _timeout_failure_outcome("bad")

    _install_active_lane_outcomes(
        controller,
        {"bad": failed_lane_outcome},
        set_names={"bad": "Bad Set"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["bad", "ok"], queue_names=["Bad Set", "OK Set"], total=2, pending_init_seed={}, pending_init_rewrite="reaction: A -> B; k=1", pending_init_applied=True)
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad",
        outcome=failed_lane_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=1.0,
    ) is True

    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    assert _batch_policy_context(controller).pending_init_applied is False


@pytest.mark.unit
def test_parallel_batch_all_scoped_failures_replays_deferred_preview(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    failed_a = _timeout_failure_outcome("bad-a")
    failed_b = _timeout_failure_outcome("bad-b", seconds=0.3)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    _install_active_lane_outcomes(
        controller,
        {"bad-a": failed_a, "bad-b": failed_b},
        set_names={"bad-a": "Bad A", "bad-b": "Bad B"},
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck", queue_ids=["bad-a", "bad-b"], queue_names=["Bad A", "Bad B"], total=2, pending_workspace_reset_set_ids=["bad-a", "bad-b"], pending_dirty_reset_generation_by_set_id={"bad-a": 1, "bad-b": 1})
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("bad-a", "bad-b")
    mw.message_box_critical = MagicMock()

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad-a",
        outcome=failed_a,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=1.0,
    ) is True
    assert scheduled == []

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="bad-b",
        outcome=failed_b,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="callback",
        completed_ts=2.0,
    ) is True

    assert mw._status_label.text == "Batch completed with 2 failed set(s)"
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("bad-a", "bad-b")
    assert scheduled == [controller._run_simulation_from_slider]
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.message_box_critical.assert_called_once()


@pytest.mark.unit
def test_consume_parallel_batch_outcome_exception_tears_down_pool_and_next_parallel_run_recreates_lane_pool(
    controller: SimulationController,
):
    submitted: list[tuple[str, dict[str, object]]] = []

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []
            self.ready_lane_count = 999

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            submitted.append((self.label, dict(task)))
            return BatchLaneOutcome(
                lane_id=self.label,
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=str(set_id),
                owner_epoch=1,
                success=True,
                payload={"success": True, "set_id": str(set_id)},
            )

        def close(self, *, kill: bool = False):
            self.close_calls.append(bool(kill))

    created: list[_FakeLanePool] = []

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool(label=f"lane-pool-{len(created) + 1}-w{int(max_workers)}")
        created.append(pool)
        return pool
    failed_outcome = _lane_outcome(
        "sid",
        success=False,
        failure={"kind": "internal_error", "phase": "lane_pool", "message": "boom"},
    )
    _install_active_lane_outcomes(controller, {"sid": failed_outcome}, set_names={"sid": "set1"})
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=1, request_id=2, fast_mode=False, cache_key="ck")
    controller._error_handling_owner.handle_error = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=failed_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="scan",
    )

    assert ok is False
    controller._error_handling_owner.handle_error.assert_called_once()
    assert _batch_policy_context(controller).active is False
    assert not controller.parallel_batch.has_lane_pool()
    assert not controller.parallel_batch.has_active_requests()

    controller.parallel_batch.lane_pool_factory = _factory
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["fresh"], queue_names=["fresh-set"], run_id=3, request_id=11, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={"fresh": _test_simulation_plan_payload(set_id="fresh", set_name="fresh-set", initials={"A": 1.0}, cache_key="ck")})
    controller.ui.batch.batch_initials_for_row = MagicMock(return_value={"A": 1.0})

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert len(created) == 1
    assert controller.parallel_batch.lane_pool_token() == id(created[0])
    assert [label for label, task in submitted if task.get("set_id") == "fresh"] == [created[0].label]

@pytest.mark.unit
def test_poll_parallel_batch_outcomes_consumes_callback_then_scan(controller: SimulationController):
    outcome_a = _lane_outcome("a", {"ok": 1}, run_id=9, request_id=8)
    outcome_b = _lane_outcome("b", {"ok": 2}, run_id=9, request_id=8)

    _install_active_lane_outcomes(controller, {"a": outcome_a, "b": outcome_b}, set_names={"a": "A", "b": "B"})
    controller._batch_parallel.drain_completion_queue()
    controller._batch_parallel.enqueue_completion("a")
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=9, request_id=8, fast_mode=False, cache_key="ck")

    controller._consume_parallel_batch_outcome = MagicMock(return_value=True)
    controller._poll_parallel_batch_completions()

    assert controller._consume_parallel_batch_outcome.call_count == 2
    sources = [kwargs["source"] for _args, kwargs in controller._consume_parallel_batch_outcome.call_args_list]
    assert sources == ["callback", "scan"]

@pytest.mark.unit
def test_poll_parallel_batch_outcomes_callback_then_scan_only_queue_one_replay_handoff(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    outcome_a = _lane_outcome("a", {"ok": 1}, run_id=11, request_id=7)
    outcome_b = _lane_outcome("b", {"ok": 2}, run_id=11, request_id=7)

    _install_active_lane_outcomes(controller, {"a": outcome_a, "b": outcome_b}, set_names={"a": "A", "b": "B"})
    controller._batch_parallel.drain_completion_queue()
    controller._batch_parallel.enqueue_completion("a")

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 8
    controller._active_run_id = 11
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=8,
        epoch=2,
        target_set_ids=("id1",),
    )
    controller.run_state.pending_slider_preview_launch = controller.run_state.pending_slider_preview_launch.__class__(
        active=True,
        request_id=7,
        target_set_ids=("id2",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=11, request_id=7, fast_mode=True, cache_key="ck", preview_owner_epoch=1, keep_lane_pool_alive=True)
    stale_policy_context = _batch_policy_context(controller)
    stale_cache_key = controller.batch_context_owner.completion_flush_context().cache_key

    def _consume_stale_completion(**kwargs):
        _complete_with_callback_identity(
            controller,
            _successful_result_payload(),
            run_id=11,
            fast_mode=True,
            request_id=7,
            owner_epoch=1,
            batch_set="A",
            batch_set_id=str(kwargs["set_id"]),
            cache_key="ck",
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
        )
        if kwargs["source"] == "callback":
            controller.batch_context_owner.serialize_completion_policy_context(stale_policy_context)
            controller.batch_context_owner.record_cache_key(stale_cache_key)
            controller._simulation_running = True
            controller._slider_simulation_active = True
        return True

    controller._consume_parallel_batch_outcome = MagicMock(side_effect=_consume_stale_completion)

    controller._poll_parallel_batch_completions()

    assert controller._consume_parallel_batch_outcome.call_count == 2
    sources = [kwargs["source"] for _args, kwargs in controller._consume_parallel_batch_outcome.call_args_list]
    assert sources == ["callback", "scan"]
    assert scheduled == [controller._run_simulation_from_slider]
    assert getattr(controller.run_state.pending_slider_preview_launch, "handoff_queued") is True


@pytest.mark.unit
def test_poll_parallel_batch_outcomes_catches_unhandled_exceptions_and_shuts_down(mw: _FakeMainWindow, controller: SimulationController):
    outcome_a = _lane_outcome("a", {"ok": 1}, run_id=9, request_id=8)

    _install_active_lane_outcomes(controller, {"a": outcome_a}, set_names={"a": "A"})
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, run_id=9, request_id=8, fast_mode=False, cache_key="ck")

    controller._error_handling_owner.handle_error = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    controller._consume_parallel_batch_outcome = MagicMock(side_effect=RuntimeError("boom"))

    controller._poll_parallel_batch_completions()
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)
    controller._error_handling_owner.handle_error.assert_called_once()


@pytest.mark.unit
def test_poll_parallel_batch_outcomes_surfaces_exception_with_captured_callback_context(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    outcome_a = _lane_outcome("a", {"ok": 1}, run_id=9, request_id=8)

    _install_active_lane_outcomes(controller, {"a": outcome_a}, set_names={"a": "A"})
    controller._active_run_id = 9
    controller._latest_sim_request_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = False
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=9,
        request_id=8,
        fast_mode=False,
        cache_key="ck",
    )

    mw.message_box_critical = MagicMock()
    controller._consume_parallel_batch_outcome = MagicMock(side_effect=RuntimeError("boom"))

    controller._poll_parallel_batch_completions()

    state = controller.batch_context_owner.completion_state()
    assert state is None or state.active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is True
    assert controller.ui.run_ui._stop_btn.isEnabled() is False
    mw.message_box_critical.assert_called_once()


@pytest.mark.unit
def test_poll_parallel_batch_outcomes_surfaces_runtime_snapshot_exception(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    outcome_a = _lane_outcome("a", {"ok": 1}, run_id=9, request_id=8)

    _install_active_lane_outcomes(controller, {"a": outcome_a}, set_names={"a": "A"})
    controller._active_run_id = 9
    controller._latest_sim_request_id = 8
    controller._simulation_running = True
    controller._slider_simulation_active = False
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        run_id=9,
        request_id=8,
        fast_mode=False,
        cache_key="ck",
    )

    mw.message_box_critical = MagicMock()
    monkeypatch.setattr(
        controller._batch_parallel._runtime_session,
        "snapshot",
        MagicMock(side_effect=RuntimeError("snapshot boom")),
    )

    controller._poll_parallel_batch_completions()

    state = controller.batch_context_owner.completion_state()
    assert state is None or state.active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is True
    assert controller.ui.run_ui._stop_btn.isEnabled() is False
    mw.message_box_critical.assert_called_once()

@pytest.mark.unit
def test_flush_pending_slider_updates_for_run_stops_timers_and_finalizes(mw: _FakeMainWindow, controller: SimulationController):
    release_timer = MagicMock()
    release_timer.isActive.return_value = True
    debounce_timer = MagicMock()
    debounce_timer.isActive.return_value = True
    mw._slider_release_commit_timer = release_timer
    mw._variable_update_timer = debounce_timer
    mw._pending_slider_values = {"a": 1}
    mw._finalize_slider_release_commit = MagicMock()
    mw._slider_triggered_simulation = True

    controller._pending_slider_simulation = True
    controller._pending_slider_plot_cache_key = "ck"

    controller._flush_pending_slider_updates_for_run()
    release_timer.stop.assert_called_once()
    debounce_timer.stop.assert_called_once()
    mw._finalize_slider_release_commit.assert_called_once()
    assert controller._pending_slider_simulation is True
    assert mw._slider_triggered_simulation is False

@pytest.mark.unit
def test_flush_pending_slider_updates_for_run_stops_species_timer_and_preserves_replay_until_success(
    mw: _FakeMainWindow, controller: SimulationController
):
    species_timer = MagicMock()
    species_timer.isActive.return_value = True
    mw._species_slider_update_timer = species_timer

    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("id1",)

    controller._flush_pending_slider_updates_for_run(reset_set_ids=("id1",))

    species_timer.stop.assert_called_once()
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("id1",)

@pytest.mark.unit
def test_run_simulation_from_slider_discards_stale_request(mw: _FakeMainWindow, controller: SimulationController):
    controller._pending_slider_sim_request_id = 1
    controller._latest_sim_request_id = 2
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=2,
        epoch=7,
        target_set_ids=("id1",),
    )

    controller._run_simulation_from_slider()
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None

@pytest.mark.unit
def test_run_simulation_from_slider_promotes_reserved_future_request_to_latest(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    controller._latest_sim_request_id = 5
    controller._pending_slider_sim_request_id = 6
    controller._pending_slider_target_set_ids = ("id1",)
    mw._run_btn.setEnabled(True)
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"

    controller._run_simulation_from_slider()

    assert controller._latest_sim_request_id == 6
    controller.run_simulation_internal.assert_called_once()
    assert controller.run_simulation_internal.call_args.kwargs["request_id"] == 6


@pytest.mark.unit
def test_run_simulation_from_slider_allocates_fresh_request_after_clearing_pending_replay_owner(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=2,
        target_set_ids=("id1",),
    )
    controller._pending_slider_sim_request_id = 5
    controller._latest_sim_request_id = 5

    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id2",),
        request_id=None,
        preserve_existing_request=False,
    )
    mw._batch_store.row_count.return_value = 2
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]

    controller._run_simulation_from_slider()

    assert controller.run_state.preview_ownership.request_id == 5
    assert controller.run_state.preview_ownership.target_set_ids == ("id1",)
    controller.run_simulation_internal.assert_called_once()
    assert controller.run_simulation_internal.call_args.kwargs["request_id"] == 6


@pytest.mark.unit
def test_run_simulation_from_slider_noops_for_stray_callback_after_consumed_replay(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller.run_simulation_internal = MagicMock()
    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id2",),
        request_id=6,
    )
    mw._batch_store.row_count.return_value = 2
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 1
    controller._simulation_running = False
    controller._slider_simulation_active = False
    seed_batch_context(controller.batch_context_owner)

    controller._run_simulation_from_slider()

    assert controller.run_simulation_internal.call_count == 1


@pytest.mark.unit
def test_public_run_simulation_from_slider_noops_when_no_deferred_replay_intent(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller.run_simulation_internal = MagicMock()

    controller.launch_pending_slider_preview_replay()

    assert controller.run_simulation_internal.call_count == 0
    pending_launch = _pending_slider_preview_launch(controller)
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()


@pytest.mark.unit
def test_run_simulation_from_slider_keeps_current_preview_owner_until_new_request_actually_starts(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=2,
        target_set_ids=("id1",),
    )
    mw._batch_store.row_count.return_value = 2
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._last_slider_change_name = "k1"

    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id2",),
        request_id=6,
    )

    assert controller.run_state.preview_ownership.request_id == 5
    assert controller.run_state.preview_ownership.target_set_ids == ("id1",)
    assert controller._pending_slider_sim_request_id == 6
    assert controller._pending_slider_target_set_ids == ("id2",)

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["request_id"] == 6
    assert kwargs["batch_rows"] == [1]
    assert controller.run_state.preview_ownership.request_id == 5


@pytest.mark.unit
def test_launch_pending_slider_preview_replay_consumes_explicit_submitted_intent(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller.run_simulation_internal = MagicMock()
    mw._batch_rows_for_scope.return_value = [2]
    mw._batch_store.row_count.return_value = 3
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2", 2: "id3"}[int(row)]
    mw._slider_gesture_target_set_ids_snapshot = ["id1"]
    mw._last_slider_change_name = "k1"

    controller.submit_slider_preview_replay_intent(
        SliderReplayIntent(target_set_ids=("id2",), source="reset"),
        preserve_existing_request=False,
    )

    controller.launch_pending_slider_preview_replay()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["batch_rows"] == [1]


@pytest.mark.unit
def test_queue_pending_slider_preview_replay_preserve_existing_rebinds_pending_request_when_owner_cleared(
    controller: SimulationController,
):
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=None,
        epoch=4,
        target_set_ids=(),
    )
    controller._pending_slider_sim_request_id = 8

    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id2",),
        request_id=None,
        preserve_existing_request=True,
    )

    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 8
    assert controller.run_state.preview_ownership.request_id is None
    assert controller.run_state.preview_ownership.target_set_ids == ()


@pytest.mark.unit
def test_queue_pending_slider_preview_replay_preserve_existing_allocates_fresh_deferred_request_without_borrowing_owner(
    controller: SimulationController,
):
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=11,
        epoch=4,
        target_set_ids=("id1",),
    )
    controller._latest_sim_request_id = 11

    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id2",),
        request_id=None,
        preserve_existing_request=True,
    )

    assert controller.run_state.preview_ownership.request_id == 11
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 12
    assert controller._pending_slider_target_set_ids == ("id2",)


@pytest.mark.unit
def test_pending_slider_preview_launch_state_preserves_single_string_target_identity(
    controller: SimulationController,
):
    controller._pending_slider_target_set_ids = "id2"

    assert controller._pending_slider_target_set_ids == ("id2",)


@pytest.mark.unit
def test_pending_slider_preview_launch_state_preserves_string_false_normalization(
    controller: SimulationController,
):
    controller._pending_slider_simulation = "false"
    assert controller._pending_slider_simulation is False


@pytest.mark.unit
def test_clear_pending_slider_preview_replay_preserves_active_preview_ownership(
    controller: SimulationController,
):
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=8,
        epoch=4,
        target_set_ids=("id2",),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 8
    controller._pending_slider_target_set_ids = ("id2",)

    controller.clear_pending_slider_preview_replay(clear_plot_updates=False)

    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None
    assert controller._pending_slider_target_set_ids == ()
    assert controller.run_state.preview_ownership.request_id == 8

@pytest.mark.unit
def test_run_simulation_from_slider_uses_snapshotted_target_rows(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_target_set_ids = ("id1", "id2")
    controller._latest_sim_request_id = int(rid)
    mw._batch_rows_for_scope.return_value = [2]
    mw._batch_store.row_count.return_value = 3
    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id3"]
    controller._effective_batch_worker_count = MagicMock(return_value=1)

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["request_id"] == int(rid)
    assert kwargs["batch_rows"] == [0, 1]


@pytest.mark.unit
def test_selected_run_runtime_snapshot_uses_batch_runtime_for_parallel_run(controller: SimulationController):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    batch_snapshot = RuntimeReadinessSnapshot(
        mode="batch",
        status="warming",
        ready=False,
        generation=4,
        required=True,
        controls_ready=False,
        polling=True,
    )
    serial_snapshot = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="ready",
        ready=True,
        generation=5,
        required=True,
        controls_ready=True,
        polling=False,
    )
    controller._interactive_runtime_rows = MagicMock(return_value=[0, 1])
    controller._effective_batch_worker_count = MagicMock(return_value=2)
    controller._parallel_batch_runtime_snapshot = MagicMock(return_value=batch_snapshot)
    controller._interactive_simulation_runtime_snapshot = MagicMock(return_value=serial_snapshot)

    assert controller.selected_run_runtime_snapshot() is batch_snapshot
    controller._parallel_batch_runtime_snapshot.assert_called_once()
    controller._interactive_simulation_runtime_snapshot.assert_not_called()


@pytest.mark.unit
def test_slider_preview_runtime_snapshot_uses_batch_runtime_for_parallel_preview(controller: SimulationController):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    batch_snapshot = RuntimeReadinessSnapshot(
        mode="batch",
        status="warming",
        ready=False,
        generation=4,
        required=True,
        controls_ready=False,
        polling=True,
    )
    serial_snapshot = RuntimeReadinessSnapshot(
        mode="preview",
        status="ready",
        ready=True,
        generation=5,
        required=True,
        controls_ready=True,
        polling=False,
    )
    controller._interactive_runtime_rows = MagicMock(return_value=[0, 1])
    controller._effective_batch_worker_count = MagicMock(return_value=2)
    controller._parallel_batch_runtime_snapshot = MagicMock(return_value=batch_snapshot)
    controller._interactive_simulation_runtime_snapshot = MagicMock(return_value=serial_snapshot)

    assert controller.slider_preview_runtime_snapshot() is batch_snapshot
    controller._parallel_batch_runtime_snapshot.assert_called_once()
    controller._interactive_simulation_runtime_snapshot.assert_not_called()


@pytest.mark.unit
def test_current_interactive_runtime_warm_skips_batch_for_single_row_serial_workflows(
    controller: SimulationController,
):
    controller._interactive_runtime_rows = MagicMock(return_value=[0])
    controller._effective_batch_worker_count = MagicMock(return_value=1)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._parallel_batch_runtime_readiness_owner.ensure = MagicMock()

    controller.ensure_current_interactive_simulation_runtimes_available(wait=False)

    assert controller._ensure_interactive_simulation_runtime_available_for_mode.call_args_list == [
        call(fast_mode=False, wait=False),
        call(fast_mode=True, wait=False),
    ]
    controller._parallel_batch_runtime_readiness_owner.ensure.assert_not_called()


@pytest.mark.unit
def test_current_interactive_runtime_warm_uses_required_lanes_for_parallel_run_and_preview(
    controller: SimulationController,
):
    controller._interactive_runtime_rows = MagicMock(return_value=[0, 1, 2])
    controller._effective_batch_worker_count = MagicMock(return_value=2)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._parallel_batch_runtime_readiness_owner.ensure = MagicMock()

    controller.ensure_current_interactive_simulation_runtimes_available(wait=False)

    controller._parallel_batch_runtime_readiness_owner.ensure.assert_called_once_with(
        wait=False,
        required_lanes=2,
    )


@pytest.mark.unit
def test_run_simulation_from_slider_ignores_stale_mechanism_snapshot_for_species_preview(
    mw: _FakeMainWindow, controller: SimulationController
):
    controller.run_simulation_internal = MagicMock()
    rid = controller._next_sim_request_id()
    controller._pending_slider_sim_request_id = int(rid)
    controller._pending_slider_target_set_ids = ("id3",)
    controller._latest_sim_request_id = int(rid)
    mw._batch_rows_for_scope.return_value = [2]
    mw._batch_store.row_count.return_value = 3
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2", 2: "id3"}[int(row)]
    mw._slider_gesture_target_set_ids_snapshot = ["id1", "id2"]
    mw._last_slider_change_name = "init:A"

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["batch_rows"] == [2]

@pytest.mark.unit
def test_run_simulation_from_slider_preflight_abort_clears_slider_triggered_flag(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return ""

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )

    rid = controller._next_sim_request_id()
    controller._latest_sim_request_id = int(rid)
    controller._pending_slider_sim_request_id = int(rid)
    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation_from_slider()

    assert mw._slider_triggered_simulation is False

@pytest.mark.unit
def test_run_simulation_from_slider_defers_when_full_run_in_progress(mw: _FakeMainWindow, controller: SimulationController):
    mw._run_btn = _FakeButton(False)
    controller._simulation_worker = None
    controller._simulation_running = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False)
    controller.queue_pending_slider_preview_replay(
        target_set_ids=("id1",),
        request_id=7,
    )
    controller._latest_sim_request_id = 0

    controller.launch_pending_slider_preview_replay()
    assert controller._pending_slider_simulation is True

@pytest.mark.unit
def test_retained_worker_finish_preserves_valid_deferred_replay_without_current_preview_owner(
    monkeypatch, controller: SimulationController
):
    worker = _FakeWorker(running=False, wait_returns=True)
    controller._retained_simulation_workers = [worker]
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 9
    controller._pending_slider_target_set_ids = ("id2",)
    controller.run_state.preview_ownership = PreviewOwnershipState()

    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._on_retained_simulation_worker_finished(worker)

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_sim_request_id == 9
    assert controller._pending_slider_target_set_ids == ("id2",)


@pytest.mark.unit
def test_cancel_active_run_for_restart_resets_ui_and_shuts_down(mw: _FakeMainWindow, controller: SimulationController):
    seed_batch_context(controller.batch_context_owner, active=True)
    controller._shutdown_batch_lane_pool = MagicMock()
    worker = _FakeWorker(running=True, wait_returns=True)
    controller._simulation_worker = worker

    controller._cancel_active_run_for_restart()
    assert _batch_policy_context(controller).active is False
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False


@pytest.mark.unit
def test_stop_simulation_does_not_close_active_contained_owner_outside_worker_thread(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._simulation_running = True
    seed_batch_context(controller.batch_context_owner, active=True, fast_mode=False)
    controller._shutdown_batch_lane_pool = MagicMock()
    owner = _FakeContainedOwner()
    controller._ordinary_simulation_owner = owner
    worker = _FakeWorker(running=True, wait_returns=False)
    controller._simulation_worker = worker

    controller._stop_simulation()

    assert worker._cancelled is True
    assert owner.close_calls == []
    assert controller._ordinary_simulation_owner is None
    assert mw._status_label.text == "Cancelling simulation..."

@pytest.mark.unit
def test_run_simulation_blocks_restart_while_retained_worker_is_still_running(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_rows_for_scope.return_value = [0]
    controller.run_simulation_internal = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True)
    controller._shutdown_batch_lane_pool = MagicMock()
    worker = make_stubborn_worker(_FakeWorker)
    controller._simulation_worker = worker
    controller._simulation_running = True
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)

    controller.run_simulation()
    controller.run_simulation()

    assert controller.run_simulation_internal.call_count == 0
    assert controller._simulation_worker is None
    assert controller._simulation_running is False
    assert controller.has_running_owned_simulation_workers() is True
    assert controller._retained_simulation_workers == [worker]
    assert mw._status_label.text == "Cancelling previous simulation..."
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

@pytest.mark.unit
def test_run_simulation_reuses_parallel_lane_pool_for_explicit_multi_set_runs(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_rows_for_scope.return_value = [0, 1]
    _install_ready_batch_lane_pool(
        controller,
        _RecordingLanePool([]),
        max_lanes=2,
    )
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    controller.run_simulation_internal.assert_called_once()
    _args, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is False
    assert kwargs["batch_rows"] == [0, 1]
    assert kwargs["reuse_parallel_lane_pool"] is True


@pytest.mark.unit
def test_batch_runtime_readiness_wait_false_returns_without_warming_on_caller_thread(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
        ParallelBatchRuntimeReadinessOwner,
    )

    class _BatchParallel:
        max_parallel_workers = 2

        def __init__(self) -> None:
            self.ensure_calls: list[dict[str, object]] = []

        def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool):
            self.ensure_calls.append(
                {
                    "max_lanes": int(max_lanes),
                    "wait": bool(wait),
                    "thread_id": threading.get_ident(),
                }
            )
            return object()

        def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
            return bool(self.ensure_calls)

        def active_request_count(self) -> int:
            return 0

        def shutdown(self, **_kwargs) -> None:
            return None

    fake_parallel = _BatchParallel()
    controller._batch_parallel_adapter = fake_parallel
    controller._parallel_batch_runtime_readiness_owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=fake_parallel,
        capacity_getter=lambda: controller.batch_runtime_lane_budget,
    )
    caller_thread_id = threading.get_ident()

    controller.ensure_parallel_batch_runtime_ready(wait=False)

    assert not [
        call
        for call in fake_parallel.ensure_calls
        if call["thread_id"] == caller_thread_id
    ]


@pytest.mark.unit
def test_interactive_batch_runtime_capacity_uses_configured_lane_budget(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
        ParallelBatchRuntimeReadinessOwner,
    )

    class _BatchParallel:
        max_parallel_workers = 16

        def __init__(self) -> None:
            self.ensure_calls: list[dict[str, object]] = []

        def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool):
            self.ensure_calls.append({"max_lanes": int(max_lanes), "wait": bool(wait)})
            return object()

        def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
            return False

        def active_request_count(self) -> int:
            return 0

        def shutdown(self, **_kwargs) -> None:
            return None

    fake_parallel = _BatchParallel()
    controller._batch_parallel_adapter = fake_parallel
    controller._parallel_batch_runtime_readiness_owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=fake_parallel,
        capacity_getter=lambda: controller.batch_runtime_lane_budget,
    )
    controller.batch_runtime_lane_budget = 6

    controller.ensure_parallel_batch_runtime_ready(wait=True)

    assert fake_parallel.ensure_calls == [{"max_lanes": 6, "wait": True}]

@pytest.mark.unit
def test_run_auto_locks_editor(mw: _FakeMainWindow, controller: SimulationController):
    mw._batch_rows_for_scope.return_value = [0]
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._auto_lock_for_run_calls == 1
    controller.run_simulation_internal.assert_called_once()

@pytest.mark.unit
def test_run_aborts_if_mechanism_invalid_while_unlocked(mw: _FakeMainWindow, controller: SimulationController):
    mw._batch_rows_for_scope.return_value = [0]
    mw._auto_lock_for_run_result = False
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    assert mw._auto_lock_for_run_calls == 1
    controller.run_simulation_internal.assert_not_called()
    assert mw._status_label.text == "Cannot run: mechanism has errors. Fix and try again."

@pytest.mark.unit
def test_start_parallel_batch_simulations_reports_unready_when_lane_pool_is_missing(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    def _raise_runtime_snapshot(_self):
        raise RuntimeError("no lane pool")

    monkeypatch.setattr(controller._batch_parallel.__class__, "runtime_snapshot", _raise_runtime_snapshot)
    controller._start_next_batch_simulation = MagicMock()
    controller._queue_run_after_runtime_ready = MagicMock()
    controller._simulation_running = True
    mw._stop_btn.setEnabled(True)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], run_id=1, effective_workers=2)

    controller._start_parallel_batch_simulations()
    state = controller.batch_context_owner.active_batch_state()
    assert state is not None
    assert state.parallel is True
    assert state.active is False
    assert state.runtime_waiting is True
    assert controller._simulation_running is False
    controller._start_next_batch_simulation.assert_not_called()
    controller._queue_run_after_runtime_ready.assert_not_called()
    assert mw._stop_btn.isEnabled() is False
    assert mw._status_label.text == "Batch runtime readiness check failed: no lane pool"


@pytest.mark.unit
def test_start_parallel_batch_simulations_requeues_unready_slider_preview_as_preview(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    scheduled: list[tuple[int, object]] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda delay_ms, fn: scheduled.append((int(delay_ms), fn)))
    controller._queue_run_after_runtime_ready = MagicMock()
    controller._ensure_parallel_batch_runtime_ready = MagicMock()
    controller._simulation_running = True
    controller._slider_simulation_active = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], run_id=1, request_id=4, effective_workers=2, fast_mode=True)

    controller._start_parallel_batch_simulations()

    controller._queue_run_after_runtime_ready.assert_not_called()
    controller._ensure_parallel_batch_runtime_ready.assert_called_once_with(
        wait=False,
        required_lanes=2,
    )
    pending = controller._pending_slider_preview_launch
    assert pending.active is True
    assert pending.request_id == 4
    assert pending.target_set_ids == ("id1", "id2")
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert scheduled == [(50, controller._run_simulation_from_slider)]


@pytest.mark.unit
def test_run_simulation_queues_unready_parallel_batch_runtime_without_fake_running(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    controller.parallel_batch.max_parallel_workers = 2
    controller._ensure_parallel_batch_runtime_ready = MagicMock()
    controller.run_simulation_internal = MagicMock()

    controller.run_simulation()

    controller.run_simulation_internal.assert_not_called()
    controller._ensure_parallel_batch_runtime_ready.assert_called()
    assert controller._pending_run_after_runtime_ready.active is True
    assert controller._simulation_running is False
    assert mw._stop_btn.isEnabled() is False
    assert "runtime" in mw._status_label.text.lower()

@pytest.mark.unit
def test_start_parallel_batch_simulations_maps_submit_failure_to_affected_set(
    mw: _FakeMainWindow,
    controller: SimulationController,
    monkeypatch,
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=2)
    original_submit_task = controller._batch_parallel.submit_task

    def _submit_task(_adapter, _task, *, set_id: str, set_name: str, **kwargs):
        if str(set_id) == "bad":
            raise RuntimeError("submit failed")
        return original_submit_task(
            _task,
            set_id=set_id,
            set_name=set_name,
            **kwargs,
        )

    monkeypatch.setattr(type(controller._batch_parallel), "submit_task", _submit_task)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["bad", "ok"], queue_names=["Bad Set", "OK Set"], run_id=1, request_id=2, effective_workers=2, cache_key="ck", full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, pending_init_seed={}, pending_init_applied=True, explicit_cache_valid_set_ids=("bad", "ok"), simulation_plan_by_set_id={"bad": _test_simulation_plan_payload(set_id="bad", set_name="Bad Set", initials={"A": 1.0}), "ok": _test_simulation_plan_payload(set_id="ok", set_name="OK Set", initials={"A": 1.0})})
    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    controller.batch_cache.active_cache_key = "ck"

    controller._start_parallel_batch_simulations()

    assert _batch_policy_context(controller).active is True
    summary = controller.batch_context_owner.completion_summary()
    completion_state = controller.batch_context_owner.completion_state()
    assert summary.failed_set_ids == ("bad",)
    assert completion_state is not None
    assert "bad" in completion_state.completed_set_ids
    assert controller._batch_parallel.active_request_count() == 1
    assert "completion_policy_context" not in controller._batch_parallel.active_request_metadata("ok")
    controller.ui.dialogs.message_box_critical = MagicMock()
    controller._batch_parallel.join_active_requests(timeout_s=2.0)
    controller._poll_parallel_batch_completions()
    assert controller.batch_cache.active_cache_valid_set_ids == ("ok",)
    assert controller.batch_cache.active_cache_invalidated_set_ids == ("bad",)
    assert submitted


@pytest.mark.unit
def test_start_parallel_batch_submit_failure_preserves_captured_callback_identity(
    mw: _FakeMainWindow,
    controller: SimulationController,
    monkeypatch,
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=1)
    captured_identities: list[object] = []

    def _raise_submit(_adapter, _task, *, set_id: str, set_name: str, **kwargs):
        _ = _adapter, _task, set_id, set_name, kwargs
        captured_identities.append(kwargs["callback_identity"])
        seed_batch_context(
            controller.batch_context_owner,
            active=True,
            parallel=True,
            rows=[0],
            queue_ids=["current-id"],
            queue_names=["current-set"],
            run_id=99,
            request_id=98,
            effective_workers=1,
            preview_owner_epoch=97,
            cache_key="current-cache",
            full_dsl="reaction: A -> C; k=3",
            solver_config={"solver": "Radau"},
            t_end=3.0,
            fast_mode=True,
            pending_init_seed={},
            pending_init_applied=True,
            simulation_identity_by_set_id={"current-id": {"fingerprint": "current-fp"}},
            preview_batch_cache_token_by_set_id={"current-id": "current-preview-token"},
        )
        raise RuntimeError("submit failed")

    monkeypatch.setattr(type(controller._batch_parallel), "submit_task", _raise_submit)
    monkeypatch.setattr(controller, "_try_handle_scoped_batch_failure", lambda **_kwargs: False)
    controller._on_simulation_error = MagicMock()
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        rows=[0],
        queue_ids=["id1"],
        queue_names=["set1"],
        run_id=1,
        request_id=2,
        effective_workers=1,
        preview_owner_epoch=7,
        cache_key="batch-cache",
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=10.0,
        fast_mode=True,
        pending_init_seed={},
        pending_init_applied=True,
        simulation_identity_by_set_id={"id1": {"fingerprint": "fp-1"}},
        simulation_plan_by_set_id={"id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}, fast_mode=True, cache_key="batch-cache", simulation_identity={"fingerprint": "fp-1"})},
        preview_batch_cache_token_by_set_id={"id1": "preview-token"},
    )

    controller._start_parallel_batch_simulations()

    controller._on_simulation_error.assert_called_once()
    callback_identity = controller._on_simulation_error.call_args.kwargs.get("callback_identity")
    assert captured_identities
    assert callback_identity is captured_identities[0]
    assert callback_identity is not None
    assert callback_identity.batch_set_id == "id1"
    assert callback_identity.owner_epoch == 7
    assert callback_identity.cache_key == "batch-cache"
    assert callback_identity.simulation_identity == {"fingerprint": "fp-1"}
    assert callback_identity.preview_batch_cache_token == "preview-token"


@pytest.mark.unit
def test_start_contained_serial_batch_worker_passes_submitted_callback_identity_to_signal_wiring(
    controller: SimulationController,
    monkeypatch,
):
    plan_payload = _test_simulation_plan_payload(
        set_id="id1",
        set_name="set1",
        cache_key="batch-cache",
        simulation_identity={"fingerprint": "submitted-fp"},
    )
    worker = MagicMock()
    monkeypatch.setattr(
        controller._contained_serial_worker_launch_owner,
        "create_worker",
        MagicMock(return_value=worker),
    )
    controller._connect_simulation_worker_application_signals = MagicMock()
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        queue_ids=["current-id"],
        queue_names=["current-set"],
        run_id=99,
        request_id=98,
        preview_owner_epoch=97,
        cache_key="current-cache",
        simulation_identity_by_set_id={"current-id": {"fingerprint": "current-fp"}},
    )

    started = controller._start_contained_serial_batch_worker(
        plan_payload=plan_payload,
        run_id=1,
        request_id=2,
        fast_mode=False,
        owner_epoch=None,
        set_name="set1",
        set_id="id1",
        cache_key="batch-cache",
        context={"active": True, "run_id": 1, "request_id": 2},
        include_mechanism_in_result_payload=True,
    )

    assert started is True
    callback_identity = controller._connect_simulation_worker_application_signals.call_args.kwargs[
        "callback_identity"
    ]
    assert callback_identity.batch_set_id == "id1"
    assert callback_identity.cache_key == "batch-cache"
    assert callback_identity.simulation_identity == {"fingerprint": "submitted-fp"}
    worker.start.assert_called_once()


@pytest.mark.unit
def test_start_contained_serial_batch_worker_rejects_missing_callback_context(
    controller: SimulationController,
    monkeypatch,
):
    plan_payload = _test_simulation_plan_payload(
        set_id="id1",
        set_name="set1",
        cache_key="batch-cache",
        simulation_identity={"fingerprint": "submitted-fp"},
    )
    controller._contained_serial_worker_launch_owner.create_worker = MagicMock()
    controller._connect_simulation_worker_application_signals = MagicMock()

    started = controller._start_contained_serial_batch_worker(
        plan_payload=plan_payload,
        run_id=1,
        request_id=2,
        fast_mode=False,
        owner_epoch=None,
        set_name="set1",
        set_id="id1",
        cache_key="batch-cache",
        context=None,
        include_mechanism_in_result_payload=True,
    )

    assert started is False
    assert "callback context" in controller._last_nonfatal_exception
    controller._contained_serial_worker_launch_owner.create_worker.assert_not_called()
    controller._connect_simulation_worker_application_signals.assert_not_called()


@pytest.mark.unit
def test_start_parallel_batch_identity_capture_failure_surfaces_original_failure(
    mw: _FakeMainWindow,
    controller: SimulationController,
    monkeypatch,
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=1)

    def _raise_capture(**_kwargs):
        raise RuntimeError("identity capture failed")

    monkeypatch.setattr(controller, "_capture_simulation_callback_identity", _raise_capture)
    monkeypatch.setattr(
        SimulationCallbackIdentity,
        "capture",
        classmethod(
            lambda cls, **_kwargs: (_ for _ in ()).throw(
                AssertionError("capture-failure fallback must not re-enter SimulationCallbackIdentity.capture")
            )
        ),
    )
    monkeypatch.setattr(controller, "_try_handle_scoped_batch_failure", lambda **_kwargs: False)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        rows=[0],
        queue_ids=["id1"],
        queue_names=["set1"],
        run_id=1,
        request_id=2,
        effective_workers=1,
        preview_owner_epoch=7,
        cache_key="batch-cache",
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=10.0,
        fast_mode=True,
        pending_init_seed={},
        pending_init_applied=True,
        simulation_plan_by_set_id={"id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}, fast_mode=True, cache_key="batch-cache")},
    )

    controller._start_parallel_batch_simulations()

    assert "identity capture failed" in controller._last_nonfatal_exception
    assert controller._batch_parallel.has_active_requests() is False
    assert _batch_policy_context(controller).active is False
    assert mw._status_label.text == "Batch simulation failed"


@pytest.mark.unit
def test_start_parallel_batch_simulations_records_preview_owner_epoch_in_submitted_lane_metadata(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=1)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        rows=[0],
        queue_ids=["id1"],
        queue_names=["set1"],
        run_id=1,
        request_id=2,
        effective_workers=1,
        preview_owner_epoch=7,
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=10.0,
        fast_mode=True,
        pending_init_seed={},
        pending_init_applied=True,
        simulation_plan_by_set_id={"id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}, fast_mode=True, cache_key="cache")},
    )

    controller._start_parallel_batch_simulations()

    metadata = controller._batch_parallel.active_request_metadata("id1")
    assert metadata["preview_owner_epoch"] == 7
    assert metadata["callback_identity"].owner_epoch == 7


@pytest.mark.unit
def test_start_parallel_batch_simulations_reuses_one_callback_context_for_submitted_sets(
    mw: _FakeMainWindow,
    controller: SimulationController,
    monkeypatch,
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=2)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        rows=[0, 1],
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        run_id=1,
        request_id=2,
        effective_workers=2,
        preview_owner_epoch=7,
        cache_key="batch-cache",
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=10.0,
        fast_mode=False,
        pending_init_seed={},
        pending_init_applied=True,
        prepared_by_set_id={
            "id1": {"prepared": True},
            "id2": {"prepared": True},
        },
        mechanism_text_by_set_id={
            "id1": "reaction: A -> B; k=1",
            "id2": "reaction: A -> C; k=2",
        },
        mechanism_signature_by_set_id={"id1": "sig-id1", "id2": "sig-id2"},
        simulation_identity_by_set_id={
            "id1": {"fingerprint": "fp-1"},
            "id2": {"fingerprint": "fp-2"},
        },
        simulation_plan_by_set_id={
            "id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", mechanism_text="reaction: A -> B; k=1", initials={"A": 1.0}, cache_key="batch-cache", simulation_identity={"fingerprint": "fp-1"}),
            "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", mechanism_text="reaction: A -> C; k=2", initials={"A": 1.0}, cache_key="batch-cache", simulation_identity={"fingerprint": "fp-2"}),
        },
        preview_batch_cache_token_by_set_id={"id1": "preview-1", "id2": "preview-2"},
    )
    original_snapshot = controller.batch_context_owner.callback_context_snapshot
    callback_context_calls = 0

    def _record_callback_context_snapshot(context=None):
        nonlocal callback_context_calls
        callback_context_calls += 1
        return original_snapshot(context)

    monkeypatch.setattr(controller.batch_context_owner, "callback_context_snapshot", _record_callback_context_snapshot)
    monkeypatch.setattr(
        controller.batch_context_owner,
        "current_context_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("full context snapshot must not be captured for callbacks")),
        raising=False,
    )

    controller._start_parallel_batch_simulations()

    metadata_id1 = controller._batch_parallel.active_request_metadata("id1")
    metadata_id2 = controller._batch_parallel.active_request_metadata("id2")
    identities = [metadata_id1["callback_identity"], metadata_id2["callback_identity"]]
    assert callback_context_calls == 1
    assert identities[0].callback_context is identities[1].callback_context
    assert not hasattr(identities[0], "context_snapshot")
    assert not hasattr(identities[1], "context_snapshot")
    assert identities[0].batch_set_id == "id1"
    assert identities[1].batch_set_id == "id2"


@pytest.mark.unit
def test_parallel_batch_callback_identity_uses_submitted_task_plan_identity(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=1)
    solver_config = {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15}
    plan_identity = SimulationIdentity.build(
        schema_id="parallel-plan-schema",
        param_fingerprint="parallel-plan-fingerprint",
        solver_config=solver_config,
        t_end=10.0,
    ).to_payload()
    stale_map_identity = SimulationIdentity.build(
        schema_id="parallel-stale-map-schema",
        param_fingerprint="parallel-stale-map-fingerprint",
        solver_config=solver_config,
        t_end=10.0,
    ).to_payload()
    plan_payload = SimulationPlan.from_execution_request(
        {
            "prepared_payload": {"prepared": True},
            "initials": {"A": 1.0},
            "t_span": (0.0, 10.0),
            "solver_config": solver_config,
            "mechanism_text": "reaction: A -> B; k=1",
            "simulation_identity": plan_identity,
        },
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
        cache_identity_payload={"cache_key": "batch-cache", "simulation_identity": plan_identity},
        cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": ["id1"]},
    ).to_payload()
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        rows=[0],
        queue_ids=["id1"],
        queue_names=["set1"],
        run_id=1,
        request_id=2,
        effective_workers=1,
        cache_key="batch-cache",
        full_dsl="reaction: A -> B; k=1",
        solver_config=solver_config,
        t_end=10.0,
        fast_mode=False,
        pending_init_seed={},
        pending_init_applied=True,
        simulation_plan_by_set_id={"id1": plan_payload},
        prepared_by_set_id={"id1": {"prepared": True}},
        mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"},
        mechanism_signature_by_set_id={"id1": "sig-id1"},
        simulation_identity_by_set_id={"id1": stale_map_identity},
    )

    controller._start_parallel_batch_simulations()

    metadata = controller._batch_parallel.active_request_metadata("id1")
    identity = metadata["callback_identity"]
    assert identity.simulation_identity == plan_identity
    assert identity.launch_provenance["temperature_K"] == pytest.approx(298.15)
    assert identity.launch_provenance["simulation_time"] == pytest.approx(10.0)
    assert identity.launch_provenance["num_points_requested"] == 10


@pytest.mark.gui
def test_main_window_parallel_dispatch_identity_publishes_per_set_callback_cache(main_window, monkeypatch):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    controller = main_window.simulation_controller
    run_id = 13
    request_id = 17
    cache_key = "main-window-dispatch-cache"
    solver_config = {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "temperature_K": 298.15}
    preview_ownership = controller._set_preview_ownership(
        request_id=request_id,
        target_set_ids=("id1", "id2"),
    )
    controller.run_state.latest_sim_request_id = request_id
    controller.run_state.active_run_id = run_id
    submitted: list[dict[str, object]] = []
    lane_pool = _RecordingLanePool(submitted)
    _install_ready_batch_lane_pool(controller, lane_pool, max_lanes=2)
    monkeypatch.setattr(
        controller._batch_dispatch_materialization_owner,
        "materialize_initials",
        lambda **_kwargs: {"A": 1.0},
    )
    identity_by_set_id = {
        set_id: SimulationIdentity.build(
            schema_id="main-window-dispatch-schema",
            param_fingerprint=set_id,
            solver_config=solver_config,
            t_end=10.0,
            preview_batch_cache_token=f"preview-token-{set_id}",
            execution_flags=("fast_mode",),
        ).to_payload()
        for set_id in ("id1", "id2")
    }
    plan_by_set_id = {
        set_id: SimulationPlan.from_execution_request(
            {
                "prepared_payload": {"prepared": True, "set_id": set_id},
                "initials": {"A": 1.0},
                "t_span": (0.0, 10.0),
                "solver_config": solver_config,
                "mechanism_text": f"reaction: A -> B ; k={idx}",
                "simulation_identity": identity_by_set_id[set_id],
            },
            execution_mode="preview",
            algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
            cache_identity_payload={"cache_key": cache_key, "simulation_identity": identity_by_set_id[set_id]},
            cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": ["id1", "id2"]},
        ).to_payload()
        for idx, set_id in enumerate(("id1", "id2"), start=1)
    }
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=True,
        keep_lane_pool_alive=True,
        run_id=run_id,
        request_id=request_id,
        cache_key=cache_key,
        rows=[0, 1],
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        primary_set_id="id1",
        total=2,
        completed_set_ids=[],
        effective_workers=2,
        preview_owner_epoch=preview_ownership.epoch,
        preview_scope_set_ids=("id1", "id2"),
        full_dsl="reaction: A -> B ; k=1",
        solver_config=solver_config,
        t_end=10.0,
        simulation_plan_by_set_id=plan_by_set_id,
        prepared_by_set_id={
            "id1": {"prepared": True, "set_id": "id1"},
            "id2": {"prepared": True, "set_id": "id2"},
        },
        mechanism_text_by_set_id={
            "id1": "reaction: A -> B ; k=1",
            "id2": "reaction: A -> B ; k=2",
        },
        mechanism_signature_by_set_id={"id1": "sig-id1", "id2": "sig-id2"},
        simulation_identity_by_set_id={
            "id1": {"schema_id": "stale-context-id1"},
            "id2": {"schema_id": "stale-context-id2"},
        },
        preview_batch_cache_token_by_set_id={
            "id1": "preview-token-id1",
            "id2": "preview-token-id2",
        },
        pending_init_seed={},
        pending_init_applied=True,
    )

    controller.start_parallel_batch_simulations()

    metadata_by_set_id = {
        set_id: controller.parallel_batch.active_request_metadata(set_id)
        for set_id in ("id1", "id2")
    }
    callback_identities = {
        set_id: metadata["callback_identity"]
        for set_id, metadata in metadata_by_set_id.items()
    }
    assert callback_identities["id1"].callback_context is callback_identities["id2"].callback_context
    for set_id, identity in callback_identities.items():
        assert identity.simulation_identity == identity_by_set_id[set_id]
        assert identity.preview_batch_cache_token == f"preview-token-{set_id}"

    for set_id, identity in callback_identities.items():
        controller.on_simulation_complete(
            _successful_result_payload(),
            callback_identity=identity,
        )

    for set_id in ("id1", "id2"):
        entry_result = controller.batch_cache.entry_for_set(
            cache_key=cache_key,
            set_id=set_id,
            is_preview=True,
        )
        assert entry_result.state == "valid"
        assert entry_result.entry is not None
        assert entry_result.entry["simulation_identity"] == identity_by_set_id[set_id]
        assert entry_result.entry["preview_batch_cache_token"] == f"preview-token-{set_id}"


@pytest.mark.unit
def test_start_parallel_batch_simulations_resizes_retained_pool_before_submit(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    old_submitted: list[dict[str, object]] = []
    new_submitted: list[dict[str, object]] = []
    old_pool = _RecordingLanePool(old_submitted)
    new_pool = _RecordingLanePool(new_submitted)

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: old_pool
    controller._batch_parallel.ensure_lane_pool(max_lanes=2)
    controller._batch_parallel.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: new_pool
    controller._batch_parallel.ensure_lane_pool(max_lanes=4)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1, 2, 3], queue_ids=["id1", "id2", "id3", "id4"], queue_names=["set1", "set2", "set3", "set4"], run_id=1, request_id=2, effective_workers=4, full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={set_id: _test_simulation_plan_payload(set_id=set_id, set_name=f"set{idx}", initials={"A": 1.0}) for idx, set_id in enumerate(("id1", "id2", "id3", "id4"), start=1)})

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert controller._batch_parallel.current_max_workers == 4
    assert old_pool.close_calls == [True]
    assert old_submitted == []
    assert len(new_submitted) == 4


@pytest.mark.unit
def test_run_simulation_internal_builds_context_and_calls_start_next(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return "state: A, kind=GS, energy=0"

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\ninitial: A=1")
            self._state_network_editor = _StateNetworkEditor()

    batch_names = ["set1"]
    mw._batch_store.row_count.side_effect = lambda: len(batch_names)
    mw._batch_store.set_names.side_effect = lambda: list(batch_names)
    mw._batch_store.ensure_set.side_effect = (
        lambda name: batch_names.index(str(name))
        if str(name) in batch_names
        else (batch_names.append(str(name)) or (len(batch_names) - 1))
    )
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": (
            {"randomname3": {"A": 1.0}},
            text.replace("initial: A=1", "# Initial concentrations moved to Batch Initial Conditions table (randomname3). Edit there."),
        ),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text.replace("initial:", "# stripped initial:"),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    mw.discard_concentration_overlays_for_rows.return_value = True

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    policy_context = _batch_policy_context(controller)
    execution_state = controller.batch_context_owner.execution_payload_state()
    cache_key = controller.batch_context_owner.completion_flush_context().cache_key
    assert policy_context.active is True
    assert policy_context.parallel is False
    assert isinstance(cache_key, str)
    assert cache_key != "ck"
    assert any("# State Network" in text for text in execution_state.mechanism_text_by_set_id.values())
    assert policy_context.queue_names == ("randomname3",)
    assert policy_context.pending_init_seed == {"randomname3": {"A": 1.0}}
    assert isinstance(policy_context.pending_init_rewrite, str) and policy_context.pending_init_rewrite
    assert policy_context.pending_init_applied is True
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_simulation_identity_for_set_uses_supplied_intervention_schedule_fingerprint(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    solver_config = {
        "solver": "BDF",
        "rtol": 1e-6,
        "atol": 1e-12,
        "grid_n": 100,
        "temperature_K": 298.15,
    }

    mw._get_mechanism_text.side_effect = AssertionError("schedule identity must not reparse controller DSL")
    first_identity = controller._simulation_identity_for_set(
        set_id="id1",
        solver_config=solver_config,
        t_end=10.0,
        intervention_schedule_fingerprint="submitted-schedule-a",
        fast_mode=False,
    )
    second_identity = controller._simulation_identity_for_set(
        set_id="id1",
        solver_config=solver_config,
        t_end=10.0,
        intervention_schedule_fingerprint="submitted-schedule-b",
        fast_mode=False,
    )

    assert first_identity.intervention_schedule_fingerprint == "submitted-schedule-a"
    assert second_identity.intervention_schedule_fingerprint == "submitted-schedule-b"
    assert first_identity.cache_key() != second_identity.cache_key()


@pytest.mark.unit
def test_run_dispatch_cache_key_includes_parsed_intervention_schedule_fingerprint(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    from kindred.core.simulation_plan import SimulationPlan

    def _configure_run(mechanism_text: str) -> None:
        _install_mechanism_editor_text(mw, mechanism_text)
        mw._batch_store.row_count.return_value = 1
        mw._batch_store.set_names.return_value = ["set1"]
        mw._batch_rows_for_scope.return_value = [0]
        mw._batch_set_id_for_row.return_value = "id1"
        mw._batch_preferred_primary_set_id.return_value = "id1"
        mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
        mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    first_dsl = "\n".join(
        [
            "reaction: A -> B; k=1",
            "intervention: time=1.0; species=A; op=set; value=2.0",
        ]
    )
    second_dsl = "\n".join(
        [
            "reaction: A -> B; k=1",
            "intervention: time=1.0; species=A; op=set; value=3.0",
        ]
    )

    _configure_run(first_dsl)
    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    first_key = controller.batch_context_owner.completion_flush_context().cache_key
    first_plan_payload = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id["id1"]
    first_identity = SimulationPlan.from_payload(first_plan_payload).simulation_identity_payload()

    seed_batch_context(controller.batch_context_owner, active=False)
    _configure_run(second_dsl)
    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0], reuse_parallel_lane_pool=False)
    second_key = controller.batch_context_owner.completion_flush_context().cache_key
    second_plan_payload = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id["id1"]
    second_identity = SimulationPlan.from_payload(second_plan_payload).simulation_identity_payload()

    assert first_identity["intervention_schedule_fingerprint"]
    assert second_identity["intervention_schedule_fingerprint"]
    assert first_identity["intervention_schedule_fingerprint"] != second_identity["intervention_schedule_fingerprint"]
    assert first_key != second_key


@pytest.mark.unit
def test_run_dispatch_normalizes_schedule_param_direct_spelling(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    from kindred.core.simulation_plan import SimulationPlan

    k1_dsl = "\n".join(
        [
            "reaction: A -> B; k=1",
            "initial: A=1.0",
            "intervention: time=1.0; species=A; op=add; amount_param=k1",
        ]
    )
    K1_dsl = "\n".join(
        [
            "reaction: A -> B; k=1",
            "initial: A=1.0",
            "intervention: time=1.0; species=A; op=add; amount_param=K1",
        ]
    )
    def _configure_run(mechanism_text: str) -> None:
        _install_mechanism_editor_text(mw, mechanism_text)
        mw._batch_store.row_count.return_value = 1
        mw._batch_store.set_names.return_value = ["set1"]
        mw._batch_rows_for_scope.return_value = [0]
        mw._batch_set_id_for_row.return_value = "id1"
        mw._batch_preferred_primary_set_id.return_value = "id1"
        mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
        mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    _configure_run(k1_dsl)
    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    first_key = controller.batch_context_owner.completion_flush_context().cache_key
    first_plan_payload = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id["id1"]
    first_identity = SimulationPlan.from_payload(first_plan_payload).simulation_identity_payload()

    seed_batch_context(controller.batch_context_owner, active=False)
    _configure_run(K1_dsl)
    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0], reuse_parallel_lane_pool=False)
    second_key = controller.batch_context_owner.completion_flush_context().cache_key
    second_plan_payload = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id["id1"]
    second_identity = SimulationPlan.from_payload(second_plan_payload).simulation_identity_payload()

    assert first_identity["intervention_schedule_fingerprint"]
    assert first_identity["intervention_schedule_fingerprint"] == second_identity["intervention_schedule_fingerprint"]
    assert first_key == second_key


@pytest.mark.unit
def test_serial_single_set_run_uses_contained_owner_lane(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _FakeContainedWorker:
        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ):
            self.owner = owner
            self.simulation_plan_payload = dict(simulation_plan_payload)
            self.include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            self.parent = parent
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()
            self.started = False

        def start(self) -> None:
            self.started = True

        def isRunning(self) -> bool:
            return False

        def cancel(self) -> None:
            return

    class _FakeOwner:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    created_workers: list[_FakeContainedWorker] = []
    owners = {False: _FakeOwner("ordinary"), True: _FakeOwner("preview")}

    def _worker_factory(**kwargs):
        worker = _FakeContainedWorker(**kwargs)
        created_workers.append(worker)
        return worker

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _worker_factory)
    controller._contained_simulation_owner_factory = lambda *, fast_mode: owners[bool(fast_mode)]
    def _ready_owner(*, fast_mode, simulation_plan_payload):
        owner = owners[bool(fast_mode)]
        setattr(controller, controller._contained_owner_attr(fast_mode=bool(fast_mode)), owner)
        return owner

    monkeypatch.setattr(
        controller._runtime_application,
        "acquire_ready_owner",
        lambda *, mode, payload: _ready_owner(
            fast_mode=(str(mode) == "preview"),
            simulation_plan_payload=payload,
        ),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    mw._parse_sim_time_seconds.return_value = 10.0

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert len(created_workers) == 1
    worker = created_workers[0]
    assert worker.owner is owners[False]
    assert controller._ordinary_simulation_owner is owners[False]
    assert controller._preview_simulation_owner is None
    assert worker.started is True
    assert worker.simulation_plan_payload["execution_mode"] == "explicit"
    assert worker.simulation_plan_payload["execution_request"]["prepared_payload"] is None


@pytest.mark.unit
def test_controller_close_teardown_closes_ordinary_and_preview_contained_owners(
    controller: SimulationController,
):
    class _FakeOwner:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    ordinary = _FakeOwner()
    preview = _FakeOwner()
    controller._ordinary_simulation_owner = ordinary
    controller._preview_simulation_owner = preview
    controller._shutdown_batch_lane_pool = MagicMock()

    assert controller._prepare_simulation_shutdown_for_close() is True

    assert ordinary.close_calls == [True]
    assert preview.close_calls == [True]
    assert controller._ordinary_simulation_owner is None
    assert controller._preview_simulation_owner is None
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_close_teardown_detaches_active_contained_owner_for_running_worker(
    controller: SimulationController,
):
    worker = make_stubborn_worker(_FakeWorker)
    owner = _FakeContainedOwner()
    controller._simulation_worker = worker
    controller._ordinary_simulation_owner = owner
    seed_batch_context(controller.batch_context_owner, active=True, fast_mode=False)
    controller._shutdown_batch_lane_pool = MagicMock()

    assert controller._prepare_simulation_shutdown_for_close() is False

    assert worker._cancelled is True
    assert owner.close_calls == []
    assert controller._ordinary_simulation_owner is None
    assert worker in controller._retained_simulation_workers
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_close_teardown_closes_detached_owner_when_worker_stops_during_cleanup(
    controller: SimulationController,
):
    worker = _FakeWorker(running=True, wait_returns=True)
    owner = _FakeContainedOwner()
    controller._simulation_worker = worker
    controller._ordinary_simulation_owner = owner
    seed_batch_context(controller.batch_context_owner, active=True, fast_mode=False)
    controller._shutdown_batch_lane_pool = MagicMock()

    assert controller._prepare_simulation_shutdown_for_close() is True

    assert worker._cancelled is True
    assert owner.close_calls == [True]
    assert controller._ordinary_simulation_owner is None
    assert worker not in controller._retained_simulation_workers
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_successful_ordinary_completion_retains_contained_owner(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    owner = _FakeContainedOwner()
    controller._ordinary_simulation_owner = owner
    controller._simulation_running = True
    controller._active_run_id = 1
    controller.run_state.latest_sim_request_id = 2
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, queue_ids=["id1"], queue_names=["set1"], completed_set_ids=[], total=1, pos=0, pending_workspace_reset_set_ids=[])
    controller._cleanup_parallel_batch_lane_pool_after_run = MagicMock()

    payload = _successful_result_payload()
    payload.update(
        {
            "success": True,
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
        }
    )

    _complete_with_callback_identity(
        controller,
        payload,
        run_id=1,
        fast_mode=False,
        request_id=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="cache-key",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert controller._ordinary_simulation_owner is owner
    assert owner.close_calls == []
    assert controller._simulation_running is False
    assert mw.run_button_is_enabled() is True
    assert mw._stop_btn.enabled is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("phase", "message"),
    [
        ("ready", "Contained simulation owner startup timed out."),
        ("accept", "Contained simulation owner accept timed out."),
        ("active_solve", "Simulation timed out during active solve."),
    ],
)
def test_ordinary_containment_timeout_error_resets_ui_and_discards_owner(
    mw: _FakeMainWindow,
    controller: SimulationController,
    phase: str,
    message: str,
):
    owner = _FakeContainedOwner()
    controller._ordinary_simulation_owner = owner
    controller._simulation_running = True
    controller._active_run_id = 7
    controller.run_state.latest_sim_request_id = 8
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, queue_ids=["id1"], queue_names=["set1"], completed_set_ids=[], total=1, pos=0)
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._shutdown_batch_lane_pool = MagicMock()
    critical_messages: list[tuple[str, str, Optional[str]]] = []
    mw.message_box_critical = lambda title, text, *, details=None: critical_messages.append(
        (str(title), str(text), str(details) if details else None)
    )
    mw.set_sim_progress_value(55)
    mw.set_run_button_enabled(False)
    mw.set_stop_button_enabled(True)

    _error_with_callback_identity(
        controller,
        build_simulation_failure(
            "timeout",
            message,
            details={"phase": phase},
            exc_type="SimulationContainmentTimeout",
        ),
        run_id=7,
        fast_mode=False,
        request_id=8,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="cache-key",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert controller._ordinary_simulation_owner is None
    assert owner.close_calls == [True]
    assert controller._simulation_running is False
    assert mw.run_button_is_enabled() is True
    assert mw._stop_btn.enabled is False
    assert mw._sim_progress.value == 0
    assert mw._status_label.text == "Simulation failed"
    assert critical_messages
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)


@pytest.mark.unit
def test_contained_owner_reuse_accepts_equivalent_copied_numpy_y0(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _normal_plan, _payload_copy_with_distinct_y0

    startup_payload = _normal_plan().to_payload()
    equivalent_payload = _payload_copy_with_distinct_y0(startup_payload)

    class _Owner:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self._payload)

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner(startup_payload)
    controller._ordinary_simulation_owner = owner

    reused = controller._contained_simulation_owner(
        fast_mode=False,
        simulation_plan_payload=equivalent_payload,
    )

    assert reused is owner
    assert controller._ordinary_simulation_owner is owner
    assert owner.close_calls == []


@pytest.mark.unit
def test_contained_owner_reuse_accepts_changed_copied_numpy_y0(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _normal_plan, _payload_copy_with_distinct_y0

    startup_payload = _normal_plan().to_payload()
    changed_payload = _payload_copy_with_distinct_y0(startup_payload, [1.0, 0.5])
    owner = _FakeContainedOwner()
    owner.simulation_plan_payload = startup_payload
    controller._ordinary_simulation_owner = owner

    reused = controller._contained_simulation_owner(
        fast_mode=False,
        simulation_plan_payload=changed_payload,
    )

    assert reused is owner
    assert controller._ordinary_simulation_owner is owner
    assert owner.close_calls == []


@pytest.mark.unit
def test_readiness_warm_starts_matching_contained_owner_off_gui_wait(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _normal_plan

    startup_payload = _normal_plan().to_payload()

    class _Owner:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = dict(payload)
            self.start_calls: list[dict[str, object]] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self._payload)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})

    owner = _Owner(startup_payload)
    controller._ordinary_simulation_owner = owner

    controller._warm_contained_simulation_owner_for_plan(
        fast_mode=False,
        simulation_plan_payload=startup_payload,
    )

    assert owner.start_calls == [{"wait": True}]


@pytest.mark.unit
def test_interactive_runtime_availability_does_not_create_generic_empty_owners(
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.start_calls: list[dict[str, object]] = []

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    controller._contained_simulation_owner_factory = _factory

    controller.ensure_interactive_simulation_runtimes_available(wait=False)

    assert created == []
    assert controller._ordinary_simulation_owner is None
    assert controller._preview_simulation_owner is None


@pytest.mark.unit
def test_interactive_runtime_availability_warms_exact_ordinary_and_preview_plans(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.start_calls: list[dict[str, object]] = []
            self.ready = False

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _factory

    controller.ensure_interactive_simulation_runtimes_available(wait=True)

    assert [(owner.fast_mode, owner.start_calls) for owner in created] == [
        (False, [{"wait": True}]),
        (True, [{"wait": True}]),
    ]
    assert all(owner.is_ready for owner in created)
    for owner, expected_mode in ((created[0], "explicit"), (created[1], "preview")):
        assert owner.payload.get("execution_mode") == expected_mode
        execution_request = owner.payload.get("execution_request")
        assert isinstance(execution_request, dict)
        assert execution_request.get("mechanism_text") == "reaction: A -> B; k=1"
        assert execution_request.get("initials") == {"A": 1.0, "B": 0.0}


@pytest.mark.unit
def test_simulation_runtime_inputs_changed_invalidates_blas_dependent_serial_runtimes(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            self.ready = True

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _factory
    controller.parallel_batch.limit_blas_threads_per_worker = True

    controller.ensure_interactive_simulation_runtimes_available(wait=True)
    initial_ordinary = controller._ordinary_simulation_owner
    initial_preview = controller._preview_simulation_owner
    assert [
        owner.payload["metadata"]["contained_owner_identity"]["contained_child_blas_threads_limited"]
        for owner in created
    ] == [True, True]

    controller.parallel_batch.limit_blas_threads_per_worker = False
    controller.simulation_runtime_inputs_changed()

    assert initial_ordinary.close_calls == [False]
    assert initial_preview.close_calls == [False]
    assert len(created) == 4
    assert [
        owner.payload["metadata"]["contained_owner_identity"]["contained_child_blas_threads_limited"]
        for owner in created[2:]
    ] == [False, False]
    assert controller._ordinary_simulation_owner is created[2]
    assert controller._preview_simulation_owner is created[3]


@pytest.mark.unit
def test_interactive_runtime_ready_is_pure_and_ensure_warms_changed_payload(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False
            self.start_calls: list[dict[str, object]] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

    created: list[_Owner] = []
    release_background_start = threading.Event()
    background_start_entered = threading.Event()

    class _BlockingOwner(_Owner):
        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            background_start_entered.set()
            release_background_start.wait(timeout=2.0)
            self.ready = True

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _BlockingOwner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _factory

    release_background_start.set()
    controller.ensure_interactive_simulation_runtimes_available(wait=True)
    assert controller.interactive_simulation_runtime_ready(fast_mode=False) is True
    release_background_start.clear()
    background_start_entered.clear()

    mw._batch_set_id_for_row.return_value = "id2"
    mw._batch_preferred_primary_set_id.return_value = "id2"
    mw._simulation_param_fingerprints = {"id2": "params-id2"}

    assert controller.interactive_simulation_runtime_ready(fast_mode=False) is False
    assert not background_start_entered.wait(timeout=0.05)

    controller._ensure_interactive_simulation_runtime_available_for_mode(fast_mode=False, wait=False)
    assert background_start_entered.wait(timeout=1.0)
    release_background_start.set()


@pytest.mark.unit
def test_interactive_runtime_availability_does_not_apply_pending_init_migration(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.start_calls: list[dict[str, object]] = []
            self.ready = False

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    apply_pending_init_migration = MagicMock(return_value=True)
    monkeypatch.setattr(
        mw._mechanism_helpers,
        "apply_pending_init_migration",
        apply_pending_init_migration,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": (
            {"set1": {"A": 1.0}},
            "reaction: A -> B; k=1",
        ),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1\ninitial: A=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _factory

    controller.ensure_interactive_simulation_runtimes_available(wait=True)

    assert created
    apply_pending_init_migration.assert_not_called()


@pytest.mark.unit
def test_interactive_runtime_snapshot_surfaces_payload_build_failure(monkeypatch, controller: SimulationController):
    def _raise_payload_error(*, fast_mode: bool):
        _ = fast_mode
        raise ValueError("invalid intervention schedule")

    monkeypatch.setattr(
        controller,
        "_interactive_runtime_plan_payloads_for_mode",
        _raise_payload_error,
    )

    snapshot = controller.interactive_simulation_runtime_snapshot(fast_mode=False)

    assert snapshot.status == "failed"
    assert snapshot.ready is False
    assert snapshot.required is True
    assert snapshot.controls_ready is False
    assert snapshot.should_poll is False
    assert "invalid intervention schedule" in str(snapshot.failure)


@pytest.mark.unit
def test_interactive_runtime_availability_uses_readiness_boundary_without_user_launcher(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            if bool(wait):
                self.ready = True

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _factory
    controller._run_simulation_internal = MagicMock(side_effect=AssertionError("readiness must not launch user simulation"))

    controller._ensure_interactive_simulation_runtime_available_for_mode(fast_mode=False, wait=True)

    assert len(created) == 1
    assert created[0].payload["execution_mode"] == "explicit"


@pytest.mark.unit
def test_serial_runtime_readiness_warms_every_selected_set_owner(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False
            self.start_calls: list[dict[str, object]] = []

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

    created: list[_Owner] = []

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.side_effect = lambda row: {"A": float(int(row) + 1), "B": 0.0}
    controller._contained_simulation_owner_factory = _factory

    controller._ensure_interactive_simulation_runtime_available_for_mode(fast_mode=False, wait=True)

    warmed_set_ids = [
        dict(owner.payload.get("metadata") or {}).get("set_id")
        for owner in created
    ]
    assert warmed_set_ids == ["id1", "id2"]
    assert all(owner.start_calls == [{"wait": True}] for owner in created)


@pytest.mark.unit
def test_first_serial_actions_reuse_startup_ready_exact_contained_owners(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []
            self.ready = False

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    class _ContainedWorker:
        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ) -> None:
            self.owner = owner
            self._owner = owner
            self.simulation_plan_payload = dict(simulation_plan_payload)
            self.include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            self.parent = parent
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()
            self.started = False

        def start(self) -> None:
            self.started = True

        def isRunning(self) -> bool:
            return False

        def cancel(self) -> None:
            return

    created_owners: list[_Owner] = []
    created_workers: list[_ContainedWorker] = []

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created_owners.append(owner)
        return owner

    def _worker_factory(**kwargs) -> _ContainedWorker:
        worker = _ContainedWorker(**kwargs)
        created_workers.append(worker)
        return worker

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _worker_factory)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    mw._variable_slider_values = {"k1": 1.0}
    controller._contained_simulation_owner_factory = _owner_factory

    controller.ensure_interactive_simulation_runtimes_available(wait=True)
    ordinary_owner = controller._ordinary_simulation_owner
    preview_owner = controller._preview_simulation_owner
    assert ordinary_owner is created_owners[0]
    assert preview_owner is created_owners[1]

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=11,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
    )
    ordinary_worker = created_workers[-1]
    controller._release_runtime_owner_from_worker(ordinary_worker)
    controller._simulation_worker = None
    mw._slider_overrides = {"k1": 2.0}
    mw._simulation_param_fingerprints = {"id1": "params-slider-2"}
    controller._run_simulation_internal(
        fast_mode=True,
        request_id=12,
        batch_rows=[0],
        reuse_parallel_lane_pool=True,
    )
    first_preview_worker = created_workers[-1]
    controller._release_runtime_owner_from_worker(first_preview_worker)
    controller._simulation_worker = None
    mw._slider_overrides = {"k1": 3.0}
    mw._simulation_param_fingerprints = {"id1": "params-slider-3"}
    controller._run_simulation_internal(
        fast_mode=True,
        request_id=13,
        batch_rows=[0],
        reuse_parallel_lane_pool=True,
    )

    assert controller._ordinary_simulation_owner is ordinary_owner
    assert controller._preview_simulation_owner is preview_owner
    assert ordinary_owner.start_calls == [{"wait": True}]
    assert preview_owner.start_calls == [{"wait": True}]
    assert ordinary_owner.close_calls == []
    assert preview_owner.close_calls == []
    assert [
        ordinary_worker.owner,
        first_preview_worker.owner,
        created_workers[-1].owner,
    ] == [ordinary_owner, preview_owner, preview_owner]


@pytest.mark.unit
def test_first_ordinary_action_requires_ready_exact_contained_owner(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.start_calls: list[dict[str, object]] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return False

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})

    created_owners: list[_Owner] = []
    created_workers: list[object] = []

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created_owners.append(owner)
        return owner

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.ContainedSimulationWorker",
        lambda **kwargs: created_workers.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _owner_factory

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=11,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
    )

    assert created_owners == []
    assert created_workers == []
    assert not mw._run_btn.isEnabled()
    assert "runtime" in mw._status_label.text.lower()
    assert "ready" in mw._status_label.text.lower()


@pytest.mark.unit
def test_contained_worker_construction_failure_releases_acquired_runtime_owner(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            if bool(wait):
                self.ready = True

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    created_owners: list[_Owner] = []

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created_owners.append(owner)
        return owner

    def _raising_worker_factory(**_kwargs):
        raise RuntimeError("worker construction failed")

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _raising_worker_factory)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _owner_factory
    controller._on_simulation_error = MagicMock()

    controller.ensure_interactive_simulation_runtimes_available(wait=True)
    ordinary_owner = controller._ordinary_simulation_owner
    assert ordinary_owner is created_owners[0]
    assert ordinary_owner.is_ready

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=11,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
    )

    assert controller._simulation_worker is None
    assert controller._ordinary_simulation_owner is ordinary_owner
    assert ordinary_owner.close_calls == []


@pytest.mark.unit
def test_contained_worker_construction_failure_release_error_preserves_original_failure(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Owner:
        def __init__(self, payload: dict[str, object], *, fast_mode: bool) -> None:
            self.payload = dict(payload)
            self.fast_mode = bool(fast_mode)
            self.ready = False

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self.payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def start(self, *, wait: bool = True) -> None:
            if bool(wait):
                self.ready = True

        def close(self, *, kill: bool = False) -> None:
            return None

    created_owners: list[_Owner] = []

    def _owner_factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]) -> _Owner:
        owner = _Owner(dict(simulation_plan_payload), fast_mode=bool(fast_mode))
        created_owners.append(owner)
        return owner

    def _raising_worker_factory(**_kwargs):
        raise RuntimeError("worker construction failed")

    def _raising_release_owner(_owner, *, kill: bool = False) -> None:
        raise RuntimeError("release failed")

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _raising_worker_factory)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._contained_simulation_owner_factory = _owner_factory
    controller._on_simulation_error = MagicMock()

    controller.ensure_interactive_simulation_runtimes_available(wait=True)
    assert created_owners
    monkeypatch.setattr(controller._runtime_application, "release_owner", _raising_release_owner)

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=11,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
    )

    error_payload = controller._on_simulation_error.call_args.args[0]
    assert error_payload["message"] == "worker construction failed"
    assert error_payload["exc_type"] == "RuntimeError"
    assert "release failed" in str(controller._last_nonfatal_exception)


@pytest.mark.unit
def test_first_preview_action_requires_ready_exact_contained_owner(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _FakeRuntime:
        species_names = ["A", "B"]

        def as_worker_payload(self) -> dict[str, object]:
            return {"prepared": True}

        def as_serializable_execution_payload(self) -> dict[str, object]:
            return {"prepared": True, "version": 2}

    created_owners: list[object] = []
    created_workers: list[object] = []

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.ContainedSimulationWorker",
        lambda **kwargs: created_workers.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_store.visible_species.return_value = ["A", "B"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    mw._slider_overrides = {"k1": 2.0}
    mw._variable_slider_values = {"k1": 1.0}
    mw._simulation_param_fingerprints = {"id1": "params-slider-2"}
    mw._prepare_slider_runtime.return_value = _FakeRuntime()
    mw._apply_slider_overrides_to_bindings.return_value = True
    controller._contained_simulation_owner_factory = (
        lambda **_kwargs: created_owners.append(object()) or created_owners[-1]
    )

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=12,
        batch_rows=[0],
        reuse_parallel_lane_pool=True,
    )

    assert created_owners == []
    assert created_workers == []
    assert mw._preview_unavailable_messages
    assert "runtime" in mw._preview_unavailable_messages[-1].lower()
    assert "ready" in mw._preview_unavailable_messages[-1].lower()


@pytest.mark.unit
def test_readiness_warm_replaces_generic_prewarmed_owner_for_first_real_plan(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _normal_plan

    startup_payload = _normal_plan().to_payload()

    class _Owner:
        def __init__(self, payload: dict[str, object] | None = None) -> None:
            self._payload = dict(payload or {})
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self._payload)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner()
    replacement = _Owner(startup_payload)
    controller._ordinary_simulation_owner = owner
    controller._contained_simulation_owner_factory = lambda *, fast_mode, simulation_plan_payload: replacement

    controller._warm_contained_simulation_owner_for_plan(
        fast_mode=False,
        simulation_plan_payload=startup_payload,
    )

    assert owner.close_calls == [False]
    assert owner.start_calls == []
    assert replacement.start_calls == [{"wait": True}]
    assert controller._ordinary_simulation_owner is replacement


@pytest.mark.unit
def test_readiness_warm_replaces_prepare_capable_owner_for_identity_change(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _energy_scheduled_plan, _normal_plan

    startup_payload = _normal_plan().to_payload()
    changed_payload = _energy_scheduled_plan().to_payload()

    class _PrepareCapableOwner:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = dict(payload)
            self.ready = True
            self.prepare_calls: list[dict[str, object]] = []
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self._payload)

        @property
        def is_ready(self) -> bool:
            return bool(self.ready)

        def prepare_runtime_payload(self, payload: dict[str, object]) -> None:
            self.prepare_calls.append(dict(payload))
            self._payload = dict(payload)
            self.ready = True

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})
            self.ready = True

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _PrepareCapableOwner(startup_payload)
    replacement = _PrepareCapableOwner(changed_payload)
    factory_calls: list[dict[str, object]] = []
    controller._ordinary_simulation_owner = owner

    def _factory(*, fast_mode: bool, simulation_plan_payload: dict[str, object]):
        factory_calls.append(dict(simulation_plan_payload))
        return replacement

    controller._contained_simulation_owner_factory = _factory

    controller._warm_contained_simulation_owner_for_plan(
        fast_mode=False,
        simulation_plan_payload=changed_payload,
        wait=True,
    )

    assert controller._ordinary_simulation_owner is replacement
    assert owner.prepare_calls == []
    assert owner.start_calls == []
    assert owner.close_calls == [False]
    assert replacement.prepare_calls == [changed_payload]
    assert replacement.start_calls == []
    assert replacement.close_calls == []
    assert factory_calls == [changed_payload]


@pytest.mark.unit
def test_readiness_warm_replaces_existing_mismatched_contained_owner(controller: SimulationController):
    from tests.test_simulation_containment_payloads import _normal_plan, _payload_copy_with_distinct_y0

    startup_payload = _normal_plan().to_payload()
    changed_payload = _payload_copy_with_distinct_y0(startup_payload, [1.0, 0.5])

    class _Owner:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = dict(payload)
            self.start_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, object]:
            return dict(self._payload)

        def start(self, *, wait: bool = True) -> None:
            self.start_calls.append({"wait": bool(wait)})

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner(startup_payload)
    replacement = _Owner(changed_payload)
    controller._ordinary_simulation_owner = owner
    controller._contained_simulation_owner_factory = lambda *, fast_mode, simulation_plan_payload: replacement

    controller._warm_contained_simulation_owner_for_plan(
        fast_mode=False,
        simulation_plan_payload=changed_payload,
    )

    assert owner.close_calls == [False]
    assert replacement.start_calls == [{"wait": True}]
    assert controller._ordinary_simulation_owner is replacement


@pytest.mark.unit
def test_run_simulation_internal_merges_empty_default_named_block_with_legacy_initials(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 1

    class _Text:
        def __init__(self, text):
            self._text = text

        def toPlainText(self):
            return self._text

        def setPlainText(self, value):
            self._text = str(value)

    class _StateNetworkEditor:
        def get_dsl_text(self) -> str:
            return ""

        def getPlainText(self) -> str:
            return ""

        def setPlainText(self, value: str) -> None:
            self._text = str(value)

        def toPlainText(self) -> str:
            return getattr(self, "_text", "")

        def get_state_network_dsl(self) -> str:
            return "state: A, kind=GS, energy=0"

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text(
                "reaction: A -> B; k=1\n\nset1 = {\n}\n\n# Initial concentrations\n[A] = 1.0\n"
            )
            self._state_network_editor = _StateNetworkEditor()

    batch_names = ["set1"]
    mw._batch_store.row_count.side_effect = lambda: len(batch_names)
    mw._batch_store.set_names.side_effect = lambda: list(batch_names)
    mw._batch_store.ensure_set.side_effect = (
        lambda name: batch_names.index(str(name))
        if str(name) in batch_names
        else (batch_names.append(str(name)) or (len(batch_names) - 1))
    )
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    policy_context = _batch_policy_context(controller)

    assert policy_context.pending_init_seed == {"set1": {"A": 1.0}}
    assert isinstance(policy_context.pending_init_rewrite, str) and policy_context.pending_init_rewrite
    assert policy_context.pending_init_applied is True
    assert policy_context.pending_init_rewrite.count(
        "Initial concentrations moved to Batch Initial Conditions table (set1). Edit there."
    ) == 2
    mw._batch_store.set_value.assert_any_call(0, "A", "1")
    mw._set_text_with_optional_undo.assert_called()
    rewritten = str(mw._set_text_with_optional_undo.call_args.args[1])
    assert "set1 = {" not in rewritten
    assert "[A] = 1.0" not in rewritten
    assert rewritten.count(
        "Initial concentrations moved to Batch Initial Conditions table (set1). Edit there."
    ) == 2

@pytest.mark.unit
def test_run_simulation_internal_fast_mode_isolates_prepared_payloads_per_set(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text(
                "\n".join(
                    [
                        "reaction: A -> B; k=1",
                        "intervention: op=set; species=A; time=0.0; value=2.0",
                    ]
                )
            )
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(param_names: Optional[list[str]] = None, *, set_id: Optional[str] = None):
        _ = param_names, set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime()
        created_runtimes.append(runtime)
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 4.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 2.5}, {"A": 5.5}])
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    execution_state = controller.batch_context_owner.execution_payload_state()
    prepared_by_set_id = execution_state.prepared_by_set_id
    simulation_plan_by_set_id = execution_state.simulation_plan_by_set_id
    assert execution_state.prepared is None
    assert prepared_by_set_id == {}
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()
    plan = SimulationPlan.from_payload(simulation_plan_by_set_id["id1"])
    assert plan.execution_mode == "preview"
    assert plan.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    request_id1 = plan.to_execution_request().to_payload()
    request_id2 = SimulationPlan.from_payload(
        simulation_plan_by_set_id["id2"]
    ).to_execution_request().to_payload()
    assert request_id1["prepared_payload"] is None
    assert request_id2["prepared_payload"] is None
    assert request_id1["initials"] == {"A": 2.5}
    assert request_id2["initials"] == {"A": 5.5}
    assert request_id1["parameter_overrides"] == {"k1": 2.0}
    assert request_id2["parameter_overrides"] == {"k1": 2.0}
    assert created_runtimes == []

@pytest.mark.unit
def test_run_simulation_internal_fast_mode_refreshes_runtime_after_multi_set_preview(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text(
                "\n".join(
                    [
                        "reaction: A -> B; k=1",
                        "intervention: op=set; species=A; time=0.0; value=2.0",
                    ]
                )
            )
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.mechanism = {"runtime": self.label}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(param_names: Optional[list[str]] = None, *, set_id: Optional[str] = None):
        _ = param_names, set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime(label=f"runtime-{len(created_runtimes) + 1}")
        created_runtimes.append(runtime)
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.side_effect = lambda rows: "id2" if list(rows) == [1] else "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    assert created_runtimes == []

    seed_batch_context(controller.batch_context_owner, active=False)
    controller._run_simulation_internal(fast_mode=True, request_id=8, batch_rows=[1], reuse_parallel_lane_pool=False)

    assert created_runtimes == []
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_fast_mode_does_not_prepare_or_dirty_gui_runtime_after_multi_set_loop(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """Multi-set fast-mode planning must not mutate GUI prepared-runtime state."""
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.mechanism = {"runtime": self.label}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(param_names: Optional[list[str]] = None, *, set_id: Optional[str] = None):
        _ = param_names, set_id
        cached = getattr(mw, "_prepared_slider_runtime_cache", None)
        if cached is not None and not bool(mw._slider_runtime_dirty):
            return cached
        runtime = _FakeRuntime(label=f"runtime-{set_id}")
        mw._prepared_slider_runtime_cache = runtime
        mw._slider_runtime_dirty = False
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.side_effect = lambda rows: "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=10, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    assert mw._slider_runtime_dirty is False
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_fast_deferral_replaces_deferred_target_snapshot_and_dispatch_uses_it(
    mw: _FakeMainWindow, controller: SimulationController
):
    active_worker = _FakeWorker(running=True, wait_returns=True)
    active_worker._fast_mode = True  # type: ignore[attr-defined]
    controller._simulation_worker = active_worker
    controller._simulation_running = True
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 6
    controller._pending_slider_target_set_ids = ("stale-id",)
    mw._batch_store.row_count.return_value = 2
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._last_slider_change_name = "k1"

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=7,
        batch_rows=[1],
        reuse_parallel_lane_pool=False,
    )

    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("id2",)

    controller._simulation_worker = None
    controller._simulation_running = False
    controller.run_simulation_internal = MagicMock()
    mw._batch_rows_for_scope.return_value = [0]

    controller._run_simulation_from_slider()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["request_id"] == 7
    assert kwargs["batch_rows"] == [1]


@pytest.mark.unit
def test_runtime_readiness_only_fast_active_preview_does_not_mutate_pending_replay(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    active_worker = _FakeWorker(running=True, wait_returns=True)
    active_worker._fast_mode = True  # type: ignore[attr-defined]
    controller._simulation_worker = active_worker
    controller._simulation_running = True
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 6
    controller._pending_slider_target_set_ids = ("stale-id",)
    mw._batch_store.row_count.return_value = 2
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=7,
        batch_rows=[1],
        reuse_parallel_lane_pool=False,
        runtime_readiness_only=True,
    )

    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 6
    assert controller._pending_slider_target_set_ids == ("stale-id",)


@pytest.mark.unit
def test_runtime_readiness_only_ordinary_warm_preserves_preview_ownership(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}
    controller._warm_contained_simulation_owner_for_plan = MagicMock()
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=3,
        target_set_ids=("id1",),
    )

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=0,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
        runtime_readiness_only=True,
    )

    assert controller.run_state.preview_ownership == PreviewOwnershipState(
        request_id=5,
        epoch=3,
        target_set_ids=("id1",),
    )
    mw._sync_batch_species_columns.assert_not_called()


@pytest.mark.unit
def test_runtime_readiness_only_ordinary_warm_does_not_mutate_status_for_solver_warning(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    mw._get_mechanism_text.return_value = "reaction: A -> B; k=1"
    mw._initial_solver = "LegacySolver"
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}

    controller._run_simulation_internal(
        fast_mode=False,
        request_id=0,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
        runtime_readiness_only=True,
    )

    assert mw._status_label.text == ""


@pytest.mark.unit
def test_fast_preview_completion_uses_dispatch_time_overlay_token_snapshot(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    row_to_set_id = {0: "id1", 1: "id2"}

    def _preview_token(rows: list[int]) -> str:
        tokens: list[str] = []
        for row in rows or []:
            set_id = row_to_set_id.get(int(row))
            if set_id:
                tokens.append(f"token:{set_id}")
        return "|".join(tokens)

    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: row_to_set_id[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(side_effect=_preview_token)

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    controller._latest_sim_request_id = 7
    controller._queue_slider_plot_update = MagicMock()
    mw._mechanism_editor._reactions_text = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=7, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    assert controller.batch_context_owner.preview_batch_cache_token_for_set("id1") == "token:id1"
    assert controller.batch_context_owner.preview_batch_cache_token_for_set("id2") == "token:id2"
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=True,
        request_id=7,
        batch_set="set1",
        batch_set_id="id1",
        cache_key=str(controller.batch_context_owner.completion_cache_key()),
        simulation_identity=controller.batch_context_owner.simulation_identity_for_set("id1"),
        preview_batch_cache_token=controller.batch_context_owner.preview_batch_cache_token_for_set("id1"),
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    row_to_set_id.clear()
    row_to_set_id.update({0: "id9", 1: "id1", 2: "id2"})

    result = _successful_result_payload()
    cache_key = str(controller.batch_context_owner.completion_cache_key())
    controller._on_simulation_complete(
        result,
        callback_identity=callback_identity,
    )

    payload = controller.batch_cache.entry_for_set(cache_key=cache_key, set_id="id1", is_preview=True).entry
    assert isinstance(payload, dict)
    assert payload.get("preview_batch_cache_token") == "token:id1"

@pytest.mark.unit
def test_run_simulation_internal_fast_mode_parallel_signatures_follow_preview_mechanism_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    def _signature(**kwargs):
        mechanism_text = str(kwargs.get("mechanism_text") or "")
        if mechanism_text:
            return f"text:{mechanism_text}"
        identity = dict(kwargs.get("simulation_identity") or {})
        return f"id:{identity.get('param_fingerprint')}"

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._parse_sim_time_seconds.return_value = 10.0
    mw._simulation_param_fingerprints = {"id1": "params-a", "id2": "params-b"}
    mw._slider_overrides = {"k1": 2.0}
    mw.apply_overrides_to_text = MagicMock(
        side_effect=lambda text, *, set_id=None: f"{text}\n# preview {set_id} k={mw._slider_overrides['k1']}"
    )
    mw.apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text, *, set_id=None: str(text))

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        _signature,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 2,
    )
    controller._start_parallel_batch_simulations = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=7,
        batch_rows=[0, 1],
        reuse_parallel_lane_pool=False,
    )
    first_state = controller.batch_context_owner.execution_payload_state()

    mw._slider_overrides = {"k1": 5.0}
    seed_batch_context(controller.batch_context_owner, active=False)
    controller._run_simulation_internal(
        fast_mode=True,
        request_id=8,
        batch_rows=[0, 1],
        reuse_parallel_lane_pool=False,
    )
    second_state = controller.batch_context_owner.execution_payload_state()

    first_text = dict(first_state.mechanism_text_by_set_id)
    second_text = dict(second_state.mechanism_text_by_set_id)
    first_sig = dict(first_state.mechanism_signature_by_set_id)
    second_sig = dict(second_state.mechanism_signature_by_set_id)

    assert first_sig == {
        set_id: f"text:{text}"
        for set_id, text in first_text.items()
    }
    assert second_sig == {
        set_id: f"text:{text}"
        for set_id, text in second_text.items()
    }
    assert first_sig["id1"] != second_sig["id1"]
    assert first_sig["id2"] != second_sig["id2"]


@pytest.mark.unit
def test_fast_mode_preview_owner_identity_uses_set_specific_staged_request_dsl(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_identity import contained_simulation_owner_identity
    from kindred.core.simulation_plan import SimulationPlan

    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._parse_sim_time_seconds.return_value = 10.0
    mw._simulation_param_fingerprints = {"id1": "params-a"}
    mw._slider_overrides = {"k1": 2.0}
    mw.apply_overrides_to_text = MagicMock(
        side_effect=lambda text, *, set_id=None: "reaction: A -> C; k=2"
    )
    mw.apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text, *, set_id=None: str(text))

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **kwargs: str(kwargs.get("mechanism_text") or ""),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(
        fast_mode=True,
        request_id=9,
        batch_rows=[0],
        reuse_parallel_lane_pool=False,
    )

    plan_payload = controller.batch_context_owner.simulation_plan_payload_for_set("id1")
    plan = SimulationPlan.from_payload(plan_payload)
    request_text = plan.to_execution_request().mechanism_text
    owner_identity = dict(plan.metadata["contained_owner_identity"])
    expected_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text=request_text,
        solver_config=plan.to_execution_request().solver_config,
        t_end=10.0,
        set_id="id1",
        parameter_names=["k1"],
        simulation_identity=plan.simulation_identity_payload(),
    )
    canonical_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; k=1",
        solver_config=plan.to_execution_request().solver_config,
        t_end=10.0,
        set_id="id1",
        parameter_names=["k1"],
        simulation_identity=plan.simulation_identity_payload(),
    )

    assert request_text == "reaction: A -> C; k=2"
    assert owner_identity == expected_identity
    assert owner_identity != canonical_identity


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_passes_scalar_override_as_request_parameter_when_bindings_cannot_apply(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\n# Algebra\nparam a = 5\n")
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"a": 2.0}
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda text: str(text))
    mw._prepare_slider_runtime = MagicMock(return_value=object())
    mw._apply_slider_overrides_to_bindings = MagicMock(return_value=False)
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=8, batch_rows=[0], reuse_parallel_lane_pool=False)

    execution_state = controller.batch_context_owner.execution_payload_state()
    assert "param a = 5" in execution_state.mechanism_text_by_set_id["id1"]
    plan = SimulationPlan.from_payload(execution_state.simulation_plan_by_set_id["id1"])
    request_payload = plan.to_execution_request().to_payload()
    assert request_payload["parameter_overrides"] == {"a": pytest.approx(2.0)}

@pytest.mark.unit
def test_run_simulation_internal_explicit_run_uses_overlay_cache_token(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "baseline-cache"
    mw._simulation_schema_id = "schema-explicit"
    mw._simulation_param_fingerprints = {"id1": "params-id1"}
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="set:id1|A=2.5")

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    solver_cfg = controller.batch_context_owner.execution_payload_state().solver_config
    expected = SimulationScopeIdentity.build(
        queue_ids=["id1"],
        identity_by_set_id={
            "id1": SimulationIdentity.build(
                schema_id="schema-explicit",
                param_fingerprint="",
                solver_config=dict(solver_cfg),
                t_end=10.0,
                preview_batch_cache_token="",
                execution_flags=(),
            )
        },
    ).cache_key()
    assert controller.batch_context_owner.completion_cache_key() == expected
    assert controller.batch_cache.active_cache_key == expected
    assert controller.batch_cache.active_cache_preview_token is None
    assert controller.batch_cache.active_cache_preview_scope_set_ids is None
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)
    assert mw.preview_batch_cache_token.call_args_list == []
    controller._start_next_batch_simulation.assert_called_once()

@pytest.mark.unit
def test_run_simulation_internal_explicit_cache_key_ignores_non_primary_set_fingerprint_changes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="")
    mw._simulation_schema_id = "schema-explicit"
    mw._simulation_param_fingerprints = {"id1": "params-id1", "id2": "params-id2a"}

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0, 1], reuse_parallel_lane_pool=False)
    first_key = str(controller.batch_context_owner.completion_cache_key())

    mw._batch_set_id_for_row.side_effect = ["id1", "id2", "id1", "id2"]
    mw._simulation_param_fingerprints = {"id1": "params-id1", "id2": "params-id2b"}
    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0, 1], reuse_parallel_lane_pool=False)
    second_key = str(controller.batch_context_owner.completion_cache_key())

    assert first_key == second_key

@pytest.mark.unit
def test_run_simulation_internal_baseline_explicit_run_leaves_overlay_cache_token_empty(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "baseline-cache"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(return_value="")

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert isinstance(controller.batch_context_owner.completion_cache_key(), str)
    assert controller.batch_context_owner.completion_cache_key() != "baseline-cache"
    assert controller.batch_cache.active_cache_key == controller.batch_context_owner.completion_cache_key()
    assert controller.batch_cache.active_cache_preview_token is None
    assert controller.batch_cache.active_cache_preview_scope_set_ids is None
    assert controller.batch_cache.active_cache_valid_set_ids == ("id1",)
    assert mw.preview_batch_cache_token.call_args_list == []
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_run_simulation_internal_fast_mode_seeds_active_preview_scope_set_ids(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "preview-cache"
    mw._mechanism_editor = _MechanismEditor()
    mw._parse_sim_time_seconds.return_value = 10.0
    mw.preview_batch_cache_token = MagicMock(side_effect=["preview:id1", "preview:id2"])
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 1.0}, {"A": 2.0}])

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=1, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    assert isinstance(controller.batch_cache.active_preview_cache_key, str)
    assert controller.batch_cache.active_preview_cache_key
    assert controller.batch_cache.active_preview_scope_set_ids == ("id1", "id2")
    controller._start_next_batch_simulation.assert_called_once()


@pytest.mark.unit
def test_start_parallel_batch_simulations_marks_only_primary_explicit_result_for_mechanism_payload(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 2.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 1.0}, {"A": 2.0}])
    pool = _RecordingLanePool(submitted)
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda _message, _exc: None,
    )
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", run_id=3, request_id=11, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", mechanism_signature_by_set_id={"id1": "sig-1", "id2": "sig-2"}, solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={"id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}), "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", initials={"A": 2.0})})

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert len(submitted) == 2
    by_set_id = {str(task["set_id"]): task for task in submitted}
    assert by_set_id["id1"]["include_mechanism_in_result_payload"] is True
    assert by_set_id["id2"]["include_mechanism_in_result_payload"] is False

@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_uses_set_specific_prepared_payload_and_mechanism_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    request_payload = {
        "prepared_payload": {"version": 2, "prepared_for": "id2"},
        "initials": {"A": 1.5},
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=3",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=1, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], full_dsl="reaction: A -> B; k=1", mechanism_text_by_set_id={
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        }, mechanism_signature="sig", mechanism_signature_by_set_id={
            "id1": "sig-2",
            "id2": "sig-3",
        }, prepared=None, prepared_by_set_id={
            "id1": {"prepared_for": "id1"},
            "id2": {"prepared_for": "id2"},
        }, simulation_plan_by_set_id={
            "id2": SimulationPlan.from_execution_request(
                request_payload,
                execution_mode="preview",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload={"cache_key": "slider-cache"},
                metadata={"set_id": "id2", "set_name": "set2", "fast_mode": True},
            ).to_payload(),
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=7, cache_key="slider-cache", pending_init_seed={}, pending_init_applied=True)

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["prepared"] == {"version": 2, "prepared_for": "id2"}
    assert created["started"] is True
    worker = controller._simulation_worker
    worker_plan = SimulationPlan.from_payload(getattr(worker, "simulation_plan_payload", None))
    worker_request = worker_plan.to_execution_request().to_payload()
    assert worker_request["prepared_payload"] == {"version": 2, "prepared_for": "id2"}
    assert worker_request["initials"] == {"A": 1.5}
    assert worker_request["mechanism_text"] == "reaction: A -> B; k=3"
    assert getattr(worker, "_execution_request", None) is None

@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_does_not_borrow_batch_global_prepared_payload(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=1, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], full_dsl="reaction: A -> B; k=1", mechanism_text_by_set_id={
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        }, mechanism_signature="sig", mechanism_signature_by_set_id={
            "id1": "sig-2",
            "id2": "sig-3",
        }, prepared={"prepared_for": "id1"}, prepared_by_set_id={
            "id1": {"prepared_for": "id1"},
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=11, cache_key="slider-cache", pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={
            "id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", mechanism_text="reaction: A -> B; k=2", initials={"A": 1.0}, fast_mode=True, cache_key="slider-cache"),
            "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", mechanism_text="reaction: A -> B; k=3", initials={"A": 4.0}, fast_mode=True, cache_key="slider-cache"),
        })

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["initials"] == {"A": 4.0}
    assert created["started"] is True
    assert getattr(controller._simulation_worker, "_execution_request", None) is None

@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_uses_target_queue_preview_inputs(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["solver_config"] = dict(solver_config)
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=1, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], full_dsl="reaction: A -> B; k=1", mechanism_text_by_set_id={
            "id1": "reaction: A -> B; k=2",
            "id2": "reaction: A -> B; k=3",
        }, mechanism_signature="sig", mechanism_signature_by_set_id={
            "id1": "sig-2",
            "id2": "sig-3",
        }, prepared=None, prepared_by_set_id={}, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=12, cache_key="slider-cache", pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={
            "id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", mechanism_text="reaction: A -> B; k=2", initials={"A": 1.0}, fast_mode=True, cache_key="slider-cache"),
            "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", mechanism_text="reaction: A -> B; k=3", initials={"A": 4.0}, fast_mode=True, cache_key="slider-cache"),
        })

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["mechanism_text"] == "reaction: A -> B; k=3"
    assert created["initials"] == {"A": 4.0}
    assert created["solver_config"] == {"solver": "BDF"}
    assert created["started"] is True
    assert getattr(controller._simulation_worker, "_execution_request", None) is None

@pytest.mark.unit
def test_start_next_batch_simulation_explicit_run_uses_canonical_pending_init_seed(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["initials"] = dict(initials)
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 0.25}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, rows=[0], queue_ids=["id1"], queue_names=["randomname3"], full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=7, cache_key="explicit-cache", pending_init_seed={"randomname3": {"A": 1.0}}, pending_init_applied=False, simulation_plan_by_set_id={
        "id1": _test_simulation_plan_payload(set_id="id1", set_name="randomname3", initials={"A": 1.0}, cache_key="explicit-cache"),
    })

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["initials"] == {"A": 1.0}
    mw.preview_initials_for_row.assert_not_called()

@pytest.mark.unit
def test_start_parallel_batch_simulations_explicit_run_uses_canonical_pending_init_seed(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    mw._batch_initials_for_row.return_value = {"A": 0.25}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    pool = _RecordingLanePool(submitted)
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda _message, _exc: None,
    )
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id1"], queue_names=["randomname3"], run_id=3, request_id=11, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={"randomname3": {"A": 1.0}}, pending_init_applied=False, simulation_plan_by_set_id={
        "id1": _test_simulation_plan_payload(set_id="id1", set_name="randomname3", initials={"A": 1.0}, cache_key="explicit-cache"),
    })

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert submitted
    from kindred.core.simulation_plan import SimulationPlan

    submitted_request = SimulationPlan.from_payload(submitted[0]["simulation_plan"]).to_execution_request().to_payload()
    assert submitted_request["initials"] == {"A": 1.0}
    assert "initials" not in submitted[0]
    mw.preview_initials_for_row.assert_not_called()

@pytest.mark.unit
def test_parallel_batch_pool_settings_changed_shuts_down_idle_pool_immediately(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False):
            self.close_calls.append(bool(kill))

    fake = _FakeLanePool()
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: fake
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner)

    controller._parallel_batch_pool_settings_changed()

    assert fake.close_calls == [False]
    assert not controller.parallel_batch.has_lane_pool()

@pytest.mark.unit
def test_parallel_batch_pool_settings_changed_defers_shutdown_until_parallel_completion(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, max_lanes: int) -> None:
            self.max_lanes = int(max_lanes)
            self.ready_lane_count = int(max_lanes)
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False):
            self.close_calls.append(bool(kill))

    current = _FakeLanePool(2)
    created: list[tuple[int, bool, _FakeLanePool]] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool(max_lanes)
        created.append((int(max_lanes), bool(limit_blas_threads), pool))
        return pool

    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: current
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    controller.parallel_batch.lane_pool_factory = _factory
    controller.parallel_batch.max_parallel_workers = 6
    controller.parallel_batch.begin_run(
        run_id=3,
        request_id=11,
        fast_mode=False,
        queue_ids=["id1"],
        queue_names=["set1"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=60.0,
        cache_key="cache-key",
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, keep_lane_pool_alive=True, queue_ids=["id1"], queue_names=["set1"], completed_set_ids=[], total=1, fast_mode=False, primary_set_id="other-id")
    controller._active_run_id = 3
    controller.run_state.latest_sim_request_id = 11
    mw._batch_current_row.return_value = None
    monkeypatch.setattr(controller._result_materialization_owner, "resolve_completion_mechanism", MagicMock(return_value=None))
    monkeypatch.setattr(
        controller._result_materialization_owner,
        "update_primary_result_materialization_contract",
        MagicMock(return_value=False),
    )

    controller._parallel_batch_pool_settings_changed()

    assert controller.parallel_batch.lane_pool_token() == id(current)
    assert controller.parallel_batch.is_pool_stale is True
    assert current.close_calls == []

    _complete_with_callback_identity(
        controller,
        {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.asarray([[1.0, 1.0]], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "",
            "solver_config": {},
            "fallback_occurred": False,
            "fallback_message": None,
        },
        run_id=3,
        fast_mode=False,
        request_id=11,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="cache-key",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert current.close_calls == [False]
    assert not controller.parallel_batch.has_lane_pool()

    recreated = controller.parallel_batch.ensure_lane_pool(max_lanes=6)

    assert created == [(6, True, recreated)]
    assert controller.parallel_batch.lane_pool_token() == id(recreated)


@pytest.mark.unit
def test_parallel_batch_pool_settings_changed_defers_shutdown_for_superseded_inflight_work(
    mw: _FakeMainWindow, controller: SimulationController
):
    started = threading.Event()
    release = threading.Event()

    class _BlockingLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            started.set()
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id="lane",
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"set_id": set_id},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            release.set()

    pool = _BlockingLanePool()
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller.parallel_batch.ensure_lane_pool(max_lanes=1)
    controller.parallel_batch.begin_run(
        run_id=3,
        request_id=11,
        fast_mode=False,
        queue_ids=["id1"],
        queue_names=["set1"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=60.0,
        cache_key="cache-key",
    )
    handle = controller.parallel_batch.submit_task(
        {"value": 1},
        set_id="id1",
        set_name="set1",
        callback_identity=_capture_callback_identity(
            controller,
            run_id=3,
            request_id=11,
            fast_mode=False,
            batch_set="set1",
            batch_set_id="id1",
            cache_key="cache-key",
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
        ),
    )
    assert started.wait(timeout=1.0)

    cancelled, running = controller.parallel_batch.soft_supersede()
    controller._parallel_batch_pool_settings_changed()

    assert (cancelled, running) == (0, 1)
    assert controller.parallel_batch.has_active_requests()
    assert controller.parallel_batch.is_pool_stale is True
    assert pool.close_calls == []

    release.set()
    handle.join(timeout=1.0)


@pytest.mark.unit
def test_ensure_parallel_batch_runtime_ready_only_once(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    created: list[tuple[int, bool]] = []
    created_pools: list[_FakeLanePool] = []

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, max_lanes: int) -> None:
            self.max_lanes = int(max_lanes)
            self.warm_calls: list[tuple[int, bool]] = []

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            return None

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            self.warm_calls.append((int(max_lanes), bool(wait)))

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        pool = _FakeLanePool(max_lanes)
        created_pools.append(pool)
        return pool

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.lane_pool_factory = _factory
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)
    first_token = controller.parallel_batch.lane_pool_token()
    controller.ensure_parallel_batch_runtime_ready()
    controller._parallel_batch_pool_settings_changed()
    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert created == [(3, True), (3, True)]
    assert controller.parallel_batch.has_lane_pool()
    assert controller.parallel_batch.lane_pool_token() != first_token
    assert created_pools[-1].warm_calls[:1] == [(3, True)]


@pytest.mark.unit
def test_ensure_parallel_batch_runtime_ready_defaults_to_nonblocking_warm(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    created: list[tuple[int, bool]] = []
    created_pools: list[_FakeLanePool] = []

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, max_lanes: int) -> None:
            self.max_lanes = int(max_lanes)
            self.warm_calls: list[tuple[int, bool]] = []

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            return None

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            self.warm_calls.append((int(max_lanes), bool(wait)))

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        pool = _FakeLanePool(max_lanes)
        created_pools.append(pool)
        return pool

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.lane_pool_factory = _factory
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert created == [(3, True)]
    assert controller.parallel_batch.has_lane_pool()
    assert created_pools[-1].warm_calls[:1] == [(3, True)]


@pytest.mark.unit
def test_ensure_parallel_batch_runtime_ready_retries_after_failure(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    attempts: list[tuple[int, bool]] = []
    recorded: list[tuple[str, str]] = []

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, max_lanes: int) -> None:
            self.max_lanes = int(max_lanes)
            self.ready_lane_count = int(max_lanes)

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            return None

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        attempts.append((int(max_lanes), bool(limit_blas_threads)))
        if len(attempts) == 1:
            raise RuntimeError("factory boom")
        return _FakeLanePool(max_lanes)

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.lane_pool_factory = _factory
    controller.parallel_batch.record_nonfatal_exception = _record
    controller._record_nonfatal_exception = _record
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert attempts == [(3, True)]
    assert not controller.parallel_batch.has_lane_pool()
    assert recorded == [("Failed to create batch lane pool", "factory boom")]

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert attempts == [(3, True), (3, True)]
    assert controller.parallel_batch.has_lane_pool()
    assert controller.parallel_batch.has_ready_lane_pool(max_lanes=3)


@pytest.mark.unit
def test_ensure_parallel_batch_runtime_ready_retries_after_warm_failure(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    attempts: list[tuple[int, bool]] = []
    recorded: list[tuple[str, str]] = []
    created_pools: list[_FakeLanePool] = []

    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self, max_lanes: int, *, fail_warm: bool) -> None:
            self.max_lanes = int(max_lanes)
            self.fail_warm = bool(fail_warm)
            self.close_calls: list[bool] = []
            self.warm_calls: list[tuple[int, bool]] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            self.warm_calls.append((int(max_lanes), bool(wait)))
            if self.fail_warm:
                raise RuntimeError("warm boom")

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        attempts.append((int(max_lanes), bool(limit_blas_threads)))
        pool = _FakeLanePool(max_lanes, fail_warm=len(attempts) == 1)
        created_pools.append(pool)
        return pool

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.lane_pool_factory = _factory
    controller.parallel_batch.record_nonfatal_exception = _record
    controller._record_nonfatal_exception = _record
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert attempts == [(3, True)]
    assert not controller.parallel_batch.has_lane_pool()
    assert recorded == [("Failed to warm batch lane pool", "warm boom")]

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    assert attempts == [(3, True), (3, True)]
    assert controller.parallel_batch.has_lane_pool()
    assert created_pools[-1].warm_calls[:1] == [(3, True)]


@pytest.mark.unit
def test_ensure_parallel_batch_runtime_ready_factory_failure_records_once(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    recorded: list[tuple[str, str]] = []

    def _factory(max_lanes: int, _limit_blas_threads: bool) -> object:
        _ = max_lanes
        raise RuntimeError("factory boom")

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    controller.parallel_batch.max_parallel_workers = 5
    controller.parallel_batch.lane_pool_factory = _factory
    controller.parallel_batch.record_nonfatal_exception = _record
    controller._record_nonfatal_exception = _record
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 4)

    controller.ensure_parallel_batch_runtime_ready()

    assert not controller.parallel_batch.has_lane_pool()
    assert recorded == [("Failed to create batch lane pool", "factory boom")]


@pytest.mark.unit
def test_start_parallel_batch_uses_prewarmed_lane_pool_without_blocking_warm(
    mw: _FakeMainWindow, controller: SimulationController
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    class _WarmLedgerLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.warm_calls: list[dict[str, object]] = []
            self.run_calls: list[dict[str, object]] = []
            self.close_calls: list[bool] = []
            self.ready_lane_count = 0

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            self.warm_calls.append({"max_lanes": int(max_lanes), "wait": bool(wait)})
            self.ready_lane_count = int(max_lanes)

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            self.run_calls.append(
                {
                    "task": dict(task or {}),
                    "run_id": int(run_id),
                    "request_id": int(request_id),
                    "set_id": str(set_id),
                    "active_timeout_s": float(active_timeout_s),
                }
            )
            return _lane_outcome(str(set_id), run_id=int(run_id), request_id=int(request_id))

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    pool = _WarmLedgerLanePool()
    controller.parallel_batch.max_parallel_workers = 2
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool

    controller.ensure_parallel_batch_runtime_ready()
    assert controller.parallel_batch_runtime_readiness_owner.wait_for_background_warm(timeout_s=1.0)

    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["a", "b"], queue_names=["A", "B"], run_id=10, request_id=20, effective_workers=2, full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=1.0, fast_mode=False, keep_lane_pool_alive=True, cache_key="cache-a", simulation_plan_by_set_id={
        "a": _test_simulation_plan_payload(set_id="a", set_name="A", cache_key="cache-a"),
        "b": _test_simulation_plan_payload(set_id="b", set_name="B", cache_key="cache-a"),
    }, mechanism_text_by_set_id={}, simulation_identity_by_set_id={})

    controller._start_parallel_batch_simulations()
    controller.parallel_batch.join_active_requests(timeout_s=1.0)

    assert pool.warm_calls == [{"max_lanes": 2, "wait": True}]
    assert len(pool.run_calls) == 2
    assert pool.close_calls == []


@pytest.mark.unit
def test_start_parallel_batch_does_not_submit_until_lane_pool_is_ready(
    mw: _FakeMainWindow, controller: SimulationController, monkeypatch
):
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    class _WarmLedgerLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.warm_calls: list[dict[str, object]] = []
            self.run_calls: list[dict[str, object]] = []
            self.ready_lane_count = 0

        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            self.warm_calls.append({"max_lanes": int(max_lanes), "wait": bool(wait)})
            if bool(wait):
                self.ready_lane_count = int(max_lanes)

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            self.run_calls.append(
                {
                    "task": dict(task or {}),
                    "run_id": int(run_id),
                    "request_id": int(request_id),
                    "set_id": str(set_id),
                    "active_timeout_s": float(active_timeout_s),
                }
            )
            return _lane_outcome(str(set_id), run_id=int(run_id), request_id=int(request_id))

        def close(self, *, kill: bool = False) -> None:
            _ = kill

    pool = _WarmLedgerLanePool()
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = lambda row: f"id{int(row) + 1}"
    controller.parallel_batch.max_parallel_workers = 2
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    monkeypatch.setattr(controller, "_ensure_selected_run_runtime_warming", MagicMock())

    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    assert controller.parallel_batch.has_ready_lane_pool(max_lanes=2) is False

    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0, 1], queue_ids=["a", "b"], queue_names=["A", "B"], run_id=10, request_id=20, effective_workers=2, full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=1.0, fast_mode=False, keep_lane_pool_alive=True, cache_key="cache-a", simulation_plan_by_set_id={
        "a": _test_simulation_plan_payload(set_id="a", set_name="A", cache_key="cache-a"),
        "b": _test_simulation_plan_payload(set_id="b", set_name="B", cache_key="cache-a"),
    }, mechanism_text_by_set_id={}, simulation_identity_by_set_id={})

    controller._start_parallel_batch_simulations()
    controller.parallel_batch.join_active_requests(timeout_s=1.0)

    assert pool.run_calls == []
    assert "runtime" in mw._status_label.text.lower()
    assert mw._run_button_requested_enabled is True
    assert mw.run_button_is_enabled() is False
    assert mw._runtime_availability_refresh_requests == 1
    pool.ready_lane_count = 2
    controller._run_simulation = MagicMock()

    controller._retry_pending_run_after_runtime_ready()

    assert controller._pending_run_after_runtime_ready.active is False
    controller._run_simulation.assert_called_once()


@pytest.mark.unit
def test_poll_parallel_batch_outcomes_shuts_down_stale_pool_after_active_requests_drain(
    mw: _FakeMainWindow, controller: SimulationController
):
    class _FakeLanePool(_ProtocolLanePool):
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False):
            self.close_calls.append(bool(kill))

    timer = MagicMock()
    timer.isActive.return_value = True
    controller._batch_completion_poll_timer = timer
    pool = _FakeLanePool()
    controller.parallel_batch.lane_pool_factory = lambda _max_lanes, _limit_blas_threads: pool
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    controller.parallel_batch.mark_pool_stale()
    seed_batch_context(controller.batch_context_owner, active=False, parallel=False)

    controller._poll_parallel_batch_completions()

    assert pool.close_calls == [False]
    assert not controller.parallel_batch.has_lane_pool()
    assert timer.stop.called

@pytest.mark.unit
def test_start_next_batch_simulation_invalid_initials_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_initials_for_row.side_effect = ValueError("bad initials")
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, rows=[0], queue_ids=["id1"], queue_names=["set1"], full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=7, cache_key="explicit-cache", pending_init_seed={}, pending_init_applied=True)

    controller._start_next_batch_simulation()

    assert warned == [("Invalid Initial Conditions", "Set 'set1' has invalid initial conditions:\n\nbad initials")]
    mw._batch_model.validate_rows.assert_called_once_with([0])
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    assert _batch_policy_context(controller).pending_init_applied is False

@pytest.mark.unit
def test_start_parallel_batch_simulations_invalid_initials_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    class _NoSubmitLanePool(_ProtocolLanePool):
        def run(self, *_args, **_kwargs):
            raise AssertionError("run should not be reached when initials are invalid")

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            return

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_initials_for_row.side_effect = ValueError("bad initials")
    _install_ready_batch_lane_pool(controller, _NoSubmitLanePool(), max_lanes=2)
    controller._shutdown_batch_lane_pool = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id1"], queue_names=["set1"], run_id=3, request_id=11, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={}, pending_init_applied=True)

    controller._start_parallel_batch_simulations()

    assert warned == [("Invalid Initial Conditions", "Set 'set1' has invalid initial conditions:\n\nbad initials")]
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._shutdown_batch_lane_pool.assert_called_once_with(force_terminate=True)
    assert _batch_policy_context(controller).pending_init_applied is False

@pytest.mark.unit
def test_run_simulation_internal_aborts_and_unlocks_on_invalid_batch_rows(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1\ninitial: A=1")
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str):
        warned.append((str(title), str(text)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda t: t)
    mw._apply_overrides_to_state_network_dsl = MagicMock(side_effect=lambda t: t)
    mw._parse_sim_time_seconds.return_value = 10.0

    mw._batch_model.validate_rows.return_value = {(0, "A")}

    class _MechTmp:
        def species_names(self):
            return ["A", "B"]

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", lambda *_a, **_k: _MechTmp())
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({"set1": {"A": 1.0}}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    controller._simulation_running = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pending_init_applied=True)
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid Initial Conditions"
    assert controller._simulation_running is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    assert _batch_policy_context(controller).pending_init_applied is False
    controller._start_next_batch_simulation.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_preview_mode_caps_points(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

        def slider_points_value(self) -> int:
            return 500

        def slider_solver_value(self) -> str:
            return "Radau"

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._mechanism_editor = _MechanismEditor()
    mw._initial_solver = "BDF"
    mw._initial_rtol = 1e-6
    mw._initial_atol = 1e-12
    mw._parse_sim_time_seconds.return_value = 10.0
    mw._slider_drag_active = True
    mw._last_slider_change_name = "Keq12"
    mw._batch_store.visible_species.return_value = ["A"]

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.migrate_reaction_dsl_initial_concentration_sets",
        lambda text, default_set_name="set1": ({}, text),
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=True, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    solver_cfg = controller.batch_context_owner.execution_payload_state().solver_config
    assert solver_cfg["solver"] == "Radau"
    assert int(solver_cfg["grid"]["N"]) <= 120
    assert int(solver_cfg["grid"]["N"]) >= 50

@pytest.mark.unit
def test_preview_ownership_same_membership_reordering_does_not_bump_epoch(controller: SimulationController):
    first = controller._claim_preview_ownership(
        request_id=7,
        target_set_ids=("id2", "id1"),
    )

    second = controller._set_preview_ownership(
        request_id=7,
        target_set_ids=("id1", "id2"),
    )

    assert first.epoch == second.epoch
    assert second.target_set_ids == ("id1", "id2")


@pytest.mark.unit

@pytest.mark.unit
def test_run_simulation_internal_invalid_t_end_does_not_schedule_pending_slider_replay_after_preflight_abort(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []
    scheduled: list[object] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._latest_sim_request_id = 4
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 3
    controller._pending_slider_target_set_ids = ("id1",)
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid t_end"
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 3
    assert controller._pending_slider_target_set_ids == ("id1",)
    assert scheduled == []
    controller._start_next_batch_simulation.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_invalid_t_end_reinvalidates_preserved_pending_init_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1\ninitial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid t_end"
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_invalidate_preserved_pending_init_results_after_failed_run_honors_explicit_flag_without_context(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    seed_batch_context(controller.batch_context_owner)

    controller._invalidate_preserved_pending_init_results_after_failed_run(
        pending_init_applied=True,
        ctx=None,
    )

    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()


@pytest.mark.unit
def test_run_simulation_internal_no_mechanism_after_pending_init_migration_reinvalidates_preserved_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "initial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "No Mechanism"
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._start_next_batch_simulation.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_fast_no_mechanism_clears_preview_ownership_and_pending_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"

    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 9
    controller._pending_slider_target_set_ids = ("id1",)

    controller._run_simulation_internal(fast_mode=True, request_id=9, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert controller.run_state.preview_ownership.request_id is None
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None
    assert controller._pending_slider_target_set_ids == ()

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit
def test_explicit_run_worker_error_reinvalidates_preserved_pending_init_results(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1\ninitial: A=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    _error_with_callback_identity(
        controller,
        {"kind": "simulation_error", "message": "ode build failed"},
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit

@pytest.mark.unit
def test_on_simulation_complete_updates_cache_and_marks_pending_init_applied(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    mw._slider_triggered_simulation = True
    controller._queue_slider_plot_update = MagicMock()
    mw._mechanism_editor = MagicMock()
    mw._mechanism_editor._reactions_text = MagicMock()

    mw._batch_store.ensure_set = MagicMock(return_value=0)
    mw._batch_store.set_value = MagicMock()

    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_names=["set1"], queue_ids=["id1"], cache_key="ck", pending_init_seed={"set1": {"A": 1.0}}, pending_init_rewrite="reaction: A -> B; k=1", pending_init_applied=False, primary_set_id="id1")

    result = {
        "t": np.linspace(0.0, 1.0, 3),
        "Y": np.asarray([[1.0, 0.5, 0.1], [0.0, 0.5, 0.9]], dtype=float),
        "species_names": ["A", "B"],
        "mechanism": object(),
        "mechanism_text": "reaction: A -> B; k=1",
        "solver_config": {"solver": "Radau", "temperature_K": 298.15},
        "algebra_scalars": {},
        "algebra_errors": [],
    }

    _complete_with_callback_identity(
        controller,
        result,
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    payload = controller.batch_cache.entry_for_set(cache_key="ck", set_id="id1", is_preview=False).entry
    assert isinstance(payload, dict)
    assert np.allclose(np.asarray(payload.get("t")), result["t"])
    assert _batch_policy_context(controller).pending_init_applied is True
    mw._arm_pending_init_result_invalidation_guard.assert_called_once_with(rewrite="reaction: A -> B; k=1")
    controller._queue_slider_plot_update.assert_called_once()


@pytest.mark.unit
def test_on_simulation_complete_writes_cache_identity_from_simulation_plan(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    solver_config = {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15}
    plan_identity = SimulationIdentity.build(
        schema_id="schema-from-plan",
        param_fingerprint="plan-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()
    stale_context_identity = SimulationIdentity.build(
        schema_id="schema-from-context",
        param_fingerprint="context-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()
    plan_payload = SimulationPlan.from_execution_request(
        {
            "prepared_payload": None,
            "initials": {"A": 1.0},
            "t_span": (0.0, 1.0),
            "solver_config": solver_config,
            "mechanism_text": "reaction: A -> B; k=1",
            "simulation_identity": plan_identity,
        },
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "ck", "simulation_identity": plan_identity},
        cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": ["id1"]},
    ).to_payload()

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._queue_slider_plot_update = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_names=["set1"], queue_ids=["id1"], cache_key="ck", primary_set_id="id1", pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={"id1": plan_payload}, simulation_identity_by_set_id={"id1": stale_context_identity})

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        simulation_identity=plan_identity,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    payload = controller.batch_cache.entry_for_set(cache_key="ck", set_id="id1", is_preview=False).entry
    assert isinstance(payload, dict)
    assert payload["simulation_identity"] == plan_identity


@pytest.mark.unit
def test_on_simulation_complete_without_batch_set_id_does_not_fallback_to_plan_or_context_set(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    solver_config = {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15}
    plan_identity = SimulationIdentity.build(
        schema_id="schema-from-plan",
        param_fingerprint="fallback-plan-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()
    stale_context_identity = SimulationIdentity.build(
        schema_id="schema-from-context",
        param_fingerprint="fallback-context-fingerprint",
        solver_config=solver_config,
        t_end=1.0,
    ).to_payload()
    plan_payload = SimulationPlan.from_execution_request(
        {
            "prepared_payload": None,
            "initials": {"A": 1.0},
            "t_span": (0.0, 1.0),
            "solver_config": solver_config,
            "mechanism_text": "reaction: A -> B; k=1",
            "simulation_identity": plan_identity,
        },
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "ck", "simulation_identity": plan_identity},
        cache_scope_payload={"scope_identity": {"schema_id": "scope"}, "queue_ids": ["id1"]},
    ).to_payload()

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._queue_slider_plot_update = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, queue_names=["set1"], queue_ids=["id1"], cache_key="ck", primary_set_id="id1", pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={"id1": plan_payload}, simulation_identity_by_set_id={"id1": stale_context_identity})

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set=None,
        batch_set_id=None,
        cache_key="ck",
        simulation_identity=plan_identity,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    payload = controller.batch_cache.entry_for_set(cache_key="ck", set_id="id1", is_preview=False).entry
    assert payload is None


@pytest.mark.unit
def test_on_simulation_complete_uses_truthful_scipy_fallback_warning_text(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    warnings: list[tuple[str, str]] = []

    def _fake_warning(_parent, title: str, message: str):
        warnings.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fake_warning)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2

    result = _successful_result_payload()
    result["fallback_occurred"] = True
    result["fallback_message"] = "BDF failed; succeeded with Radau"

    _complete_with_callback_identity(
        controller,
        result,
        run_id=2,
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert warnings == [("Solver fallback", warnings[0][1])]
    message = warnings[0][1]
    assert "BDF failed; succeeded with Radau" in message
    assert "alternative stiff SciPy solver" in message
    assert "RK4" not in message
    assert "fixed-step" not in message

@pytest.mark.unit
def test_on_simulation_error_cancelled_schedules_pending_slider(monkeypatch, mw: _FakeMainWindow, controller: SimulationController):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    scheduled = {}

    def _fake_single_shot(_ms: int, fn: Callable[[], Any]) -> None:
        scheduled["fn"] = fn

    monkeypatch.setattr(QtCore.QTimer, "singleShot", _fake_single_shot)

    controller._latest_sim_request_id = 1
    controller._active_run_id = 2
    controller._pending_slider_simulation = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("id1",),
    )

    _error_with_callback_identity(
        controller,
        {"kind": "cancelled", "message": "Simulation cancelled by user"},
        run_id=2,
        fast_mode=True,
        request_id=1,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    assert "fn" in scheduled
    mw._variable_update_timer.stop.assert_called_once_with()
    mw._species_slider_update_timer.stop.assert_called_once_with()

@pytest.mark.unit
def test_on_simulation_error_non_cancelled_explicit_requeues_preserved_pending_slider_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=5)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._last_slider_change_name = "k1"
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 4
    controller._pending_slider_target_set_ids = ("id2",)

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 4
    assert controller._pending_slider_target_set_ids == ("id2",)
    critical.assert_called_once()
    controller.run_simulation_internal = MagicMock()
    scheduled[0]()
    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["batch_rows"] == [1]


@pytest.mark.unit
def test_on_simulation_error_ignores_stale_runtime_input_epoch_before_ui_publication(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)

    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._authoritative_runtime_input_epoch = 2
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_epoch=1,
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
    )
    callback_identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)

    controller._on_simulation_error(
        "stale boom",
        callback_identity=callback_identity,
    )

    critical.assert_not_called()
    assert mw._status_label.text != "Simulation failed"
    assert _batch_policy_context(controller).active is True
    assert controller._simulation_running is True


@pytest.mark.unit
def test_completion_uses_captured_context_for_runtime_stale_after_context_turnover(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1},
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        simulation_identity={},
        preview_batch_cache_token="",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 2},
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    cache_truth = MagicMock()
    cache_entry = MagicMock()
    display = MagicMock()
    annotations = MagicMock()
    provenance = MagicMock()
    controller._cache_admin.publish_completion_cache_truth = cache_truth
    controller._cache_admin.publish_completion_cache = cache_entry
    controller.ui.results.publish_simulation_completion_result = display
    controller.ui.results.publish_completion_intervention_annotations = annotations
    controller.ui.provenance.publish_simulation_completion_provenance = provenance

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=identity,
    )

    cache_truth.assert_not_called()
    cache_entry.assert_not_called()
    display.assert_not_called()
    annotations.assert_not_called()
    provenance.assert_not_called()
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert callback_context.runtime_input_set_epoch_by_set_id == {"id1": 2}
    assert callback_context.stale_runtime_input_set_ids == ()


@pytest.mark.unit
def test_error_uses_captured_context_for_runtime_stale_after_context_turnover(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1},
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 2},
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )

    controller._on_simulation_error(
        "stale boom",
        callback_identity=identity,
    )

    critical.assert_not_called()
    assert mw._status_label.text != "Simulation failed"
    completion_state = controller.batch_context_owner.completion_state()
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert completion_state is not None
    assert completion_state.active is True
    assert callback_context.runtime_input_set_epoch_by_set_id == {"id1": 2}
    assert callback_context.stale_runtime_input_set_ids == ()


@pytest.mark.unit
def test_error_missing_callback_context_does_not_use_current_preview_owner_epoch(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 8
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = True
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=4,
        target_set_ids=("id2",),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=True,
        request_id=7,
        run_id=5,
        preview_owner_epoch=4,
        cache_key="preview-cache",
        queue_ids=["id2"],
        queue_names=["set2"],
    )
    controller._error_handling_owner._deps.freshness.assess_callback = MagicMock(
        side_effect=AssertionError("freshness must not be consulted without callback context")
    )

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_error(
            "stale boom",
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=True,
                request_id=7,
                owner_epoch=None,
                batch_set="set2",
                batch_set_id="id2",
                cache_key="preview-cache",
            ),
        ),

    controller._error_handling_owner._deps.freshness.assess_callback.assert_not_called()
    state = controller.batch_context_owner.completion_state()
    assert state is not None
    assert state.active is True


@pytest.mark.unit
def test_error_missing_callback_context_does_not_use_current_queue_or_stale_context(
    monkeypatch, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )
    controller._error_handling_owner._deps.freshness.assess_callback = MagicMock(
        side_effect=AssertionError("freshness must not be consulted without callback context")
    )

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_error(
            "boom",
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="current-cache",
            ),
        ),

    controller._error_handling_owner._deps.freshness.assess_callback.assert_not_called()


@pytest.mark.unit
def test_error_missing_callback_context_does_not_deactivate_current_batch_context(
    monkeypatch, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_error(
            "boom",
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="current-cache",
            ),
        ),

    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_error_missing_callback_context_does_not_deactivate_mismatched_current_run(
    monkeypatch, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_error(
            "boom",
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="captured-cache",
            ),
        ),

    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_error_malformed_callback_context_does_not_deactivate_current_run(
    monkeypatch, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )

    with pytest.raises(ValueError, match="callback_identity.callback_context"):
        controller._on_simulation_error(
            "boom",
            callback_identity=_malformed_callback_identity_without_context(
                run_id=5,
                fast_mode=False,
                request_id=7,
                owner_epoch=None,
                batch_set=None,
                batch_set_id=None,
                cache_key="current-cache",
            ),
        ),

    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_scalar_error_without_callback_identity_is_rejected_without_current_context_capture(
    monkeypatch, controller: SimulationController
):
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    controller._latest_sim_request_id = 7
    controller._active_run_id = 5
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller.ui.run_ui.set_run_button_enabled(False)
    controller.ui.run_ui.set_stop_button_enabled(True)
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=7,
        run_id=5,
        cache_key="current-cache",
        queue_ids=["current-id"],
        queue_names=["current-set"],
        pos=0,
    )

    with pytest.raises(TypeError, match="callback_identity"):
        controller._on_simulation_error(
            "boom",
        )

    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
    assert controller._simulation_running is True
    assert controller._slider_simulation_active is False
    assert controller.ui.run_ui.run_button_is_enabled() is False
    assert controller.ui.run_ui._stop_btn.isEnabled() is True


@pytest.mark.unit
def test_parallel_stale_runtime_completion_consumes_current_batch_after_progress_turnover(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 0, "id2": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 1},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=[],
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        simulation_identity={},
        preview_batch_cache_token="",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 2},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=["id1"],
        primary_set_id="id1",
    )
    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller.ui.results.publish_simulation_completion_result = MagicMock()

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=identity,
    )

    policy_context = _batch_policy_context(controller)
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert policy_context.active is False
    assert policy_context.completed_set_ids == ("id1", "id2")
    assert callback_context.stale_runtime_input_set_ids == ("id2",)
    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()


@pytest.mark.unit
def test_parallel_stale_runtime_error_consumes_current_batch_after_progress_turnover(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 0, "id2": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 1},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=[],
        primary_set_id="id1",
    )
    identity = _capture_callback_identity(
        controller,
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 2},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        completed_set_ids=["id1"],
        primary_set_id="id1",
    )

    controller._on_simulation_error(
        "stale boom",
        callback_identity=identity,
    )

    policy_context = _batch_policy_context(controller)
    callback_context = controller.batch_context_owner.callback_context_snapshot()
    assert policy_context.active is False
    assert policy_context.completed_set_ids == ("id1", "id2")
    assert callback_context.stale_runtime_input_set_ids == ("id2",)
    critical.assert_not_called()
    assert mw._status_label.text != "Simulation failed"


@pytest.mark.unit
def test_scoped_runtime_input_supersede_rejects_affected_completion_but_accepts_unaffected(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 0},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=0,
        total=2,
        primary_set_id="id1",
    )
    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.set_data.assert_not_called()
    assert "ck::id1" not in controller.batch_cache.result_cache
    state = controller.batch_context_owner.completion_state()
    assert state is not None
    assert state.active is True

    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 0},
        cache_key="ck",
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        rows=[0, 1],
        pos=1,
        total=2,
        primary_set_id="id2",
    )
    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.set_data.assert_called_once()
    assert "ck::id2" in controller.batch_cache.result_cache


@pytest.mark.unit
def test_scoped_runtime_input_supersede_rejects_completion_before_any_publication_surface(
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1},
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    cache_truth = MagicMock()
    cache_entry = MagicMock()
    display = MagicMock()
    annotations = MagicMock()
    provenance = MagicMock()
    controller._cache_admin.publish_completion_cache_truth = cache_truth
    controller._cache_admin.publish_completion_cache = cache_entry
    controller.ui.results.publish_simulation_completion_result = display
    controller.ui.results.publish_completion_intervention_annotations = annotations
    controller.ui.provenance.publish_simulation_completion_provenance = provenance

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    cache_truth.assert_not_called()
    cache_entry.assert_not_called()
    display.assert_not_called()
    annotations.assert_not_called()
    provenance.assert_not_called()


@pytest.mark.unit
def test_malformed_completion_payload_does_not_publish_cache_truth(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    cache_truth = MagicMock()
    mw.message_box_critical = MagicMock()
    controller._cache_admin.publish_completion_cache_truth = cache_truth

    _complete_with_callback_identity(
        controller,
        {"t": np.array([0.0])},
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    cache_truth.assert_not_called()


@pytest.mark.unit
def test_materialization_failure_does_not_publish_cache_truth(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    class _Mechanism:
        def species_names(self) -> list[str]:
            return ["A", "B"]

    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        cache_key="ck",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    cache_truth = MagicMock()
    mw.message_box_critical = MagicMock()
    mw._is_energy_mode_mechanism = MagicMock(side_effect=RuntimeError("materialization boom"))
    controller._cache_admin.publish_completion_cache_truth = cache_truth
    payload = _successful_result_payload()
    payload["mechanism"] = _Mechanism()

    _complete_with_callback_identity(
        controller,
        payload,
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    cache_truth.assert_not_called()
    mw.message_box_critical.assert_called_once()


@pytest.mark.unit
def test_scoped_runtime_input_supersede_rejects_affected_error_but_surfaces_unaffected(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    mw.message_box_critical = MagicMock()

    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 0},
    )
    _error_with_callback_identity(
        controller,
        "affected boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.message_box_critical.assert_not_called()
    assert _batch_policy_context(controller).active is True
    assert controller._simulation_running is True

    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 0},
    )
    _error_with_callback_identity(
        controller,
        "unaffected boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.message_box_critical.assert_called_once()


@pytest.mark.unit
def test_global_authoritative_supersede_rejects_unaffected_completion_and_error(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._authoritative_runtime_input_epoch = 2
    controller._authoritative_runtime_input_global_epoch = 2
    controller._authoritative_runtime_input_set_epoch_by_set_id = {}
    mw.message_box_critical = MagicMock()

    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=1,
        runtime_input_set_epoch_by_set_id={"id2": 0},
    )
    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.set_data.assert_not_called()
    assert "ck::id2" not in controller.batch_cache.result_cache

    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        request_id=5,
        runtime_input_global_epoch=1,
        runtime_input_set_epoch_by_set_id={"id2": 0},
    )
    _error_with_callback_identity(
        controller,
        "global stale boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_on_simulation_error_non_cancelled_explicit_replays_existing_owned_pending_slider_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock(return_value=QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", critical)
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, request_id=5)
    controller._simulation_running = True
    controller._slider_simulation_active = False
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._last_slider_change_name = "k1"
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("id2",)
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=7,
        epoch=3,
        target_set_ids=("id2",),
    )

    _error_with_callback_identity(
        controller,
        "boom",
        run_id=3,
        fast_mode=False,
        request_id=5,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    assert scheduled == [controller._run_simulation_from_slider]
    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller.run_state.preview_ownership.request_id == 7
    critical.assert_called_once()

@pytest.mark.unit
def test_on_simulation_error_surfaces_stack_trace_as_dialog_details_and_log(
    caplog, mw: _FakeMainWindow, controller: SimulationController
):
    critical = MagicMock()
    mw.message_box_critical = critical
    controller._latest_sim_request_id = 5
    controller._active_run_id = 3
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=5,
        epoch=1,
        target_set_ids=("id1",),
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=True, request_id=5)
    controller._simulation_running = True
    controller._slider_simulation_active = True

    stack_trace = "Traceback line 1\nTraceback line 2"

    with caplog.at_level("WARNING", logger="kindred.gui.controllers.simulation_controller"):
        _error_with_callback_identity(
            controller,
            {
                "kind": "simulation_error",
                "message": "solver blew up",
                "context": {"stack_trace": stack_trace},
            },
            run_id=3,
            fast_mode=True,
            request_id=5,
            callback_context=controller.batch_context_owner.callback_context_snapshot(),
        )

    critical.assert_called_once_with(
        "Simulation Error",
        "Simulation failed:\n\nsolver blew up",
        details=stack_trace,
    )
    assert "Traceback" not in critical.call_args.args[1]
    messages = [record.getMessage() for record in caplog.records]
    assert "Simulation error surfaced to UI: solver blew up" in messages
    assert stack_trace in messages
    assert mw._status_label.text == "Simulation failed"
    assert mw._sim_progress.value == 0
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False

@pytest.mark.unit
def test_consume_parallel_batch_outcome_error_payload_calls_on_error(controller: SimulationController):
    outcome = _lane_outcome(
        "sid",
        {
            "success": False,
            "error": {"kind": "simulation_error", "message": "solver blew up", "code": "E301"},
        },
        request_id=1,
    )

    _install_active_lane_outcomes(controller, {"sid": outcome}, set_names={"sid": "set1"})
    controller._error_handling_owner.handle_error = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="sid",
        outcome=outcome,
        run_id=1,
        request_id=1,
        fast_mode=False,
        source="scan",
    )

    assert ok is False
    controller._error_handling_owner.handle_error.assert_called_once()


@pytest.mark.unit
def test_parallel_batch_stale_runtime_input_error_is_consumed_without_failure_publication(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    outcome = _timeout_failure_outcome("id1")

    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 4, "id2": 0}
    controller._batch_cache.active_cache_key = "ck"
    controller._batch_cache.active_cache_valid_set_ids = ("id1", "id2")
    controller._batch_cache.active_cache_invalidated_set_ids = None
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=False, run_id=1, request_id=2, cache_key="ck", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], total=2, completed_set_ids=[], runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 0}, explicit_cache_valid_set_ids=("id1", "id2"), explicit_cache_invalidated_set_ids=None, pending_workspace_reset_set_ids=["id1", "id2"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2})
    _install_active_lane_outcomes(
        controller,
        {"id1": outcome},
        set_names={"id1": "set1"},
        callback_identities={
            "id1": _capture_callback_identity(
                controller,
                run_id=1,
                fast_mode=False,
                request_id=2,
                batch_set="set1",
                batch_set_id="id1",
                cache_key="ck",
                callback_context=controller.batch_context_owner.callback_context_snapshot(),
            )
        },
    )
    mw._dirty_state_generations = {"id1": 1, "id2": 2}
    controller._error_handling_owner.handle_error = MagicMock()
    mw.message_box_critical = MagicMock()
    mw._status_label.setText("Running 2 sets in parallel")

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id1",
        outcome=outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="test",
    )

    assert ok is True
    state = controller.batch_context_owner.completion_state()
    assert state is not None
    assert state.completed_set_ids == ("id1",)
    summary = controller.batch_context_owner.completion_summary()
    assert summary.failed_set_ids == ()
    assert summary.failed_errors == {}
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.pending_workspace_reset_set_ids == ("id2",)
    assert policy_context.pending_dirty_reset_generation_by_set_id == {"id2": 2}
    assert controller._batch_cache.active_cache_valid_set_ids == ("id1", "id2")
    assert controller._batch_cache.active_cache_invalidated_set_ids is None
    controller._error_handling_owner.handle_error.assert_not_called()
    mw.message_box_critical.assert_not_called()
    assert mw._status_label.text == "Running 2 sets in parallel"

    controller._active_run_id = 1
    controller._latest_sim_request_id = 2
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True

    _complete_with_callback_identity(
        controller,
        _successful_result_payload(),
        run_id=1,
        fast_mode=False,
        request_id=2,
        batch_set="set2",
        batch_set_id="id2",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id2"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id2"])
    pending_reset_state = controller.batch_context_owner.pending_dirty_reset_state()
    assert pending_reset_state.set_ids == ()
    assert pending_reset_state.generation_by_set_id == {}


@pytest.mark.unit
def test_parallel_batch_all_stale_runtime_input_callbacks_finish_run_cleanly(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    outcome = _timeout_failure_outcome("id1")

    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 4}
    controller._simulation_running = True
    controller._slider_simulation_active = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._sim_progress.setValue(42)
    mw._status_label.setText("Running 1 set in parallel")
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=False, run_id=1, request_id=2, cache_key="ck", queue_ids=["id1"], queue_names=["set1"], total=1, completed_set_ids=[], runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0})
    _install_active_lane_outcomes(
        controller,
        {"id1": outcome},
        set_names={"id1": "set1"},
        callback_identities={
            "id1": _capture_callback_identity(
                controller,
                run_id=1,
                fast_mode=False,
                request_id=2,
                batch_set="set1",
                batch_set_id="id1",
                cache_key="ck",
                callback_context=controller.batch_context_owner.callback_context_snapshot(),
            )
        },
    )
    controller.queue_pending_slider_preview_replay(target_set_ids=("id2",), request_id=9)
    controller._error_handling_owner.handle_error = MagicMock()
    mw.message_box_critical = MagicMock()

    ok = _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id1",
        outcome=outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="test",
    )

    assert ok is True
    assert _batch_policy_context(controller).active is False
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 0
    assert mw._status_label.text == "Ready"
    assert scheduled == [controller._run_simulation_from_slider]
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert _pending_slider_preview_launch(controller).request_id == 9
    controller._error_handling_owner.handle_error.assert_not_called()
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_parallel_batch_multiset_all_stale_runtime_input_callbacks_finish_run_cleanly(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    id1_outcome = _timeout_failure_outcome("id1")
    id2_outcome = _timeout_failure_outcome("id2")

    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 4, "id2": 4}
    controller._simulation_running = True
    controller._slider_simulation_active = False
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._sim_progress.setValue(42)
    mw._status_label.setText("Running 2 sets in parallel")
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, fast_mode=False, run_id=1, request_id=2, cache_key="ck", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], total=2, completed_set_ids=[], runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 0})
    _install_active_lane_outcomes(
        controller,
        {"id1": id1_outcome, "id2": id2_outcome},
        set_names={"id1": "set1", "id2": "set2"},
        callback_identities={
            "id1": _capture_callback_identity(
                controller,
                run_id=1,
                fast_mode=False,
                request_id=2,
                batch_set="set1",
                batch_set_id="id1",
                cache_key="ck",
                callback_context=controller.batch_context_owner.callback_context_snapshot(),
            ),
            "id2": _capture_callback_identity(
                controller,
                run_id=1,
                fast_mode=False,
                request_id=2,
                batch_set="set2",
                batch_set_id="id2",
                cache_key="ck",
                callback_context=controller.batch_context_owner.callback_context_snapshot(),
            ),
        },
    )
    controller._error_handling_owner.handle_error = MagicMock()
    mw.message_box_critical = MagicMock()

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id1",
        outcome=id1_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="test",
    ) is True

    assert _batch_policy_context(controller).active is True
    assert controller._simulation_running is True
    assert mw._status_label.text == "Running 2 sets in parallel"

    assert _consume_parallel_batch_outcome_for_test(
        controller,
        set_id="id2",
        outcome=id2_outcome,
        run_id=1,
        request_id=2,
        fast_mode=False,
        source="test",
    ) is True

    assert _batch_policy_context(controller).active is False
    policy_context = controller.batch_context_owner.completion_policy_context()
    summary = controller.batch_context_owner.completion_summary()
    assert policy_context is not None
    assert policy_context.completed_set_ids == ("id1", "id2")
    assert summary.failed_set_ids == ()
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 0
    assert mw._status_label.text == "Ready"
    controller._error_handling_owner.handle_error.assert_not_called()
    mw.message_box_critical.assert_not_called()


@pytest.mark.unit
def test_scoped_runtime_input_supersede_preserves_unaffected_serial_queue_tail(
    monkeypatch,
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    created: dict[str, object] = {}
    scheduled: list[object] = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    _install_recording_contained_worker(monkeypatch, created, controller)

    active_worker = _FakeWorker(running=True, wait_returns=True)
    active_worker._batch_set_id = "id1"
    active_worker._fast_mode = False
    controller._simulation_worker = active_worker
    controller._simulation_running = True
    controller._slider_simulation_active = False
    mw._batch_initials_for_row.side_effect = lambda row: {0: {"A": 1.0}, 1: {"A": 2.0}}[int(row)]
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, run_id=3, request_id=5, cache_key="ck", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], rows=[0, 1], pos=0, total=2, full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, completed_set_ids=[], pending_workspace_reset_set_ids=["id1", "id2"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2}, runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 0}, simulation_plan_by_set_id={
        "id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}, cache_key="ck"),
        "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", initials={"A": 2.0}, cache_key="ck"),
    })

    controller._supersede_active_work_for_authoritative_mechanism_transition(
        epoch=2,
        affected_set_ids=("id1",),
    )

    assert active_worker._cancelled is True
    assert _batch_policy_context(controller).active is True
    assert _batch_policy_context(controller).pos == 1
    completion_state = controller.batch_context_owner.completion_state()
    assert completion_state is not None
    assert completion_state.completed_set_ids == ("id1",)
    pending_reset_state = controller.batch_context_owner.pending_dirty_reset_state()
    assert pending_reset_state.set_ids == ("id2",)
    assert pending_reset_state.generation_by_set_id == {"id2": 2}
    assert scheduled == [controller._start_next_batch_simulation]

    scheduled[0]()

    assert created["started"] is True
    assert created["initials"] == {"A": 2.0}
    assert controller._simulation_worker._batch_set_id == "id2"


@pytest.mark.unit
def test_start_next_batch_simulation_stale_serial_tail_uses_completion_cleanup(
    mw: _FakeMainWindow,
    controller: SimulationController,
):
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id2": 4}
    controller._simulation_worker = _FakeWorker(running=False, wait_returns=True)
    controller._simulation_running = True
    controller._slider_simulation_active = True
    mw.set_slider_triggered_simulation(True)
    mw._run_btn.setEnabled(False)
    mw._stop_btn.setEnabled(True)
    mw._sim_progress.setValue(50)
    mw._status_label.setText("Completed set1 (1/2)")
    mw._dirty_state_generations = {"id1": 1, "id2": 2}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, fast_mode=False, run_id=3, request_id=5, cache_key="ck", queue_ids=["id1", "id2"], queue_names=["set1", "set2"], rows=[0, 1], pos=1, total=2, completed_set_ids=["id1"], pending_workspace_reset_set_ids=["id1", "id2"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2}, runtime_input_global_epoch=0, runtime_input_set_epoch_by_set_id={"id1": 0, "id2": 0})

    controller._start_next_batch_simulation()

    assert _batch_policy_context(controller).active is False
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.completed_set_ids == ("id1", "id2")
    pending_reset_state = controller.batch_context_owner.pending_dirty_reset_state()
    assert pending_reset_state.set_ids == ()
    assert pending_reset_state.generation_by_set_id == {}
    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    assert controller._simulation_running is False
    assert controller._slider_simulation_active is False
    assert mw.slider_triggered_simulation() is False
    assert mw._run_btn.isEnabled() is True
    assert mw._stop_btn.isEnabled() is False
    assert mw._sim_progress.value == 100
    assert mw._status_label.text == "Simulation complete"


@pytest.mark.unit
def test_has_running_workers_is_pure_query(controller: SimulationController):
    worker = _FakeWorker(running=False)
    controller._simulation_worker = worker
    controller._retained_simulation_workers = [worker]
    controller._delete_worker_if_stopped = MagicMock()

    assert controller._has_running_owned_simulation_workers() is False
    controller._delete_worker_if_stopped.assert_not_called()
    assert controller._simulation_worker is worker
    assert controller._retained_simulation_workers == [worker]

# ---------------------------------------------------------------------------
# Structured execution request: non-fast-mode regression tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_does_not_build_prepared_payloads(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """Explicit runs must stay canonical-only and avoid preview prepared payloads."""

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=1")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    created_runtimes: list[_FakeRuntime] = []

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        runtime = _FakeRuntime()
        created_runtimes.append(runtime)
        return runtime

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    # Explicit run: fast_mode=False
    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0], reuse_parallel_lane_pool=False)

    execution_state = controller.batch_context_owner.execution_payload_state()
    prepared_by_set_id = execution_state.prepared_by_set_id
    assert prepared_by_set_id == {}
    assert execution_state.prepared is None
    assert created_runtimes == []
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()

@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_builds_simulation_plans(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text(mechanism_text)
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        return _FakeRuntime()

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0], reuse_parallel_lane_pool=False)

    simulation_plan_by_set_id = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id
    assert "id1" in simulation_plan_by_set_id
    plan = SimulationPlan.from_payload(simulation_plan_by_set_id["id1"])
    request = plan.to_execution_request().to_payload()
    assert plan.execution_mode == "explicit"
    assert plan.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 1.0}
    assert tuple(request["t_span"]) == (0.0, 10.0)
    assert request["mechanism_text"] == mechanism_text
    assert request["intervention_schedule"]["instant_events"] == [
        {"op": "set", "species": "A", "time": 0.0, "value": 2.0}
    ]
    assert request["simulation_identity"]["schema_id"] != ""
    assert request["simulation_identity"]["param_fingerprint"] == ""
    mw.preview_initials_for_row.assert_not_called()
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()


@pytest.mark.unit
def test_run_simulation_internal_builds_plan_initials_from_materialization_owner(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan

    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    mw._parse_sim_time_seconds.return_value = 10.0
    controller._batch_dispatch_materialization_owner.materialize_initials = MagicMock(return_value={"A": 9.0})
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0], reuse_parallel_lane_pool=False)

    simulation_plan_by_set_id = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id
    request = SimulationPlan.from_payload(simulation_plan_by_set_id["id1"]).to_execution_request().to_payload()
    assert request["initials"] == {"A": 9.0}
    controller._batch_dispatch_materialization_owner.materialize_initials.assert_called_once_with(
        row=0,
        set_name="set1",
        fast_mode=False,
        pending_init_seed={},
        pending_init_applied=False,
    )


@pytest.mark.unit
def test_run_simulation_internal_non_fast_mode_multiset_plans_do_not_inherit_primary_dsl(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("reaction: A -> B; k=PRIMARY")
            self._state_network_editor = _StateNetworkEditor()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.mechanism = {"bound_set_id": None}
            self.param_names = ["k1"]
            self.species_names = ["A"]

        def as_worker_payload(self) -> dict[str, object]:
            return {
                "version": 1,
                "mechanism": self.mechanism,
                "rhs": object(),
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=PRIMARY",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

        def as_serializable_execution_payload(self) -> dict[str, object]:
            return {
                "version": 2,
                "mechanism": self.mechanism,
                "y0": np.array([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> B; k=PRIMARY",
                "temperature_schedule": None,
                "jacobian_func": None,
            }

    def _prepare_slider_runtime(*, set_id: Optional[str] = None):
        _ = set_id
        return _FakeRuntime()

    def _apply_slider_overrides(runtime: _FakeRuntime, *, set_id: Optional[str] = None) -> bool:
        runtime.mechanism["bound_set_id"] = str(set_id)
        return True

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0, 1]
    mw._batch_set_id_for_row.side_effect = ["id1", "id2"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._prepare_slider_runtime = MagicMock(side_effect=_prepare_slider_runtime)
    mw._apply_slider_overrides_to_bindings = MagicMock(side_effect=_apply_slider_overrides)
    mw._apply_overrides_to_text = MagicMock(side_effect=lambda text: str(text))
    mw._batch_initials_for_row.side_effect = [{"A": 1.0}, {"A": 4.0}]
    mw.preview_initials_for_row = MagicMock(side_effect=[{"A": 2.5}, {"A": 5.5}])
    mw._slider_overrides = {"k1": 2.0}
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=42, batch_rows=[0, 1], reuse_parallel_lane_pool=False)

    plan_by_set_id = controller.batch_context_owner.execution_payload_state().simulation_plan_by_set_id
    request_id1 = SimulationPlan.from_payload(plan_by_set_id["id1"]).to_execution_request().to_payload()
    request_id2 = SimulationPlan.from_payload(plan_by_set_id["id2"]).to_execution_request().to_payload()
    assert request_id1["prepared_payload"] is None
    assert request_id2["prepared_payload"] is None
    assert request_id1["mechanism_text"] == "reaction: A -> B; k=PRIMARY"
    assert request_id2["mechanism_text"] == "reaction: A -> B; k=PRIMARY"
    assert request_id1["initials"] == {"A": 1.0}
    assert request_id2["initials"] == {"A": 4.0}
    mw.preview_initials_for_row.assert_not_called()
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()

@pytest.mark.unit
def test_start_next_batch_simulation_non_fast_mode_ignores_prepared_payload(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    """Serial explicit runs must parse canonical DSL instead of using preview prepared payloads."""

    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, rows=[0], queue_ids=["id1"], queue_names=["set1"], full_dsl="reaction: A -> B; k=1", mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"}, mechanism_signature="sig", mechanism_signature_by_set_id={"id1": "sig-1"}, simulation_plan_by_set_id={"id1": _test_simulation_plan_payload(set_id="id1", set_name="set1", initials={"A": 1.0}, cache_key="explicit-cache")}, prepared=None, prepared_by_set_id={
            "id1": {"prepared_for": "id1", "version": 1},
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=42, cache_key="explicit-cache", pending_init_seed={}, pending_init_applied=True)

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["prepared"] is None
    assert created["started"] is True
    mw.preview_initials_for_row.assert_not_called()

@pytest.mark.unit
def test_start_next_batch_simulation_non_fast_mode_sets_plan_only_execution_boundary(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    created: dict[str, object] = {}
    request_payload = {
        "prepared_payload": {"version": 1, "prepared_for": "id1"},
        "initials": {"A": 3.0},
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=1",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }
    plan_payload = SimulationPlan.from_execution_request(
        request_payload,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "explicit-cache"},
        metadata={"set_id": "id1", "set_name": "set1"},
    ).to_payload()

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["prepared"] = dict(prepared) if isinstance(prepared, dict) else prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, rows=[0], queue_ids=["id1"], queue_names=["set1"], full_dsl="reaction: A -> B; k=1", simulation_plan_by_set_id={"id1": plan_payload}, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=42, cache_key="explicit-cache", pending_init_seed={}, pending_init_applied=True)

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["mechanism_text"] == "reaction: A -> B; k=1"
    worker = controller._simulation_worker
    worker_plan = SimulationPlan.from_payload(getattr(worker, "simulation_plan_payload", None))
    assert worker_plan.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    worker_request = worker_plan.to_execution_request().to_payload()
    assert worker_request["initials"] == {"A": 3.0}
    assert worker_request["prepared_payload"] is None
    assert worker_request["mechanism_text"] == "reaction: A -> B; k=1"
    assert getattr(worker, "_execution_request", None) is None


@pytest.mark.unit
def test_start_parallel_batch_simulations_submits_simulation_plan_payload(
    mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    request_payload = {
        "prepared_payload": {"version": 1, "prepared_for": "id1"},
        "initials": {"A": 3.0},
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=1",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }
    plan_payload = SimulationPlan.from_execution_request(
        request_payload,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "explicit-cache"},
        metadata={"set_id": "id1", "set_name": "set1"},
    ).to_payload()
    submitted_tasks: list[dict[str, object]] = []

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    _install_ready_batch_lane_pool(
        controller,
        _RecordingLanePool(submitted_tasks),
        max_lanes=2,
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id1"], queue_names=["set1"], run_id=101, effective_workers=2, full_dsl="reaction: A -> B; k=1", simulation_plan_by_set_id={"id1": plan_payload}, mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"}, mechanism_signature_by_set_id={"id1": "sig"}, simulation_identity_by_set_id={
            "id1": {"schema_id": "schema", "param_fingerprint": "fingerprint"}
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=42, pending_init_seed={}, pending_init_applied=True)

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    task = submitted_tasks[0]
    assert "simulation_plan" in task
    assert "execution_request" not in task
    assert "mechanism_signature" not in task
    submitted_plan = SimulationPlan.from_payload(task["simulation_plan"])  # type: ignore[arg-type]
    assert submitted_plan.algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT
    request = submitted_plan.to_execution_request().to_payload()
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 3.0}


@pytest.mark.unit
def test_start_parallel_batch_simulations_fast_existing_plan_submits_preview_plan_without_execution_request(
    mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan

    submitted_tasks: list[dict[str, object]] = []

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    _install_ready_batch_lane_pool(
        controller,
        _RecordingLanePool(submitted_tasks),
        max_lanes=2,
    )
    plan_payload = _test_simulation_plan_payload(
        set_id="id1",
        set_name="set1",
        mechanism_text="reaction: A -> B; k=4",
        initials={"A": 4.0},
        fast_mode=True,
        cache_key="preview-cache",
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id1"], queue_names=["set1"], run_id=101, effective_workers=2, full_dsl="reaction: A -> B; k=1", simulation_plan_by_set_id={"id1": plan_payload}, mechanism_text_by_set_id={"id1": "reaction: A -> B; k=4"}, mechanism_signature_by_set_id={"id1": "sig"}, simulation_identity_by_set_id={
            "id1": {"schema_id": "schema", "param_fingerprint": "fingerprint"}
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=42, pending_init_seed={}, pending_init_applied=True)

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    task = submitted_tasks[0]
    assert "execution_request" not in task
    plan = SimulationPlan.from_payload(task["simulation_plan"])  # type: ignore[index]
    assert plan.execution_mode == "preview"
    from kindred.core.simulation_plan import SimulationAlgebraPolicy

    assert plan.algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT
    request = plan.to_execution_request().to_payload()
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 4.0}
    assert request["mechanism_text"] == "reaction: A -> B; k=4"


@pytest.mark.unit
def test_start_parallel_batch_simulations_fast_existing_gui_plan_submits_batch_policy_and_preserves_payloads(
    mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    request_payload = {
        "prepared_payload": {"version": 1, "prepared_for": "id1"},
        "initials": {"A": 4.0},
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=4",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }
    cache_identity_payload = {
        "cache_key": "preview-cache",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
        "preview_batch_cache_token": "preview-token",
    }
    cache_scope_payload = {
        "scope_identity": {"schema_id": "scope", "queue_fingerprint": "queue"},
        "queue_ids": ["id1"],
    }
    metadata = {"set_id": "id1", "set_name": "set1", "fast_mode": True}
    plan_payload = SimulationPlan.from_execution_request(
        request_payload,
        execution_mode="preview",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload=cache_identity_payload,
        cache_scope_payload=cache_scope_payload,
        metadata=metadata,
    ).to_payload()
    submitted_tasks: list[dict[str, object]] = []

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    _install_ready_batch_lane_pool(
        controller,
        _RecordingLanePool(submitted_tasks),
        max_lanes=2,
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id1"], queue_names=["set1"], run_id=101, effective_workers=2, full_dsl="reaction: A -> B; k=1", simulation_plan_by_set_id={"id1": plan_payload}, mechanism_text_by_set_id={"id1": "reaction: A -> B; k=4"}, mechanism_signature_by_set_id={"id1": "sig"}, simulation_identity_by_set_id={
            "id1": {"schema_id": "schema", "param_fingerprint": "fingerprint"}
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=42, pending_init_seed={}, pending_init_applied=True)

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    task = submitted_tasks[0]
    assert "execution_request" not in task
    submitted_plan = SimulationPlan.from_payload(task["simulation_plan"])  # type: ignore[arg-type]
    assert submitted_plan.execution_mode == "preview"
    assert submitted_plan.algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT
    assert submitted_plan.cache_identity_payload == cache_identity_payload
    assert submitted_plan.cache_scope_payload == cache_scope_payload
    assert submitted_plan.metadata == metadata
    assert submitted_plan.version == SimulationPlan.from_payload(plan_payload).version
    expected_request_payload = {**request_payload, "version": 1}
    assert submitted_plan.to_execution_request().to_payload() == expected_request_payload
    assert "execution_request" not in task


@pytest.mark.unit
def test_start_next_batch_simulation_non_primary_explicit_worker_uses_secondary_result_payload_mode(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload,
            parent,
        ):
            created["owner"] = owner
            created["simulation_plan_payload"] = dict(simulation_plan_payload)
            _ = parent
            created["include_mechanism_in_result_payload"] = bool(include_mechanism_in_result_payload)
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 1.0})
    controller._release_current_simulation_worker = MagicMock()
    request_payload = {
        "prepared_payload": {"version": 1, "prepared_for": "id2"},
        "initials": {"A": 3.0},
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=2",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=1, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], primary_set_id="id1", full_dsl="reaction: A -> B; k=1", simulation_plan_by_set_id={
            "id2": SimulationPlan.from_execution_request(
                request_payload,
                execution_mode="explicit",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                cache_identity_payload={"cache_key": "explicit-cache"},
                metadata={"set_id": "id2", "set_name": "set2", "fast_mode": False},
            ).to_payload(),
        }, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=42, cache_key="explicit-cache", pending_init_seed={}, pending_init_applied=True)

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _RecordingWorker)
    controller._contained_simulation_owner_factory = lambda *, fast_mode: "ordinary-owner"
    monkeypatch.setattr(controller._runtime_application, "acquire_ready_owner", lambda **_kwargs: "ordinary-owner")

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["owner"] == "ordinary-owner"
    assert created["include_mechanism_in_result_payload"] is False
    worker_plan = SimulationPlan.from_payload(created["simulation_plan_payload"])
    worker_request = worker_plan.to_execution_request().to_payload()
    assert worker_request["initials"] == {"A": 3.0}
    assert worker_request["prepared_payload"] is None
    assert worker_request["mechanism_text"] == "reaction: A -> B; k=2"
    assert getattr(controller._simulation_worker, "_execution_request", None) is None


@pytest.mark.unit
def test_start_next_batch_simulation_fast_mode_existing_plan_attaches_preview_plan_without_execution_request(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan

    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=None,
            parent,
        ):
            created["owner"] = owner
            created["simulation_plan_payload"] = dict(simulation_plan_payload)
            created["include_mechanism_in_result_payload"] = bool(include_mechanism_in_result_payload)
            _ = parent
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 4.0})
    mw._slider_overrides = {}
    controller._release_current_simulation_worker = MagicMock()
    plan_payload = _test_simulation_plan_payload(
        set_id="id2",
        set_name="set2",
        mechanism_text="reaction: A -> B; k=3",
        initials={"A": 4.0},
        fast_mode=True,
        cache_key="slider-cache",
    )
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=1, rows=[0, 1], queue_ids=["id1", "id2"], queue_names=["set1", "set2"], full_dsl="reaction: A -> B; k=1", mechanism_text_by_set_id={"id2": "reaction: A -> B; k=3"}, mechanism_signature_by_set_id={"id2": "sig-3"}, simulation_identity_by_set_id={
            "id2": {"schema_id": "schema-b", "param_fingerprint": "fingerprint-b"}
        }, simulation_plan_by_set_id={"id2": plan_payload}, prepared_by_set_id={}, solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=True, request_id=12, cache_key="slider-cache", pending_init_seed={}, pending_init_applied=True)

    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _RecordingWorker)
    controller._contained_simulation_owner_factory = lambda *, fast_mode: "preview-owner"
    monkeypatch.setattr(controller._runtime_application, "acquire_ready_owner", lambda **_kwargs: "preview-owner")

    controller._start_next_batch_simulation()

    assert created["started"] is True
    assert created["owner"] == "preview-owner"
    assert created["include_mechanism_in_result_payload"] is False
    assert getattr(controller._simulation_worker, "_execution_request", None) is None
    plan = SimulationPlan.from_payload(created["simulation_plan_payload"])
    assert plan.execution_mode == "preview"
    assert plan.to_execution_request().to_payload()["initials"] == {"A": 4.0}
    assert plan.to_execution_request().to_payload()["mechanism_text"] == "reaction: A -> B; k=3"

@pytest.mark.unit
def test_run_simulation_internal_energy_mode_builds_simulation_plans(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_plan import SimulationPlan
    from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return "\n".join(
                [
                    "energy=kJ/mol",
                    "T=298.15",
                    "state: A, kind=GS, energy=0, members=A",
                    "state: B, kind=GS, energy=5, members=B",
                    "state: TS1, kind=TS, energy=25",
                    "edge: A,TS1",
                    "edge: TS1,B",
                ]
            )

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text("")
            self._state_network_editor = _StateNetworkEditor()

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = ["id1"]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {
        "dGact_fwd__TS1__A__B": 32.0,
        "dG_eq__TS1__A__B": 7.0,
    }
    mw._slider_runtime_dirty = False
    mw._prepared_slider_runtime_cache = None
    mw._parse_sim_time_seconds.return_value = 10.0
    runtime = MainWindowVariableRuntime(mw)
    runtime.set_variable_metadata(
        {
            "dGact_fwd__TS1__A__B": {
                "type": "energy",
                "role": "dG_act_fwd",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
            "dG_eq__TS1__A__B": {
                "type": "energy",
                "role": "dG_eq",
                "ts": "TS1",
                "reactant": "A",
                "product": "B",
            },
        }
    )
    mw._prepare_slider_runtime = MagicMock(
        side_effect=lambda *, set_id=None: runtime.prepare_slider_runtime(set_id=set_id)
    )
    mw._apply_slider_overrides_to_bindings = MagicMock(
        side_effect=lambda prepared, *, set_id=None: runtime.apply_slider_overrides_to_bindings(
            prepared,
            set_id=set_id,
        )
    )

    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=99, batch_rows=[0], reuse_parallel_lane_pool=False)

    execution_state = controller.batch_context_owner.execution_payload_state()
    prepared_by_set_id = execution_state.prepared_by_set_id
    simulation_plan_by_set_id = execution_state.simulation_plan_by_set_id

    assert prepared_by_set_id == {}
    request = SimulationPlan.from_payload(simulation_plan_by_set_id["id1"]).to_execution_request().to_payload()
    assert request["prepared_payload"] is None
    mw._prepare_slider_runtime.assert_not_called()
    mw._apply_slider_overrides_to_bindings.assert_not_called()


def test_start_next_batch_simulation_explicit_run_ignores_staged_concentration_overlay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    created: dict[str, object] = {}

    class _RecordingWorker:
        def __init__(
            self,
            *,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent,
            prepared,
            include_mechanism_in_result_payload=None,
        ):
            created["mechanism_text"] = str(mechanism_text)
            created["initials"] = dict(initials)
            created["t_span"] = tuple(t_span)
            created["solver_config"] = dict(solver_config)
            created["prepared"] = prepared
            created["include_mechanism_in_result_payload"] = include_mechanism_in_result_payload
            self.progress = _FakeSignal()
            self.result_ready = _FakeSignal()
            self.error = _FakeSignal()

        def start(self) -> None:
            created["started"] = True

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    controller._release_current_simulation_worker = MagicMock()
    seed_batch_context(controller.batch_context_owner, active=True, parallel=False, pos=0, rows=[0], queue_ids=["id2"], queue_names=["set2"], full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=10.0, fast_mode=False, request_id=7, cache_key="explicit-cache", pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={
        "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", initials={"A": 1.0}, cache_key="explicit-cache"),
    })

    _install_recording_contained_worker(monkeypatch, created, controller)

    controller._start_next_batch_simulation()

    assert created["initials"] == {"A": 1.0}
    assert created["started"] is True
    mw.preview_initials_for_row.assert_not_called()


def test_start_parallel_batch_simulations_explicit_run_ignores_staged_concentration_overlay(
    mw: _FakeMainWindow, controller: SimulationController
):
    submitted: list[dict[str, object]] = []

    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.preview_initials_for_row = MagicMock(return_value={"A": 2.5})
    pool = _RecordingLanePool(submitted)
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda _message, _exc: None,
    )
    controller.parallel_batch.lane_pool_factory = MagicMock(return_value=pool)
    controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    seed_batch_context(controller.batch_context_owner, active=True, parallel=True, rows=[0], queue_ids=["id2"], queue_names=["set2"], run_id=3, request_id=11, full_dsl="reaction: A -> B; k=1", mechanism_signature="sig", solver_config={"solver": "BDF"}, t_end=10.0, effective_workers=2, fast_mode=False, pending_init_seed={}, pending_init_applied=True, simulation_plan_by_set_id={
        "id2": _test_simulation_plan_payload(set_id="id2", set_name="set2", initials={"A": 1.0}, cache_key="explicit-cache"),
    })

    controller._start_parallel_batch_simulations()
    _join_active_batch_requests(controller)

    assert submitted
    from kindred.core.simulation_plan import SimulationPlan

    submitted_request = SimulationPlan.from_payload(submitted[0]["simulation_plan"]).to_execution_request().to_payload()
    assert submitted_request["initials"] == {"A": 1.0}
    assert "initials" not in submitted[0]
    mw.preview_initials_for_row.assert_not_called()


def test_explicit_run_worker_error_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert controller.batch_context_owner.pending_dirty_reset_state().set_ids == ("id1",)
    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.discard_concentration_overlays_for_rows.assert_not_called()

    _error_with_callback_identity(
        controller,
        {"kind": "simulation_error", "message": "ode build failed"},
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )

    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    mw.discard_concentration_overlays_for_rows.assert_not_called()


def test_explicit_run_success_clears_targeted_concentration_overlays_by_set_id_after_row_reorder(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}.get(int(row))
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert controller.batch_context_owner.pending_dirty_reset_state().set_ids == ("id1",)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id2", 1: "id1"}.get(int(row))

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_rows.assert_not_called()


def test_explicit_run_success_clears_targeted_concentration_overlays_by_set_id_not_row(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    mw._batch_set_id_for_row.return_value = "id2"
    mw.discard_concentration_overlays_for_set_ids.return_value = True

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_rows.assert_not_called()


def test_explicit_run_success_cancels_pending_species_preview_after_targeted_overlay_reset(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    assert mw.discard_concentration_overlays_for_set_ids.call_count == 1
    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is False
    assert controller._pending_slider_sim_request_id is None
    assert scheduled == []


def test_explicit_run_success_preserves_pending_slider_replay_for_non_targeted_dirty_set(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1, "id2": 3}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7
    controller._pending_slider_target_set_ids = ("id2",)

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    mw.reset_mechanism_workspaces.assert_called_once_with(["id1"])
    mw.discard_concentration_overlays_for_set_ids.assert_called_once_with(["id1"])
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert controller._pending_slider_sim_request_id == 7
    assert controller._pending_slider_target_set_ids == ("id2",)
    assert scheduled == [controller._run_simulation_from_slider]


def test_explicit_run_preflight_abort_does_not_schedule_pending_slider_replay(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return ""

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda _text: "",
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_current_row.return_value = 0
    mw._last_slider_change_name = "k1"

    controller._latest_sim_request_id = 2
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 1
    controller._pending_slider_target_set_ids = ("id1", "id2")

    controller._run_simulation()

    assert controller._pending_slider_simulation is True
    assert controller._pending_slider_sim_request_id == 1
    assert controller._pending_slider_target_set_ids == ("id1", "id2")
    assert scheduled == []


def test_run_simulation_replays_selected_run_after_runtime_becomes_ready(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    scheduled: list[object] = []

    warming = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="warming",
        ready=False,
        generation=3,
        message="Preparing simulation runtime...",
        required=True,
        controls_ready=False,
        polling=True,
    )
    ready = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="ready",
        ready=True,
        generation=3,
        required=True,
        controls_ready=True,
        polling=False,
    )
    snapshots = [warming, ready]

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    controller._selected_run_runtime_snapshot = MagicMock(side_effect=lambda: snapshots.pop(0) if snapshots else ready)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._ensure_selected_run_runtime_warming = MagicMock()
    controller._run_intent_signature_for_rows = MagicMock(return_value="intent-a")
    controller.run_simulation_internal = MagicMock()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_set_name_for_id.return_value = "set1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation()

    controller.run_simulation_internal.assert_not_called()
    controller._ensure_selected_run_runtime_warming.assert_called_once()
    assert mw._runtime_availability_refresh_requests == 0
    assert mw._status_label.text == "Preparing simulation runtime..."
    assert mw.run_button_is_enabled() is False
    assert len(scheduled) == 1

    scheduled.pop(0)()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is False
    assert kwargs["batch_rows"] == [0]
    assert kwargs["reuse_parallel_lane_pool"] is False


def test_run_simulation_replays_selected_run_after_failed_runtime_restarts(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    scheduled: list[object] = []

    failed = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="failed",
        ready=False,
        generation=3,
        failure="boom",
        message="Simulation runtime failed to start. boom",
        required=True,
        controls_ready=False,
        polling=False,
    )
    warming = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="warming",
        ready=False,
        generation=4,
        message="Preparing simulation runtime...",
        required=True,
        controls_ready=False,
        polling=True,
    )
    ready = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="ready",
        ready=True,
        generation=4,
        required=True,
        controls_ready=True,
        polling=False,
    )
    snapshots = [failed, warming, ready]

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    controller._selected_run_runtime_snapshot = MagicMock(side_effect=lambda: snapshots.pop(0) if snapshots else ready)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._ensure_selected_run_runtime_warming = MagicMock()
    controller._run_intent_signature_for_rows = MagicMock(return_value="intent-a")
    controller.run_simulation_internal = MagicMock()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_set_name_for_id.return_value = "set1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation()

    controller.run_simulation_internal.assert_not_called()
    controller._ensure_selected_run_runtime_warming.assert_not_called()
    assert mw._runtime_availability_refresh_requests == 0
    assert mw._status_label.text == "Simulation runtime failed to start. boom"
    assert scheduled == []
    assert not controller._pending_run_after_runtime_ready.active


def test_run_simulation_replays_when_runtime_becomes_ready_during_existing_poll(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    warming = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="warming",
        ready=False,
        generation=3,
        message="Preparing simulation runtime...",
        required=True,
        controls_ready=False,
        polling=True,
    )
    ready = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="ready",
        ready=True,
        generation=4,
        required=True,
        controls_ready=True,
        polling=False,
    )
    snapshots = [warming, ready]
    scheduled: list[object] = []

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    controller._selected_run_runtime_snapshot = MagicMock(side_effect=lambda: snapshots.pop(0) if snapshots else ready)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._ensure_selected_run_runtime_warming = MagicMock()
    controller._run_intent_signature_for_rows = MagicMock(return_value="intent-a")
    controller.run_simulation_internal = MagicMock()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_set_name_for_id.return_value = "set1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation()

    controller._ensure_selected_run_runtime_warming.assert_called_once()
    controller.run_simulation_internal.assert_not_called()
    assert len(scheduled) == 1
    assert controller._pending_run_after_runtime_ready.active

    scheduled.pop(0)()

    controller.run_simulation_internal.assert_called_once()
    assert not controller._pending_run_after_runtime_ready.active


def test_pending_run_after_runtime_ready_cancels_when_run_intent_changes(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    scheduled: list[object] = []
    warming = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="warming",
        ready=False,
        generation=3,
        message="Preparing simulation runtime...",
        required=True,
        controls_ready=False,
        polling=True,
    )
    ready = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="ready",
        ready=True,
        generation=4,
        required=True,
        controls_ready=True,
        polling=False,
    )
    snapshots = [warming, warming, ready]

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))
    controller._selected_run_runtime_snapshot = MagicMock(side_effect=lambda: snapshots.pop(0) if snapshots else ready)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller._run_intent_signature_for_rows = MagicMock(side_effect=["clicked-intent", "changed-intent"])
    controller.run_simulation_internal = MagicMock()

    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_set_name_for_id.return_value = "set1"
    mw._batch_preferred_primary_set_id.return_value = "id1"

    controller._run_simulation()

    assert len(scheduled) == 1
    scheduled.pop(0)()

    controller.run_simulation_internal.assert_not_called()
    assert not controller._pending_run_after_runtime_ready.active
    assert mw._status_label.text == "Preparing simulation runtime..."


def test_pending_run_after_runtime_ready_cancel_keeps_runtime_controls_gated(
    mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot
    from kindred.gui.controllers.simulation_run_state import PendingRunAfterRuntimeReadyState

    warming = RuntimeReadinessSnapshot(
        mode="ordinary",
        status="warming",
        ready=False,
        generation=4,
        message="Preparing simulation runtime...",
        required=True,
        controls_ready=False,
        polling=True,
    )
    controller._run_state.pending_run_after_runtime_ready = PendingRunAfterRuntimeReadyState(
        active=True,
        rows=(0,),
        target_set_ids=("id1",),
        intent_signature="intent-a",
    )
    controller._selected_run_runtime_snapshot = MagicMock(return_value=warming)
    controller._ensure_interactive_simulation_runtime_available_for_mode = MagicMock()
    controller.run_simulation_internal = MagicMock()

    mw._batch_rows_for_scope.return_value = [1]
    mw._batch_set_id_for_row.side_effect = lambda row: "id2" if int(row) == 1 else "id1"

    controller._retry_pending_run_after_runtime_ready()

    controller.run_simulation_internal.assert_not_called()
    assert not controller._pending_run_after_runtime_ready.active
    assert mw.run_button_is_enabled() is False
    assert mw._runtime_availability_refresh_requests == 1
    assert mw._status_label.text == "Preparing simulation runtime..."


def test_explicit_run_success_requeues_surviving_pending_slider_replay_with_fresh_request_id(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 2
    mw._batch_store.set_names.return_value = ["set1", "set2"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.side_effect = lambda row: {0: "id1", 1: "id2"}[int(row)]
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1, "id2": 2}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._last_slider_change_name = "k1"

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()
    controller._interactive_simulation_runtime_snapshot = MagicMock(
        return_value=RuntimeReadinessSnapshot(
            mode="preview",
            status="ready",
            ready=True,
            generation=1,
            required=True,
            controls_ready=True,
            polling=False,
        )
    )

    controller._run_simulation_internal(fast_mode=False, request_id=2, batch_rows=[0], reuse_parallel_lane_pool=False)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=2,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 1
    controller._pending_slider_target_set_ids = ("id1", "id2")

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    controller.run_simulation_internal = MagicMock()
    assert scheduled == [controller._run_simulation_from_slider]
    controller._latest_sim_request_id = 2
    scheduled[0]()

    controller.run_simulation_internal.assert_called_once()
    _, kwargs = controller.run_simulation_internal.call_args
    assert kwargs["fast_mode"] is True
    assert kwargs["batch_rows"] == [1]


def test_explicit_run_success_preserves_targeted_dirty_state_edited_after_run_start(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw._dirty_state_generations = {"id1": 1}
    mw.reset_mechanism_workspaces.return_value = True
    mw.discard_concentration_overlays_for_set_ids.return_value = True
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    mw._dirty_state_generations["id1"] = 2

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    mw.reset_mechanism_workspaces.assert_not_called()
    mw.discard_concentration_overlays_for_set_ids.assert_not_called()
    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert controller._pending_slider_sim_request_id == 7
    assert scheduled == [controller._run_simulation_from_slider]


def test_run_simulation_internal_invalid_t_end_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._parse_sim_time_seconds.side_effect = ValueError("bad t_end")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid t_end"
    mw.reset_mechanism_workspaces.assert_not_called()
    controller._start_next_batch_simulation.assert_not_called()


def test_run_simulation_internal_invalid_schedule_protected_name_uses_run_validation_cleanup(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "\n".join(
                [
                    "equilibrium: A <-> B; kf=1.0; kr=0.5",
                    "initial: A=1.0",
                    "intervention: op=add; species=A; time=1.0; amount_param=K1",
                ]
            )

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0, "B": 0.0}

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid Intervention Schedule"
    assert "K1" in warned[0][1]
    assert "not a valid indexed parameter identifier" in warned[0][1]
    assert mw.run_button_is_enabled() is True
    assert controller._slider_simulation_active is False
    mw._invalidate_pending_init_preserved_results_after_failed_run.assert_called_once_with()
    controller._start_next_batch_simulation.assert_not_called()


def test_run_simulation_internal_invalid_initials_preserves_targeted_dirty_workspaces(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    warned: list[tuple[str, str]] = []

    def _warning(_parent, title, message):
        warned.append((str(title), str(message)))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )

    class _MechTmp:
        def species_names(self):
            return ["A", "B"]

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", lambda *_a, **_k: _MechTmp())

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._slider_overrides = {"k1": 2.0}
    mw._batch_initials_for_row.side_effect = ValueError("bad initials")

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)

    assert warned and warned[0][0] == "Invalid Initial Conditions"
    mw.reset_mechanism_workspaces.assert_not_called()
    controller._start_next_batch_simulation.assert_not_called()


def test_explicit_run_success_preserves_pending_species_preview_replay_when_no_targeted_dirty_reset_occurred(
    monkeypatch, mw: _FakeMainWindow, controller: SimulationController
):
    class _Text:
        def toPlainText(self) -> str:
            return "reaction: A -> B; k=1"

    class _StateNetworkEditor:
        def get_state_network_dsl(self) -> str:
            return ""

    class _MechanismEditor:
        def __init__(self):
            self._reactions_text = _Text()
            self._state_network_editor = _StateNetworkEditor()

    class _ActiveTimer:
        def __init__(self) -> None:
            self._active = True
            self.stop_calls = 0

        def isActive(self) -> bool:
            return bool(self._active)

        def stop(self) -> None:
            self.stop_calls += 1
            self._active = False

    scheduled: list[object] = []

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.strip_reaction_dsl_initial_concentrations",
        lambda text: text,
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_run_preparation.batch_mechanism_signature",
        lambda **_kwargs: "sig",
    )
    monkeypatch.setattr(
        "kindred.gui.controllers.simulation_controller.compute_effective_batch_workers",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: scheduled.append(fn))

    mw._mechanism_editor = _MechanismEditor()
    mw._batch_store.row_count.return_value = 1
    mw._batch_store.set_names.return_value = ["set1"]
    mw._batch_rows_for_scope.return_value = [0]
    mw._batch_set_id_for_row.return_value = "id1"
    mw._batch_preferred_primary_set_id.return_value = "id1"
    mw._batch_cache_key.return_value = "ck"
    mw._batch_initials_for_row.return_value = {"A": 1.0}
    mw.reset_mechanism_workspaces.return_value = False
    mw.discard_concentration_overlays_for_set_ids.return_value = False
    mw._species_slider_update_timer = _ActiveTimer()

    controller._start_next_batch_simulation = MagicMock()
    controller._shutdown_batch_lane_pool = MagicMock()

    controller._run_simulation_internal(fast_mode=False, request_id=1, batch_rows=[0], reuse_parallel_lane_pool=False)
    callback_identity = _capture_callback_identity(
        controller,
        run_id=int(controller._active_run_id),
        fast_mode=False,
        request_id=1,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="ck",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
    )
    controller._pending_slider_simulation = True
    controller._pending_slider_sim_request_id = 7

    controller._on_simulation_complete(
        _successful_result_payload(),
        callback_identity=callback_identity,
    )

    assert mw._species_slider_update_timer.stop_calls == 1
    assert controller._pending_slider_simulation is True
    assert _pending_slider_preview_launch(controller).handoff_queued is True
    assert controller._pending_slider_sim_request_id == 7
    assert scheduled == [controller._run_simulation_from_slider]

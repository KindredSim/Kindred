from __future__ import annotations

import inspect
import multiprocessing
import os
import pickle
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from kindred.core.containment_kernel import (
    ContainmentChildFatal,
    ContainmentHandlerResponse,
    ContainmentKernelAcceptTimeout,
    ContainmentKernelActiveTimeout,
    ContainmentKernelCancelled,
    ContainmentKernelChildFailure,
    ContainmentKernelProtocolError,
    ContainmentKernelStartupTimeout,
    _start_process_with_env,
)
from kindred.core.exceptions import SimulationCancelled
from kindred.core.simulation_failure import (
    build_simulation_failure,
    coerce_simulation_failure,
    simulation_failure_from_exception,
)
from kindred.core.simulation_plan import SimulationPlan
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    SimulationPreparationError,
    prepared_simulation_run_for_execution_request,
    prepare_simulation_worker_run,
)
from kindred.core.simulation_result_finalization import build_finalized_simulation_result_payload
from kindred.core.simulator.solvers import SimulationRequest, solve_ode
from kindred.core.simulation_runtime_service import SimulationRuntimeOwner
from kindred.core.runtime_defaults import contained_child_blas_thread_env

_DEFAULT_SIMULATION_ACTIVE_TIMEOUT_S = 60.0
_OWNER_READY_TIMEOUT_S = 30.0
_OWNER_ACCEPT_TIMEOUT_S = 10.0
_OWNER_POLL_INTERVAL_S = 0.02
_PROCESS_JOIN_TIMEOUT_S = 0.5


class SimulationContainmentPayloadError(ValueError):
    """Raised when a simulation containment payload cannot safely cross spawn."""


class SimulationContainmentProtocolError(RuntimeError):
    """Raised when a warm simulation child violates the containment protocol."""


class SimulationContainmentChildFailure(SimulationContainmentProtocolError):
    """Raised when the warm simulation child returns a structured fatal/error payload."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = coerce_simulation_failure(failure)
        super().__init__(
            str(self.failure.get("message") or "Contained simulation owner failed.")
        )


class SimulationContainmentStartupTimeout(SimulationContainmentProtocolError):
    """Raised when a warm simulation child does not become READY."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"Contained simulation owner did not become ready within {self.timeout_s:.3g} seconds."
        )


class SimulationContainmentAcceptTimeout(SimulationContainmentProtocolError):
    """Raised when a warm simulation child does not ACCEPT a submitted request."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"Contained simulation owner did not accept the request within {self.timeout_s:.3g} seconds."
        )


class SimulationContainmentTimeout(RuntimeError):
    """Raised when an accepted contained simulation exceeds its active timeout."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        message = f"Simulation timed out after {self.timeout_s:.3g} seconds."
        self.failure = build_simulation_failure(
            "timeout",
            message,
            code="E306",
            details={"active_solve_timeout_s": self.timeout_s},
            exc_type=type(self).__name__,
        )
        super().__init__(message)


def _formatted_exception_stack_trace(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _failure_payload_with_stack_trace(payload: dict[str, Any], *, exc: BaseException) -> dict[str, Any]:
    context = payload.get("context")
    if isinstance(context, dict) and str(context.get("stack_trace") or "").strip():
        return payload

    stack_trace = _formatted_exception_stack_trace(exc)
    if not stack_trace:
        return payload

    enriched_payload = dict(payload)
    enriched_context = dict(context) if isinstance(context, dict) else {}
    enriched_context["stack_trace"] = stack_trace
    enriched_payload["context"] = enriched_context
    return enriched_payload


@dataclass(frozen=True)
class SimulationReplyGate:
    owner_epoch: int
    request_id: int

    def is_current(self, message: Mapping[str, Any]) -> bool:
        return (
            int(message.get("owner_epoch", -1)) == int(self.owner_epoch)
            and int(message.get("request_id", -1)) == int(self.request_id)
        )


def _path_for(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "payload"


def _is_qt_like(value: Any) -> bool:
    module = str(getattr(type(value), "__module__", "") or "")
    return module.startswith(("PySide6", "PyQt", "shiboken6"))


def _is_known_serializable_callable_object(value: Any) -> bool:
    module = str(getattr(type(value), "__module__", "") or "")
    return module.startswith("kindred.core.temperature")


def _reject_unsafe_value(value: Any, *, path: tuple[str, ...]) -> None:
    location = _path_for(path)
    if isinstance(value, (SimulationRequest, SimulationExecutionRequest)):
        raise SimulationContainmentPayloadError(
            f"{location} contains {type(value).__name__}; pass only serialized mappings."
        )
    if _is_qt_like(value):
        raise SimulationContainmentPayloadError(f"{location} contains a Qt object; Qt cannot cross spawn.")
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        raise SimulationContainmentPayloadError(f"{location} contains callable {value!r}.")
    if callable(value) and not _is_known_serializable_callable_object(value):
        raise SimulationContainmentPayloadError(f"{location} contains callable {value!r}.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_unsafe_value(item, path=(*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_unsafe_value(item, path=(*path, str(index)))


def _prepared_payload_from_plan_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    execution_request = payload.get("execution_request")
    if not isinstance(execution_request, Mapping):
        return None
    prepared_payload = execution_request.get("prepared_payload")
    return prepared_payload if isinstance(prepared_payload, Mapping) else None


def contained_payloads_equal(left: Any, right: Any) -> bool:
    """Compare contained payload values without NumPy truth-value ambiguity."""
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        return bool(
            left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        if left_keys != right_keys:
            return False
        for key in left_keys:
            left_value = left.get(key)
            right_value = right.get(key)
            if not contained_payloads_equal(left_value, right_value):
                return False
        return True
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            return False
        if len(left) != len(right):
            return False
        return all(contained_payloads_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    try:
        return bool(left == right)
    except ValueError:
        return False


def contained_owner_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the runtime-owner identity carried by a contained simulation plan payload."""
    if not isinstance(payload, Mapping):
        return {}
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        identity = metadata.get("contained_owner_identity")
        if isinstance(identity, Mapping):
            return validate_contained_simulation_payload(dict(identity))
    return validate_contained_simulation_payload(dict(payload))


def contained_owner_payloads_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare the owner lifecycle identity for contained simulation reuse."""
    left_identity = contained_owner_identity_payload(left)
    right_identity = contained_owner_identity_payload(right)
    return contained_payloads_equal(left_identity, right_identity)


def validate_contained_simulation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, (SimulationRequest, SimulationExecutionRequest)):
        raise SimulationContainmentPayloadError(
            f"{type(payload).__name__} instances cannot cross simulation containment; pass a payload mapping."
        )
    if not isinstance(payload, Mapping):
        raise SimulationContainmentPayloadError("Contained simulation payload must be a mapping.")

    payload_dict = dict(payload)
    prepared_payload = _prepared_payload_from_plan_payload(payload_dict)
    if prepared_payload is not None:
        version = int(prepared_payload.get("version", 1))
        if version == 1 or "rhs" in prepared_payload:
            raise SimulationContainmentPayloadError(
                "Contained simulation payload rejects version-1 prepared payload with rhs."
            )

    _reject_unsafe_value(payload_dict, path=("payload",))
    try:
        pickle.dumps(payload_dict)
    except Exception as exc:
        raise SimulationContainmentPayloadError(
            f"Contained simulation payload is not pickle-safe for spawn: {exc}"
        ) from exc
    return payload_dict


def build_contained_simulation_plan_payload(
    simulation_plan: SimulationPlan | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(simulation_plan, SimulationPlan):
        payload = simulation_plan.to_payload()
    elif isinstance(simulation_plan, Mapping):
        payload = SimulationPlan.from_payload(simulation_plan).to_payload()
    else:
        raise SimulationContainmentPayloadError(
            "Contained simulation plan payload requires a SimulationPlan or mapping."
        )
    return validate_contained_simulation_payload(payload)


def _terminate_process(proc: multiprocessing.Process, *, join_timeout_s: float = _PROCESS_JOIN_TIMEOUT_S) -> None:
    if not proc.is_alive():
        proc.join(timeout=join_timeout_s)
        return
    proc.terminate()
    proc.join(timeout=join_timeout_s)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(timeout=join_timeout_s)


@dataclass(frozen=True)
class _PreparedContainedSimulationRequest:
    request_payload: dict[str, Any]
    plan_payload: dict[str, Any]
    plan: SimulationPlan
    execution_request: dict[str, Any]
    prepared: Any


class _SimulationChildHandler:
    def __init__(self, simulation_plan_payload: Mapping[str, Any]) -> None:
        self._startup_plan_payload: dict[str, Any] | None = None
        self._startup_prepared = None
        self._prepared_by_request_id: dict[int, _PreparedContainedSimulationRequest] = {}
        if isinstance(simulation_plan_payload, Mapping) and "execution_request" in simulation_plan_payload:
            self._startup_plan_payload = validate_contained_simulation_payload(simulation_plan_payload)
            startup_plan = SimulationPlan.from_payload(self._startup_plan_payload)
            startup_execution_request = startup_plan.to_execution_request().to_payload()
            self._startup_prepared = prepare_simulation_worker_run(execution_request=startup_execution_request)
            self._prewarm_solver_imports()

    @staticmethod
    def _prewarm_solver_imports() -> None:
        from kindred.core.scipy_integrate import load_scipy_integrate

        load_scipy_integrate()

    def _prepare_request(self, request_payload: Mapping[str, Any]) -> _PreparedContainedSimulationRequest:
        payload = dict(request_payload or {})
        request_plan_payload = payload.get("simulation_plan_payload")
        if isinstance(request_plan_payload, Mapping):
            plan_payload = validate_contained_simulation_payload(request_plan_payload)
            plan = SimulationPlan.from_payload(plan_payload)
            execution_request = plan.to_execution_request().to_payload()
            if (
                self._startup_prepared is not None
                and self._startup_plan_payload is not None
                and contained_owner_payloads_match(self._startup_plan_payload, plan_payload)
            ):
                prepared = prepared_simulation_run_for_execution_request(
                    self._startup_prepared,
                    execution_request,
                )
            else:
                prepared = prepare_simulation_worker_run(execution_request=execution_request)
        elif self._startup_prepared is not None and self._startup_plan_payload is not None:
            plan_payload = dict(self._startup_plan_payload)
            plan = SimulationPlan.from_payload(self._startup_plan_payload)
            execution_request = plan.to_execution_request().to_payload()
            prepared = self._startup_prepared
        else:
            raise SimulationContainmentPayloadError(
                "Contained simulation request missing simulation_plan_payload."
            )
        return _PreparedContainedSimulationRequest(
            request_payload=payload,
            plan_payload=dict(plan_payload),
            plan=plan,
            execution_request=execution_request,
            prepared=prepared,
        )

    def before_accept(self, request_payload: Mapping[str, Any], context: Any) -> None:
        try:
            self._prepared_by_request_id[int(context.request_id)] = self._prepare_request(request_payload)
        except BaseException as exc:  # noqa: BLE001 - child boundary must serialize reconstruction failures
            reconstruction_failure = isinstance(
                exc,
                (
                    SimulationContainmentPayloadError,
                    SimulationPreparationError,
                    TypeError,
                    ValueError,
                ),
            )
            raise ContainmentChildFatal(
                _failure_payload_with_stack_trace(
                    simulation_failure_from_exception(
                        exc,
                        kind="simulation_containment_reconstruction" if reconstruction_failure else None,
                    ),
                    exc=exc,
                )
            ) from exc

    def handle_request(self, request_payload: Mapping[str, Any], _context: Any) -> dict[str, Any] | ContainmentHandlerResponse:
        try:
            request_id = int(getattr(_context, "request_id", -1))
            prepared_request = self._prepared_by_request_id.pop(request_id, None)
            if prepared_request is None:
                prepared_request = self._prepare_request(request_payload)
            if bool(prepared_request.request_payload.get("prepare_only")):
                self._startup_plan_payload = dict(prepared_request.plan_payload)
                self._startup_prepared = prepared_request.prepared
                self._prewarm_solver_imports()
                return {"success": True, "prepared": True}

            result = solve_ode(prepared_request.prepared.request)
            include_mechanism = bool(prepared_request.request_payload.get("include_mechanism_in_result_payload"))
            return build_finalized_simulation_result_payload(
                mechanism=prepared_request.prepared.mechanism,
                result=result,
                species_names=list(prepared_request.prepared.species_names),
                initials_for_algebra=prepared_request.prepared.initials_for_algebra,
                simulation_plan=prepared_request.plan,
                preparation_warnings=list(getattr(prepared_request.prepared, "warnings", None) or []),
                solver=str(getattr(prepared_request.prepared.request, "solver", "") or ""),
                mechanism_text=str(prepared_request.execution_request.get("mechanism_text") or ""),
                solver_config=dict(prepared_request.execution_request.get("solver_config") or {}),
                include_mechanism=include_mechanism,
            )
        except SimulationCancelled as exc:
            return ContainmentHandlerResponse(
                kind="cancelled",
                failure=simulation_failure_from_exception(exc),
            )
        except BaseException as exc:  # noqa: BLE001 - process boundary must serialize all failures
            reconstruction_failure = isinstance(
                exc,
                (
                    SimulationContainmentPayloadError,
                    SimulationPreparationError,
                    TypeError,
                    ValueError,
                ),
            )
            return ContainmentHandlerResponse(
                kind="fatal" if reconstruction_failure else "error",
                failure=_failure_payload_with_stack_trace(
                    simulation_failure_from_exception(
                        exc,
                        kind="simulation_containment_reconstruction" if reconstruction_failure else None,
                    ),
                    exc=exc,
                ),
            )


def create_simulation_child_handler(simulation_plan_payload: Mapping[str, Any]) -> _SimulationChildHandler:
    try:
        return _SimulationChildHandler(simulation_plan_payload)
    except BaseException as exc:  # noqa: BLE001 - startup/reconstruction failure is fatal to the owner
        raise ContainmentChildFatal(
            _failure_payload_with_stack_trace(
                simulation_failure_from_exception(exc, kind="simulation_containment_startup"),
                exc=exc,
            )
        ) from exc


def _simulation_owner_child(
    simulation_plan_payload: Mapping[str, Any],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    owner_epoch: int,
) -> None:
    try:
        startup_plan_payload: dict[str, Any] | None = None
        startup_prepared = None
        if isinstance(simulation_plan_payload, Mapping) and "execution_request" in simulation_plan_payload:
            startup_plan_payload = validate_contained_simulation_payload(simulation_plan_payload)
            startup_plan = SimulationPlan.from_payload(startup_plan_payload)
            startup_execution_request = startup_plan.to_execution_request().to_payload()
            startup_prepared = prepare_simulation_worker_run(execution_request=startup_execution_request)
            _SimulationChildHandler._prewarm_solver_imports()
        output_queue.put(
            {
                "kind": "ready",
                "owner_epoch": int(owner_epoch),
                "pid": int(os.getpid()),
            }
        )
        while True:
            request = input_queue.get()
            if not isinstance(request, Mapping):
                continue
            kind = str(request.get("kind") or "")
            if kind == "close":
                return
            if kind != "solve":
                continue
            request_id = int(request.get("request_id", -1))
            try:
                request_payload = dict(request.get("payload") or {})
                request_plan_payload = request_payload.get("simulation_plan_payload")
                if isinstance(request_plan_payload, Mapping):
                    plan_payload = validate_contained_simulation_payload(request_plan_payload)
                    plan = SimulationPlan.from_payload(plan_payload)
                    execution_request = plan.to_execution_request().to_payload()
                    prepared = prepare_simulation_worker_run(execution_request=execution_request)
                elif startup_prepared is not None and startup_plan_payload is not None:
                    plan_payload = dict(startup_plan_payload)
                    plan = SimulationPlan.from_payload(startup_plan_payload)
                    execution_request = plan.to_execution_request().to_payload()
                    prepared = startup_prepared
                else:
                    raise SimulationContainmentPayloadError(
                        "Contained simulation request missing simulation_plan_payload."
                    )
                output_queue.put(
                    {
                        "kind": "accepted",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                    }
                )
                if bool(request_payload.get("prepare_only")):
                    startup_plan_payload = dict(plan_payload)
                    startup_prepared = prepared
                    _SimulationChildHandler._prewarm_solver_imports()
                    output_queue.put(
                        {
                            "kind": "result",
                            "owner_epoch": int(owner_epoch),
                            "request_id": request_id,
                            "payload": {"success": True, "prepared": True},
                        }
                    )
                    continue
                result = solve_ode(prepared.request)
                include_mechanism = bool(request_payload.get("include_mechanism_in_result_payload"))
                payload = build_finalized_simulation_result_payload(
                    mechanism=prepared.mechanism,
                    result=result,
                    species_names=list(prepared.species_names),
                    initials_for_algebra=prepared.initials_for_algebra,
                    simulation_plan=plan,
                    preparation_warnings=list(getattr(prepared, "warnings", None) or []),
                    solver=str(getattr(prepared.request, "solver", "") or ""),
                    mechanism_text=str(execution_request.get("mechanism_text") or ""),
                    solver_config=dict(execution_request.get("solver_config") or {}),
                    include_mechanism=include_mechanism,
                )
                output_queue.put(
                    {
                        "kind": "result",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "payload": payload,
                    }
                )
            except SimulationCancelled as exc:
                output_queue.put(
                    {
                        "kind": "cancelled",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "failure": simulation_failure_from_exception(exc),
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - process boundary must serialize all failures
                output_queue.put(
                    {
                        "kind": (
                            "fatal"
                            if isinstance(
                                exc,
                                (
                                    SimulationContainmentPayloadError,
                                    SimulationPreparationError,
                                    TypeError,
                                    ValueError,
                                ),
                            )
                            else "error"
                        ),
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "failure": _failure_payload_with_stack_trace(
                            simulation_failure_from_exception(
                                exc,
                                kind=(
                                    "simulation_containment_reconstruction"
                                    if isinstance(
                                        exc,
                                        (
                                            SimulationContainmentPayloadError,
                                            SimulationPreparationError,
                                            TypeError,
                                            ValueError,
                                        ),
                                    )
                                    else None
                                ),
                            ),
                            exc=exc,
                        ),
                    }
                )
    except BaseException as exc:  # noqa: BLE001 - startup/reconstruction failure is fatal to the owner
        output_queue.put(
            {
                "kind": "fatal",
                "owner_epoch": int(owner_epoch),
                "failure": _failure_payload_with_stack_trace(
                    simulation_failure_from_exception(exc, kind="simulation_containment_startup"),
                    exc=exc,
                ),
            }
        )


def _simulation_owner_child_with_env(
    child_target: Callable[..., Any],
    handler_env: Mapping[str, str],
    simulation_plan_payload: Mapping[str, Any],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    owner_epoch: int,
) -> None:
    for name, value in dict(handler_env or {}).items():
        os.environ[str(name)] = str(value)
    child_target(
        simulation_plan_payload,
        input_queue,
        output_queue,
        owner_epoch,
    )


class WarmSimulationOwner:
    def __init__(
        self,
        simulation_plan_payload: Mapping[str, Any],
        *,
        active_timeout_s: float = _DEFAULT_SIMULATION_ACTIVE_TIMEOUT_S,
        ready_timeout_s: float = _OWNER_READY_TIMEOUT_S,
        accept_timeout_s: float = _OWNER_ACCEPT_TIMEOUT_S,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
        child_target: Optional[Callable[..., Any]] = None,
        handler_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._simulation_plan_payload = dict(simulation_plan_payload)
        self._active_timeout_s = max(0.001, float(active_timeout_s))
        self._ready_timeout_s = max(0.001, float(ready_timeout_s))
        self._accept_timeout_s = max(0.001, float(accept_timeout_s))
        self._mp_context = mp_context or multiprocessing.get_context("spawn")
        self._child_target = child_target or _simulation_owner_child
        self._handler_env = (
            contained_child_blas_thread_env()
            if handler_env is None
            else dict(handler_env)
        )
        self._process: Optional[multiprocessing.Process] = None
        self._input_queue: Optional[multiprocessing.Queue] = None
        self._output_queue: Optional[multiprocessing.Queue] = None
        self._owner_epoch = 0
        self._request_id = 0
        self._runtime_owner: Optional[SimulationRuntimeOwner] = None
        self._lock = threading.RLock()
        if child_target is None:
            self._runtime_owner = SimulationRuntimeOwner(
                handler_import_path="kindred.core.simulation_containment:create_simulation_child_handler",
                startup_payload=dict(self._simulation_plan_payload),
                handler_env=dict(self._handler_env),
                ready_timeout_s=self._ready_timeout_s,
                accept_timeout_s=self._accept_timeout_s,
                mp_context=self._mp_context,
            )

    @property
    def owner_epoch(self) -> int:
        if self._runtime_owner is not None:
            return int(self._runtime_owner.owner_epoch)
        return int(self._owner_epoch)

    @property
    def simulation_plan_payload(self) -> dict[str, Any]:
        return dict(self._simulation_plan_payload)

    @property
    def is_running(self) -> bool:
        if self._runtime_owner is not None:
            return bool(self._runtime_owner.is_running)
        proc = self._process
        return bool(proc is not None and proc.is_alive())

    @property
    def is_ready(self) -> bool:
        if self._runtime_owner is not None:
            return bool(getattr(self._runtime_owner, "is_ready", False))
        return bool(self.is_running)

    def start(
        self,
        *,
        wait: bool = False,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        with self._lock:
            if self._runtime_owner is not None:
                try:
                    self._runtime_owner.warm(wait=bool(wait), cancellation_check=cancellation_check)
                except ContainmentKernelCancelled as exc:
                    raise SimulationCancelled() from exc
                except ContainmentKernelStartupTimeout as exc:
                    raise SimulationContainmentStartupTimeout(exc.timeout_s) from exc
                except ContainmentKernelChildFailure as exc:
                    raise SimulationContainmentChildFailure(exc.failure) from exc
                except ContainmentKernelProtocolError as exc:
                    raise SimulationContainmentProtocolError(str(exc)) from exc
                return
            if bool(wait):
                self._ensure_started(cancellation_check=cancellation_check)

    def prepare_runtime_payload(
        self,
        simulation_plan_payload: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        plan_payload = validate_contained_simulation_payload(dict(simulation_plan_payload or {}))
        with self._lock:
            if self._runtime_owner is not None:
                try:
                    self._runtime_owner.solve(
                        {
                            "simulation_plan_payload": dict(plan_payload),
                            "prepare_only": True,
                        },
                        active_timeout_s=self._active_timeout_s,
                    )
                except ContainmentKernelStartupTimeout as exc:
                    raise SimulationContainmentStartupTimeout(exc.timeout_s) from exc
                except ContainmentKernelAcceptTimeout as exc:
                    raise SimulationContainmentAcceptTimeout(exc.timeout_s) from exc
                except ContainmentKernelActiveTimeout as exc:
                    raise SimulationContainmentTimeout(exc.timeout_s) from exc
                except ContainmentKernelChildFailure as exc:
                    raise SimulationContainmentChildFailure(exc.failure) from exc
                except ContainmentKernelProtocolError as exc:
                    raise SimulationContainmentProtocolError(str(exc)) from exc
                self._simulation_plan_payload = dict(plan_payload)
                return
            if not self.is_running:
                self._simulation_plan_payload = dict(plan_payload)
                if bool(wait):
                    self._ensure_started(cancellation_check=None)
                return
            self.solve({"simulation_plan_payload": dict(plan_payload), "prepare_only": True})
            self._simulation_plan_payload = dict(plan_payload)

    def solve(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._runtime_owner is not None:
                if self._cancel_requested(cancellation_check):
                    self.close(kill=True)
                    raise SimulationCancelled()
                try:
                    return self._runtime_owner.solve(
                        dict(payload or {}),
                        active_timeout_s=self._active_timeout_s,
                        cancellation_check=cancellation_check,
                    )
                except ContainmentKernelCancelled as exc:
                    raise SimulationCancelled() from exc
                except ContainmentKernelStartupTimeout as exc:
                    raise SimulationContainmentStartupTimeout(exc.timeout_s) from exc
                except ContainmentKernelAcceptTimeout as exc:
                    raise SimulationContainmentAcceptTimeout(exc.timeout_s) from exc
                except ContainmentKernelActiveTimeout as exc:
                    raise SimulationContainmentTimeout(exc.timeout_s) from exc
                except ContainmentKernelChildFailure as exc:
                    raise SimulationContainmentChildFailure(exc.failure) from exc
                except ContainmentKernelProtocolError as exc:
                    raise SimulationContainmentProtocolError(str(exc)) from exc
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise SimulationCancelled()
            self._ensure_started(cancellation_check=cancellation_check)
            if self._input_queue is None or self._output_queue is None:
                raise SimulationContainmentProtocolError("Contained simulation owner queues are unavailable.")

            self._request_id += 1
            request_id = int(self._request_id)
            gate = SimulationReplyGate(owner_epoch=int(self._owner_epoch), request_id=request_id)
            self._input_queue.put(
                {
                    "kind": "solve",
                    "owner_epoch": int(self._owner_epoch),
                    "request_id": request_id,
                    "payload": dict(payload or {}),
                }
            )
            self._wait_for_acceptance(
                gate,
                cancellation_check=cancellation_check,
            )

            deadline = time.monotonic() + self._active_timeout_s
            while True:
                if self._cancel_requested(cancellation_check):
                    self.close(kill=True)
                    raise SimulationCancelled()
                if time.monotonic() >= deadline:
                    self.close(kill=True)
                    raise SimulationContainmentTimeout(self._active_timeout_s)
                try:
                    message = self._output_queue.get(timeout=_OWNER_POLL_INTERVAL_S)
                except queue.Empty:
                    self._raise_if_process_exited()
                    continue
                if not isinstance(message, Mapping):
                    raise SimulationContainmentProtocolError("Contained simulation owner returned a non-mapping reply.")
                if not gate.is_current(message):
                    continue
                kind = str(message.get("kind") or "")
                if kind == "progress":
                    continue
                if kind == "accepted":
                    continue
                if kind == "result":
                    result_payload = message.get("payload")
                    if isinstance(result_payload, Mapping):
                        return dict(result_payload)
                    raise SimulationContainmentProtocolError("Contained simulation owner returned malformed result payload.")
                if kind == "cancelled":
                    raise SimulationCancelled()
                if kind == "error":
                    failure = message.get("failure")
                    raise SimulationContainmentChildFailure(failure if isinstance(failure, Mapping) else {})
                if kind == "fatal":
                    failure = message.get("failure")
                    raise SimulationContainmentChildFailure(failure if isinstance(failure, Mapping) else {})
                raise SimulationContainmentProtocolError(
                    f"Contained simulation owner returned unexpected reply kind: {kind!r}."
                )

    def cancel(self) -> dict[str, Any]:
        self.close(kill=True)
        return build_simulation_failure(
            "cancelled",
            "Simulation cancelled by user",
            code="E305",
            exc_type="SimulationCancelled",
        )

    def close(self, *, kill: bool = False) -> None:
        if self._runtime_owner is not None:
            self._runtime_owner.close(kill=bool(kill))
            return
        proc = self._process
        input_queue = self._input_queue
        output_queue = self._output_queue
        self._process = None
        self._input_queue = None
        self._output_queue = None

        if proc is not None:
            if not kill and proc.is_alive() and input_queue is not None:
                try:
                    input_queue.put({"kind": "close", "owner_epoch": int(self._owner_epoch)})
                    proc.join(timeout=_PROCESS_JOIN_TIMEOUT_S)
                except (OSError, EOFError, BrokenPipeError, ValueError):
                    pass
            if proc.is_alive():
                _terminate_process(proc)
            else:
                proc.join(timeout=_PROCESS_JOIN_TIMEOUT_S)

        for owned_queue in (input_queue, output_queue):
            if owned_queue is None:
                continue
            try:
                owned_queue.close()
                owned_queue.join_thread()
            except (OSError, ValueError, AttributeError):
                pass

    def _ensure_started(self, *, cancellation_check: Optional[Callable[[], bool]]) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self.close(kill=True)
        self._owner_epoch += 1
        self._input_queue = self._mp_context.Queue()
        self._output_queue = self._mp_context.Queue()
        target = self._child_target
        args = (
            dict(self._simulation_plan_payload),
            self._input_queue,
            self._output_queue,
            int(self._owner_epoch),
        )
        if self._handler_env:
            target = _simulation_owner_child_with_env
            args = (self._child_target, dict(self._handler_env), *args)
        self._process = self._mp_context.Process(
            target=target,
            args=args,
        )
        _start_process_with_env(self._process, self._handler_env)

        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise SimulationCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                raise SimulationContainmentStartupTimeout(self._ready_timeout_s)
            assert self._output_queue is not None
            try:
                message = self._output_queue.get(timeout=_OWNER_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise SimulationContainmentProtocolError("Contained simulation owner returned a non-mapping startup reply.")
            if int(message.get("owner_epoch", -1)) != int(self._owner_epoch):
                continue
            kind = str(message.get("kind") or "")
            if kind == "ready":
                return
            if kind == "fatal":
                failure = message.get("failure")
                raise SimulationContainmentChildFailure(failure if isinstance(failure, Mapping) else {})
            raise SimulationContainmentProtocolError(
                f"Contained simulation owner returned unexpected startup reply: {kind!r}."
            )

    def _wait_for_acceptance(
        self,
        gate: SimulationReplyGate,
        *,
        cancellation_check: Optional[Callable[[], bool]],
    ) -> None:
        if self._output_queue is None:
            raise SimulationContainmentProtocolError("Contained simulation owner output queue is unavailable.")
        deadline = time.monotonic() + self._accept_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise SimulationCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                raise SimulationContainmentAcceptTimeout(self._accept_timeout_s)
            try:
                message = self._output_queue.get(timeout=_OWNER_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise SimulationContainmentProtocolError("Contained simulation owner returned a non-mapping reply.")
            if not gate.is_current(message):
                continue
            kind = str(message.get("kind") or "")
            if kind == "accepted":
                return
            if kind == "fatal":
                failure = message.get("failure")
                raise SimulationContainmentChildFailure(failure if isinstance(failure, Mapping) else {})
            raise SimulationContainmentProtocolError(
                f"Contained simulation owner replied before accepting request {gate.request_id}: {kind!r}."
            )

    def _raise_if_process_exited(self) -> None:
        proc = self._process
        if proc is None:
            raise SimulationContainmentProtocolError("Contained simulation owner is not running.")
        if proc.is_alive():
            return
        proc.join(timeout=0.1)
        raise SimulationContainmentProtocolError(
            f"Contained simulation owner exited unexpectedly with code {proc.exitcode}."
        )

    @staticmethod
    def _cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> bool:
        if cancellation_check is None:
            return False
        check = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
        return bool(check())

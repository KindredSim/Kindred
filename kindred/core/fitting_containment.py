from __future__ import annotations

import multiprocessing
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from kindred.core.exceptions import FitSimulationError, FittingCancelled, SimulationCancelled
from kindred.core.fitting_evaluation import SerialFittingEvaluator
from kindred.core.simulation_failure import (
    build_simulation_failure,
    coerce_simulation_failure,
    simulation_failure_from_exception,
)
from kindred.core.simulation_series_payload import coerce_simulation_series_payload

_DEFAULT_FITTING_SOLVE_TIMEOUT_S = 60.0
_LANE_READY_TIMEOUT_S = 30.0
_LANE_ACCEPT_TIMEOUT_S = 10.0
_LANE_POLL_INTERVAL_S = 0.02


@dataclass(frozen=True)
class _FittingLaneReplyGate:
    owner_epoch: int

    def is_current(self, message: Mapping[str, Any], *, request_id: int) -> bool:
        return (
            int(message.get("owner_epoch", -1)) == int(self.owner_epoch)
            and int(message.get("request_id", -1)) == int(request_id)
        )


class FittingLaneProtocolError(RuntimeError):
    """Raised when the contained fitting lane violates the request protocol."""


class FittingLaneTimeout(FitSimulationError):
    """Nonfatal fitting-candidate timeout reported after a lane accepts work."""

    def __init__(
        self,
        timeout_s: float,
        *,
        failed_params: Optional[Dict[str, float]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        resolved_message = message or f"Fitting simulation timed out after {self.timeout_s:.3g} seconds."
        super().__init__(
            resolved_message,
            failed_params=failed_params,
            details={
                "fatal": False,
                "failure": build_simulation_failure(
                    "timeout",
                    resolved_message,
                    code="E306",
                    details={"active_solve_timeout_s": self.timeout_s},
                    exc_type=type(self).__name__,
                ),
            },
        )


def _fatal_fit_simulation_error(
    kind: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    failure: Optional[Mapping[str, Any]] = None,
) -> FitSimulationError:
    if failure is None:
        failure_payload = build_simulation_failure(
            kind,
            message,
            exc_type=type(exc).__name__ if exc is not None else None,
        )
    else:
        failure_payload = coerce_simulation_failure(dict(failure))
    return FitSimulationError(
        message,
        details={
            "fatal": True,
            "failure": failure_payload,
        },
    )


def _terminate_process(proc: multiprocessing.Process, *, join_timeout_s: float = 0.5) -> None:
    if not proc.is_alive():
        proc.join(timeout=join_timeout_s)
        return
    proc.terminate()
    proc.join(timeout=join_timeout_s)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(timeout=join_timeout_s)


def _serialize_fit_error(exc: FitSimulationError) -> Dict[str, Any]:
    details = dict(getattr(exc, "details", {}) or {})
    failure = details.get("failure")
    if failure is None:
        failure = simulation_failure_from_exception(exc)
    return {
        "message": str(getattr(exc, "message", None) or str(exc)),
        "failed_params": dict(getattr(exc, "failed_params", None) or {}),
        "details": details,
        "failure": coerce_simulation_failure(failure),
    }


def _build_fit_error_from_payload(payload: Mapping[str, Any]) -> FitSimulationError:
    details = dict(payload.get("details") or {})
    if "failure" not in details:
        details["failure"] = coerce_simulation_failure(payload.get("failure"))
    failed_params = payload.get("failed_params")
    return FitSimulationError(
        str(payload.get("message") or "Contained fitting simulation failed."),
        failed_params=dict(failed_params) if isinstance(failed_params, Mapping) else None,
        details=details,
    )


def _fitting_lane_child(
    process_payload: Mapping[str, Any],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    owner_epoch: int,
) -> None:
    try:
        evaluator = SerialFittingEvaluator.from_process_payload(process_payload)
        evaluator._ensure_prepared()
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
            if kind != "evaluate":
                continue
            request_id = int(request.get("request_id", -1))
            output_queue.put(
                {
                    "kind": "accepted",
                    "owner_epoch": int(owner_epoch),
                    "request_id": request_id,
                }
            )
            try:
                result = evaluator.evaluate_series_with_parameter_origins(
                    dict(request.get("params") or {}),
                    dict(request.get("origins") or {}),
                    failed_params=dict(request.get("failed_params") or {}),
                )
                output_queue.put(
                    {
                        "kind": "result",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "payload": coerce_simulation_series_payload(result).to_legacy_dict(),
                    }
                )
            except FitSimulationError as exc:
                output_queue.put(
                    {
                        "kind": "fit_error",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "error": _serialize_fit_error(exc),
                    }
                )
            except (FittingCancelled, SimulationCancelled) as exc:
                output_queue.put(
                    {
                        "kind": "cancelled",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "failure": simulation_failure_from_exception(exc),
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - process boundary must serialize failures
                output_queue.put(
                    {
                        "kind": "fit_error",
                        "owner_epoch": int(owner_epoch),
                        "request_id": request_id,
                        "error": {
                            "message": f"Fitting simulation failed: {exc}",
                            "failed_params": dict(request.get("failed_params") or {}),
                            "details": {"fatal": False},
                            "failure": simulation_failure_from_exception(exc),
                        },
                    }
                )
    except BaseException as exc:  # noqa: BLE001 - child startup/prewarm failure is fatal to the lane
        output_queue.put(
            {
                "kind": "fatal",
                "owner_epoch": int(owner_epoch),
                "failure": simulation_failure_from_exception(exc, kind="fitting_containment_prewarm"),
            }
        )


class WarmFittingEvaluatorLane:
    def __init__(
        self,
        process_payload: Mapping[str, Any],
        *,
        request_timeout_s: float = _DEFAULT_FITTING_SOLVE_TIMEOUT_S,
        ready_timeout_s: float = _LANE_READY_TIMEOUT_S,
        accept_timeout_s: float = _LANE_ACCEPT_TIMEOUT_S,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
        child_target: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._process_payload = dict(process_payload)
        self._request_timeout_s = max(0.001, float(request_timeout_s))
        self._ready_timeout_s = max(0.001, float(ready_timeout_s))
        self._accept_timeout_s = max(0.001, float(accept_timeout_s))
        self._mp_context = mp_context or multiprocessing.get_context("spawn")
        self._child_target = child_target or _fitting_lane_child
        self._process: Optional[multiprocessing.Process] = None
        self._input_queue: Optional[multiprocessing.Queue] = None
        self._output_queue: Optional[multiprocessing.Queue] = None
        self._owner_epoch = 0
        self._request_id = 0
        self._closed = False

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ):
        if self._cancel_requested(cancellation_check):
            raise FittingCancelled()
        self._ensure_started(cancellation_check=cancellation_check)
        if self._input_queue is None or self._output_queue is None:
            raise FittingLaneProtocolError("Contained fitting lane queues are unavailable.")

        self._request_id += 1
        request_id = int(self._request_id)
        gate = _FittingLaneReplyGate(owner_epoch=int(self._owner_epoch))
        self._input_queue.put(
            {
                "kind": "evaluate",
                "owner_epoch": int(self._owner_epoch),
                "request_id": request_id,
                "params": dict(params or {}),
                "origins": dict(origins or {}),
                "failed_params": dict(failed_params or {}),
            }
        )

        self._wait_for_acceptance(
            gate,
            request_id=request_id,
            cancellation_check=cancellation_check,
        )
        deadline = time.monotonic() + self._request_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise FittingCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                raise FittingLaneTimeout(self._request_timeout_s, failed_params=failed_params)
            try:
                message = self._output_queue.get(timeout=_LANE_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise FittingLaneProtocolError("Contained fitting lane returned a non-mapping reply.")
            if not gate.is_current(message, request_id=request_id):
                continue
            kind = str(message.get("kind") or "")
            if kind == "result":
                return coerce_simulation_series_payload(message.get("payload") or {})
            if kind == "fit_error":
                error_payload = message.get("error")
                if isinstance(error_payload, Mapping):
                    raise _build_fit_error_from_payload(error_payload)
                raise FittingLaneProtocolError("Contained fitting lane returned a malformed error reply.")
            if kind == "cancelled":
                raise FittingCancelled()
            if kind == "fatal":
                failure = message.get("failure")
                raise _fatal_fit_simulation_error(
                    "fitting_containment_protocol",
                    "Contained fitting lane failed fatally.",
                    failure=failure if isinstance(failure, Mapping) else None,
                )
            if kind == "accepted":
                continue
            raise FittingLaneProtocolError(f"Contained fitting lane returned unexpected reply kind: {kind!r}.")

    def warm(self, *, cancellation_check: Optional[Callable[[], bool]] = None) -> None:
        self._ensure_started(cancellation_check=cancellation_check)

    def close(self, *, kill: bool = False) -> None:
        self._closed = True
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
                    proc.join(timeout=0.5)
                except (OSError, EOFError, BrokenPipeError):
                    pass
            if proc.is_alive():
                _terminate_process(proc)
            else:
                proc.join(timeout=0.5)
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
        self._closed = False
        self._owner_epoch += 1
        self._input_queue = self._mp_context.Queue()
        self._output_queue = self._mp_context.Queue()
        self._process = self._mp_context.Process(
            target=self._child_target,
            args=(
                dict(self._process_payload),
                self._input_queue,
                self._output_queue,
                int(self._owner_epoch),
            ),
        )
        self._process.start()
        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise FittingCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                raise _fatal_fit_simulation_error(
                    "fitting_containment_prewarm",
                    "Contained fitting lane did not become ready.",
                )
            assert self._output_queue is not None
            try:
                message = self._output_queue.get(timeout=_LANE_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise FittingLaneProtocolError("Contained fitting lane returned a non-mapping startup reply.")
            if int(message.get("owner_epoch", -1)) != int(self._owner_epoch):
                continue
            kind = str(message.get("kind") or "")
            if kind == "ready":
                return
            if kind == "fatal":
                failure = message.get("failure")
                raise _fatal_fit_simulation_error(
                    "fitting_containment_prewarm",
                    "Contained fitting lane failed before accepting work.",
                    failure=failure if isinstance(failure, Mapping) else None,
                )
            raise FittingLaneProtocolError(f"Contained fitting lane returned unexpected startup reply: {kind!r}.")

    def _wait_for_acceptance(
        self,
        gate: _FittingLaneReplyGate,
        *,
        request_id: int,
        cancellation_check: Optional[Callable[[], bool]],
    ) -> None:
        if self._output_queue is None:
            raise FittingLaneProtocolError("Contained fitting lane output queue is unavailable.")
        deadline = time.monotonic() + self._accept_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise FittingCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                raise FittingLaneProtocolError("Contained fitting lane did not accept the request.")
            try:
                message = self._output_queue.get(timeout=_LANE_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise FittingLaneProtocolError("Contained fitting lane returned a non-mapping reply.")
            if not gate.is_current(message, request_id=request_id):
                continue
            kind = str(message.get("kind") or "")
            if kind == "accepted":
                return
            if kind == "fatal":
                failure = message.get("failure")
                raise _fatal_fit_simulation_error(
                    "fitting_containment_protocol",
                    "Contained fitting lane failed before accepting the request.",
                    failure=failure if isinstance(failure, Mapping) else None,
                )
            raise FittingLaneProtocolError(
                f"Contained fitting lane replied before accepting request {request_id}: {kind!r}."
            )

    def _raise_if_process_exited(self) -> None:
        proc = self._process
        if proc is None:
            raise FittingLaneProtocolError("Contained fitting lane is not running.")
        if proc.is_alive():
            return
        proc.join(timeout=0.1)
        raise FittingLaneProtocolError(
            f"Contained fitting lane exited unexpectedly with code {proc.exitcode}."
        )

    @staticmethod
    def _cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> bool:
        if cancellation_check is None:
            return False
        check = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
        return bool(check())

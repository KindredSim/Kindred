from __future__ import annotations

import importlib
import multiprocessing
import os
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

_DEFAULT_READY_TIMEOUT_S = 30.0
_DEFAULT_ACCEPT_TIMEOUT_S = 10.0
_DEFAULT_ACTIVE_TIMEOUT_S = 60.0
_DEFAULT_EVENT_HISTORY_LIMIT = 256
_POLL_INTERVAL_S = 0.02
_PROCESS_JOIN_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class ContainmentHandlerSpec:
    import_path: str
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not str(self.import_path or "").strip():
            raise ValueError("Containment handler import_path must not be empty.")
        object.__setattr__(self, "import_path", str(self.import_path))
        object.__setattr__(self, "env", {str(k): str(v) for k, v in dict(self.env or {}).items()})


@dataclass(frozen=True)
class ContainmentKernelEvent:
    kind: str
    owner_epoch: int
    request_id: Optional[int] = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "owner_epoch", int(self.owner_epoch))
        object.__setattr__(self, "request_id", None if self.request_id is None else int(self.request_id))
        object.__setattr__(self, "details", dict(self.details or {}))


@dataclass(frozen=True)
class ContainmentHandlerResponse:
    kind: str
    payload: Mapping[str, Any] | None = None
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "payload", None if self.payload is None else dict(self.payload))
        object.__setattr__(self, "failure", None if self.failure is None else dict(self.failure))


@dataclass(frozen=True)
class ContainmentChildContext:
    output_queue: Any
    owner_epoch: int
    request_id: int
    reply_fields: Mapping[str, Any] | None = None

    def put(self, message: Mapping[str, Any]) -> None:
        self.output_queue.put(dict(message))


@dataclass(frozen=True)
class _ReplyGate:
    owner_epoch: int
    request_id: int
    expected_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_fields", dict(self.expected_fields or {}))

    def is_current(self, message: Mapping[str, Any]) -> bool:
        if not (
            int(message.get("owner_epoch", -1)) == int(self.owner_epoch)
            and int(message.get("request_id", -1)) == int(self.request_id)
        ):
            return False
        return all(message.get(name) == value for name, value in dict(self.expected_fields or {}).items())


class ContainmentKernelProtocolError(RuntimeError):
    pass


class ContainmentKernelChildFailure(ContainmentKernelProtocolError):
    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure or {})
        super().__init__(str(self.failure.get("message") or "Contained child failed."))


class ContainmentKernelStartupTimeout(ContainmentKernelProtocolError):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(f"Contained child did not become ready within {self.timeout_s:.3g} seconds.")


class ContainmentKernelAcceptTimeout(ContainmentKernelProtocolError):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(f"Contained child did not accept the request within {self.timeout_s:.3g} seconds.")


class ContainmentKernelActiveTimeout(RuntimeError):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(f"Contained request timed out after {self.timeout_s:.3g} seconds.")


class ContainmentKernelCancelled(RuntimeError):
    pass


class ContainmentChildFatal(RuntimeError):
    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure or {})
        super().__init__(str(self.failure.get("message") or "Contained child failed."))


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _failure_from_exception(exc: BaseException, *, kind: str = "containment_error") -> dict[str, Any]:
    return {
        "kind": str(kind),
        "message": str(exc),
        "exc_type": type(exc).__name__,
        "context": {"stack_trace": _format_exception(exc)},
    }


def _apply_env(env: Mapping[str, str]) -> None:
    for name, value in dict(env or {}).items():
        os.environ[str(name)] = str(value)


def _import_from_path(import_path: str) -> Callable[..., Any]:
    module_name, sep, attr_name = str(import_path).partition(":")
    if not sep:
        module_name, _, attr_name = str(import_path).rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid containment handler import path: {import_path!r}")
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if not callable(target):
        raise TypeError(f"Containment handler is not callable: {import_path!r}")
    return target


def _call_optional_before_accept(handler: Any, payload: Mapping[str, Any], context: ContainmentChildContext) -> None:
    before_accept = getattr(handler, "before_accept", None)
    if callable(before_accept):
        before_accept(dict(payload or {}), context)


def _handle_request(handler: Any, payload: Mapping[str, Any], context: ContainmentChildContext) -> Any:
    handle = getattr(handler, "handle_request", None)
    if callable(handle):
        return handle(dict(payload or {}), context)
    if callable(handler):
        return handler(dict(payload or {}), context)
    raise TypeError("Containment handler must define handle_request() or be callable.")


def _put_handler_response(
    output_queue: Any,
    *,
    owner_epoch: int,
    request_id: int,
    reply_fields: Mapping[str, Any],
    response: Any,
) -> None:
    if isinstance(response, ContainmentHandlerResponse):
        message: dict[str, Any] = {
            "kind": str(response.kind),
            "owner_epoch": int(owner_epoch),
            "request_id": int(request_id),
            **dict(reply_fields or {}),
        }
        if response.payload is not None:
            message["payload"] = dict(response.payload)
        if response.failure is not None:
            message["failure"] = dict(response.failure)
        output_queue.put(message)
        return
    output_queue.put(
        {
            "kind": "result",
            "owner_epoch": int(owner_epoch),
            "request_id": int(request_id),
            **dict(reply_fields or {}),
            "payload": dict(response) if isinstance(response, Mapping) else response,
        }
    )


def _containment_child_main(
    handler_spec: ContainmentHandlerSpec,
    startup_payload: Mapping[str, Any],
    input_queue: Any,
    output_queue: Any,
    owner_epoch: int,
) -> None:
    try:
        _apply_env(dict(handler_spec.env or {}))
        factory = _import_from_path(handler_spec.import_path)
        startup = dict(startup_payload or {})
        startup.setdefault("owner_epoch", int(owner_epoch))
        handler = factory(startup)
        output_queue.put({"kind": "ready", "owner_epoch": int(owner_epoch), "pid": int(os.getpid())})
        while True:
            request = input_queue.get()
            if not isinstance(request, Mapping):
                continue
            kind = str(request.get("kind") or "")
            if kind == "close":
                return
            if kind != "request":
                continue
            request_id = int(request.get("request_id", -1))
            payload = dict(request.get("payload") or {})
            reply_fields = dict(request.get("reply_fields") or {})
            context = ContainmentChildContext(
                output_queue=output_queue,
                owner_epoch=int(owner_epoch),
                request_id=int(request_id),
                reply_fields=dict(reply_fields),
            )
            try:
                _call_optional_before_accept(handler, payload, context)
                output_queue.put(
                    {
                        "kind": "accepted",
                        "owner_epoch": int(owner_epoch),
                        "request_id": int(request_id),
                        **dict(reply_fields),
                    }
                )
                response = _handle_request(handler, payload, context)
                _put_handler_response(
                    output_queue,
                    owner_epoch=int(owner_epoch),
                    request_id=int(request_id),
                    reply_fields=dict(reply_fields),
                    response=response,
                )
            except ContainmentChildFatal as exc:
                output_queue.put(
                    {
                        "kind": "fatal",
                        "owner_epoch": int(owner_epoch),
                        "request_id": int(request_id),
                        **dict(reply_fields),
                        "failure": dict(exc.failure),
                    }
                )
                return
            except BaseException as exc:  # noqa: BLE001 - process boundary must serialize failures
                output_queue.put(
                    {
                        "kind": "error",
                        "owner_epoch": int(owner_epoch),
                        "request_id": int(request_id),
                        **dict(reply_fields),
                        "failure": _failure_from_exception(exc),
                    }
                )
    except ContainmentChildFatal as exc:
        output_queue.put({"kind": "fatal", "owner_epoch": int(owner_epoch), "failure": dict(exc.failure)})
    except BaseException as exc:  # noqa: BLE001 - startup/import failures must cross the boundary
        output_queue.put(
            {
                "kind": "fatal",
                "owner_epoch": int(owner_epoch),
                "failure": _failure_from_exception(exc, kind="containment_startup"),
            }
        )


def _process_was_started(proc: multiprocessing.Process) -> bool:
    try:
        return getattr(proc, "pid", None) is not None
    except Exception:
        return False


def _join_started_process(proc: multiprocessing.Process, *, timeout: float) -> None:
    if not _process_was_started(proc):
        return
    proc.join(timeout=timeout)


def _terminate_process(proc: multiprocessing.Process, *, join_timeout_s: float = _PROCESS_JOIN_TIMEOUT_S) -> None:
    if not _process_was_started(proc):
        return
    if not proc.is_alive():
        _join_started_process(proc, timeout=join_timeout_s)
        return
    proc.terminate()
    _join_started_process(proc, timeout=join_timeout_s)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        _join_started_process(proc, timeout=join_timeout_s)


class ContainmentKernelOwner:
    def __init__(
        self,
        handler_spec: ContainmentHandlerSpec,
        *,
        startup_payload: Mapping[str, Any] | None = None,
        ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        accept_timeout_s: float = _DEFAULT_ACCEPT_TIMEOUT_S,
        event_history_limit: int = _DEFAULT_EVENT_HISTORY_LIMIT,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        self._handler_spec = handler_spec
        self._startup_payload = dict(startup_payload or {})
        self._ready_timeout_s = max(0.001, float(ready_timeout_s))
        self._accept_timeout_s = max(0.001, float(accept_timeout_s))
        self._event_history_limit = max(1, int(event_history_limit))
        self._mp_context = mp_context or multiprocessing.get_context("spawn")
        self._process: Optional[multiprocessing.Process] = None
        self._input_queue: Any = None
        self._output_queue: Any = None
        self._owner_epoch = 0
        self._request_id = 0
        self._ready = False
        self._events: deque[ContainmentKernelEvent] = deque(maxlen=self._event_history_limit)
        self._lifecycle_lock = threading.RLock()

    @property
    def owner_epoch(self) -> int:
        return int(self._owner_epoch)

    @property
    def startup_payload(self) -> dict[str, Any]:
        return dict(self._startup_payload)

    @property
    def is_running(self) -> bool:
        proc = self._process
        return bool(proc is not None and proc.is_alive())

    @property
    def is_ready(self) -> bool:
        return bool(self._ready and self.is_running)

    def drain_events(self) -> list[ContainmentKernelEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def start(
        self,
        *,
        wait: bool = True,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        with self._lifecycle_lock:
            if self._process is not None and self._process.is_alive():
                if wait and not self._ready:
                    self._wait_for_ready(cancellation_check=cancellation_check)
                return
            self.close(kill=True)
            self._owner_epoch += 1
            self._ready = False
            self._input_queue = self._mp_context.Queue()
            self._output_queue = self._mp_context.Queue()
            process = self._mp_context.Process(
                target=_containment_child_main,
                args=(
                    self._handler_spec,
                    dict(self._startup_payload),
                    self._input_queue,
                    self._output_queue,
                    int(self._owner_epoch),
                ),
            )
            self._process = process
            self._events.append(ContainmentKernelEvent(kind="owner_starting", owner_epoch=int(self._owner_epoch)))
            process.start()
        if wait:
            self._wait_for_ready(cancellation_check=cancellation_check)

    def request(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        active_timeout_s: float = _DEFAULT_ACTIVE_TIMEOUT_S,
        cancellation_check: Optional[Callable[[], bool]] = None,
        reply_fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._cancel_requested(cancellation_check):
            self.close(kill=True)
            raise ContainmentKernelCancelled()
        self.start(wait=True, cancellation_check=cancellation_check)
        if self._input_queue is None:
            raise ContainmentKernelProtocolError("Contained child input queue is unavailable.")
        self._request_id += 1
        request_id = int(self._request_id)
        expected_fields = dict(reply_fields or {})
        gate = _ReplyGate(
            owner_epoch=int(self._owner_epoch),
            request_id=request_id,
            expected_fields=expected_fields,
        )
        self._events.append(
            ContainmentKernelEvent(kind="request_submitted", owner_epoch=int(self._owner_epoch), request_id=request_id)
        )
        self._input_queue.put(
            {
                "kind": "request",
                "owner_epoch": int(self._owner_epoch),
                "request_id": request_id,
                "reply_fields": expected_fields,
                "payload": dict(payload or {}),
            }
        )
        self._wait_for_acceptance(gate, cancellation_check=cancellation_check)
        return self._wait_for_result(
            gate,
            active_timeout_s=max(0.001, float(active_timeout_s)),
            cancellation_check=cancellation_check,
        )

    def close(self, *, kill: bool = False) -> None:
        with self._lifecycle_lock:
            proc = self._process
            input_queue = self._input_queue
            output_queue = self._output_queue
            self._process = None
            self._input_queue = None
            self._output_queue = None
            self._ready = False

        if proc is not None and _process_was_started(proc):
            if not kill and proc.is_alive() and input_queue is not None:
                try:
                    input_queue.put({"kind": "close", "owner_epoch": int(self._owner_epoch)})
                    _join_started_process(proc, timeout=_PROCESS_JOIN_TIMEOUT_S)
                except (OSError, EOFError, BrokenPipeError, ValueError):
                    pass
            if proc.is_alive():
                _terminate_process(proc)
            else:
                _join_started_process(proc, timeout=_PROCESS_JOIN_TIMEOUT_S)

        for owned_queue in (input_queue, output_queue):
            if owned_queue is None:
                continue
            try:
                owned_queue.close()
                owned_queue.join_thread()
            except (OSError, ValueError, AttributeError):
                pass

    def _wait_for_ready(self, *, cancellation_check: Optional[Callable[[], bool]]) -> None:
        if self._output_queue is None:
            raise ContainmentKernelProtocolError("Contained child output queue is unavailable.")
        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise ContainmentKernelCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                self._events.append(
                    ContainmentKernelEvent(kind="startup_timeout", owner_epoch=int(self._owner_epoch))
                )
                raise ContainmentKernelStartupTimeout(self._ready_timeout_s)
            try:
                message = self._output_queue.get(timeout=_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise ContainmentKernelProtocolError("Contained child returned a non-mapping startup reply.")
            if int(message.get("owner_epoch", -1)) != int(self._owner_epoch):
                self._events.append(ContainmentKernelEvent(kind="stale_ignored", owner_epoch=int(self._owner_epoch)))
                continue
            kind = str(message.get("kind") or "")
            if kind == "ready":
                self._ready = True
                self._events.append(ContainmentKernelEvent(kind="owner_ready", owner_epoch=int(self._owner_epoch)))
                return
            if kind == "fatal":
                self.close(kill=True)
                failure = message.get("failure")
                raise ContainmentKernelChildFailure(failure if isinstance(failure, Mapping) else {})
            raise ContainmentKernelProtocolError(f"Contained child returned unexpected startup reply: {kind!r}.")

    def _wait_for_acceptance(
        self,
        gate: _ReplyGate,
        *,
        cancellation_check: Optional[Callable[[], bool]],
    ) -> None:
        if self._output_queue is None:
            raise ContainmentKernelProtocolError("Contained child output queue is unavailable.")
        deadline = time.monotonic() + self._accept_timeout_s
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise ContainmentKernelCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                self._events.append(
                    ContainmentKernelEvent(
                        kind="accept_timeout",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                raise ContainmentKernelAcceptTimeout(self._accept_timeout_s)
            try:
                message = self._output_queue.get(timeout=_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise ContainmentKernelProtocolError("Contained child returned a non-mapping reply.")
            if not gate.is_current(message):
                self._events.append(
                    ContainmentKernelEvent(
                        kind="stale_ignored",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                continue
            kind = str(message.get("kind") or "")
            if kind == "accepted":
                self._events.append(
                    ContainmentKernelEvent(
                        kind="request_accepted",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                return
            if kind == "cancelled":
                self.close(kill=True)
                raise ContainmentKernelCancelled()
            if kind in {"fatal", "error"}:
                failure = message.get("failure")
                if kind == "fatal":
                    self.close(kill=True)
                raise ContainmentKernelChildFailure(failure if isinstance(failure, Mapping) else {})
            raise ContainmentKernelProtocolError(
                f"Contained child replied before accepting request {gate.request_id}: {kind!r}."
            )

    def _wait_for_result(
        self,
        gate: _ReplyGate,
        *,
        active_timeout_s: float,
        cancellation_check: Optional[Callable[[], bool]],
    ) -> dict[str, Any]:
        if self._output_queue is None:
            raise ContainmentKernelProtocolError("Contained child output queue is unavailable.")
        deadline = time.monotonic() + float(active_timeout_s)
        while True:
            if self._cancel_requested(cancellation_check):
                self.close(kill=True)
                raise ContainmentKernelCancelled()
            if time.monotonic() >= deadline:
                self.close(kill=True)
                self._events.append(
                    ContainmentKernelEvent(
                        kind="request_timeout",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                        details={"active_timeout_s": float(active_timeout_s)},
                    )
                )
                raise ContainmentKernelActiveTimeout(active_timeout_s)
            try:
                message = self._output_queue.get(timeout=_POLL_INTERVAL_S)
            except queue.Empty:
                self._raise_if_process_exited()
                continue
            if not isinstance(message, Mapping):
                raise ContainmentKernelProtocolError("Contained child returned a non-mapping reply.")
            if not gate.is_current(message):
                self._events.append(
                    ContainmentKernelEvent(
                        kind="stale_ignored",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                continue
            kind = str(message.get("kind") or "")
            if kind == "accepted":
                continue
            if kind == "result":
                payload = message.get("payload")
                if isinstance(payload, Mapping):
                    result = dict(payload)
                else:
                    raise ContainmentKernelProtocolError("Contained child returned malformed result payload.")
                self._events.append(
                    ContainmentKernelEvent(
                        kind="request_result",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                return result
            if kind in {"fatal", "error", "cancelled"}:
                failure = message.get("failure")
                self._events.append(
                    ContainmentKernelEvent(
                        kind=f"request_{kind}",
                        owner_epoch=int(gate.owner_epoch),
                        request_id=int(gate.request_id),
                    )
                )
                if kind == "cancelled":
                    self.close(kill=True)
                    raise ContainmentKernelCancelled()
                if kind == "fatal":
                    self.close(kill=True)
                raise ContainmentKernelChildFailure(failure if isinstance(failure, Mapping) else {})
            raise ContainmentKernelProtocolError(f"Contained child returned unexpected reply kind: {kind!r}.")

    def _raise_if_process_exited(self) -> None:
        proc = self._process
        if proc is None:
            raise ContainmentKernelProtocolError("Contained child is not running.")
        if not _process_was_started(proc):
            raise ContainmentKernelProtocolError("Contained child has not started.")
        if proc.is_alive():
            return
        _join_started_process(proc, timeout=0.1)
        raise ContainmentKernelProtocolError(f"Contained child exited unexpectedly with code {proc.exitcode}.")

    @staticmethod
    def _cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> bool:
        if cancellation_check is None:
            return False
        check = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
        try:
            return bool(check())
        except TypeError:
            return bool(cancellation_check())

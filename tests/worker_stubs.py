from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PySide6 import QtCore


class ImmediateWorker(QtCore.QObject):
    """Lightweight SimulationWorker replacement for tests (immediate completion)."""

    progress = QtCore.Signal(int, str)
    result_ready = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(
        self,
        mechanism_text,
        initials,
        t_span,
        solver_config,
        parent=None,
        prepared=None,
        include_mechanism_in_result_payload=True,
    ):
        super().__init__(parent)
        self._running = False
        self._mechanism_text = str(mechanism_text)
        self._initials = dict(initials or {})
        self._t_span = tuple(t_span) if t_span is not None else (0.0, 0.0)
        self._solver_config = dict(solver_config or {})
        self._prepared = prepared
        self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
        self._payload = self._build_payload()
        self._fast_mode = False

    @staticmethod
    def _build_payload() -> dict:
        t = np.linspace(0.0, 1.0, 6)
        y = np.vstack(
            [
                np.linspace(1.0, 0.2, t.size),
                np.linspace(0.0, 0.8, t.size),
            ]
        )
        return {
            "t": t,
            "Y": y,
            "species_names": ["A", "B"],
            "mechanism": None,
            "mechanism_text": "reaction: A -> B; k=0.5\ninitial: A=1.0\ninitial: B=0.0",
            "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12},
        }

    def start(self) -> None:
        self._running = True
        self.progress.emit(100, "complete")
        self.result_ready.emit(self._payload)
        self._running = False

    def cancel(self) -> None:
        self._running = False

    def isRunning(self) -> bool:
        return bool(self._running)

    def wait(self, *_args, **_kwargs) -> bool:
        self._running = False
        return True

    def terminate(self) -> None:
        self._running = False


def make_simulation_worker_stub(
    *,
    on_init: Optional[Callable[[Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
    payload_factory: Optional[Callable[[Any], dict]] = None,
    emit_progress: Optional[tuple[int, str]] = None,
    stop_after_start: bool = True,
    wait_returns: bool = True,
) -> type:
    """
    Create a SimulationWorker-like QObject class for monkeypatching.

    The returned class implements the minimal interface expected by the GUI/controller:
    - signals: progress(int,str), result_ready(dict), error(str)
    - methods: start(), cancel(), isRunning(), wait(), terminate()
    """

    class _Worker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(
            self,
            mechanism_text,
            initials,
            t_span,
            solver_config,
            parent=None,
            prepared=None,
            include_mechanism_in_result_payload=True,
        ):
            super().__init__(parent)
            self._running = False
            self._mechanism_text = str(mechanism_text)
            self._initials = dict(initials or {})
            self._t_span = tuple(t_span) if t_span is not None else (0.0, 0.0)
            self._t_start = float(self._t_span[0]) if len(self._t_span) >= 1 else 0.0
            self._t_end = float(self._t_span[-1]) if len(self._t_span) >= 1 else 0.0
            self._solver_config = dict(solver_config or {})
            self._prepared = prepared
            self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            if on_init is not None:
                on_init(self)

        def start(self) -> None:
            self._running = True
            if on_start is not None:
                on_start(self)
            if payload_factory is not None:
                if emit_progress is not None:
                    self.progress.emit(int(emit_progress[0]), str(emit_progress[1]))
                self.result_ready.emit(payload_factory(self))
            if bool(stop_after_start):
                self._running = False

        def cancel(self) -> None:
            self._running = False

        def isRunning(self) -> bool:
            return bool(self._running)

        def wait(self, *_args, **_kwargs) -> bool:
            if bool(wait_returns):
                self._running = False
            return bool(wait_returns)

        def terminate(self) -> None:
            self._running = False

    if payload_factory is not None:
        payload_qualname = getattr(payload_factory, "__qualname__", payload_factory.__class__.__qualname__)
        _Worker.__qualname__ = f"WorkerStub[{payload_qualname}]"

    return _Worker


def make_contained_simulation_worker_stub(
    *,
    on_init: Optional[Callable[[Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
    payload_factory: Optional[Callable[[Any], dict]] = None,
    emit_progress: Optional[tuple[int, str]] = None,
    stop_after_start: bool = True,
    wait_returns: bool = True,
) -> type:
    """
    Create a ContainedSimulationWorker-like QObject class for monkeypatching.

    The returned class exposes the same attrs most legacy SimulationWorker tests
    inspect, but derives them from the serialized SimulationPlan payload that the
    contained GUI path now passes to the worker adapter.
    """

    class _ContainedWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        result_ready = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(
            self,
            *,
            owner,
            simulation_plan_payload,
            include_mechanism_in_result_payload=True,
            parent=None,
        ):
            super().__init__(parent)
            from kindred.core.simulation_plan import SimulationPlan

            self._running = False
            self._owner = owner
            self._simulation_plan_payload = dict(simulation_plan_payload or {})
            execution_request = (
                SimulationPlan.from_payload(self._simulation_plan_payload)
                .to_execution_request()
                .to_payload()
            )
            self._mechanism_text = str(execution_request.get("mechanism_text") or "")
            self._initials = dict(execution_request.get("initials") or {})
            self._t_span = tuple(execution_request.get("t_span") or (0.0, 0.0))
            self._t_start = float(self._t_span[0]) if len(self._t_span) >= 1 else 0.0
            self._t_end = float(self._t_span[-1]) if len(self._t_span) >= 1 else 0.0
            self._solver_config = dict(execution_request.get("solver_config") or {})
            self._prepared = execution_request.get("prepared_payload")
            self._include_mechanism_in_result_payload = bool(include_mechanism_in_result_payload)
            if on_init is not None:
                on_init(self)

        def start(self) -> None:
            self._running = True
            if on_start is not None:
                on_start(self)
            if payload_factory is not None:
                if emit_progress is not None:
                    self.progress.emit(int(emit_progress[0]), str(emit_progress[1]))
                self.result_ready.emit(payload_factory(self))
            if bool(stop_after_start):
                self._running = False

        def cancel(self) -> None:
            self._running = False
            owner = getattr(self, "_owner", None)
            if owner is not None and hasattr(owner, "cancel"):
                owner.cancel()

        def isRunning(self) -> bool:
            return bool(self._running)

        def wait(self, *_args, **_kwargs) -> bool:
            if bool(wait_returns):
                self._running = False
            return bool(wait_returns)

        def terminate(self) -> None:
            self._running = False

    if payload_factory is not None:
        payload_qualname = getattr(payload_factory, "__qualname__", payload_factory.__class__.__qualname__)
        _ContainedWorker.__qualname__ = f"ContainedWorkerStub[{payload_qualname}]"

    return _ContainedWorker


def make_stubborn_worker(fake_worker_cls: type) -> Any:
    """
    Return a FakeWorker-derived instance that stays 'running' even after cancel().

    This is used to validate _cleanup_worker_safely() behavior (finished->deleteLater
    hookup and avoiding sendPostedEvents while still running).
    """

    class _StubbornWorker(fake_worker_cls):
        def __init__(self) -> None:
            super().__init__(running=True, wait_returns=False, signal_disconnect_typeerror=False)
            self._delete_later_called = False

        def cancel(self) -> None:
            self._cancelled = True
            self._running = True

        def isRunning(self) -> bool:
            return True

        def deleteLater(self) -> None:
            self._delete_later_called = True

    return _StubbornWorker()

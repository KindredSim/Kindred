"""Fitting runtime readiness owner for the global fitting window."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
import logging
import threading
from typing import Any, Callable, Optional

from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.dataset_parameter_overrides import FitDatasetParameterOverrides
from kindred.core.exceptions import FittingCancelled
from kindred.core.fitting_evaluation import SerialFittingEvaluator, coerce_fitting_series_evaluator
from kindred.core.fitting_runtime_session import FittingRuntimeSession
from kindred.core.simulation_preparation import coerce_prepared_simulation_metadata
from kindred.gui.fitting.run_stamp import (
    finalize_global_fit_run_stamp_prepared_simulation,
    hash_global_fit_run_stamp,
)

logger = logging.getLogger(__name__)


def _session_required_for_fit_evaluator(fit_evaluator: Any) -> bool:
    normalized = coerce_fitting_series_evaluator(fit_evaluator)
    return type(normalized) is SerialFittingEvaluator


class FittingRuntimeReadinessState(str, Enum):
    EMPTY = "empty"
    BLOCKED = "blocked"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class FittingRuntimeLaunchDecisionState(str, Enum):
    ACCEPTED = "accepted"
    PREPARING = "preparing"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class FittingRuntimeIdentity:
    datasets: tuple[FitDatasetSpec, ...]
    config: dict[str, Any]
    dataset_overrides: tuple[FitDatasetParameterOverrides, ...]
    weights: Optional[dict[str, float]]
    requested_solver: str
    requested_rtol: float
    requested_atol: float
    fit_evaluator: Any
    stamp: dict[str, Any]
    stamp_hash: str
    stamp_short: str
    lane_count: int
    readiness_required: bool = True
    fit_evaluator_factory: Optional[Callable[[], Any]] = None
    base_evaluator: Any = None
    launch_request_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", deepcopy(dict(self.config or {})))
        object.__setattr__(self, "weights", None if self.weights is None else deepcopy(dict(self.weights)))
        object.__setattr__(self, "stamp", deepcopy(dict(self.stamp or {})))
        if not str(self.launch_request_hash or ""):
            object.__setattr__(self, "launch_request_hash", str(self.stamp_hash or ""))


@dataclass(frozen=True)
class PreparedFitEvaluator:
    base_evaluator: Any
    fit_evaluator: Any


@dataclass
class FittingRuntimeReadinessLedger:
    desired_identity_changes: int = 0
    blocked_transitions: int = 0
    session_creations: int = 0
    ready_reuses: int = 0
    preparation_starts: int = 0
    preparation_cancels: int = 0
    preparation_supersedes: int = 0
    preparation_ready: int = 0
    preparation_failed: int = 0
    stale_completions: int = 0
    session_closes: int = 0
    close_requests: int = 0
    close_deferred: int = 0
    close_completed: int = 0


@dataclass(frozen=True)
class FittingRuntimeReadinessSnapshot:
    state: FittingRuntimeReadinessState
    desired_hash: str = ""
    active_hash: str = ""
    ready_hash: str = ""
    session: Optional[FittingRuntimeSession] = None
    identity: Optional[FittingRuntimeIdentity] = None
    error: Optional[BaseException] = None
    ledger: FittingRuntimeReadinessLedger = field(default_factory=FittingRuntimeReadinessLedger)

    @property
    def is_preparing(self) -> bool:
        return self.state is FittingRuntimeReadinessState.PREPARING

    @property
    def is_ready(self) -> bool:
        return self.state is FittingRuntimeReadinessState.READY

    @property
    def is_closing(self) -> bool:
        return self.state is FittingRuntimeReadinessState.CLOSING


@dataclass(frozen=True)
class FittingRuntimeAcceptedLaunch:
    identity: FittingRuntimeIdentity
    session: Optional[FittingRuntimeSession]

    @property
    def datasets(self) -> tuple[FitDatasetSpec, ...]:
        return self.identity.datasets

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.identity.config)

    @property
    def dataset_overrides(self) -> tuple[FitDatasetParameterOverrides, ...]:
        return self.identity.dataset_overrides

    @property
    def weights(self) -> Optional[dict[str, float]]:
        return dict(self.identity.weights) if self.identity.weights is not None else None

    @property
    def fit_evaluator(self) -> Any:
        return self.identity.fit_evaluator

    @property
    def base_evaluator(self) -> Any:
        return self.identity.base_evaluator

    @property
    def stamp(self) -> dict[str, Any]:
        return dict(self.identity.stamp)

    @property
    def stamp_hash(self) -> str:
        return str(self.identity.stamp_hash)

    @property
    def stamp_short(self) -> str:
        return str(self.identity.stamp_short)

    @property
    def lane_count(self) -> int:
        return int(self.identity.lane_count)


@dataclass(frozen=True)
class FittingRuntimeLaunchDecision:
    state: FittingRuntimeLaunchDecisionState
    accepted_launch: Optional[FittingRuntimeAcceptedLaunch] = None
    snapshot: Optional[FittingRuntimeReadinessSnapshot] = None


class FittingRuntimePreparationWorker:
    def __init__(
        self,
        *,
        identity: FittingRuntimeIdentity,
        session_factory: Callable[[Any, int], Optional[FittingRuntimeSession]],
        stamp_hash: str,
        lane_count: int,
        finished_callback: Callable[[], None],
    ) -> None:
        self._identity = identity
        self._session_factory = session_factory
        self._session: Optional[FittingRuntimeSession] = None
        self._prepared_identity: Optional[FittingRuntimeIdentity] = None
        self._stamp_hash = str(stamp_hash or "")
        self._lane_count = max(1, int(lane_count))
        self._cancelled = threading.Event()
        self._lock = threading.RLock()
        self._status = ""
        self._error: Optional[BaseException] = None
        self._session_created = False
        self._finished_callback = finished_callback
        self._thread = threading.Thread(
            target=self._run,
            name="kindred-fitting-runtime-prepare",
            daemon=True,
        )

    @property
    def stamp_hash(self) -> str:
        return self._stamp_hash

    @property
    def status(self) -> str:
        return str(self._status or "")

    @property
    def error_object(self) -> Optional[BaseException]:
        return self._error

    @property
    def prepared_identity(self) -> Optional[FittingRuntimeIdentity]:
        return self._prepared_identity

    @property
    def prepared_session(self) -> Optional[FittingRuntimeSession]:
        with self._lock:
            return self._session

    @property
    def session_created(self) -> bool:
        return bool(self._session_created)

    def start(self) -> None:
        self._thread.start()

    def isRunning(self) -> bool:
        return self._thread.is_alive()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            session = self._session
        cancel_run = getattr(session, "cancel_run", None)
        if callable(cancel_run):
            try:
                cancel_run()
            except Exception as exc:
                logger.debug("Failed to cancel fitting runtime preparation: %s", exc, exc_info=True)

    def _cancel_requested(self) -> bool:
        return bool(self._cancelled.is_set())

    def _run(self) -> None:
        try:
            self._raise_if_cancelled()
            evaluator_factory = self._identity.fit_evaluator_factory
            evaluator_result = evaluator_factory() if callable(evaluator_factory) else self._identity.fit_evaluator
            base_evaluator = self._identity.base_evaluator
            if isinstance(evaluator_result, PreparedFitEvaluator):
                base_evaluator = evaluator_result.base_evaluator
                fit_evaluator = evaluator_result.fit_evaluator
            else:
                fit_evaluator = evaluator_result
            if base_evaluator is None:
                base_evaluator = fit_evaluator
            session_required = _session_required_for_fit_evaluator(fit_evaluator)
            prepared_identity = replace(
                self._identity,
                fit_evaluator=fit_evaluator,
                readiness_required=bool(session_required),
                fit_evaluator_factory=None,
                base_evaluator=base_evaluator,
            )
            self._raise_if_cancelled()
            session = (
                self._session_factory(fit_evaluator, int(self._lane_count))
                if bool(session_required)
                else None
            )
            with self._lock:
                self._session = session
                self._session_created = session is not None
                self._prepared_identity = prepared_identity
            self._raise_if_cancelled()
            if session is not None:
                session.warm(
                    cancellation_check=self._cancel_requested,
                    lane_count=self._lane_count,
                )
        except FittingCancelled:
            self._status = "cancelled"
        except BaseException as exc:  # noqa: BLE001 - surfaced through the fitting UI
            if self._cancel_requested():
                self._status = "cancelled"
            else:
                self._error = exc
                self._status = "error"
        else:
            self._status = "cancelled" if self._cancel_requested() else "prepared"
        finally:
            try:
                self._finished_callback()
            except Exception as exc:
                logger.debug("Failed to notify fitting runtime preparation completion: %s", exc, exc_info=True)

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested():
            raise FittingCancelled()


class FittingRuntimeReadinessController:
    def __init__(
        self,
        *,
        session_factory: Callable[[Any, int], Optional[FittingRuntimeSession]],
        finished_callback: Callable[[], None],
        ledger: Optional[FittingRuntimeReadinessLedger] = None,
    ) -> None:
        self._session_factory = session_factory
        self._finished_callback = finished_callback
        self._ledger = ledger if ledger is not None else FittingRuntimeReadinessLedger()
        self._state = FittingRuntimeReadinessState.EMPTY
        self._desired_identity: Optional[FittingRuntimeIdentity] = None
        self._active_identity: Optional[FittingRuntimeIdentity] = None
        self._active_session: Optional[FittingRuntimeSession] = None
        self._ready_identity: Optional[FittingRuntimeIdentity] = None
        self._ready_session: Optional[FittingRuntimeSession] = None
        self._worker: Optional[FittingRuntimePreparationWorker] = None
        self._error: Optional[BaseException] = None

    @property
    def worker(self) -> Optional[FittingRuntimePreparationWorker]:
        return self._worker

    @property
    def ledger(self) -> FittingRuntimeReadinessLedger:
        return self._ledger

    def snapshot(self) -> FittingRuntimeReadinessSnapshot:
        return FittingRuntimeReadinessSnapshot(
            state=self._state,
            desired_hash="" if self._desired_identity is None else self._desired_identity.stamp_hash,
            active_hash="" if self._active_identity is None else self._active_identity.stamp_hash,
            ready_hash="" if self._ready_identity is None else self._ready_identity.stamp_hash,
            session=self._ready_session,
            identity=self._ready_identity,
            error=self._error,
            ledger=self._ledger,
        )

    def set_blocked(self, error: Optional[BaseException] = None) -> None:
        self._desired_identity = None
        self._error = error
        self._ledger.blocked_transitions += 1
        self._cancel_active_preparation()
        self._close_ready_session(kill=False)
        self._state = FittingRuntimeReadinessState.BLOCKED

    def set_desired_identity(self, identity: Optional[FittingRuntimeIdentity]) -> None:
        if self._state is FittingRuntimeReadinessState.CLOSING:
            return
        old_hash = "" if self._desired_identity is None else self._desired_identity.stamp_hash
        new_hash = "" if identity is None else identity.stamp_hash
        if old_hash != new_hash:
            self._ledger.desired_identity_changes += 1
        self._desired_identity = identity
        self._error = None
        if identity is None:
            self._cancel_active_preparation()
            self._close_ready_session(kill=False)
            self._state = FittingRuntimeReadinessState.EMPTY
            return
        if not identity.readiness_required and not callable(identity.fit_evaluator_factory):
            self._cancel_active_preparation()
            self._close_ready_session(kill=False)
            ready_identity = self._finalize_identity_for_accepted_launch(identity)
            self._desired_identity = ready_identity
            self._ready_identity = ready_identity
            self._ready_session = None
            self._state = FittingRuntimeReadinessState.READY
            return
        if self._ready_identity is not None and not self._same_ready_identity(self._ready_identity, identity):
            self._close_ready_session(kill=False)
        if self._ready_identity is not None and self._same_ready_identity(self._ready_identity, identity):
            if (
                not bool(self._ready_identity.readiness_required)
                or self._session_ready(self._ready_session, lane_count=identity.lane_count)
            ):
                self._ledger.ready_reuses += 1
                self._state = FittingRuntimeReadinessState.READY
                return
            self._close_ready_session(kill=True)
        if self._worker is not None:
            if self._active_identity is not None and not self._same_ready_identity(self._active_identity, identity):
                self._ledger.preparation_supersedes += 1
                self._cancel_active_preparation()
            if self._worker is not None:
                self._state = FittingRuntimeReadinessState.PREPARING
                return
        self._start_preparation(identity)

    def ready_identity_for(self, stamp_hash: str) -> Optional[FittingRuntimeIdentity]:
        if self._ready_identity is None:
            return None
        requested = str(stamp_hash or "")
        return (
            self._ready_identity
            if requested in {
                str(self._ready_identity.stamp_hash or ""),
                str(self._ready_identity.launch_request_hash or ""),
            }
            else None
        )

    def ready_session_for(self, stamp_hash: str) -> Optional[FittingRuntimeSession]:
        identity = self.ready_identity_for(stamp_hash)
        if identity is None:
            return None
        if not identity.readiness_required:
            return None
        if not self._session_ready(self._ready_session, lane_count=identity.lane_count):
            return None
        return self._ready_session

    def is_ready_for(self, identity: Optional[FittingRuntimeIdentity]) -> bool:
        if identity is None:
            return False
        if self._state is not FittingRuntimeReadinessState.READY:
            return False
        if self._ready_identity is None or not self._same_ready_identity(self._ready_identity, identity):
            return False
        if callable(self._ready_identity.fit_evaluator_factory):
            return False
        if not bool(self._ready_identity.readiness_required):
            return True
        return self._session_ready(self._ready_session, lane_count=self._ready_identity.lane_count)

    def accepted_launch_for(self, identity: Optional[FittingRuntimeIdentity]) -> Optional[FittingRuntimeAcceptedLaunch]:
        if not self.is_ready_for(identity):
            return None
        if self._ready_identity is None:
            return None
        session = self._ready_session if self._ready_identity.readiness_required else None
        if self._ready_identity.readiness_required and session is None:
            return None
        return FittingRuntimeAcceptedLaunch(identity=self._ready_identity, session=session)

    def prepare_or_accept_launch(
        self,
        identity: Optional[FittingRuntimeIdentity],
    ) -> FittingRuntimeLaunchDecision:
        self.set_desired_identity(identity)
        if identity is None:
            return FittingRuntimeLaunchDecision(
                FittingRuntimeLaunchDecisionState.NOT_READY,
                snapshot=self.snapshot(),
            )
        accepted = self.accepted_launch_for(identity)
        if accepted is not None:
            return FittingRuntimeLaunchDecision(
                FittingRuntimeLaunchDecisionState.ACCEPTED,
                accepted_launch=accepted,
                snapshot=self.snapshot(),
            )
        snapshot = self.snapshot()
        decision_state = (
            FittingRuntimeLaunchDecisionState.PREPARING
            if snapshot.state is FittingRuntimeReadinessState.PREPARING
            else FittingRuntimeLaunchDecisionState.NOT_READY
        )
        return FittingRuntimeLaunchDecision(decision_state, snapshot=snapshot)

    def cancel(self, *, kill: bool = True) -> bool:
        self._desired_identity = None
        if self._worker is None:
            return True
        self._cancel_active_preparation()
        if self._worker is not None and kill:
            if self._state is not FittingRuntimeReadinessState.CLOSING:
                self._state = FittingRuntimeReadinessState.EMPTY
            return False
        return self._worker is None

    def close_ready_session(self, *, kill: bool = False) -> None:
        self._close_ready_session(kill=kill)

    def close(self, *, kill: bool = True) -> bool:
        self._ledger.close_requests += 1
        self._desired_identity = None
        if self._worker is not None:
            self._state = FittingRuntimeReadinessState.CLOSING
            self._ledger.close_deferred += 1
            self._cancel_active_preparation()
            if self._worker is None and self._state is FittingRuntimeReadinessState.CLOSED:
                return True
            return False
        self._close_ready_session(kill=kill)
        self._state = FittingRuntimeReadinessState.CLOSED
        self._ledger.close_completed += 1
        return True

    def handle_worker_finished(self) -> bool:
        worker = self._worker
        was_closing = self._state is FittingRuntimeReadinessState.CLOSING
        if worker is None:
            return True
        if self._worker_running(worker):
            return False
        self._worker = None
        active_identity = worker.prepared_identity or self._active_identity
        active_session = worker.prepared_session
        self._active_identity = None
        self._active_session = None
        status = worker.status
        if worker.session_created:
            self._ledger.session_creations += 1
        if (
            status == "prepared"
            and active_identity is not None
            and self._desired_identity is not None
            and self._same_ready_identity(active_identity, self._desired_identity)
        ):
            if active_identity.readiness_required and not self._session_ready(active_session, lane_count=active_identity.lane_count):
                if active_session is not None:
                    self._close_session(active_session, kill=True)
                self._error = RuntimeError("Prepared fitting runtime did not produce a required fitting runtime session.")
                self._ledger.preparation_failed += 1
                self._state = FittingRuntimeReadinessState.FAILED
                self._desired_identity = None
                return True
            ready_identity = self._finalize_identity_for_accepted_launch(active_identity)
            self._desired_identity = ready_identity
            self._ready_identity = ready_identity
            self._ready_session = active_session
            self._ledger.preparation_ready += 1
            self._state = FittingRuntimeReadinessState.READY
            return True
        if active_session is not None:
            self._close_session(active_session, kill=was_closing or status != "prepared")
        if was_closing:
            self._error = None
            self._desired_identity = None
            self._close_ready_session(kill=True)
            self._state = FittingRuntimeReadinessState.CLOSED
            self._ledger.close_completed += 1
            return True
        if status == "prepared":
            self._ledger.stale_completions += 1
        elif status == "cancelled":
            self._ledger.preparation_cancels += 1
        else:
            self._error = worker.error_object or RuntimeError("Unknown fitting runtime preparation failure.")
            self._ledger.preparation_failed += 1
            self._state = FittingRuntimeReadinessState.FAILED
            self._desired_identity = None
            return True
        if self._desired_identity is not None:
            self._start_preparation(self._desired_identity)
            return True
        if status != "error":
            self._state = FittingRuntimeReadinessState.EMPTY
        return True

    def _start_preparation(self, identity: FittingRuntimeIdentity) -> None:
        worker = FittingRuntimePreparationWorker(
            identity=identity,
            session_factory=self._session_factory,
            stamp_hash=identity.stamp_hash,
            lane_count=int(identity.lane_count),
            finished_callback=self._finished_callback,
        )
        self._active_identity = identity
        self._active_session = None
        self._worker = worker
        self._state = FittingRuntimeReadinessState.PREPARING
        self._ledger.preparation_starts += 1
        worker.start()

    def _cancel_active_preparation(self) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            worker.cancel()
        except Exception as exc:
            logger.debug("Failed to cancel fitting runtime preparation: %s", exc, exc_info=True)
        if not self._worker_running(worker):
            self.handle_worker_finished()

    def _close_ready_session(self, *, kill: bool) -> None:
        session = self._ready_session
        self._ready_session = None
        self._ready_identity = None
        if session is not None:
            self._close_session(session, kill=kill)

    def _close_session(self, session: FittingRuntimeSession, *, kill: bool) -> None:
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close(kill=bool(kill))
            except Exception as exc:
                logger.debug("Failed to close fitting runtime session: %s", exc, exc_info=True)
            else:
                self._ledger.session_closes += 1

    @staticmethod
    def _worker_running(worker: FittingRuntimePreparationWorker) -> bool:
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    @staticmethod
    def _same_ready_identity(left: FittingRuntimeIdentity, right: FittingRuntimeIdentity) -> bool:
        left_hashes = {
            str(value)
            for value in (
                left.launch_request_hash,
                left.stamp_hash,
            )
            if str(value or "").strip()
        }
        right_hashes = {
            str(value)
            for value in (
                right.launch_request_hash,
                right.stamp_hash,
            )
            if str(value or "").strip()
        }
        return (
            bool(left_hashes.intersection(right_hashes))
            and int(left.lane_count) == int(right.lane_count)
        )

    @staticmethod
    def _prepared_simulation_meta(fit_evaluator: Any):
        if fit_evaluator is None:
            return None
        try:
            prepared = getattr(fit_evaluator, "prepared_metadata", None)
        except Exception:
            prepared = None
        meta = coerce_prepared_simulation_metadata(prepared)
        if meta is not None:
            return meta
        try:
            prepared = getattr(fit_evaluator, "_kindred_prepared_simulation_meta", None)
        except Exception:
            return None
        return coerce_prepared_simulation_metadata(prepared)

    def _finalize_identity_for_accepted_launch(self, identity: FittingRuntimeIdentity) -> FittingRuntimeIdentity:
        if identity is None:
            return identity
        prepared_simulation = self._prepared_simulation_meta(identity.fit_evaluator)
        if identity.stamp.get("prepared_simulation") and prepared_simulation is None:
            return replace(
                identity,
                fit_evaluator_factory=None,
                launch_request_hash=str(identity.stamp_hash or ""),
            )
        if prepared_simulation is None:
            return replace(
                identity,
                fit_evaluator_factory=None,
                launch_request_hash=str(identity.stamp_hash or ""),
            )
        stamp = finalize_global_fit_run_stamp_prepared_simulation(identity.stamp, prepared_simulation)
        stamp_hash = hash_global_fit_run_stamp(stamp)
        launch_request_hash = str(identity.launch_request_hash or identity.stamp_hash or "")
        if (
            prepared_simulation.symbolic_jacobian_identity
            or prepared_simulation.symbolic_wegscheider_identity
        ):
            launch_request_hash = str(stamp_hash)
        return replace(
            identity,
            stamp=stamp,
            stamp_hash=str(stamp_hash),
            stamp_short=str(stamp_hash)[:12],
            fit_evaluator_factory=None,
            launch_request_hash=launch_request_hash,
        )

    @staticmethod
    def _session_ready(session: Optional[FittingRuntimeSession], *, lane_count: int) -> bool:
        if session is None:
            return False
        is_ready = getattr(session, "is_ready", None)
        if callable(is_ready):
            try:
                return bool(is_ready(lane_count=max(1, int(lane_count))))
            except Exception as exc:
                logger.debug("Failed to inspect fitting runtime readiness: %s", exc, exc_info=True)
                return False
        return False

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from kindred.core.exceptions import FitSimulationError, FittingCancelled
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR,
    SerialFittingEvaluator,
)
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulation_series_payload import coerce_simulation_series_payload

logger = logging.getLogger(__name__)


@dataclass
class FittingRuntimeLedger:
    payload_preparations: int = 0
    lane_creations: int = 0
    lane_warms: int = 0
    candidate_evaluations: int = 0
    run_cancellations: int = 0
    lane_closes: int = 0
    lane_kills: int = 0

    def snapshot(self) -> Dict[str, int]:
        return {
            "payload_preparations": int(self.payload_preparations),
            "lane_creations": int(self.lane_creations),
            "lane_warms": int(self.lane_warms),
            "candidate_evaluations": int(self.candidate_evaluations),
            "run_cancellations": int(self.run_cancellations),
            "lane_closes": int(self.lane_closes),
            "lane_kills": int(self.lane_kills),
        }


@dataclass(frozen=True)
class FittingRuntimeRequest:
    params: Mapping[str, float]
    origins: Mapping[str, str] = field(default_factory=dict)
    failed_params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params or {}))
        object.__setattr__(self, "origins", dict(self.origins or {}))
        object.__setattr__(self, "failed_params", dict(self.failed_params or {}))


class FittingRuntimeSession:
    """Own reusable fitting evaluator lanes for one prepared fitting runtime identity."""

    def __init__(
        self,
        *,
        process_payload: Mapping[str, Any],
        max_lanes: int,
        lane_factory: Optional[Callable[..., Any]] = None,
        ledger: Optional[FittingRuntimeLedger] = None,
        request_timeout_s: Optional[float] = None,
    ) -> None:
        if int(max_lanes) < 1:
            raise ValueError("Fitting runtime max_lanes must be at least 1.")
        self._process_payload = dict(process_payload)
        self._max_lanes = int(max_lanes)
        self._lane_factory = lane_factory
        self._request_timeout_s = request_timeout_s
        self._ledger = ledger if ledger is not None else FittingRuntimeLedger()
        self._ledger.payload_preparations += 1
        self._lanes: list[Optional[Any]] = []
        self._warmed_lane_ids: set[int] = set()
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_serial_evaluator(
        cls,
        evaluator: SerialFittingEvaluator,
        *,
        max_lanes: int,
        lane_factory: Optional[Callable[..., Any]] = None,
        ledger: Optional[FittingRuntimeLedger] = None,
        request_timeout_s: Optional[float] = None,
    ) -> "FittingRuntimeSession":
        try:
            process_payload = evaluator.to_process_payload()
        except Exception as exc:
            raise _fatal_fit_runtime_error(
                "fitting_containment_payload",
                f"Failed to build fitting containment payload: {exc}",
                exc=exc,
            ) from exc
        return cls(
            process_payload=process_payload,
            max_lanes=int(max_lanes),
            lane_factory=lane_factory,
            ledger=ledger,
            request_timeout_s=request_timeout_s,
        )

    @property
    def ledger(self) -> FittingRuntimeLedger:
        return self._ledger

    def begin_run(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Fitting runtime session is closed.")

    def evaluator(self, *, cancellation_check: Optional[Callable[[], bool]] = None) -> "FittingRuntimeEvaluator":
        return FittingRuntimeEvaluator(self, cancellation_check=cancellation_check)

    def warm(self, *, cancellation_check: Optional[Callable[[], bool]] = None, lane_count: Optional[int] = None) -> None:
        lanes = self._ensure_lanes(lane_count=lane_count)
        for lane in lanes:
            self._raise_if_cancelled(cancellation_check)
            lane_id = id(lane)
            if lane_id in self._warmed_lane_ids:
                continue
            warm = getattr(lane, "warm", None)
            try:
                if callable(warm):
                    warm(cancellation_check=cancellation_check)
            except BaseException:
                self._invalidate_lane(lane, kill=True)
                raise
            self._warmed_lane_ids.add(lane_id)
            with self._lock:
                self._ledger.lane_warms += 1

    def evaluate_one(
        self,
        request: FittingRuntimeRequest,
        *,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ):
        result = self.evaluate_batch([request], cancellation_check=cancellation_check)
        return _coerce_single_evaluation_result(result[0])

    def evaluate_batch(
        self,
        requests: Sequence[FittingRuntimeRequest],
        *,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> list[object]:
        request_list = [request if isinstance(request, FittingRuntimeRequest) else FittingRuntimeRequest(**request) for request in requests]
        if not request_list:
            return []
        self._raise_if_cancelled(cancellation_check)
        lane_count = min(len(request_list), self._max_lanes)
        self.warm(cancellation_check=cancellation_check, lane_count=lane_count)

        def _evaluate_on_lane(lane, index: int, runtime_request: FittingRuntimeRequest) -> tuple[int, object]:
            self._raise_if_cancelled(cancellation_check)
            try:
                value = lane.evaluate_series_with_parameter_origins(
                    dict(runtime_request.params),
                    dict(runtime_request.origins),
                    failed_params=dict(runtime_request.failed_params),
                    cancellation_check=cancellation_check,
                )
                with self._lock:
                    self._ledger.candidate_evaluations += 1
                return index, coerce_simulation_series_payload(value)
            except BaseException as exc:  # noqa: BLE001 - fitting objective owns penalty/fatal policy
                if _exception_invalidates_lane(exc):
                    self._invalidate_lane(lane, kill=True)
                return index, exc

        if lane_count == 1:
            ordered = [
                _evaluate_on_lane(
                    self._lane_for_slot(
                        0,
                        cancellation_check=cancellation_check,
                    ),
                    index,
                    request,
                )
                for index, request in enumerate(request_list)
            ]
        else:
            groups: list[list[tuple[int, FittingRuntimeRequest]]] = [[] for _ in range(lane_count)]
            for index, request in enumerate(request_list):
                groups[index % lane_count].append((index, request))

            def _evaluate_group(group_index: int) -> list[tuple[int, object]]:
                return [
                    _evaluate_on_lane(
                        self._lane_for_slot(
                            group_index,
                            cancellation_check=cancellation_check,
                        ),
                        index,
                        runtime_request,
                    )
                    for index, runtime_request in groups[group_index]
                ]

            with ThreadPoolExecutor(max_workers=lane_count, thread_name_prefix="kindred-fitting-runtime") as executor:
                ordered = [
                    item
                    for group_result in executor.map(_evaluate_group, range(lane_count))
                    for item in group_result
                ]
        ordered.sort(key=lambda pair: int(pair[0]))
        return [value for _index, value in ordered]

    def cancel_run(self) -> None:
        with self._lock:
            self._ledger.run_cancellations += 1
        self.close(kill=True)

    def close(self, *, kill: bool = False) -> None:
        with self._lock:
            lanes = [lane for lane in self._lanes if lane is not None]
            self._lanes = []
            self._warmed_lane_ids.clear()
            self._closed = True
        for lane in lanes:
            close = getattr(lane, "close", None)
            if callable(close):
                try:
                    close(kill=bool(kill))
                except Exception as exc:
                    logger.debug("Failed to close fitting runtime lane: %s", exc, exc_info=True)
            self._ledger.lane_closes += 1
            if kill:
                self._ledger.lane_kills += 1

    def _ensure_lanes(self, *, lane_count: Optional[int] = None) -> list[Any]:
        required = self._max_lanes if lane_count is None else min(self._max_lanes, max(1, int(lane_count)))
        with self._lock:
            if self._closed:
                raise RuntimeError("Fitting runtime session is closed.")
            while len(self._lanes) < required:
                self._lanes.append(self._make_lane())
                self._ledger.lane_creations += 1
            for index in range(required):
                if self._lanes[index] is None:
                    self._lanes[index] = self._make_lane()
                    self._ledger.lane_creations += 1
            return [lane for lane in self._lanes[:required] if lane is not None]

    def _make_lane(self):
        if self._lane_factory is None:
            from kindred.core.fitting_containment import WarmFittingEvaluatorLane

            factory = WarmFittingEvaluatorLane
        else:
            factory = self._lane_factory
        kwargs: dict[str, object] = {}
        if self._request_timeout_s is not None:
            kwargs["request_timeout_s"] = float(self._request_timeout_s)
        return factory(dict(self._process_payload), **kwargs)

    def _lane_for_slot(
        self,
        slot_index: int,
        *,
        cancellation_check: Optional[Callable[[], bool]],
    ) -> Any:
        self.warm(cancellation_check=cancellation_check, lane_count=int(slot_index) + 1)
        lanes = self._ensure_lanes(lane_count=int(slot_index) + 1)
        return lanes[int(slot_index)]

    def _invalidate_lane(self, lane: Any, *, kill: bool) -> None:
        with self._lock:
            self._warmed_lane_ids.discard(id(lane))
            for index, existing in enumerate(self._lanes):
                if existing is lane:
                    self._lanes[index] = None
        close = getattr(lane, "close", None)
        if callable(close):
            try:
                close(kill=bool(kill))
            except Exception as exc:
                logger.debug("Failed to close invalid fitting runtime lane: %s", exc, exc_info=True)
        self._ledger.lane_closes += 1
        if kill:
            self._ledger.lane_kills += 1

    @staticmethod
    def _cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> bool:
        if cancellation_check is None:
            return False
        check = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
        return bool(check())

    def _raise_if_cancelled(self, cancellation_check: Optional[Callable[[], bool]]) -> None:
        if self._cancel_requested(cancellation_check):
            raise FittingCancelled()


def _fatal_fit_runtime_error(
    kind: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
) -> FitSimulationError:
    return FitSimulationError(
        message,
        details={
            "fatal": True,
            "failure": build_simulation_failure(
                kind,
                message,
                exc_type=type(exc).__name__ if exc is not None else None,
            ),
        },
    )


def _failure_kind_from_exception(exc: BaseException) -> str:
    details = getattr(exc, "details", None)
    if not isinstance(details, Mapping):
        return ""
    failure = details.get("failure")
    if not isinstance(failure, Mapping):
        return ""
    return str(failure.get("kind") or "")


def _exception_invalidates_lane(exc: BaseException) -> bool:
    if isinstance(exc, FittingCancelled):
        return True
    if exc.__class__.__name__ in {"FittingLaneTimeout", "FittingLaneProtocolError"}:
        return True
    return isinstance(exc, FitSimulationError) and _failure_kind_from_exception(exc).startswith("fitting_containment_")


def _coerce_single_evaluation_result(value: object) -> object:
    if not isinstance(value, BaseException):
        return value
    if value.__class__.__name__ == "FittingLaneProtocolError":
        raise _fatal_fit_runtime_error(
            "fitting_containment_protocol",
            str(value),
            exc=value,
        ) from value
    raise value


class FittingRuntimeEvaluator:
    def __init__(
        self,
        session: FittingRuntimeSession,
        *,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._session = session
        self._cancellation_check = cancellation_check

    def __call__(self, params: Mapping[str, float]):
        return self.evaluate_series(params)

    def evaluate_series(self, params: Mapping[str, float]):
        configured_origins = {
            str(name): FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR
            for name in dict(params or {})
            if str(name).strip()
        }
        return self.evaluate_series_with_parameter_origins(
            params,
            configured_origins,
            failed_params=None,
        )

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ):
        return self._session.evaluate_one(
            FittingRuntimeRequest(
                params=dict(params or {}),
                origins=dict(origins or {}),
                failed_params=dict(failed_params or {}),
            ),
            cancellation_check=self._cancellation_check,
        )

    def evaluate_fitting_runtime_batch(
        self,
        requests: Sequence[FittingRuntimeRequest],
        *,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> list[object]:
        effective_check = cancellation_check if cancellation_check is not None else self._cancellation_check
        return self._session.evaluate_batch(
            list(requests),
            cancellation_check=effective_check,
        )

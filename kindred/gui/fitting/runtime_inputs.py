"""Typed Global Fit runtime-input snapshots and active-window publication."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

from kindred.core.batch_parallel import compute_effective_batch_workers
from kindred.gui.project_schema import SIMULATION_TEMPERATURE_K_RANGE


def _coerce_temperature(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{source} requires numeric temperature_K.")
    try:
        temperature = float(value)
    except Exception as exc:
        raise RuntimeError(f"{source} requires numeric temperature_K.") from exc
    min_temperature, max_temperature = SIMULATION_TEMPERATURE_K_RANGE
    if (
        not math.isfinite(temperature)
        or temperature < float(min_temperature)
        or temperature > float(max_temperature)
    ):
        raise RuntimeError(f"{source} requires temperature_K within the simulation temperature range.")
    return float(temperature)


def _coerce_runtime_bool(value: Any, *, key: str, source: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{source} requires explicit {key}.")
    return bool(value)


def _coerce_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Fitting runtime inputs require integer {key}.")
    if int(value) <= 0:
        raise RuntimeError(f"Fitting runtime inputs require positive {key}.")
    return int(value)


@dataclass(frozen=True)
class FittingEvaluatorRuntimeSettings:
    temperature_K: float
    use_sparse_jacobian: bool
    wegscheider_cyclicity_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature_K",
            _coerce_temperature(
                self.temperature_K,
                source="Fitting evaluator runtime settings",
            ),
        )
        object.__setattr__(
            self,
            "use_sparse_jacobian",
            _coerce_runtime_bool(
                self.use_sparse_jacobian,
                key="use_sparse_jacobian",
                source="Fitting evaluator runtime settings",
            ),
        )
        object.__setattr__(
            self,
            "wegscheider_cyclicity_enabled",
            _coerce_runtime_bool(
                self.wegscheider_cyclicity_enabled,
                key="wegscheider_cyclicity_enabled",
                source="Fitting evaluator runtime settings",
            ),
        )

    def builder_kwargs(self) -> dict[str, object]:
        return {
            "temperature_K": float(self.temperature_K),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled),
        }

    def identity_key(self) -> tuple[float, bool, bool]:
        return (
            float(self.temperature_K),
            bool(self.use_sparse_jacobian),
            bool(self.wegscheider_cyclicity_enabled),
        )

    def to_stamp_payload(self) -> dict[str, object]:
        return self.builder_kwargs()


@dataclass(frozen=True)
class FittingRuntimeInputs:
    evaluator: FittingEvaluatorRuntimeSettings
    batch_runtime_lane_budget: int
    generation: int = 0

    def __post_init__(self) -> None:
        evaluator = self.evaluator
        if not isinstance(evaluator, FittingEvaluatorRuntimeSettings):
            raise RuntimeError("Fitting runtime inputs require typed evaluator runtime settings.")
        object.__setattr__(self, "evaluator", evaluator)
        object.__setattr__(
            self,
            "batch_runtime_lane_budget",
            _coerce_positive_int(
                self.batch_runtime_lane_budget,
                key="batch_runtime_lane_budget",
            ),
        )
        generation = _coerce_positive_or_zero_int(self.generation, key="generation")
        object.__setattr__(self, "generation", generation)

    def identity_key(self) -> tuple[tuple[float, bool, bool], int]:
        return (
            self.evaluator.identity_key(),
            int(self.batch_runtime_lane_budget),
        )

    def lane_count_for_dataset_count(self, dataset_count: int) -> int:
        return compute_effective_batch_workers(
            num_sets=max(1, int(dataset_count)),
            max_parallel_workers=int(self.batch_runtime_lane_budget),
        )

    def to_stamp_payload(self) -> dict[str, object]:
        return {
            "evaluator": self.evaluator.to_stamp_payload(),
            "batch_runtime_lane_budget": int(self.batch_runtime_lane_budget),
        }

    @classmethod
    def from_stamp_payload(cls, payload: Mapping[str, Any]) -> "FittingRuntimeInputs":
        if not isinstance(payload, Mapping):
            raise RuntimeError("Fitting runtime input stamp payload must be a mapping.")
        evaluator_payload = payload.get("evaluator")
        if not isinstance(evaluator_payload, Mapping):
            raise RuntimeError("Fitting runtime input stamp payload requires evaluator settings.")
        return cls(
            evaluator=FittingEvaluatorRuntimeSettings(
                temperature_K=evaluator_payload.get("temperature_K"),
                use_sparse_jacobian=evaluator_payload.get("use_sparse_jacobian"),
                wegscheider_cyclicity_enabled=evaluator_payload.get(
                    "wegscheider_cyclicity_enabled"
                ),
            ),
            batch_runtime_lane_budget=payload.get("batch_runtime_lane_budget"),
        )


def _coerce_positive_or_zero_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Fitting runtime inputs require integer {key}.")
    if int(value) < 0:
        raise RuntimeError(f"Fitting runtime inputs require non-negative {key}.")
    return int(value)


class FittingRuntimeInputPublisher:
    """Own active Global Fit windows and publish typed runtime-input snapshots."""

    def __init__(
        self,
        *,
        capture_inputs: Callable[[], FittingRuntimeInputs],
        record_failure: Callable[..., None],
    ) -> None:
        if not callable(capture_inputs):
            raise RuntimeError("Fitting runtime input publisher requires a capture callback.")
        if not callable(record_failure):
            raise RuntimeError("Fitting runtime input publisher requires a failure recorder.")
        self._capture_inputs = capture_inputs
        self._record_failure = record_failure
        self._windows: list[object] = []
        self._generation = 0
        initial_inputs = self._capture_inputs_typed()
        self._last_identity_key = initial_inputs.identity_key()
        self._transaction_depth = 0
        self._transaction_force_requested = False

    def current_inputs(self) -> FittingRuntimeInputs:
        inputs = self._capture_inputs_typed()
        return self._sync_handoff_inputs(inputs, reason="runtime inputs requested")

    def register_window(
        self,
        window: object,
        *,
        runtime_inputs: FittingRuntimeInputs | None = None,
    ) -> object:
        if window is None:
            raise RuntimeError("Cannot register a missing fitting window.")
        if runtime_inputs is not None:
            if not isinstance(runtime_inputs, FittingRuntimeInputs):
                raise RuntimeError("Registered fitting window requires typed runtime inputs.")
            if self._transaction_depth == 0 and runtime_inputs.identity_key() != self._last_identity_key:
                if self._live_windows():
                    self._publish_inputs(runtime_inputs, reason="fitting window registered")
                else:
                    self._last_identity_key = runtime_inputs.identity_key()
        if window not in self._windows:
            self._windows.append(window)
            self._connect_destroyed_cleanup(window)
        self._show_window(window)
        return window

    def active_windows(self) -> tuple[object, ...]:
        return tuple(self._live_windows())

    def publish_if_changed(self, *, reason: str) -> None:
        if self._transaction_depth > 0:
            return
        inputs = self._capture_inputs_typed_or_record(reason=reason)
        if inputs is None:
            return
        if inputs.identity_key() == self._last_identity_key:
            return
        self._publish_inputs(inputs, reason=reason)

    def publish_current(self, *, reason: str, force: bool = False) -> None:
        if self._transaction_depth > 0:
            self._transaction_force_requested = bool(self._transaction_force_requested or force)
            return
        inputs = self._capture_inputs_typed_or_record(reason=reason)
        if inputs is None:
            return
        if not force and inputs.identity_key() == self._last_identity_key:
            return
        self._publish_inputs(inputs, reason=reason)

    def publish_datasets_removed(self, dataset_ids: Sequence[str]) -> None:
        removed_ids = tuple(str(dataset_id) for dataset_id in (dataset_ids or ()) if str(dataset_id))
        if not removed_ids:
            return
        for window in self._live_windows():
            try:
                handler = getattr(window, "handle_external_datasets_removed", None)
                if callable(handler):
                    handler(removed_ids)
            except Exception as exc:
                self._record_publication_failure(
                    "active_fit_window_datasets_removed",
                    message="Failed to notify an active fitting window about removed datasets",
                    exc=exc,
                )

    @contextmanager
    def transaction(self, *, reason: str) -> Iterator[None]:
        self._transaction_depth += 1
        try:
            yield
        except Exception:
            if self._transaction_depth == 1:
                self._transaction_force_requested = False
            raise
        finally:
            self._transaction_depth = max(0, int(self._transaction_depth) - 1)
        if self._transaction_depth > 0:
            return
        force = bool(self._transaction_force_requested)
        self._transaction_force_requested = False
        inputs = self._capture_inputs_typed_or_record(reason=reason)
        if inputs is None:
            return
        if force or inputs.identity_key() != self._last_identity_key:
            self._publish_inputs(inputs, reason=reason)

    def _capture_inputs_typed_or_record(self, *, reason: str) -> FittingRuntimeInputs | None:
        try:
            return self._capture_inputs_typed()
        except Exception as exc:
            self._record_publication_failure(
                "active_fit_window_runtime_inputs_capture",
                message=f"Failed to capture fitting runtime inputs for {reason}",
                exc=exc,
            )
            return None

    def _capture_inputs_typed(self) -> FittingRuntimeInputs:
        inputs = self._capture_inputs()
        if not isinstance(inputs, FittingRuntimeInputs):
            raise RuntimeError("Fitting runtime input capture must return FittingRuntimeInputs.")
        return inputs

    def _sync_handoff_inputs(self, inputs: FittingRuntimeInputs, *, reason: str) -> FittingRuntimeInputs:
        handed = replace(inputs, generation=int(self._generation))
        if self._transaction_depth > 0:
            return handed
        if handed.identity_key() == self._last_identity_key:
            return handed
        if self._live_windows():
            return self._publish_inputs(inputs, reason=reason)
        self._last_identity_key = handed.identity_key()
        return handed

    def _publish_inputs(self, inputs: FittingRuntimeInputs, *, reason: str) -> FittingRuntimeInputs:
        self._generation += 1
        published = replace(inputs, generation=int(self._generation))
        self._last_identity_key = published.identity_key()
        for window in self._live_windows():
            try:
                handler = getattr(window, "apply_runtime_inputs", None)
                if not callable(handler):
                    raise RuntimeError("Active fitting window has no typed runtime-input subscriber.")
                handler(runtime_inputs=published)
            except Exception as exc:
                self._record_publication_failure(
                    "active_fit_window_runtime_inputs_changed",
                    message=f"Failed to notify an active fitting window about runtime input changes for {reason}",
                    exc=exc,
                )
        return published

    def _live_windows(self) -> list[object]:
        live: list[object] = []
        for window in list(self._windows):
            if window is None:
                continue
            try:
                _ = getattr(window, "objectName", lambda: "")()
            except RuntimeError:
                continue
            live.append(window)
        self._windows = live
        return list(live)

    def _connect_destroyed_cleanup(self, window: object) -> None:
        destroyed = getattr(window, "destroyed", None)
        connect = getattr(destroyed, "connect", None)
        if not callable(connect):
            return

        def _cleanup(*_args: object) -> None:
            self._remove_window(window)

        try:
            connect(_cleanup)
        except Exception:
            return

    def _remove_window(self, window: object) -> None:
        self._windows = [candidate for candidate in self._windows if candidate is not window]

    @staticmethod
    def _show_window(window: object) -> None:
        for method_name in ("show", "raise_", "activateWindow"):
            method = getattr(window, method_name, None)
            if callable(method):
                method()

    def _record_publication_failure(self, key: str, *, message: str, exc: Exception) -> None:
        self._record_failure(str(key), message=str(message), exc=exc)

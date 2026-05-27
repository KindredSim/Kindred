"""
Global fitting worker thread.

This module owns worker orchestration and best-payload emission separately from
the fitting window UI implementation.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING, TypedDict

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    coerce_fit_dataset_specs,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
    coerce_fit_dataset_parameter_overrides,
    split_fit_dataset_parameter_overrides,
)
from kindred.core.analysis.global_fit_execution import (
    assemble_global_fit_result,
    build_parameter_layout,
    normalize_weights,
)
from kindred.core.analysis.global_fit_projection import (
    FitRenderProjection,
    projection_from_global_fit_result,
)
from kindred.core.fitting_evaluation import SerialFittingEvaluator, coerce_fitting_series_evaluator
from kindred.core.exceptions import FitSimulationError, FittingCancelled
from kindred.core.simulation_failure import build_simulation_failure, coerce_simulation_failure
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER

if TYPE_CHECKING:
    from kindred.core.analysis.global_fit_execution import GlobalFitResult


logger = logging.getLogger(__name__)


__all__ = [
    "GlobalFitWorker",
]


class GlobalFitFinishedPayloadV1(TypedDict):
    version: int
    result: "GlobalFitResult"
    weights: Dict[str, float]
    run_stamp: Dict[str, Any]
    run_stamp_hash: str
    run_stamp_short: str


class GlobalFitBestUpdatedPayloadV1(TypedDict):
    version: int
    iteration: int
    cost: float
    shared_params: Dict[str, float]
    dataset_params: Dict[str, Dict[str, float]]
    run_stamp: Dict[str, Any]
    run_stamp_hash: str
    run_stamp_short: str
    render_projection: Any


class GlobalFitWorker(QtCore.QThread):
    """Worker that drives multi-dataset fitting via an injected fitter."""

    progress = Signal(int, str)
    bestUpdated = Signal(dict)
    finished = Signal(dict)
    error = Signal(object)

    def __init__(
        self,
        datasets: List[object],
        shared_params: Dict[str, float],
        *,
        dataset_overrides: Optional[List[object]] = None,
        dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        weights: Optional[Dict[str, float]] = None,
        method: str = "trf",
        max_nfev: int = 1000,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        seed: Optional[int] = None,
        log10_params: Optional[Dict[str, bool]] = None,
        fit_evaluator: Optional[object] = None,
        fit_runtime_session: Optional[object] = None,
        fit_runtime_max_lanes: Optional[int] = None,
        fit_runtime_ledger: Optional[object] = None,
        fit_func: Optional[Callable[..., "GlobalFitResult"]] = None,
        solver: str = FITTING_DEFAULT_SOLVER,
        rtol: float = 1e-6,
        atol: float = 1e-12,
        best_update_interval_s: float = 0.25,
        render_projection_interval_s: float = 2.0,
        run_stamp: Optional[Dict[str, Any]] = None,
        run_stamp_hash: Optional[str] = None,
        run_stamp_short: Optional[str] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._dataset_specs: list[FitDatasetSpec] = coerce_fit_dataset_specs(list(datasets))
        self._datasets: list[FitDatasetSpec] = list(self._dataset_specs)
        self._shared_params = dict(shared_params)
        self._dataset_overrides: list[FitDatasetParameterOverrides] = coerce_fit_dataset_parameter_overrides(
            dataset_ids=[spec.dataset_id for spec in self._dataset_specs],
            dataset_overrides=dataset_overrides,
            dataset_params=dataset_params,
            dataset_variable_params=dataset_variable_params,
        )
        dataset_params_map, dataset_variable_params_map = split_fit_dataset_parameter_overrides(self._dataset_overrides)
        self._dataset_params = {
            ds_id: {name: float(value) for name, value in param_map.items()}
            for ds_id, param_map in dataset_params_map.items()
        }
        self._dataset_variable_params = {
            ds_id: {
                name: {
                    "initial": float(spec["initial"]),
                    "min": float(spec["min"]),
                    "max": float(spec["max"]),
                    "log10": bool(spec.get("log10", False)),
                }
                for name, spec in param_map.items()
            }
            for ds_id, param_map in dataset_variable_params_map.items()
        }
        self._bounds = dict(bounds or {})
        self._weights = dict(weights or {})
        self._method = method
        self._max_nfev = max(1, int(max_nfev))
        self._ftol = float(ftol)
        self._xtol = float(xtol)
        self._seed = seed
        self._log10_params = {str(k): bool(v) for k, v in (log10_params or {}).items()}
        self._fit_evaluator = coerce_fitting_series_evaluator(fit_evaluator) if fit_evaluator is not None else None
        self._fit_runtime_session = fit_runtime_session
        self._fit_runtime_max_lanes = None if fit_runtime_max_lanes is None else max(1, int(fit_runtime_max_lanes))
        self._fit_runtime_ledger = fit_runtime_ledger
        if fit_func is None:
            from kindred.core.analysis.global_fitting import fit_global as default_fit_global

            fit_func = default_fit_global
        self._fit_func = fit_func
        solver_label = str(solver or FITTING_DEFAULT_SOLVER).strip() or FITTING_DEFAULT_SOLVER
        solver_method, _solver_warning = normalize_solver_name(solver_label)
        self._solver = str(solver_method)
        # Tolerances are baked into the simulation closure; kwargs accepted here for API consistency.
        self._best_update_interval_s = max(0.0, float(best_update_interval_s))
        self._render_projection_interval_s = max(0.0, float(render_projection_interval_s))
        self._cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._run_thread_ident: int | None = None
        self._run_stamp = dict(run_stamp or {}) if isinstance(run_stamp, dict) else {}
        self._run_stamp_hash = str(run_stamp_hash or "")
        self._run_stamp_short = str(run_stamp_short or "")

        self._best_cost = float("inf")
        self._best_iteration = 0
        self._best_params: Dict[str, float] = {}
        self._pending_best = False
        self._last_best_emit_ts = 0.0
        self._last_render_projection_emit_ts = 0.0

    def cancel(self) -> None:
        """Request cancellation from the worker."""
        self._cancelled = True
        self._pause_event.set()
        cancel_runtime = getattr(self._fit_runtime_session, "cancel_run", None)
        if callable(cancel_runtime):
            cancel_runtime()

    def pause(self) -> None:
        """Pause optimization at the next cooperative boundary."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume a paused optimization run."""
        self._pause_event.set()

    def _wait_if_paused(self) -> None:
        if self._cancelled:
            return
        self._pause_event.wait()

    def run(self) -> None:
        self._run_thread_ident = threading.get_ident()
        try:
            payload = self._execute()
        except FittingCancelled as exc:
            self.error.emit(self._failure_payload_from_exception(exc))
            return
        except RuntimeError as exc:  # pragma: no cover - defensive boundary
            self.error.emit(self._failure_payload_from_exception(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Global fit failed: %s", exc, exc_info=True)
            self.error.emit(self._failure_payload_from_exception(exc))
        else:
            if payload is not None:
                self.finished.emit(payload)
        finally:
            self._run_thread_ident = None

    def _failure_payload_from_exception(self, exc: BaseException) -> dict[str, Any]:
        def _with_run_stamp(payload: dict[str, Any]) -> dict[str, Any]:
            payload["run_stamp"] = dict(self._run_stamp)
            payload["run_stamp_hash"] = str(self._run_stamp_hash)
            payload["run_stamp_short"] = str(self._run_stamp_short)
            return payload

        if isinstance(exc, FitSimulationError):
            nested = exc.details.get("failure") if isinstance(getattr(exc, "details", None), Mapping) else None
            if isinstance(nested, Mapping) and "kind" in nested and "message" in nested:
                return _with_run_stamp(coerce_simulation_failure(nested))
            return _with_run_stamp(
                build_simulation_failure(
                    "fitting_error",
                    str(getattr(exc, "message", None) or str(exc)),
                    code=getattr(exc, "code", None),
                    context=getattr(exc, "context", None),
                    details=getattr(exc, "details", None),
                    exc_type=exc.__class__.__name__,
                )
            )
        if isinstance(exc, FittingCancelled):
            details = dict(getattr(exc, "details", None) or {})
            original_message = str(getattr(exc, "message", None) or str(exc) or "")
            if original_message and original_message != "Fit cancelled by user":
                details.setdefault("origin_message", original_message)
            return _with_run_stamp(
                build_simulation_failure(
                    "cancelled",
                    "Fit cancelled by user",
                    code=getattr(exc, "code", None),
                    context=getattr(exc, "context", None),
                    details=details,
                    exc_type=exc.__class__.__name__,
                )
            )
        return _with_run_stamp(
            build_simulation_failure(
                "fitting_error",
                str(exc) or exc.__class__.__name__,
                exc_type=exc.__class__.__name__,
            )
        )

    def _execute(self) -> Optional[GlobalFitFinishedPayloadV1]:
        """Execute fit_global with progress and cancellation hooks."""
        if not self._datasets:
            raise RuntimeError("No datasets were provided.")
        if self._fit_evaluator is None:
            raise RuntimeError("Fit evaluator is not configured.")
        if type(self._fit_evaluator) is SerialFittingEvaluator and not self._runtime_session_ready_for_fit():
            raise RuntimeError("A required fitting runtime session is not ready for this global fit worker.")

        def progress_callback(iteration: int, cost: float, params: Dict[str, float]) -> None:
            self._wait_if_paused()
            if self._cancelled:
                raise FittingCancelled()
            fraction = int(100 * iteration / max(1, self._max_nfev))
            percent = min(95, max(5, fraction))
            message = f"Iteration {iteration}: SSQ={cost:.4g}"
            self.progress.emit(percent, message)
            self._maybe_emit_best(iteration, cost, params)

        def cancellation_check() -> bool:
            self._wait_if_paused()
            return self._cancelled

        def wait_for_resume(timeout_s: float) -> bool:
            if self._cancelled:
                return True
            return bool(self._pause_event.wait(timeout=float(timeout_s)))

        cancellation_check._kindred_nonblocking_cancelled = lambda: self._cancelled
        cancellation_check._kindred_nonblocking_paused = lambda: not self._pause_event.is_set()
        cancellation_check._kindred_wait_for_resume = wait_for_resume

        self._best_cost = float("inf")
        self._best_iteration = 0
        self._best_params = {}
        self._pending_best = False
        self._last_best_emit_ts = 0.0
        self._last_render_projection_emit_ts = 0.0

        self.progress.emit(5, f"Running global fit... [{self._solver}]")
        result = self._fit_func(
            self._fit_evaluator,
            self._datasets,
            dict(self._shared_params),
            dataset_overrides=list(self._dataset_overrides),
            bounds=self._bounds,
            weights=self._weights,
            method=self._method,
            max_nfev=self._max_nfev,
            ftol=self._ftol,
            xtol=self._xtol,
            seed=self._seed,
            log10_params=self._log10_params,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            runtime_session=self._fit_runtime_session,
            max_runtime_lanes=self._fit_runtime_max_lanes,
            runtime_ledger=self._fit_runtime_ledger,
        )
        if self._cancelled:
            raise FittingCancelled()

        self._flush_pending_best()
        self.progress.emit(100, "Fit complete")
        return self._build_finished_payload(result)

    def _runtime_session_ready_for_fit(self) -> bool:
        runtime_session = self._fit_runtime_session
        if runtime_session is None:
            return False
        is_ready = getattr(runtime_session, "is_ready", None)
        if not callable(is_ready):
            return False
        try:
            if self._fit_runtime_max_lanes is None:
                lane_count = None
            else:
                lane_count = max(1, int(self._fit_runtime_max_lanes))
            return bool(is_ready(lane_count=lane_count))
        except Exception:
            return False

    def _build_finished_payload(self, result: "GlobalFitResult") -> GlobalFitFinishedPayloadV1:
        return {
            "version": 1,
            "result": result,
            "weights": dict(self._weights),
            "run_stamp": dict(self._run_stamp),
            "run_stamp_hash": str(self._run_stamp_hash),
            "run_stamp_short": str(self._run_stamp_short),
        }

    def _maybe_emit_best(self, iteration: int, cost: float, params: Dict[str, float]) -> None:
        """Emit bestUpdated when cost improves (throttled)."""
        try:
            cost_val = float(cost)
        except Exception:
            return
        if not np.isfinite(cost_val):
            return
        if cost_val >= self._best_cost:
            return

        self._best_cost = cost_val
        self._best_iteration = int(iteration)
        self._best_params = {str(k): float(v) for k, v in (params or {}).items()}
        self._pending_best = True
        now = time.monotonic()
        if self._best_update_interval_s <= 0.0 or (now - self._last_best_emit_ts) >= self._best_update_interval_s:
            self._emit_best_payload()

    def _flush_pending_best(self) -> None:
        if not self._best_params:
            return
        self._pending_best = True
        self._emit_best_payload()

    def _emit_best_payload(self) -> None:
        self._wait_if_paused()
        if self._cancelled or not self._pending_best or not self._best_params:
            self._pending_best = False
            return

        now = time.monotonic()
        shared_params = self._best_payload_shared_params()
        dataset_params = self._best_payload_dataset_params()
        render_projection = self._maybe_build_live_render_projection(
            now=now,
            shared_params=shared_params,
            dataset_params=dataset_params,
        )

        payload: GlobalFitBestUpdatedPayloadV1 = {
            "version": 1,
            "iteration": int(self._best_iteration),
            "cost": float(self._best_cost),
            "shared_params": dict(shared_params),
            "dataset_params": dataset_params,
            "run_stamp": dict(self._run_stamp),
            "run_stamp_hash": str(self._run_stamp_hash),
            "run_stamp_short": str(self._run_stamp_short),
            "render_projection": render_projection,
        }
        self._pending_best = False
        self._last_best_emit_ts = now
        self.bestUpdated.emit(payload)

    def _maybe_build_live_render_projection(
        self,
        *,
        now: float,
        shared_params: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
    ) -> Optional[FitRenderProjection]:
        if self._cancelled or not self._run_stamp_hash:
            return None
        if self._fit_evaluator is None:
            return None
        elapsed = float(now) - float(self._last_render_projection_emit_ts)
        if self._last_render_projection_emit_ts > 0.0 and elapsed < self._render_projection_interval_s:
            return None
        try:
            layout = build_parameter_layout(
                payloads=list(self._dataset_specs),
                shared_params=dict(self._shared_params),
                dataset_variable_params=dict(self._dataset_variable_params),
                bounds=dict(self._bounds),
                log10_params=dict(self._log10_params),
            )
            weights = normalize_weights(list(self._dataset_specs), dict(self._weights))
            fit_evaluator = self._live_projection_evaluator()
            result = assemble_global_fit_result(
                fit_evaluator=fit_evaluator,
                payloads=list(self._dataset_specs),
                layout=layout,
                fitted_params=dict(shared_params),
                combined_dataset_params={str(ds_id): dict(values) for ds_id, values in dataset_params.items()},
                weights=weights,
                penalty_value=1e6,
                cancellation_check=lambda: bool(self._cancelled),
                success=True,
                message="Live best projection",
                nfev=int(self._best_iteration),
                covariance=None,
                objective_residuals=None,
                uncertainties=None,
                optimizer_diagnostic=None,
            )
            projection = projection_from_global_fit_result(
                result,
                run_stamp_hash=str(self._run_stamp_hash),
                phase="live",
                cost=float(self._best_cost),
            )
        except Exception as exc:
            logger.debug("Skipping live fit render projection: %s", exc)
            return None
        self._last_render_projection_emit_ts = float(now)
        return projection

    def _live_projection_evaluator(self) -> object:
        runtime_session = self._fit_runtime_session
        if runtime_session is not None:
            evaluator = getattr(runtime_session, "evaluator", None)
            if callable(evaluator):
                return evaluator(cancellation_check=lambda: bool(self._cancelled))
        return self._fit_evaluator

    def _best_payload_shared_params(self) -> dict[str, float]:
        shared_params = dict(self._shared_params)
        for name in list(shared_params.keys()):
            if name in self._best_params:
                shared_params[name] = float(self._best_params[name])
        return {str(k): float(v) for k, v in shared_params.items()}

    def _best_payload_dataset_params(self) -> dict[str, dict[str, float]]:
        dataset_params: dict[str, dict[str, float]] = {
            str(spec.dataset_id): dict(self._dataset_params.get(spec.dataset_id, {}))
            for spec in self._dataset_specs
            if str(getattr(spec, "dataset_id", "") or "").strip()
        }
        for key, value in self._best_params.items():
            if "::" not in key:
                continue
            ds_id, param_name = key.split("::", 1)
            if not ds_id or not param_name:
                continue
            dataset_params.setdefault(ds_id, {})
            dataset_params[ds_id][param_name] = float(value)
        return dataset_params

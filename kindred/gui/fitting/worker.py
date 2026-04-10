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
from kindred.core.fitting_evaluation import coerce_fitting_series_evaluator
from kindred.core.simulation_series_payload import coerce_simulation_series_payload
from kindred.core.exceptions import FitSimulationError, FittingCancelled
from kindred.core.api.fitting import fit_global
from kindred.core.simulation_failure import build_simulation_failure, coerce_simulation_failure
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER

if TYPE_CHECKING:
    from kindred.core.api.fitting import GlobalFitResult


logger = logging.getLogger(__name__)


__all__ = [
    "GlobalFitWorker",
    "fit_global",
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
    model_series: Any
    residual_series: Any
    plot_model_series: Any
    plot_model_x: Any
    dataset_stats: Any


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
        fit_func: Optional[Callable[..., "GlobalFitResult"]] = None,
        solver: str = FITTING_DEFAULT_SOLVER,
        rtol: float = 1e-6,
        atol: float = 1e-12,
        best_update_interval_s: float = 0.25,
        plot_update_interval_s: float = 2.0,
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
        self._fit_func = fit_func or fit_global
        solver_label = str(solver or FITTING_DEFAULT_SOLVER).strip() or FITTING_DEFAULT_SOLVER
        solver_method, _solver_warning = normalize_solver_name(solver_label)
        self._solver = str(solver_method)
        # Tolerances are baked into the simulation closure; kwargs accepted here for API consistency.
        self._best_update_interval_s = max(0.0, float(best_update_interval_s))
        self._heavy_update_interval_s = max(0.0, float(plot_update_interval_s))
        self._cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._run_stamp = dict(run_stamp or {}) if isinstance(run_stamp, dict) else {}
        self._run_stamp_hash = str(run_stamp_hash or "")
        self._run_stamp_short = str(run_stamp_short or "")

        self._best_cost = float("inf")
        self._best_iteration = 0
        self._best_params: Dict[str, float] = {}
        self._pending_best = False
        self._last_best_emit_ts = 0.0
        self._last_heavy_emit_ts = time.monotonic()
        self._last_best_emit_had_plot_payload = False
        self._best_payload_exception_counts: Dict[str, int] = {}

    def _record_best_payload_exception(self, key: str, *, message: str, exc: Exception) -> None:
        count = int(self._best_payload_exception_counts.get(key, 0)) + 1
        self._best_payload_exception_counts[key] = count
        if count <= 3:
            logger.debug("%s (count=%d): %s", message, count, exc, exc_info=True)

    def cancel(self) -> None:
        """Request cancellation from the worker."""
        self._cancelled = True
        self._pause_event.set()

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

    def _failure_payload_from_exception(self, exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, FitSimulationError):
            nested = exc.details.get("failure") if isinstance(getattr(exc, "details", None), Mapping) else None
            if isinstance(nested, Mapping) and "kind" in nested and "message" in nested:
                return coerce_simulation_failure(nested)
            return build_simulation_failure(
                "fitting_error",
                str(getattr(exc, "message", None) or str(exc)),
                code=getattr(exc, "code", None),
                context=getattr(exc, "context", None),
                details=getattr(exc, "details", None),
                exc_type=exc.__class__.__name__,
            )
        if isinstance(exc, FittingCancelled):
            details = dict(getattr(exc, "details", None) or {})
            original_message = str(getattr(exc, "message", None) or str(exc) or "")
            if original_message and original_message != "Fit cancelled by user":
                details.setdefault("origin_message", original_message)
            return build_simulation_failure(
                "cancelled",
                "Fit cancelled by user",
                code=getattr(exc, "code", None),
                context=getattr(exc, "context", None),
                details=details,
                exc_type=exc.__class__.__name__,
            )
        return build_simulation_failure(
            "fitting_error",
            str(exc) or exc.__class__.__name__,
            exc_type=exc.__class__.__name__,
        )

    def _execute(self) -> Optional[GlobalFitFinishedPayloadV1]:
        """Execute fit_global with progress and cancellation hooks."""
        if not self._datasets:
            raise RuntimeError("No datasets were provided.")
        if self._fit_evaluator is None:
            raise RuntimeError("Fit evaluator is not configured.")

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

        self._best_cost = float("inf")
        self._best_iteration = 0
        self._best_params = {}
        self._pending_best = False
        self._last_best_emit_ts = 0.0
        self._last_heavy_emit_ts = time.monotonic()
        self._last_best_emit_had_plot_payload = False

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
        )
        if self._cancelled:
            raise FittingCancelled()

        self._flush_pending_best()
        self.progress.emit(100, "Fit complete")
        return self._build_finished_payload(result)

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
        if not self._pending_best and self._last_best_emit_had_plot_payload:
            return
        self._pending_best = True
        self._emit_best_payload(force_heavy=True)

    def _emit_best_payload(self, *, force_heavy: bool = False) -> None:
        self._wait_if_paused()
        if self._cancelled or not self._pending_best or not self._best_params:
            self._pending_best = False
            return

        now = time.monotonic()
        shared_params = self._best_payload_shared_params()
        dataset_params = self._best_payload_dataset_params()
        include_plot_payload = bool(
            force_heavy
            or self._heavy_update_interval_s <= 0.0
            or (now - self._last_heavy_emit_ts) >= self._heavy_update_interval_s
        )
        if include_plot_payload:
            model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = self._build_best_payload_series(
                shared_params=shared_params,
                dataset_params=dataset_params,
            )
            self._last_heavy_emit_ts = now
        else:
            model_series = None
            residual_series = None
            plot_model_series = None
            plot_model_x = None
            dataset_stats = None

        payload: GlobalFitBestUpdatedPayloadV1 = {
            "version": 1,
            "iteration": int(self._best_iteration),
            "cost": float(self._best_cost),
            "shared_params": dict(shared_params),
            "dataset_params": dataset_params,
            "model_series": model_series,
            "residual_series": residual_series,
            "plot_model_series": plot_model_series,
            "plot_model_x": plot_model_x,
            "dataset_stats": dataset_stats,
        }
        self._pending_best = False
        self._last_best_emit_ts = now
        self._last_best_emit_had_plot_payload = include_plot_payload
        self.bestUpdated.emit(payload)

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

    def _build_best_payload_series(
        self,
        *,
        shared_params: dict[str, float],
        dataset_params: dict[str, dict[str, float]],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
        dict[str, np.ndarray],
        dict[str, dict[str, float]],
    ]:
        model_series: dict[str, dict[str, np.ndarray]] = {}
        residual_series: dict[str, dict[str, np.ndarray]] = {}
        plot_model_series: dict[str, dict[str, np.ndarray]] = {}
        plot_model_x: dict[str, np.ndarray] = {}
        dataset_stats: dict[str, dict[str, float]] = {}

        for spec in self._dataset_specs:
            ds_id = str(spec.dataset_id).strip()
            if not ds_id:
                continue
            if self._cancelled:
                break
            try:
                self._wait_if_paused()
                if self._cancelled:
                    break
                dataset_payload = self._best_payload_for_dataset(
                    spec,
                    shared_params=shared_params,
                    dataset_params=dataset_params,
                )
                if dataset_payload is None:
                    continue
                (
                    model_for_ds,
                    residual_for_ds,
                    plot_for_ds,
                    plot_x,
                    stats_for_ds,
                ) = dataset_payload
                if model_for_ds:
                    model_series[ds_id] = model_for_ds
                if residual_for_ds:
                    residual_series[ds_id] = residual_for_ds
                if plot_for_ds:
                    plot_model_series[ds_id] = plot_for_ds
                if plot_x is not None:
                    plot_model_x[ds_id] = plot_x
                if stats_for_ds is not None:
                    dataset_stats[ds_id] = stats_for_ds
            except Exception as exc:
                self._record_best_payload_exception(
                    f"dataset::{ds_id}",
                    message=f"Best-update payload construction failed for dataset '{ds_id}'",  # nosec B608 - not SQL construction
                    exc=exc,
                )
                continue

        return model_series, residual_series, plot_model_series, plot_model_x, dataset_stats

    def _best_payload_for_dataset(
        self,
        spec: FitDatasetSpec,
        *,
        shared_params: dict[str, float],
        dataset_params: dict[str, dict[str, float]],
    ) -> Optional[
        tuple[
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            dict[str, np.ndarray],
            Optional[np.ndarray],
            Optional[dict[str, float]],
        ]
    ]:
        ds_id = str(spec.dataset_id).strip()
        if not ds_id:
            return None
        species_list = [str(x) for x in (spec.species_list or []) if str(x)]
        y_matrix = np.asarray(spec.y_matrix, dtype=float)
        t_exp = np.asarray(spec.t_exp, dtype=float).reshape(-1)
        x_name = str(spec.x_name or "t").strip() or "t"
        x_obs = (
            np.asarray(spec.x_obs, dtype=float).reshape(-1)
            if spec.x_obs is not None and x_name != "t"
            else None
        )
        x_mode = str(spec.x_mode or "auto").strip() or "auto"

        full_params = dict(shared_params)
        full_params.update(dataset_params.get(ds_id, {}))
        sim_time, sim_species = self._simulate_best_payload_result(full_params)

        plot_x_and_mask = self._best_payload_plot_x_and_mask(
            ds_id=ds_id,
            t_exp=t_exp,
            x_name=x_name,
            x_obs=x_obs,
            x_mode=x_mode,
            sim_time=sim_time,
            sim_species=sim_species,
        )
        if plot_x_and_mask is None:
            return None
        plot_x, plot_mask = plot_x_and_mask

        model_for_ds, residual_for_ds, plot_for_ds, stats_for_ds = self._best_payload_species_series(
            ds_id=ds_id,
            species_list=species_list,
            y_matrix=y_matrix,
            t_exp=t_exp,
            x_name=x_name,
            x_obs=x_obs,
            x_mode=x_mode,
            sim_time=sim_time,
            sim_species=sim_species,
            plot_mask=plot_mask,
        )
        return model_for_ds, residual_for_ds, plot_for_ds, plot_x, stats_for_ds

    def _simulate_best_payload_result(
        self,
        full_params: dict[str, float],
    ) -> tuple[Optional[np.ndarray], dict[str, np.ndarray]]:
        if self._fit_evaluator is None:
            sim_result = {}
        else:
            sim_result = self._fit_evaluator.evaluate_series(full_params)
        payload = coerce_simulation_series_payload(sim_result)
        sim_time = np.asarray(payload.t, dtype=float).reshape(-1)
        return (sim_time if sim_time.size else None), dict(payload.species)

    def _best_payload_plot_x_and_mask(
        self,
        *,
        ds_id: str,
        t_exp: np.ndarray,
        x_name: str,
        x_obs: Optional[np.ndarray],
        x_mode: str,
        sim_time: Optional[np.ndarray],
        sim_species: dict[str, np.ndarray],
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        if x_name == "t":
            return np.asarray(t_exp, dtype=float).reshape(-1), None
        if sim_time is None:
            return None
        x_model_series = sim_species.get(x_name)
        if x_model_series is None:
            return None
        t0 = float(np.min(t_exp))
        t1 = float(np.max(t_exp))
        if t0 > t1:
            t0, t1 = t1, t0
        t_scale = max(1.0, abs(t0), abs(t1))
        t_pad = 1e-12 * t_scale
        plot_mask = (sim_time >= (t0 - t_pad)) & (sim_time <= (t1 + t_pad))
        if not np.any(plot_mask):
            return None
        x_seg = np.asarray(x_model_series, dtype=float).reshape(-1)[plot_mask]
        if x_seg.size < 2 or not np.all(np.isfinite(x_seg)):
            return None
        if x_mode == "monotone":
            x_scale = max(1.0, float(np.max(np.abs(x_seg))))
            x_tol = 1e-12 * x_scale
            diffs = np.diff(x_seg)
            inc = bool(np.all(diffs > x_tol))
            dec = bool(np.all(diffs < -x_tol))
            if not (inc or dec):
                return None
            if x_obs is None:
                return None
            if bool(
                np.any(x_obs < (float(np.min(x_seg)) - x_tol))
                or np.any(x_obs > (float(np.max(x_seg)) + x_tol))
            ):
                return None
        return np.asarray(x_seg, dtype=float).reshape(-1), plot_mask

    def _best_payload_species_series(
        self,
        *,
        ds_id: str,
        species_list: list[str],
        y_matrix: np.ndarray,
        t_exp: np.ndarray,
        x_name: str,
        x_obs: Optional[np.ndarray],
        x_mode: str,
        sim_time: Optional[np.ndarray],
        sim_species: dict[str, np.ndarray],
        plot_mask: Optional[np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], Optional[dict[str, float]]]:
        model_for_ds: dict[str, np.ndarray] = {}
        residual_for_ds: dict[str, np.ndarray] = {}
        plot_for_ds: dict[str, np.ndarray] = {}

        residual_blocks: list[np.ndarray] = []
        exp_blocks: list[np.ndarray] = []
        for idx, species_name in enumerate(species_list):
            if species_name not in sim_species:
                continue
            y_exp = y_matrix[idx].reshape(-1)
            y_sim_raw = sim_species[species_name]
            if y_sim_raw.size == t_exp.size:
                y_sim_time = y_sim_raw
            elif sim_time is not None and y_sim_raw.size == sim_time.size:
                y_sim_time = np.interp(t_exp, sim_time, y_sim_raw)
            else:
                continue
            model_for_ds[species_name] = y_sim_time

            if x_name == "t":
                y_sim_resid = y_sim_time
                plot_for_ds[species_name] = y_sim_time
            else:
                if sim_time is None or x_obs is None:
                    continue
                x_model_series = sim_species.get(x_name)
                if x_model_series is None:
                    continue
                try:
                    y_sim_resid = self._align_y_sim_for_best_payload(
                        ds_id=ds_id,
                        species_name=species_name,
                        x_name=x_name,
                        x_mode=x_mode,
                        t_exp=t_exp,
                        x_obs=x_obs,
                        sim_time=sim_time,
                        x_model_series=np.asarray(x_model_series, dtype=float),
                        y_sim_raw=np.asarray(y_sim_raw, dtype=float),
                    )
                except Exception as exc:
                    self._record_best_payload_exception(
                        f"align::{ds_id}",
                        message=f"Best-update alignment failed for dataset '{ds_id}' ({x_name} vs t) y='{species_name}'",  # nosec B608 - not SQL construction
                        exc=exc,
                    )
                    continue
                if plot_mask is not None:
                    plot_vals = np.asarray(y_sim_raw, dtype=float).reshape(-1)
                    plot_for_ds[species_name] = np.asarray(plot_vals[plot_mask], dtype=float).reshape(-1)

            residual = np.asarray(y_sim_resid, dtype=float).reshape(-1) - y_exp
            residual_for_ds[species_name] = residual
            residual_blocks.append(residual)
            exp_blocks.append(y_exp)

        stats_for_ds: Optional[dict[str, float]] = None
        if residual_blocks and exp_blocks:
            residuals = np.concatenate(residual_blocks)
            y_exp_all = np.concatenate(exp_blocks)
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y_exp_all - float(np.mean(y_exp_all))) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            chi_squared = ss_res / residuals.size if residuals.size else float("inf")
            stats_for_ds = {"r_squared": r_squared, "chi_squared": chi_squared}
        return model_for_ds, residual_for_ds, plot_for_ds, stats_for_ds

    def _align_y_sim_for_best_payload(
        self,
        *,
        ds_id: str,
        species_name: str,
        x_name: str,
        x_mode: str,
        t_exp: np.ndarray,
        x_obs: np.ndarray,
        sim_time: np.ndarray,
        x_model_series: np.ndarray,
        y_sim_raw: np.ndarray,
    ) -> np.ndarray:
        from kindred.core.analysis.parametric_alignment import (
            align_y_on_x_obs,
            align_y_on_x_obs_time_guided,
            is_non_monotone_in_sampled_window_error,
        )
        from kindred.core.exceptions import FitSimulationError

        if x_mode == "monotone":
            return align_y_on_x_obs(
                t_obs=t_exp,
                x_obs=x_obs,
                t_sim=sim_time,
                x_model=x_model_series,
                y_model=y_sim_raw,
                dataset_label=ds_id,
                x_name=x_name,
                y_name=species_name,
            )
        if x_mode == "time_guided":
            return align_y_on_x_obs_time_guided(
                t_obs=t_exp,
                x_obs=x_obs,
                t_sim=sim_time,
                x_model=x_model_series,
                y_model=y_sim_raw,
                dataset_label=ds_id,
                x_name=x_name,
                y_name=species_name,
            )
        try:
            return align_y_on_x_obs(
                t_obs=t_exp,
                x_obs=x_obs,
                t_sim=sim_time,
                x_model=x_model_series,
                y_model=y_sim_raw,
                dataset_label=ds_id,
                x_name=x_name,
                y_name=species_name,
            )
        except FitSimulationError as exc:
            if not is_non_monotone_in_sampled_window_error(exc):
                raise
        return align_y_on_x_obs_time_guided(
            t_obs=t_exp,
            x_obs=x_obs,
            t_sim=sim_time,
            x_model=x_model_series,
            y_model=y_sim_raw,
            dataset_label=ds_id,
            x_name=x_name,
            y_name=species_name,
        )

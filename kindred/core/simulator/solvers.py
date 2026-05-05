# SPDX-License-Identifier: MIT
"""
ODE solver wrappers for Kindred.

SciPy is a hard dependency; all integration routes through `scipy.integrate.solve_ivp` with:
- Deterministic `t_eval` output grid
- Optional TemperatureSchedule wrapper that injects T into RHS(T=...) calls
- Optional progress callback wrapper
- Optional positivity clamp (post-processing)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
from kindred.core.scipy_integrate import load_scipy_integrate

from kindred.core.temperature import TemperatureScheduleDictProtocol, TemperatureScheduleProtocol
from kindred.core.intervention_schedule import (
    InterventionInstantEvent,
    InterventionInterval,
    InterventionSchedule,
    InterventionScheduleError,
    active_interval_boundaries,
    coerce_intervention_schedule,
    events_at_time,
    intervals_active_at,
)
from kindred.core.exceptions import (
    InitialConditionError,
    SimulationCancelled,
    TimeGridError,
    create_solver_error,
)
from kindred.core.time_grid import build_time_grid
from .jacobian import JacobianConfig, compute_jacobian

logger = logging.getLogger(__name__)

DEFAULT_SOLVER_NAME = "BDF"


def _solve_ivp(*, fun: Callable[[float, np.ndarray], np.ndarray], t_span: Tuple[float, float], y0: np.ndarray, **kwargs: Any):
    solve_ivp = load_scipy_integrate()
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


class ODERhsNoTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray: ...


class ODERhsWithTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


ODERhs = ODERhsNoTemp | ODERhsWithTemp

Rhs2 = Callable[[float, np.ndarray], np.ndarray]


__all__ = [
    "SimulationRequest",
    "SimulationOutput",
    "DEFAULT_SOLVER_NAME",
    "build_time_grid",
    "normalize_solver_name",
    "solve_ode",
]


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
    t_span: Tuple[float, float]
    y0: np.ndarray
    solver: str = DEFAULT_SOLVER_NAME
    rtol: float = 1e-6
    atol: float = 1e-12
    max_step: Optional[float] = None
    first_step: Optional[float] = None
    t_eval: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, float | int]] = None
    rosenbrock_jacobian: JacobianConfig = JacobianConfig()
    jacobian_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None
    event_terminal: Optional[Iterable[bool]] = None
    positivity: Optional[str] = None
    pos_indices: Optional[Iterable[int]] = None

    progress_callback: Optional[Callable[[float, float, float], None]] = None
    temperature_schedule: TemperatureScheduleProtocol | None = None
    intervention_schedule: InterventionSchedule | Mapping[str, Any] | None = None
    species_names: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class SimulationOutput:
    t: np.ndarray
    Y: np.ndarray
    provenance: Dict[str, object]
    fallback_occurred: bool = False
    fallback_message: Optional[str] = None


def _scipy_method_for(name: object) -> Tuple[str, Optional[str]]:
    """Map solver name to SciPy method name with correct capitalization."""
    n = str(name or "").strip().upper()
    scipy_methods = {
        "RADAU": "Radau",
        "BDF": "BDF",
    }
    if n in scipy_methods:
        return scipy_methods[n], None
    return DEFAULT_SOLVER_NAME, f"Unknown solver name; using {DEFAULT_SOLVER_NAME}"


def normalize_solver_name(name: object) -> Tuple[str, Optional[str]]:
    """
    Normalize a user-specified solver name to a SciPy `solve_ivp` method name.

    Returns (method, warning). The warning is non-empty when an unknown
    name is mapped to a supported solver.
    """
    return _scipy_method_for(name)


def _unpack_banded_jacobian(J: np.ndarray, *, ml: int, mu: int) -> np.ndarray:
    n = J.shape[1]
    Jd = np.zeros((n, n), float)
    for j in range(n):
        i_min = max(0, j - mu)
        i_max = min(n - 1, j + ml)
        band_rows = slice(mu + i_min - j, mu + i_max - j + 1)
        Jd[i_min : i_max + 1, j] = J[band_rows, j]
    return Jd


def _make_scipy_jac(rhs: Callable[[float, np.ndarray], np.ndarray], cfg: "JacobianConfig"):
    def jac(t: float, y: np.ndarray) -> np.ndarray:
        J, kind = compute_jacobian(rhs, t, y, cfg=cfg)
        J_arr = np.asarray(J, dtype=float)
        if kind.startswith("banded("):
            ml, mu = cfg.validate_for(J_arr.shape[1])
            return _unpack_banded_jacobian(J_arr, ml=ml, mu=mu)
        return J_arr

    return jac


class _TemperatureInjectedRhs:
    def __init__(self, rhs: ODERhsWithTemp, schedule: TemperatureScheduleProtocol) -> None:
        self._rhs = rhs
        self._schedule = schedule

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        T = float(self._schedule(t))
        return self._rhs(t, y, T=T)


class _ProgressRhs:
    def __init__(
        self,
        rhs: Rhs2,
        *,
        callback: Callable[[float, float, float], None],
        t0: float,
        t1: float,
        every: int = 10,
    ) -> None:
        self._rhs = rhs
        self._callback = callback
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._every = int(every)
        self._span = float(self._t1 - self._t0)
        self._last_bucket = -1

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        if self._every > 0 and np.isfinite(self._span):
            if self._span <= 0.0:
                bucket = 0
            else:
                frac = (float(t) - self._t0) / self._span
                if frac < 0.0:
                    frac = 0.0
                elif frac > 1.0:
                    frac = 1.0
                bucket = int(frac * float(self._every))
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                self._callback(float(t), self._t0, self._t1)
        return self._rhs(t, y)


class _InterventionRhs:
    def __init__(
        self,
        rhs: Rhs2,
        *,
        intervals: Sequence[InterventionInterval],
        species_index: Mapping[str, int],
    ) -> None:
        self._rhs = rhs
        self._intervals = tuple(intervals)
        self._species_index = dict(species_index)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        y_eval = np.asarray(y, dtype=float).copy()
        fixed: dict[int, float] = {}
        for interval in self._intervals:
            idx = self._species_index.get(interval.species)
            if idx is None:
                continue
            if interval.kind in {"reservoir", "clamp"}:
                value = float(interval.value if interval.value is not None else 0.0)
                y_eval[idx] = value
                fixed[idx] = value
        dy = np.asarray(self._rhs(t, y_eval), dtype=float).copy()
        for interval in self._intervals:
            idx = self._species_index.get(interval.species)
            if idx is None:
                continue
            if interval.kind == "source":
                dy[idx] += float(interval.rate if interval.rate is not None else 0.0)
            elif interval.kind == "sink":
                dy[idx] -= float(interval.rate if interval.rate is not None else 0.0)
        for idx in fixed:
            dy[idx] = 0.0
        return dy


def _prepare_time_grid(req: SimulationRequest, *, t0: float, t1: float) -> np.ndarray:
    if req.t_eval is None:
        return build_time_grid(t0, t1, req.grid or {"N": 2})
    t_eval = np.asarray(req.t_eval, dtype=float).reshape(-1)
    if t_eval.size == 0:
        raise TimeGridError("t_eval must contain at least one time point")
    if np.any(~np.isfinite(t_eval)):
        raise TimeGridError("t_eval must contain only finite values")
    if np.any(np.diff(t_eval) <= 0):
        raise TimeGridError("t_eval must be strictly increasing")
    if t_eval[0] < t0 or t_eval[-1] > t1:
        raise TimeGridError("t_eval must lie within t_span")
    return t_eval


def _prepare_rhs(req: SimulationRequest, *, t0: float, t1: float) -> tuple[Rhs2, Rhs2]:
    if req.temperature_schedule is None:
        rhs_for_jac: Rhs2 = cast(ODERhsNoTemp, req.rhs)
    else:
        rhs_for_jac = _TemperatureInjectedRhs(cast(ODERhsWithTemp, req.rhs), req.temperature_schedule)

    rhs: Rhs2 = rhs_for_jac
    if req.progress_callback is not None:
        rhs = _ProgressRhs(rhs_for_jac, callback=req.progress_callback, t0=t0, t1=t1)
    return rhs, rhs_for_jac


def _build_provenance(req: SimulationRequest, *, t_eval: np.ndarray) -> Dict[str, object]:
    prov: Dict[str, object] = {
        "solver_requested": req.solver,
        "rtol": float(req.rtol),
        "atol": float(req.atol),
        "max_step": (None if req.max_step is None else float(req.max_step)),
        "first_step": (None if req.first_step is None else float(req.first_step)),
        "grid_type": ("explicit" if req.t_eval is not None else ("dt" if "dt" in (req.grid or {}) else "N")),
        "grid_params": ({"t_eval": list(t_eval)} if req.t_eval is not None else dict(req.grid or {})),
        "rosenbrock_jacobian": {
            "mode": getattr(req.rosenbrock_jacobian, "mode", None),
            "ml": getattr(req.rosenbrock_jacobian, "ml", None),
            "mu": getattr(req.rosenbrock_jacobian, "mu", None),
        },
        "custom_jacobian": bool(req.jacobian_func),
        "positivity": (req.positivity or None),
        "pos_indices": (list(req.pos_indices) if req.pos_indices is not None else None),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    schedule = coerce_intervention_schedule(req.intervention_schedule)
    prov["has_intervention_schedule"] = schedule is not None
    if schedule is not None:
        prov["intervention_schedule"] = schedule.to_payload()
        prov["intervention_schedule_fingerprint"] = schedule.fingerprint

    if req.temperature_schedule is not None:
        if isinstance(req.temperature_schedule, TemperatureScheduleDictProtocol):
            try:
                prov["temperature_schedule"] = req.temperature_schedule.to_dict()
            except Exception:
                prov["temperature_schedule"] = str(req.temperature_schedule)
        else:
            prov["temperature_schedule"] = str(req.temperature_schedule)

    return prov


def _implicit_scipy_alternatives(primary: str) -> List[str]:
    order = {
        "BDF": ["Radau"],
        "Radau": ["BDF"],
    }
    return [m for m in order.get(primary, []) if m != primary]


def _apply_positivity_to_trajectory(
    Y: np.ndarray,
    *,
    mode: Optional[str],
    indices: Optional[Iterable[int]],
) -> np.ndarray:
    if not mode:
        return Y
    if indices is None:
        return np.maximum(Y, 0.0)
    Y2 = np.asarray(Y, float).copy()
    for j in indices:
        jj = int(j)
        if 0 <= jj < Y2.shape[0]:
            Y2[jj, :] = np.maximum(Y2[jj, :], 0.0)
    return Y2


def _solver_trajectory_array(sol_y: object, *, y0: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    sample_count = int(np.asarray(t_out, dtype=float).reshape(-1).size)
    species_count = int(np.asarray(y0, dtype=float).reshape(-1).size)
    Y = np.asarray(sol_y, dtype=float)
    if Y.ndim == 2:
        return Y
    if sample_count == 0:
        return np.empty((species_count, 0), dtype=float)
    if Y.ndim == 1 and species_count == 1 and Y.size == sample_count:
        return Y.reshape(1, sample_count)
    if Y.ndim == 1 and sample_count == 1 and Y.size == species_count:
        return Y.reshape(species_count, 1)
    return Y.reshape(species_count, sample_count)


def _execute_scipy(
    req: SimulationRequest,
    *,
    rhs: Rhs2,
    rhs_for_jac: Rhs2,
    t0: float,
    t1: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    prov: Dict[str, object],
    method: str,
    note: Optional[str],
) -> SimulationOutput:
    if note:
        prov["emulation_note"] = note

    base_kwargs: Dict[str, Any] = dict(t_eval=t_eval, rtol=float(req.rtol), atol=float(req.atol))
    if req.max_step is not None:
        base_kwargs["max_step"] = float(req.max_step)
    if req.first_step is not None:
        base_kwargs["first_step"] = float(req.first_step)
    events_list = _event_callables_for_request(req)
    if events_list is not None:
        base_kwargs["events"] = events_list

    jac_callable = req.jacobian_func or _make_scipy_jac(rhs_for_jac, req.rosenbrock_jacobian)
    banded_jacobian_active = (
        getattr(req.rosenbrock_jacobian, "mode", None) == "banded"
        and req.rosenbrock_jacobian.ml is not None
        and req.rosenbrock_jacobian.mu is not None
    )

    def _kwargs_for_method(method_name: str) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["method"] = method_name
        if method_name in ("Radau", "BDF"):
            kwargs["jac"] = jac_callable
            if banded_jacobian_active:
                # SciPy implicit solvers expect an n x n Jacobian when a jac
                # callable is supplied; jac_sparsity is ignored in that case.
                kwargs.pop("jac_sparsity", None)
        else:
            kwargs.pop("jac", None)
            kwargs.pop("jac_sparsity", None)
        return kwargs

    attempted_methods: List[str] = []
    fallback_occurred = False
    fallback_message: Optional[str] = None
    primary_method = method

    def _looks_like_brentq_sign_error(exc: BaseException) -> bool:
        return "different signs" in str(exc).lower()

    def _cancel_requested_via_event() -> bool:
        for ev in events_list or []:
            if not bool(getattr(ev, "_kindred_cancel_event", False)):
                continue
            cancelled_cb = getattr(ev, "_kindred_cancelled", None)
            if callable(cancelled_cb) and bool(cancelled_cb()):
                return True
        return False

    def _solve_or_cancelled(method_name: str):
        try:
            return _solve_ivp(fun=rhs, t_span=(t0, t1), y0=y0, **_kwargs_for_method(method_name))  # type: ignore[misc]
        except ValueError as exc:
            # SciPy event root-finder can raise on cancellation races (brentq sign error).
            # If we have an explicit cancellation event and it reports cancellation requested,
            # translate to a typed cancellation signal.
            if events_list and _looks_like_brentq_sign_error(exc) and _cancel_requested_via_event():
                raise SimulationCancelled() from exc
            raise

    sol = _solve_or_cancelled(method)
    attempted_methods.append(method)

    if events_list and getattr(sol, "status", 0) == 1 and getattr(sol, "t_events", None):
        for idx, ev in enumerate(events_list):
            if not bool(getattr(ev, "_kindred_cancel_event", False)):
                continue
            t_events = sol.t_events[idx] if idx < len(sol.t_events) else None
            if t_events is not None and len(t_events) > 0:
                raise SimulationCancelled()

    if not sol.success:
        prov["solver_failure"] = str(sol.message)
        last_message = str(sol.message)
        for alt_method in _implicit_scipy_alternatives(method):
            alt_sol = _solve_or_cancelled(alt_method)
            attempted_methods.append(alt_method)
            if alt_sol.success:
                sol = alt_sol
                prov["solver_alternative_used"] = alt_method
                fallback_occurred = True
                fallback_message = f"{primary_method} failed; succeeded with {alt_method}"
                method = alt_method
                break
            last_message = str(alt_sol.message)
        if not sol.success:
            t_fail = float(sol.t[-1]) if getattr(sol, "t", None) is not None and len(sol.t) > 0 else float(t0)
            attempt_list = ", ".join(attempted_methods)
            raise create_solver_error(
                req.solver,
                t_fail,
                f"{last_message}; attempted methods: {attempt_list}",
            )

    prov["solver_used"] = method
    t_out = np.asarray(sol.t, float)
    Y_out = _apply_positivity_to_trajectory(
        _solver_trajectory_array(sol.y, y0=y0, t_out=t_out),
        mode=req.positivity,
        indices=(list(req.pos_indices) if req.pos_indices is not None else None),
    )
    if hasattr(sol, "t_events") and sol.t_events:
        prov["events"] = [list(te) for te in sol.t_events]
    return SimulationOutput(
        t=t_out,
        Y=Y_out,
        provenance=prov,
        fallback_occurred=fallback_occurred,
        fallback_message=fallback_message,
    )


def _event_terminal_flags(
    req: SimulationRequest,
    events: Sequence[Callable[[float, np.ndarray], float]],
) -> list[bool]:
    explicit = [] if req.event_terminal is None else list(req.event_terminal)
    flags: list[bool] = []
    for idx, event in enumerate(events):
        if idx < len(explicit):
            flags.append(bool(explicit[idx]))
        else:
            flags.append(bool(getattr(event, "terminal", False)))
    return flags


def _event_callables_for_request(req: SimulationRequest) -> Optional[List[Callable[[float, np.ndarray], float]]]:
    if req.events is None:
        return None
    raw_events = list(req.events)
    terminal_flags = _event_terminal_flags(req, raw_events)
    event_callables: List[Callable[[float, np.ndarray], float]] = []
    for idx, event in enumerate(raw_events):
        terminal = bool(terminal_flags[idx]) if idx < len(terminal_flags) else bool(getattr(event, "terminal", False))

        def _wrapped_event(t: float, y: np.ndarray, _event=event) -> float:
            return float(_event(t, y))

        _wrapped_event.terminal = terminal  # type: ignore[attr-defined]
        if hasattr(event, "direction"):
            _wrapped_event.direction = getattr(event, "direction")  # type: ignore[attr-defined]
        if bool(getattr(event, "_kindred_cancel_event", False)):
            _wrapped_event._kindred_cancel_event = True  # type: ignore[attr-defined]
            cancelled_cb = getattr(event, "_kindred_cancelled", None)
            if callable(cancelled_cb):
                _wrapped_event._kindred_cancelled = cancelled_cb  # type: ignore[attr-defined]
        event_callables.append(_wrapped_event)
    return event_callables


def _normal_event_provenance(
    events: object,
    *,
    count: int,
) -> list[list[float]]:
    out = [[] for _ in range(max(0, int(count)))]
    if not isinstance(events, Sequence):
        return out
    for idx, raw in enumerate(events):
        if idx >= len(out):
            break
        try:
            out[idx].extend(float(value) for value in raw)  # type: ignore[union-attr]
        except TypeError:
            continue
    return out


def _non_cancel_terminal_event_hit(
    *,
    events: Sequence[Callable[[float, np.ndarray], float]],
    terminal_flags: Sequence[bool],
    event_provenance: Sequence[Sequence[float]],
) -> bool:
    for idx, values in enumerate(event_provenance):
        if not values:
            continue
        event = events[idx] if idx < len(events) else None
        if event is not None and bool(getattr(event, "_kindred_cancel_event", False)):
            continue
        if idx < len(terminal_flags) and bool(terminal_flags[idx]):
            return True
    return False


def _species_index_for_request(req: SimulationRequest, schedule: InterventionSchedule) -> dict[str, int]:
    names = tuple(str(name) for name in (req.species_names or ()))
    if not names:
        raise InterventionScheduleError("species_names are required when intervention_schedule is present.")
    schedule.validate_species(names)
    return {name: idx for idx, name in enumerate(names)}


def _apply_instant_events(
    y: np.ndarray,
    events: Iterable[InterventionInstantEvent],
    *,
    species_index: Mapping[str, int],
) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for event in events:
        idx = species_index.get(event.species)
        if idx is None:
            raise InterventionScheduleError(f"Unknown intervention species: {event.species}")
        if event.op == "set":
            out[idx] = float(event.value if event.value is not None else 0.0)
        elif event.op == "clear":
            out[idx] = 0.0
        elif event.op == "add":
            out[idx] += float(event.amount if event.amount is not None else 0.0)
        elif event.op == "remove":
            amount = float(event.amount if event.amount is not None else 0.0)
            next_value = out[idx] - amount
            if next_value < 0.0:
                raise InterventionScheduleError(
                    f"remove intervention would make {event.species} negative at t={event.time:g}."
                )
            out[idx] = next_value
        else:
            raise InterventionScheduleError(f"Unsupported instant intervention op: {event.op!r}.")
    return out


def _segment_eval_mask(t_eval: np.ndarray, *, seg_start: float, seg_end: float, is_final: bool) -> np.ndarray:
    start_floor = float(np.nextafter(float(seg_start), -np.inf))
    end_floor = float(np.nextafter(float(seg_end), -np.inf))
    if is_final:
        end_ceiling = float(np.nextafter(float(seg_end), np.inf))
        return (t_eval >= start_floor) & (t_eval <= end_ceiling)
    return (t_eval >= start_floor) & (t_eval < end_floor)


def _snap_eval_times_to_segment_boundaries(
    t_eval: np.ndarray,
    *,
    seg_start: float,
    seg_end: float,
) -> np.ndarray:
    snapped = np.asarray(t_eval, dtype=float).reshape(-1).copy()
    if not snapped.size:
        return snapped
    start_floor = float(np.nextafter(float(seg_start), -np.inf))
    start_ceiling = float(np.nextafter(float(seg_start), np.inf))
    end_floor = float(np.nextafter(float(seg_end), -np.inf))
    end_ceiling = float(np.nextafter(float(seg_end), np.inf))
    snapped[(snapped >= start_floor) & (snapped <= start_ceiling)] = float(seg_start)
    snapped[(snapped >= end_floor) & (snapped <= end_ceiling)] = float(seg_end)
    return snapped


def _deduplicate_snapped_eval_times(t_eval: np.ndarray) -> np.ndarray:
    values = np.asarray(t_eval, dtype=float).reshape(-1)
    if values.size <= 1:
        return values
    keep = np.concatenate([np.array([True]), np.diff(values) > 0.0])
    return values[keep]


def _eval_times_include_boundary(t_eval: np.ndarray, boundary: float) -> bool:
    values = np.asarray(t_eval, dtype=float).reshape(-1)
    if not values.size:
        return False
    boundary_value = float(boundary)
    lower = float(np.nextafter(boundary_value, -np.inf))
    upper = float(np.nextafter(boundary_value, np.inf))
    return bool(np.any((values >= lower) & (values <= upper)))


def _execute_with_intervention_schedule(
    req: SimulationRequest,
    *,
    rhs: Rhs2,
    rhs_for_jac: Rhs2,
    t0: float,
    t1: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    prov: Dict[str, object],
    method: str,
    note: Optional[str],
    schedule: InterventionSchedule,
) -> SimulationOutput:
    segment_req_base = req
    events_tuple: tuple[Callable[[float, np.ndarray], float], ...] = ()
    if req.events is not None:
        events_tuple = tuple(req.events)
        segment_req_base = replace(req, events=events_tuple)
    event_provenance = _normal_event_provenance([], count=len(events_tuple))
    terminal_flags = _event_terminal_flags(segment_req_base, events_tuple)
    species_index = _species_index_for_request(req, schedule)
    boundaries = active_interval_boundaries(schedule, t0=float(t0), t1=float(t1))
    if len(boundaries) < 2:
        return _execute_scipy(
            segment_req_base,
            rhs=rhs,
            rhs_for_jac=rhs_for_jac,
            t0=t0,
            t1=t1,
            y0=y0,
            t_eval=t_eval,
            prov=prov,
            method=method,
            note=note,
        )

    current_y = _apply_instant_events(y0, events_at_time(schedule, float(t0)), species_index=species_index)
    outputs_t: list[np.ndarray] = []
    outputs_y: list[np.ndarray] = []
    fallback_occurred = False
    fallback_message = None
    segment_count = 0
    segment_solvers: list[str] = []
    segment_alternatives: list[str] = []
    custom_jacobian_disabled = False
    terminal_stop = False

    for idx in range(len(boundaries) - 1):
        seg_start = float(boundaries[idx])
        seg_end = float(boundaries[idx + 1])
        if idx > 0:
            current_y = _apply_instant_events(
                current_y,
                events_at_time(schedule, seg_start),
                species_index=species_index,
            )
        if seg_end <= seg_start:
            continue
        is_final = idx == len(boundaries) - 2
        requested_mask = _segment_eval_mask(
            t_eval,
            seg_start=seg_start,
            seg_end=seg_end,
            is_final=is_final,
        )
        requested_eval = _snap_eval_times_to_segment_boundaries(
            t_eval[requested_mask],
            seg_start=seg_start,
            seg_end=seg_end,
        )
        requested_eval = _deduplicate_snapped_eval_times(requested_eval)
        if requested_eval.size and float(requested_eval[-1]) == float(seg_end):
            internal_eval = requested_eval
            requested_count = int(requested_eval.size)
        elif requested_eval.size:
            internal_eval = np.concatenate([requested_eval, np.array([seg_end], dtype=float)])
            requested_count = int(requested_eval.size)
        else:
            internal_eval = np.array([seg_end], dtype=float)
            requested_count = 0
        active_intervals = intervals_active_at(schedule, seg_start)
        if active_intervals:
            current_y = np.asarray(current_y, dtype=float).copy()
            for interval in active_intervals:
                if interval.kind not in {"reservoir", "clamp"}:
                    continue
                fixed_idx = species_index.get(interval.species)
                if fixed_idx is not None:
                    current_y[fixed_idx] = float(interval.value if interval.value is not None else 0.0)
        seg_rhs = rhs
        seg_rhs_for_jac = rhs_for_jac
        if active_intervals:
            seg_rhs = _InterventionRhs(rhs, intervals=active_intervals, species_index=species_index)
            seg_rhs_for_jac = _InterventionRhs(rhs_for_jac, intervals=active_intervals, species_index=species_index)
        seg_req = segment_req_base
        if active_intervals and segment_req_base.jacobian_func is not None:
            custom_jacobian_disabled = True
            seg_req = replace(segment_req_base, jacobian_func=None)
        seg_prov: Dict[str, object] = dict(prov)
        seg_prov["intervention_segment_index"] = int(idx)
        seg_out = _execute_scipy(
            seg_req,
            rhs=seg_rhs,
            rhs_for_jac=seg_rhs_for_jac,
            t0=seg_start,
            t1=seg_end,
            y0=current_y,
            t_eval=internal_eval,
            prov=seg_prov,
            method=method,
            note=note if idx == 0 else None,
        )
        segment_count += 1
        fallback_occurred = fallback_occurred or bool(seg_out.fallback_occurred)
        fallback_message = fallback_message or seg_out.fallback_message
        segment_solver = str(seg_out.provenance.get("solver_used") or method)
        segment_solvers.append(segment_solver)
        alternative = seg_out.provenance.get("solver_alternative_used")
        if alternative:
            segment_alternatives.append(str(alternative))
        seg_events = _normal_event_provenance(seg_out.provenance.get("events"), count=len(events_tuple))
        if event_provenance:
            for event_idx, values in enumerate(seg_events):
                event_provenance[event_idx].extend(values)
        terminal_stop = _non_cancel_terminal_event_hit(
            events=events_tuple,
            terminal_flags=terminal_flags,
            event_provenance=seg_events,
        )
        seg_t = np.asarray(seg_out.t, dtype=float).reshape(-1)
        seg_Y = np.asarray(seg_out.Y, dtype=float)
        if seg_Y.ndim == 1:
            seg_Y = _solver_trajectory_array(seg_Y, y0=current_y, t_out=seg_t)
        if seg_Y.shape[1]:
            current_y = np.asarray(seg_Y[:, -1], dtype=float).reshape(-1)
            for interval in active_intervals:
                if interval.kind not in {"reservoir", "clamp"}:
                    continue
                fixed_idx = species_index.get(interval.species)
                if fixed_idx is not None:
                    current_y[fixed_idx] = float(interval.value if interval.value is not None else 0.0)
        elif not terminal_stop:
            raise create_solver_error(
                req.solver,
                seg_start,
                "Scheduled simulation segment produced no state samples without a terminal event.",
            )
        if requested_count:
            actual_count = min(
                int(requested_count),
                int(seg_t.size),
            )
            if actual_count:
                outputs_t.append(seg_t[:actual_count])
                outputs_y.append(seg_Y[:, :actual_count])
        if terminal_stop and seg_t.size:
            terminal_time = float(seg_t[-1])
            if _eval_times_include_boundary(t_eval, terminal_time):
                already_emitted = bool(outputs_t and outputs_t[-1].size and float(outputs_t[-1][-1]) == terminal_time)
                if not already_emitted:
                    outputs_t.append(seg_t[-1:])
                    outputs_y.append(seg_Y[:, -1:])
        if terminal_stop:
            break

    if boundaries and not terminal_stop:
        final_time = float(boundaries[-1])
        current_y = _apply_instant_events(
            current_y,
            events_at_time(schedule, final_time),
            species_index=species_index,
        )
        if t_eval.size and _eval_times_include_boundary(t_eval, final_time):
            if outputs_t and outputs_t[-1].size and float(outputs_t[-1][-1]) == final_time:
                outputs_y[-1][:, -1] = current_y
            else:
                outputs_t.append(np.array([final_time], dtype=float))
                outputs_y.append(current_y.reshape(-1, 1))

    if outputs_t:
        t_out = np.concatenate(outputs_t)
        Y_out = np.concatenate(outputs_y, axis=1)
    elif terminal_stop:
        t_out = np.empty(0, dtype=float)
        Y_out = np.empty((int(np.asarray(current_y, dtype=float).reshape(-1).size), 0), dtype=float)
    else:
        t_out = np.asarray(t_eval, dtype=float)
        Y_out = np.repeat(current_y.reshape(-1, 1), repeats=t_out.size, axis=1)
    distinct_segment_solvers = list(dict.fromkeys(segment_solvers))
    if len(distinct_segment_solvers) == 1:
        prov["solver_used"] = distinct_segment_solvers[0]
    elif distinct_segment_solvers:
        prov["solver_used"] = "mixed"
    else:
        prov["solver_used"] = method
    if segment_solvers:
        prov["intervention_segment_solvers"] = list(segment_solvers)
    distinct_alternatives = list(dict.fromkeys(segment_alternatives))
    if len(distinct_alternatives) == 1:
        prov["solver_alternative_used"] = distinct_alternatives[0]
    elif distinct_alternatives:
        prov["solver_alternative_used"] = list(distinct_alternatives)
    if custom_jacobian_disabled:
        prov["intervention_custom_jacobian_disabled"] = True
    prov["intervention_segments"] = int(segment_count)
    if event_provenance:
        prov["events"] = [list(values) for values in event_provenance]
    return SimulationOutput(
        t=t_out,
        Y=_apply_positivity_to_trajectory(
            Y_out,
            mode=req.positivity,
            indices=(list(req.pos_indices) if req.pos_indices is not None else None),
        ),
        provenance=prov,
        fallback_occurred=fallback_occurred,
        fallback_message=fallback_message,
    )

def _request_from_mapping(payload: Mapping[str, Any], *, allow_unknown_keys: bool) -> SimulationRequest:
    field_names = {f.name for f in fields(SimulationRequest)}
    payload_dict = dict(payload)
    missing = [name for name in ("rhs", "t_span", "y0") if name not in payload_dict]
    if missing:
        raise TypeError(f"Missing required SimulationRequest field(s): {', '.join(missing)}")
    unknown = sorted(set(payload_dict) - field_names)
    if unknown and not allow_unknown_keys:
        raise TypeError(f"Unknown SimulationRequest field(s): {', '.join(unknown)}")
    filtered = {k: v for k, v in payload_dict.items() if k in field_names}
    return SimulationRequest(**filtered)  # type: ignore[arg-type]


def solve_ode(req: SimulationRequest | Mapping[str, Any], *, allow_unknown_keys: bool = False) -> SimulationOutput:
    if not isinstance(req, SimulationRequest):
        if isinstance(req, Mapping):
            req = _request_from_mapping(req, allow_unknown_keys=bool(allow_unknown_keys))
        else:
            raise TypeError(f"solve_ode expects SimulationRequest or Mapping, got {type(req)!r}")
    y0 = np.asarray(req.y0, float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise InitialConditionError("y0 must be a 1D finite array")
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    rhs, rhs_for_jac = _prepare_rhs(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method, note = _scipy_method_for(req.solver)
    schedule = coerce_intervention_schedule(req.intervention_schedule)
    if schedule is not None:
        return _execute_with_intervention_schedule(
            req,
            rhs=rhs,
            rhs_for_jac=rhs_for_jac,
            t0=t0,
            t1=t1,
            y0=y0,
            t_eval=t_eval,
            prov=prov,
            method=method,
            note=note,
            schedule=schedule,
        )
    return _execute_scipy(
        req,
        rhs=rhs,
        rhs_for_jac=rhs_for_jac,
        t0=t0,
        t1=t1,
        y0=y0,
        t_eval=t_eval,
        prov=prov,
        method=method,
        note=note,
    )

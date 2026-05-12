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
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution

from kindred.core.temperature import TemperatureScheduleDictProtocol, TemperatureScheduleProtocol
from kindred.core.intervention_schedule import (
    InterventionInstantEvent,
    InterventionInterval,
    InterventionSchedule,
    InterventionScheduleError,
    InterventionTriggerEvent,
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
from .jacobian import JacobianConfig

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
    jac_sparsity: Optional[Any] = None
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None
    event_terminal: Optional[Iterable[bool]] = None
    positivity: Optional[str] = None
    pos_indices: Optional[Iterable[int]] = None

    progress_callback: Optional[Callable[[float, float, float], None]] = None
    temperature_schedule: TemperatureScheduleProtocol | None = None
    intervention_schedule: InterventionSchedule | Mapping[str, Any] | None = None
    species_names: Optional[Tuple[str, ...]] = None
    symbolic_wegscheider_identity: Optional[Mapping[str, Any]] = None
    symbolic_jacobian_status: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class SimulationOutput:
    t: np.ndarray
    Y: np.ndarray
    provenance: Dict[str, object]
    fallback_occurred: bool = False
    fallback_message: Optional[str] = None


def _normal_event_states(raw_states: object, *, count: int, species_count: int) -> list[list[np.ndarray]]:
    states: list[list[np.ndarray]] = [[] for _ in range(max(0, int(count)))]
    if raw_states is None or count <= 0:
        return states
    if not isinstance(raw_states, Sequence):
        return states
    for idx in range(min(int(count), len(raw_states))):
        raw = raw_states[idx]
        arr = np.asarray(raw, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if int(species_count) == 1:
                arr = arr.reshape(-1, 1)
            elif arr.size == int(species_count):
                arr = arr.reshape(1, int(species_count))
            else:
                continue
        elif arr.ndim != 2:
            try:
                arr = arr.reshape(-1, int(species_count))
            except Exception:
                continue
        states[idx] = [np.asarray(row, dtype=float).reshape(-1) for row in arr]
    return states


def _attach_event_states(
    output: SimulationOutput,
    *,
    raw_states: object,
    count: int,
    species_count: int,
) -> SimulationOutput:
    object.__setattr__(
        output,
        "_kindred_event_states",
        _normal_event_states(raw_states, count=count, species_count=species_count),
    )
    return output


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
    symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
        jacobian_func=req.jacobian_func,
        jac_sparsity=req.jac_sparsity,
        status=req.symbolic_jacobian_status,
    )
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
        "positivity": (req.positivity or None),
        "pos_indices": (list(req.pos_indices) if req.pos_indices is not None else None),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    prov.update(symbolic_jacobian.provenance_fields())
    prov["jacobian_sparsity_hint"] = req.jac_sparsity is not None
    if isinstance(req.symbolic_wegscheider_identity, Mapping) and req.symbolic_wegscheider_identity:
        prov["symbolic_wegscheider_identity"] = dict(req.symbolic_wegscheider_identity)
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


def _scrub_unused_jacobian_provenance(
    prov: Dict[str, object],
    *,
    method: str,
    req: SimulationRequest,
) -> None:
    if str(method) in {"Radau", "BDF"}:
        return
    disabled_symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
        jacobian_func=req.jacobian_func,
        jac_sparsity=req.jac_sparsity,
        status=req.symbolic_jacobian_status,
    ).with_runtime_disabled(
        partially=False,
        code="non-implicit-solver",
        reason=f"Symbolic Jacobian disabled because solver {method} does not consume Jacobian callables.",
    )
    prov.pop("symbolic_jacobian_identity", None)
    prov.update(disabled_symbolic_jacobian.provenance_fields())


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

    jac_callable = req.jacobian_func
    jac_sparsity = req.jac_sparsity
    banded_jacobian_active = (
        jac_callable is not None
        and
        getattr(req.rosenbrock_jacobian, "mode", None) == "banded"
        and req.rosenbrock_jacobian.ml is not None
        and req.rosenbrock_jacobian.mu is not None
    )

    def _kwargs_for_method(method_name: str) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["method"] = method_name
        if method_name in ("Radau", "BDF") and jac_callable is not None:
            kwargs["jac"] = jac_callable
            if banded_jacobian_active:
                # SciPy implicit solvers expect an n x n Jacobian when a jac
                # callable is supplied; jac_sparsity is ignored in that case.
                kwargs.pop("jac_sparsity", None)
        elif method_name in ("Radau", "BDF") and jac_sparsity is not None:
            kwargs["jac_sparsity"] = jac_sparsity
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
    output = SimulationOutput(
        t=t_out,
        Y=Y_out,
        provenance=prov,
        fallback_occurred=fallback_occurred,
        fallback_message=fallback_message,
    )
    return _attach_event_states(
        output,
        raw_states=getattr(sol, "y_events", None),
        count=len(events_list or ()),
        species_count=int(np.asarray(y0, dtype=float).reshape(-1).size),
    )


def _request_for_internal_segment(req: SimulationRequest, *, t0: float, t1: float) -> SimulationRequest:
    if req.first_step is None:
        return req
    segment_span = abs(float(t1) - float(t0))
    if float(req.first_step) <= segment_span:
        return req
    return replace(req, first_step=None)


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


def _apply_fixed_interval_values(
    y: np.ndarray,
    intervals: Iterable[InterventionInterval],
    *,
    species_index: Mapping[str, int],
) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for interval in intervals:
        if interval.kind not in {"reservoir", "clamp"}:
            continue
        fixed_idx = species_index.get(interval.species)
        if fixed_idx is not None:
            out[fixed_idx] = float(interval.value if interval.value is not None else 0.0)
    return out


def _trigger_direction_value(direction: str) -> float:
    if direction == "rising":
        return 1.0
    if direction == "falling":
        return -1.0
    return 0.0


def _trigger_available(
    trigger: InterventionTriggerEvent,
    *,
    trigger_index: int,
    segment_start: float,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
) -> bool:
    if int(trigger_counts[trigger_index]) >= int(trigger.max_count):
        return False
    ready_time = _trigger_next_eligible_time(
        trigger,
        trigger_index=trigger_index,
        trigger_counts=trigger_counts,
        trigger_last_times=trigger_last_times,
    )
    if ready_time is None:
        return True
    return float(segment_start) >= float(ready_time)


def _trigger_next_eligible_time(
    trigger: InterventionTriggerEvent,
    *,
    trigger_index: int,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
) -> float | None:
    if int(trigger_counts[trigger_index]) >= int(trigger.max_count):
        return None
    last_time = trigger_last_times[trigger_index]
    if last_time is None:
        return None
    min_ready = float(last_time) + max(0.0, float(trigger.min_interval))
    just_after_last = float(np.nextafter(float(last_time), np.inf))
    return max(min_ready, just_after_last)


def _next_trigger_rearm_boundary(
    triggers: Sequence[InterventionTriggerEvent],
    *,
    segment_start: float,
    segment_end: float,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
) -> float | None:
    candidates: list[float] = []
    start_value = float(segment_start)
    end_value = float(segment_end)
    for trigger_index, trigger in enumerate(triggers):
        ready_time = _trigger_next_eligible_time(
            trigger,
            trigger_index=trigger_index,
            trigger_counts=trigger_counts,
            trigger_last_times=trigger_last_times,
        )
        if ready_time is None:
            continue
        if start_value < float(ready_time) < end_value:
            candidates.append(float(ready_time))
    if not candidates:
        return None
    return min(candidates)


def _trigger_callables_for_segment(
    triggers: Sequence[InterventionTriggerEvent],
    *,
    species_index: Mapping[str, int],
    segment_start: float,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
) -> tuple[tuple[Callable[[float, np.ndarray], float], ...], tuple[int, ...]]:
    callables: list[Callable[[float, np.ndarray], float]] = []
    mapping: list[int] = []
    for trigger_index, trigger in enumerate(triggers):
        if not _trigger_available(
            trigger,
            trigger_index=trigger_index,
            segment_start=segment_start,
            trigger_counts=trigger_counts,
            trigger_last_times=trigger_last_times,
        ):
            continue
        species_idx = species_index.get(trigger.trigger_species)
        if species_idx is None:
            raise InterventionScheduleError(f"Unknown intervention trigger species: {trigger.trigger_species}")
        threshold = float(trigger.threshold if trigger.threshold is not None else 0.0)

        def _trigger_event(_t: float, y: np.ndarray, *, _idx=species_idx, _threshold=threshold) -> float:
            return float(np.asarray(y, dtype=float).reshape(-1)[_idx] - _threshold)

        _trigger_event.terminal = True  # type: ignore[attr-defined]
        _trigger_event.direction = _trigger_direction_value(trigger.direction)  # type: ignore[attr-defined]
        _trigger_event._kindred_intervention_trigger_index = int(trigger_index)  # type: ignore[attr-defined]
        callables.append(_trigger_event)
        mapping.append(int(trigger_index))
    return tuple(callables), tuple(mapping)


def _trigger_hits_from_event_provenance(
    event_provenance: Sequence[Sequence[float]],
    *,
    trigger_mapping: Sequence[int],
    user_event_count: int,
) -> tuple[float, tuple[int, ...]] | None:
    hits: list[tuple[float, int]] = []
    for offset, trigger_index in enumerate(trigger_mapping):
        event_idx = int(user_event_count) + int(offset)
        if event_idx >= len(event_provenance):
            continue
        for raw_time in event_provenance[event_idx]:
            hits.append((float(raw_time), int(trigger_index)))
    if not hits:
        return None
    trigger_time = min(time for time, _trigger_index in hits)
    lower = float(np.nextafter(trigger_time, -np.inf))
    upper = float(np.nextafter(trigger_time, np.inf))
    trigger_indices = tuple(
        dict.fromkeys(trigger_index for time, trigger_index in hits if lower <= float(time) <= upper)
    )
    return float(trigger_time), trigger_indices


def _trigger_state_from_event_states(
    event_provenance: Sequence[Sequence[float]],
    event_states: Sequence[Sequence[np.ndarray]],
    *,
    trigger_mapping: Sequence[int],
    user_event_count: int,
    trigger_time: float,
    trigger_indices: Sequence[int],
) -> np.ndarray | None:
    trigger_set = {int(trigger_index) for trigger_index in trigger_indices}
    lower = float(np.nextafter(float(trigger_time), -np.inf))
    upper = float(np.nextafter(float(trigger_time), np.inf))
    for offset, trigger_index in enumerate(trigger_mapping):
        if int(trigger_index) not in trigger_set:
            continue
        event_idx = int(user_event_count) + int(offset)
        if event_idx >= len(event_provenance) or event_idx >= len(event_states):
            continue
        states = event_states[event_idx]
        for state_idx, raw_time in enumerate(event_provenance[event_idx]):
            if not (lower <= float(raw_time) <= upper):
                continue
            if state_idx >= len(states):
                continue
            return np.asarray(states[state_idx], dtype=float).reshape(-1)
    return None


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


def _append_segment_output_samples(
    outputs_t: list[np.ndarray],
    outputs_y: list[np.ndarray],
    *,
    seg_t: np.ndarray,
    seg_Y: np.ndarray,
    count: int,
) -> None:
    actual_count = min(int(count), int(seg_t.size))
    if actual_count <= 0:
        return
    emit_t = np.asarray(seg_t[:actual_count], dtype=float).reshape(-1)
    emit_Y = np.asarray(seg_Y[:, :actual_count], dtype=float)
    if outputs_t and outputs_t[-1].size and emit_t.size:
        previous_t = float(outputs_t[-1][-1])
        first_t = float(emit_t[0])
        if float(np.nextafter(previous_t, -np.inf)) <= first_t <= float(np.nextafter(previous_t, np.inf)):
            emit_t = emit_t[1:]
            emit_Y = emit_Y[:, 1:]
    if emit_t.size:
        outputs_t.append(emit_t)
        outputs_y.append(emit_Y)


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
    segment_symbolic_jacobians: list[bool] = []
    segment_sparsity_hints: list[bool] = []
    symbolic_jacobian_disabled = False
    symbolic_jacobian_used = False
    terminal_stop = False
    trigger_events = tuple(schedule.trigger_events)
    trigger_counts = [0 for _ in trigger_events]
    trigger_last_times: list[float | None] = [None for _ in trigger_events]
    trigger_provenance: list[dict[str, object]] = []

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
        sub_start = seg_start
        while sub_start < seg_end:
            sub_end = seg_end
            rearm_boundary = _next_trigger_rearm_boundary(
                trigger_events,
                segment_start=sub_start,
                segment_end=seg_end,
                trigger_counts=trigger_counts,
                trigger_last_times=trigger_last_times,
            )
            if rearm_boundary is not None:
                sub_end = float(rearm_boundary)
            is_sub_final = bool(is_final and float(sub_end) == float(seg_end))
            requested_mask = _segment_eval_mask(
                t_eval,
                seg_start=sub_start,
                seg_end=sub_end,
                is_final=is_sub_final,
            )
            requested_eval = _snap_eval_times_to_segment_boundaries(
                t_eval[requested_mask],
                seg_start=sub_start,
                seg_end=sub_end,
            )
            requested_eval = _deduplicate_snapped_eval_times(requested_eval)
            if requested_eval.size and float(requested_eval[-1]) == float(sub_end):
                internal_eval = requested_eval
                requested_count = int(requested_eval.size)
            elif requested_eval.size:
                internal_eval = np.concatenate([requested_eval, np.array([sub_end], dtype=float)])
                requested_count = int(requested_eval.size)
            else:
                internal_eval = np.array([sub_end], dtype=float)
                requested_count = 0
            active_intervals = intervals_active_at(schedule, sub_start)
            if active_intervals:
                current_y = _apply_fixed_interval_values(
                    current_y,
                    active_intervals,
                    species_index=species_index,
                )
            seg_rhs = rhs
            seg_rhs_for_jac = rhs_for_jac
            if active_intervals:
                seg_rhs = _InterventionRhs(rhs, intervals=active_intervals, species_index=species_index)
                seg_rhs_for_jac = _InterventionRhs(rhs_for_jac, intervals=active_intervals, species_index=species_index)
            trigger_callables, trigger_mapping = _trigger_callables_for_segment(
                trigger_events,
                species_index=species_index,
                segment_start=sub_start,
                trigger_counts=trigger_counts,
                trigger_last_times=trigger_last_times,
            )
            segment_events = events_tuple + trigger_callables
            seg_req = segment_req_base
            segment_symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
                jacobian_func=segment_req_base.jacobian_func,
                jac_sparsity=segment_req_base.jac_sparsity,
                status=segment_req_base.symbolic_jacobian_status,
            )
            if active_intervals and segment_req_base.jacobian_func is not None:
                if segment_symbolic_jacobian.has_executable_jacobian:
                    symbolic_jacobian_disabled = True
                seg_req = replace(seg_req, jacobian_func=None, jac_sparsity=None)
            elif active_intervals and segment_req_base.jac_sparsity is not None:
                seg_req = replace(seg_req, jac_sparsity=None)
            if trigger_callables:
                segment_terminal_flags = tuple(
                    _event_terminal_flags(segment_req_base, events_tuple)
                ) + tuple(True for _trigger in trigger_callables)
                seg_req = replace(
                    seg_req,
                    events=segment_events,
                    event_terminal=segment_terminal_flags,
                )
            seg_req = _request_for_internal_segment(seg_req, t0=sub_start, t1=sub_end)
            seg_prov: Dict[str, object] = dict(prov)
            seg_symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
                jacobian_func=seg_req.jacobian_func,
                jac_sparsity=seg_req.jac_sparsity,
                status=seg_req.symbolic_jacobian_status,
            )
            seg_prov.pop("symbolic_jacobian_identity", None)
            seg_prov.update(seg_symbolic_jacobian.provenance_fields())
            seg_prov["jacobian_sparsity_hint"] = seg_req.jac_sparsity is not None
            seg_prov["intervention_segment_index"] = int(segment_count)
            seg_out = _execute_scipy(
                seg_req,
                rhs=seg_rhs,
                rhs_for_jac=seg_rhs_for_jac,
                t0=sub_start,
                t1=sub_end,
                y0=current_y,
                t_eval=internal_eval,
                prov=seg_prov,
                method=method,
                note=note if segment_count == 0 else None,
            )
            segment_count += 1
            fallback_occurred = fallback_occurred or bool(seg_out.fallback_occurred)
            fallback_message = fallback_message or seg_out.fallback_message
            symbolic_jacobian_used = symbolic_jacobian_used or bool(
                seg_out.provenance.get("symbolic_jacobian")
            )
            segment_symbolic_jacobians.append(bool(seg_out.provenance.get("symbolic_jacobian")))
            segment_sparsity_hints.append(bool(seg_out.provenance.get("jacobian_sparsity_hint")))
            segment_solver = str(seg_out.provenance.get("solver_used") or method)
            segment_solvers.append(segment_solver)
            alternative = seg_out.provenance.get("solver_alternative_used")
            if alternative:
                segment_alternatives.append(str(alternative))
            seg_events_all = _normal_event_provenance(seg_out.provenance.get("events"), count=len(segment_events))
            seg_event_states_all = getattr(seg_out, "_kindred_event_states", None)
            if seg_event_states_all is None:
                seg_event_states_all = _normal_event_states(
                    None,
                    count=len(segment_events),
                    species_count=int(np.asarray(current_y, dtype=float).reshape(-1).size),
                )
            seg_events = seg_events_all[: len(events_tuple)]
            if event_provenance:
                for event_idx, values in enumerate(seg_events):
                    event_provenance[event_idx].extend(values)
            terminal_stop = _non_cancel_terminal_event_hit(
                events=events_tuple,
                terminal_flags=terminal_flags,
                event_provenance=seg_events,
            )
            trigger_hit = None if terminal_stop else _trigger_hits_from_event_provenance(
                seg_events_all,
                trigger_mapping=trigger_mapping,
                user_event_count=len(events_tuple),
            )
            seg_t = np.asarray(seg_out.t, dtype=float).reshape(-1)
            seg_Y = np.asarray(seg_out.Y, dtype=float)
            if seg_Y.ndim == 1:
                seg_Y = _solver_trajectory_array(seg_Y, y0=current_y, t_out=seg_t)
            if seg_Y.shape[1]:
                current_y = np.asarray(seg_Y[:, -1], dtype=float).reshape(-1)
                current_y = _apply_fixed_interval_values(
                    current_y,
                    active_intervals,
                    species_index=species_index,
                )
            elif not terminal_stop and trigger_hit is None:
                raise create_solver_error(
                    req.solver,
                    sub_start,
                    "Scheduled simulation segment produced no state samples without a terminal event.",
                )
            if requested_count:
                _append_segment_output_samples(
                    outputs_t,
                    outputs_y,
                    seg_t=seg_t,
                    seg_Y=seg_Y,
                    count=int(requested_count),
                )
            if terminal_stop and seg_t.size:
                terminal_time = float(seg_t[-1])
                if _eval_times_include_boundary(t_eval, terminal_time):
                    already_emitted = bool(outputs_t and outputs_t[-1].size and float(outputs_t[-1][-1]) == terminal_time)
                    if not already_emitted:
                        outputs_t.append(seg_t[-1:])
                        outputs_y.append(seg_Y[:, -1:])
            if terminal_stop:
                break
            if trigger_hit is None:
                if rearm_boundary is not None and float(sub_end) < float(seg_end):
                    sub_start = float(sub_end)
                    continue
                break
            trigger_time, trigger_indices = trigger_hit
            trigger_state = _trigger_state_from_event_states(
                seg_events_all,
                seg_event_states_all,
                trigger_mapping=trigger_mapping,
                user_event_count=len(events_tuple),
                trigger_time=trigger_time,
                trigger_indices=trigger_indices,
            )
            if trigger_state is not None:
                current_y = trigger_state
            else:
                raise create_solver_error(
                    req.solver,
                    trigger_time,
                    "State-trigger event did not provide an event-time state for scheduled intervention resume.",
                )
            trigger_instants = [trigger_events[trigger_index].to_instant_event(time=trigger_time) for trigger_index in trigger_indices]
            current_y = _apply_instant_events(current_y, trigger_instants, species_index=species_index)
            current_y = _apply_fixed_interval_values(
                current_y,
                intervals_active_at(schedule, float(trigger_time)),
                species_index=species_index,
            )
            if outputs_t and outputs_t[-1].size and _eval_times_include_boundary(outputs_t[-1], trigger_time):
                outputs_y[-1][:, -1] = current_y
            for trigger_index in trigger_indices:
                trigger_counts[trigger_index] += 1
                trigger_last_times[trigger_index] = float(trigger_time)
                trigger = trigger_events[trigger_index]
                trigger_provenance.append(
                    {
                        "time": float(trigger_time),
                        "trigger_species": str(trigger.trigger_species),
                        "species": str(trigger.species),
                        "action": str(trigger.op),
                    }
                )
            sub_start = float(trigger_time)
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
    if segment_sparsity_hints and any(segment_sparsity_hints) != all(segment_sparsity_hints):
        prov["intervention_segment_jacobian_sparsity_hints"] = list(segment_sparsity_hints)
        prov["intervention_jacobian_sparsity_hint_partially_disabled"] = True
    elif segment_sparsity_hints:
        prov["jacobian_sparsity_hint"] = bool(segment_sparsity_hints[-1])
    distinct_alternatives = list(dict.fromkeys(segment_alternatives))
    if len(distinct_alternatives) == 1:
        prov["solver_alternative_used"] = distinct_alternatives[0]
    elif distinct_alternatives:
        prov["solver_alternative_used"] = list(distinct_alternatives)
    if symbolic_jacobian_disabled:
        prov["intervention_symbolic_jacobian_disabled"] = True
        if segment_symbolic_jacobians:
            prov["intervention_segment_symbolic_jacobians"] = list(segment_symbolic_jacobians)
        runtime_symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
            jacobian_func=req.jacobian_func,
            jac_sparsity=req.jac_sparsity,
            status=req.symbolic_jacobian_status,
        ).with_runtime_disabled(
            partially=bool(symbolic_jacobian_used),
            code="active-intervention-interval",
            reason="Symbolic Jacobian disabled for active intervention interval segments.",
        )
        if symbolic_jacobian_used:
            prov["intervention_symbolic_jacobian_partially_disabled"] = True
        if not symbolic_jacobian_used:
            prov.pop("symbolic_jacobian_identity", None)
        prov.update(runtime_symbolic_jacobian.provenance_fields())
    prov["intervention_segments"] = int(segment_count)
    if trigger_provenance:
        prov["intervention_trigger_events"] = list(trigger_provenance)
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
    _scrub_unused_jacobian_provenance(prov, method=method, req=req)
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

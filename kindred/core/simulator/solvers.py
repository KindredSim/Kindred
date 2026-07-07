# SPDX-License-Identifier: MIT
"""
ODE solver wrappers for Kindred.

SciPy is a hard dependency; all integration routes through `scipy.integrate.solve_ivp` with:
- Deterministic `t_eval` output grid
- Optional TemperatureSchedule wrapper that injects T into RHS(T=...) calls
- Optional progress callback wrapper
- Non-negative concentration invariant enforcement at output boundaries
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
from kindred.core.scipy_integrate import load_scipy_integrate
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution

from kindred.core.temperature import (
    TemperatureScheduleDictProtocol,
    TemperatureScheduleProtocol,
    coerce_temperature_schedule,
)
from kindred.core.intervention_schedule import coerce_intervention_schedule
from kindred.core.intervention_schedule_compiler import compile_intervention_schedule
from kindred.core.exceptions import (
    InitialConditionError,
    SimulationCancelled,
    SolverError,
    TimeGridError,
    create_solver_error,
)
from kindred.core.time_grid import build_time_grid
from .nonnegative_invariant import project_nonnegative_trajectory
from .solver_output import solver_trajectory_array
from .intervention_schedule_execution import (
    InterventionScheduleExecutionOwner,
    ScheduleExecutionRequest,
    SegmentExecutionRequest,
    SegmentRunResult,
)
from .solver_types import (
    DEFAULT_SOLVER_NAME,
    ODERhsNoTemp,
    ODERhsWithTemp,
    Rhs2,
    SimulationOutput,
    SimulationRequest,
)

logger = logging.getLogger(__name__)

def _solve_ivp(*, fun: Callable[[float, np.ndarray], np.ndarray], t_span: Tuple[float, float], y0: np.ndarray, **kwargs: Any):
    solve_ivp = load_scipy_integrate()
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


__all__ = [
    "SimulationRequest",
    "SimulationOutput",
    "DEFAULT_SOLVER_NAME",
    "build_time_grid",
    "normalize_solver_name",
    "solve_ode",
]


def _normal_event_states_with_projection(
    raw_states: object,
    *,
    count: int,
    species_count: int,
) -> tuple[list[list[np.ndarray]], dict[str, object]]:
    states: list[list[np.ndarray]] = [[] for _ in range(max(0, int(count)))]
    projection_count = 0
    max_negative_magnitude = 0.0
    if raw_states is None or count <= 0:
        return states, {"projected": False, "count": 0, "max_negative_magnitude": 0.0}
    if not isinstance(raw_states, Sequence):
        return states, {"projected": False, "count": 0, "max_negative_magnitude": 0.0}
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
        projected_rows: list[np.ndarray] = []
        for row in arr:
            projected, provenance = project_nonnegative_trajectory(
                np.asarray(row, dtype=float).reshape(-1, 1),
                context="Simulation event state",
            )
            projected_rows.append(projected.reshape(-1))
            projection_count += int(provenance.get("count", 0) or 0)
            max_negative_magnitude = max(
                max_negative_magnitude,
                float(provenance.get("max_negative_magnitude", 0.0) or 0.0),
            )
        states[idx] = projected_rows
    return states, {
        "projected": bool(projection_count),
        "count": int(projection_count),
        "max_negative_magnitude": float(max_negative_magnitude),
    }


def _normal_event_states(raw_states: object, *, count: int, species_count: int) -> list[list[np.ndarray]]:
    states, _provenance = _normal_event_states_with_projection(
        raw_states,
        count=count,
        species_count=species_count,
    )
    return states


def _attach_event_states(
    output: SimulationOutput,
    *,
    raw_states: object,
    count: int,
    species_count: int,
) -> SimulationOutput:
    event_states, event_projection = _normal_event_states_with_projection(
        raw_states,
        count=count,
        species_count=species_count,
    )
    if bool(event_projection.get("projected")):
        provenance = dict(output.provenance)
        provenance["event_state_nonnegative_projection"] = event_projection
        output = SimulationOutput(
            t=output.t,
            Y=output.Y,
            provenance=provenance,
        )
    object.__setattr__(
        output,
        "_kindred_event_states",
        event_states,
    )
    return output


def _scipy_method_for(name: object) -> str:
    """Map solver name to SciPy method name with correct capitalization."""
    n = str(name or "").strip().upper()
    scipy_methods = {
        "RADAU": "Radau",
        "BDF": "BDF",
    }
    if n in scipy_methods:
        return scipy_methods[n]
    allowed = ", ".join(sorted(scipy_methods.values()))
    raise SolverError(f"Unsupported solver name {name!r}; expected one of: {allowed}.")


def normalize_solver_name(name: object) -> str:
    """
    Normalize a user-specified solver name to a SciPy `solve_ivp` method name.

    Unknown names raise `SolverError`.
    """
    return _scipy_method_for(name)


class _TemperatureInjectedRhs:
    def __init__(self, rhs: ODERhsWithTemp, schedule: TemperatureScheduleProtocol) -> None:
        self._rhs = rhs
        schedule_obj = coerce_temperature_schedule(schedule)
        if schedule_obj is None:
            raise TypeError("temperature_schedule must not be None when injecting temperature.")
        self._schedule = schedule_obj

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


def _prepare_time_grid(req: SimulationRequest, *, t0: float, t1: float) -> np.ndarray:
    if req.t_eval is None:
        if not isinstance(req.grid, Mapping) or not req.grid:
            raise TimeGridError("SimulationRequest.grid is required when t_eval is not provided")
        try:
            return build_time_grid(t0, t1, req.grid)
        except ValueError as exc:
            raise TimeGridError(str(exc)) from exc
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
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    prov.update(symbolic_jacobian.provenance_fields())
    prov["jacobian_sparsity_hint"] = req.jac_sparsity is not None
    if isinstance(req.symbolic_wegscheider_identity, Mapping) and req.symbolic_wegscheider_identity:
        prov["symbolic_wegscheider_identity"] = dict(req.symbolic_wegscheider_identity)
    schedule = coerce_intervention_schedule(req.intervention_schedule)
    prov["has_intervention_schedule"] = schedule is not None
    if schedule is not None:
        compiled_schedule = compile_intervention_schedule(schedule)
        prov["intervention_schedule_declarative"] = compiled_schedule.normalized_declarative_payload
        prov["intervention_schedule_declarative_fingerprint"] = compiled_schedule.declarative_fingerprint
        prov["intervention_schedule_executable"] = compiled_schedule.executable_payload
        prov["intervention_schedule_executable_fingerprint"] = compiled_schedule.executable_fingerprint
        if compiled_schedule.lineage:
            prov["intervention_schedule_lineage"] = [dict(item) for item in compiled_schedule.lineage]
        metadata = compiled_schedule.provenance.get("metadata")
        if isinstance(metadata, Mapping):
            prov["intervention_schedule_metadata"] = dict(metadata)
        primitive_metadata = compiled_schedule.provenance.get("primitive_metadata")
        if isinstance(primitive_metadata, Sequence) and not isinstance(primitive_metadata, (str, bytes)):
            entries = [dict(item) for item in primitive_metadata if isinstance(item, Mapping)]
            if entries:
                prov["intervention_schedule_primitive_metadata"] = entries
        prov["intervention_schedule_metadata_uses_internal_numeric_values"] = bool(
            compiled_schedule.provenance.get("metadata_uses_internal_numeric_values")
        )

    if req.temperature_schedule is not None:
        temperature_schedule = coerce_temperature_schedule(req.temperature_schedule)
        if isinstance(temperature_schedule, TemperatureScheduleDictProtocol):
            prov["temperature_schedule"] = dict(temperature_schedule.to_dict())
        else:
            prov["temperature_schedule"] = str(temperature_schedule)

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
) -> SimulationOutput:
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

    if events_list and getattr(sol, "status", 0) == 1 and getattr(sol, "t_events", None):
        for idx, ev in enumerate(events_list):
            if not bool(getattr(ev, "_kindred_cancel_event", False)):
                continue
            t_events = sol.t_events[idx] if idx < len(sol.t_events) else None
            if t_events is not None and len(t_events) > 0:
                raise SimulationCancelled()

    if not sol.success:
        prov["solver_failure"] = str(sol.message)
        t_fail = float(sol.t[-1]) if getattr(sol, "t", None) is not None and len(sol.t) > 0 else float(t0)
        raise create_solver_error(req.solver, t_fail, str(sol.message))

    prov["solver_used"] = method
    t_out = np.asarray(sol.t, float)
    Y_out, nonnegative_projection = project_nonnegative_trajectory(
        solver_trajectory_array(sol.y, y0=y0, t_out=t_out),
        context="Simulation",
    )
    if bool(nonnegative_projection.get("projected")):
        prov["nonnegative_projection"] = nonnegative_projection
    if hasattr(sol, "t_events") and sol.t_events:
        prov["events"] = [list(te) for te in sol.t_events]
    output = SimulationOutput(
        t=t_out,
        Y=Y_out,
        provenance=prov,
    )
    return _attach_event_states(
        output,
        raw_states=getattr(sol, "y_events", None),
        count=len(events_list or ()),
        species_count=int(np.asarray(y0, dtype=float).reshape(-1).size),
    )


def _run_scipy_segment(request: SegmentExecutionRequest) -> SegmentRunResult:
    output = _execute_scipy(
        request.request,
        rhs=request.rhs,
        rhs_for_jac=request.rhs_for_jac,
        t0=request.t0,
        t1=request.t1,
        y0=request.y0,
        t_eval=request.t_eval,
        prov=request.provenance,
        method=request.method,
    )
    events = output.provenance.get("events")
    return SegmentRunResult(
        output=output,
        event_times=_normal_event_provenance(events, count=len(_event_callables_for_request(request.request) or ())),
        event_states=_normal_event_states(
            getattr(output, "_kindred_event_states", None),
            count=len(_event_callables_for_request(request.request) or ()),
            species_count=int(np.asarray(request.y0, dtype=float).reshape(-1).size),
        ),
        solver_used=str(output.provenance.get("solver_used") or request.method),
        symbolic_jacobian_used=bool(output.provenance.get("symbolic_jacobian")),
        jacobian_sparsity_hint=bool(output.provenance.get("jacobian_sparsity_hint")),
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

def solve_ode(req: SimulationRequest) -> SimulationOutput:
    if not isinstance(req, SimulationRequest):
        raise TypeError(f"solve_ode expects SimulationRequest, got {type(req)!r}")
    if req.temperature_schedule is not None:
        req = replace(req, temperature_schedule=coerce_temperature_schedule(req.temperature_schedule))
    y0 = np.asarray(req.y0, float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise InitialConditionError("y0 must be a 1D finite array")
    if np.any(y0 < 0.0):
        raise InitialConditionError("y0 must contain non-negative concentrations")
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    rhs, rhs_for_jac = _prepare_rhs(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method = _scipy_method_for(req.solver)
    _scrub_unused_jacobian_provenance(prov, method=method, req=req)
    schedule = coerce_intervention_schedule(req.intervention_schedule)
    if schedule is not None:
        compiled_schedule = compile_intervention_schedule(schedule)
        if not compiled_schedule.executable_schedule.is_empty():
            return InterventionScheduleExecutionOwner(_run_scipy_segment).execute(
                ScheduleExecutionRequest(
                    request=req,
                    rhs=rhs,
                    rhs_for_jac=rhs_for_jac,
                    t0=t0,
                    t1=t1,
                    y0=y0,
                    t_eval=t_eval,
                    provenance=prov,
                    method=method,
                    schedule=compiled_schedule.executable_schedule,
                )
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
    )

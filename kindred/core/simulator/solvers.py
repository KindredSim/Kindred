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
    Mapping,
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
    temperature_schedule_execution_identity,
)
from kindred.core.exceptions import (
    InitialConditionError,
    SimulationCancelled,
    SolverError,
    TimeGridError,
    create_solver_error,
)
from kindred.core.time_grid import build_time_grid as _build_time_grid
from .nonnegative_invariant import (
    NonnegativeProjection,
    project_nonnegative_trajectory,
)
from .solver_output import solver_trajectory_array
from .intervention_schedule_execution import (
    InterventionScheduleExecutionOwner,
    ScheduleExecutionRequest,
    SegmentExecutionRequest,
    SegmentIdentity,
    SegmentRunResult,
)
from .solver_types import (
    ODERhsNoTemp as _ODERhsNoTemp,
    ODERhsWithTemp as _ODERhsWithTemp,
    Rhs2 as _Rhs2,
    SimulationCancellationControl as _SimulationCancellationControl,
    SimulationOutput as _SimulationOutput,
    SimulationRequest as _SimulationRequest,
)

logger = logging.getLogger(__name__)


class _NonfiniteSymbolicJacobianError(RuntimeError):
    """Internal signal that permits one same-segment no-Jacobian retry."""

    def __init__(self, *, time: float) -> None:
        self.time = float(time)
        super().__init__("Symbolic Jacobian evaluation was non-finite or invalid.")


def _solve_ivp(*, fun: Callable[[float, np.ndarray], np.ndarray], t_span: Tuple[float, float], y0: np.ndarray, **kwargs: Any):
    solve_ivp = load_scipy_integrate()
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


__all__ = [
    "normalize_solver_name",
    "solve_ode",
]


def _normal_event_states_with_projection(
    raw_states: object,
    *,
    count: int,
    species_count: int,
) -> tuple[
    list[list[np.ndarray]],
    NonnegativeProjection,
]:
    raw_output = _normal_event_states(
        raw_states,
        count=count,
        species_count=species_count,
    )
    projections: list[NonnegativeProjection] = []
    for exact_rows in raw_output:
        for exact in exact_rows:
            _projected, projection = project_nonnegative_trajectory(
                np.asarray(exact, dtype=float).reshape(-1, 1).copy(),
                context="Simulation event state",
            )
            projections.append(projection)
    return raw_output, NonnegativeProjection.aggregate(projections)


def _normal_event_states(raw_states: object, *, count: int, species_count: int) -> list[list[np.ndarray]]:
    output: list[list[np.ndarray]] = [
        [] for _ in range(max(0, int(count)))
    ]
    if raw_states is None or count <= 0 or not isinstance(raw_states, Sequence):
        return output
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
            except ValueError:
                continue
        output[idx] = [
            np.asarray(row, dtype=float).reshape(-1).copy()
            for row in arr
        ]
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
    def __init__(self, rhs: _ODERhsWithTemp, schedule: TemperatureScheduleProtocol) -> None:
        self._rhs = rhs
        schedule_obj = (
            schedule
            if isinstance(schedule, TemperatureScheduleDictProtocol)
            else coerce_temperature_schedule(schedule)
        )
        if schedule_obj is None:
            raise TypeError("temperature_schedule must not be None when injecting temperature.")
        self._schedule = schedule_obj

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        T = float(self._schedule(t))
        return self._rhs(t, y, T=T)


class _ProgressRhs:
    def __init__(
        self,
        rhs: _Rhs2,
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


class _CancellationRhs:
    """Poll request cancellation immediately before scientific RHS work."""

    def __init__(
        self,
        rhs: _Rhs2,
        control: _SimulationCancellationControl,
    ) -> None:
        self._rhs = rhs
        self._control = control

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        if self._control.cancel_requested():
            raise SimulationCancelled()
        return self._rhs(t, y)


def _prepare_time_grid(req: _SimulationRequest, *, t0: float, t1: float) -> np.ndarray:
    if req.t_eval is None:
        if not isinstance(req.grid, Mapping) or not req.grid:
            raise TimeGridError("SimulationRequest.grid is required when t_eval is not provided")
        try:
            return _build_time_grid(t0, t1, req.grid)
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


def _prepare_rhs(req: _SimulationRequest, *, t0: float, t1: float) -> _Rhs2:
    if req.temperature_schedule is None:
        rhs: _Rhs2 = cast(_ODERhsNoTemp, req.rhs)
    else:
        rhs = _TemperatureInjectedRhs(
            cast(_ODERhsWithTemp, req.rhs), req.temperature_schedule
        )

    if req.progress_callback is not None:
        rhs = _ProgressRhs(rhs, callback=req.progress_callback, t0=t0, t1=t1)
    if req.cancellation_control is not None:
        rhs = _CancellationRhs(rhs, req.cancellation_control)
    return rhs


def _build_provenance(req: _SimulationRequest, *, t_eval: np.ndarray) -> Dict[str, object]:
    symbolic_jacobian = req.symbolic_jacobian
    prov: Dict[str, object] = {
        "solver_requested": req.solver,
        "rtol": float(req.rtol),
        "atol": float(req.atol),
        "max_step": (None if req.max_step is None else float(req.max_step)),
        "first_step": (None if req.first_step is None else float(req.first_step)),
        "grid_type": ("explicit" if req.t_eval is not None else ("dt" if "dt" in (req.grid or {}) else "N")),
        "grid_params": ({"t_eval": list(t_eval)} if req.t_eval is not None else dict(req.grid or {})),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    prov.update(symbolic_jacobian.provenance_fields())
    prov["jacobian_sparsity_hint"] = symbolic_jacobian.jac_sparsity is not None
    if isinstance(req.symbolic_wegscheider_identity, Mapping) and req.symbolic_wegscheider_identity:
        prov["symbolic_wegscheider_identity"] = dict(req.symbolic_wegscheider_identity)
    intervention_execution = req.intervention_execution
    prov["has_intervention_schedule"] = not intervention_execution.is_empty
    if not intervention_execution.is_empty:
        prov["intervention_execution"] = (
            intervention_execution.provenance_projection()
        )

    if req.temperature_schedule is not None:
        temperature_schedule = coerce_temperature_schedule(req.temperature_schedule)
        schedule_identity = temperature_schedule_execution_identity(
            temperature_schedule
        )
        prov["temperature_schedule"] = (
            dict(schedule_identity)
            if schedule_identity is not None
            else str(temperature_schedule)
        )

    return prov


def _scrub_unused_jacobian_provenance(
    prov: Dict[str, object],
    *,
    method: str,
) -> None:
    if str(method) in {"Radau", "BDF"}:
        return
    disabled_symbolic_jacobian = SymbolicJacobianExecution.disabled(
        code="non-implicit-solver",
        reason=f"Symbolic Jacobian disabled because solver {method} does not consume Jacobian callables.",
    )
    prov.pop("symbolic_jacobian_identity", None)
    prov.update(disabled_symbolic_jacobian.provenance_fields())


def _guard_runtime_symbolic_jacobian(
    jacobian_func: Callable[[float, np.ndarray], np.ndarray],
    cancellation_control: _SimulationCancellationControl | None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Reject a non-finite symbolic Jacobian without reclassifying RHS errors."""

    def guarded(time: float, state: np.ndarray) -> np.ndarray:
        if (
            cancellation_control is not None
            and cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        values = np.asarray(state, dtype=float).reshape(-1)
        try:
            matrix = np.asarray(
                jacobian_func(float(time), values),
                dtype=float,
            )
        except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
            raise _NonfiniteSymbolicJacobianError(time=float(time)) from exc
        if (
            matrix.shape != (values.size, values.size)
            or not np.all(np.isfinite(matrix))
        ):
            raise _NonfiniteSymbolicJacobianError(time=float(time))
        return matrix

    return guarded


def _execute_scipy(
    req: _SimulationRequest,
    *,
    segment_identity: SegmentIdentity,
    rhs: _Rhs2,
    t0: float,
    t1: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    prov: Dict[str, object],
    method: str,
    boundary_events: Sequence[Callable[[float, np.ndarray], float]],
    boundary_terminal_flags: Sequence[bool],
) -> SegmentRunResult:
    base_kwargs: Dict[str, Any] = dict(t_eval=t_eval, rtol=float(req.rtol), atol=float(req.atol))
    if req.max_step is not None:
        base_kwargs["max_step"] = float(req.max_step)
    if req.first_step is not None:
        base_kwargs["first_step"] = float(req.first_step)
    boundary_events = tuple(boundary_events)
    events_list: list[Callable[[float, np.ndarray], float]] = []
    for index, event in enumerate(boundary_events):
        terminal = (
            bool(boundary_terminal_flags[index])
            if index < len(boundary_terminal_flags)
            else bool(getattr(event, "terminal", False))
        )

        def _wrapped_event(
            t: float,
            y: np.ndarray,
            *,
            _event=event,
        ) -> float:
            return float(_event(t, y))

        _wrapped_event.terminal = terminal  # type: ignore[attr-defined]
        if hasattr(event, "direction"):
            _wrapped_event.direction = getattr(event, "direction")  # type: ignore[attr-defined]
        events_list.append(_wrapped_event)
    if req.cancellation_control is not None:
        events_list.append(_cancellation_event(req.cancellation_control))
    if events_list:
        base_kwargs["events"] = events_list

    symbolic_jacobian = req.symbolic_jacobian
    jac_callable = symbolic_jacobian.jacobian_func
    jac_sparsity = symbolic_jacobian.jac_sparsity

    def _disable_symbolic_jacobian_for_segment(
        *,
        code: str,
        reason: str,
    ) -> None:
        nonlocal jac_callable, jac_sparsity
        jac_callable = None
        prov["symbolic_jacobian"] = False
        prov["jacobian_sparsity_hint"] = jac_sparsity is not None
        if symbolic_jacobian.has_executable_jacobian:
            runtime_symbolic_jacobian = (
                symbolic_jacobian.with_runtime_disabled(
                    partially=False,
                    code=code,
                    reason=reason,
                )
            )
            prov.pop("symbolic_jacobian_identity", None)
            prov.update(runtime_symbolic_jacobian.provenance_fields())

    implicit_method = str(method) in {"Radau", "BDF"}
    if (
        implicit_method
        and jac_callable is not None
        and symbolic_jacobian.has_executable_jacobian
    ):
        jac_callable = _guard_runtime_symbolic_jacobian(
            jac_callable,
            req.cancellation_control,
        )
    prov["symbolic_jacobian"] = bool(
        implicit_method and jac_callable is not None
    )
    prov["jacobian_sparsity_hint"] = bool(
        implicit_method
        and jac_callable is None
        and jac_sparsity is not None
    )
    def _kwargs_for_method(method_name: str) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["method"] = method_name
        if method_name in ("Radau", "BDF") and jac_callable is not None:
            kwargs["jac"] = jac_callable
        elif method_name in ("Radau", "BDF") and jac_sparsity is not None:
            kwargs["jac_sparsity"] = jac_sparsity
        else:
            kwargs.pop("jac", None)
            kwargs.pop("jac_sparsity", None)
        return kwargs

    def _looks_like_brentq_sign_error(exc: BaseException) -> bool:
        return "different signs" in str(exc).lower()

    def _solve_once(method_name: str):
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        try:
            solved = _solve_ivp(
                fun=rhs,
                t_span=(t0, t1),
                y0=y0,
                **_kwargs_for_method(method_name),
            )  # type: ignore[misc]
        except ValueError as exc:
            # SciPy event root-finder can raise on cancellation races (brentq sign error).
            if (
                _looks_like_brentq_sign_error(exc)
                and req.cancellation_control is not None
                and req.cancellation_control.cancel_requested()
            ):
                raise SimulationCancelled() from exc
            raise
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        return solved

    try:
        sol = _solve_once(method)
    except _NonfiniteSymbolicJacobianError as exc:
        _disable_symbolic_jacobian_for_segment(
            code="nonfinite-runtime",
            reason=(
                "Symbolic Jacobian disabled for this segment because a runtime "
                "evaluation was non-finite or invalid."
            ),
        )
        prov["symbolic_jacobian_runtime_fallback"] = {
            "code": "nonfinite-runtime",
            "time": float(exc.time),
            "segment": segment_identity.to_payload(),
        }
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled() from exc
        sol = _solve_once(method)

    if not sol.success:
        prov["solver_failure"] = str(sol.message)
        t_fail = float(sol.t[-1]) if getattr(sol, "t", None) is not None and len(sol.t) > 0 else float(t0)
        raise create_solver_error(req.solver, t_fail, str(sol.message))

    prov["solver_used"] = method
    t_out = np.asarray(sol.t, float)
    raw_trajectory = solver_trajectory_array(
        sol.y,
        y0=y0,
        t_out=t_out,
    )
    raw_terminal_state = None
    if (
        t_out.size
        and float(t_out[-1]) == float(t1)
        and raw_trajectory.ndim == 2
        and raw_trajectory.shape[1]
    ):
        raw_terminal_state = np.asarray(
            raw_trajectory[:, -1],
            dtype=float,
        ).reshape(-1).copy()
    Y_out, nonnegative_projection = project_nonnegative_trajectory(
        raw_trajectory,
        context="Simulation",
    )
    if nonnegative_projection.projected:
        prov["nonnegative_projection"] = nonnegative_projection.to_payload()
    event_count = len(boundary_events)
    event_times = _normal_event_provenance(
        getattr(sol, "t_events", None),
        count=event_count,
    )
    raw_event_states, event_state_projection = (
        _normal_event_states_with_projection(
            getattr(sol, "y_events", None),
            count=event_count,
            species_count=int(np.asarray(y0, dtype=float).reshape(-1).size),
        )
    )
    if event_state_projection.projected:
        prov["event_state_nonnegative_projection"] = (
            event_state_projection.to_payload()
        )
    if (
        req.cancellation_control is not None
        and req.cancellation_control.cancel_requested()
    ):
        raise SimulationCancelled()
    output = _SimulationOutput(
        t=t_out,
        Y=Y_out,
        provenance=prov,
        intervention_execution=req.intervention_execution,
    )
    return SegmentRunResult(
        identity=segment_identity,
        output=output,
        event_times=event_times,
        raw_event_states=raw_event_states,
        raw_terminal_state=raw_terminal_state,
        trajectory_projection=nonnegative_projection,
        event_state_projection=event_state_projection,
        solver_used=str(output.provenance.get("solver_used") or method),
        symbolic_jacobian_used=bool(
            output.provenance.get("symbolic_jacobian")
        ),
        jacobian_sparsity_hint=bool(
            output.provenance.get("jacobian_sparsity_hint")
        ),
    )


def _run_scipy_segment(request: SegmentExecutionRequest) -> SegmentRunResult:
    return _execute_scipy(
        request.request,
        segment_identity=request.identity,
        rhs=request.rhs,
        t0=request.t0,
        t1=request.t1,
        y0=request.y0,
        t_eval=request.t_eval,
        prov=request.provenance,
        method=request.method,
        boundary_events=request.boundary_events,
        boundary_terminal_flags=request.boundary_terminal_flags,
    )


def _cancellation_event(
    control: _SimulationCancellationControl,
) -> Callable[[float, np.ndarray], float]:
    """Create the solver-owned event used only for secondary early stop."""

    def event(_time: float, _state: np.ndarray) -> float:
        return -1.0 if control.cancel_requested() else 1.0

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1.0  # type: ignore[attr-defined]
    return event


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

def solve_ode(req: _SimulationRequest) -> _SimulationOutput:
    if not isinstance(req, _SimulationRequest):
        raise TypeError(f"solve_ode expects SimulationRequest, got {type(req)!r}")
    if (
        req.cancellation_control is not None
        and req.cancellation_control.cancel_requested()
    ):
        raise SimulationCancelled()
    if req.temperature_schedule is not None:
        schedule = (
            req.temperature_schedule
            if isinstance(
                req.temperature_schedule,
                TemperatureScheduleDictProtocol,
            )
            else coerce_temperature_schedule(req.temperature_schedule)
        )
        req = replace(req, temperature_schedule=schedule)
    y0 = np.asarray(req.y0, float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise InitialConditionError("y0 must be a 1D finite array")
    if np.any(y0 < 0.0):
        raise InitialConditionError("y0 must contain non-negative concentrations")
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    rhs = _prepare_rhs(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method = _scipy_method_for(req.solver)
    _scrub_unused_jacobian_provenance(prov, method=method)
    return InterventionScheduleExecutionOwner(_run_scipy_segment).execute(
        ScheduleExecutionRequest(
            request=req,
            rhs=rhs,
            t0=t0,
            t1=t1,
            y0=y0,
            t_eval=t_eval,
            provenance=prov,
            method=method,
            schedule=req.intervention_execution,
        )
    )

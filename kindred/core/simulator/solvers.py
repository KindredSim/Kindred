# SPDX-License-Identifier: MIT
"""
ODE solver wrappers for Kindred.

SciPy is a hard dependency; all integration routes through `scipy.integrate.solve_ivp` with:
- Deterministic `t_eval` output grid
- Schedule execution supplies the selected temperature to RHS and Jacobian calls
- Non-negative concentration invariant enforcement at output boundaries
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Tuple,
)

import numpy as np
from kindred.core.scipy_integrate import load_scipy_integrate
from kindred.core.exceptions import (
    SimulationCancelled,
    SolverError,
    create_solver_error,
)
from kindred.core.time_grid import build_time_grid as _build_time_grid
from .nonnegative_invariant import (
    NonnegativeProjection,
    project_nonnegative_trajectory,
)
from .intervention_schedule_execution import (
    InterventionScheduleExecutionOwner,
    ScheduleExecutionRequest,
    SegmentExecutionRequest,
    SegmentIdentity,
    SegmentRunResult,
)
from .solver_types import (
    SIMULATION_NONFINITE_MESSAGE,
    Rhs2 as _Rhs2,
    SimulationCancellationControl as _SimulationCancellationControl,
    FinalSimulationOutput as _FinalSimulationOutput,
    SimulationScientificUnavailability as _SimulationScientificUnavailability,
    _SimulationScientificUnavailabilitySignal,
    SimulationRequest as _SimulationRequest,
)

logger = logging.getLogger(__name__)


class _SymbolicJacobianEvaluationError(RuntimeError):
    """Internal signal for one failed admitted analytic evaluation."""

    def __init__(self, *, time: float, reason: str) -> None:
        self.time = float(time)
        self.reason = str(reason)
        super().__init__(self.reason)


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
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    NonnegativeProjection,
]:
    raw_output = _normal_event_states(raw_states, count=count)
    projections: list[NonnegativeProjection] = []
    for exact_rows in raw_output:
        for exact in exact_rows:
            _projected, projection = project_nonnegative_trajectory(
                np.asarray(exact, dtype=float).reshape(-1, 1),
                context="Simulation event state",
            )
            projections.append(projection)
    return raw_output, NonnegativeProjection.aggregate(projections)


def _normal_event_states(
    raw_states: object,
    *,
    count: int,
) -> tuple[tuple[np.ndarray, ...], ...]:
    # SciPy gives None without events, else one (hits, states) array per event,
    # shaped (0,) when the event never fired; the first ``count`` events are
    # the boundary events.
    output: list[list[np.ndarray]] = [[] for _ in range(count)]
    for idx, raw in enumerate(tuple(raw_states or ())[:count]):
        arr = np.asarray(raw, dtype=float)
        if arr.size == 0:
            continue
        for row in arr:
            state = np.asarray(row, dtype=float).reshape(-1).copy()
            state.setflags(write=False)
            output[idx].append(state)
    return tuple(tuple(rows) for rows in output)


def normalize_solver_name(name: object) -> str:
    """
    Normalize a user-specified solver name to a SciPy `solve_ivp` method name.

    Unknown names raise `SolverError`.
    """
    normalized = str(name or "").strip().upper()
    scipy_methods = {
        "RADAU": "Radau",
        "BDF": "BDF",
    }
    if normalized in scipy_methods:
        return scipy_methods[normalized]
    allowed = ", ".join(sorted(scipy_methods.values()))
    raise SolverError(f"Unsupported solver name {name!r}. Expected one of: {allowed}.")


def _prepare_time_grid(req: _SimulationRequest, *, t0: float, t1: float) -> np.ndarray:
    # Run, Preview and Scan requests realized this grid when they were built;
    # Fit supplies its admitted sample times inside the span.
    if req.t_eval is None:
        return _build_time_grid(t0, t1, req.grid)
    return req.t_eval


def _build_provenance(req: _SimulationRequest, *, t_eval: np.ndarray) -> Dict[str, object]:
    symbolic_jacobian = req.symbolic_jacobian
    prov: Dict[str, object] = {
        "solver_requested": req.solver,
        "rtol": float(req.rtol),
        "atol": float(req.atol),
        "grid_type": ("explicit" if req.t_eval is not None else "N"),
        "grid_params": ({"t_eval": list(t_eval)} if req.t_eval is not None else dict(req.grid or {})),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    prov.update(symbolic_jacobian.provenance_fields())
    intervention_execution = req.intervention_execution
    prov["has_intervention_schedule"] = not intervention_execution.is_empty
    if not intervention_execution.is_empty:
        prov["intervention_execution"] = (
            intervention_execution.provenance_projection()
        )

    if req.temperature_schedule is not None:
        prov["temperature_schedule"] = req.temperature_schedule.to_dict()

    return prov


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
        matrix = np.asarray(jacobian_func(float(time), values), dtype=float)
        expected_shape = (values.size, values.size)
        if matrix.shape != expected_shape:
            raise _SymbolicJacobianEvaluationError(
                time=float(time),
                reason=f"its analytic Jacobian had shape {matrix.shape} instead of {expected_shape}",
            )
        if not np.all(np.isfinite(matrix)):
            raise _SymbolicJacobianEvaluationError(
                time=float(time),
                reason="its analytic Jacobian had nonfinite values",
            )
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
    method: str,
    boundary_events: Sequence[Callable[[float, np.ndarray], float]],
    boundary_terminal_flags: Sequence[bool],
    volume=None,
) -> SegmentRunResult:
    species_count = len(y0)
    # SciPy integrates on a segment-local clock.  BDF does not rescale its
    # difference history when ``t + h`` rounds, so an absolute origin makes a
    # fast transient's error estimate depend on ulp(t0) and collapses the step.
    # Callbacks, samples, roots and dense output keep the absolute clock, and
    # both segment ends map exactly.
    origin = float(t0)
    local_end = float(t1) - origin

    def absolute(tau: float) -> float:
        return float(t1) if tau >= local_end else min(origin + float(tau), float(t1))

    local_rhs = rhs if origin == 0.0 else (lambda tau, state: rhs(absolute(tau), state))
    base_kwargs: Dict[str, Any] = dict(t_eval=t_eval - origin, rtol=float(req.rtol), atol=float(req.atol))
    if volume is not None:
        y0 = np.append(y0, volume.value)
        base_kwargs["atol"] = np.append(np.full(species_count, req.atol), req.atol * volume.initial_value)
    from kindred.core.intervention_schedule import InterventionStopAction
    retain_dense = volume is not None or any(isinstance(item.then, InterventionStopAction) for item in req.intervention_execution.triggers)
    dense_output = retain_dense or req.sample_solver_steps
    if dense_output:
        base_kwargs["dense_output"] = True
    boundary_events = tuple(boundary_events)
    events_list: list[Callable[[float, np.ndarray], float]] = []
    for index, event in enumerate(boundary_events):
        terminal = bool(boundary_terminal_flags[index])

        def _wrapped_event(
            t: float,
            y: np.ndarray,
            *,
            _event=event,
        ) -> float:
            # Events see the solver state the RHS sees: species, then vessel volume.
            return float(_event(absolute(t), y))

        _wrapped_event.terminal = terminal  # type: ignore[attr-defined]
        if hasattr(event, "direction"):
            _wrapped_event.direction = getattr(event, "direction")  # type: ignore[attr-defined]
        events_list.append(_wrapped_event)
    if req.cancellation_control is not None:
        events_list.append(_cancellation_event(req.cancellation_control))
    # SciPy evaluates every event at each accepted step (root searches stay
    # inside that step), so the largest observed time is the last accepted
    # solver time; a failed step returns before events are evaluated.
    last_accepted_time = origin

    def _observe_accepted_time(tau: float, _y: np.ndarray) -> float:
        nonlocal last_accepted_time
        last_accepted_time = max(last_accepted_time, absolute(tau))
        return 1.0

    events_list.append(_observe_accepted_time)
    base_kwargs["events"] = events_list

    symbolic_jacobian = req.symbolic_jacobian
    jac_callable = symbolic_jacobian.jacobian_func

    if jac_callable is not None:
        guarded_jacobian = _guard_runtime_symbolic_jacobian(
            jac_callable,
            req.cancellation_control,
        )
        jac_callable = (
            guarded_jacobian if origin == 0.0
            else (lambda tau, state: guarded_jacobian(absolute(tau), state))
        )
    solve_kwargs = dict(base_kwargs)
    solve_kwargs["method"] = method
    if jac_callable is not None:
        solve_kwargs["jac"] = jac_callable

    def _solve():
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        rhs_values = np.asarray(
            rhs(float(t0), np.asarray(y0, dtype=float).reshape(-1).copy()),
            dtype=float,
        ).reshape(-1)
        if not np.all(np.isfinite(rhs_values)):
            raise _SimulationScientificUnavailabilitySignal(
                _SimulationScientificUnavailability(
                    producer="scipy",
                    stage="rhs_preflight",
                    message=SIMULATION_NONFINITE_MESSAGE,
                    details={
                        "solver": req.solver,
                        "time": float(t0),
                        "segment": segment_identity.to_payload(),
                    },
                )
            )
        try:
            solved = _solve_ivp(
                fun=local_rhs,
                t_span=(0.0, local_end),
                y0=y0,
                **solve_kwargs,
            )  # type: ignore[misc]
        except ValueError as exc:
            # SciPy's event root finder raises when an event's sign change
            # does not survive re-evaluation on the step interpolant.  Its
            # dense LU factorization or solve raises when a step's Newton
            # matrix or residual has overflowed to non-finite values.
            if "different signs" in str(exc).lower():
                stage, reason = "event_root_finding", "it could not find the time at which a trigger threshold is crossed"
            elif str(exc) == "array must not contain infs or NaNs":
                stage, reason = "nonfinite_step", "a step produced values that are not finite"
            else:
                raise
            if (
                req.cancellation_control is not None
                and req.cancellation_control.cancel_requested()
            ):
                raise SimulationCancelled() from exc
            error = create_solver_error(req.solver, last_accepted_time, reason, str(exc))
            if req.candidate_evaluation:
                raise _SimulationScientificUnavailabilitySignal(
                    _SimulationScientificUnavailability(
                        producer="scipy",
                        stage=stage,
                        message=error.message,
                        details={
                            "solver": req.solver,
                            "time": last_accepted_time,
                            "message": str(exc),
                        },
                    )
                ) from exc
            raise error from exc
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            if dense_output:
                solved.sol = None
            raise SimulationCancelled()
        return solved

    try:
        sol = _solve()
    except _SymbolicJacobianEvaluationError as exc:
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled() from exc
        details = {
            "solver": req.solver,
            "time": float(exc.time),
            "segment": segment_identity.to_payload(),
            "reason": exc.reason,
        }
        error = create_solver_error(req.solver, exc.time, exc.reason, "")
        error.details.update(details)
        if req.candidate_evaluation:
            raise _SimulationScientificUnavailabilitySignal(
                _SimulationScientificUnavailability(
                    producer="analytic_jacobian",
                    stage="runtime_evaluation",
                    message=error.message,
                    details=details,
                )
            ) from exc
        raise error from exc

    try:
        if not sol.success:
            if dense_output:
                sol.sol = None
            t_fail = last_accepted_time
            # BDF and Radau report one failure: the step size fell below
            # the spacing of floating-point numbers at the current time.
            error = create_solver_error(
                req.solver, t_fail, "its step size became too small to continue", str(sol.message),
            )
            if req.candidate_evaluation:
                raise _SimulationScientificUnavailabilitySignal(
                    _SimulationScientificUnavailability(
                        producer="scipy",
                        stage="solve_ivp_unsuccessful",
                        message=error.message,
                        details={
                            "solver": req.solver,
                            "time": t_fail,
                            "message": str(sol.message),
                        },
                    )
                )
            raise error

        # Detach SciPy's time vector at ingress.  The trajectory copy below is
        # owned exactly once by project_nonnegative_trajectory; retain the exact
        # pre-projection terminal column separately for continuation semantics.
        t_out = np.asarray(t_eval, dtype=float)[:len(sol.t)].copy()
        # SciPy returns a 1-D empty array when no requested time was reached.
        raw_trajectory = np.asarray(sol.y, dtype=float).reshape(len(y0), -1)
        local_dense = sol.sol
        if req.sample_solver_steps:
            # Every accepted step before the solver stopped (an event can end
            # the segment early).  Steps the absolute clock cannot tell apart
            # become one row.
            local = np.asarray(local_dense.ts, dtype=float)
            local = local[(local > 0.0) & (local < local_dense.t_max)]
            step_times, first = np.unique(origin + local, return_index=True)
            extra = ((step_times > origin) & (step_times < absolute(float(local_dense.t_max)))
                     & ~np.isin(step_times, t_out))
            if extra.any():
                times = np.concatenate((t_out, step_times[extra]))
                order = np.argsort(times, kind="stable")
                t_out = times[order]
                raw_trajectory = np.concatenate(
                    (raw_trajectory,
                     np.asarray(local_dense(local[first][extra]), dtype=float).reshape(len(y0), -1)),
                    axis=1,
                )[:, order]
        dense = (None if local_dense is None else local_dense if origin == 0.0
                 else (lambda times: local_dense(np.asarray(times, dtype=float) - origin)))
        if volume is not None:
            volume.record_segment(t0, absolute(float(local_dense.t_max)), dense)
            raw_trajectory = raw_trajectory[:species_count]
        raw_terminal_state = None
        if (
            t_out.size
            and float(t_out[-1]) == float(t1)
            and raw_trajectory.ndim == 2
            and raw_trajectory.shape[1]
        ):
            raw_terminal_state = raw_trajectory[:, -1].copy()
        Y_out, nonnegative_projection = project_nonnegative_trajectory(
            raw_trajectory,
            context="Simulation",
        )
        event_count = len(boundary_events)
        event_times = tuple(
            tuple(absolute(value) for value in values)
            for values in _normal_event_provenance(
                sol.t_events,
                count=event_count,
            )
        )
        raw_event_states, event_state_projection = (
            _normal_event_states_with_projection(
                (tuple(np.asarray(states).reshape(-1, len(y0))[:, :species_count] for states in (sol.y_events or ()))
                 if volume is not None else sol.y_events),
                count=event_count,
            )
        )
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        return SegmentRunResult(
            t=t_out,
            Y=Y_out,
            event_times=event_times,
            raw_event_states=raw_event_states,
            raw_terminal_state=raw_terminal_state,
            trajectory_projection=nonnegative_projection,
            event_state_projection=event_state_projection,
            dense_output=((lambda times, dense=dense: dense(times)[:species_count]) if volume is not None
                          else dense if retain_dense else None),
            symbolic_jacobian_used=jac_callable is not None,
        )
    finally:
        if dense_output:
            sol.sol = None


def _run_scipy_segment(request: SegmentExecutionRequest) -> SegmentRunResult:
    return _execute_scipy(
        request.request,
        segment_identity=request.identity,
        rhs=request.rhs,
        t0=request.t0,
        t1=request.t1,
        y0=request.y0,
        t_eval=request.t_eval,
        method=request.method,
        boundary_events=request.boundary_events,
        boundary_terminal_flags=request.boundary_terminal_flags,
        volume=request.volume,
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
) -> tuple[tuple[float, ...], ...]:
    out = [[] for _ in range(count)]
    # The first ``count`` SciPy events are the boundary events.
    for idx, raw in enumerate(tuple(events or ())[:count]):
        out[idx].extend(float(value) for value in raw)
    return tuple(tuple(values) for values in out)

def solve_ode(
    req: _SimulationRequest,
) -> _FinalSimulationOutput | _SimulationScientificUnavailability:
    if (
        req.cancellation_control is not None
        and req.cancellation_control.cancel_requested()
    ):
        raise SimulationCancelled()
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method = req.solver
    try:
        output = InterventionScheduleExecutionOwner(_run_scipy_segment).execute(
            ScheduleExecutionRequest(
                request=req,
                t_eval=t_eval,
                provenance=prov,
                method=method,
            )
        )
    except _SimulationScientificUnavailabilitySignal as exc:
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled() from exc
        if req.candidate_evaluation:
            return exc.outcome
        raise SolverError(exc.outcome.message, details={"solver": req.solver}) from exc
    return output

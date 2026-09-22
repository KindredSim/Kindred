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
    Mapping,
    Sequence,
    Tuple,
)

import numpy as np
from kindred.core.scipy_integrate import load_scipy_integrate
from kindred.core.temperature import (
    TemperatureScheduleDictProtocol,
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
    species_count: int,
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
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
                np.asarray(exact, dtype=float).reshape(-1, 1),
                context="Simulation event state",
            )
            projections.append(projection)
    return raw_output, NonnegativeProjection.aggregate(projections)


def _normal_event_states(
    raw_states: object,
    *,
    count: int,
    species_count: int,
) -> tuple[tuple[np.ndarray, ...], ...]:
    output: list[list[np.ndarray]] = [
        [] for _ in range(max(0, int(count)))
    ]
    if raw_states is None or count <= 0 or not isinstance(raw_states, Sequence):
        return tuple(tuple(rows) for rows in output)
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
    raise SolverError(f"Unsupported solver name {name!r}; expected one of: {allowed}.")


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


def _build_provenance(req: _SimulationRequest, *, t_eval: np.ndarray) -> Dict[str, object]:
    symbolic_jacobian = req.symbolic_jacobian
    prov: Dict[str, object] = {
        "solver_requested": req.solver,
        "rtol": float(req.rtol),
        "atol": float(req.atol),
        "max_step": (None if req.max_step is None else float(req.max_step)),
        "first_step": (None if req.first_step is None else float(req.first_step)),
        "grid_type": ("explicit" if req.t_eval is not None else "N"),
        "grid_params": ({"t_eval": list(t_eval)} if req.t_eval is not None else dict(req.grid or {})),
        "has_temperature_schedule": req.temperature_schedule is not None,
    }
    prov.update(symbolic_jacobian.provenance_fields())
    prov["jacobian_sparsity_hint"] = symbolic_jacobian.jac_sparsity is not None
    intervention_execution = req.intervention_execution
    prov["has_intervention_schedule"] = not intervention_execution.is_empty
    if not intervention_execution.is_empty:
        prov["intervention_execution"] = (
            intervention_execution.provenance_projection()
        )

    if req.temperature_schedule is not None:
        temperature_schedule = req.temperature_schedule
        if isinstance(temperature_schedule, TemperatureScheduleDictProtocol):
            prov["temperature_schedule"] = dict(temperature_schedule.to_dict())
        else:
            prov["temperature_schedule"] = str(temperature_schedule)

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
        try:
            matrix = np.asarray(
                jacobian_func(float(time), values),
                dtype=float,
            )
        except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
            raise _SymbolicJacobianEvaluationError(
                time=float(time),
                reason=(
                    "Analytic Jacobian callback raised "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        expected_shape = (values.size, values.size)
        if matrix.shape != expected_shape:
            raise _SymbolicJacobianEvaluationError(
                time=float(time),
                reason=(
                    "Analytic Jacobian callback returned shape "
                    f"{matrix.shape}; expected {expected_shape}."
                ),
            )
        if not np.all(np.isfinite(matrix)):
            raise _SymbolicJacobianEvaluationError(
                time=float(time),
                reason="Analytic Jacobian callback returned non-finite values.",
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
    prov: Dict[str, object],
    method: str,
    boundary_events: Sequence[Callable[[float, np.ndarray], float]],
    boundary_terminal_flags: Sequence[bool],
    volume=None,
) -> SegmentRunResult:
    species_count = len(y0)
    base_kwargs: Dict[str, Any] = dict(t_eval=t_eval, rtol=float(req.rtol), atol=float(req.atol))
    if volume is not None:
        y0 = np.append(y0, volume.value)
        base_kwargs["atol"] = np.append(np.full(species_count, req.atol), req.atol * volume.initial_value)
    from kindred.core.intervention_schedule import InterventionStopAction
    retain_dense = volume is not None or any(isinstance(item.then, InterventionStopAction) for item in req.intervention_execution.triggers)
    if retain_dense:
        base_kwargs["dense_output"] = True
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
            return float(_event(t, y[:species_count]))

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

    if jac_callable is not None and symbolic_jacobian.has_executable_jacobian:
        jac_callable = _guard_runtime_symbolic_jacobian(
            jac_callable,
            req.cancellation_control,
        )
    prov["symbolic_jacobian"] = bool(jac_callable is not None)
    prov["jacobian_sparsity_hint"] = bool(
        jac_callable is None and jac_sparsity is not None
    )
    def _kwargs_for_method(method_name: str) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["method"] = method_name
        if jac_callable is not None:
            kwargs["jac"] = jac_callable
        elif jac_sparsity is not None:
            kwargs["jac_sparsity"] = jac_sparsity
        return kwargs

    def _looks_like_brentq_sign_error(exc: BaseException) -> bool:
        return "different signs" in str(exc).lower()

    rhs_preflighted = False

    def _solve_once(method_name: str):
        nonlocal rhs_preflighted
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled()
        if not rhs_preflighted:
            rhs_values = np.asarray(
                rhs(float(t0), np.asarray(y0, dtype=float).reshape(-1).copy()),
                dtype=float,
            ).reshape(-1)
            if not np.all(np.isfinite(rhs_values)):
                raise _SimulationScientificUnavailabilitySignal(
                    _SimulationScientificUnavailability(
                        producer="scipy",
                        stage="rhs_preflight",
                        details={
                            "solver": req.solver,
                            "time": float(t0),
                            "segment": segment_identity.to_payload(),
                        },
                    )
                )
            rhs_preflighted = True
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
            if retain_dense:
                solved.sol = None
            raise SimulationCancelled()
        return solved

    try:
        sol = _solve_once(method)
    except _SymbolicJacobianEvaluationError as exc:
        if (
            req.cancellation_control is not None
            and req.cancellation_control.cancel_requested()
        ):
            raise SimulationCancelled() from exc
        if req.candidate_evaluation:
            raise _SimulationScientificUnavailabilitySignal(
                _SimulationScientificUnavailability(
                    producer="analytic_jacobian",
                    stage="runtime_evaluation",
                    details={
                        "solver": req.solver,
                        "time": float(exc.time),
                        "segment": segment_identity.to_payload(),
                        "reason": exc.reason,
                    },
                )
            ) from exc
        raise SolverError(
            "Analytic Jacobian evaluation failed at "
            f"t={exc.time:.4e}: {exc.reason}",
            suggestion=(
                "Disable the analytic Jacobian preference to execute with the "
                "solver's numerical Jacobian, or correct the reported "
                "analytic evaluation condition."
            ),
            details={
                "solver": req.solver,
                "time": float(exc.time),
                "segment": segment_identity.to_payload(),
                "reason": exc.reason,
            },
        ) from exc

    try:
        if not sol.success:
            dense_failure_time = float(sol.sol.ts[-1]) if retain_dense and sol.sol is not None else None
            if retain_dense:
                sol.sol = None
            prov["solver_failure"] = str(sol.message)
            t_fail = dense_failure_time if dense_failure_time is not None else (float(sol.t[-1]) if getattr(sol, "t", None) is not None and len(sol.t) > 0 else float(t0))
            if req.candidate_evaluation:
                raise _SimulationScientificUnavailabilitySignal(
                    _SimulationScientificUnavailability(
                        producer="scipy",
                        stage="solve_ivp_unsuccessful",
                        details={
                            "solver": req.solver,
                            "time": t_fail,
                            "message": str(sol.message),
                        },
                    )
                )
            raise create_solver_error(
                req.solver,
                t_fail,
                str(sol.message),
            )

        prov["solver_used"] = method
        # Detach SciPy's time vector at ingress.  The trajectory copy below is
        # owned exactly once by project_nonnegative_trajectory; retain the exact
        # pre-projection terminal column separately for continuation semantics.
        t_out = np.asarray(sol.t, dtype=float).copy()
        raw_trajectory = solver_trajectory_array(
            sol.y,
            y0=y0,
            t_out=t_out,
        )
        if volume is not None:
            volume.record_segment(t0, float(sol.sol.t_max), sol.sol)
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
        if nonnegative_projection.projected:
            prov["nonnegative_projection"] = nonnegative_projection.to_payload()
        event_count = len(boundary_events)
        event_times = _normal_event_provenance(
            getattr(sol, "t_events", None),
            count=event_count,
        )
        raw_event_states, event_state_projection = (
            _normal_event_states_with_projection(
                (tuple(np.asarray(states).reshape(-1, len(y0))[:, :species_count] for states in (sol.y_events or ()))
                 if volume is not None else getattr(sol, "y_events", None)),
                count=event_count,
                species_count=species_count,
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
        return SegmentRunResult(
            t=t_out,
            Y=Y_out,
            event_times=event_times,
            raw_event_states=raw_event_states,
            raw_terminal_state=raw_terminal_state,
            trajectory_projection=nonnegative_projection,
            event_state_projection=event_state_projection,
            solver_used=str(prov.get("solver_used") or method),
            dense_output=((lambda times, dense=sol.sol: dense(times)[:species_count]) if volume is not None
                          else sol.sol if retain_dense else None),
            symbolic_jacobian_used=bool(
                prov.get("symbolic_jacobian")
            ),
            jacobian_sparsity_hint=bool(
                prov.get("jacobian_sparsity_hint")
            ),
        )
    finally:
        if retain_dense:
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
        prov=request.provenance,
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
    out = [[] for _ in range(max(0, int(count)))]
    if not isinstance(events, Sequence):
        return tuple(tuple(values) for values in out)
    for idx, raw in enumerate(events):
        if idx >= len(out):
            break
        try:
            out[idx].extend(float(value) for value in raw)  # type: ignore[union-attr]
        except TypeError:
            continue
    return tuple(tuple(values) for values in out)

def solve_ode(
    req: _SimulationRequest,
) -> _FinalSimulationOutput | _SimulationScientificUnavailability:
    if not isinstance(req, _SimulationRequest):
        raise TypeError(f"solve_ode expects SimulationRequest, got {type(req)!r}")
    if (
        req.cancellation_control is not None
        and req.cancellation_control.cancel_requested()
    ):
        raise SimulationCancelled()
    y0 = np.asarray(req.y0, float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise InitialConditionError("y0 must be a 1D finite array")
    if np.any(y0 < 0.0):
        raise InitialConditionError("y0 must contain non-negative concentrations")
    t0, t1 = map(float, req.t_span)
    t_eval = _prepare_time_grid(req, t0=t0, t1=t1)
    prov = _build_provenance(req, t_eval=t_eval)

    method = normalize_solver_name(req.solver)
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
        raise create_solver_error(
            req.solver,
            t0,
            "Simulation produced non-finite values.",
        ) from exc
    return output

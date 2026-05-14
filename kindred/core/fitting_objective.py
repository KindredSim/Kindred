"""
Objective construction helpers for fitting.

Builds residual-vector objectives that rely on prepared simulation execution.
Optimization is intentionally kept separate (see `kindred.core.fitting_optimization`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from kindred.core.cache import cache_simulation, fingerprint_simulation_request
from kindred.core.exceptions import FittingCancelled, FitSimulationError, SimulationCancelled
from kindred.core.objective import ObjectiveContext, ObjectiveWrapper
from kindred.core.runtime_defaults import WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT
from kindred.core.simulation_preparation import (
    PreparedFittingObjectiveContext,
    SimulationPreparationError,
    materialize_request_intervention_schedule_for_parameter_values,
    prepare_fitting_objective_context,
)
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

logger = logging.getLogger(__name__)

__all__ = ["build_fitting_objective", "build_prepared_fitting_objective"]

if TYPE_CHECKING:
    from kindred.core.simulator.solvers import SimulationOutput


@cache_simulation(maxsize=256)
def _solve_request_cached(_mechanism, *, _request) -> "SimulationOutput":
    from kindred.core.simulator.solvers import solve_ode

    return solve_ode(_request)


def build_fitting_objective(
    mechanism_text: str,
    param_names: List[str],
    t_exp: np.ndarray,
    y_exp: np.ndarray,
    target_species: str,
    temperature_K: float = 298.15,
    initials: Optional[Dict[str, float]] = None,
    solver: str = DEFAULT_SOLVER_NAME,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    wegscheider_cyclicity_enabled: bool = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    prepare_func: Callable[..., object] | None = None,
    prepare_context_func: Callable[..., PreparedFittingObjectiveContext] | None = None,
    solve_policy_factory: Callable[[PreparedFittingObjectiveContext], Callable[[object, np.ndarray], "SimulationOutput"]] | None = None,
    parameter_algebra_policy_factory: Callable[[PreparedFittingObjectiveContext], Callable[[Dict[str, float]], None]] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if initials is None:
        initials = {}

    t_exp = np.asarray(t_exp, dtype=float).reshape(-1)
    y_exp = np.asarray(y_exp, dtype=float).reshape(-1)

    if t_exp.size < 2 or np.any(~np.isfinite(t_exp)):
        return _build_failure_objective(
            y_exp,
            "Experimental time points must be finite and contain at least two samples.",
        )
    if np.any(np.diff(t_exp) <= 0):
        return _build_failure_objective(
            y_exp,
            "Experimental time points must be strictly increasing for solver evaluation.",
        )
    if np.any(~np.isfinite(y_exp)):
        return _build_failure_objective(y_exp, "Experimental observations must be finite for fitting.")

    prepare_context = prepare_fitting_objective_context if prepare_context_func is None else prepare_context_func
    try:
        prepared = prepare_context(
            mechanism_text=mechanism_text,
            param_names=param_names,
            t_exp=t_exp,
            target_species=target_species,
            temperature_K=temperature_K,
            initials=initials,
            solver=solver,
            rtol=rtol,
            atol=atol,
            wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
            prepare_func=prepare_func,
        )
    except (FitSimulationError, ValueError, KeyError, TypeError) as exc:
        logger.error("Failed to prepare mechanism during fitting setup: %s", exc)
        error = exc
        if not isinstance(error, FitSimulationError):
            error = FitSimulationError(f"Failed to prepare mechanism for fitting: {exc}")
        return _build_failure_objective(
            y_exp,
            f"Failed to prepare mechanism for fitting: {exc}",
            error=error,
        )

    solve_policy = (
        _build_cached_solver_policy(prepared)
        if solve_policy_factory is None
        else solve_policy_factory(prepared)
    )
    parameter_algebra_policy = (
        _build_parameter_algebra_policy(prepared)
        if parameter_algebra_policy_factory is None
        else parameter_algebra_policy_factory(prepared)
    )
    return build_prepared_fitting_objective(
        prepared,
        y_exp=y_exp,
        solve_request=solve_policy,
        parameter_algebra_policy=parameter_algebra_policy,
    )


def build_prepared_fitting_objective(
    prepared: PreparedFittingObjectiveContext,
    *,
    y_exp: np.ndarray,
    solve_request: Callable[[object, np.ndarray], "SimulationOutput"],
    parameter_algebra_policy: Callable[[Dict[str, float]], None],
) -> Callable[[np.ndarray], np.ndarray]:
    ctx = ObjectiveContext()
    penalty_residual = np.full_like(y_exp, 1e6, dtype=float)
    mechanism = prepared.bound.mechanism

    def objective(param_values: np.ndarray) -> np.ndarray:
        param_state = _snapshot_params(prepared.requested_param_names, param_values)
        if mechanism is None:
            raise FitSimulationError("Prepared mechanism unavailable during fitting.", failed_params=param_state)

        def _record_error(err: FitSimulationError, provenance=None) -> None:
            ctx.set_error(err, provenance)

        try:
            _update_parameter_bindings(prepared, param_values, param_state)
            parameter_algebra_policy(param_state)
            request = materialize_request_intervention_schedule_for_parameter_values(
                mechanism=prepared.bound.mechanism,
                request=prepared.request,
                unresolved_intervention_schedule=prepared.unresolved_intervention_schedule,
                parameter_values=param_state,
                species_names=prepared.bound.species_names,
                runtime_parameter_names=prepared.bound.bindings.keys(),
            )
            result = solve_request(request, param_values)
        except (FittingCancelled, SimulationCancelled) as exc:
            raise FittingCancelled() from exc
        except SimulationPreparationError as exc:
            err = FitSimulationError(
                str(exc),
                failed_params=param_state,
                details={"fatal": True, "stage": exc.stage},
            )
            _record_error(err)
            raise err from exc
        except FitSimulationError as exc:
            _record_error(exc)
            raise
        except Exception as exc:
            logger.error("Simulation failed during fitting: %s", exc)
            err = FitSimulationError(
                f"Simulation failed for parameters {param_state}: {exc}",
                failed_params=param_state,
            )
            _record_error(err)
            raise err from exc

        try:
            y_model = _extract_model_series(
                prepared,
                result=result,
                param_state=param_state,
                expected_size=int(y_exp.shape[0]),
            )
        except FitSimulationError as exc:
            _record_error(exc, getattr(result, "provenance", None))
            raise
        except Exception as exc:
            logger.error("Failed to extract model series during fitting: %s", exc)
            err = FitSimulationError(
                f"Failed to extract model series during fitting: {exc}",
                failed_params=param_state,
            )
            _record_error(err, getattr(result, "provenance", None))
            raise err from exc

        ctx.set_model(y_model)
        if not np.all(np.isfinite(y_model)):
            err = FitSimulationError(
                f"Simulation produced non-finite values for {prepared.target_species}.",
                failed_params=param_state,
            )
            _record_error(err, getattr(result, "provenance", None))
            return penalty_residual

        residuals = y_model - y_exp
        if not np.all(np.isfinite(residuals)):
            err = FitSimulationError(
                "Residuals contain non-finite values; rejecting simulation output.",
                failed_params=param_state,
            )
            _record_error(err, getattr(result, "provenance", None))
            return penalty_residual

        return residuals

    wrapper = ObjectiveWrapper(objective, ctx)
    wrapper.y_exp = y_exp  # type: ignore[attr-defined]
    wrapper._kindred_vector_objective = True  # type: ignore[attr-defined]
    wrapper._kindred_t_span = tuple(map(float, prepared.request.t_span))  # type: ignore[attr-defined]
    return wrapper


def _build_failure_objective(
    y_exp: np.ndarray,
    reason: str,
    *,
    error: FitSimulationError | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ctx = ObjectiveContext()

    def _fail(_: np.ndarray) -> np.ndarray:
        failure = error
        if failure is None:
            failure = FitSimulationError(reason)
        ctx.set_error(failure, None)
        raise failure

    wrapper = ObjectiveWrapper(_fail, ctx)
    wrapper.y_exp = y_exp  # type: ignore[attr-defined]
    wrapper._kindred_vector_objective = True  # type: ignore[attr-defined]
    wrapper.failure_reason = reason  # type: ignore[attr-defined]
    return wrapper


def _snapshot_params(param_names: List[str], values: np.ndarray) -> Dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(param_names)}


def _update_parameter_bindings(
    prepared: PreparedFittingObjectiveContext,
    param_values: np.ndarray,
    param_state: Dict[str, float],
) -> None:
    try:
        for i, name in enumerate(prepared.requested_param_names):
            binding = prepared.bound.bindings.get(name)
            if binding is not None:
                binding.set(float(param_values[i]))
    except Exception as exc:
        logger.error("Failed to update parameter bindings: %s", exc)
        raise FitSimulationError(
            f"Failed to update parameter bindings: {exc}",
            failed_params=param_state,
        ) from exc


def _build_parameter_algebra_policy(
    prepared: PreparedFittingObjectiveContext,
) -> Callable[[Dict[str, float]], None]:
    from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

    def _apply(param_state: Dict[str, float]) -> None:
        try:
            _ = apply_parameter_algebra_to_mechanism(
                prepared.bound.mechanism_text,
                mechanism=prepared.bound.mechanism,
                require_mutable=True,
            )
        except Exception as exc:
            raise FitSimulationError(
                f"Parameter algebra failed for parameters {param_state}: {exc}",
                failed_params=param_state,
            ) from exc

    return _apply


def _build_cached_solver_policy(
    prepared: PreparedFittingObjectiveContext,
) -> Callable[[object, np.ndarray], "SimulationOutput"]:
    def _solve(request: object, param_values: np.ndarray) -> "SimulationOutput":
        req_fp = fingerprint_simulation_request(request)
        fingerprint = None
        if req_fp is not None:
            params_key = tuple(float(v) for v in np.asarray(param_values, dtype=float).reshape(-1))
            try:
                serialized = json.dumps(
                    {"req_fp": req_fp, "params_key": params_key},
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            except TypeError:
                serialized = repr((req_fp, params_key))
            fingerprint = hashlib.sha256(serialized.encode("utf-8", "ignore")).hexdigest()
        return _solve_request_cached(
            prepared.bound.mechanism,
            _cache_fingerprint=fingerprint,
            _request=request,
        )

    return _solve


def _extract_model_series(
    prepared: PreparedFittingObjectiveContext,
    *,
    result: "SimulationOutput",
    param_state: Dict[str, float],
    expected_size: int,
) -> np.ndarray:
    from kindred.core.algebra.errors import (
        AlgebraError,
        AlgebraNameError,
        AlgebraShadowError,
        AlgebraSyntaxError,
    )
    from kindred.core.algebra.simulation_series import evaluate_compiled_algebra_series_for_simulation
    from kindred.core.exceptions import ErrorContext

    try:
        if prepared.target_is_species:
            if prepared.target_species_index is None:
                raise FitSimulationError(
                    f"Internal error: missing species index for target '{prepared.target_species}'.",
                    failed_params=param_state,
                )
            y_model = np.asarray(result.Y[int(prepared.target_species_index), :], dtype=float).reshape(-1)
        else:
            if prepared.compiled_algebra is None:
                raise FitSimulationError(
                    "Algebra target requested but mechanism contains no Algebra observables.",
                    failed_params=param_state,
                )
            species_series = {
                name: result.Y[idx, :] for idx, name in enumerate(prepared.bound.species_names)
            }
            algebra_series, algebra_scalars = evaluate_compiled_algebra_series_for_simulation(
                prepared.bound.mechanism,
                prepared.compiled_algebra,
                t=result.t,
                species_series=species_series,
                initials=prepared.initials_for_algebra,
                temperature_K=float(prepared.temperature_K),
            )
            if prepared.target_species in (algebra_series or {}):
                y_model = np.asarray(algebra_series[prepared.target_species], dtype=float).reshape(-1)
            elif prepared.target_species in (algebra_scalars or {}):
                y_model = np.full(expected_size, float(algebra_scalars[prepared.target_species]), dtype=float)
            else:
                raise FitSimulationError(
                    f"Algebra target '{prepared.target_species}' was not produced by Algebra evaluation.",
                    failed_params=param_state,
                )
        if y_model.shape[0] != expected_size:
            raise FitSimulationError(
                f"Simulation returned {y_model.shape[0]} points but expected {expected_size}.",
                failed_params=param_state,
            )
        return y_model
    except AlgebraError as exc:
        is_fatal = isinstance(exc, (AlgebraNameError, AlgebraShadowError, AlgebraSyntaxError))
        raise FitSimulationError(
            f"Algebra evaluation failed during fitting: {exc}",
            failed_params=param_state,
            details={"fatal": bool(is_fatal)},
            context=ErrorContext(line=exc.line, col=exc.col, line_text=exc.line_text),
        ) from exc

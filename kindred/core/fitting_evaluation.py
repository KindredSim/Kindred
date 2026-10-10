"""
Shared per-dataset fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds one evaluator per dataset from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from kindred.core.exceptions import (
    FitCandidateDomainError,
    FitSimulationError,
    FittingCancelled,
    KineticRuntimeDomainError,
    SimulationCancelled,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
from kindred.core.algebra.errors import AlgebraError
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    PreparedSimulationRun,
    fit_simulation_error_from_preparation_error,
    build_simulation_request_from_prepared_run,
    prepare_simulation_run,
    SimulationPreparationError,
)
from kindred.core.simulation_preparation_stages import PreparedSolverConfig
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_result_finalization import finalize_simulation_result
from kindred.core.simulator.solver_types import (
    SimulationCancellationControl,
    SimulationScientificUnavailability,
)
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.time_grid import build_time_grid
from kindred.core.volume import PhysicalDomainError
from kindred.core.simulation_failure import factual_cause_from_exception

logger = logging.getLogger(__name__)

__all__ = [
    "FITTING_PARAM_ORIGIN_CONFIGURED_DATASET",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED",
    "FittingEvaluator",
    "prepare_fitting_execution_context",
]

FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED = "optimizer_shared"
FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET = "optimizer_dataset"
FITTING_PARAM_ORIGIN_CONFIGURED_DATASET = "configured_dataset"
FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR = "configured_evaluator"

def _parameter_origin_for(name: str, origins: Optional[Mapping[str, str]]) -> str:
    origin = str((origins or {}).get(name, FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR) or "").strip()
    return origin or FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR


def _coerce_consumed_parameter_value(
    *,
    name: str,
    raw_value: object,
    origins: Optional[Mapping[str, str]],
    failed_params: Optional[Dict[str, float]],
) -> float:
    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        error_type = (
            FitCandidateDomainError
            if _parameter_origin_for(name, origins)
            in {
                FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
                FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
            }
            else FitSimulationError
        )
        raise error_type(
            f"Invalid parameter value for {name!r}: {raw_value!r}",
            **(
                {"stage": "parameter_binding"}
                if error_type is FitCandidateDomainError
                else {}
            ),
            failed_params=dict(failed_params or {}) or None,
            details={
                "fatal": error_type is FitSimulationError,
                "parameter_origin": _parameter_origin_for(name, origins),
            },
        ) from exc
    if np.isfinite(value):
        return value
    origin = _parameter_origin_for(name, origins)
    error_type = (
        FitCandidateDomainError
        if origin
        in {
            FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
            FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
        }
        else FitSimulationError
    )
    raise error_type(
        f"Nonfinite parameter value for {name!r}: {raw_value!r}",
        **(
            {"stage": "parameter_binding"}
            if error_type is FitCandidateDomainError
            else {}
        ),
        failed_params=dict(failed_params or {}) or None,
        details={
            "fatal": error_type is FitSimulationError,
            "parameter_origin": origin,
        },
    )


def _scientific_unavailability_from_exception(
    exc: object,
    *,
    producer: str,
    stage: str,
    failed_params: Optional[Dict[str, float]],
) -> ScientificUnavailable:
    details = dict(getattr(exc, "details", None) or {})
    return ScientificUnavailable(
        producer=producer,
        stage=stage,
        details={
            **details,
            "failed_params": dict(failed_params or {}),
        },
        factual_cause=getattr(exc, "factual_cause", None),
    )


def _known_candidate_unavailability(
    exc: BaseException,
    *,
    failed_params: Optional[Dict[str, float]],
) -> ScientificUnavailable | None:
    """Translate producer-owned typed candidate failures at this boundary.

    Explicit candidate-domain and physical-predicate failures reject candidates.
    Preparation, Algebra, kinetic, and wrapped failures remain fatal; the
    solver's typed ``SimulationScientificUnavailability`` is settled separately
    below.
    """

    if isinstance(exc, PhysicalDomainError):
        return ScientificUnavailable(
            producer="physical_intervention", stage="execution",
            details={**exc.details, "failed_params": dict(failed_params or {})},
            factual_cause=factual_cause_from_exception(exc),
        )
    if isinstance(exc, FitCandidateDomainError):
        details = getattr(exc, "details", {})
        stage = (
            str(details.get("stage", "candidate_evaluation"))
            if isinstance(details, Mapping)
            else "candidate_evaluation"
        )
        return _scientific_unavailability_from_exception(
            exc,
            producer="fit_candidate",
            stage=stage,
            failed_params=failed_params,
        )
    return None


def _fatal_evaluation_error(
    exc: BaseException,
    *,
    stage: str,
    failed_params: Optional[Dict[str, float]],
) -> FitSimulationError:
    """A fatal Fit failure states its cause's sentence and authored line;
    the evaluation stage stays a structured fact."""
    details = dict(getattr(exc, "details", None) or {})
    details.update(
        {
            "fatal": True,
            "stage": str(stage),
        }
    )
    return FitSimulationError(
        str(getattr(exc, "message", None) or exc).strip() or type(exc).__name__,
        failed_params=failed_params,
        details=details,
        context=getattr(exc, "context", None),
    )


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Typed fitting request and preparation metadata.

    Its sole producer unions the admitted regular grid with the admitted
    observation times inside the Fit span.
    """

    prepared_run: PreparedSimulationRun
    requested_sample_times: tuple[float, ...]
    fit_sampling_policy: str = "fit_internal_regular_grid"
    fit_sampling_identity: Mapping[str, object] = field(default_factory=dict)
    initial_prefix: str = "init:"


def prepare_fitting_execution_context(
    *,
    prepared_dsl: PreparedDslRequestSet,
    param_names: List[str],
    t_start: float = 0.0,
    t_end: float,
    objective_sample_times: Sequence[float],
    prepared_solver_config: PreparedSolverConfig,
    initial_prefix: str = "init:",
    fit_sampling_policy: str = "fit_internal_regular_grid",
    fit_sampling_identity: Mapping[str, object] | None = None,
    cancellation_control: SimulationCancellationControl | None = None,
) -> PreparedFittingExecutionContext:
    """Prepare structured execution data for one per-dataset evaluator.

    The Fit request admits the span, solver policy, regular grid and the
    observation times inside the span; this builds one dataset's context.
    """

    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    initial_prefix = str(initial_prefix or "init:")
    mechanism_param_names = [
        str(name)
        for name in (param_names or ())
        if str(name).strip()
        and not str(name).startswith(initial_prefix)
    ]
    # The Fit request realized this regular grid on this span before launch.
    regular_grid = build_time_grid(t_start, t_end, prepared_solver_config.grid)
    objective_times = np.asarray(tuple(objective_sample_times), dtype=float).reshape(-1)
    requested_sample_times = tuple(
        float(value)
        for value in np.unique(
            np.concatenate(
                (
                    regular_grid,
                    objective_times,
                )
            )
        )
    )

    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    # The candidate names are the Fit setup's parameters, enumerated from this
    # prepared mechanism's writable coordinates and independent scalars.
    species_names = tuple(prepared_dsl.accepted_mechanism_for_structure().species_names())
    # Every Fit initial concentration is a candidate or configured value
    # applied per evaluation; the retained request starts from exact zeros.
    y0 = np.zeros(len(species_names), dtype=float)
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    execution_request = SimulationExecutionRequest(
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=tuple(float(value) for value in y0),
        t_span=(float(t_start), float(t_end)),
        solver_config=prepared_solver_config,
        requested_parameter_names=tuple(mechanism_param_names),
        use_authored_stops=False,
    )
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    prepared_run = prepare_simulation_run(
        execution_request=execution_request,
        cancellation_control=cancellation_control,
    )
    return PreparedFittingExecutionContext(
        prepared_run=prepared_run,
        requested_sample_times=requested_sample_times,
        fit_sampling_policy=str(fit_sampling_policy or "fit_internal_regular_grid"),
        fit_sampling_identity=dict(fit_sampling_identity or {}),
        initial_prefix=str(initial_prefix),
    )


class FittingEvaluator:
    """Per-dataset in-process evaluator built from admitted execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
    ) -> None:
        self._context = context
        self._cancellation_control: SimulationCancellationControl | None = None

    def set_cancellation_control(
        self,
        cancellation_control: SimulationCancellationControl | None,
    ) -> "FittingEvaluator":
        self._cancellation_control = cancellation_control
        return self

    def _raise_if_cancel_requested(self) -> None:
        if (
            self._cancellation_control is not None
            and self._cancellation_control.cancel_requested()
        ):
            raise FittingCancelled()

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResultValue | ScientificUnavailable:
        """Evaluate one candidate, returning only natural scientific values.

        Cancellation and fatal causes remain factual exceptions.  Candidate
        domain and solver scientific-unavailability facts are returned
        directly, so no per-series outcome transport is needed.
        """

        try:
            result = self._evaluate_series(
                params,
                origins,
                failed_params=failed_params,
            )
        except (FittingCancelled, SimulationCancelled) as exc:
            raise FittingCancelled() from exc
        except Exception as exc:
            unavailable = _known_candidate_unavailability(
                exc,
                failed_params=failed_params,
            )
            if unavailable is not None:
                return unavailable
            raise
        if isinstance(result, SimulationScientificUnavailability):
            unavailability = result
            unavailable = _scientific_unavailability_from_exception(
                unavailability,
                producer=unavailability.producer,
                stage=unavailability.stage,
                failed_params=failed_params,
            )
            return unavailable
        return result

    def _evaluate_series(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResultValue | SimulationScientificUnavailability:
        self._raise_if_cancel_requested()
        param_map = {
            str(name): raw_value
            for raw_name, raw_value in dict(params or {}).items()
            for name in (str(raw_name),)
            if name.strip()
        }
        origin_map = dict(origins or {})

        initial_overrides: Dict[str, float] = {}
        non_initial_parameters = {
            str(name): value
            for name, value in param_map.items()
            if not str(name).startswith(self._context.initial_prefix)
        }

        non_initial_parameters = {
            name: _coerce_consumed_parameter_value(
                name=name,
                raw_value=value,
                origins=origin_map,
                failed_params=failed_params,
            )
            for name, value in non_initial_parameters.items()
        }
        prepared_request = self._context.prepared_run.request
        try:
            # Every candidate, including the first, receives a fresh isolated
            # request and one canonical preparation.  A rejected candidate can
            # therefore never contaminate the next candidate.
            prepared_run = prepare_simulation_run(
                execution_request=replace(
                    prepared_request,
                    parameter_overrides=non_initial_parameters or None,
                ),
                cancellation_control=self._cancellation_control,
            )
        except SimulationPreparationError as exc:
            if exc.stage == "ordinary_parameter_domain" and exc.parameter_names and all(
                _parameter_origin_for(name, origin_map) in {
                    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
                    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
                }
                for name in exc.parameter_names
            ):
                raise FitCandidateDomainError(
                    str(exc), stage=exc.stage, failed_params=failed_params,
                    details=dict(exc.details),
                ) from exc
            raise fit_simulation_error_from_preparation_error(
                exc,
                failed_params=failed_params,
            ) from exc
        except SimulationCancelled as exc:
            raise FittingCancelled() from exc
        except Exception as exc:
            if isinstance(exc, KineticRuntimeDomainError):
                raise
            if isinstance(exc, AlgebraError):
                raise
            raise _fatal_evaluation_error(
                exc,
                failed_params=failed_params,
                stage="candidate_preparation",
            ) from exc

        symbolic_jacobian = prepared_run.symbolic_jacobian

        species_index = {
            name: idx for idx, name in enumerate(prepared_run.species_names)
        }
        for key, raw_val in param_map.items():
            name = str(key)
            if name.startswith(self._context.initial_prefix):
                species_name = name[len(self._context.initial_prefix) :]
                if species_name not in species_index:
                    continue
                initial_overrides[species_name] = _coerce_consumed_parameter_value(
                    name=name,
                    raw_value=raw_val,
                    origins=origin_map,
                    failed_params=failed_params,
                )

        y0 = np.asarray(prepared_run.y0, dtype=float).copy()
        for species_name, value in initial_overrides.items():
            idx = species_index.get(species_name)
            if idx is None:
                continue
            y0[idx] = float(value)

        # Candidate preparation resolves and compiles every schedule (an empty
        # one included) while its admitted target map is alive.
        intervention_execution = prepared_run.intervention_execution

        symbolic_jacobian = prepared_run.symbolic_jacobian

        request = build_simulation_request_from_prepared_run(
            prepared_run,
            y0=y0,
            intervention_execution=intervention_execution,
            symbolic_jacobian=symbolic_jacobian,
            cancellation_control=self._cancellation_control,
            t_eval=np.asarray(
                self._context.requested_sample_times,
                dtype=float,
            ),
            candidate_evaluation=True,
        )
        try:
            result = solve_ode(request)
        except (PhysicalDomainError, FitSimulationError, KineticRuntimeDomainError):
            raise
        except SimulationCancelled as exc:
            raise FittingCancelled() from exc
        except Exception as exc:
            raise _fatal_evaluation_error(
                exc,
                stage="solver",
                failed_params=failed_params,
            ) from exc
        if isinstance(result, SimulationScientificUnavailability):
            return result
        self._raise_if_cancel_requested()

        initials_map = {
            name: float(y0[index])
            for index, name in enumerate(prepared_run.species_names)
        }
        try:
            admitted = finalize_simulation_result(
                mechanism=prepared_run.mechanism,
                result=result,
                species_names=list(prepared_run.species_names),
                initials_for_algebra=initials_map,
                parameter_values_at_temperature=prepared_run.parameter_values_at_temperature,
                parameter_algebra_spec=prepared_run.parameter_algebra_spec,
                cancellation_control=self._cancellation_control,
                solver_provenance_updates={
                    "fit_sampling_policy": self._context.fit_sampling_policy,
                    "fit_sampling_identity": dict(
                        self._context.fit_sampling_identity
                    ),
                    "fit_final_sampling_times": tuple(
                        float(value)
                        for value in np.asarray(result.t, dtype=float)
                    ),
                },
            )
        except Exception as exc:
            if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from exc
            if isinstance(exc, AlgebraError):
                raise
            raise _fatal_evaluation_error(
                exc,
                stage="result_finalization",
                failed_params=failed_params,
            ) from exc
        self._raise_if_cancel_requested()
        return admitted

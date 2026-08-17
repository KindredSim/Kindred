"""
Shared serial fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds a serial evaluator from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    KindredError,
    KineticRuntimeDomainError,
    SimulationCancelled,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
from kindred.core.analysis import fit_evaluation_outcome as _fit_outcome
from kindred.core.algebra.errors import (
    AlgebraError,
)
from kindred.core.algebra.observable_introspection import AdmittedAlgebraProgram
from kindred.core.algebra.simulation_series import (
    evaluate_compiled_algebra_series_for_simulation,
)
from kindred.core.intervention_schedule_compiler import (
    InterventionExecutionContract,
)
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    SimulationExecutionRequest,
    SimulationExecutionSnapshot,
    fit_simulation_error_from_preparation_error,
    prepare_preparation_failure,
    raise_unowned_request_parameter_values,
    reject_requested_algebra_owned_mechanism_parameters_for_fitting,
    bind_simulation_execution_request_interventions,
    build_simulation_request_from_prepared_run,
    canonicalize_request_parameter_names,
    partition_simulation_parameter_values,
    metadata_view_for_mechanism,
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
    resolve_prepared_run_intervention_execution,
    SimulationPreparationError,
)
from kindred.core.simulation_preparation_stages import PreparedSolverConfig
from kindred.core.simulation_algebra_policy import SimulationAlgebraPolicy
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulation_result_finalization import (
    build_simulation_row_algebra_context,
    finalize_simulation_result,
)
from kindred.core.simulator.solver_types import (
    SimulationCancellationControl,
    SimulationExecutionOutcome,
    SimulationScientificUnavailability,
)
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.executable_identity import compose_target_executable_identity
from kindred.core.time_grid import build_time_grid

logger = logging.getLogger(__name__)

__all__ = [
    "FITTING_PARAM_ORIGIN_CONFIGURED_DATASET",
    "FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET",
    "FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED",
    "PreparedFittingExecutionContext",
    "SerialFittingEvaluator",
    "evaluate_fitting_series",
    "prepare_fitting_execution_context",
]

_FittingSeriesOutcome = _fit_outcome.FittingSeriesOutcome

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
        raise FitSimulationError(
            f"Invalid parameter value for {name!r}: {raw_value!r}",
            failed_params=dict(failed_params or {}) or None,
            details={
                "fatal": True,
                "parameter_origin": _parameter_origin_for(name, origins),
            },
        ) from exc
    if np.isfinite(value):
        return value
    origin = _parameter_origin_for(name, origins)
    raise FitSimulationError(
        f"Non-finite parameter value for {name!r}: {raw_value!r}",
        failed_params=dict(failed_params or {}) or None,
        details={
            "fatal": True,
            "parameter_origin": origin,
        },
    )


def _scientific_unavailability_from_exception(
    exc: object,
    *,
    producer: str,
    stage: str,
    failed_params: Optional[Dict[str, float]],
    value_dependencies: Sequence[str] = (),
    extra_details: Mapping[str, object] | None = None,
) -> ScientificUnavailable:
    details = dict(getattr(exc, "details", None) or {})
    normalized_dependencies = tuple(
        sorted(
            str(name)
            for name in value_dependencies
            if str(name).strip()
        )
    )
    if normalized_dependencies:
        details["value_dependencies"] = normalized_dependencies
    details.update(dict(extra_details or {}))
    return ScientificUnavailable(
        producer=producer,
        stage=stage,
        details={
            **details,
            "failed_params": dict(failed_params or {}),
        },
        factual_cause=getattr(exc, "factual_cause", None),
    )


def _candidate_dependency_unavailability(
    exc: object,
    *,
    origins: Optional[Mapping[str, str]],
    producer: str,
    stage: str,
    failed_params: Optional[Dict[str, float]],
) -> ScientificUnavailable | None:
    details = getattr(exc, "details", None)
    if not isinstance(details, Mapping):
        details = {}
    raw_dependencies = getattr(exc, "value_dependencies", ())
    if not raw_dependencies:
        raw_dependencies = details.get("value_dependencies", ())
    if isinstance(raw_dependencies, str):
        raw_dependencies = (raw_dependencies,)
    dependencies = frozenset(
        str(item).strip()
        for item in raw_dependencies
        if str(item).strip()
    ) if isinstance(raw_dependencies, (list, tuple, set, frozenset)) else frozenset()
    optimizer_names = frozenset(
        str(name).strip()
        for name, origin in (origins or {}).items()
        if str(name).strip()
        and str(origin).strip()
        in {
            FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
            FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
        }
    )
    optimizer_dependencies = dependencies.intersection(optimizer_names)
    if not optimizer_dependencies and not (
        isinstance(exc, SimulationScientificUnavailability) and not dependencies
    ):
        return None
    return _scientific_unavailability_from_exception(
        exc,
        producer=producer,
        stage=stage,
        failed_params=failed_params,
        extra_details={
            "value_dependencies": tuple(sorted(dependencies)),
            "optimizer_dependencies": tuple(sorted(optimizer_dependencies)),
        },
        value_dependencies=dependencies,
    )


def _is_fit_local_algebra_failure(exc: BaseException) -> bool:
    """Identify the admitted Algebra producer without settling dependency policy."""

    if isinstance(exc, AlgebraError):
        return True
    details = getattr(exc, "details", None)
    return isinstance(details, Mapping) and "algebra_statement" in details


def _fatal_evaluation_error(
    exc: BaseException,
    *,
    message: str,
    stage: str,
    failed_params: Optional[Dict[str, float]],
    extra_details: Mapping[str, object] | None = None,
) -> FitSimulationError:
    details = dict(getattr(exc, "details", None) or {})
    details.update(dict(extra_details or {}))
    details.update(
        {
            "fatal": True,
            "stage": str(stage),
        }
    )
    source_message = str(getattr(exc, "message", None) or str(exc)).strip()
    rendered_message = str(message or "").strip()
    if source_message and source_message in rendered_message:
        prefix, separator, _ = rendered_message.partition(":")
        rendered_message = (
            prefix.strip()
            if separator
            else rendered_message.replace(source_message, "").rstrip(" :")
        )
    return FitSimulationError(
        rendered_message or "Fitting evaluation failed.",
        failed_params=failed_params,
        details=details,
        context=getattr(exc, "context", None),
    )


def _strict_algebra_failure_details(
    exc: BaseException,
    *,
    admission: AdmittedAlgebraProgram | None = None,
) -> Mapping[str, object] | None:
    """Attach only the bounded dependency truth retained at Fit admission.

    Algebra's exception type and message identify a producer failure, but they
    do not decide whether the failure is candidate-dependent.  The admitted
    AST does: an exact statement location is resolved to its transitive local
    scalar plan.  Dynamic or incomplete mappings intentionally return no
    dependency truth and therefore remain conservative at the Fit boundary.
    """

    if not isinstance(exc, AlgebraError) or admission is None:
        return None
    line = getattr(exc, "line", None)
    line_text = str(getattr(exc, "line_text", "") or "")
    statements = tuple(admission.compiled.block.lines)
    matches = [
        statement
        for statement in statements
        if (line is not None and int(statement.line) == int(line))
        or (line_text and str(statement.line_text) == line_text)
    ]
    if len(matches) != 1:
        return None
    statement = matches[0]
    plan = {
        str(name): tuple(str(dep) for dep in dependencies)
        for name, dependencies in admission.dependency_plan
    }
    local_scalars = set(admission.scalar_names)
    local_observables = set(admission.observable_names)
    dependencies: set[str] = set()
    pending = list(plan.get(str(statement.name), ()))
    seen: set[str] = set()
    while pending:
        name = str(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        if name in local_scalars:
            dependencies.add(name)
        elif name in local_observables:
            pending.extend(plan.get(name, ()))
        else:
            # The admitted plan is incomplete for this statement.  Do not
            # manufacture a candidate dependency from an unknown AST atom.
            return None
    # A dynamic expression (species/current row or a dynamic local observable)
    # is not an exact local scalar arithmetic failure.
    if statement.expr.has_species_ref() or any(
        name in local_observables for name in seen
    ):
        return None
    return {
        "algebra_code": str(getattr(exc, "code", "") or ""),
        "algebra_location": {
            "line": int(getattr(exc, "line", statement.line)),
            "column": int(getattr(exc, "col", statement.col)),
        },
        "algebra_statement": {
            "name": str(statement.name),
            "source": str(statement.line_text),
        },
        "value_dependencies": tuple(sorted(dependencies)),
    }


class FittingSeriesEvaluator(Protocol):
    """Exact evaluator contract consumed by the Fit optimizer."""

    def evaluate_series_outcome_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> _FittingSeriesOutcome: ...


def evaluate_fitting_series(
    evaluator: FittingSeriesEvaluator,
    params: Mapping[str, float],
    *,
    origins: Optional[Mapping[str, str]] = None,
    failed_params: Optional[Dict[str, float]] = None,
) -> _FittingSeriesOutcome:
    outcome_evaluator = getattr(
        evaluator,
        "evaluate_series_outcome_with_parameter_origins",
        None,
    )
    if not callable(outcome_evaluator):
        raise TypeError(
            "Fit evaluator must implement "
            "evaluate_series_outcome_with_parameter_origins(...)."
        )
    outcome = outcome_evaluator(
        dict(params or {}),
        dict(origins or {}),
        failed_params=failed_params,
    )
    if not isinstance(outcome, _FittingSeriesOutcome):
        raise TypeError("Fit evaluator must return a FittingSeriesOutcome.")
    return outcome


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Typed fitting request, fixed policy, and preparation metadata."""

    execution_request: SimulationExecutionRequest
    algebra_policy: SimulationAlgebraPolicy
    requested_param_names: tuple[str, ...]
    prepared_metadata: PreparedSimulationMetadata
    initial_prefix: str
    requested_sample_times: tuple[float, ...]
    fit_sampling_policy: str = "fit_internal_regular_grid"
    fit_sampling_identity: Mapping[str, object] = field(default_factory=dict)
    fit_local_admission: AdmittedAlgebraProgram | None = None

    def __post_init__(self) -> None:
        execution_request = self.execution_request
        if not isinstance(execution_request, SimulationExecutionRequest):
            raise TypeError(
                "execution_request must be SimulationExecutionRequest."
            )
        from kindred.core.simulation_algebra_policy import (
            ensure_fitting_strict_algebra_policy,
        )

        ensure_fitting_strict_algebra_policy(self.algebra_policy)
        if not isinstance(self.prepared_metadata, PreparedSimulationMetadata):
            raise TypeError(
                "prepared_metadata must be PreparedSimulationMetadata."
            )
        if (
            self.prepared_metadata.solver_config
            != execution_request.solver_config
        ):
            raise ValueError(
                "Fit metadata and execution request must share one solver policy."
            )
        names = tuple(
            dict.fromkeys(
                str(name)
                for name in self.requested_param_names
                if str(name).strip()
            )
        )
        object.__setattr__(self, "execution_request", execution_request)
        object.__setattr__(
            self,
            "algebra_policy",
            SimulationAlgebraPolicy.FITTING_STRICT,
        )
        object.__setattr__(
            self,
            "requested_param_names",
            names,
        )
        object.__setattr__(
            self,
            "initial_prefix",
            str(self.initial_prefix or "init:"),
        )
        sample_times = tuple(float(value) for value in self.requested_sample_times)
        t_start, t_end = execution_request.t_span
        if (
            not sample_times
            or any(not np.isfinite(value) for value in sample_times)
            or any(right <= left for left, right in zip(sample_times, sample_times[1:]))
            or sample_times[0] < float(t_start)
            or sample_times[-1] > float(t_end)
        ):
            raise ValueError(
                "Fitting requested sample times must be finite, strictly "
                "increasing, and inside the execution span."
            )
        object.__setattr__(self, "requested_sample_times", sample_times)
        object.__setattr__(
            self,
            "fit_sampling_policy",
            str(self.fit_sampling_policy or "fit_internal_regular_grid"),
        )
        object.__setattr__(
            self,
            "fit_sampling_identity",
            dict(self.fit_sampling_identity or {}),
        )
        admission = self.fit_local_admission
        if admission is not None and not isinstance(admission, AdmittedAlgebraProgram):
            raise TypeError("Fit-local admission must be a typed value.")

def prepare_fitting_execution_context(
    *,
    prepared_dsl: PreparedDslRequestSet,
    param_names: List[str],
    t_start: float = 0.0,
    t_end: float,
    num_points: int,
    objective_sample_times: Sequence[float],
    prepared_solver_config: PreparedSolverConfig,
    initial_prefix: str = "init:",
    fit_sampling_policy: str = "fit_internal_regular_grid",
    fit_sampling_identity: Mapping[str, object] | None = None,
    fit_local_admission: AdmittedAlgebraProgram | None = None,
    cancellation_control: SimulationCancellationControl | None = None,
) -> PreparedFittingExecutionContext:
    """Prepare structured execution data for serial fitting evaluation."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    if cancellation_control is not None and not isinstance(
        cancellation_control,
        SimulationCancellationControl,
    ):
        raise TypeError(
            "cancellation_control must be a SimulationCancellationControl."
        )
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    t_start = float(t_start)
    t_end = float(t_end)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("Fitting execution requires finite t_end > t_start.")
    if type(num_points) is not int:
        raise TypeError("Fitting num_points must be an integer.")
    if num_points < 2:
        raise ValueError("Fitting num_points must be >= 2.")
    grid_n = num_points
    initial_prefix = str(initial_prefix or "init:")
    if fit_local_admission is not None:
        if not isinstance(fit_local_admission, AdmittedAlgebraProgram):
            raise TypeError("fit_local_admission must be a typed value.")
        # Re-entry is deliberate: namespace and domain drift must reject before
        # Fit setup or solver mutation.  The returned admission is the only
        # request-local Algebra authority used below.
        fit_local_admission = fit_local_admission.reenter(prepared_dsl)
    local_scalar_names = set(
        ()
        if fit_local_admission is None
        else fit_local_admission.scalar_names
    )
    mechanism_param_names = [
        str(name)
        for name in (param_names or ())
        if str(name).strip()
        and not str(name).startswith(initial_prefix)
        and str(name) not in local_scalar_names
    ]
    if not isinstance(prepared_solver_config, PreparedSolverConfig):
        raise TypeError("Fit preparation requires a PreparedSolverConfig.")
    if prepared_solver_config.grid != {"N": grid_n}:
        raise ValueError(
            "Fit prepared solver grid must exactly match requested sampling grid."
        )
    try:
        # Validate and realize the regular N grid before any exact observation
        # union/deduplication; an unrepresentable grid must fail prelaunch.
        build_time_grid(t_start, t_end, prepared_solver_config.grid)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fit regular grid cannot be realized on this time span.") from exc
    temperature_K = float(prepared_dsl.analysis.temperature_K)
    objective_times = np.asarray(tuple(objective_sample_times), dtype=float).reshape(-1)
    if (
        np.any(~np.isfinite(objective_times))
        or np.any(objective_times < t_start)
        or np.any(objective_times > t_end)
    ):
        raise ValueError(
            "Objective observation times must be finite and inside the Fit span."
        )
    requested_sample_times = tuple(
        float(value)
        for value in np.unique(
            np.concatenate(
                (
                    np.linspace(t_start, t_end, grid_n, dtype=float),
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
    bound = prepare_bound_mechanism(
        prepared_dsl,
        mechanism_param_names,
        wegscheider_cyclicity_enabled=(
            prepared_solver_config.wegscheider_cyclicity_enabled
        ),
    )
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    try:
        requested_partition = partition_simulation_parameter_values(
            mechanism=bound.mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=bound.unresolved_intervention_schedule,
            requested_parameter_names=mechanism_param_names,
        )
        raise_unowned_request_parameter_values(requested_partition)
        canonical_requested_param_names = canonicalize_request_parameter_names(
            requested_partition,
            mechanism_param_names,
        )
        reject_requested_algebra_owned_mechanism_parameters_for_fitting(
            bound.mechanism,
            bound.parameter_algebra_spec,
            requested_partition.bindable_mechanism_parameter_names,
        )
    except SimulationPreparationError as exc:
        raise fit_simulation_error_from_preparation_error(exc) from exc
    except ValueError as exc:
        raise fit_simulation_error_from_preparation_error(
            prepare_preparation_failure("parameter_binding", exc)
        ) from exc
    intervention_schedule = bound.unresolved_intervention_schedule
    intervention_execution = bound.intervention_execution
    intervention_schedule_declarative_fingerprint = (
        intervention_execution.declarative_fingerprint
        if intervention_execution is not None
        else intervention_schedule.fingerprint
        if intervention_schedule is not None
        else ""
    )
    intervention_schedule_executable_fingerprint = (
        intervention_execution.executable_fingerprint
        if intervention_execution is not None
        else ""
    )

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256(
            prepared_dsl.analysis.normalized_text.encode("utf-8")
        ).hexdigest(),
        mechanism_text_len=len(prepared_dsl.analysis.normalized_text),
        param_names=sorted(
            {
                *(str(x) for x in canonical_requested_param_names if str(x).strip()),
                *local_scalar_names,
            }
        ),
        t_end=float(t_end),
        num_points=grid_n,
        temperature_K=float(temperature_K),
        solver_config=prepared_solver_config,
        initial_prefix=str(initial_prefix),
        intervention_schedule_declarative_fingerprint=str(intervention_schedule_declarative_fingerprint),
        intervention_schedule_executable_fingerprint=str(intervention_schedule_executable_fingerprint),
    )

    execution_request = SimulationExecutionRequest(
        snapshot=SimulationExecutionSnapshot(
            analysis=bound.dsl_analysis,
            mechanism=bound.mechanism,
            species_names=tuple(bound.species_names),
            y0=tuple(float(value) for value in bound.y0),
            temperature_schedule=metadata_view_for_mechanism(
                bound.mechanism
            ).temperature_schedule,
            parameter_algebra_spec=bound.parameter_algebra_spec,
            intervention_execution=intervention_execution,
            base_executable_identity=bound.base_executable_identity,
            target_executable_identity=(
                None
                if intervention_execution is None
                else compose_target_executable_identity(
                    bound.base_executable_identity,
                    intervention_execution,
                )
            ),
            parameterized_intervention_schedule=intervention_schedule,
            requested_parameter_names=tuple(
                canonical_requested_param_names
            ),
            wegscheider_admission_report=bound.wegscheider_admission_report,
        ),
        t_span=(float(t_start), float(t_end)),
        solver_config=prepared_solver_config,
    )
    execution_request = bind_simulation_execution_request_interventions(
        execution_request
    )
    bound_intervention_execution = (
        execution_request.snapshot.intervention_execution
    )
    if bound_intervention_execution is not None:
        prepared_meta = replace(
            prepared_meta,
            intervention_schedule_declarative_fingerprint=(
                bound_intervention_execution.declarative_fingerprint
            ),
            intervention_schedule_executable_fingerprint=(
                bound_intervention_execution.executable_fingerprint
            ),
        )
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    return PreparedFittingExecutionContext(
        execution_request=execution_request,
        algebra_policy=SimulationAlgebraPolicy.FITTING_STRICT,
        requested_param_names=tuple(
            sorted(
                {
                    *(str(x) for x in canonical_requested_param_names if str(x).strip()),
                    *local_scalar_names,
                }
            )
        ),
        prepared_metadata=prepared_meta,
        initial_prefix=str(initial_prefix),
        requested_sample_times=requested_sample_times,
        fit_sampling_policy=str(fit_sampling_policy or "fit_internal_regular_grid"),
        fit_sampling_identity=dict(fit_sampling_identity or {}),
        fit_local_admission=fit_local_admission,
    )


class SerialFittingEvaluator:
    """Serial in-process evaluator built from structured fitting execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
    ) -> None:
        self._context = replace(
            context,
            execution_request=context.execution_request.isolated_copy(),
        )
        self._cancellation_control: SimulationCancellationControl | None = None

    @property
    def prepared_metadata(self) -> PreparedSimulationMetadata:
        return self._context.prepared_metadata

    def set_cancellation_control(
        self,
        cancellation_control: SimulationCancellationControl | None,
    ) -> "SerialFittingEvaluator":
        if cancellation_control is not None and not isinstance(
            cancellation_control,
            SimulationCancellationControl,
        ):
            raise TypeError(
                "cancellation_control must be a SimulationCancellationControl."
            )
        self._cancellation_control = cancellation_control
        return self

    def _raise_if_cancel_requested(self) -> None:
        if (
            self._cancellation_control is not None
            and self._cancellation_control.cancel_requested()
        ):
            raise FittingCancelled()

    def _raise_unrequested_schedule_candidate_authority(
        self,
        parameter_partition: Any,
        candidate_parameter_names: set[str],
    ) -> None:
        if not candidate_parameter_names:
            return
        schedule_names = {
            str(name)
            for name in getattr(parameter_partition, "schedule_parameter_names", ())
            if str(name).strip()
        }
        if not schedule_names:
            return
        requested_names = {
            str(name)
            for name in (self._context.requested_param_names or ())
            if str(name).strip()
        }
        schedule_name_by_raw = dict(
            getattr(parameter_partition, "schedule_parameter_name_by_raw", {}) or {}
        )
        owner_by_raw = {
            **dict(getattr(parameter_partition, "mechanism_parameter_name_by_raw", {}) or {}),
            **dict(getattr(parameter_partition, "unbound_mechanism_parameter_name_by_raw", {}) or {}),
        }
        schedule_owner_by_raw = {
            raw_name: str(schedule_name_by_raw.get(raw_name, raw_name))
            for raw_name in schedule_names
        }
        for candidate_name in sorted(str(name) for name in candidate_parameter_names if str(name).strip()):
            candidate_owner = str(owner_by_raw.get(candidate_name, candidate_name))
            for schedule_raw_name, schedule_owner in sorted(schedule_owner_by_raw.items()):
                if candidate_name != schedule_raw_name and candidate_owner != schedule_owner:
                    continue
                if (
                    candidate_name in requested_names
                    or candidate_owner in requested_names
                    or schedule_raw_name in requested_names
                    or schedule_owner in requested_names
                ):
                    continue
                raise FitSimulationError(
                    (
                        f"Unrequested intervention schedule parameter {candidate_name!r} "
                        "cannot be used as fitting candidate authority."
                    ),
                    details={"fatal": True, "stage": "parameter_binding"},
                )

    def evaluate_series_outcome_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> _FittingSeriesOutcome:
        """Return the producer-owned per-dataset semantic status."""

        try:
            result = self._evaluate_series(
                params,
                origins,
                failed_params=failed_params,
            )
        except (FittingCancelled, SimulationCancelled) as exc:
            return _FittingSeriesOutcome.cancelled(exc)
        except Exception as exc:
            is_algebra_failure = _is_fit_local_algebra_failure(exc)
            unavailable = _candidate_dependency_unavailability(
                exc,
                origins=origins,
                producer=(
                    "fitting_algebra"
                    if is_algebra_failure
                    else "serial_fitting_evaluator"
                ),
                stage=(
                    "fit_algebra"
                    if is_algebra_failure
                    else "candidate_evaluation"
                ),
                failed_params=failed_params,
            )
            if unavailable is not None:
                return _FittingSeriesOutcome.scientific_unavailable(
                    unavailable,
                    provenance={"stage": "serial_fitting_evaluator"},
                )
            return _FittingSeriesOutcome.fatal(
                exc,
                provenance={"stage": "serial_fitting_evaluator"},
            )
        if isinstance(result, SimulationExecutionOutcome):
            if result.status != "scientific_unavailable":
                raise RuntimeError("Unsupported simulation execution outcome status.")
            unavailability = result.unavailability
            if unavailability is None:
                raise RuntimeError("Simulation outcome omitted its unavailability result.")
            unavailable = _candidate_dependency_unavailability(
                unavailability,
                origins=origins,
                producer=unavailability.producer,
                stage=unavailability.stage,
                failed_params=None,
            )
            if unavailable is None:
                return _FittingSeriesOutcome.fatal(
                    FitSimulationError(
                        "Simulation scientific unavailability depends on fixed or unknown authority.",
                        details=dict(unavailability.details),
                    ),
                    provenance={"stage": "solver"},
                )
            return _FittingSeriesOutcome.scientific_unavailable(
                unavailable,
                provenance={"stage": "solver"},
            )
        if not isinstance(result, SimulationResultValue):
            raise TypeError("Fit evaluator must return a SimulationResultValue.")
        return _FittingSeriesOutcome.available(result)

    def _evaluate_series(
        self,
        params: Mapping[str, float],
        origins: Optional[Mapping[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResultValue | SimulationExecutionOutcome:
        self._raise_if_cancel_requested()
        param_map = {
            str(name): raw_value
            for raw_name, raw_value in dict(params or {}).items()
            for name in (str(raw_name),)
            if name.strip()
        }
        origin_map = dict(origins or {})

        admission = self._context.fit_local_admission
        local_scalar_names = set(
            () if admission is None else admission.scalar_names
        )
        missing_local_scalars = sorted(local_scalar_names - set(param_map))
        if missing_local_scalars:
            raise FitSimulationError(
                "Fit-local evaluation is missing request scalar value(s): "
                + ", ".join(repr(name) for name in missing_local_scalars),
                details={"fatal": True, "stage": "fit_local_evaluation"},
            )
        local_scalar_values = {
            name: _coerce_consumed_parameter_value(
                name=name,
                raw_value=param_map[name],
                origins=origin_map,
                failed_params=failed_params,
            )
            for name in local_scalar_names
        }
        initial_overrides: Dict[str, float] = {}
        candidate_parameter_names = {
            str(name)
            for name in dict(params or {})
            if (
                str(name).strip()
                and not str(name).startswith(self._context.initial_prefix)
                and str(name) not in local_scalar_names
                and _parameter_origin_for(str(name), origin_map)
                in {
                    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
                    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
                }
            )
        }
        non_initial_parameters = {
            str(name): value
            for name, value in param_map.items()
            if not str(name).startswith(self._context.initial_prefix)
            and str(name) not in local_scalar_names
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
        typed_request = self._context.execution_request
        typed_schedule = typed_request.snapshot.parameterized_intervention_schedule
        try:
            preflight_partition = partition_simulation_parameter_values(
                mechanism=typed_request.snapshot.mechanism,
                parameter_overrides=non_initial_parameters,
                unresolved_intervention_schedule=typed_schedule,
                validate_values=False,
            )
            raise_unowned_request_parameter_values(preflight_partition)
            self._raise_unrequested_schedule_candidate_authority(
                preflight_partition,
                candidate_parameter_names,
            )
        except SimulationPreparationError as exc:
            raise fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                failed_params=failed_params,
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc
        try:
            # Every candidate, including the first, receives a fresh isolated
            # request and one canonical preparation.  A rejected candidate can
            # therefore never contaminate the next candidate.
            prepared_run = prepare_simulation_worker_run(
                execution_request=replace(
                    typed_request,
                    parameter_overrides=non_initial_parameters or None,
                ),
                cancellation_control=self._cancellation_control,
            )
        except SimulationPreparationError as exc:
            preparation_cause = exc.__cause__ or exc
            algebra_details = _strict_algebra_failure_details(preparation_cause)
            if algebra_details is not None:
                raise preparation_cause from exc
            raise fit_simulation_error_from_preparation_error(
                exc,
                failed_params=failed_params,
            ) from exc
        except SimulationCancelled as exc:
            raise FittingCancelled() from exc
        except Exception as exc:
            if isinstance(exc, KineticRuntimeDomainError):
                raise
            algebra_details = _strict_algebra_failure_details(exc)
            if algebra_details is not None:
                raise
            raise _fatal_evaluation_error(
                exc,
                message=f"Structured fitting candidate preparation failed: {exc}",
                failed_params=failed_params,
                stage="candidate_preparation",
            ) from exc

        symbolic_jacobian = prepared_run.symbolic_jacobian
        metadata_updates: dict[str, object] = {}
        if symbolic_jacobian.status or symbolic_jacobian.identity:
            metadata_updates.update(symbolic_jacobian.metadata_kwargs())
        symbolic_wegscheider_identity = prepared_run.symbolic_wegscheider_identity
        if (
            isinstance(symbolic_wegscheider_identity, Mapping)
            and symbolic_wegscheider_identity
        ):
            metadata_updates[
                "symbolic_wegscheider_identity"
            ] = dict(symbolic_wegscheider_identity)
        prepared_intervention = prepared_run.intervention_execution
        if isinstance(prepared_intervention, InterventionExecutionContract):
            metadata_updates.update(
                {
                    "intervention_schedule_declarative_fingerprint": (
                        prepared_intervention.declarative_fingerprint
                    ),
                    "intervention_schedule_executable_fingerprint": (
                        prepared_intervention.executable_fingerprint
                    ),
                }
            )
        if metadata_updates:
            self._context = replace(
                self._context,
                prepared_metadata=replace(
                    self._context.prepared_metadata,
                    **metadata_updates,
                ),
            )

        species_index = {
            name: idx for idx, name in enumerate(prepared_run.species_names)
        }
        try:
            parameter_partition = partition_simulation_parameter_values(
                mechanism=prepared_run.mechanism,
                parameter_overrides=non_initial_parameters,
                unresolved_intervention_schedule=typed_schedule,
                validate_values=False,
            )
            raise_unowned_request_parameter_values(parameter_partition)
            self._raise_unrequested_schedule_candidate_authority(
                parameter_partition,
                candidate_parameter_names,
            )
        except SimulationPreparationError as exc:
            raise fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                failed_params=failed_params,
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc

        requested_parameter_names = set(self._context.requested_param_names or ())
        consumed_parameter_names = {
            *parameter_partition.mechanism_parameter_names,
            *parameter_partition.unbound_mechanism_parameter_names,
            *parameter_partition.scalar_parameter_names,
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
            elif name not in local_scalar_names:
                binding_name = parameter_partition.mechanism_parameter_name_by_raw.get(name, name)
                if binding_name not in consumed_parameter_names:
                    if name in parameter_partition.schedule_only_parameter_names:
                        if name not in requested_parameter_names:
                            raise FitSimulationError(
                                (
                                    f"Unrequested intervention schedule parameter {name!r} "
                                    "cannot be used as fitting candidate authority."
                                ),
                                details={"fatal": True, "stage": "parameter_binding"},
                            )
                        _coerce_consumed_parameter_value(
                            name=name,
                            raw_value=raw_val,
                            origins=origin_map,
                            failed_params=failed_params,
                        )
                        continue
                    if name in parameter_partition.invalid_parameter_identifier_messages:
                        raise FitSimulationError(
                            parameter_partition.invalid_parameter_identifier_messages[name],
                            details={"fatal": True, "stage": "parameter_binding"},
                        )
                    if name in set(self._context.requested_param_names or ()):
                        raise FitSimulationError(
                            (
                                f"Requested fitting parameter {name!r} is unavailable after preparation; "
                                "it may be algebra-derived or unsupported as a fitted dimension."
                            ),
                            details={"fatal": True, "stage": "parameter_binding"},
                        )
                    continue
                _coerce_consumed_parameter_value(
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

        intervention_execution = prepared_run.intervention_execution
        if intervention_execution is None:
            try:
                intervention_execution = resolve_prepared_run_intervention_execution(
                    prepared_run,
                    parameter_partition,
                    cancellation_check=(
                        self._cancellation_control.cancel_requested
                        if self._cancellation_control is not None
                        else None
                    ),
                )
            except SimulationPreparationError as exc:
                raise fit_simulation_error_from_preparation_error(
                    exc,
                    failed_params=failed_params,
                ) from exc

        symbolic_jacobian = prepared_run.symbolic_jacobian

        request = build_simulation_request_from_prepared_run(
            prepared_run,
            y0=y0,
            intervention_execution=intervention_execution,
            symbolic_jacobian=symbolic_jacobian,
            cancellation_control=self._cancellation_control,
        )
        request = replace(
            request,
            t_eval=np.asarray(
                self._context.requested_sample_times,
                dtype=float,
            ),
            candidate_evaluation=True,
        )
        try:
            result = solve_ode(request)
        except FitSimulationError:
            raise
        except SimulationCancelled as exc:
            raise FittingCancelled() from exc
        except KindredError as exc:
            if isinstance(exc, KineticRuntimeDomainError):
                raise
            raise _fatal_evaluation_error(
                exc,
                message=f"Fitting simulation failed: {exc}",
                stage="solver",
                failed_params=failed_params,
            ) from exc
        except Exception as exc:
            raise _fatal_evaluation_error(
                exc,
                message=f"Fitting simulation failed: {exc}",
                stage="solver",
                failed_params=failed_params,
            ) from exc
        if isinstance(result, SimulationExecutionOutcome):
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
                temperature_schedule=prepared_run.temperature_schedule,
                parameter_algebra_spec=prepared_run.parameter_algebra_spec,
                algebra_policy=self._context.algebra_policy,
            )
        except Exception as exc:
            if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from exc
            algebra_details = _strict_algebra_failure_details(exc)
            if algebra_details is not None:
                raise
            raise _fatal_evaluation_error(
                exc,
                message=f"Fitting result finalization failed: {exc}",
                stage="result_finalization",
                failed_params=failed_params,
                extra_details=algebra_details,
            ) from exc
        final_solver_provenance = dict(admitted.provenance.solver)
        final_solver_provenance.update(
            {
                "fit_sampling_policy": self._context.fit_sampling_policy,
                "fit_sampling_identity": dict(
                    self._context.fit_sampling_identity
                ),
                "fit_final_sampling_times": tuple(
                    float(value) for value in np.asarray(admitted.t, dtype=float)
                ),
            }
        )
        admitted = replace(
            admitted,
            provenance=replace(
                admitted.provenance,
                solver=final_solver_provenance,
            ),
        )
        if admission is not None:
            self._raise_if_cancel_requested()
            try:
                symbol_series, symbol_values = build_simulation_row_algebra_context(
                    prepared_run.mechanism,
                    np.asarray(admitted.t, dtype=float),
                    parameter_algebra_spec=prepared_run.parameter_algebra_spec,
                    temperature_schedule=prepared_run.temperature_schedule,
                )
                symbol_series.update(
                    {
                        str(name): np.asarray(values, dtype=float)
                        for name, values in admitted.series.items()
                    }
                )
                symbol_values.update(
                    {str(name): float(value) for name, value in admitted.scalars.items()}
                )
                symbol_values.update(local_scalar_values)
                species_series = {
                    str(name): np.asarray(admitted.series[name], dtype=float)
                    for name in prepared_run.species_names
                }
                local_series, local_scalars = (
                    evaluate_compiled_algebra_series_for_simulation(
                        prepared_run.mechanism,
                        admission.compiled,
                        t=np.asarray(admitted.t, dtype=float),
                        species_series=species_series,
                        initials=initials_map,
                        temperature_K=float(
                            metadata_view_for_mechanism(
                                prepared_run.mechanism
                            ).temperature_K
                        ),
                        symbol_series=symbol_series,
                        symbol_values=symbol_values,
                    )
                )
            except Exception as exc:
                if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                    raise FittingCancelled() from exc
                details = _strict_algebra_failure_details(
                    exc,
                    admission=admission,
                )
                if details is not None:
                    raise _fatal_evaluation_error(
                        exc,
                        message=f"Fit-local observable evaluation failed: {exc}",
                        stage="fit_local_evaluation",
                        failed_params=failed_params,
                        extra_details=details,
                    ) from exc
                raise _fatal_evaluation_error(
                    exc,
                    message=f"Fit-local observable evaluation failed: {exc}",
                    stage="fit_local_evaluation",
                    failed_params=failed_params,
                ) from exc
            local_names = tuple(admission.observable_names)
            solver_provenance = dict(admitted.provenance.solver)
            solver_provenance["fit_local_admission_identity"] = admission.identity
            admitted = replace(
                admitted,
                series={**dict(admitted.series), **local_series},
                scalars={**dict(admitted.scalars), **local_scalars},
                provenance=replace(
                    admitted.provenance,
                    solver=solver_provenance,
                    requested_algebra_outputs=(
                        *admitted.provenance.requested_algebra_outputs,
                        *local_names,
                    ),
                    succeeded_algebra_outputs=(
                        *admitted.provenance.succeeded_algebra_outputs,
                        *local_names,
                    ),
                ),
            )
        segment_ids = np.asarray(admitted.segment_ids, dtype=np.int64)
        segment_keys = tuple(admitted.segment_keys)
        discontinuity_ids = np.asarray(
            admitted.discontinuity_ids,
            dtype=np.int64,
        )
        times = np.asarray(admitted.t, dtype=float)
        result_execution = admitted.intervention_execution
        if not isinstance(result_execution, InterventionExecutionContract):
            raise FitSimulationError(
                "Fitting result is missing its intervention execution contract.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if (
            result_execution.with_trigger_firings(())
            != intervention_execution.with_trigger_firings(())
        ):
            raise FitSimulationError(
                "Fitting result intervention execution conflicts with the "
                "resolved request contract.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if not (
            discontinuity_ids.size
            == len(segment_keys)
            == segment_ids.size
            == times.size
        ):
            raise FitSimulationError(
                "Fitting result lost aligned segment/discontinuity identity.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        if np.any(np.diff(segment_ids) < 0) or np.any(np.diff(discontinuity_ids) < 0):
            raise FitSimulationError(
                "Fitting result segment identity is not ordered.",
                details={"fatal": True, "stage": "intervention_execution"},
            )
        key_by_segment_id: dict[int, str] = {}
        for segment_id, segment_key in zip(
            segment_ids,
            segment_keys,
            strict=True,
        ):
            normalized_key = str(segment_key or "").strip()
            existing = key_by_segment_id.setdefault(
                int(segment_id),
                normalized_key,
            )
            if not normalized_key or existing != normalized_key:
                raise FitSimulationError(
                    "Fitting result segment identity is internally "
                    "inconsistent.",
                    details={
                        "fatal": True,
                        "stage": "intervention_execution",
                    },
                )
        self._raise_if_cancel_requested()
        return admitted

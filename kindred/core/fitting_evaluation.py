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
    KindredError,
    KineticRuntimeDomainError,
    SimulationCancelled,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
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
from kindred.core.intervention_schedule import coerce_intervention_schedule
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    PreparedSimulationRun,
    _canonical_step_override_name,
    _classify_request_parameter_names,
    _parameter_algebra_scalar_authority_names,
    _schedule_resolution_values_for_request,
    fit_simulation_error_from_preparation_error,
    prepare_preparation_failure,
    raise_unowned_request_parameter_values,
    reject_requested_algebra_owned_mechanism_parameters_for_fitting,
    bind_simulation_execution_request_interventions,
    build_simulation_request_from_prepared_run,
    metadata_view_for_mechanism,
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
    SimulationScientificUnavailability,
)
from kindred.core.simulator.solvers import solve_ode
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism
from kindred.core.time_grid import build_time_grid

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
        f"Non-finite parameter value for {name!r}: {raw_value!r}",
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

    Only the explicit candidate-domain exception owns candidate rejection here.
    Preparation, Algebra, kinetic, and wrapped failures remain fatal; the
    solver's typed ``SimulationScientificUnavailability`` is settled separately
    below.
    """

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
    message: str,
    stage: str,
    failed_params: Optional[Dict[str, float]],
) -> FitSimulationError:
    details = dict(getattr(exc, "details", None) or {})
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


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Typed fitting request, fixed policy, and preparation metadata."""

    prepared_run: PreparedSimulationRun
    algebra_policy: SimulationAlgebraPolicy
    requested_sample_times: tuple[float, ...]
    fit_sampling_policy: str = "fit_internal_regular_grid"
    fit_sampling_identity: Mapping[str, object] = field(default_factory=dict)
    fit_local_admission: AdmittedAlgebraProgram | None = None

    def __post_init__(self) -> None:
        prepared_run = self.prepared_run
        if not isinstance(prepared_run, PreparedSimulationRun):
            raise TypeError(
                "prepared_run must be PreparedSimulationRun."
            )
        from kindred.core.simulation_algebra_policy import (
            ensure_fitting_strict_algebra_policy,
        )

        ensure_fitting_strict_algebra_policy(self.algebra_policy)
        object.__setattr__(self, "prepared_run", prepared_run)
        object.__setattr__(
            self,
            "algebra_policy",
            SimulationAlgebraPolicy.FITTING_STRICT,
        )
        sample_times = tuple(float(value) for value in self.requested_sample_times)
        request = prepared_run.request
        t_start, t_end = request.t_span
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
    """Prepare structured execution data for one per-dataset evaluator."""

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
    mechanism = prepared_dsl.accepted_mechanism_for_structure()
    unresolved_intervention_schedule = coerce_intervention_schedule(
        prepared_dsl.parameterized_intervention_schedule_for_binding()
    )
    parameter_algebra_spec = prepared_dsl.parameter_algebra_spec_for_execution()
    species_names = tuple(mechanism.species_names())
    y0 = np.asarray([mechanism.species[sp].initial_conc for sp in species_names], dtype=float)
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    try:
        mechanism_namespace = build_namespace_from_mechanism(mechanism)
        namespace_info = mechanism_namespace.info_by_name
        declared_scalar_names, derived_scalar_names = _parameter_algebra_scalar_authority_names(
            mechanism, parameter_algebra_spec
        )
        requested_derived_scalar_names = sorted(
            name for name in mechanism_param_names if str(name) in derived_scalar_names
        )
        if requested_derived_scalar_names:
            raise SimulationPreparationError(
                "parameter_binding",
                "derived scalar parameter(s) cannot be used as fitting write authority: "
                + ", ".join(repr(name) for name in requested_derived_scalar_names),
            )
        raw_values: dict[str, float] = {}
        mechanism_names: set[str] = set()
        unbound_mechanism_names: set[str] = set()
        schedule_names: set[str] = set()
        scalar_names: set[str] = set()
        unknown_names: set[str] = set()
        mechanism_name_by_raw: dict[str, str] = {}
        unbound_mechanism_name_by_raw: dict[str, str] = {}
        schedule_name_by_raw: dict[str, str] = {}
        invalid_messages: dict[str, str] = {}
        _classify_request_parameter_names(
            mechanism=mechanism,
            parameter_overrides=None,
            unresolved_intervention_schedule=unresolved_intervention_schedule,
            raw_values=raw_values,
            mechanism_names=mechanism_names,
            unbound_mechanism_names=unbound_mechanism_names,
            schedule_names=schedule_names,
            scalar_names=scalar_names,
            unknown_names=unknown_names,
            mechanism_name_by_raw=mechanism_name_by_raw,
            unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
            schedule_name_by_raw=schedule_name_by_raw,
            invalid_messages=invalid_messages,
            requested_parameter_names=mechanism_param_names,
            declared_scalar_names=declared_scalar_names,
        )
        raise_unowned_request_parameter_values(
            raw_values=raw_values,
            mechanism_name_by_raw=mechanism_name_by_raw,
            unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
            unbound_mechanism_names=unbound_mechanism_names,
            unknown_names=unknown_names,
            invalid_messages=invalid_messages,
            allow_unbound_mechanism_parameters=True,
        )
        canonical_by_raw = {**mechanism_name_by_raw, **unbound_mechanism_name_by_raw}
        canonical_requested_param_names: list[str] = []
        owner_to_raw: dict[str, str] = {}
        for raw_name in mechanism_param_names:
            raw = str(raw_name or "").strip()
            if not raw:
                continue
            owner = str(canonical_by_raw.get(raw, raw))
            previous = owner_to_raw.get(owner)
            if previous is not None and previous != raw:
                raise ValueError(
                    f"Duplicate request parameter names {previous!r} and {raw!r} resolve to {owner!r}."
                )
            owner_to_raw[owner] = raw
            canonical_requested_param_names.append(owner)
        reject_requested_algebra_owned_mechanism_parameters_for_fitting(
            mechanism,
            parameter_algebra_spec,
            set(mechanism_names) | set(unbound_mechanism_names),
        )
        constrained = {
            str(stmt.name)
            for stmt in (parameter_algebra_spec.param_statements or [])
            if str(stmt.name) in namespace_info
        }
        constrained_keq_targets = {
            name for name in constrained if namespace_info[name].role == "Keq"
        }
        requested = {
            _canonical_step_override_name(mechanism, str(name))
            for name in (set(mechanism_names) | set(unbound_mechanism_names))
            if str(name).strip()
        }
        requested_dependent_keq = sorted(requested & constrained_keq_targets)
        if requested_dependent_keq:
            raise SimulationPreparationError(
                "parameter_algebra",
                "Dependent equilibrium parameter(s) cannot be fitted directly because "
                "Algebra constraints overwrite them: "
                + ", ".join(requested_dependent_keq),
            )
    except SimulationPreparationError as exc:
        raise fit_simulation_error_from_preparation_error(exc) from exc
    except ValueError as exc:
        raise fit_simulation_error_from_preparation_error(
            prepare_preparation_failure("parameter_binding", exc)
        ) from exc
    execution_request = SimulationExecutionRequest(
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=tuple(float(value) for value in y0),
        t_span=(float(t_start), float(t_end)),
        solver_config=prepared_solver_config,
        requested_parameter_names=tuple(canonical_requested_param_names),
    )
    execution_request = bind_simulation_execution_request_interventions(
        execution_request
    )
    if (
        cancellation_control is not None
        and cancellation_control.cancel_requested()
    ):
        raise FittingCancelled()
    prepared_dsl.ensure_scientific_structures()
    prepared_run = prepare_simulation_worker_run(
        execution_request=execution_request,
        cancellation_control=cancellation_control,
    )
    prepared_run = replace(
        prepared_run,
        version=1,
        mechanism_text_len=len(prepared_dsl.analysis.normalized_text),
        param_names=tuple(sorted({*(str(x) for x in canonical_requested_param_names if str(x).strip()), *local_scalar_names})),
        t_end=float(t_end),
        num_points=grid_n,
        temperature_K=float(temperature_K),
        initial_prefix=str(initial_prefix),
        symbolic_jacobian_status=prepared_run.symbolic_jacobian.status,
    )
    return PreparedFittingExecutionContext(
        prepared_run=prepared_run,
        algebra_policy=SimulationAlgebraPolicy.FITTING_STRICT,
        requested_sample_times=requested_sample_times,
        fit_sampling_policy=str(fit_sampling_policy or "fit_internal_regular_grid"),
        fit_sampling_identity=dict(fit_sampling_identity or {}),
        fit_local_admission=fit_local_admission,
    )


class FittingEvaluator:
    """Per-dataset in-process evaluator built from admitted execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
    ) -> None:
        self._context = context
        self._cancellation_control: SimulationCancellationControl | None = None

    @property
    def prepared_metadata(self) -> PreparedSimulationRun:
        return self._context.prepared_run

    def set_cancellation_control(
        self,
        cancellation_control: SimulationCancellationControl | None,
    ) -> "FittingEvaluator":
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
        if not isinstance(result, SimulationResultValue):
            raise TypeError("Fit evaluator must return a SimulationResultValue.")
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
        non_initial_parameters = {
            str(name): value
            for name, value in param_map.items()
            if not str(name).startswith(self._context.prepared_run.initial_prefix)
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
        prepared_request = self._context.prepared_run.request
        typed_schedule = prepared_request.prepared_dsl.parameterized_intervention_schedule_for_binding()
        try:
            # Every candidate, including the first, receives a fresh isolated
            # request and one canonical preparation.  A rejected candidate can
            # therefore never contaminate the next candidate.
            prepared_run = prepare_simulation_worker_run(
                execution_request=replace(
                    prepared_request,
                    parameter_overrides=non_initial_parameters or None,
                ),
                cancellation_control=self._cancellation_control,
            )
        except SimulationPreparationError as exc:
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
                message=f"Structured fitting candidate preparation failed: {exc}",
                failed_params=failed_params,
                stage="candidate_preparation",
            ) from exc

        symbolic_jacobian = prepared_run.symbolic_jacobian
        metadata_updates: dict[str, object] = {}
        if symbolic_jacobian.status:
            metadata_updates.update(symbolic_jacobian.metadata_kwargs())
        if metadata_updates:
            self._context = replace(
                self._context,
                prepared_run=replace(self._context.prepared_run, **metadata_updates),
            )

        species_index = {
            name: idx for idx, name in enumerate(prepared_run.species_names)
        }
        try:
            raw_values: dict[str, float] = {}
            mechanism_names: set[str] = set()
            unbound_mechanism_names: set[str] = set()
            schedule_names: set[str] = set()
            scalar_names: set[str] = set()
            unknown_names: set[str] = set()
            mechanism_name_by_raw: dict[str, str] = {}
            unbound_mechanism_name_by_raw: dict[str, str] = {}
            schedule_name_by_raw: dict[str, str] = {}
            invalid_messages: dict[str, str] = {}
            _classify_request_parameter_names(
                mechanism=prepared_run.mechanism,
                parameter_overrides=non_initial_parameters,
                unresolved_intervention_schedule=typed_schedule,
                raw_values=raw_values,
                mechanism_names=mechanism_names,
                unbound_mechanism_names=unbound_mechanism_names,
                schedule_names=schedule_names,
                scalar_names=scalar_names,
                unknown_names=unknown_names,
                mechanism_name_by_raw=mechanism_name_by_raw,
                unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
                schedule_name_by_raw=schedule_name_by_raw,
                invalid_messages=invalid_messages,
                validate_values=False,
            )
            raise_unowned_request_parameter_values(
                raw_values=raw_values,
                mechanism_name_by_raw=mechanism_name_by_raw,
                unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
                unbound_mechanism_names=unbound_mechanism_names,
                unknown_names=unknown_names,
                invalid_messages=invalid_messages,
            )
        except SimulationPreparationError as exc:
            raise fit_simulation_error_from_preparation_error(exc) from exc
        except ValueError as exc:
            raise FitSimulationError(
                str(exc),
                failed_params=failed_params,
                details={"fatal": True, "stage": "parameter_binding"},
            ) from exc

        requested_parameter_names = set(self._context.prepared_run.param_names or ())
        consumed_parameter_names = {
            *mechanism_names,
            *unbound_mechanism_names,
            *scalar_names,
        }
        schedule_only_parameter_names = {
            name
            for name in schedule_names
            if schedule_name_by_raw.get(name, name) not in mechanism_names
            and schedule_name_by_raw.get(name, name) not in unbound_mechanism_names
            and name not in scalar_names
        }
        for key, raw_val in param_map.items():
            name = str(key)
            if name.startswith(self._context.prepared_run.initial_prefix):
                species_name = name[len(self._context.prepared_run.initial_prefix) :]
                if species_name not in species_index:
                    continue
                initial_overrides[species_name] = _coerce_consumed_parameter_value(
                    name=name,
                    raw_value=raw_val,
                    origins=origin_map,
                    failed_params=failed_params,
                )
            elif name not in local_scalar_names:
                binding_name = mechanism_name_by_raw.get(
                    name,
                    unbound_mechanism_name_by_raw.get(name, name),
                )
                if binding_name not in consumed_parameter_names:
                    if name in schedule_only_parameter_names:
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
                    if name in invalid_messages:
                        raise FitSimulationError(
                            invalid_messages[name],
                            details={"fatal": True, "stage": "parameter_binding"},
                        )
                    if name in set(self._context.prepared_run.param_names or ()):
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
                    parameter_values=_schedule_resolution_values_for_request(
                        typed_schedule,
                        raw_values=raw_values,
                        mechanism_name_by_raw=mechanism_name_by_raw,
                        unbound_mechanism_name_by_raw=unbound_mechanism_name_by_raw,
                        schedule_name_by_raw=schedule_name_by_raw,
                        mechanism=prepared_run.mechanism,
                    ) or {},
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
                temperature_schedule=prepared_run.temperature_schedule,
                parameter_algebra_spec=prepared_run.parameter_algebra_spec,
                algebra_policy=self._context.algebra_policy,
            )
        except Exception as exc:
            if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from exc
            if isinstance(exc, AlgebraError):
                raise
            raise _fatal_evaluation_error(
                exc,
                message=f"Fitting result finalization failed: {exc}",
                stage="result_finalization",
                failed_params=failed_params,
            ) from exc
        final_solver_provenance = dict(admitted.solver_provenance)
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
            solver_provenance=final_solver_provenance,
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
                if isinstance(exc, AlgebraError):
                    raise
                raise _fatal_evaluation_error(
                    exc,
                    message=f"Fit-local observable evaluation failed: {exc}",
                    stage="fit_local_evaluation",
                    failed_params=failed_params,
                ) from exc
            local_names = tuple(admission.observable_names)
            solver_provenance = dict(admitted.solver_provenance)
            admitted = replace(
                admitted,
                series={**dict(admitted.series), **local_series},
                scalars={**dict(admitted.scalars), **local_scalars},
                solver_provenance=solver_provenance,
                requested_algebra_outputs=(
                        *admitted.requested_algebra_outputs,
                        *local_names,
                ),
                succeeded_algebra_outputs=(
                        *admitted.succeeded_algebra_outputs,
                        *local_names,
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

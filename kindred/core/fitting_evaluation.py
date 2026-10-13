"""
Shared per-dataset fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds one evaluator per dataset from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Mapping, Sequence

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
from kindred.core.algebra.errors import UNCOMPUTABLE_ERRORS
from kindred.core.algebra.simulation_series import observable_closure
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
from kindred.core.simulation_failure import FactualCause, factual_cause_from_exception

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
_OPTIMIZER_ORIGINS = frozenset({
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
})


def _scientific_unavailability_from_exception(
    exc: object,
    *,
    producer: str,
    stage: str,
) -> ScientificUnavailable:
    return ScientificUnavailable(
        producer=producer,
        stage=stage,
        factual_cause=FactualCause(
            domain="simulation", origin=producer, code=None,
            type="SimulationScientificUnavailability", message=exc.message, details=exc.details,
        ),
    )


def _known_candidate_unavailability(
    exc: BaseException,
) -> ScientificUnavailable | None:
    """Translate producer-owned typed candidate failures at this boundary.

    Explicit candidate-domain and physical-predicate failures reject candidates.
    Calculated values outside their domain are decided from their recorded
    cause by ``_candidate_domain_rejection``.  Other preparation, Algebra, and
    wrapped failures remain fatal; the solver's typed
    ``SimulationScientificUnavailability`` is settled separately.
    """

    if isinstance(exc, PhysicalDomainError):
        return ScientificUnavailable(
            producer="physical_intervention", stage="execution",
            factual_cause=factual_cause_from_exception(exc),
        )
    if isinstance(exc, FitCandidateDomainError):
        return ScientificUnavailable(
            producer="fit_candidate", stage=exc.details["stage"],
            factual_cause=factual_cause_from_exception(exc),
        )
    return None


# A calculated value, param or let, outside its domain or not computable.
_DOMAIN_CAUSES = frozenset({
    "KineticRuntimeDomainError", *(error_type.__name__ for error_type in UNCOMPUTABLE_ERRORS),
})


def _candidate_domain_rejection(
    cause: FactualCause,
    origins: Mapping[str, str],
) -> ScientificUnavailable | None:
    """Reject the candidate when a calculated value outside its domain depends
    on an optimizer value; one fed only by configured values stays fatal."""
    # A let that reads a concentration depends on every value the candidate
    # varies, as the solved curve does.
    if cause.type in _DOMAIN_CAUSES and any(
        origins.get(name) in _OPTIMIZER_ORIGINS
        for name in (origins if cause.details.get("reads_species") else cause.details["parameter_names"])
    ):
        return ScientificUnavailable(
            producer="fit_candidate", stage=cause.details["stage"],
            factual_cause=cause,
        )
    return None


def _fatal_evaluation_error(
    exc: BaseException,
    *,
    stage: str,
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
        details=details,
        context=getattr(exc, "context", None),
    )


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Typed fitting request and preparation metadata.

    Its sole producer unions the admitted regular grid with the admitted
    observation times inside the Fit span.  A measured-X dataset has no
    observation times; its curve also takes the solver's accepted steps.
    """

    prepared_run: PreparedSimulationRun
    requested_sample_times: tuple[float, ...]
    sample_solver_steps: bool
    initial_prefix: str = "init:"


def prepare_fitting_execution_context(
    *,
    prepared_dsl: PreparedDslRequestSet,
    param_names: List[str],
    t_start: float = 0.0,
    t_end: float,
    objective_sample_times: Sequence[float],
    prepared_solver_config: PreparedSolverConfig,
    sample_solver_steps: bool,
    initial_prefix: str = "init:",
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
        requested_parameter_names=tuple(param_names),
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
        sample_solver_steps=sample_solver_steps,
        initial_prefix=initial_prefix,
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
        origins: Mapping[str, str],
        outputs: tuple[str, ...],
    ) -> SimulationResultValue | ScientificUnavailable:
        """Evaluate one candidate, returning only natural scientific values.

        Cancellation and fatal causes remain factual exceptions.  Candidate
        domain and solver scientific-unavailability facts are returned
        directly, so no per-series outcome transport is needed.  Of the
        Algebra outputs, only the dataset's ``outputs`` and the lets they
        read can reject the candidate; other failures stay diagnostics.
        """

        try:
            result = self._evaluate_series(
                params,
                origins,
            )
        except (FittingCancelled, SimulationCancelled) as exc:
            raise FittingCancelled() from exc
        except KineticRuntimeDomainError as exc:
            cause = self._with_fit_values(factual_cause_from_exception(exc), params)
            rejection = _candidate_domain_rejection(cause, origins)
            if rejection is not None:
                return rejection
            exc.factual_cause = cause
            raise
        except Exception as exc:
            unavailable = _known_candidate_unavailability(
                exc,
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
            )
            return unavailable
        # A let, or a value it reads, that fails records its own cause.
        result = replace(result, algebra_errors=tuple(
            self._with_fit_values(cause, params) for cause in result.algebra_errors
        ))
        # A selected output that was calculated is final; one that failed is
        # judged by the failures among the lets it reads. One undefined at some
        # time points only is judged where data compares with it.
        rejection = self.failure_rejection(
            result, tuple(name for name in outputs if name in result.failed_algebra_outputs), origins,
        )
        return result if rejection is None else rejection

    def failure_rejection(
        self,
        result: SimulationResultValue,
        outputs: tuple[str, ...],
        origins: Mapping[str, str],
    ) -> ScientificUnavailable | None:
        """Reject the candidate for the first failure, among the lets the
        outputs read, that depends on an optimizer value; None leaves it fatal."""
        used = observable_closure(
            self._context.prepared_run.parameter_algebra_spec.observable_block, outputs,
        )
        for cause in result.algebra_errors:
            if cause.details["name"] not in used:
                continue
            rejection = _candidate_domain_rejection(cause, origins)
            if rejection is not None:
                return rejection
        return None

    def _with_fit_values(self, cause: FactualCause, params: Mapping[str, float]) -> FactualCause:
        """Name the Fit's own values for the failing value's inputs, which can
        differ from the numbers on the authored line the cause quotes."""
        names = [name for name in cause.details.get("parameter_names", ()) if name in params]
        if not names:
            return cause
        units = dict(self._context.prepared_run.parameter_algebra_spec.input_units)
        values = ", ".join(
            f"{name} = {params[name]:.6g}" + ("" if units[name].symbol == "1" else f" {units[name].symbol}")
            for name in names
        )
        return replace(cause, message=f"{cause.message} Fit values: {values}.")

    def _evaluate_series(
        self,
        params: Mapping[str, float],
        origins: Mapping[str, str],
    ) -> SimulationResultValue | SimulationScientificUnavailability:
        self._raise_if_cancel_requested()
        prefix = self._context.initial_prefix
        non_initial_parameters = {
            name: value for name, value in params.items() if not name.startswith(prefix)
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
                origins[name] in _OPTIMIZER_ORIGINS for name in exc.parameter_names
            ):
                raise FitCandidateDomainError(
                    str(exc), stage=exc.stage,
                    details=dict(exc.details),
                ) from exc
            raise fit_simulation_error_from_preparation_error(
                exc,
            ) from exc
        except SimulationCancelled as exc:
            raise FittingCancelled() from exc
        except KineticRuntimeDomainError:
            raise
        except Exception as exc:
            raise _fatal_evaluation_error(
                exc,
                stage="candidate_preparation",
            ) from exc

        species_index = {
            name: idx for idx, name in enumerate(prepared_run.species_names)
        }
        y0 = np.asarray(prepared_run.y0, dtype=float).copy()
        for name, value in params.items():
            if name.startswith(prefix):
                y0[species_index[name[len(prefix):]]] = value

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
            sample_solver_steps=self._context.sample_solver_steps,
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
                result=result,
                species_names=list(prepared_run.species_names),
                initials_for_algebra=initials_map,
                parameter_values_at_temperature=prepared_run.parameter_values_at_temperature,
                parameter_algebra_spec=prepared_run.parameter_algebra_spec,
                cancellation_control=self._cancellation_control,
            )
        except Exception as exc:
            if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from exc
            raise _fatal_evaluation_error(
                exc,
                stage="result_finalization",
            ) from exc
        self._raise_if_cancel_requested()
        return admitted

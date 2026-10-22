"""
Shared per-dataset fitting evaluation boundary.

This module owns the fitting-side execution seam below the optimizer and above
the solver. It builds one evaluator per dataset from structured execution-request data
so the GUI/global-fit path no longer depends on the mutable prepared closure as
its architectural contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from kindred.core.exceptions import FittingCancelled
from kindred.core.analysis.fit_evaluation_outcome import (
    ScientificUnavailable,
)
from kindred.core.algebra.errors import UNCOMPUTABLE_ERRORS
from kindred.core.algebra.simulation_series import observable_closure
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_simulation_request_from_prepared_run,
    prepare_simulation_run,
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


# A calculated value outside its domain, or a physical predicate the solved
# state violates.
_VALUE_FAILURES = (*UNCOMPUTABLE_ERRORS, PhysicalDomainError)


def _depends_on_candidate(inputs: Sequence[str], reads_species: bool, origins: Mapping[str, str]) -> bool:
    """Whether a failure's inputs include an optimizer value; one that reads a
    current concentration depends on every value the candidate varies, as the
    solved curve does."""
    return any(origins.get(name) in _OPTIMIZER_ORIGINS for name in (origins if reads_species else inputs))


def _rejection(cause: FactualCause, *, producer: str = "fit_candidate") -> ScientificUnavailable:
    return ScientificUnavailable(producer=producer, stage="candidate_domain", factual_cause=cause)


@dataclass(frozen=True)
class PreparedFittingExecutionContext:
    """Typed fitting request and preparation metadata.

    Its sole producer unions the admitted regular grid with the admitted
    observation times inside the Fit span.  A measured-X dataset has no
    observation times; its curve also takes the solver's accepted steps and
    can end at an authored stop.
    """

    request: SimulationExecutionRequest
    requested_sample_times: tuple[float, ...]
    sample_solver_steps: bool
    use_authored_stops: bool
    initial_prefix: str = "init:"


def prepare_fitting_execution_context(
    *,
    prepared_dsl: PreparedDslRequestSet,
    t_start: float = 0.0,
    t_end: float,
    objective_sample_times: Sequence[float],
    prepared_solver_config: PreparedSolverConfig,
    sample_solver_steps: bool,
    use_authored_stops: bool,
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
    species_names = tuple(prepared_dsl.accepted_mechanism_for_structure().species_names())
    # A form each candidate completes: its starting concentrations and values
    # join the request before it is prepared, so nothing is prepared here.
    request = SimulationExecutionRequest(
        prepared_dsl=prepared_dsl,
        species_names=species_names,
        y0=(0.0,) * len(species_names),
        t_span=(float(t_start), float(t_end)),
        solver_config=prepared_solver_config,
    )
    return PreparedFittingExecutionContext(
        request=request,
        requested_sample_times=requested_sample_times,
        sample_solver_steps=sample_solver_steps,
        use_authored_stops=use_authored_stops,
        initial_prefix=initial_prefix,
    )


class FittingEvaluator:
    """Per-dataset in-process evaluator built from admitted execution data."""

    def __init__(
        self,
        context: PreparedFittingExecutionContext,
    ) -> None:
        self._context = context
        self._spec = context.request.prepared_dsl.parameter_algebra_spec_for_execution()

    def evaluate_series_with_parameter_origins(
        self,
        params: Mapping[str, float],
        origins: Mapping[str, str],
        outputs: tuple[str, ...],
        cancellation_control: SimulationCancellationControl,
    ) -> SimulationResultValue | ScientificUnavailable:
        """Evaluate one candidate, returning only natural scientific values.

        A value outside its domain rejects the candidate when it depends on an
        optimizer value, and a physical predicate the solved state violates
        always does; one fed only by configured values stays an error, as do
        cancellation and every other failure. Of the Algebra outputs, only the
        dataset's ``outputs`` and the lets they read can reject the candidate;
        other failures stay diagnostics.  Concurrent candidates of one dataset
        each pass their own ``cancellation_control``.
        """

        try:
            result = self._evaluate_series(params, cancellation_control)
        except _VALUE_FAILURES as exc:
            # The values a refusal names, through any value calculated from them.
            inputs = self._spec.input_dependencies(
                exc.details.get("parameter_names", ())
            )
            exc.annotate(suffix=self._fit_values(inputs, params))
            if isinstance(exc, PhysicalDomainError):
                return _rejection(factual_cause_from_exception(exc), producer="physical_intervention")
            if _depends_on_candidate(self._candidate_names(inputs), exc.details.get("reads_species", False), origins):
                return _rejection(factual_cause_from_exception(exc))
            raise
        if isinstance(result, SimulationScientificUnavailability):
            return ScientificUnavailable(
                producer=result.producer,
                stage=result.stage,
                factual_cause=FactualCause(
                    domain="simulation", origin=result.producer, code=None,
                    message=result.message, details=result.details,
                ),
            )
        # A let, or a value it reads, that fails records its own cause.
        result = replace(result, algebra_errors=tuple(
            replace(cause, message=cause.message + self._fit_values(cause.details.get("parameter_names", ()), params))
            for cause in result.algebra_errors
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
        """Reject the candidate for the first value failure, among the lets the
        outputs read, that depends on an optimizer value; None leaves it fatal."""
        used = observable_closure(
            self._spec.observable_block, outputs,
        )
        for cause in result.algebra_errors:
            if (
                cause.details["name"] in used
                and cause.details.get("value_failure")
                and _depends_on_candidate(
                    self._candidate_names(cause.details.get("parameter_names", ())),
                    cause.details.get("reads_species", False), origins,
                )
            ):
                return _rejection(cause)
        return None

    def parameter_values(
        self,
        params: Mapping[str, float],
        names: tuple[str, ...],
    ) -> Mapping[str, float]:
        """The named values one candidate calculates at the mechanism's
        temperature, ``[X]_0`` its starting concentration of X; nothing is
        solved."""
        request = self._candidate_request(params)
        prepared_run = prepare_simulation_run(execution_request=request)
        starting = {f"[{name}]_0": value for name, value in zip(request.species_names, request.y0)}
        calculated = prepared_run.parameter_values_at_temperature(
            float(request.prepared_dsl.analysis.temperature_K), tuple(name for name in names if name not in starting)
        )
        return {name: starting[name] if name in starting else calculated[name] for name in names}

    def _candidate_request(self, params: Mapping[str, float]) -> SimulationExecutionRequest:
        """The dataset's request with a candidate's starting concentrations
        and, as overrides, its other values."""
        prefix = self._context.initial_prefix
        request = self._context.request
        species_index = {name: idx for idx, name in enumerate(request.species_names)}
        y0 = list(request.y0)
        overrides = {}
        for name, value in params.items():
            if name.startswith(prefix):
                y0[species_index[name[len(prefix):]]] = value
            else:
                overrides[name] = value
        return replace(request, y0=tuple(y0), parameter_overrides=overrides or None)

    def _candidate_names(self, inputs: Sequence[str]) -> tuple[str, ...]:
        """A failure's inputs as the candidate names them; ``[X]_0`` is X's
        starting concentration."""
        prefix = self._context.initial_prefix
        return tuple(prefix + name[1:-3] if name.startswith("[") else name for name in inputs)

    def _fit_values(self, names: Sequence[str], params: Mapping[str, float]) -> str:
        """The Fit's own values for a failing value's inputs, which can differ
        from the numbers on the authored line its message quotes."""
        names = [(name, key) for name, key in zip(names, self._candidate_names(names)) if key in params]
        if not names:
            return ""
        units = {name: unit.symbol for name, unit in self._spec.input_units}
        # A starting concentration is in M.
        values = ", ".join(
            f"{name} = {params[key]:.6g}" + ("" if units.get(name, "M") == "1" else f" {units.get(name, 'M')}")
            for name, key in names
        )
        return f" Fit values: {values}."

    def _evaluate_series(
        self,
        params: Mapping[str, float],
        cancellation_control: SimulationCancellationControl,
    ) -> SimulationResultValue | SimulationScientificUnavailability:
        def raise_if_cancel_requested() -> None:
            if cancellation_control.cancel_requested():
                raise FittingCancelled()

        raise_if_cancel_requested()
        # Every candidate, including the first, receives a fresh isolated
        # request and one canonical preparation.  A rejected candidate can
        # therefore never contaminate the next candidate.
        prepared_run = prepare_simulation_run(
            execution_request=replace(
                self._candidate_request(params),
                use_authored_stops=self._context.use_authored_stops,
            ),
            cancellation_control=cancellation_control,
        )
        request = build_simulation_request_from_prepared_run(
            prepared_run,
            t_eval=np.asarray(
                self._context.requested_sample_times,
                dtype=float,
            ),
            candidate_evaluation=True,
            sample_solver_steps=self._context.sample_solver_steps,
        )
        result = solve_ode(request)
        if isinstance(result, SimulationScientificUnavailability):
            return result
        raise_if_cancel_requested()
        admitted = finalize_simulation_result(
            result=result,
            species_names=list(prepared_run.species_names),
            initials_for_algebra=prepared_run.initials_for_algebra,
            parameter_values_at_temperature=prepared_run.parameter_values_at_temperature,
            parameter_algebra_spec=prepared_run.parameter_algebra_spec,
            cancellation_control=cancellation_control,
        )
        raise_if_cancel_requested()
        return admitted

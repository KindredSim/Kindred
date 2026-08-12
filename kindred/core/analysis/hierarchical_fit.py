"""Nonlinear mixed-effects fitting with Gaussian random effects.

This module owns the Fit-local Hierarchical extension.  The extension is a
first-order Laplace approximation to a weighted-Gaussian nonlinear
mixed-effects model:

* every dataset has one Gaussian random effect on the configured optimizer
  scale;
* the random-effect mode is conditional on a jointly optimized population
  location, dispersion, and nuisance/fixed-effect coordinates; and
* the dataset callback evaluates exactly one dataset for one complete
  candidate, never a global all-dataset objective.

The callback returns the repository's typed scientific evaluation outcome.  A
residual vector is evidence only when its outcome is ``"converged"``.  In
particular, optimizer penalties, nonconvergence, candidate-domain rejection,
and fatal structural failures never enter a Laplace likelihood calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from kindred.core.exceptions import (
    FitFinalReplayError,
    FitSimulationError,
    FittingCancelled,
    OptimizationError,
    SimulationCancelled,
)
from kindred.core.fitting_completion import FitTerminalExtension
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.project_document import validate_fit_method
from kindred.core.simulation_failure import simulation_failure_from_exception

__all__ = [
    "HierarchicalFitConfig",
    "HierarchicalNuisanceParameter",
    "hierarchical_fit_extension",
]


_METHOD = (
    "nonlinear mixed-effects weighted-Gaussian likelihood with Gaussian "
    "random effects and Laplace marginalization"
)
_VALID_EVALUATION_STATUSES = frozenset(
    {"converged", "nonconverged", "candidate_domain_rejected", "cancelled", "fatal"}
)
_POPULATION_COORDINATE = "__hierarchical_population__"
_LOG_DISPERSION_COORDINATE = "__hierarchical_log_dispersion__"


def _private_coordinate(prefix: str, occupied: set[str]) -> str:
    """Return a deterministic private coordinate absent from user names."""

    candidate = prefix
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{prefix}{suffix}"
    occupied.add(candidate)
    return candidate


def _raise_wrapped_programming_failure(exc: BaseException) -> None:
    details = getattr(exc, "details", None)
    if isinstance(details, Mapping) and details.get("stage") == "hierarchical_likelihood":
        return
    cause = exc.__cause__
    if isinstance(cause, (AssertionError, AttributeError, KeyError, TypeError, ValueError)):
        raise cause


def _finite(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _validate_bounds(
    *,
    name: str,
    initial: float,
    minimum: float,
    maximum: float,
    log10: bool,
) -> tuple[str, float, float, float, bool]:
    parameter_name = str(name or "").strip()
    if not parameter_name:
        raise ValueError("Hierarchical parameter name must be nonblank.")
    initial_value = _finite(initial, f"{parameter_name} initial")
    low = _finite(minimum, f"{parameter_name} minimum")
    high = _finite(maximum, f"{parameter_name} maximum")
    if not low < high or not low <= initial_value <= high:
        raise ValueError(f"{parameter_name} value and bounds are inconsistent.")
    use_log10 = bool(log10)
    if use_log10 and low <= 0.0:
        raise ValueError("Log-scale hierarchical fitting requires positive bounds.")
    return parameter_name, initial_value, low, high, use_log10


@dataclass(frozen=True, slots=True)
class HierarchicalNuisanceParameter:
    """One jointly optimized nuisance/fixed-effect coordinate.

    Values are supplied on the physical scale and encoded to the optimizer
    scale only inside this module.  ``name`` is the exact coordinate key that
    the dataset evaluator expects (normally a shared name or ``dataset::name``
    for a non-hierarchical dataset coordinate).
    """

    name: str
    initial: float
    minimum: float
    maximum: float
    log10: bool = False

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("Hierarchical nuisance parameter name must be nonblank.")
        initial = _finite(self.initial, f"{name} initial")
        low = float(self.minimum)
        high = float(self.maximum)
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or not low < high
            or not low <= initial <= high
        ):
            raise ValueError(f"{name} value and bounds are inconsistent.")
        log10 = bool(self.log10)
        if log10 and (low <= 0.0 or not math.isfinite(high)):
            raise ValueError(
                "Log-scale hierarchical nuisance bounds must be finite and positive."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", low)
        object.__setattr__(self, "maximum", high)
        object.__setattr__(self, "log10", log10)

    def optimizer_value(self, physical: float) -> float:
        value = _finite(physical, f"{self.name} physical value")
        if self.log10 and value <= 0.0:
            raise ValueError(f"{self.name} requires a positive physical value.")
        return math.log10(value) if self.log10 else value

    def physical_value(self, optimizer_value: float) -> float:
        raw = _finite(optimizer_value, f"{self.name} optimizer value")
        if self.log10:
            try:
                value = math.pow(10.0, raw)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{self.name} optimizer value overflowed.") from exc
        else:
            value = raw
        if not math.isfinite(value):
            raise ValueError(f"{self.name} physical value is non-finite.")
        return value

    @property
    def optimizer_bounds(self) -> tuple[float, float]:
        return self.optimizer_value(self.minimum), self.optimizer_value(self.maximum)


@dataclass(frozen=True, slots=True)
class HierarchicalFitConfig:
    """One declared Hierarchical scalar on its configured optimizer scale."""

    parameter_name: str
    initial_population: float
    minimum: float
    maximum: float
    log10: bool = False
    initial_dispersion: float = 0.1
    minimum_dispersion: float = 1e-8
    maximum_dispersion: float = 10.0
    optimizer_method: str = "trf"
    max_nfev: int = 500
    inner_max_nfev: int = 200

    def __post_init__(self) -> None:
        name, initial, low, high, log10 = _validate_bounds(
            name=self.parameter_name,
            initial=self.initial_population,
            minimum=self.minimum,
            maximum=self.maximum,
            log10=self.log10,
        )
        initial_dispersion = _finite(
            self.initial_dispersion,
            "hierarchical initial dispersion",
        )
        minimum_dispersion = _finite(
            self.minimum_dispersion,
            "hierarchical minimum dispersion",
        )
        maximum_dispersion = _finite(
            self.maximum_dispersion,
            "hierarchical maximum dispersion",
        )
        if not 0.0 < minimum_dispersion <= initial_dispersion < maximum_dispersion:
            raise ValueError(
                "Hierarchical dispersion requires 0 < minimum <= initial < maximum."
            )
        method = validate_fit_method(self.optimizer_method)
        if int(self.max_nfev) <= 0 or int(self.inner_max_nfev) <= 0:
            raise ValueError("Hierarchical fitting budgets must be positive.")
        object.__setattr__(self, "parameter_name", name)
        object.__setattr__(self, "initial_population", initial)
        object.__setattr__(self, "minimum", low)
        object.__setattr__(self, "maximum", high)
        object.__setattr__(self, "log10", log10)
        object.__setattr__(self, "initial_dispersion", initial_dispersion)
        object.__setattr__(self, "minimum_dispersion", minimum_dispersion)
        object.__setattr__(self, "maximum_dispersion", maximum_dispersion)
        object.__setattr__(self, "optimizer_method", method)
        object.__setattr__(self, "max_nfev", int(self.max_nfev))
        object.__setattr__(self, "inner_max_nfev", int(self.inner_max_nfev))

    def optimizer_value(self, physical: float) -> float:
        value = _finite(physical, f"{self.parameter_name} physical value")
        if self.log10 and value <= 0.0:
            raise ValueError(
                f"{self.parameter_name} requires a positive physical value."
            )
        return math.log10(value) if self.log10 else value

    def physical_value(self, optimizer_value: float) -> float:
        raw = _finite(optimizer_value, f"{self.parameter_name} optimizer value")
        if self.log10:
            try:
                value = math.pow(10.0, raw)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    f"{self.parameter_name} optimizer value overflowed."
                ) from exc
        else:
            value = raw
        if not math.isfinite(value):
            raise ValueError(f"{self.parameter_name} physical value is non-finite.")
        return value

    @property
    def optimizer_bounds(self) -> tuple[float, float]:
        return self.optimizer_value(self.minimum), self.optimizer_value(self.maximum)


@dataclass(frozen=True, slots=True)
class _ConditionalMode:
    optimizer_value: float
    curvature: float | None
    marginal_nll: float | None
    status: str
    nfev: int
    boundary_active: bool = False
    laplace_correction_applied: bool = False
    outcome: Mapping[str, object] | None = None


class _EvaluationState(Exception):
    def __init__(self, status: str, details: Mapping[str, object] | None = None) -> None:
        self.status = status
        self.details = dict(details or {})
        super().__init__(str(self.details.get("message") or status))


def _callback_status(value: object) -> str:
    if not isinstance(value, FitEvaluationOutcome):
        raise TypeError(
            "Hierarchical dataset evaluation must return FitEvaluationOutcome."
        )
    status = str(getattr(value, "status", "") or "").strip().lower()
    if status not in _VALID_EVALUATION_STATUSES:
        raise TypeError(
            "Hierarchical dataset evaluation must return the canonical typed "
            f"outcome status {_VALID_EVALUATION_STATUSES}; got {status!r}."
        )
    return status


def _callback_details(value: object) -> dict[str, object]:
    details: dict[str, object] = dict(value.provenance)
    cause = value.cause_payload()
    if cause is not None:
        details["cause"] = dict(cause)
    return details


def _callback_residuals(value: object, *, dataset_id: str) -> np.ndarray:
    residuals = value.residuals
    if residuals is None:
        raise FitSimulationError(
            f"Hierarchical evaluation omitted residuals for dataset {dataset_id!r}.",
            details={"fatal": True, "candidate_domain": False, "stage": "hierarchical_likelihood"},
        )
    result = np.asarray(residuals, dtype=float).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise FitSimulationError(
            f"Hierarchical residuals are invalid for dataset {dataset_id!r}.",
            details={"fatal": True, "candidate_domain": False, "stage": "hierarchical_likelihood"},
        )
    return result


def _coerce_callback_outcome(
    value: object,
    *,
    dataset_id: str,
) -> tuple[str, np.ndarray | None, dict[str, object]]:
    status = _callback_status(value)
    details = _callback_details(value)
    if status == "converged":
        return status, _callback_residuals(value, dataset_id=dataset_id), details
    # A non-converged or rejected evaluation is not scientific evidence.  It
    # is deliberately forbidden from exposing a residual vector to callers.
    if value.residuals is not None:
        details["penalty_residuals_discarded"] = True
    if status == "cancelled":
        raise FittingCancelled(str(details.get("message") or "Fit cancelled by user"))
    return status, None, details


def _outcome_from_exception(exc: BaseException) -> tuple[str, dict[str, object]]:
    if isinstance(exc, (FittingCancelled, SimulationCancelled)):
        raise FittingCancelled() from exc
    if isinstance(exc, FitSimulationError):
        outcome = FitEvaluationOutcome.from_exception(exc)
        if outcome.status == "candidate_domain_rejected":
            return "candidate_domain_rejected", {
                "message": str(exc),
                "cause": dict(outcome.cause_payload() or {}),
            }
    return "fatal", {
        "message": str(exc),
        "cause": {
            "type": type(exc).__name__,
            "message": str(exc),
            "details": dict(getattr(exc, "details", {}) or {}),
        },
    }


def _observed_information(
    function: Callable[[np.ndarray], float],
    optimum: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    dimension = int(optimum.size)
    if dimension == 0:
        return None, "Observed information requires at least one coordinate."
    steps = np.maximum(1e-5 * np.maximum(1.0, np.abs(optimum)), 1e-6)
    if np.any(optimum - steps <= lower) or np.any(optimum + steps >= upper):
        return None, "Observed information is unavailable at an optimizer bound."
    try:
        center = float(function(optimum))
        hessian = np.empty((dimension, dimension), dtype=float)
        for index in range(dimension):
            delta = np.zeros(dimension, dtype=float)
            delta[index] = steps[index]
            hessian[index, index] = (
                function(optimum + delta)
                - 2.0 * center
                + function(optimum - delta)
            ) / (steps[index] ** 2)
        for first_index in range(dimension):
            for second_index in range(first_index):
                first = np.zeros(dimension, dtype=float)
                second = np.zeros(dimension, dtype=float)
                first[first_index] = steps[first_index]
                second[second_index] = steps[second_index]
                cross = (
                    function(optimum + first + second)
                    - function(optimum + first - second)
                    - function(optimum - first + second)
                    + function(optimum - first - second)
                ) / (4.0 * steps[first_index] * steps[second_index])
                hessian[first_index, second_index] = cross
                hessian[second_index, first_index] = cross
    except _EvaluationState as exc:
        return None, f"Observed information evaluation was {exc.status}."
    except (FitSimulationError, OptimizationError, ValueError, FloatingPointError) as exc:
        details = getattr(exc, "details", None)
        if isinstance(details, Mapping) and details.get("fatal") is True:
            raise
        return None, str(exc)
    if not np.all(np.isfinite(hessian)):
        return None, "Observed information was non-finite."
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= 0.0):
        return None, "Observed information was not positive definite."
    covariance = np.linalg.inv(hessian)
    return covariance, None


def _unavailable(
    *,
    provenance: Mapping[str, object],
    details: Mapping[str, object],
    reason: str,
) -> FitTerminalExtension:
    return FitTerminalExtension(
        kind="hierarchical",
        status="unavailable",
        method=_METHOD,
        provenance=provenance,
        details=details,
        unavailable_reason=reason,
    )


def hierarchical_fit_extension(
    *,
    dataset_ids: Sequence[str],
    config: HierarchicalFitConfig,
    dataset_evaluator: Callable[[str, Mapping[str, float]], FitEvaluationOutcome],
    primary_evaluation: FitEvaluationOutcome,
    nuisance_parameters: Sequence[HierarchicalNuisanceParameter] = (),
    initial_effects: Mapping[str, float] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> FitTerminalExtension:
    """Fit one nonlinear mixed-effects coordinate by Laplace marginalization.

    ``dataset_evaluator`` is the integration handoff.  It is called as
    ``dataset_evaluator(dataset_id, physical_parameter_values)`` and must
    return the canonical typed scientific evaluation outcome.  The parameter
    mapping contains every nuisance coordinate and exactly one random-effect
    coordinate named ``f"{dataset_id}::{config.parameter_name}"``.  The
    callback owns simulation and alignment for that one dataset only.

    ``primary_evaluation`` is the terminal outcome of the ordinary primary
    Fit.  Hierarchical analysis is unavailable unless that outcome is
    ``"converged"``; a finite optimizer vector alone is not sufficient.
    """

    if not isinstance(config, HierarchicalFitConfig):
        raise TypeError("Hierarchical fitting requires HierarchicalFitConfig.")
    if not callable(dataset_evaluator):
        raise TypeError("Hierarchical fitting requires a dataset evaluator callback.")
    if not isinstance(primary_evaluation, FitEvaluationOutcome):
        raise TypeError(
            "Hierarchical fitting requires the canonical primary FitEvaluationOutcome."
        )
    ids = tuple(str(value or "").strip() for value in dataset_ids)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Hierarchical dataset IDs must be unique and nonblank.")
    nuisance = tuple(nuisance_parameters)
    if any(not isinstance(item, HierarchicalNuisanceParameter) for item in nuisance):
        raise TypeError("Hierarchical nuisance parameters require typed declarations.")
    nuisance_names = tuple(item.name for item in nuisance)
    if len(nuisance_names) != len(set(nuisance_names)):
        raise ValueError("Hierarchical nuisance parameter names must be unique.")
    random_names = tuple(f"{dataset_id}::{config.parameter_name}" for dataset_id in ids)
    if set(nuisance_names).intersection(random_names):
        raise ValueError("Hierarchical nuisance coordinates overlap random effects.")
    occupied_coordinates = set(nuisance_names) | set(random_names)
    population_coordinate = _private_coordinate(
        _POPULATION_COORDINATE,
        occupied_coordinates,
    )
    log_dispersion_coordinate = _private_coordinate(
        _LOG_DISPERSION_COORDINATE,
        occupied_coordinates,
    )

    provenance = {
        "likelihood_model": "weighted Gaussian observation residuals",
        "random_effects": "independent Gaussian dataset effects",
        "marginalization": "first-order Laplace approximation at conditional modes",
        "optimizer_scale": "log10" if config.log10 else "linear",
        "optimizer_method": config.optimizer_method,
        "max_nfev": config.max_nfev,
        "inner_max_nfev": config.inner_max_nfev,
        "joint_nuisance_reoptimization": True,
        "dataset_specific_evaluation": True,
        "fit_local_only": True,
        "publication_requires_shared_refit": True,
        "primary_evaluation_status": primary_evaluation.status,
    }
    base_details = {
        "parameter_name": config.parameter_name,
        "dataset_ids": ids,
        "nuisance_coordinates": nuisance_names,
        "random_effect_coordinates": random_names,
        "private_population_coordinate": population_coordinate,
        "private_log_dispersion_coordinate": log_dispersion_coordinate,
        "fit_local_only": True,
        "publication_requires_shared_refit": True,
        "primary_evaluation_status": primary_evaluation.status,
    }
    if primary_evaluation.status != "converged":
        return _unavailable(
            provenance=provenance,
            details={
                **base_details,
                "identifiability": "primary Fit is not converged",
                "primary_evaluation_cause": dict(
                    primary_evaluation.cause_payload() or {}
                ),
            },
            reason="Hierarchical analysis requires a converged primary Fit.",
        )
    if len(ids) < 2:
        return _unavailable(
            provenance=provenance,
            details={**base_details, "identifiability": "insufficient information"},
            reason="Hierarchical dispersion requires at least two datasets.",
        )

    low, high = config.optimizer_bounds
    population_initial = config.optimizer_value(config.initial_population)
    initial_effects_raw = {
        dataset_id: config.optimizer_value(
            float((initial_effects or {}).get(dataset_id, config.initial_population))
        )
        for dataset_id in ids
    }
    nuisance_initial = {item.name: item.optimizer_value(item.initial) for item in nuisance}
    nuisance_bounds = {item.name: item.optimizer_bounds for item in nuisance}
    last_states: dict[str, Mapping[str, object]] = {}

    def parameter_values(
        nuisance_raw: Mapping[str, float],
        dataset_id: str,
        effect_raw: float,
    ) -> dict[str, float]:
        values = {
            item.name: item.physical_value(float(nuisance_raw[item.name]))
            for item in nuisance
        }
        values[f"{dataset_id}::{config.parameter_name}"] = config.physical_value(effect_raw)
        return values

    def evaluate_data(
        dataset_id: str,
        nuisance_raw: Mapping[str, float],
        effect_raw: float,
    ) -> tuple[str, np.ndarray | None, dict[str, object]]:
        fatal_cause: BaseException | None = None
        try:
            outcome = dataset_evaluator(
                dataset_id,
                parameter_values(nuisance_raw, dataset_id, effect_raw),
            )
            fatal_cause = outcome.cause
            status, residuals, details = _coerce_callback_outcome(
                outcome,
                dataset_id=dataset_id,
            )
        except BaseException as exc:
            status, details = _outcome_from_exception(exc)
            residuals = None
        last_states[dataset_id] = {"status": status, **details}
        if status == "converged" and residuals is not None:
            return status, residuals, details
        if status == "fatal":
            error = FitSimulationError(
                str(details.get("message") or f"Fatal hierarchical evaluation for {dataset_id!r}."),
                details={
                    "fatal": True,
                    "candidate_domain": False,
                    "stage": "hierarchical_likelihood",
                    **details,
                },
            )
            if fatal_cause is not None:
                raise error from fatal_cause
            raise error
        raise _EvaluationState(status, details)

    def conditional_mode(
        *,
        dataset_id: str,
        population: float,
        dispersion: float,
        initial: float,
        nuisance_raw: Mapping[str, float],
    ) -> _ConditionalMode:
        initial_clipped = min(max(float(initial), low), high)
        def evaluate_residuals(raw: float) -> np.ndarray:
            _status, residuals, _details = evaluate_data(dataset_id, nuisance_raw, raw)
            if residuals is None:
                raise _EvaluationState("fatal")
            return residuals

        def objective(vector: np.ndarray) -> np.ndarray:
            raw = float(np.asarray(vector, dtype=float).reshape(-1)[0])
            try:
                residuals = evaluate_residuals(raw)
            except _EvaluationState as exc:
                if exc.status == "fatal":
                    raise FitSimulationError(
                        str(exc),
                        details={"fatal": True, "candidate_domain": False, "stage": "hierarchical_likelihood"},
                    ) from exc
                return np.asarray([1e6], dtype=float)
            random_effect = (raw - population) / dispersion
            # Keep the optimizer-facing vector scalar so an admitted penalty
            # has a stable shape before any valid candidate establishes the
            # observation-vector length.  Scientific residuals remain separate
            # and are consumed only by the converged evidence path.
            return np.asarray(
                [
                    math.sqrt(
                        float(np.dot(residuals, residuals))
                        + random_effect * random_effect
                    )
                ],
                dtype=float,
            )

        try:
            fitted = fit_parameters(
                objective,
                {"effect": initial_clipped},
                bounds={"effect": (low, high)},
                method=config.optimizer_method,
                max_nfev=config.inner_max_nfev,
                cancellation_check=cancellation_check,
            )
        except _EvaluationState as exc:
            return _ConditionalMode(
                optimizer_value=initial_clipped,
                curvature=None,
                marginal_nll=None,
                status=exc.status,
                nfev=0,
                outcome=exc.details,
            )
        mode = float(fitted.parameters["effect"])
        try:
            residuals = evaluate_residuals(mode)
        except _EvaluationState as exc:
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=None,
                marginal_nll=None,
                status=exc.status,
                nfev=int(fitted.nfev),
                outcome=exc.details,
            )
        if not fitted.success:
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=None,
                marginal_nll=None,
                status="nonconverged",
                nfev=int(fitted.nfev),
                outcome={"message": fitted.message},
            )

        boundary_tolerance = max(1e-8, 1e-6 * max(1.0, abs(mode)))
        boundary_active = (mode - low <= boundary_tolerance) or (high - mode <= boundary_tolerance)
        z = (mode - population) / dispersion
        data_nll = 0.5 * float(np.dot(residuals, residuals))
        if boundary_active:
            # The usual Laplace correction assumes an interior, two-sided
            # mode.  At a bound we report the conditional result but do not
            # claim that invalid correction or its uncertainty.
            marginal_nll = data_nll + 0.5 * z * z
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=None,
                marginal_nll=max(0.0, marginal_nll),
                status="bound_limited",
                nfev=int(fitted.nfev),
                boundary_active=True,
                laplace_correction_applied=False,
                outcome={"message": "Conditional mode is optimizer-bound."},
            )

        width = high - low
        step = max(1e-6 * max(1.0, abs(mode)), 1e-7 * width)
        left, right = mode - step, mode + step
        try:
            left_residuals = evaluate_residuals(left)
            right_residuals = evaluate_residuals(right)
        except _EvaluationState as exc:
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=None,
                marginal_nll=None,
                status=exc.status,
                nfev=int(fitted.nfev),
                outcome=exc.details,
            )
        if left_residuals.size != residuals.size or right_residuals.size != residuals.size:
            raise FitSimulationError(
                "Hierarchical evaluator changed residual-vector identity.",
                details={"fatal": True, "candidate_domain": False, "stage": "hierarchical_likelihood"},
            )
        derivative = (right_residuals - left_residuals) / (2.0 * step)
        data_curvature = float(np.dot(derivative, derivative))
        curvature = data_curvature + 1.0 / (dispersion * dispersion)
        if not math.isfinite(curvature) or curvature <= 0.0:
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=None,
                marginal_nll=None,
                status="candidate_domain_rejected",
                nfev=int(fitted.nfev),
                outcome={"message": "Conditional curvature is unavailable."},
            )
        marginal_nll = (
            data_nll
            + 0.5 * z * z
            + math.log(dispersion)
            + 0.5 * math.log(curvature)
        )
        if not math.isfinite(marginal_nll):
            return _ConditionalMode(
                optimizer_value=mode,
                curvature=curvature,
                marginal_nll=None,
                status="candidate_domain_rejected",
                nfev=int(fitted.nfev),
                outcome={"message": "Laplace likelihood is non-finite."},
            )
        return _ConditionalMode(
            optimizer_value=mode,
            curvature=curvature,
            marginal_nll=max(0.0, marginal_nll),
            status="converged",
            nfev=int(fitted.nfev),
            laplace_correction_applied=True,
        )

    def modes(
        population: float,
        dispersion: float,
        nuisance_raw: Mapping[str, float],
    ) -> dict[str, _ConditionalMode]:
        result: dict[str, _ConditionalMode] = {}
        for dataset_id in ids:
            if cancellation_check is not None and cancellation_check():
                raise FittingCancelled()
            item = conditional_mode(
                dataset_id=dataset_id,
                population=population,
                dispersion=dispersion,
                # Keep each objective call pure: a cached mode from a prior
                # nuisance candidate would make the outer likelihood depend
                # on evaluation order and can bias joint reoptimization.
                initial=initial_effects_raw.get(dataset_id, population),
                nuisance_raw=nuisance_raw,
            )
            result[dataset_id] = item
        return result

    outer_names = (
        population_coordinate,
        log_dispersion_coordinate,
        *nuisance_names,
    )
    outer_bounds = {
        population_coordinate: (low, high),
        log_dispersion_coordinate: (
            math.log(config.minimum_dispersion),
            math.log(config.maximum_dispersion),
        ),
        **nuisance_bounds,
    }
    outer_initial = {
        population_coordinate: population_initial,
        log_dispersion_coordinate: math.log(config.initial_dispersion),
        **nuisance_initial,
    }
    penalty = 1e6

    def unpack(vector: np.ndarray) -> tuple[float, float, dict[str, float]]:
        values = np.asarray(vector, dtype=float).reshape(-1)
        population = float(values[0])
        dispersion = math.exp(float(values[1]))
        nuisance_raw = {
            name: float(values[index + 2])
            for index, name in enumerate(nuisance_names)
        }
        return population, dispersion, nuisance_raw

    def outer_objective(vector: np.ndarray) -> np.ndarray:
        population, dispersion, nuisance_raw = unpack(vector)
        fitted_modes = modes(population, dispersion, nuisance_raw)
        invalid = [item for item in fitted_modes.values() if item.status != "converged"]
        if invalid:
            # This is an optimizer steering value only.  It is never consumed
            # as a dataset residual or likelihood evidence.
            return np.full(len(ids), penalty, dtype=float)
        return np.asarray(
            [math.sqrt(2.0 * float(item.marginal_nll)) for item in fitted_modes.values()],
            dtype=float,
        )

    try:
        fitted = fit_parameters(
            outer_objective,
            outer_initial,
            bounds=outer_bounds,
            method=config.optimizer_method,
            max_nfev=config.max_nfev,
            cancellation_check=cancellation_check,
        )
        fitted_vector = np.asarray([fitted.parameters[name] for name in outer_names], dtype=float)
        population, estimated_dispersion, fitted_nuisance_raw = unpack(fitted_vector)
        final_modes = modes(population, estimated_dispersion, fitted_nuisance_raw)
    except (FittingCancelled, SimulationCancelled):
        raise FittingCancelled()
    except (FitSimulationError, OptimizationError) as exc:
        if isinstance(exc, FitFinalReplayError):
            raise
        _raise_wrapped_programming_failure(exc)
        details = getattr(exc, "details", None)
        if isinstance(details, Mapping) and details.get("fatal") is True:
            raise
        return _unavailable(
            provenance=provenance,
            details={
                **base_details,
                "failure": simulation_failure_from_exception(exc),
                "evaluation_outcomes": dict(last_states),
                "identifiability": "likelihood evaluation failed",
            },
            reason="The hierarchical likelihood could not be optimized.",
        )

    invalid_modes = {
        dataset_id: item
        for dataset_id, item in final_modes.items()
        if item.status != "converged"
    }
    if invalid_modes:
        statuses = {dataset_id: item.status for dataset_id, item in invalid_modes.items()}
        reason = "Hierarchical evaluation did not produce converged scientific evidence."
        boundary_modes = {
            dataset_id: item
            for dataset_id, item in invalid_modes.items()
            if item.status == "bound_limited"
        }
        if boundary_modes:
            reason = (
                "Hierarchical marginal likelihood is unavailable because a "
                "conditional random-effect mode is bound-limited."
            )
        elif any(item.status == "fatal" for item in invalid_modes.values()):
            reason = "A fatal structural failure occurred during hierarchical evaluation."
        elif any(item.status == "candidate_domain_rejected" for item in invalid_modes.values()):
            reason = "A hierarchical candidate was outside the admitted scientific domain."
        elif any(item.status == "nonconverged" for item in invalid_modes.values()):
            reason = "A hierarchical conditional mode did not converge."
        return _unavailable(
            provenance=provenance,
            details={
                **base_details,
                "evaluation_outcomes": {
                    **dict(last_states),
                    **{dataset_id: {"status": status} for dataset_id, status in statuses.items()},
                },
                "optimizer_converged": bool(fitted.success),
                "boundary_modes": {
                    dataset_id: {
                        "optimizer_value": item.optimizer_value,
                        "physical_value": config.physical_value(item.optimizer_value),
                        "status": item.status,
                        "laplace_correction_applied": item.laplace_correction_applied,
                    }
                    for dataset_id, item in boundary_modes.items()
                },
                "identifiability": "insufficient valid evaluations",
            },
            reason=reason,
        )

    dispersion_boundary = math.isclose(
        estimated_dispersion,
        config.minimum_dispersion,
        rel_tol=1e-5,
        abs_tol=1e-12,
    )
    reported_dispersion = estimated_dispersion
    effects = {
        dataset_id: {
            "optimizer_value": item.optimizer_value,
            "physical_value": config.physical_value(item.optimizer_value),
            "random_effect_optimizer_scale": item.optimizer_value - population,
            "random_effect_physical_scale": config.physical_value(item.optimizer_value)
            - config.physical_value(population),
            "conditional_curvature": item.curvature,
            "conditional_converged": item.status == "converged",
            "conditional_status": item.status,
            "conditional_nfev": item.nfev,
            "boundary_active": item.boundary_active,
            "laplace_correction_applied": item.laplace_correction_applied,
        }
        for dataset_id, item in final_modes.items()
    }

    nuisance_result = {
        item.name: {
            "optimizer_value": fitted_nuisance_raw[item.name],
            "physical_value": item.physical_value(fitted_nuisance_raw[item.name]),
            "optimizer_bounds": item.optimizer_bounds,
            "physical_bounds": (item.minimum, item.maximum),
            "log10": item.log10,
        }
        for item in nuisance
    }
    optimum = fitted_vector
    lower = np.asarray([outer_bounds[name][0] for name in outer_names], dtype=float)
    upper = np.asarray([outer_bounds[name][1] for name in outer_names], dtype=float)

    def marginal_total(vector: np.ndarray) -> float:
        location, tau, nuisance_values = unpack(vector)
        fitted_modes = modes(location, tau, nuisance_values)
        if any(item.status != "converged" for item in fitted_modes.values()):
            raise _EvaluationState("nonconverged")
        return float(sum(float(item.marginal_nll or 0.0) for item in fitted_modes.values()))

    boundary_modes = tuple(
        dataset_id for dataset_id, item in final_modes.items() if item.boundary_active
    )
    if not fitted.success:
        covariance = None
        uncertainty_reason = "Outer Hierarchical optimizer did not converge."
    elif boundary_modes:
        covariance = None
        uncertainty_reason = "Interior Laplace uncertainty is unavailable for bound-limited modes."
    else:
        covariance, uncertainty_reason = _observed_information(
            marginal_total,
            optimum,
            lower,
            upper,
        )
    if covariance is None:
        uncertainty: Mapping[str, object] = {
            "status": "unavailable",
            "method": "finite-difference observed information",
            "reason": uncertainty_reason,
        }
    else:
        standard_errors = {
            name: math.sqrt(float(covariance[index, index]))
            for index, name in enumerate(outer_names)
        }
        population_se_optimizer = standard_errors[population_coordinate]
        population_se_physical = (
            math.log(10.0) * config.physical_value(population) * population_se_optimizer
            if config.log10
            else population_se_optimizer
        )
        nuisance_se_physical = {
            item.name: (
                math.log(10.0)
                * item.physical_value(fitted_nuisance_raw[item.name])
                * standard_errors[item.name]
                if item.log10
                else standard_errors[item.name]
            )
            for item in nuisance
        }
        uncertainty = {
            "status": "available",
            "method": "finite-difference observed information",
            "standard_errors_optimizer_scale": standard_errors,
            "population_standard_error_optimizer_scale": population_se_optimizer,
            "population_standard_error_physical_scale": population_se_physical,
            "dispersion_standard_error": estimated_dispersion * standard_errors[log_dispersion_coordinate],
            "nuisance_standard_errors_physical_scale": nuisance_se_physical,
            "covariance": tuple(tuple(float(value) for value in row) for row in covariance),
        }

    diagnostics: list[str] = []
    if len(ids) < 3:
        diagnostics.append(
            "Between-dataset dispersion is weakly identified with fewer than three datasets."
        )
    if dispersion_boundary:
        diagnostics.append("Between-dataset dispersion lower-bound active.")
    if boundary_modes:
        diagnostics.append(
            "Conditional modes are bound-limited; interior Laplace correction and uncertainty are unavailable."
        )
    if fitted.identifiability_warning:
        diagnostics.append(str(fitted.identifiability_warning))
    overall_converged = bool(fitted.success) and all(
        item.status == "converged" for item in final_modes.values()
    )
    status = "available" if overall_converged and not diagnostics else "partial"
    return FitTerminalExtension(
        kind="hierarchical",
        status=status,
        method=_METHOD,
        provenance={
            **provenance,
            "outer_nfev": int(fitted.nfev),
            "uncertainty_method": uncertainty["method"],
            "nuisance_coordinates": nuisance_names,
        },
        details={
            **base_details,
            "population_location": config.physical_value(population),
            "population_location_optimizer_scale": population,
            "between_dataset_dispersion": reported_dispersion,
            "estimated_positive_dispersion": estimated_dispersion,
            "estimated_positive_dispersion_optimizer_scale": math.log(estimated_dispersion),
            "dispersion_parameterization": "natural_log",
            "dispersion_boundary": dispersion_boundary,
            "dispersion_boundary_status": (
                "lower-bound active" if dispersion_boundary else "interior"
            ),
            "individual_dataset_effects": effects,
            "nuisance_parameters": nuisance_result,
            "overall_converged": overall_converged,
            "optimizer_converged": bool(fitted.success),
            "optimizer_message": fitted.message,
            "physical_bounds": (config.minimum, config.maximum),
            "optimizer_bounds": (low, high),
            "dispersion_bounds": (config.minimum_dispersion, config.maximum_dispersion),
            "dispersion_bounds_optimizer_scale": (
                math.log(config.minimum_dispersion),
                math.log(config.maximum_dispersion),
            ),
            "boundary_active_modes": boundary_modes,
            "identifiability_diagnostics": tuple(diagnostics),
            "evaluation_outcomes": dict(last_states),
            "uncertainty": uncertainty,
            "fit_local_only": True,
            "publication_requires_shared_refit": True,
        },
    )

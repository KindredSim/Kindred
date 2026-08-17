"""Named, provenance-complete uncertainty methods for completed Fits.

The functions in this module do not evaluate models or own an optimizer.  They
drive the canonical Fit objective and optimizer supplied by the global fitting
composition boundary, then return immutable terminal extensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis.fit_evaluation_outcome import (
    FitEvaluationOutcome,
)
from kindred.core.analysis.parametric_alignment import ParametricXProjection
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_completion import (
    FitExtensionProgress,
    FitTerminalExtension,
)
from kindred.core.fitting_optimization import fit_parameters_with_fixed
from kindred.core.fitting_method_policy import validate_fit_method
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.core.species import validate_evaluation_count

__all__ = [
    "BOOTSTRAP_SEED_RANGE",
    "bootstrap_de_seed",
    "validate_bootstrap_seed",
    "ProfileLikelihoodConfig",
    "ResidualBootstrapConfig",
    "profile_likelihood_extension",
    "residual_bootstrap_extension",
]


_PROFILE_METHOD = "weighted-Gaussian profile likelihood"
_BOOTSTRAP_METHOD = "stratified within-dataset/target residual bootstrap"
_PROFILE_MAX_POINTS_PER_SIDE = 10_000
_BOOTSTRAP_MAX_REPLICATES = 1_000_000
BOOTSTRAP_SEED_RANGE = (0, 2_000_000_000)


def validate_bootstrap_seed(value: object) -> int:
    """Admit the immutable residual-Bootstrap seed before launch."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("Bootstrap seed must be an integer.")
    normalized = int(value)
    lower, upper = BOOTSTRAP_SEED_RANGE
    if not lower <= normalized <= upper:
        raise ValueError(
            f"Bootstrap seed must be within {lower} and {upper}."
        )
    return normalized


def bootstrap_de_seed(
    bootstrap_seed: object,
    replicate_index: object,
) -> int:
    """Derive one deterministic DE seed for a zero-based replicate."""

    seed = validate_bootstrap_seed(bootstrap_seed)
    if isinstance(replicate_index, bool) or not isinstance(
        replicate_index, (int, np.integer)
    ):
        raise TypeError("Bootstrap replicate index must be an integer.")
    index = int(replicate_index)
    if not 0 <= index <= int(np.iinfo(np.uint32).max):
        raise ValueError(
            "Bootstrap replicate index must be zero-based within the uint32 domain."
        )
    return int(
        np.random.SeedSequence(
            [seed, 0x424F4F54, 1, index]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _bounded_positive_integer(
    value: object,
    *,
    field: str,
    maximum: int,
) -> int:
    """Validate a user-owned count without silently truncating it."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    if not math.isfinite(exact) or exact != result:
        raise ValueError(f"{field} must be an integer.")
    if result <= 0 or result > int(maximum):
        raise ValueError(f"{field} must be within 1 and {int(maximum)}.")
    return result


def _confidence(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("Confidence level must lie strictly between zero and one.")
    return result


def _coordinate_display(value: object) -> str:
    return str(getattr(value, "display_name", None) or value)


def _names(values: Sequence[object], *, field: str) -> tuple[object, ...]:
    names = tuple(
        value if hasattr(value, "display_name") else str(value or "").strip()
        for value in values
    )
    if not names or any(not value for value in names) or len(names) != len(set(names)):
        raise ValueError(f"{field} must contain unique nonblank names.")
    return names


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodConfig:
    """Configuration recorded with a weighted-Gaussian profile interval."""

    parameter_names: tuple[object, ...]
    confidence_level: float = 0.95
    max_bracketing_steps_per_side: int = 8
    max_nfev_per_point: int = 300
    optimizer_method: str = "trf"
    likelihood_ratio_threshold: float | None = None

    def __post_init__(self) -> None:
        requested = tuple(self.parameter_names)
        object.__setattr__(
            self,
            "parameter_names",
            _names(requested, field="Profile parameter names"),
        )
        for coordinate in self.parameter_names:
            from kindred.core.analysis.global_fit_execution import FitCoordinate

            if not isinstance(coordinate, FitCoordinate):
                raise TypeError(
                    "Profile coordinates require typed FitCoordinate identities."
                )
            scope = getattr(coordinate, "scope", None)
            if scope in {"coupled", "centred_deviation"}:
                raise ValueError(
                    "Coupled and centred dataset-deviation coordinates are not "
                    "public Profile targets."
                )
        confidence = _confidence(self.confidence_level)
        points = _bounded_positive_integer(
            self.max_bracketing_steps_per_side,
            field="Profile maximum bracketing steps per side",
            maximum=_PROFILE_MAX_POINTS_PER_SIDE,
        )
        evaluations = validate_evaluation_count(
            self.max_nfev_per_point,
            field="Profile evaluations per refit",
        )
        if evaluations <= 0:
            raise ValueError("Profile likelihood requires a positive evaluation budget.")
        method = validate_fit_method(self.optimizer_method)
        if not method:
            raise ValueError("Profile likelihood requires a named optimizer method.")
        threshold = self.likelihood_ratio_threshold
        if threshold is None:
            # For one profiled coordinate, chi-square(1) is the square of a
            # standard normal.  This avoids a second statistical dependency.
            threshold = NormalDist().inv_cdf((1.0 + confidence) / 2.0) ** 2
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Profile likelihood-ratio threshold must be positive.")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "max_bracketing_steps_per_side", points)
        object.__setattr__(self, "max_nfev_per_point", evaluations)
        object.__setattr__(self, "optimizer_method", method)
        object.__setattr__(self, "likelihood_ratio_threshold", threshold)


@dataclass(frozen=True, slots=True)
class ResidualBootstrapConfig:
    """Deterministic, stratified residual-bootstrap configuration."""

    parameter_names: tuple[object, ...]
    seed: int
    replicates: int = 200
    confidence_level: float = 0.95
    minimum_successful_replicates: int = 50
    minimum_success_fraction: float = 0.8
    optimizer_method: str = "trf"

    def __post_init__(self) -> None:
        requested = tuple(self.parameter_names)
        object.__setattr__(
            self,
            "parameter_names",
            _names(requested, field="Bootstrap parameter names"),
        )
        for coordinate in self.parameter_names:
            from kindred.core.analysis.global_fit_execution import FitCoordinate

            if not isinstance(coordinate, FitCoordinate):
                raise TypeError(
                    "Bootstrap coordinates require typed FitCoordinate identities."
                )
            scope = getattr(coordinate, "scope", None)
            if scope in {"coupled", "centred_deviation"}:
                raise ValueError(
                    "Coupled and centred dataset-deviation coordinates are not "
                    "residual Bootstrap targets."
                )
        seed = validate_bootstrap_seed(self.seed)
        replicates = _bounded_positive_integer(
            self.replicates,
            field="Bootstrap replicate count",
            maximum=_BOOTSTRAP_MAX_REPLICATES,
        )
        minimum = _bounded_positive_integer(
            self.minimum_successful_replicates,
            field="Bootstrap minimum successful replicates",
            maximum=_BOOTSTRAP_MAX_REPLICATES,
        )
        fraction = float(self.minimum_success_fraction)
        if minimum <= 0 or minimum > replicates:
            raise ValueError(
                "Bootstrap minimum successful replicates must be within the roster."
            )
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("Bootstrap minimum success fraction must be in (0, 1].")
        method = validate_fit_method(self.optimizer_method)
        if not method:
            raise ValueError("Residual bootstrap requires a named optimizer method.")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))
        object.__setattr__(self, "minimum_successful_replicates", minimum)
        object.__setattr__(self, "minimum_success_fraction", fraction)
        object.__setattr__(self, "optimizer_method", method)


def _sse(residuals: object) -> float:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Likelihood evaluation requires finite nonempty residuals.")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    result = float(scale * scale * np.dot(scaled, scaled))
    if not math.isfinite(result):
        raise ValueError("Likelihood residual sum of squares overflowed.")
    return result


def _objective_scientific_residuals(
    objective: Callable[[np.ndarray], np.ndarray],
    vector: np.ndarray,
) -> tuple[np.ndarray | None, FitEvaluationOutcome | None]:
    """Evaluate an objective while refusing optimizer-only penalty vectors."""

    outcome_evaluator = getattr(objective, "evaluate_outcome", None)
    if callable(outcome_evaluator):
        outcome = outcome_evaluator(vector)
        if not isinstance(outcome, FitEvaluationOutcome):
            raise TypeError("Fit objective evaluate_outcome() returned an invalid DTO.")
        if outcome.residuals is None:
            return None, outcome
        return np.asarray(outcome.residuals, dtype=float).reshape(-1), outcome
    raw = np.asarray(objective(vector), dtype=float).reshape(-1)
    return raw, None


@dataclass(frozen=True, slots=True)
class _LikelihoodRatioDetails:
    transport_value: float
    raw_value: float | None
    log_value: float | None
    outcome: str


def _likelihood_ratio_details(
    *,
    candidate_sse: float,
    best_sse: float,
    n: int,
) -> _LikelihoodRatioDetails:
    """Calculate an un-clamped finite transport value and crossing outcome.

    The profile comparison is a log likelihood ratio.  Its raw/log values are
    retained when representable.  Exact-fit and overflow limits use a finite
    transport sentinel plus a typed outcome; they are never falsely clamped to
    the configured threshold before crossing interpolation.
    """

    candidate = float(candidate_sse)
    best = float(best_sse)
    count = int(n)
    if not all(math.isfinite(value) and value >= 0.0 for value in (candidate, best)):
        raise ValueError("Likelihood SSE values must be finite and nonnegative.")
    if count <= 0:
        raise ValueError("Likelihood evaluation requires a positive observation count.")
    if candidate <= best:
        raw_ratio = 1.0 if best == 0.0 else max(0.0, candidate / best)
        return _LikelihoodRatioDetails(0.0, raw_ratio, 0.0, "finite")
    if best == 0.0:
        finite_transport = float(np.finfo(float).max / 4.0)
        return _LikelihoodRatioDetails(
            finite_transport,
            None,
            None,
            "threshold_exceeded_exact_fit",
        )
    raw_ratio = candidate / best
    raw_value = raw_ratio if math.isfinite(raw_ratio) else None
    log_ratio = float(count) * (math.log(candidate) - math.log(best))
    if not math.isfinite(log_ratio):
        finite_transport = float(np.finfo(float).max / 4.0)
        return _LikelihoodRatioDetails(
            finite_transport,
            None,
            None,
            "threshold_exceeded_overflow",
        )
    return _LikelihoodRatioDetails(log_ratio, raw_value, log_ratio, "finite")


def profile_likelihood_extension(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    initial_parameters: Mapping[object, float],
    bounds: Mapping[object, tuple[float, float]],
    config: ProfileLikelihoodConfig,
    decode_coordinate: Callable[[object, float], float],
    cancellation_check: Callable[[], bool] | None = None,
    terminal_converged: bool | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> FitTerminalExtension:
    """Fix selected coordinates and reoptimize nuisances on a bounded profile.

    Discovery moves monotonically away from the optimum and stops at the first
    successful threshold bracket.  A bracket is then refined only through
    actual nuisance refits; no endpoint is produced by interpolation.
    """

    if not isinstance(config, ProfileLikelihoodConfig):
        raise TypeError("Profile likelihood requires ProfileLikelihoodConfig.")
    if not config.parameter_names:
        raise ValueError("Ordinary Profile likelihood requires at least one coordinate.")
    forbidden_coordinates = tuple(
        item for item in config.parameter_names
        if getattr(item, "scope", None) in {"coupled", "centred_deviation"}
    )
    if forbidden_coordinates:
        raise ValueError(
            "Coupled and centred dataset-deviation coordinates cannot be public "
            "Profile targets."
        )
    initial = {name: float(value) for name, value in initial_parameters.items()}
    bound_map = {
        name: (float(value[0]), float(value[1]))
        for name, value in bounds.items()
    }
    for coordinate, (low, high) in bound_map.items():
        if not (
            math.isfinite(low)
            and math.isfinite(high)
            and low < high
        ):
            raise ValueError(
                f"Profile coordinate {coordinate!r} requires finite ordered bounds."
            )
    missing = tuple(
        item for item in config.parameter_names if item not in initial
    )
    if missing:
        raise KeyError(
            "Requested Profile optimizer coordinates are absent: "
            + ", ".join(_coordinate_display(item) for item in missing)
        )
    threshold = float(config.likelihood_ratio_threshold)
    planned_work = len(config.parameter_names) * 2 * (
        config.max_bracketing_steps_per_side + 64
    )
    work_done = 0
    active_coordinate: str | None = None
    active_side: str | None = None
    active_discovery: int | None = None
    active_refinement: int | None = None

    def emit_progress(
        *,
        status: str = "running",
        stage: str = "settlement",
        coordinate: str | None = None,
        side: str | None = None,
        discovery_step: int | None = None,
        refinement_step: int | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            FitExtensionProgress(
                phase="profile_likelihood",
                status=status,
                stage=stage,
                coordinate=coordinate,
                side=side,
                discovery_step=discovery_step,
                refinement_step=refinement_step,
                completed_work=work_done,
                planned_work=planned_work,
            )
        )

    def finish(extension: FitTerminalExtension) -> FitTerminalExtension:
        emit_progress(
            status=extension.status,
            stage="settlement",
            coordinate=active_coordinate,
            side=active_side,
            discovery_step=active_discovery,
            refinement_step=active_refinement,
        )
        return extension

    emit_progress(stage="primary_complete")

    def failure_extension(
        cause: FactualCause,
        *,
        details: Mapping[str, object] | None = None,
    ) -> FitTerminalExtension:
        return finish(
            FitTerminalExtension(
                kind="profile_likelihood",
                status="failed",
                method=_PROFILE_METHOD,
                provenance={
                    "confidence_level": config.confidence_level,
                    "likelihood_ratio_threshold": threshold,
                    "optimizer_method": config.optimizer_method,
                    "warm_start_policy": "inner_successful_refit",
                    "max_bracketing_steps_per_side": (
                        config.max_bracketing_steps_per_side
                    ),
                    "primary_fit_preserved": True,
                    "settlement": "failed",
                },
                details={
                    "requested_parameters": tuple(
                        _coordinate_display(item) for item in config.parameter_names
                    ),
                    **dict(details or {}),
                },
                cause=cause,
            )
        )

    if terminal_converged is False:
        return finish(FitTerminalExtension(
            kind="profile_likelihood",
            status="unavailable",
            method=_PROFILE_METHOD,
            provenance={
                "confidence_level": config.confidence_level,
                "likelihood_ratio_threshold": config.likelihood_ratio_threshold,
                "likelihood_model": "weighted Gaussian residuals with profiled variance",
                "optimizer_method": config.optimizer_method,
                "primary_fit_required": "converged",
            },
            details={
                "requested_parameters": tuple(
                    _coordinate_display(item) for item in config.parameter_names
                )
            },
            unavailable_reason="Profile likelihood requires a converged primary Fit.",
        ))
    try:
        base_vector = np.asarray([initial[name] for name in initial], dtype=float)
        base_residuals, base_outcome = _objective_scientific_residuals(
            objective,
            base_vector,
        )
    except (FittingCancelled, SimulationCancelled):
        raise
    except Exception as exc:
        return failure_extension(
            factual_cause_from_exception(
                exc,
                domain="fit",
                origin="profile_likelihood",
            ),
            details={"stage": "primary_profile_evaluation"},
        )
    if base_residuals is None or (
        base_outcome is not None and base_outcome.status != "converged"
    ):
        cause = (
            base_outcome.to_factual_cause()
            if base_outcome is not None
            else None
        )
        if base_outcome is not None and base_outcome.status == "fatal":
            return failure_extension(
                cause
                or FactualCause(
                    domain="fit",
                    origin="profile_likelihood",
                    code=None,
                    type="FatalProfileEvaluation",
                    message="Profile base evaluation failed fatally.",
                ),
                details={"stage": "primary_profile_evaluation"},
            )
        return finish(FitTerminalExtension(
            kind="profile_likelihood",
            status="unavailable",
            method=_PROFILE_METHOD,
            provenance={
                "confidence_level": config.confidence_level,
                "likelihood_ratio_threshold": config.likelihood_ratio_threshold,
                "likelihood_model": "weighted Gaussian residuals with profiled variance",
                "optimizer_method": config.optimizer_method,
                "primary_fit_required": "converged",
            },
            details={
                "requested_parameters": tuple(
                    _coordinate_display(item) for item in config.parameter_names
                ),
                "base_evaluation": (
                    {
                        "status": base_outcome.status,
                        "cause": cause,
                        "provenance": dict(base_outcome.provenance),
                    }
                    if base_outcome is not None
                    else {"status": "invalid"}
                ),
            },
            unavailable_reason=(
                "The primary Fit did not provide a valid converged scientific evaluation."
            ),
        ))
    base_sse = _sse(base_residuals)
    n = int(base_residuals.size)
    intervals: dict[object, object] = {}
    any_partial = False
    for name in config.parameter_names:
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        low, high = bound_map[name]
        best = initial[name]
        if not low <= best <= high:
            raise ValueError(
                f"Profile optimum for {_coordinate_display(name)!r} lies outside bounds."
            )
        active_coordinate = _coordinate_display(name)

        def side_profile(
            side: str,
            bound: float,
        ) -> tuple[float | None, str, list[Mapping[str, object]], Mapping[str, object]]:
            nonlocal work_done, active_side, active_discovery, active_refinement
            active_side = side
            records: list[Mapping[str, object]] = []
            span = abs(bound - best)
            if span == 0.0:
                return bound, "bound", records, {
                    "status": "bound",
                    "evaluation_index": None,
                    "optimizer_value": bound,
                    "coordinate_tolerance": 0.0,
                    "likelihood_ratio_tolerance": max(1e-6, 1e-4 * threshold),
                }
            inner_x = best
            inner_parameters = dict(initial)
            crossed = False
            bracket: tuple[float, float, float, Mapping[str, float]] | None = None

            def append_fit(
                fixed: float,
                *,
                stage: str,
                discovery_step: int | None,
                refinement_step: int | None,
                warm_start: Mapping[str, float],
            ) -> tuple[Mapping[str, object], object | None]:
                nonlocal work_done, active_discovery, active_refinement
                if cancellation_check is not None and cancellation_check():
                    raise FittingCancelled()
                try:
                    fit = fit_parameters_with_fixed(
                        objective,
                        dict(warm_start),
                        fixed_parameters={name: float(fixed)},
                        bounds=bound_map,
                        method=config.optimizer_method,
                        max_nfev=config.max_nfev_per_point,
                        cancellation_check=cancellation_check,
                    )
                except (FittingCancelled, SimulationCancelled):
                    raise
                except Exception as exc:
                    raise RuntimeError("Fatal Profile nuisance refit failed.") from exc
                work_done += 1
                active_discovery = discovery_step
                active_refinement = refinement_step
                emit_progress(
                    stage=stage,
                    coordinate=active_coordinate,
                    side=side,
                    discovery_step=discovery_step,
                    refinement_step=refinement_step,
                )
                evaluation = fit.evaluation
                if evaluation is not None and evaluation.status == "fatal":
                    if isinstance(evaluation.cause, BaseException):
                        raise evaluation.cause
                    raise RuntimeError("Fatal Profile nuisance evaluation failed.")
                if (
                    not fit.success
                    or evaluation is None
                    or evaluation.status != "converged"
                    or evaluation.residuals is None
                ):
                    failure = evaluation.to_factual_cause() if evaluation is not None else None
                    if failure is None:
                        failure = FactualCause(
                            domain="fit",
                            origin="profile_likelihood",
                            code=None,
                            type="NonconvergedRefit",
                            message=str(fit.message or "Profile refit did not converge."),
                            details=(
                                {
                                    "status": str(evaluation.status),
                                    "provenance": dict(evaluation.provenance),
                                }
                                if evaluation is not None
                                else {}
                            ),
                        )
                    record = {
                        "evaluation_index": len(records),
                        "stage": stage,
                        "discovery_step": discovery_step,
                        "refinement_step": refinement_step,
                        "fixed_optimizer_value": float(fixed),
                        "fixed_value": decode_coordinate(name, float(fixed)),
                        "status": "failed",
                        "warm_start_policy": "inner_successful_refit",
                        "failure": failure,
                    }
                    records.append(record)
                    return record, None
                candidate_sse = _sse(evaluation.residuals)
                lr_details = _likelihood_ratio_details(
                    candidate_sse=candidate_sse,
                    best_sse=base_sse,
                    n=n,
                )
                record = {
                    "evaluation_index": len(records),
                    "stage": stage,
                    "discovery_step": discovery_step,
                    "refinement_step": refinement_step,
                    "fixed_optimizer_value": float(fixed),
                    "fixed_value": decode_coordinate(name, float(fixed)),
                    "status": "success",
                    "warm_start_policy": "inner_successful_refit",
                    "optimizer_converged": bool(fit.success),
                    "optimized_parameters": dict(fit.parameters),
                    "weighted_residual_sse": candidate_sse,
                    "likelihood_ratio": (
                        None
                        if lr_details.outcome == "threshold_exceeded_exact_fit"
                        else lr_details.transport_value
                    ),
                    "likelihood_ratio_raw": lr_details.raw_value,
                    "likelihood_ratio_log": lr_details.log_value,
                    "likelihood_ratio_outcome": lr_details.outcome,
                }
                records.append(record)
                return record, fit.parameters

            for discovery_step in range(1, config.max_bracketing_steps_per_side + 1):
                fixed = (
                    bound
                    if discovery_step == config.max_bracketing_steps_per_side
                    else best
                    + (bound - best)
                    * (
                        float(discovery_step)
                        / float(config.max_bracketing_steps_per_side)
                    )
                )
                if fixed == inner_x:
                    break
                record, fitted_parameters = append_fit(
                    fixed,
                    stage="discovery",
                    discovery_step=discovery_step,
                    refinement_step=None,
                    warm_start=inner_parameters,
                )
                if record["status"] != "success":
                    return None, "failed", records, {
                        "status": "failed",
                        "evaluation_index": record["evaluation_index"],
                        "reason": "scientific_unavailable_or_nonconverged",
                    }
                outcome = str(record["likelihood_ratio_outcome"])
                if outcome == "threshold_exceeded_exact_fit":
                    return best, "limiting", records, {
                        "status": "limiting",
                        "evaluation_index": record["evaluation_index"],
                        "optimizer_value": best,
                        "likelihood_ratio_tolerance": max(1e-6, 1e-4 * threshold),
                    }
                lr = float(record["likelihood_ratio"])
                if lr >= threshold:
                    bracket = (
                        inner_x,
                        float(fixed),
                        lr,
                        dict(inner_parameters),
                    )
                    crossed = True
                    break
                inner_x = float(fixed)
                inner_parameters = dict(fitted_parameters or inner_parameters)
            if not crossed:
                if records and records[-1]["status"] == "success":
                    last_x = float(records[-1]["fixed_optimizer_value"])
                    if last_x == bound:
                        return bound, "bound", records, {
                            "status": "bound",
                            "evaluation_index": records[-1]["evaluation_index"],
                            "optimizer_value": bound,
                            "coordinate_tolerance": 0.0,
                            "likelihood_ratio_tolerance": max(1e-6, 1e-4 * threshold),
                        }
                return None, "unavailable", records, {
                    "status": "unavailable",
                    "reason": "threshold_not_bracketed_within_discovery_budget",
                }

            assert bracket is not None
            left_x, right_x, right_lr, left_parameters = bracket
            lr_tolerance = max(1e-6, 1e-4 * threshold)
            best_candidate: tuple[float, int] | None = None
            representability_exhausted = False
            for refinement_step in range(1, 65):
                coordinate_scale = max(
                    abs(best),
                    abs(bound),
                    abs(bound - best),
                )
                coordinate_tolerance = max(
                    1e-6 * coordinate_scale,
                    32.0
                    * max(
                        math.ulp(float(left_x)),
                        math.ulp(float(right_x)),
                    ),
                )
                midpoint = left_x + (right_x - left_x) / 2.0
                if midpoint == left_x or midpoint == right_x:
                    representability_exhausted = True
                    break
                record, fitted_parameters = append_fit(
                    midpoint,
                    stage="refinement",
                    discovery_step=None,
                    refinement_step=refinement_step,
                    warm_start=left_parameters,
                )
                if record["status"] != "success":
                    return None, "failed", records, {
                        "status": "failed",
                        "evaluation_index": record["evaluation_index"],
                        "reason": "scientific_unavailable_or_nonconverged",
                    }
                if str(record["likelihood_ratio_outcome"]) == "threshold_exceeded_exact_fit":
                    return best, "limiting", records, {
                        "status": "limiting",
                        "evaluation_index": record["evaluation_index"],
                        "optimizer_value": best,
                        "likelihood_ratio_tolerance": lr_tolerance,
                    }
                midpoint_lr = float(record["likelihood_ratio"])
                width = abs(right_x - left_x)
                if abs(midpoint_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return midpoint, "threshold", records, {
                        "status": "threshold",
                        "evaluation_index": record["evaluation_index"],
                        "optimizer_value": midpoint,
                        "likelihood_ratio": midpoint_lr,
                        "coordinate_tolerance": coordinate_tolerance,
                        "likelihood_ratio_tolerance": lr_tolerance,
                        "bracket_width": width,
                    }
                if abs(right_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return right_x, "threshold", records, {
                        "status": "threshold",
                        "evaluation_index": next(
                            int(item["evaluation_index"])
                            for item in reversed(records)
                            if float(item["fixed_optimizer_value"]) == right_x
                        ),
                        "optimizer_value": right_x,
                        "likelihood_ratio": right_lr,
                        "coordinate_tolerance": coordinate_tolerance,
                        "likelihood_ratio_tolerance": lr_tolerance,
                        "bracket_width": width,
                    }
                if midpoint_lr >= threshold:
                    right_x, right_lr = midpoint, midpoint_lr
                else:
                    left_x = midpoint
                    left_parameters = dict(fitted_parameters or left_parameters)
                best_candidate = (midpoint, int(record["evaluation_index"]))
            return None, "unavailable", records, {
                "status": "unavailable",
                "reason": (
                    "binary64_representability_exhausted"
                    if representability_exhausted
                    else "midpoint_refit_budget_exhausted"
                ),
                "coordinate_tolerance": coordinate_tolerance,
                "likelihood_ratio_tolerance": lr_tolerance,
                "bracket_width": abs(right_x - left_x),
                "last_evaluation_index": (
                    None if best_candidate is None else best_candidate[1]
                ),
            }

        side_results: dict[str, tuple[float | None, str, list[Mapping[str, object]], Mapping[str, object]]] = {}
        for side, bound in (("lower", low), ("upper", high)):
            try:
                side_results[side] = side_profile(side, bound)
            except (FittingCancelled, SimulationCancelled):
                raise
            except Exception as exc:
                return failure_extension(
                    factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                    details={
                        "coordinate": _coordinate_display(name),
                        "side": side,
                        "evaluation_count": len(side_results.get(side, (None, None, (), {}))[2]),
                    },
                )
        lower, lower_outcome, lower_records, lower_resolution = side_results["lower"]
        upper, upper_outcome, upper_records, upper_resolution = side_results["upper"]
        endpoint_messages: dict[str, str] = {}
        for side, outcome, bound in (("lower", lower_outcome, low), ("upper", upper_outcome, high)):
            if outcome == "bound":
                endpoint_messages[side] = (
                    f"{_coordinate_display(name)} reached its configured {side} bound; "
                    "the profile is bound-limited."
                )
            elif outcome == "failed":
                endpoint_messages[side] = (
                    f"{_coordinate_display(name)} {side} profile stopped at the first "
                    "scientifically unavailable refit; no later trial was bridged."
                )
        if lower is not None and upper is not None:
            interval_status = "available"
            if lower_outcome == upper_outcome == "threshold":
                interval_kind = "two-sided"
            elif lower_outcome == upper_outcome == "limiting":
                interval_kind = "exact-fit limiting"
            else:
                interval_kind = "bound-limited"
        elif lower is not None or upper is not None:
            interval_status = "partial"
            interval_kind = (
                "bound-limited"
                if lower_outcome == "bound" or upper_outcome == "bound"
                else "one-sided"
            )
            any_partial = True
        else:
            interval_status = "unavailable"
            interval_kind = "unavailable"
            any_partial = True
        intervals[name] = {
            "status": interval_status,
            "interval_kind": interval_kind,
            "estimate": decode_coordinate(name, best),
            "lower": None if lower is None else decode_coordinate(name, lower),
            "upper": None if upper is None else decode_coordinate(name, upper),
            "lower_outcome": lower_outcome,
            "upper_outcome": upper_outcome,
            "optimizer_bounds": (low, high),
            "physical_bounds": (
                decode_coordinate(name, low),
                decode_coordinate(name, high),
            ),
            "evaluations": {
                "lower": tuple(lower_records),
                "upper": tuple(upper_records),
            },
            "resolution": {
                "lower": lower_resolution,
                "upper": upper_resolution,
                "messages": endpoint_messages,
            },
        }

    interval_statuses = {
        str(item.get("status"))
        for item in intervals.values()
        if isinstance(item, Mapping)
    }
    if interval_statuses == {"unavailable"}:
        extension_status = "unavailable"
        unavailable_reason = "No requested profile interval could be determined."
    elif any_partial:
        extension_status = "partial"
        unavailable_reason = None
    else:
        extension_status = "available"
        unavailable_reason = None
    return finish(FitTerminalExtension(
        kind="profile_likelihood",
        status=extension_status,
        method=_PROFILE_METHOD,
        provenance={
            "confidence_level": config.confidence_level,
            "likelihood_ratio_threshold": config.likelihood_ratio_threshold,
            "likelihood_model": "weighted Gaussian residuals with profiled variance",
            "likelihood_ratio_transport": (
                "finite un-clamped log-ratio with typed exact-fit/overflow crossing outcome"
            ),
            "fixed_coordinate_policy": "selected coordinate fixed; all others reoptimized",
            "warm_start_policy": "inner_successful_refit",
            "optimizer_method": config.optimizer_method,
            "max_bracketing_steps_per_side": config.max_bracketing_steps_per_side,
            "max_nfev_per_point": config.max_nfev_per_point,
            "observation_count": n,
            "terminal_weighted_residual_sse": base_sse,
        },
        details={"intervals": MappingProxyType(intervals)},
        unavailable_reason=unavailable_reason,
    ))


def _parameter_value(result: object, name: object) -> float:
    """Resolve one typed uncertainty coordinate from terminal Fit custody."""

    from kindred.core.analysis.global_fit_execution import FitCoordinate

    if not isinstance(name, FitCoordinate):
        raise TypeError("Bootstrap coordinates require typed FitCoordinate identities.")
    scope = getattr(name, "scope", None)
    parameter_name = str(getattr(name, "name", "") or "").strip()
    if scope not in {
        "shared",
        "dataset",
        "coupled",
        "centred_overall",
        "centred_deviation",
    }:
        raise TypeError("Bootstrap coordinates require a typed FitCoordinate scope.")
    if not parameter_name:
        raise ValueError("Bootstrap coordinates require a nonblank parameter name.")
    if scope in {"coupled", "centred_deviation"}:
        raise ValueError(
            "Coupled and centred dataset-deviation coordinates cannot be bootstrapped."
        )
    shared = getattr(result, "shared_params", None)
    datasets = getattr(result, "dataset_params", None)
    if scope == "shared":
        if not isinstance(shared, Mapping) or parameter_name not in shared:
            raise KeyError(name)
        value = float(shared[parameter_name])
    elif scope == "dataset":
        dataset_id = str(getattr(name, "dataset_id", "") or "").strip()
        if not dataset_id or not isinstance(datasets, Mapping):
            raise KeyError(name)
        values = datasets.get(dataset_id)
        if not isinstance(values, Mapping) or parameter_name not in values:
            raise KeyError(name)
        value = float(values[parameter_name])
    elif scope == "centred_overall":
        settlements = getattr(result, "centred_fixed_effects_settlements", ())
        matching = tuple(
            item
            for item in settlements
            if getattr(item, "parameter_name", None) == parameter_name
        )
        if len(matching) != 1:
            raise KeyError(name)
        value = float(matching[0].overall_decoded_value)
    else:
        raise KeyError(name)
    if not math.isfinite(value):
        raise ValueError(
            f"Bootstrap Fit returned non-finite parameter {_coordinate_display(name)!r}."
        )
    return value


_BootstrapRefitOutcome = tuple[
    _fit_execution.GlobalFitResult | None,
    FitEvaluationOutcome,
]


def _refit_converged(
    result: object,
    evaluation: FitEvaluationOutcome,
) -> tuple[bool, Mapping[str, object] | None]:
    """Require an explicit typed outcome before using refit values."""

    if not isinstance(evaluation, FitEvaluationOutcome):
        raise TypeError("Bootstrap refits require their typed evaluation outcome.")
    if evaluation.status == "fatal":
        cause = evaluation.cause
        if isinstance(cause, FittingCancelled):
            raise cause
        return False, {
            "status": "fatal",
            "provenance": dict(evaluation.provenance),
            "cause": evaluation.to_factual_cause()
            or FactualCause(
                domain="fit",
                origin="bootstrap",
                code=None,
                type="FatalBootstrapRefit",
                message="Fatal Bootstrap refit omitted its exception.",
            ),
        }
    if evaluation.status == "cancelled":
        cause = evaluation.cause
        if isinstance(cause, FittingCancelled):
            raise cause
        raise FittingCancelled()
    if evaluation.status == "scientific_unavailable":
        return False, {
            "status": evaluation.status,
            "provenance": dict(evaluation.provenance),
            "cause": evaluation.to_factual_cause(),
        }
    if evaluation.status == "converged":
        if result is None:
            raise RuntimeError("Converged Bootstrap refit omitted its result.")
        return True, None

    completion = getattr(result, "completion", None)
    if completion is not None:
        converged = bool(getattr(completion, "optimizer_converged", False))
        if converged:
            return True, None
        message = str(
            getattr(result, "message", "") or "Fit refit did not converge."
        )
        optimizer_diagnostic = getattr(completion, "optimizer_diagnostic", None)
        return False, {
            "status": "nonconverged",
            "message": message,
            "cause": FactualCause(
                domain="fit",
                origin="bootstrap",
                code=None,
                type="NonconvergedRefit",
                message=message,
                cause=(
                    optimizer_diagnostic.failure
                    if optimizer_diagnostic is not None
                    else None
                ),
            ),
        }
    return False, {
        "status": "nonconverged",
        "message": "Bootstrap refit did not expose an explicit convergence outcome.",
        "cause": FactualCause(
            domain="fit",
            origin="bootstrap",
            code=None,
            type="MissingConvergenceOutcome",
            message="Bootstrap refit did not expose an explicit convergence outcome.",
        ),
    }


_BOOTSTRAP_STRATUM_SCHEMA_VERSION = 1
_BOOTSTRAP_DRAW_SCHEMA_VERSION = 2


def _bootstrap_stratum_manifest(
    payloads: Sequence[FitDatasetSpec],
) -> Mapping[str, object]:
    """Capture invariant stratum structure once for all replicates."""

    manifest: dict[str, object] = {}
    for payload in payloads:
        targets: dict[str, object] = {}
        for target in payload.species_list:
            source = payload.observations[target]
            targets[target] = {
                "raw_observation_id": source.raw_observation_id,
                "raw_column": source.raw_column,
                "time": tuple(float(value) for value in source.t),
                "source_rows": tuple(int(value) for value in source.source_rows),
                "target_weight": float(payload.target_weights[target]),
                "x_name": payload.x_name,
            }
        if payload.x_name != "t" and payload.x_observations is not None:
            x_source = payload.x_observations
            targets["__x__"] = {
                "raw_observation_id": x_source.raw_observation_id,
                "raw_column": x_source.raw_column,
                "time": tuple(float(value) for value in x_source.t),
                "source_rows": tuple(int(value) for value in x_source.source_rows),
                "target_weight": float(payload.target_weights[payload.x_name]),
                "x": tuple(float(value) for value in x_source.y),
                "x_name": payload.x_name,
            }
        manifest[payload.dataset_id] = targets
    return manifest


def _bootstrap_payloads(
    *,
    payloads: Sequence[FitDatasetSpec],
    fitted_model: Mapping[str, Mapping[str, np.ndarray]],
    fitted_x_model: Mapping[str, ParametricXProjection] | None,
    rng: np.random.Generator,
) -> tuple[FitDatasetSpec, ...]:
    captured: list[FitDatasetSpec] = []
    for payload in payloads:
        dataset_model = fitted_model.get(payload.dataset_id)
        if not isinstance(dataset_model, Mapping):
            raise ValueError(
                f"Bootstrap terminal model omitted dataset {payload.dataset_id!r}."
            )
        observations: dict[str, FitObservationSeries] = {}
        for target in payload.species_list:
            source = payload.observations[target]
            model = np.asarray(dataset_model.get(target), dtype=float).reshape(-1)
            observed = np.asarray(source.y, dtype=float).reshape(-1)
            if model.size != observed.size or not np.all(np.isfinite(model)):
                raise ValueError(
                    f"Bootstrap terminal model is misaligned for "
                    f"{payload.dataset_id!r}/{target!r}."
                )
            residuals = observed - model
            centred_residuals = residuals - float(np.mean(residuals))
            indexes = rng.integers(0, centred_residuals.size, size=centred_residuals.size)
            resampled_y = model + centred_residuals[indexes]
            observations[target] = FitObservationSeries(
                raw_observation_id=source.raw_observation_id,
                raw_column=source.raw_column,
                t=source.t,
                y=resampled_y,
                source_rows=source.source_rows,
            )
        x_observations = None
        if payload.x_name != "t":
            if payload.x_observations is None or fitted_x_model is None:
                raise ValueError(
                    f"Bootstrap dataset {payload.dataset_id!r} omitted its canonical X model."
                )
            projection = fitted_x_model.get(payload.dataset_id)
            if projection is None:
                raise ValueError(
                    f"Bootstrap dataset {payload.dataset_id!r} omitted its X projection."
                )
            x_source = payload.x_observations
            observed_x = np.asarray(
                [float(x_source.y[x_source.source_rows.index(row)]) for row in projection.source_rows],
                dtype=float,
            )
            x_residuals = observed_x - np.asarray(projection.selected_model_x, dtype=float)
            centered_x = x_residuals - float(np.mean(x_residuals))
            indexes = rng.integers(0, centered_x.size, size=centered_x.size)
            synthetic_x = np.asarray(projection.selected_model_x, dtype=float) + centered_x[indexes]
            values_by_row = dict(zip(projection.source_rows, synthetic_x, strict=True))
            full_values = np.asarray(
                [values_by_row.get(int(row), float(value)) for row, value in zip(x_source.source_rows, x_source.y, strict=True)],
                dtype=float,
            )
            x_observations = FitObservationSeries(
                raw_observation_id=x_source.raw_observation_id,
                raw_column=x_source.raw_column,
                t=x_source.t,
                y=full_values,
                source_rows=x_source.source_rows,
            )
        captured.append(
            FitDatasetSpec(
                dataset_id=payload.dataset_id,
                species_list=payload.species_list,
                observations=observations,
                target_weights=payload.target_weights,
                x_name=payload.x_name,
                x_mode=payload.x_mode,
                x_observations=x_observations,
            )
        )
    return tuple(captured)


def residual_bootstrap_extension(
    *,
    terminal_result: object,
    terminal_outcome: FitEvaluationOutcome,
    payloads: Sequence[FitDatasetSpec],
    config: ResidualBootstrapConfig,
    refit: Callable[[Sequence[FitDatasetSpec], int], _BootstrapRefitOutcome],
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> FitTerminalExtension:
    """Run deterministic full refits for stratified residual replicates."""

    if not isinstance(config, ResidualBootstrapConfig):
        raise TypeError("Residual bootstrap requires ResidualBootstrapConfig.")
    if not config.parameter_names:
        raise ValueError("Ordinary residual Bootstrap requires at least one coordinate.")
    primary_converged, primary_failure = _refit_converged(
        terminal_result,
        terminal_outcome,
    )
    def emit_progress(
        *,
        status: str = "running",
        stage: str = "replicate",
        completed: int = 0,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            FitExtensionProgress(
                phase="bootstrap",
                status=status,
                stage=stage,
                completed_work=completed,
                planned_work=config.replicates,
                completed_replicates=completed,
                planned_replicates=config.replicates,
            )
        )

    def finish(extension: FitTerminalExtension) -> FitTerminalExtension:
        emit_progress(
            status=extension.status,
            stage="settlement",
            completed=len(replicate_records),
        )
        return extension

    replicate_records: list[Mapping[str, object]] = []
    emit_progress(stage="primary_complete")
    if not primary_converged:
        return finish(FitTerminalExtension(
            kind="bootstrap",
            status="unavailable",
            method=_BOOTSTRAP_METHOD,
            provenance={
                "seed": config.seed,
                "replicates": config.replicates,
                "confidence_level": config.confidence_level,
                "resampling": "within each dataset/target residual stratum",
                "primary_fit_required": "converged",
            },
            details={
                "requested_parameters": tuple(
                    _coordinate_display(item) for item in config.parameter_names
                ),
                "primary_fit": primary_failure or {"status": "nonconverged"},
            },
            unavailable_reason="Residual bootstrap requires a converged primary Fit.",
        ))
    fitted_model = getattr(terminal_result, "model_series", None)
    if not isinstance(fitted_model, Mapping) or not fitted_model:
        raise ValueError(
            "The terminal Fit did not retain aligned model series for Bootstrap."
        )
    fitted_x_model = getattr(terminal_result, "model_x_projection_by_dataset", None)
    for name in config.parameter_names:
        _parameter_value(terminal_result, name)

    stratum_manifest = _bootstrap_stratum_manifest(payloads)
    rng = np.random.default_rng(config.seed)
    successes: dict[str, list[float]] = {
        name: [] for name in config.parameter_names
    }
    for index in range(config.replicates):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        synthetic = _bootstrap_payloads(
            payloads=payloads,
            fitted_model=fitted_model,
            fitted_x_model=fitted_x_model,
            rng=rng,
        )
        try:
            fitted, fitted_outcome = refit(synthetic, index)
        except (FittingCancelled, SimulationCancelled):
            raise
        except Exception as exc:
            return finish(FitTerminalExtension(
                kind="bootstrap",
                status="failed",
                method=_BOOTSTRAP_METHOD,
                provenance={
                    "seed": config.seed,
                    "replicates": config.replicates,
                    "primary_fit_preserved": True,
                    "settlement": "failed",
                },
                details={
                    "replicate": index,
                    "replicate_roster": tuple(replicate_records),
                },
                cause=factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="bootstrap",
                ),
            ))
        converged, failure = _refit_converged(fitted, fitted_outcome)
        if not converged:
            if isinstance(failure, Mapping) and failure.get("status") == "fatal":
                raw_cause = failure.get("cause")
                cause = (
                    raw_cause
                    if isinstance(raw_cause, FactualCause)
                    else FactualCause(
                        domain="fit",
                        origin="bootstrap",
                        code=None,
                        type="FatalBootstrapRefit",
                        message="Fatal Bootstrap refit failed.",
                    )
                )
                return finish(FitTerminalExtension(
                    kind="bootstrap",
                    status="failed",
                    method=_BOOTSTRAP_METHOD,
                    provenance={
                        "seed": config.seed,
                        "replicates": config.replicates,
                        "primary_fit_preserved": True,
                        "settlement": "failed",
                    },
                    details={
                        "replicate": index,
                        "replicate_roster": tuple(replicate_records),
                    },
                    cause=cause,
                ))
            replicate_records.append(
                {
                    "replicate": index,
                    "status": "failed",
                    "failure": (
                        failure.get("cause")
                        if isinstance(failure, Mapping)
                        and isinstance(failure.get("cause"), FactualCause)
                        else FactualCause(
                            domain="fit",
                            origin="bootstrap",
                            code=None,
                            type="NonconvergedRefit",
                            message="Bootstrap refit did not converge.",
                        )
                    ),
                }
            )
            emit_progress(completed=index + 1)
            continue
        if fitted is None:
            raise RuntimeError("Converged Bootstrap refit omitted its result.")
        values = {
            name: _parameter_value(fitted, name)
            for name in config.parameter_names
        }
        for name, value in values.items():
            successes[name].append(value)
        replicate_records.append(
            {
                "replicate": index,
                "status": "success",
                "parameters": values,
            }
        )
        emit_progress(completed=index + 1)

    success_count = sum(item["status"] == "success" for item in replicate_records)
    failure_count = len(replicate_records) - success_count
    required = max(
        config.minimum_successful_replicates,
        int(math.ceil(config.replicates * config.minimum_success_fraction)),
    )
    valid = success_count >= required
    alpha = 1.0 - config.confidence_level
    intervals: dict[str, object] = {}
    for name, values in successes.items():
        if not valid:
            intervals[name] = {
                "status": "unavailable",
                "lower": None,
                "upper": None,
                "reason": (
                    f"Only {success_count} of {config.replicates} full refits "
                    f"succeeded; {required} are required."
                ),
            }
            continue
        interval = np.percentile(
            np.asarray(values, dtype=float),
            [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
        )
        intervals[name] = {
            "status": "available",
            "lower": float(interval[0]),
            "upper": float(interval[1]),
            "construction": "equal-tailed percentile interval",
            "successful_values": tuple(float(value) for value in values),
        }

    return finish(FitTerminalExtension(
        kind="bootstrap",
        status="available" if valid else "unavailable",
        method=_BOOTSTRAP_METHOD,
        provenance={
            "seed": config.seed,
            "replicates": config.replicates,
            "confidence_level": config.confidence_level,
            "resampling": "within each dataset/target residual stratum",
            "stratum_manifest_version": _BOOTSTRAP_STRATUM_SCHEMA_VERSION,
            "stratum_manifest": stratum_manifest,
            "draw_recipe_version": _BOOTSTRAP_DRAW_SCHEMA_VERSION,
            "draw_algorithm": (
                "numpy.default_rng(seed).integers(0, stratum_residual_count); "
                "integer draws sample independently mean-centered residuals"
            ),
            "residual_centering": (
                "independent arithmetic mean subtraction within each dataset/target stratum"
            ),
            "replay_authority": "seed plus versioned draw algorithm and stratum manifest",
            "preserved_fields": (
                "time",
                "X",
                "source-row",
                "dataset weight",
                "target weight",
                "full-refit",
            ),
            "replicate_execution": "full global refit",
            "refit_validity": "only explicitly converged scientific refits count",
            "minimum_successful_replicates": config.minimum_successful_replicates,
            "minimum_success_fraction": config.minimum_success_fraction,
            "required_successes": required,
            "successful_replicates": success_count,
            "failed_replicates": failure_count,
            "interval_construction": "equal-tailed percentile",
        },
        details={
            "intervals": intervals,
            "replicate_roster": tuple(replicate_records),
        },
        unavailable_reason=(
            None
            if valid
            else (
                f"Bootstrap validity threshold was not met: {success_count} "
                f"successful full refits; {required} required."
            )
        ),
    ))

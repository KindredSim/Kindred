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
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.exceptions import (
    FitFinalReplayError,
    FitSimulationError,
    FittingCancelled,
)
from kindred.core.fitting_completion import FitTerminalExtension
from kindred.core.fitting_optimization import fit_parameters_with_fixed
from kindred.core.project_document import validate_fit_method
from kindred.core.simulation_failure import simulation_failure_from_exception

__all__ = [
    "ProfileLikelihoodConfig",
    "ResidualBootstrapConfig",
    "profile_likelihood_extension",
    "residual_bootstrap_extension",
]


_PROFILE_METHOD = "weighted-Gaussian profile likelihood"
_BOOTSTRAP_METHOD = "stratified within-dataset/target residual bootstrap"


def _is_candidate_domain_failure(exc: BaseException) -> bool:
    details = getattr(exc, "details", None)
    return bool(
        isinstance(exc, FitSimulationError)
        and isinstance(details, Mapping)
        and details.get("candidate_domain") is True
        and details.get("fatal") is False
    )


def _confidence(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("Confidence level must lie strictly between zero and one.")
    return result


def _names(values: Sequence[object], *, field: str) -> tuple[str, ...]:
    names = tuple(str(value or "").strip() for value in values)
    if not names or any(not value for value in names) or len(names) != len(set(names)):
        raise ValueError(f"{field} must contain unique nonblank names.")
    return names


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodConfig:
    """Configuration recorded with a weighted-Gaussian profile interval."""

    parameter_names: tuple[str, ...]
    confidence_level: float = 0.95
    points_per_side: int = 8
    max_nfev_per_point: int = 300
    optimizer_method: str = "trf"
    likelihood_ratio_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_names",
            _names(self.parameter_names, field="Profile parameter names"),
        )
        confidence = _confidence(self.confidence_level)
        points = int(self.points_per_side)
        evaluations = int(self.max_nfev_per_point)
        method = validate_fit_method(self.optimizer_method)
        if points < 2:
            raise ValueError("Profile likelihood requires at least two points per side.")
        if evaluations <= 0:
            raise ValueError("Profile likelihood requires a positive evaluation budget.")
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
        object.__setattr__(self, "points_per_side", points)
        object.__setattr__(self, "max_nfev_per_point", evaluations)
        object.__setattr__(self, "optimizer_method", method)
        object.__setattr__(self, "likelihood_ratio_threshold", threshold)


@dataclass(frozen=True, slots=True)
class ResidualBootstrapConfig:
    """Deterministic, stratified residual-bootstrap configuration."""

    parameter_names: tuple[str, ...]
    seed: int
    replicates: int = 200
    confidence_level: float = 0.95
    minimum_successful_replicates: int = 50
    minimum_success_fraction: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_names",
            _names(self.parameter_names, field="Bootstrap parameter names"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("Bootstrap seed must be an integer.")
        replicates = int(self.replicates)
        minimum = int(self.minimum_successful_replicates)
        fraction = float(self.minimum_success_fraction)
        if replicates <= 0:
            raise ValueError("Bootstrap replicate count must be positive.")
        if minimum <= 0 or minimum > replicates:
            raise ValueError(
                "Bootstrap minimum successful replicates must be within the roster."
            )
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("Bootstrap minimum success fraction must be in (0, 1].")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))
        object.__setattr__(self, "minimum_successful_replicates", minimum)
        object.__setattr__(self, "minimum_success_fraction", fraction)


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


def _profile_grid(best: float, low: float, high: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    if not all(math.isfinite(value) for value in (best, low, high)):
        raise ValueError("Profile likelihood requires finite coordinate bounds.")
    if not low <= best <= high or low >= high:
        raise ValueError("Profile likelihood received inconsistent coordinate bounds.")
    lower = np.linspace(best, low, count + 1, dtype=float)[1:]
    upper = np.linspace(best, high, count + 1, dtype=float)[1:]
    return lower, upper


def _crossing(
    *,
    best: float,
    bound: float,
    evaluations: Sequence[Mapping[str, object]],
    threshold: float,
) -> tuple[float | None, str]:
    previous_x = float(best)
    previous_lr = 0.0
    saw_success = False
    for item in evaluations:
        if item["status"] != "success":
            # A profile side is only a continuous chain of valid refits.
            # Later points cannot bridge an invalid/nonconverged region.
            return None, "failed"
        saw_success = True
        x = float(item["fixed_optimizer_value"])
        outcome = str(item.get("likelihood_ratio_outcome", "finite"))
        if outcome == "threshold_exceeded_exact_fit":
            # With an exact fit, every positive SSE has an infinite limiting
            # likelihood ratio.  The truthful finite representation is the
            # limiting point at the optimum, never the first grid sample.
            return float(best), "limiting"
        lr = float(item["likelihood_ratio"])
        if lr >= threshold:
            if outcome != "finite":
                return x, "threshold"
            width = lr - previous_lr
            if width <= 0.0:
                return x, "threshold"
            fraction = (threshold - previous_lr) / width
            return float(previous_x + fraction * (x - previous_x)), "threshold"
        previous_x = x
        previous_lr = lr
    if saw_success and math.isclose(previous_x, bound, rel_tol=0.0, abs_tol=1e-12):
        return float(bound), "bound"
    return None, "unavailable"


def profile_likelihood_extension(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    initial_parameters: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    config: ProfileLikelihoodConfig,
    decode_coordinate: Callable[[str, float], float],
    cancellation_check: Callable[[], bool] | None = None,
    terminal_converged: bool | None = None,
) -> FitTerminalExtension:
    """Fix each selected coordinate and reoptimize every other coordinate."""

    if not isinstance(config, ProfileLikelihoodConfig):
        raise TypeError("Profile likelihood requires ProfileLikelihoodConfig.")
    initial = {str(name): float(value) for name, value in initial_parameters.items()}
    bound_map = {
        str(name): (float(value[0]), float(value[1]))
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
    missing = sorted(set(config.parameter_names) - set(initial))
    if missing:
        raise KeyError(
            "Requested Profile optimizer coordinates are absent: "
            + ", ".join(missing)
        )
    if terminal_converged is False:
        return FitTerminalExtension(
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
            details={"requested_parameters": config.parameter_names},
            unavailable_reason="Profile likelihood requires a converged primary Fit.",
        )
    base_vector = np.asarray([initial[name] for name in initial], dtype=float)
    base_residuals, base_outcome = _objective_scientific_residuals(
        objective,
        base_vector,
    )
    if base_residuals is None or (
        base_outcome is not None and base_outcome.status != "converged"
    ):
        cause_payload = (
            dict(base_outcome.cause_payload())
            if base_outcome is not None and base_outcome.cause_payload() is not None
            else {}
        )
        return FitTerminalExtension(
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
                "requested_parameters": config.parameter_names,
                "base_evaluation": (
                    {
                        "status": base_outcome.status,
                        "cause": cause_payload,
                        "provenance": dict(base_outcome.provenance),
                    }
                    if base_outcome is not None
                    else {"status": "invalid"}
                ),
            },
            unavailable_reason=(
                "The primary Fit did not provide a valid converged scientific evaluation."
            ),
        )
    base_sse = _sse(base_residuals)
    n = int(base_residuals.size)
    intervals: dict[str, object] = {}
    any_partial = False

    for name in config.parameter_names:
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        low, high = bound_map[name]
        best = initial[name]
        lower_grid, upper_grid = _profile_grid(
            best,
            low,
            high,
            config.points_per_side,
        )

        side_records: dict[str, list[Mapping[str, object]]] = {"lower": [], "upper": []}
        warm_start = dict(initial)
        for side, grid in (("lower", lower_grid), ("upper", upper_grid)):
            warm_start = dict(initial)
            for fixed in grid:
                if cancellation_check is not None and cancellation_check():
                    raise FittingCancelled()
                try:
                    fit = fit_parameters_with_fixed(
                        objective,
                        warm_start,
                        fixed_parameters={name: float(fixed)},
                        bounds=bound_map,
                        method=config.optimizer_method,
                        max_nfev=config.max_nfev_per_point,
                        cancellation_check=cancellation_check,
                    )
                except FitFinalReplayError as exc:
                    typed_cause = exc.typed_cause
                    side_records[side].append(
                        {
                            "fixed_optimizer_value": float(fixed),
                            "fixed_value": decode_coordinate(name, float(fixed)),
                            "status": "failed",
                            "failure": simulation_failure_from_exception(typed_cause),
                        }
                    )
                    continue
                except FitSimulationError as exc:
                    if not _is_candidate_domain_failure(exc):
                        raise
                    side_records[side].append(
                        {
                            "fixed_optimizer_value": float(fixed),
                            "fixed_value": decode_coordinate(name, float(fixed)),
                            "status": "failed",
                            "failure": simulation_failure_from_exception(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    typed_cause = getattr(exc, "typed_cause", None)
                    if not isinstance(typed_cause, FitSimulationError):
                        raise
                    side_records[side].append(
                        {
                            "fixed_optimizer_value": float(fixed),
                            "fixed_value": decode_coordinate(name, float(fixed)),
                            "status": "failed",
                            "failure": simulation_failure_from_exception(typed_cause),
                        }
                    )
                    continue
                evaluation = fit.evaluation
                if (
                    not fit.success
                    or evaluation is None
                    or evaluation.status != "converged"
                    or evaluation.residuals is None
                ):
                    failure: dict[str, object] = {
                        "kind": "nonconverged_refit"
                        if evaluation is None
                        or evaluation.status == "nonconverged"
                        else str(evaluation.status),
                        "message": str(fit.message),
                    }
                    if evaluation is not None:
                        failure["provenance"] = dict(evaluation.provenance)
                        cause = evaluation.cause_payload()
                        if cause is not None:
                            failure["cause"] = dict(cause)
                    side_records[side].append(
                        {
                            "fixed_optimizer_value": float(fixed),
                            "fixed_value": decode_coordinate(name, float(fixed)),
                            "status": "failed",
                            "failure": failure,
                        }
                    )
                    # A failed refit is not a warm-start source for a later
                    # coordinate on the same profile side.
                    continue
                candidate_sse = _sse(evaluation.residuals)
                likelihood_ratio_details = _likelihood_ratio_details(
                    candidate_sse=candidate_sse,
                    best_sse=base_sse,
                    n=n,
                )
                likelihood_ratio = (
                    None
                    if likelihood_ratio_details.outcome == "threshold_exceeded_exact_fit"
                    else likelihood_ratio_details.transport_value
                )
                warm_start = dict(fit.parameters)
                side_records[side].append(
                    {
                        "fixed_optimizer_value": float(fixed),
                        "fixed_value": decode_coordinate(name, float(fixed)),
                        "status": "success",
                        "optimizer_converged": bool(fit.success),
                        "optimized_parameters": dict(fit.parameters),
                        "weighted_residual_sse": candidate_sse,
                        "likelihood_ratio": likelihood_ratio,
                        "likelihood_ratio_raw": likelihood_ratio_details.raw_value,
                        "likelihood_ratio_log": likelihood_ratio_details.log_value,
                        "likelihood_ratio_outcome": likelihood_ratio_details.outcome,
                    }
                )

        lower, lower_outcome = _crossing(
            best=best,
            bound=low,
            evaluations=side_records["lower"],
            threshold=float(config.likelihood_ratio_threshold),
        )
        upper, upper_outcome = _crossing(
            best=best,
            bound=high,
            evaluations=side_records["upper"],
            threshold=float(config.likelihood_ratio_threshold),
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
            interval_kind = "one-sided"
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
                "lower": tuple(side_records["lower"]),
                "upper": tuple(side_records["upper"]),
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
    return FitTerminalExtension(
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
            "optimizer_method": config.optimizer_method,
            "points_per_side": config.points_per_side,
            "max_nfev_per_point": config.max_nfev_per_point,
            "observation_count": n,
            "terminal_weighted_residual_sse": base_sse,
        },
        details={"intervals": MappingProxyType(intervals)},
        unavailable_reason=unavailable_reason,
    )


def _parameter_value(result: object, name: str) -> float:
    shared = getattr(result, "shared_params", None)
    datasets = getattr(result, "dataset_params", None)
    if isinstance(shared, Mapping) and name in shared:
        value = float(shared[name])
    elif "::" in name and isinstance(datasets, Mapping):
        dataset_id, parameter = name.split("::", 1)
        values = datasets.get(dataset_id)
        if not isinstance(values, Mapping) or parameter not in values:
            raise KeyError(name)
        value = float(values[parameter])
    else:
        raise KeyError(name)
    if not math.isfinite(value):
        raise ValueError(f"Bootstrap Fit returned non-finite parameter {name!r}.")
    return value


def _refit_converged(result: object) -> tuple[bool, Mapping[str, object] | None]:
    """Require an explicit converged terminal refit before using its values."""

    completion = getattr(result, "completion", None)
    if completion is not None:
        converged = bool(getattr(completion, "optimizer_converged", False))
        if converged:
            return True, None
        return False, {
            "status": "nonconverged",
            "message": str(getattr(result, "message", "") or ""),
        }
    evaluation = getattr(result, "evaluation", None)
    if isinstance(evaluation, FitEvaluationOutcome):
        if evaluation.status == "converged":
            return True, None
        return False, {
            "status": evaluation.status,
            "provenance": dict(evaluation.provenance),
            **(
                {"cause": dict(evaluation.cause_payload())}
                if evaluation.cause_payload() is not None
                else {}
            ),
        }
    return False, {
        "status": "nonconverged",
        "message": "Bootstrap refit did not expose an explicit convergence outcome.",
    }


_BOOTSTRAP_STRATUM_SCHEMA_VERSION = 1
_BOOTSTRAP_DRAW_SCHEMA_VERSION = 1


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
                "x": (
                    tuple(float(value) for value in payload.x_obs_by_species[target])
                    if payload.x_name != "t"
                    else tuple(float(value) for value in source.t)
                ),
                "x_name": payload.x_name,
            }
        manifest[payload.dataset_id] = targets
    return manifest


def _bootstrap_payloads(
    *,
    payloads: Sequence[FitDatasetSpec],
    fitted_model: Mapping[str, Mapping[str, np.ndarray]],
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
            indexes = rng.integers(0, residuals.size, size=residuals.size)
            resampled_y = model + residuals[indexes]
            observations[target] = FitObservationSeries(
                raw_observation_id=source.raw_observation_id,
                raw_column=source.raw_column,
                t=source.t,
                y=resampled_y,
                source_rows=source.source_rows,
            )
        captured.append(
            FitDatasetSpec(
                dataset_id=payload.dataset_id,
                species_list=payload.species_list,
                observations=observations,
                target_weights=payload.target_weights,
                x_name=payload.x_name,
                x_mode=payload.x_mode,
                x_obs_by_species=payload.x_obs_by_species,
            )
        )
    return tuple(captured)


def residual_bootstrap_extension(
    *,
    terminal_result: object,
    payloads: Sequence[FitDatasetSpec],
    config: ResidualBootstrapConfig,
    refit: Callable[[Sequence[FitDatasetSpec]], object],
    cancellation_check: Callable[[], bool] | None = None,
) -> FitTerminalExtension:
    """Run deterministic full refits for stratified residual replicates."""

    if not isinstance(config, ResidualBootstrapConfig):
        raise TypeError("Residual bootstrap requires ResidualBootstrapConfig.")
    if hasattr(terminal_result, "completion") or hasattr(terminal_result, "evaluation"):
        primary_converged, primary_failure = _refit_converged(terminal_result)
        if not primary_converged:
            return FitTerminalExtension(
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
                    "requested_parameters": config.parameter_names,
                    "primary_fit": primary_failure or {"status": "nonconverged"},
                },
                unavailable_reason="Residual bootstrap requires a converged primary Fit.",
            )
    fitted_model = getattr(terminal_result, "model_series", None)
    if not isinstance(fitted_model, Mapping) or not fitted_model:
        raise ValueError(
            "The terminal Fit did not retain aligned model series for Bootstrap."
        )
    for name in config.parameter_names:
        _parameter_value(terminal_result, name)

    stratum_manifest = _bootstrap_stratum_manifest(payloads)
    rng = np.random.default_rng(config.seed)
    successes: dict[str, list[float]] = {
        name: [] for name in config.parameter_names
    }
    replicate_records: list[Mapping[str, object]] = []
    for index in range(config.replicates):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        synthetic = _bootstrap_payloads(
            payloads=payloads,
            fitted_model=fitted_model,
            rng=rng,
        )
        try:
            fitted = refit(synthetic)
            converged, failure = _refit_converged(fitted)
            if not converged:
                replicate_records.append(
                    {
                        "replicate": index,
                        "status": "failed",
                        "failure": failure or {"status": "nonconverged"},
                    }
                )
                continue
            values = {
                name: _parameter_value(fitted, name)
                for name in config.parameter_names
            }
        except FitFinalReplayError as exc:
            replicate_records.append(
                {
                    "replicate": index,
                    "status": "failed",
                    "failure": simulation_failure_from_exception(exc.typed_cause),
                }
            )
            continue
        except FitSimulationError as exc:
            if not _is_candidate_domain_failure(exc):
                raise
            replicate_records.append(
                {
                    "replicate": index,
                    "status": "failed",
                    "failure": simulation_failure_from_exception(exc),
                }
            )
            continue
        for name, value in values.items():
            successes[name].append(value)
        replicate_records.append(
            {
                "replicate": index,
                "status": "success",
                "parameters": values,
            }
        )

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

    return FitTerminalExtension(
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
            "draw_algorithm": "numpy.default_rng(seed).integers(0, stratum_residual_count)",
            "replay_authority": "seed plus versioned draw algorithm and stratum manifest",
            "preserved_fields": (
                "time",
                "parametric X",
                "source row",
                "dataset weight",
                "target weight",
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
    )

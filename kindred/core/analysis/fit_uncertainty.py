"""Named, provenance-complete uncertainty methods for completed Fits.

The functions in this module do not evaluate models or own an optimizer.  They
drive the canonical Fit objective and optimizer supplied by the global fitting
composition boundary, then return immutable Profile or Bootstrap results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence, cast

import numpy as np

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    FitObservationSeries,
)
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.analysis.fit_evaluation_outcome import ScientificUnavailable
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_completion import FitExtensionProgress
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
    "ProfileLikelihoodEndpoint",
    "ProfileLikelihoodInterval",
    "ProfileLikelihoodResult",
    "BootstrapPercentileInterval",
    "ResidualBootstrapResult",
    "profile_likelihood_settlement_result",
    "residual_bootstrap_settlement_result",
    "profile_likelihood_result",
    "residual_bootstrap_result",
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


def _bounded_nonnegative_integer(
    value: object,
    *,
    field: str,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer.")
    try:
        result = int(value)
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be an integer.") from exc
    if not math.isfinite(exact) or exact != result:
        raise ValueError(f"{field} must be an integer.")
    if result < 0 or result > int(maximum):
        raise ValueError(f"{field} must be within 0 and {int(maximum)}.")
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


ProfileLikelihoodStatus = Literal[
    "available",
    "partial",
    "unavailable",
    "stopped",
    "failed",
]
ProfileLikelihoodEndpointStatus = Literal[
    "bound",
    "limiting",
    "threshold",
    "failed",
    "unavailable",
]
BootstrapStatus = Literal["available", "unavailable", "stopped", "failed"]


def _freeze_fact(value: object) -> object:
    """Freeze a factual result field without introducing a transport envelope."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_fact(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_fact(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodEndpoint:
    """Resolution facts for one Profile lower or upper endpoint."""

    status: ProfileLikelihoodEndpointStatus
    evaluation_index: int | None = None
    optimizer_value: float | None = None
    likelihood_ratio: float | None = None
    coordinate_tolerance: float | None = None
    likelihood_ratio_tolerance: float | None = None
    bracket_width: float | None = None
    reason: str | None = None
    last_evaluation_index: int | None = None
    evaluation_count: int = 0
    failure: FactualCause | None = None

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"bound", "limiting", "threshold", "failed", "unavailable"}:
            raise ValueError("Profile endpoint status is invalid.")
        object.__setattr__(self, "status", status)
        for name in (
            "evaluation_index",
            "last_evaluation_index",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = int(value)
                if normalized < 0:
                    raise ValueError("Profile endpoint evaluation index cannot be negative.")
                object.__setattr__(self, name, normalized)
        evaluation_count = int(self.evaluation_count)
        if evaluation_count < 0:
            raise ValueError("Profile endpoint evaluation count cannot be negative.")
        object.__setattr__(self, "evaluation_count", evaluation_count)
        optimizer_value = self.optimizer_value
        if optimizer_value is not None:
            optimizer_value = float(optimizer_value)
            if not math.isfinite(optimizer_value):
                raise ValueError("Profile endpoint optimizer_value must be finite.")
            object.__setattr__(self, "optimizer_value", optimizer_value)
        for name in (
            "likelihood_ratio",
            "coordinate_tolerance",
            "likelihood_ratio_tolerance",
            "bracket_width",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = float(value)
                if not math.isfinite(normalized) or normalized < 0.0:
                    raise ValueError(f"Profile endpoint {name} must be finite and nonnegative.")
                object.__setattr__(self, name, normalized)
        if self.failure is not None and not isinstance(self.failure, FactualCause):
            raise TypeError("Profile endpoint failure must be a FactualCause.")


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodInterval:
    """One coordinate's estimate, endpoints, and bounded resolution facts."""

    status: Literal["available", "partial", "unavailable"]
    interval_kind: str
    estimate: float
    lower: float | None
    upper: float | None
    lower_outcome: ProfileLikelihoodEndpointStatus
    upper_outcome: ProfileLikelihoodEndpointStatus
    optimizer_bounds: tuple[float, float]
    physical_bounds: tuple[float, float]
    lower_resolution: ProfileLikelihoodEndpoint
    upper_resolution: ProfileLikelihoodEndpoint
    endpoint_messages: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "partial", "unavailable"}:
            raise ValueError("Profile interval status is invalid.")
        bounds = tuple(float(value) for value in self.optimizer_bounds)
        physical_bounds = tuple(float(value) for value in self.physical_bounds)
        if len(bounds) != 2 or len(physical_bounds) != 2:
            raise ValueError("Profile interval bounds require two values.")
        if any(not math.isfinite(value) for value in (*bounds, *physical_bounds)):
            raise ValueError("Profile interval bounds must be finite.")
        estimate = float(self.estimate)
        if not math.isfinite(estimate):
            raise ValueError("Profile interval estimate must be finite.")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Profile interval endpoints must be finite.")
                object.__setattr__(self, name, value)
        if not isinstance(self.lower_resolution, ProfileLikelihoodEndpoint):
            raise TypeError("Profile lower resolution must be a typed endpoint.")
        if not isinstance(self.upper_resolution, ProfileLikelihoodEndpoint):
            raise TypeError("Profile upper resolution must be a typed endpoint.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "optimizer_bounds", bounds)
        object.__setattr__(self, "physical_bounds", physical_bounds)
        object.__setattr__(
            self,
            "endpoint_messages",
            MappingProxyType({str(key): str(value) for key, value in self.endpoint_messages.items()}),
        )


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodResult:
    """Immutable factual result of a weighted-Gaussian Profile analysis."""

    status: ProfileLikelihoodStatus
    method: str
    requested_parameters: tuple[str, ...]
    confidence_level: float
    likelihood_ratio_threshold: float
    likelihood_model: str
    likelihood_ratio_transport: str
    fixed_coordinate_policy: str
    warm_start_policy: str
    optimizer_method: str
    max_bracketing_steps_per_side: int
    max_nfev_per_point: int
    observation_count: int | None
    terminal_weighted_residual_sse: float | None
    intervals: Mapping[object, ProfileLikelihoodInterval]
    unavailable_reason: str | None = None
    diagnostic: FactualCause | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "partial", "unavailable", "stopped", "failed"}:
            raise ValueError("Profile result status is invalid.")
        names = tuple(str(value) for value in self.requested_parameters)
        if len(names) != len(set(names)) or any(not value.strip() for value in names):
            raise ValueError("Profile result parameter names must be unique and nonblank.")
        confidence = _confidence(self.confidence_level)
        threshold = float(self.likelihood_ratio_threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Profile result threshold must be positive.")
        for name in ("max_bracketing_steps_per_side", "max_nfev_per_point"):
            value = _bounded_positive_integer(
                getattr(self, name),
                field=f"Profile result {name}",
                maximum=(
                    _PROFILE_MAX_POINTS_PER_SIDE
                    if name == "max_bracketing_steps_per_side"
                    else 100_000
                ),
            )
            object.__setattr__(self, name, value)
        count = self.observation_count
        if count is not None:
            count = _bounded_nonnegative_integer(
                count,
                field="Profile observation count",
                maximum=2_147_483_647,
            )
        sse = self.terminal_weighted_residual_sse
        if sse is not None:
            sse = float(sse)
            if not math.isfinite(sse) or sse < 0.0:
                raise ValueError("Profile terminal SSE must be finite and nonnegative.")
        intervals = dict(self.intervals)
        if any(not isinstance(item, ProfileLikelihoodInterval) for item in intervals.values()):
            raise TypeError("Profile intervals require typed interval records.")
        if self.diagnostic is not None and not isinstance(self.diagnostic, FactualCause):
            raise TypeError("Profile result diagnostic must be a FactualCause.")
        reason = None if self.unavailable_reason is None else str(self.unavailable_reason).strip() or None
        if status == "unavailable" and reason is None:
            raise ValueError("Unavailable Profile results require a factual reason.")
        if status in {"stopped", "failed"} and self.diagnostic is None:
            raise ValueError("Stopped or failed Profile results require a diagnostic.")
        method = str(self.method).strip()
        if not method:
            raise ValueError("Profile results require a named method.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", _PROFILE_METHOD)
        object.__setattr__(self, "requested_parameters", names)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "likelihood_ratio_threshold", threshold)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "terminal_weighted_residual_sse", sse)
        object.__setattr__(self, "intervals", MappingProxyType(intervals))
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance or {})))


@dataclass(frozen=True, slots=True)
class BootstrapPercentileInterval:
    """One parameter's equal-tailed bootstrap percentile interval."""

    status: Literal["available", "unavailable"]
    lower: float | None
    upper: float | None
    construction: str
    reason: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "unavailable"}:
            raise ValueError("Bootstrap interval status is invalid.")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Bootstrap percentile endpoints must be finite.")
                object.__setattr__(self, name, number)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "construction", str(self.construction))
        object.__setattr__(self, "reason", None if self.reason is None else str(self.reason))


@dataclass(frozen=True, slots=True)
class ResidualBootstrapResult:
    """Immutable factual result of deterministic residual Bootstrap refits."""

    status: BootstrapStatus
    method: str
    requested_parameters: tuple[str, ...]
    seed: int
    replicates: int
    confidence_level: float
    resampling: str
    draw_algorithm: str
    residual_centering: str
    replicate_execution: str
    refit_validity: str
    minimum_successful_replicates: int
    minimum_success_fraction: float
    required_successes: int
    successful_replicates: int
    failed_replicates: int
    interval_construction: str
    intervals: Mapping[object, BootstrapPercentileInterval]
    primary_fit_diagnostic: FactualCause | None = None
    primary_fit_unavailable: ScientificUnavailable | None = None
    first_failed_replicate_diagnostic: FactualCause | None = None
    first_failed_replicate_unavailable: ScientificUnavailable | None = None
    unavailable_reason: str | None = None
    diagnostic: FactualCause | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "unavailable", "stopped", "failed"}:
            raise ValueError("Bootstrap result status is invalid.")
        names = tuple(str(value) for value in self.requested_parameters)
        if len(names) != len(set(names)) or any(not value.strip() for value in names):
            raise ValueError("Bootstrap result parameter names must be unique and nonblank.")
        seed = validate_bootstrap_seed(self.seed)
        replicates = _bounded_positive_integer(
            self.replicates,
            field="Bootstrap result replicate count",
            maximum=_BOOTSTRAP_MAX_REPLICATES,
        )
        confidence = _confidence(self.confidence_level)
        minimum = _bounded_positive_integer(
            self.minimum_successful_replicates,
            field="Bootstrap result minimum successful replicates",
            maximum=replicates,
        )
        required = _bounded_nonnegative_integer(
            self.required_successes,
            field="Bootstrap result required successes",
            maximum=replicates,
        )
        successful = _bounded_nonnegative_integer(
            self.successful_replicates,
            field="Bootstrap result successful replicates",
            maximum=replicates,
        )
        failed = _bounded_nonnegative_integer(
            self.failed_replicates,
            field="Bootstrap result failed replicates",
            maximum=replicates,
        )
        if not 0 < minimum <= replicates or not 0 <= required <= replicates:
            raise ValueError("Bootstrap result success thresholds are invalid.")
        if successful < 0 or failed < 0 or successful + failed > replicates:
            raise ValueError("Bootstrap result replicate counts are invalid.")
        fraction = float(self.minimum_success_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("Bootstrap result success fraction is invalid.")
        intervals = dict(self.intervals)
        if any(not isinstance(item, BootstrapPercentileInterval) for item in intervals.values()):
            raise TypeError("Bootstrap intervals require typed percentile records.")
        for cause_name in (
            "primary_fit_diagnostic",
            "diagnostic",
            "first_failed_replicate_diagnostic",
        ):
            cause = getattr(self, cause_name)
            if cause is not None and not isinstance(cause, FactualCause):
                raise TypeError(f"Bootstrap {cause_name} must be a FactualCause.")
        if self.primary_fit_unavailable is not None and not isinstance(
            self.primary_fit_unavailable, ScientificUnavailable
        ):
            raise TypeError("Bootstrap primary-fit unavailability must be typed.")
        if self.first_failed_replicate_unavailable is not None and not isinstance(
            self.first_failed_replicate_unavailable, ScientificUnavailable
        ):
            raise TypeError("Bootstrap first-failure unavailability must be typed.")
        reason = None if self.unavailable_reason is None else str(self.unavailable_reason).strip() or None
        if status == "unavailable" and reason is None:
            raise ValueError("Unavailable Bootstrap results require a factual reason.")
        if status in {"stopped", "failed"} and self.diagnostic is None:
            raise ValueError("Stopped or failed Bootstrap results require a diagnostic.")
        method = str(self.method).strip()
        if not method:
            raise ValueError("Bootstrap results require a named method.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", _BOOTSTRAP_METHOD)
        object.__setattr__(self, "requested_parameters", names)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "minimum_successful_replicates", minimum)
        object.__setattr__(self, "minimum_success_fraction", fraction)
        object.__setattr__(self, "required_successes", required)
        object.__setattr__(self, "successful_replicates", successful)
        object.__setattr__(self, "failed_replicates", failed)
        object.__setattr__(self, "intervals", MappingProxyType(intervals))
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance or {})))


def profile_likelihood_settlement_result(
    *,
    config: ProfileLikelihoodConfig,
    status: Literal["stopped", "failed"],
    cause: FactualCause,
    observation_count: int,
    terminal_weighted_residual_sse: float,
) -> ProfileLikelihoodResult:
    """Construct a stopped/failed Profile result from call-local facts."""

    return ProfileLikelihoodResult(
        status=status,
        method=_PROFILE_METHOD,
        requested_parameters=tuple(_coordinate_display(item) for item in config.parameter_names),
        confidence_level=config.confidence_level,
        likelihood_ratio_threshold=float(config.likelihood_ratio_threshold),
        likelihood_model="weighted Gaussian residuals with profiled variance",
        likelihood_ratio_transport="finite log likelihood ratio",
        fixed_coordinate_policy="selected coordinate fixed; all others reoptimized",
        warm_start_policy="previous scientifically available nuisance endpoint",
        optimizer_method=config.optimizer_method,
        max_bracketing_steps_per_side=config.max_bracketing_steps_per_side,
        max_nfev_per_point=config.max_nfev_per_point,
        observation_count=int(observation_count),
        terminal_weighted_residual_sse=float(terminal_weighted_residual_sse),
        intervals={},
        diagnostic=cause,
        provenance={"settlement": status, "primary_fit_preserved": True},
    )


def residual_bootstrap_settlement_result(
    *,
    config: ResidualBootstrapConfig,
    status: Literal["stopped", "failed"],
    cause: FactualCause,
    settlement: str,
    successful_replicates: int = 0,
    failed_replicates: int = 0,
    completed_replicates: int = 0,
) -> ResidualBootstrapResult:
    """Construct a stopped/failed Bootstrap result from local facts."""

    required_successes = max(
        config.minimum_successful_replicates,
        int(math.ceil(config.replicates * config.minimum_success_fraction)),
    )
    return ResidualBootstrapResult(
        status=status,
        method=_BOOTSTRAP_METHOD,
        requested_parameters=tuple(_coordinate_display(item) for item in config.parameter_names),
        seed=config.seed,
        replicates=config.replicates,
        confidence_level=config.confidence_level,
        resampling="within each dataset/target residual stratum",
        draw_algorithm="numpy.default_rng(seed) residual-index draws",
        residual_centering="independent arithmetic mean subtraction within each dataset/target stratum",
        replicate_execution="full global refit",
        refit_validity="scientifically available completed refits count regardless of optimizer convergence",
        minimum_successful_replicates=config.minimum_successful_replicates,
        minimum_success_fraction=config.minimum_success_fraction,
        required_successes=required_successes,
        successful_replicates=int(successful_replicates),
        failed_replicates=int(failed_replicates),
        interval_construction="equal-tailed percentile",
        intervals={},
        diagnostic=cause,
        provenance={
            "settlement": str(settlement),
            "primary_fit_preserved": True,
            "completed_replicates": int(completed_replicates),
        },
    )


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
) -> np.ndarray | None:
    """Evaluate one scientific residual call, never an optimizer penalty."""

    scientific_evaluator = getattr(objective, "evaluate_scientific_residuals", None)
    if not callable(scientific_evaluator):
        raise TypeError(
            "Profile objective must expose its canonical scientific residual evaluator."
        )
    raw = scientific_evaluator(np.asarray(vector, dtype=float))
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float).reshape(-1)
    return values if values.size and np.all(np.isfinite(values)) else None


@dataclass(frozen=True, slots=True)
class _LikelihoodRatioDetails:
    transport_value: float
    outcome: str


def _likelihood_ratio_details(
    *,
    candidate_sse: float,
    best_sse: float,
    n: int,
) -> _LikelihoodRatioDetails:
    """Calculate an un-clamped finite transport value and crossing outcome.

    The profile comparison is a log likelihood ratio.
    Exact-fit and overflow limits use a finite
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
        return _LikelihoodRatioDetails(0.0, "finite")
    if best == 0.0:
        finite_transport = float(np.finfo(float).max / 4.0)
        return _LikelihoodRatioDetails(
            finite_transport,
            "threshold_exceeded_exact_fit",
        )
    log_ratio = float(count) * (math.log(candidate) - math.log(best))
    if not math.isfinite(log_ratio):
        finite_transport = float(np.finfo(float).max / 4.0)
        return _LikelihoodRatioDetails(
            finite_transport,
            "threshold_exceeded_overflow",
        )
    return _LikelihoodRatioDetails(log_ratio, "finite")


def _profile_endpoint_record(
    record: Mapping[str, object],
) -> ProfileLikelihoodEndpoint:
    return ProfileLikelihoodEndpoint(
        status=str(record["status"]),  # type: ignore[arg-type]
        evaluation_index=(
            None if record.get("evaluation_index") is None else int(record["evaluation_index"])
        ),
        optimizer_value=(
            None if record.get("optimizer_value") is None else float(record["optimizer_value"])
        ),
        likelihood_ratio=(
            None if record.get("likelihood_ratio") is None else float(record["likelihood_ratio"])
        ),
        coordinate_tolerance=(
            None
            if record.get("coordinate_tolerance") is None
            else float(record["coordinate_tolerance"])
        ),
        likelihood_ratio_tolerance=(
            None
            if record.get("likelihood_ratio_tolerance") is None
            else float(record["likelihood_ratio_tolerance"])
        ),
        bracket_width=(
            None if record.get("bracket_width") is None else float(record["bracket_width"])
        ),
        reason=None if record.get("reason") is None else str(record["reason"]),
        last_evaluation_index=(
            None
            if record.get("last_evaluation_index") is None
            else int(record["last_evaluation_index"])
        ),
        evaluation_count=int(record.get("evaluation_count", 0)),
        failure=(
            record.get("failure")
            if isinstance(record.get("failure"), FactualCause)
            else None
        ),
    )


def _profile_interval_record(
    record: Mapping[str, object],
) -> ProfileLikelihoodInterval:
    lower_resolution = record.get("resolution", {})
    if not isinstance(lower_resolution, Mapping):
        raise TypeError("Profile interval resolution must be a mapping before settlement.")
    lower_endpoint = lower_resolution.get("lower", {})
    upper_endpoint = lower_resolution.get("upper", {})
    if not isinstance(lower_endpoint, Mapping) or not isinstance(upper_endpoint, Mapping):
        raise TypeError("Profile endpoint resolutions must be mappings before settlement.")
    return ProfileLikelihoodInterval(
        status=str(record["status"]),  # type: ignore[arg-type]
        interval_kind=str(record["interval_kind"]),
        estimate=float(record["estimate"]),
        lower=None if record.get("lower") is None else float(record["lower"]),
        upper=None if record.get("upper") is None else float(record["upper"]),
        lower_outcome=str(record["lower_outcome"]),  # type: ignore[arg-type]
        upper_outcome=str(record["upper_outcome"]),  # type: ignore[arg-type]
        optimizer_bounds=tuple(record["optimizer_bounds"]),  # type: ignore[arg-type]
        physical_bounds=tuple(record["physical_bounds"]),  # type: ignore[arg-type]
        lower_resolution=_profile_endpoint_record(lower_endpoint),
        upper_resolution=_profile_endpoint_record(upper_endpoint),
        endpoint_messages={
            str(key): str(value)
            for key, value in dict(lower_resolution.get("messages", {})).items()
        },
    )


def profile_likelihood_result(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    initial_parameters: Mapping[object, float],
    bounds: Mapping[object, tuple[float, float]],
    config: ProfileLikelihoodConfig,
    decode_coordinate: Callable[[object, float], float],
    base_residuals: np.ndarray | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> ProfileLikelihoodResult:
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

    def finish(result: ProfileLikelihoodResult) -> ProfileLikelihoodResult:
        emit_progress(
            status=result.status,
            stage="settlement",
            coordinate=active_coordinate,
            side=active_side,
            discovery_step=active_discovery,
            refinement_step=active_refinement,
        )
        return result

    emit_progress(stage="primary_complete")

    base_sse: float | None = None
    observation_count: int | None = None

    def failure_result(
        cause: FactualCause,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> ProfileLikelihoodResult:
        return finish(
            ProfileLikelihoodResult(
                status="failed",
                method=_PROFILE_METHOD,
                requested_parameters=tuple(
                    _coordinate_display(item) for item in config.parameter_names
                ),
                confidence_level=config.confidence_level,
                likelihood_ratio_threshold=threshold,
                likelihood_model="weighted Gaussian residuals with profiled variance",
                likelihood_ratio_transport=(
                    "finite un-clamped log-ratio with typed exact-fit/overflow crossing outcome"
                ),
                fixed_coordinate_policy="selected coordinate fixed; all others reoptimized",
                warm_start_policy="previous scientifically available nuisance endpoint",
                optimizer_method=config.optimizer_method,
                max_bracketing_steps_per_side=config.max_bracketing_steps_per_side,
                max_nfev_per_point=config.max_nfev_per_point,
                observation_count=observation_count,
                terminal_weighted_residual_sse=base_sse,
                intervals={},
                diagnostic=cause,
                provenance={"settlement": "failed", **dict(provenance or {})},
            )
        )

    if base_residuals is None:
        try:
            base_vector = np.asarray([initial[name] for name in initial], dtype=float)
            base_residuals = _objective_scientific_residuals(objective, base_vector)
        except (FittingCancelled, SimulationCancelled):
            raise
        except Exception as exc:
            return failure_result(
                factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="profile_likelihood",
                ),
                provenance={"stage": "primary_profile_evaluation"},
            )
    if base_residuals is None:
        return finish(ProfileLikelihoodResult(
            status="unavailable",
            method=_PROFILE_METHOD,
            requested_parameters=tuple(
                _coordinate_display(item) for item in config.parameter_names
            ),
            confidence_level=config.confidence_level,
            likelihood_ratio_threshold=threshold,
            likelihood_model="weighted Gaussian residuals with profiled variance",
            likelihood_ratio_transport="finite log likelihood ratio",
            fixed_coordinate_policy="selected coordinate fixed; all others reoptimized",
            warm_start_policy="previous scientifically available nuisance endpoint",
            optimizer_method=config.optimizer_method,
            max_bracketing_steps_per_side=config.max_bracketing_steps_per_side,
            max_nfev_per_point=config.max_nfev_per_point,
            observation_count=None,
            terminal_weighted_residual_sse=None,
            intervals={},
            unavailable_reason="The primary Fit did not provide a scientifically available evaluation.",
            provenance={"base_evaluation": "scientifically_unavailable"},
        ))
    base_sse = _sse(base_residuals)
    observation_count = int(base_residuals.size)
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
                try:
                    evaluation_residuals = _objective_scientific_residuals(
                        objective,
                        np.asarray(
                            [fit.parameters[item] for item in initial],
                            dtype=float,
                        ),
                    )
                except Exception as exc:
                    raise RuntimeError("Fatal Profile nuisance evaluation failed.") from exc
                if evaluation_residuals is None:
                    record = {
                        "evaluation_index": len(records),
                        "fixed_optimizer_value": float(fixed),
                        "status": "unavailable",
                        "reason": "scientific_unavailable",
                    }
                    records.append(record)
                    return record, None
                candidate_sse = _sse(evaluation_residuals)
                lr_details = _likelihood_ratio_details(
                    candidate_sse=candidate_sse,
                    best_sse=base_sse,
                    n=observation_count,
                )
                record = {
                    "evaluation_index": len(records),
                    "fixed_optimizer_value": float(fixed),
                    "status": "success",
                    "likelihood_ratio": (
                        None
                        if lr_details.outcome == "threshold_exceeded_exact_fit"
                        else lr_details.transport_value
                    ),
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
                        "reason": "scientific_unavailable",
                        "failure": record.get("failure"),
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
                        "reason": "scientific_unavailable",
                        "failure": record.get("failure"),
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
                return failure_result(
                    factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                    provenance={
                        "coordinate": _coordinate_display(name),
                        "side": side,
                        "evaluation_count": len(side_results.get(side, (None, None, (), {}))[2]),
                    },
                )
        lower, lower_outcome, lower_records, lower_resolution = side_results["lower"]
        upper, upper_outcome, upper_records, upper_resolution = side_results["upper"]
        lower_resolution = {
            **lower_resolution,
            "evaluation_count": len(lower_records),
        }
        upper_resolution = {
            **upper_resolution,
            "evaluation_count": len(upper_records),
        }
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
        result_status = "unavailable"
        unavailable_reason = "No requested profile interval could be determined."
    elif any_partial:
        result_status = "partial"
        unavailable_reason = None
    else:
        result_status = "available"
        unavailable_reason = None
    return finish(ProfileLikelihoodResult(
        status=result_status,
        method=_PROFILE_METHOD,
        requested_parameters=tuple(
            _coordinate_display(item) for item in config.parameter_names
        ),
        confidence_level=config.confidence_level,
        likelihood_ratio_threshold=threshold,
        likelihood_model="weighted Gaussian residuals with profiled variance",
        likelihood_ratio_transport=(
            "finite un-clamped log-ratio with typed exact-fit/overflow crossing outcome"
        ),
        fixed_coordinate_policy="selected coordinate fixed; all others reoptimized",
        warm_start_policy="previous scientifically available nuisance endpoint",
        optimizer_method=config.optimizer_method,
        max_bracketing_steps_per_side=config.max_bracketing_steps_per_side,
        max_nfev_per_point=config.max_nfev_per_point,
        observation_count=observation_count,
        terminal_weighted_residual_sse=base_sse,
        intervals={
            name: _profile_interval_record(record)
            for name, record in intervals.items()
        },
        unavailable_reason=unavailable_reason,
        provenance={
            "primary_fit_preserved": True,
            "settlement": result_status,
        },
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


_BootstrapRefitOutcome = _fit_execution.GlobalFitResult | ScientificUnavailable


def _refit_scientifically_available(
    result: object,
) -> tuple[bool, Mapping[str, object] | None]:
    """Classify one completed refit by scientific availability only."""

    if isinstance(result, ScientificUnavailable):
        return False, {
            "status": "scientific_unavailable",
            "cause": result,
        }
    # The refit callback is an admitted typed internal route.  Its concrete
    # result owner has already established scientific availability; optimizer
    # convergence is only a fact on that owner, never a second admission gate.
    return True, None


def _bootstrap_payloads(
    *,
    payloads: Sequence[FitDatasetSpec],
    fitted_info: Mapping[str, object],
    aligned_observed_x_by_dataset: Mapping[str, np.ndarray],
    projection_positions_by_dataset: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[FitDatasetSpec, ...]:
    captured: list[FitDatasetSpec] = []
    for payload in payloads:
        info = fitted_info.get(payload.dataset_id)
        if not isinstance(info, _fit_execution.DatasetFitInfo):
            raise ValueError(
                f"Bootstrap terminal facts omitted dataset {payload.dataset_id!r}."
            )
        dataset_model = info.model_series
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
            if payload.x_observations is None or info.x_projection is None:
                raise ValueError(
                    f"Bootstrap dataset {payload.dataset_id!r} omitted its canonical X model."
                )
            projection = info.x_projection
            x_source = payload.x_observations
            observed_x = aligned_observed_x_by_dataset[payload.dataset_id]
            x_residuals = observed_x - np.asarray(projection.selected_model_x, dtype=float)
            centered_x = x_residuals - float(np.mean(x_residuals))
            indexes = rng.integers(0, centered_x.size, size=centered_x.size)
            synthetic_x = np.asarray(projection.selected_model_x, dtype=float) + centered_x[indexes]
            projection_positions = projection_positions_by_dataset[payload.dataset_id]
            full_values = np.asarray(x_source.y, dtype=float).copy()
            projected = projection_positions >= 0
            full_values[projected] = synthetic_x[projection_positions[projected]]
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


def _bootstrap_interval_record(
    record: Mapping[str, object],
) -> BootstrapPercentileInterval:
    return BootstrapPercentileInterval(
        status=str(record["status"]),  # type: ignore[arg-type]
        lower=None if record.get("lower") is None else float(record["lower"]),
        upper=None if record.get("upper") is None else float(record["upper"]),
        construction=str(record.get("construction", "")),
        reason=None if record.get("reason") is None else str(record["reason"]),
    )


def residual_bootstrap_result(
    *,
    terminal_result: object,
    payloads: Sequence[FitDatasetSpec],
    config: ResidualBootstrapConfig,
    refit: Callable[[Sequence[FitDatasetSpec], int], _BootstrapRefitOutcome],
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> ResidualBootstrapResult:
    """Run deterministic full refits for stratified residual replicates."""

    if not isinstance(config, ResidualBootstrapConfig):
        raise TypeError("Residual bootstrap requires ResidualBootstrapConfig.")
    if not config.parameter_names:
        raise ValueError("Ordinary residual Bootstrap requires at least one coordinate.")
    primary_available, primary_failure = _refit_scientifically_available(terminal_result)
    completed = 0
    successful = 0
    failed = 0
    first_failed_replicate_diagnostic: FactualCause | None = None
    first_failed_replicate_unavailable: ScientificUnavailable | None = None
    required_successes = max(
        config.minimum_successful_replicates,
        int(math.ceil(config.replicates * config.minimum_success_fraction)),
    )

    def build_result(
        status: BootstrapStatus,
        *,
        intervals: Mapping[object, BootstrapPercentileInterval] | None = None,
        diagnostic: FactualCause | None = None,
        primary_fit_diagnostic: FactualCause | None = None,
        primary_fit_unavailable: ScientificUnavailable | None = None,
        unavailable_reason: str | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> ResidualBootstrapResult:
        """Build one bounded Bootstrap settlement from current call-local facts."""

        return ResidualBootstrapResult(
            status=status,
            method=_BOOTSTRAP_METHOD,
            requested_parameters=tuple(
                _coordinate_display(item) for item in config.parameter_names
            ),
            seed=config.seed,
            replicates=config.replicates,
            confidence_level=config.confidence_level,
            resampling="within each dataset/target residual stratum",
            draw_algorithm=(
                "numpy.default_rng(seed).integers(0, stratum_residual_count); "
                "integer draws sample independently mean-centered residuals"
            ),
            residual_centering=(
                "independent arithmetic mean subtraction within each dataset/target stratum"
            ),
            replicate_execution="full global refit",
            refit_validity="scientifically available completed refits count regardless of optimizer convergence",
            minimum_successful_replicates=config.minimum_successful_replicates,
            minimum_success_fraction=config.minimum_success_fraction,
            required_successes=required_successes,
            successful_replicates=successful,
            failed_replicates=failed,
            interval_construction="equal-tailed percentile",
            intervals={} if intervals is None else intervals,
            primary_fit_diagnostic=primary_fit_diagnostic,
            primary_fit_unavailable=primary_fit_unavailable,
            first_failed_replicate_diagnostic=first_failed_replicate_diagnostic,
            first_failed_replicate_unavailable=first_failed_replicate_unavailable,
            unavailable_reason=unavailable_reason,
            diagnostic=diagnostic,
            provenance={} if provenance is None else provenance,
        )

    def stopped_result(
        cause: FactualCause,
        *,
        settlement: str = "stopped",
    ) -> ResidualBootstrapResult:
        return build_result(
            "stopped",
            diagnostic=cause,
            provenance={
                "primary_fit_preserved": True,
                "settlement": settlement,
                "completed_replicates": completed,
            },
        )

    def emit_progress(
        *,
        status: str = "running",
        stage: str = "replicate",
        completed: int = 0,
    ) -> FactualCause | None:
        if progress_callback is None:
            return None
        try:
            progress_callback(
                FitExtensionProgress(
                    phase="bootstrap", status=status, stage=stage,
                    completed_work=completed, planned_work=config.replicates,
                    completed_replicates=completed, planned_replicates=config.replicates,
                )
            )
        except (FittingCancelled, SimulationCancelled) as exc:
            return factual_cause_from_exception(exc, domain="fit", origin="bootstrap")
        return None

    def finish(result: ResidualBootstrapResult) -> ResidualBootstrapResult:
        emit_progress(
            status=result.status,
            stage="settlement",
            completed=completed,
        )
        return result

    progress_cause = emit_progress(stage="primary_complete")
    if progress_cause is not None:
        return finish(stopped_result(progress_cause))
    if not primary_available:
        primary_cause = (
            primary_failure.get("cause")
            if isinstance(primary_failure, Mapping)
            and isinstance(primary_failure.get("cause"), FactualCause)
            else None
        )
        primary_unavailable = (
            primary_failure.get("cause")
            if isinstance(primary_failure, Mapping)
            and isinstance(primary_failure.get("cause"), ScientificUnavailable)
            else None
        )
        return finish(build_result(
            "unavailable",
            primary_fit_diagnostic=primary_cause,
            primary_fit_unavailable=primary_unavailable,
            unavailable_reason="Residual bootstrap requires a scientifically available primary Fit.",
            provenance={"primary_fit": primary_failure or {"status": "scientific_unavailable"}},
        ))
    terminal_infos = getattr(terminal_result, "dataset_info", None)
    if not isinstance(terminal_infos, (tuple, list)) or not terminal_infos:
        raise ValueError(
            "The terminal Fit did not retain aligned dataset facts for Bootstrap."
        )
    fitted_info = {
        info.dataset_id: info
        for info in terminal_infos
        if isinstance(info, _fit_execution.DatasetFitInfo)
    }
    if len(fitted_info) != len(terminal_infos):
        raise ValueError("The terminal Fit dataset roster is malformed for Bootstrap.")
    for name in config.parameter_names:
        _parameter_value(terminal_result, name)

    aligned_observed_x_by_dataset: dict[str, np.ndarray] = {}
    projection_positions_by_dataset: dict[str, np.ndarray] = {}
    for payload in payloads:
        if payload.x_name == "t":
            continue
        if payload.x_observations is None:
            raise ValueError(
                f"Bootstrap dataset {payload.dataset_id!r} omitted its canonical X model."
            )
        info = cast(_fit_execution.DatasetFitInfo, fitted_info[payload.dataset_id])
        projection = info.x_projection
        if projection is None:
            raise ValueError(
                f"Bootstrap dataset {payload.dataset_id!r} omitted its canonical X model."
            )
        observed_by_row = {
            int(row): float(value)
            for row, value in zip(
                payload.x_observations.source_rows,
                payload.x_observations.y,
                strict=True,
            )
        }
        aligned = np.asarray(
            [observed_by_row[int(row)] for row in projection.source_rows],
            dtype=float,
        )
        aligned.setflags(write=False)
        aligned_observed_x_by_dataset[payload.dataset_id] = aligned
        projection_positions_by_row = {
            int(row): index
            for index, row in enumerate(projection.source_rows)
        }
        projection_positions = np.asarray(
            [
                projection_positions_by_row.get(int(row), -1)
                for row in payload.x_observations.source_rows
            ],
            dtype=np.intp,
        )
        projection_positions.setflags(write=False)
        projection_positions_by_dataset[payload.dataset_id] = projection_positions
    rng = np.random.default_rng(config.seed)
    successes: dict[str, list[float]] = {
        name: [] for name in config.parameter_names
    }
    for index in range(config.replicates):
        if cancellation_check is not None and cancellation_check():
            return finish(stopped_result(factual_cause_from_exception(
                FittingCancelled("Residual Bootstrap stopped by user."),
                domain="fit", origin="bootstrap",
            )))
        try:
            synthetic = _bootstrap_payloads(
                payloads=payloads,
                fitted_info=fitted_info,
                aligned_observed_x_by_dataset=aligned_observed_x_by_dataset,
                projection_positions_by_dataset=projection_positions_by_dataset,
                rng=rng,
            )
            fitted = refit(synthetic, index)
        except (FittingCancelled, SimulationCancelled) as exc:
            return finish(stopped_result(factual_cause_from_exception(
                exc, domain="fit", origin="bootstrap",
            )))
        except Exception as exc:
            completed += 1
            failed += 1
            if first_failed_replicate_diagnostic is None:
                first_failed_replicate_diagnostic = factual_cause_from_exception(
                    exc, domain="fit", origin="bootstrap"
                )
            return finish(build_result(
                "failed",
                diagnostic=first_failed_replicate_diagnostic,
                provenance={"replicate": index, "settlement": "failed"},
            ))
        scientifically_available, failure = _refit_scientifically_available(fitted)
        if not scientifically_available:
            completed += 1
            failed += 1
            raw_failure = failure.get("cause") if isinstance(failure, Mapping) else None
            if (
                first_failed_replicate_diagnostic is None
                and isinstance(raw_failure, FactualCause)
            ):
                first_failed_replicate_diagnostic = raw_failure
            if (
                first_failed_replicate_unavailable is None
                and isinstance(raw_failure, ScientificUnavailable)
            ):
                first_failed_replicate_unavailable = raw_failure
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
                if not isinstance(cause, FactualCause):
                    cause = FactualCause(
                        domain="fit",
                        origin="bootstrap",
                        code=None,
                        type="FatalBootstrapRefit",
                        message="Fatal Bootstrap refit failed.",
                    )
                if first_failed_replicate_diagnostic is None:
                    first_failed_replicate_diagnostic = cause
                return finish(build_result(
                    "failed",
                    diagnostic=cause,
                    provenance={"replicate": index, "settlement": "failed"},
                ))
            progress_cause = emit_progress(completed=index + 1)
            if progress_cause is not None:
                return finish(stopped_result(progress_cause))
            continue
        values = {
            name: _parameter_value(fitted, name)
            for name in config.parameter_names
        }
        for name, value in values.items():
            successes[name].append(value)
        completed += 1
        successful += 1
        progress_cause = emit_progress(completed=index + 1)
        if progress_cause is not None:
            return finish(stopped_result(progress_cause))

    success_count = successful
    required = required_successes
    threshold_met = success_count >= required
    alpha = 1.0 - config.confidence_level
    intervals: dict[str, object] = {}
    for name, values in successes.items():
        if not threshold_met:
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
        }

    return finish(build_result(
        "available" if threshold_met else "unavailable",
        intervals={
            name: _bootstrap_interval_record(record)
            for name, record in intervals.items()
        },
        unavailable_reason=(
            None
            if threshold_met
            else (
                f"Bootstrap validity threshold was not met: {success_count} "
                f"successful full refits; {required} required."
            )
        ),
        provenance={"primary_fit_preserved": True},
    ))

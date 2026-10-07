"""Named, provenance-complete uncertainty methods for completed Fits.

The functions in this module do not evaluate models or own an optimizer.  They
drive the canonical Fit objective and optimizer supplied by the global fitting
composition boundary, then return immutable Profile or Bootstrap results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

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
from kindred.core.fitting_optimization import fit_bound_position, fit_parameters_with_fixed
from kindred.core.fitting_method_policy import UNCERTAINTY_FIT_METHODS, validate_fit_method
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.core.species import validate_evaluation_count

__all__ = [
    "BOOTSTRAP_SEED_RANGE",
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


_PROFILE_METHOD = "weighted Gaussian profile likelihood"
_BOOTSTRAP_METHOD = "stratified residual bootstrap"
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
            if scope == "coupled":
                raise ValueError(
                    "Coupled coordinates are not public Profile targets."
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
        if method not in UNCERTAINTY_FIT_METHODS:
            raise ValueError("Profile likelihood refits with Trust Region Reflective or Dogbox.")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "max_bracketing_steps_per_side", points)
        object.__setattr__(self, "max_nfev_per_point", evaluations)
        object.__setattr__(self, "optimizer_method", method)


def _profile_likelihood_threshold(
    confidence_level: float,
    observation_count: int,
    fitted_value_count: int,
) -> float | None:
    """Likelihood-ratio threshold n·ln(1 + F_level(1, n−p)/(n−p)).

    ``n`` is the weighted residual count and ``p`` the optimizer coordinate
    count.  The threshold does not exist when n ≤ p.
    """

    from scipy.special import fdtri

    freedom = int(observation_count) - int(fitted_value_count)
    if freedom <= 0:
        return None
    quantile = float(fdtri(1.0, float(freedom), float(confidence_level)))
    return float(observation_count) * math.log1p(quantile / float(freedom))


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
            if scope == "coupled":
                raise ValueError(
                    "Coupled coordinates are not residual Bootstrap targets."
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
        if method not in UNCERTAINTY_FIT_METHODS:
            raise ValueError("Residual bootstrap refits with Trust Region Reflective or Dogbox.")
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
    """Resolution facts for one Profile lower or upper endpoint.

    ``bound`` means the side is a one-sided limit: the profile reached its
    search limit, or the estimate already sits at that limit, without crossing
    the threshold.  At a threshold crossing, ``limiting_bounds`` names each
    other fitted coordinate that reached a configured bound along the profile,
    with that bound's side; bounds already active at the optimum are left out.
    """

    status: ProfileLikelihoodEndpointStatus
    reason: str | None = None
    limiting_bounds: tuple[tuple[object, str], ...] = ()

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"bound", "limiting", "threshold", "failed", "unavailable"}:
            raise ValueError("Profile endpoint status is invalid.")
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodInterval:
    """One coordinate's endpoints and bounded resolution facts.

    A ``one-sided`` interval has at least one side that is a limit rather
    than a threshold crossing; its value on that side is ``None``.
    """

    status: Literal["available", "one-sided", "partial", "unavailable"]
    lower: float | None
    upper: float | None
    lower_resolution: ProfileLikelihoodEndpoint
    upper_resolution: ProfileLikelihoodEndpoint

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "one-sided", "partial", "unavailable"}:
            raise ValueError("Profile interval status is invalid.")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Profile interval endpoints must be finite.")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodResult:
    """Immutable factual result of a weighted-Gaussian Profile analysis."""

    status: ProfileLikelihoodStatus
    method: str
    confidence_level: float
    likelihood_ratio_threshold: float | None
    likelihood_model: str
    observation_count: int
    fitted_value_count: int
    intervals: Mapping[object, ProfileLikelihoodInterval]
    unavailable_reason: str | None = None
    diagnostic: FactualCause | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "partial", "unavailable", "stopped", "failed"}:
            raise ValueError("Profile result status is invalid.")
        confidence = _confidence(self.confidence_level)
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
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "intervals", MappingProxyType(intervals))
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance or {})))


@dataclass(frozen=True, slots=True)
class BootstrapPercentileInterval:
    """One parameter's equal-tailed bootstrap percentile interval."""

    status: Literal["available", "unavailable"]
    lower: float | None
    upper: float | None

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


@dataclass(frozen=True, slots=True)
class ResidualBootstrapResult:
    """Immutable factual result of deterministic residual Bootstrap refits."""

    status: BootstrapStatus
    method: str
    seed: int
    replicates: int
    confidence_level: float
    resampling: Mapping[str, str]
    residual_centering: Mapping[str, str]
    required_successes: int
    successful_replicates: int
    failed_replicates: int
    intervals: Mapping[object, BootstrapPercentileInterval]
    first_failed_replicate_unavailable: ScientificUnavailable | None = None
    unavailable_reason: str | None = None
    diagnostic: FactualCause | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status)
        if status not in {"available", "unavailable", "stopped", "failed"}:
            raise ValueError("Bootstrap result status is invalid.")
        seed = validate_bootstrap_seed(self.seed)
        replicates = _bounded_positive_integer(
            self.replicates,
            field="Bootstrap result replicate count",
            maximum=_BOOTSTRAP_MAX_REPLICATES,
        )
        confidence = _confidence(self.confidence_level)
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
        if successful + failed > replicates:
            raise ValueError("Bootstrap result replicate counts are invalid.")
        intervals = dict(self.intervals)
        if any(not isinstance(item, BootstrapPercentileInterval) for item in intervals.values()):
            raise TypeError("Bootstrap intervals require typed percentile records.")
        if self.diagnostic is not None and not isinstance(self.diagnostic, FactualCause):
            raise TypeError("Bootstrap diagnostic must be a FactualCause.")
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
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "required_successes", required)
        object.__setattr__(self, "successful_replicates", successful)
        object.__setattr__(self, "failed_replicates", failed)
        object.__setattr__(self, "intervals", MappingProxyType(intervals))
        object.__setattr__(self, "resampling", MappingProxyType(dict(self.resampling)))
        object.__setattr__(self, "residual_centering", MappingProxyType(dict(self.residual_centering)))
        object.__setattr__(self, "unavailable_reason", reason)
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance or {})))


def profile_likelihood_settlement_result(
    *,
    config: ProfileLikelihoodConfig,
    status: Literal["stopped", "failed"],
    cause: FactualCause,
    observation_count: int,
    fitted_value_count: int,
) -> ProfileLikelihoodResult:
    """Construct a stopped/failed Profile result from call-local facts."""

    return ProfileLikelihoodResult(
        status=status,
        method=_PROFILE_METHOD,
        confidence_level=config.confidence_level,
        likelihood_ratio_threshold=_profile_likelihood_threshold(
            config.confidence_level, observation_count, fitted_value_count,
        ),
        likelihood_model="weighted Gaussian residuals with profiled variance",
        observation_count=int(observation_count),
        fitted_value_count=int(fitted_value_count),
        intervals={},
        diagnostic=cause,
        provenance={"settlement": status, "primary_fit_preserved": True},
    )


def _bootstrap_resampling_descriptions(
    payloads: Sequence[FitDatasetSpec],
) -> tuple[dict[str, str], dict[str, str]]:
    """Describe each dataset's actual donor grouping and centering policy, by dataset."""
    resampling: dict[str, str] = {}
    centering: dict[str, str] = {}
    for payload in payloads:
        if payload.x_name == "t":
            resampling[payload.dataset_id] = "Residuals are drawn independently for each target."
            centering[payload.dataset_id] = "The residuals of each target are centered on their mean."
        else:
            resampling[payload.dataset_id] = (
                "Joint X and Y residual rows are drawn within each missing-value pattern, "
                "and one row serves every available quantity."
            )
            centering[payload.dataset_id] = (
                "The residuals of each quantity are centered on their mean within each "
                "missing-value pattern."
            )
    return resampling, centering


def residual_bootstrap_settlement_result(
    *,
    config: ResidualBootstrapConfig,
    payloads: Sequence[FitDatasetSpec],
    status: Literal["stopped", "failed"],
    cause: FactualCause,
    settlement: str,
) -> ResidualBootstrapResult:
    """Construct a stopped/failed Bootstrap result from local facts."""

    required_successes = max(
        config.minimum_successful_replicates,
        int(math.ceil(config.replicates * config.minimum_success_fraction)),
    )
    resampling, centering = _bootstrap_resampling_descriptions(payloads)
    return ResidualBootstrapResult(
        status=status,
        method=_BOOTSTRAP_METHOD,
        seed=config.seed,
        replicates=config.replicates,
        confidence_level=config.confidence_level,
        resampling=resampling,
        residual_centering=centering,
        required_successes=required_successes,
        successful_replicates=0,
        failed_replicates=0,
        intervals={},
        diagnostic=cause,
        provenance={
            "settlement": str(settlement),
            "primary_fit_preserved": True,
            "completed_replicates": 0,
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
    the likelihood-ratio threshold before crossing interpolation.
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
        reason=None if record.get("reason") is None else str(record["reason"]),
        limiting_bounds=tuple(record.get("limiting_bounds", ())),
    )


def _profile_interval_record(
    record: Mapping[str, object],
) -> ProfileLikelihoodInterval:
    resolution = record.get("resolution", {})
    lower_endpoint = resolution.get("lower", {})
    upper_endpoint = resolution.get("upper", {})
    return ProfileLikelihoodInterval(
        status=str(record["status"]),  # type: ignore[arg-type]
        lower=None if record.get("lower") is None else float(record["lower"]),
        upper=None if record.get("upper") is None else float(record["upper"]),
        lower_resolution=_profile_endpoint_record(lower_endpoint),
        upper_resolution=_profile_endpoint_record(upper_endpoint),
    )


def profile_likelihood_result(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    initial_parameters: Mapping[object, float],
    bounds: Mapping[object, tuple[float, float]],
    config: ProfileLikelihoodConfig,
    decode_coordinate: Callable[[object, float], float],
    ftol: float,
    xtol: float,
    base_residuals: np.ndarray,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> ProfileLikelihoodResult:
    """Fix selected coordinates and reoptimize nuisances on a bounded profile.

    Discovery moves monotonically away from the optimum and stops at the first
    successful threshold bracket.  A bracket is then refined only through
    actual nuisance refits; no endpoint is produced by interpolation.  The
    likelihood-ratio threshold is calibrated when the analysis runs, from the
    weighted residual count and the optimizer coordinate count.  Refits use
    ``ftol`` and ``xtol``.
    """

    if not isinstance(config, ProfileLikelihoodConfig):
        raise TypeError("Profile likelihood requires ProfileLikelihoodConfig.")
    if not config.parameter_names:
        raise ValueError("Ordinary Profile likelihood requires at least one coordinate.")
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
    fitted_value_count = len(initial)
    base_sse = _sse(base_residuals)
    observation_count = int(base_residuals.size)
    threshold = _profile_likelihood_threshold(
        config.confidence_level, observation_count, fitted_value_count,
    )
    optimum_positions = {
        coordinate: fit_bound_position(initial[coordinate], low, high)
        for coordinate, (low, high) in bound_map.items()
    }
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

    def failure_result(
        cause: FactualCause,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> ProfileLikelihoodResult:
        return finish(
            ProfileLikelihoodResult(
                status="failed",
                method=_PROFILE_METHOD,
                confidence_level=config.confidence_level,
                likelihood_ratio_threshold=threshold,
                likelihood_model="weighted Gaussian residuals with profiled variance",
                observation_count=observation_count,
                fitted_value_count=fitted_value_count,
                intervals={},
                diagnostic=cause,
                provenance={"settlement": "failed", **dict(provenance or {})},
            )
        )

    if threshold is None:
        return finish(ProfileLikelihoodResult(
            status="unavailable",
            method=_PROFILE_METHOD,
            confidence_level=config.confidence_level,
            likelihood_ratio_threshold=None,
            likelihood_model="weighted Gaussian residuals with profiled variance",
            observation_count=observation_count,
            fitted_value_count=fitted_value_count,
            intervals={},
            unavailable_reason=(
                f"Profile likelihood is unavailable: {observation_count} weighted "
                f"residuals do not exceed {fitted_value_count} fitted values."
            ),
            provenance={"settlement": "unavailable"},
        ))
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
        ) -> tuple[float | None, str, Mapping[str, object]]:
            nonlocal work_done, active_side, active_discovery, active_refinement
            active_side = side
            if bound == best:
                # The estimate sits on this side's bound, so the search range
                # leaves nothing to profile on this side.
                return None, "bound", {"status": "bound", "reason": "search_limit_reached"}
            inner_x = best
            inner_parameters = dict(initial)
            bracket: tuple[
                float, float, float, Mapping[str, float], Mapping[str, float]
            ] | None = None

            def threshold_resolution(parameters: Mapping[str, float]) -> Mapping[str, object]:
                """Name each other fitted coordinate that reached a bound along the profile.

                Bounds already active at the optimum are left out.
                """

                return {
                    "status": "threshold",
                    "limiting_bounds": tuple(
                        (coordinate, position)
                        for coordinate, (other_low, other_high) in bound_map.items()
                        if coordinate != name
                        and (
                            position := fit_bound_position(
                                float(parameters[coordinate]), other_low, other_high,
                            )
                        ) not in {"interior", optimum_positions[coordinate]}
                    ),
                }

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
                        ftol=ftol,
                        xtol=xtol,
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
                        "fixed_optimizer_value": float(fixed),
                        "status": "unavailable",
                    }
                    return record, None
                candidate_sse = _sse(evaluation_residuals)
                lr_details = _likelihood_ratio_details(
                    candidate_sse=candidate_sse,
                    best_sse=base_sse,
                    n=observation_count,
                )
                record = {
                    "fixed_optimizer_value": float(fixed),
                    "status": "success",
                    "likelihood_ratio": (
                        None
                        if lr_details.outcome == "threshold_exceeded_exact_fit"
                        else lr_details.transport_value
                    ),
                    "likelihood_ratio_outcome": lr_details.outcome,
                }
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
                    # A step that rounds back onto the estimate moves on.
                    continue
                record, fitted_parameters = append_fit(
                    fixed,
                    stage="discovery",
                    discovery_step=discovery_step,
                    refinement_step=None,
                    warm_start=inner_parameters,
                )
                if record["status"] != "success":
                    return None, "failed", {
                        "status": "failed",
                        "reason": "scientific_unavailable",
                    }
                outcome = str(record["likelihood_ratio_outcome"])
                if outcome == "threshold_exceeded_exact_fit":
                    return best, "limiting", {"status": "limiting"}
                lr = float(record["likelihood_ratio"])
                if lr >= threshold:
                    bracket = (
                        inner_x,
                        float(fixed),
                        lr,
                        dict(inner_parameters),
                        dict(fitted_parameters),
                    )
                    break
                inner_x = float(fixed)
                inner_parameters = dict(fitted_parameters or inner_parameters)
            if bracket is None:
                # The last discovery step refits at the bound itself.
                return None, "bound", {"status": "bound", "reason": "search_limit_reached"}
            left_x, right_x, right_lr, left_parameters, right_parameters = bracket
            lr_tolerance = max(1e-6, 1e-4 * threshold)
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
                    return None, "failed", {
                        "status": "failed",
                        "reason": "scientific_unavailable",
                    }
                if str(record["likelihood_ratio_outcome"]) == "threshold_exceeded_exact_fit":
                    return best, "limiting", {"status": "limiting"}
                midpoint_lr = float(record["likelihood_ratio"])
                width = abs(right_x - left_x)
                if abs(midpoint_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return midpoint, "threshold", threshold_resolution(fitted_parameters)
                if abs(right_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return right_x, "threshold", threshold_resolution(right_parameters)
                if midpoint_lr >= threshold:
                    right_x, right_lr = midpoint, midpoint_lr
                    right_parameters = dict(fitted_parameters)
                else:
                    left_x = midpoint
                    left_parameters = dict(fitted_parameters or left_parameters)
            return None, "unavailable", {
                "status": "unavailable",
                "reason": (
                    "binary64_representability_exhausted"
                    if representability_exhausted
                    else "midpoint_refit_budget_exhausted"
                ),
            }

        side_results: dict[str, tuple[float | None, str, Mapping[str, object]]] = {}
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
                    },
                )
        lower, lower_outcome, lower_resolution = side_results["lower"]
        upper, upper_outcome, upper_resolution = side_results["upper"]
        outcomes = (lower_outcome, upper_outcome)
        if any(outcome in {"failed", "unavailable"} for outcome in outcomes):
            interval_status = (
                "unavailable" if lower is None and upper is None and "bound" not in outcomes
                else "partial"
            )
            any_partial = True
        elif "bound" in outcomes:
            interval_status = "one-sided"
        else:
            interval_status = "available"
        intervals[name] = {
            "status": interval_status,
            "lower": None if lower is None else decode_coordinate(name, lower),
            "upper": None if upper is None else decode_coordinate(name, upper),
            "resolution": {
                "lower": lower_resolution,
                "upper": upper_resolution,
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
        confidence_level=config.confidence_level,
        likelihood_ratio_threshold=threshold,
        likelihood_model="weighted Gaussian residuals with profiled variance",
        observation_count=observation_count,
        fitted_value_count=fitted_value_count,
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


def _parameter_value(
    result: object,
    name: object,
    group_member_ids: Mapping[object, str],
) -> float:
    """Resolve one typed uncertainty coordinate from terminal Fit custody.

    A parameter group's value is read from one of its included members, all
    of which carry the same fitted value.
    """

    scope = name.scope
    if scope == "shared":
        value = float(result.shared_params[name.name])
    elif scope == "dataset":
        value = float(result.dataset_params[name.dataset_id][name.name])
    else:
        # The configurations admit only shared, dataset and group targets.
        value = float(result.dataset_params[group_member_ids[name]][name.name])
    if not math.isfinite(value):
        raise ValueError(
            f"Bootstrap fit returned nonfinite parameter {_coordinate_display(name)!r}."
        )
    return value


_BootstrapRefitOutcome = _fit_execution.GlobalFitResult | ScientificUnavailable


def _bootstrap_payloads(
    *,
    payloads: Sequence[FitDatasetSpec],
    fitted_info: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[FitDatasetSpec, ...]:
    captured: list[FitDatasetSpec] = []
    for payload in payloads:
        info = fitted_info[payload.dataset_id]
        sources = dict(payload.observations)
        models = dict(info.model_series)
        if payload.x_name != "t":
            sources[payload.x_name] = payload.x_observations
            projection = info.x_projection
            x_by_row = dict(zip(projection.source_rows, projection.selected_model_x, strict=True))
            models[payload.x_name] = np.asarray([
                x_by_row[row] for row in payload.x_observations.source_rows
            ])
        synthetic = {name: np.asarray(source.y).copy() for name, source in sources.items()}
        if payload.x_name == "t":
            for name, source in sources.items():
                model = models[name]
                residual = source.y - model
                residual = residual - np.mean(residual)
                indexes = rng.integers(0, len(residual), size=len(residual))
                synthetic[name] = model + residual[indexes]
        else:
            # Resample complete residual vectors jointly. Group by observed
            # quantities so missing cells never borrow another column's row.
            positions = {name: {row: i for i, row in enumerate(source.source_rows)}
                         for name, source in sources.items()}
            patterns: dict[tuple[str, ...], list[int]] = {}
            for row in info.x_projection.source_rows:
                pattern = tuple(name for name in sorted(sources) if row in positions[name])
                patterns.setdefault(pattern, []).append(row)
            for pattern, rows in patterns.items():
                donors = rng.integers(0, len(rows), size=len(rows))
                for name in pattern:
                    indexes = np.asarray([positions[name][row] for row in rows])
                    model = np.asarray(models[name])[indexes]
                    residual = sources[name].y[indexes] - model
                    residual = residual - np.mean(residual)
                    synthetic[name][indexes] = model + residual[donors]
        series = {
            name: FitObservationSeries(
                raw_observation_id=source.raw_observation_id,
                raw_column=source.raw_column, t=source.t,
                y=synthetic[name], source_rows=source.source_rows,
            ) for name, source in sources.items()
        }
        captured.append(FitDatasetSpec(
            dataset_id=payload.dataset_id, species_list=payload.species_list,
            observations={name: series[name] for name in payload.species_list},
            target_weights=payload.target_weights, x_name=payload.x_name,
            x_observations=series.get(payload.x_name),
        ))
    return tuple(captured)


def _bootstrap_interval_record(
    record: Mapping[str, object],
) -> BootstrapPercentileInterval:
    return BootstrapPercentileInterval(
        status=str(record["status"]),  # type: ignore[arg-type]
        lower=None if record.get("lower") is None else float(record["lower"]),
        upper=None if record.get("upper") is None else float(record["upper"]),
    )


def residual_bootstrap_result(
    *,
    terminal_result: object,
    payloads: Sequence[FitDatasetSpec],
    config: ResidualBootstrapConfig,
    refit: Callable[[Sequence[FitDatasetSpec]], _BootstrapRefitOutcome],
    group_member_ids: Mapping[object, str],
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> ResidualBootstrapResult:
    """Run deterministic full refits for stratified residual replicates.

    ``group_member_ids`` names one included member of each group target.
    """

    if not isinstance(config, ResidualBootstrapConfig):
        raise TypeError("Residual bootstrap requires ResidualBootstrapConfig.")
    if not config.parameter_names:
        raise ValueError("Ordinary residual Bootstrap requires at least one coordinate.")
    completed = 0
    successful = 0
    failed = 0
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
        unavailable_reason: str | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> ResidualBootstrapResult:
        """Build one bounded Bootstrap settlement from current call-local facts."""

        resampling, centering = _bootstrap_resampling_descriptions(payloads)
        return ResidualBootstrapResult(
            status=status,
            method=_BOOTSTRAP_METHOD,
            seed=config.seed,
            replicates=config.replicates,
            confidence_level=config.confidence_level,
            resampling=resampling,
            residual_centering=centering,
            required_successes=required_successes,
            successful_replicates=successful,
            failed_replicates=failed,
            intervals={} if intervals is None else intervals,
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
        _parameter_value(terminal_result, name, group_member_ids)

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
                rng=rng,
            )
            fitted = refit(synthetic)
        except (FittingCancelled, SimulationCancelled) as exc:
            return finish(stopped_result(factual_cause_from_exception(
                exc, domain="fit", origin="bootstrap",
            )))
        except Exception as exc:
            completed += 1
            failed += 1
            return finish(build_result(
                "failed",
                diagnostic=factual_cause_from_exception(exc, domain="fit", origin="bootstrap"),
                provenance={"replicate": index, "settlement": "failed"},
            ))
        if isinstance(fitted, ScientificUnavailable):
            completed += 1
            failed += 1
            if first_failed_replicate_unavailable is None:
                first_failed_replicate_unavailable = fitted
            progress_cause = emit_progress(completed=index + 1)
            if progress_cause is not None:
                return finish(stopped_result(progress_cause))
            continue
        values = {
            name: _parameter_value(fitted, name, group_member_ids)
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
            # The result's unavailable reason states the shortfall once.
            intervals[name] = {"status": "unavailable", "lower": None, "upper": None}
            continue
        interval = np.percentile(
            np.asarray(values, dtype=float),
            [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
        )
        intervals[name] = {
            "status": "available",
            "lower": float(interval[0]),
            "upper": float(interval[1]),
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
                f"successful full refits, {required} required."
            )
        ),
        provenance={"primary_fit_preserved": True},
    ))

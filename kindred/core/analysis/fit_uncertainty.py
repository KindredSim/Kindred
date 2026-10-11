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
from kindred.core.analysis.fit_statistics import finite_sum_of_squares
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_completion import FitExtensionProgress
from kindred.core.fitting_optimization import fit_bound_position, fit_parameters_with_fixed
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)

__all__ = [
    "BOOTSTRAP_SEED_RANGE",
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
BOOTSTRAP_SEED_RANGE = (0, 2_000_000_000)


def _confidence(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("Confidence level must lie strictly between zero and one.")
    return result


def _coordinate_display(value: _fit_execution.FitCoordinate) -> str:
    return value.display_name


def _names(values: Sequence[object], *, field: str) -> tuple[object, ...]:
    names = tuple(values)
    if not names:
        raise ValueError(f"{field} must contain at least one coordinate.")
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
        # The dialog offers only the ordinary (shared, dataset and group)
        # coordinates of the Fit setup.
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))


def _profile_likelihood_threshold(
    confidence_level: float,
    observation_count: int,
    fitted_value_count: int,
) -> float | None:
    """Likelihood-ratio threshold n·ln(1 + F_level(1, n−p)/(n−p)).

    ``n`` is the weighted residual count and ``p`` the fitted value count:
    the optimizer coordinates plus one fitted curve location per measured-X
    row.  The threshold does not exist when n ≤ p.
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
        # The dialog offers only the ordinary (shared, dataset and group)
        # coordinates of the Fit setup.
        fraction = self.minimum_success_fraction
        if self.minimum_successful_replicates > self.replicates:
            raise ValueError(
                "Bootstrap minimum successful replicates must be within the roster."
            )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Bootstrap minimum success fraction must be in (0, 1].")
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))


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

    # The Profile loop builds it from literal statuses.


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
        # The Profile owner builds every result from its admitted
        # configuration, literal statuses, a factual reason for an unavailable
        # result and a diagnostic for a stopped or failed one.
        object.__setattr__(self, "method", _PROFILE_METHOD)
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class BootstrapPercentileInterval:
    """One parameter's equal-tailed bootstrap percentile interval."""

    status: Literal["available", "unavailable"]
    lower: float | None
    upper: float | None



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
        # The Bootstrap owner builds every result from its admitted
        # configuration and its own replicate counters, with literal statuses,
        # a factual reason for an unavailable result and a diagnostic for a
        # stopped or failed one.
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "resampling", MappingProxyType(dict(self.resampling)))
        object.__setattr__(self, "residual_centering", MappingProxyType(dict(self.residual_centering)))
        object.__setattr__(self, "provenance", _freeze_fact(dict(self.provenance)))


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
    """Describe each dataset's actual resampling and centering policy, by dataset."""
    resampling: dict[str, str] = {}
    centering: dict[str, str] = {}
    for payload in payloads:
        if payload.x_name == "t":
            resampling[payload.dataset_id] = "Residuals are drawn independently for each target."
            centering[payload.dataset_id] = "The residuals of each target are centered on their mean."
        else:
            resampling[payload.dataset_id] = (
                "Each row keeps its own X and Y residuals, multiplied by one random sign per row."
            )
            centering[payload.dataset_id] = "Residuals are not centered."
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

    candidate = candidate_sse
    best = best_sse
    count = n
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
    fitted_value_count: int,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[FitExtensionProgress], None] | None = None,
) -> ProfileLikelihoodResult:
    """Fix selected coordinates and reoptimize nuisances on a bounded profile.

    Discovery moves monotonically away from the optimum and stops at the first
    successful threshold bracket.  A bracket is then refined only through
    actual nuisance refits; no endpoint is produced by interpolation.  The
    likelihood-ratio threshold is calibrated when the analysis runs, from the
    weighted residual count and ``fitted_value_count``.  Refits use
    ``ftol`` and ``xtol``.
    """

    # The admitted configuration names at least one coordinate, and the Fit
    # layout supplies finite ordered bounds.
    initial = {name: float(value) for name, value in initial_parameters.items()}
    bound_map = {
        name: (float(value[0]), float(value[1]))
        for name, value in bounds.items()
    }
    base_sse = finite_sum_of_squares(base_residuals)
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
                f"{observation_count} observations do not exceed "
                f"{fitted_value_count} fitted values."
            ),
            provenance={"settlement": "unavailable"},
        ))
    intervals: dict[object, ProfileLikelihoodInterval] = {}
    any_partial = False
    for name in config.parameter_names:
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        # The fitted optimum lies inside its admitted bounds.
        low, high = bound_map[name]
        best = initial[name]
        active_coordinate = _coordinate_display(name)

        def side_profile(
            side: str,
            bound: float,
        ) -> tuple[float | None, ProfileLikelihoodEndpoint]:
            nonlocal work_done, active_side, active_discovery, active_refinement
            active_side = side
            if bound == best:
                # The estimate sits on this side's bound, so the search range
                # leaves nothing to profile on this side.
                return None, ProfileLikelihoodEndpoint("bound", "search_limit_reached")
            inner_x = best
            inner_parameters = dict(initial)
            bracket: tuple[
                float, float, float, Mapping[str, float], Mapping[str, float]
            ] | None = None

            def threshold_resolution(parameters: Mapping[str, float]) -> ProfileLikelihoodEndpoint:
                """Name each other fitted coordinate that reached a bound along the profile.

                Bounds already active at the optimum are left out.
                """

                return ProfileLikelihoodEndpoint(
                    "threshold",
                    limiting_bounds=tuple(
                        (coordinate, position)
                        for coordinate, (other_low, other_high) in bound_map.items()
                        if coordinate != name
                        and (
                            position := fit_bound_position(
                                float(parameters[coordinate]), other_low, other_high,
                            )
                        ) not in {"interior", optimum_positions[coordinate]}
                    ),
                )

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
                # A scientific residual call, never an optimizer penalty.
                evaluation_residuals = objective.evaluate_scientific_residuals(
                    np.asarray(
                        [fit.parameters[item] for item in initial],
                        dtype=float,
                    ),
                )
                if evaluation_residuals is None:
                    record = {
                        "fixed_optimizer_value": float(fixed),
                        "status": "unavailable",
                    }
                    return record, None
                candidate_sse = finite_sum_of_squares(evaluation_residuals)
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
                    return None, ProfileLikelihoodEndpoint("failed", "scientific_unavailable")
                outcome = str(record["likelihood_ratio_outcome"])
                if outcome == "threshold_exceeded_exact_fit":
                    return best, ProfileLikelihoodEndpoint("limiting")
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
                return None, ProfileLikelihoodEndpoint("bound", "search_limit_reached")
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
                    return None, ProfileLikelihoodEndpoint("failed", "scientific_unavailable")
                if str(record["likelihood_ratio_outcome"]) == "threshold_exceeded_exact_fit":
                    return best, ProfileLikelihoodEndpoint("limiting")
                midpoint_lr = float(record["likelihood_ratio"])
                width = abs(right_x - left_x)
                if abs(midpoint_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return midpoint, threshold_resolution(fitted_parameters)
                if abs(right_lr - threshold) <= lr_tolerance and width <= coordinate_tolerance:
                    return right_x, threshold_resolution(right_parameters)
                if midpoint_lr >= threshold:
                    right_x, right_lr = midpoint, midpoint_lr
                    right_parameters = dict(fitted_parameters)
                else:
                    left_x = midpoint
                    left_parameters = dict(fitted_parameters or left_parameters)
            return None, ProfileLikelihoodEndpoint(
                "unavailable",
                "binary64_representability_exhausted"
                if representability_exhausted
                else "midpoint_refit_budget_exhausted",
            )

        side_results: dict[str, tuple[float | None, ProfileLikelihoodEndpoint]] = {}
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
        lower, lower_resolution = side_results["lower"]
        upper, upper_resolution = side_results["upper"]
        outcomes = (lower_resolution.status, upper_resolution.status)
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
        intervals[name] = ProfileLikelihoodInterval(
            status=interval_status,
            lower=None if lower is None else decode_coordinate(name, lower),
            upper=None if upper is None else decode_coordinate(name, upper),
            lower_resolution=lower_resolution,
            upper_resolution=upper_resolution,
        )

    interval_statuses = {item.status for item in intervals.values()}
    if interval_statuses == {"unavailable"}:
        result_status = "unavailable"
        unavailable_reason = "no requested interval could be determined."
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
        intervals=intervals,
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
        return float(result.shared_params[name.name])
    if scope == "dataset":
        return float(result.dataset_params[name.dataset_id][name.name])
    # The configurations admit only shared, dataset and group targets.
    return float(result.dataset_params[group_member_ids[name]][name.name])


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
            # A measured-X row's residual vector points away from the curve at
            # that row's own location, so it stays on its row with one random
            # sign for the whole vector.
            rows = info.x_projection.source_rows
            signs = dict(zip(rows, rng.choice((-1.0, 1.0), size=len(rows)), strict=True))
            for name, source in sources.items():
                model = np.asarray(models[name])
                row_signs = np.asarray([signs[row] for row in source.source_rows])
                synthetic[name] = model + row_signs * (source.y - model)
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

    # The admitted configuration names at least one coordinate.
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
    # The terminal Fit retains one DatasetFitInfo per included dataset.
    fitted_info = {info.dataset_id: info for info in terminal_result.dataset_info}

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
    intervals: dict[object, BootstrapPercentileInterval] = {}
    for name, values in successes.items():
        if not threshold_met:
            # The result's unavailable reason states the shortfall once.
            intervals[name] = BootstrapPercentileInterval("unavailable", None, None)
            continue
        interval = np.percentile(
            np.asarray(values, dtype=float),
            [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
        )
        intervals[name] = BootstrapPercentileInterval(
            "available", float(interval[0]), float(interval[1]),
        )

    return finish(build_result(
        "available" if threshold_met else "unavailable",
        intervals=intervals,
        unavailable_reason=(
            None
            if threshold_met
            else (
                f"{success_count} successful refit{'' if success_count == 1 else 's'}, "
                f"{required} required."
            )
        ),
        provenance={"primary_fit_preserved": True},
    ))

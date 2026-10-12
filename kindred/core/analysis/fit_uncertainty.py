"""Named, provenance-complete uncertainty methods for completed Fits.

The functions in this module do not evaluate models or own an optimizer.  They
drive the canonical Fit objective and optimizer supplied by the global fitting
composition boundary, then return immutable Profile results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.analysis.fit_statistics import finite_sum_of_squares
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_completion import FitExtensionProgress
from kindred.core.fitting_optimization import (
    _JACOBIAN_RANK_RTOL,
    fit_bound_position,
    fit_parameters_with_fixed,
)
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)

__all__ = [
    "ProfileLikelihoodConfig",
    "ProfileLikelihoodOptions",
    "ProfileLikelihoodEndpoint",
    "ProfileLikelihoodInterval",
    "ProfileLikelihoodResult",
    "ProfileCutoff",
    "ProfileNoiseBlock",
    "ProfileNoiseModel",
    "profile_likelihood_settlement_result",
    "profile_likelihood_result",
    "profile_noise_model",
    "undetermined_coordinates",
]


_PROFILE_METHOD = "weighted Gaussian profile likelihood"
_LIKELIHOOD_MODEL = "independent Gaussian residuals with one noise level per series"


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
class ProfileLikelihoodOptions:
    """The Profile settings the user chooses, independent of its targets."""

    confidence_level: float = 0.95
    max_bracketing_steps_per_side: int = 8
    max_nfev_per_point: int = 300
    optimizer_method: str = "trf"

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodConfig:
    """One Profile run: its targets and the options it uses."""

    parameter_names: tuple[object, ...]
    options: ProfileLikelihoodOptions

    def __post_init__(self) -> None:
        # The Fit setup supplies only ordinary (shared, dataset and group)
        # coordinates of its roster.
        object.__setattr__(
            self,
            "parameter_names",
            _names(tuple(self.parameter_names), field="Profile parameter names"),
        )


@dataclass(frozen=True, slots=True)
class ProfileNoiseBlock:
    """One noise block's estimate: one target of a time-X dataset, or all
    series of a measured-X dataset (``weighted``)."""

    dataset_id: str
    series: str
    observations: int
    degrees_of_freedom: float
    # The noise standard deviation in the series' scientific unit, or in
    # weighted units for a measured-X block; None when the fitted values
    # explain all of the block's observations.
    noise_sd: float | None
    weighted: bool = False


@dataclass(frozen=True, slots=True)
class ProfileCutoff:
    """One coordinate's noise variance and its Satterthwaite degrees of freedom."""

    noise_variance: float
    degrees_of_freedom: float

    def rise(self, level: float) -> float:
        """The weighted-SSE rise κ·F_level(1, ν) that ends a profile at ``level``."""

        from scipy.special import fdtri

        return self.noise_variance * float(fdtri(1.0, self.degrees_of_freedom, level))


@dataclass(frozen=True, slots=True)
class ProfileNoiseModel:
    """Every block's noise estimate and each optimizer coordinate's cut-off;
    no coordinate has a cut-off when no block has an estimate."""

    blocks: tuple[ProfileNoiseBlock, ...]
    cutoffs: Mapping[object, ProfileCutoff]


def profile_noise_model(
    *,
    objective: object,
    vector: np.ndarray,
    keys: Sequence[object],
    bounds: Mapping[object, tuple[float, float]],
    dataset_info: Sequence[object],
) -> ProfileNoiseModel | None:
    """Estimate each noise block's Gaussian noise from the fit's own residuals.

    A block, one target of a time-X dataset or all series of a measured-X
    dataset, has the variance of its weighted residuals over its residual
    degrees of freedom: its residuals minus their leverages, minus one per
    measured-X curve location matched inside a segment.  A residual the fit
    reproduces exactly observes no noise and does not count.  Leverages come
    from the column-scaled weighted Jacobian at the estimate, by central
    differences inside the bounds.  A coordinate's variance averages the
    variances of the blocks with an estimate by the share of its information
    each carries: the part of its Jacobian column the other coordinates do not
    explain, within the Jacobian's numerical rank, so values the data see only
    in combination share that combination's information.  A value converged
    onto a bound at zero, like an exact zero, is stepped by one unit within its
    bounds; any other value by a step relative to it.  Returns None when a
    Jacobian step is scientifically unavailable.
    """

    weighted = np.concatenate([info.weighted_residuals for info in dataset_info])
    raw = np.concatenate([info.residuals for info in dataset_info])
    x = np.asarray(vector, dtype=float)
    columns = []
    for index, key in enumerate(keys):
        low, high = bounds[key]
        # Relative to the value, as SciPy's relative step.  A value at zero,
        # or converged onto a bound at zero, has no scale of its own and takes
        # a floor of one unit, within its bounds.
        position = fit_bound_position(float(x[index]), low, high)
        at_zero = x[index] == 0.0 or (position == "lower" and low == 0.0) or (position == "upper" and high == 0.0)
        step = min(
            np.finfo(float).eps ** (1.0 / 3.0) * (min(1.0, high - low) if at_zero else abs(x[index])),
            (high - low) / 4.0,
        )
        forward, backward = x.copy(), x.copy()
        forward[index] = min(x[index] + step, high)
        backward[index] = max(x[index] - step, low)
        upper = objective.evaluate_scientific_residuals(forward)
        lower = objective.evaluate_scientific_residuals(backward)
        if upper is None or lower is None:
            return None
        columns.append((upper - lower) / (forward[index] - backward[index]))
    jacobian = np.column_stack(columns)
    norms = np.linalg.norm(jacobian, axis=0)
    scaled = np.divide(jacobian, norms, out=np.zeros_like(jacobian), where=norms > 0.0)
    basis, singular, rows = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(singular > _JACOBIAN_RANK_RTOL * singular[0])) if singular[0] > 0.0 else 0
    leverage = np.sum(basis[:, :rank] ** 2, axis=1)

    blocks: list[ProfileNoiseBlock] = []
    estimated: list[tuple[slice, float, float]] = []
    start = 0
    for info in dataset_info:
        pooled = info.x_projection is not None
        for series, count in info.residual_blocks:
            span = slice(start, start + count)
            start += count
            informative = weighted[span] ** 2 > 0.0
            freedom = float(np.sum(1.0 - leverage[span][informative])) - info.matched_locations
            # A block the fitted values explain completely has no estimate.
            estimable = freedom > 1e-6 * count
            blocks.append(ProfileNoiseBlock(
                dataset_id=str(info.dataset_id), series=str(series), observations=int(count),
                degrees_of_freedom=freedom if estimable else 0.0,
                noise_sd=math.sqrt(float(np.sum((weighted if pooled else raw)[span] ** 2)) / freedom)
                if estimable else None,
                weighted=pooled,
            ))
            if estimable:
                estimated.append((span, float(np.sum(weighted[span] ** 2)) / freedom, freedom))

    # Without a block estimate no coordinate has a cut-off.
    cutoffs: dict[object, ProfileCutoff] = {}
    for index, key in enumerate(keys if estimated else ()):
        column = basis[:, :rank] @ (rows[:rank, index] / singular[:rank])
        information = [float(column[span] @ column[span]) for span, _variance, _freedom in estimated]
        if sum(information) <= 0.0:
            # A coordinate that informs no block with an estimate draws on
            # every such block by its freedom.
            information = [freedom for _span, _variance, freedom in estimated]
        total = sum(information)
        terms = [
            (part / total * variance, freedom)
            for part, (_span, variance, freedom) in zip(information, estimated)
            if part > 0.0
        ]
        variance = sum(term for term, _freedom in terms)
        cutoffs[key] = ProfileCutoff(
            variance, variance ** 2 / sum(term ** 2 / freedom for term, freedom in terms),
        )
    return ProfileNoiseModel(tuple(blocks), MappingProxyType(cutoffs))


def undetermined_coordinates(
    *,
    objective: object,
    vector: np.ndarray,
    keys: Sequence[object],
    bounds: Mapping[object, tuple[float, float]],
    base_residuals: np.ndarray,
    cutoffs: Mapping[object, ProfileCutoff],
) -> tuple[object, ...]:
    """The coordinates the data do not determine within their bounds.

    Each coordinate alone moves to its lower and then its upper bound, the
    others held at the estimate.  It is undetermined when the weighted SSE
    rises by less than its 95% Profile cut-off at both bounds.  A coordinate
    at a bound counts as unchanged on that side; a bound whose evaluation is
    unavailable or fails leaves the coordinate unflagged.
    """

    x = np.asarray(vector, dtype=float)
    base_sse = finite_sum_of_squares(base_residuals)
    flagged = []
    for index, key in enumerate(keys):
        limit = cutoffs[key].rise(0.95)
        determined = False
        for bound in bounds[key]:
            if bound == x[index]:
                continue
            trial = x.copy()
            trial[index] = bound
            try:
                residuals = objective.evaluate_scientific_residuals(trial)
            except (FittingCancelled, SimulationCancelled):
                raise
            except Exception:
                residuals = None
            if residuals is None or finite_sum_of_squares(residuals) - base_sse >= limit:
                determined = True
                break
        if not determined:
            flagged.append(key)
    return tuple(flagged)


ProfileLikelihoodStatus = Literal[
    "available",
    "partial",
    "unavailable",
    "stopped",
    "failed",
]
ProfileLikelihoodEndpointStatus = Literal[
    "bound",
    "threshold",
    "failed",
    "unavailable",
]


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
    # The weighted-SSE rise that ends this profile and its F degrees of freedom.
    cutoff: float
    degrees_of_freedom: float



@dataclass(frozen=True, slots=True)
class ProfileLikelihoodResult:
    """Immutable factual result of a weighted-Gaussian Profile analysis."""

    status: ProfileLikelihoodStatus
    method: str
    confidence_level: float
    likelihood_model: str
    intervals: Mapping[object, ProfileLikelihoodInterval]
    noise_blocks: tuple[ProfileNoiseBlock, ...] = ()
    unavailable_reason: str | None = None
    diagnostic: FactualCause | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The Profile owner builds every result from its admitted
        # configuration, literal statuses, a factual reason for an unavailable
        # result and a diagnostic for a stopped or failed one.
        object.__setattr__(self, "method", _PROFILE_METHOD)
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def profile_likelihood_settlement_result(
    *,
    config: ProfileLikelihoodConfig,
    status: Literal["stopped", "failed"],
    cause: FactualCause,
) -> ProfileLikelihoodResult:
    """Construct a stopped/failed Profile result from call-local facts."""

    return ProfileLikelihoodResult(
        status=status,
        method=_PROFILE_METHOD,
        confidence_level=config.options.confidence_level,
        likelihood_model=_LIKELIHOOD_MODEL,
        intervals={},
        diagnostic=cause,
        provenance={"settlement": status, "primary_fit_preserved": True},
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
    noise: ProfileNoiseModel,
    undetermined: frozenset[object],
    progress_callback: Callable[[FitExtensionProgress], None],
    cancellation_check: Callable[[], bool],
) -> ProfileLikelihoodResult:
    """Fix selected coordinates and reoptimize nuisances on a bounded profile.

    Discovery moves monotonically away from the optimum and stops at the first
    successful threshold bracket.  A bracket is then refined only through
    actual nuisance refits; no endpoint is produced by interpolation.  A
    profile ends where the weighted SSE has risen by its coordinate's cut-off
    from ``noise``.  ``undetermined`` coordinates, which the data do not
    determine, are never named as limiting another interval.  Refits use
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
    optimum_positions = {
        coordinate: fit_bound_position(initial[coordinate], low, high)
        for coordinate, (low, high) in bound_map.items()
    }
    options = config.options
    planned_work = len(config.parameter_names) * 2 * (
        options.max_bracketing_steps_per_side + 64
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
        progress_callback(
            FitExtensionProgress(
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

    def failure_result(
        cause: FactualCause,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> ProfileLikelihoodResult:
        return finish(
            ProfileLikelihoodResult(
                status="failed",
                method=_PROFILE_METHOD,
                confidence_level=options.confidence_level,
                likelihood_model=_LIKELIHOOD_MODEL,
                intervals={},
                diagnostic=cause,
                provenance={"settlement": "failed", **dict(provenance or {})},
            )
        )

    if not noise.cutoffs:
        return finish(ProfileLikelihoodResult(
            status="unavailable",
            method=_PROFILE_METHOD,
            confidence_level=options.confidence_level,
            likelihood_model=_LIKELIHOOD_MODEL,
            intervals={},
            unavailable_reason="the fitted values explain every observation, so the noise cannot be estimated.",
            provenance={"settlement": "unavailable"},
        ))
    intervals: dict[object, ProfileLikelihoodInterval] = {}
    any_partial = False
    for name in config.parameter_names:
        if cancellation_check():
            raise FittingCancelled()
        # The fitted optimum lies inside its admitted bounds.
        low, high = bound_map[name]
        best = initial[name]
        active_coordinate = _coordinate_display(name)
        cutoff = noise.cutoffs[name]
        threshold = cutoff.rise(options.confidence_level)

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
                        and coordinate not in undetermined
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
                if cancellation_check():
                    raise FittingCancelled()
                fit = fit_parameters_with_fixed(
                    objective,
                    dict(warm_start),
                    fixed_parameters={name: float(fixed)},
                    bounds=bound_map,
                    method=options.optimizer_method,
                    max_nfev=options.max_nfev_per_point,
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
                record = {
                    "fixed_optimizer_value": float(fixed),
                    "status": "success",
                    "rise": finite_sum_of_squares(evaluation_residuals) - base_sse,
                    "converged": bool(fit.success),
                }
                return record, fit.parameters

            for discovery_step in range(1, options.max_bracketing_steps_per_side + 1):
                fixed = (
                    bound
                    if discovery_step == options.max_bracketing_steps_per_side
                    else best
                    + (bound - best)
                    * (
                        float(discovery_step)
                        / float(options.max_bracketing_steps_per_side)
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
                rise = float(record["rise"])
                # An unfinished refit overstates the rise, so it can show a
                # side has not crossed but never that it has.
                if rise >= threshold and not record["converged"]:
                    return None, ProfileLikelihoodEndpoint("unavailable", "refit_not_converged")
                if rise >= threshold:
                    bracket = (
                        inner_x,
                        float(fixed),
                        rise,
                        dict(inner_parameters),
                        dict(fitted_parameters),
                    )
                    break
                inner_x = float(fixed)
                inner_parameters = dict(fitted_parameters or inner_parameters)
            if bracket is None:
                # The last discovery step refits at the bound itself.
                return None, ProfileLikelihoodEndpoint("bound", "search_limit_reached")
            left_x, right_x, right_rise, left_parameters, right_parameters = bracket
            rise_tolerance = 1e-4 * threshold
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
                midpoint_rise = float(record["rise"])
                if midpoint_rise >= threshold - rise_tolerance and not record["converged"]:
                    return None, ProfileLikelihoodEndpoint("unavailable", "refit_not_converged")
                width = abs(right_x - left_x)
                if abs(midpoint_rise - threshold) <= rise_tolerance and width <= coordinate_tolerance:
                    return midpoint, threshold_resolution(fitted_parameters)
                if abs(right_rise - threshold) <= rise_tolerance and width <= coordinate_tolerance:
                    return right_x, threshold_resolution(right_parameters)
                if midpoint_rise >= threshold:
                    right_x, right_rise = midpoint, midpoint_rise
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
            cutoff=threshold,
            degrees_of_freedom=cutoff.degrees_of_freedom,
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
        confidence_level=options.confidence_level,
        likelihood_model=_LIKELIHOOD_MODEL,
        intervals=intervals,
        unavailable_reason=unavailable_reason,
        provenance={
            "primary_fit_preserved": True,
            "settlement": result_status,
        },
    ))

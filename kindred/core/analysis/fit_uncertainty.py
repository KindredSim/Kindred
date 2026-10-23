"""Named, provenance-complete uncertainty methods for completed Fits.

The functions in this module do not evaluate models or own an optimizer.  They
drive the canonical Fit objective, evaluators and optimizer supplied by the
global fitting composition boundary, then return immutable results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from kindred.core.algebra.errors import UNCOMPUTABLE_ERRORS
from kindred.core.algebra.simulation_series import observable_parameter_roots
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
from kindred.core.intervention_schedule import InterventionTrigger, intervention_schedule_parameter_dependencies
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.core.simulator.step_indexing import iter_canonical_parameters

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
    "rate_scale_undetermined",
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
        # One pair at a time: an unavailable column ends the model before
        # the next column runs.
        upper, lower = objective.evaluate_scientific_residuals([forward, backward])
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
                [residuals] = objective.evaluate_scientific_residuals([trial])
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


class _SpeedUnavailable(Exception):
    """A rate factor that cannot be calculated at a point of a fitted curve."""


def rate_scale_undetermined(
    *,
    prepared_dsl: object,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[object],
    fixed_dataset_params: Mapping[str, Mapping[str, float]],
    layout: _fit_execution.FitParameterLayout,
    vector: np.ndarray,
    dataset_info: Sequence[_fit_execution.DatasetFitInfo],
) -> bool:
    """Whether the data leave the overall rate scale undetermined.

    With nothing tied to the clock (no temperature schedule, and no
    intervention other than a state trigger without a minimum interval), a
    curve plotted against a measured X is the same when every step's speed is
    multiplied by one factor; only the time to reach each point changes.  A
    step's speed is its rate constant, times its rate factor at each
    point of the dataset's fitted curve when it has one.  The
    scale is undetermined when the fitted values can multiply one measured-X
    dataset's speeds by a common factor without changing a time-X dataset's
    speeds, any value a target reads, a trigger's inputs or a starting
    concentration.
    A value a target's formula names that cannot be calculated at the fitted
    values lies in an ifelse branch no target took, and is not read.
    The test is linear, on the relative changes of the speeds and of those
    values when each fitted coordinate (log10 under Log Scale) moves by 1e-5
    of itself (of its range at zero) within its bounds; a value of zero stays
    zero.  A value that cannot be calculated at a moved point leaves the
    scale unflagged.
    """

    schedule = (prepared_dsl.parameterized_intervention_schedule_for_binding()
                or prepared_dsl.intervention_execution_contract().authored)
    spec = prepared_dsl.parameter_algebra_spec_for_execution()
    mechanism = prepared_dsl.accepted_mechanism_for_structure()
    rates, factors = [], {}
    for name, entry, role in iter_canonical_parameters(mechanism):
        if role in {"k", "kf", "kr"}:
            rates.append(name)
            step = (mechanism.reactions[entry["reaction_index"]] if entry["kind"] == "reaction"
                    else mechanism.equilibria[entry["equilibrium_index"]])
            if step.factor is not None:
                factors[name] = step.factor
    if (
        all(item.x_name == "t" for item in datasets)
        or prepared_dsl.temperature_changes()
        or any(
            not isinstance(item, InterventionTrigger) or item.min_interval
            for item in schedule.without_stops().operations
        )
    ):
        return False
    # A factored step's speed is a row per point of the fitted curve.
    factored = tuple(name for name in rates if name in factors)
    rates = tuple(name for name in rates if name not in factors)
    lets = tuple(dict.fromkeys(factors[name] for name in factored))
    curves = {info.dataset_id: info.simulation_result for info in dataset_info}
    # A trigger's threshold and amounts shape the curve as a target's reads do.
    triggers = {
        name for names in intervention_schedule_parameter_dependencies(schedule.without_stops()).values()
        for name in names if name in spec.evaluation_dependencies
    }
    x = np.asarray(vector, dtype=float)
    # Relative to each value, so a small rate constant keeps positive
    # neighbours. Within a step of a bound, the difference is a second-order
    # one-sided one into the interior.
    steps = 1e-5 * np.where(x != 0.0, np.abs(x), layout.upper - layout.lower)
    central = (x - steps >= layout.lower) & (x + steps <= layout.upper)
    steps = np.where(central | (layout.upper - x >= x - layout.lower), steps, -steps)
    blocks, scales = [], []
    for payload in datasets:
        roots = {
            root for target in (*payload.species_list, payload.x_name)
            if target in prepared_dsl.analysis.observable_names
            for root in observable_parameter_roots(spec.observable_block, target)
        }
        # The curve starts from the starting concentrations, which a common
        # factor leaves unchanged.
        starting = {f"[{name}]_0" for name in prepared_dsl.analysis.species_names}
        reads = tuple(sorted({root for root in roots if root in spec.evaluation_dependencies} | triggers | starting))

        evaluator = fit_evaluators[payload.dataset_id]

        def values(names: tuple[str, ...], index: int | None = None, offset: float = 0.0, *,
                   speeds: bool = True) -> np.ndarray:
            trial = x.copy()
            if index is not None:
                trial[index] += offset
            params = _fit_execution._objective_dataset_input(
                payload, layout=layout, vector=trial,
                shared_values=layout.shared_param_dict_from_vector(trial),
                fixed_values=fixed_dataset_params[payload.dataset_id],
            ).full_params
            calculated = evaluator.parameter_values(params, names)
            row = np.asarray([calculated[name] for name in names], dtype=float)
            if not (speeds and factored):
                return row
            constants = evaluator.parameter_values(params, factored)
            curve = evaluator.factor_values(params, curves[payload.dataset_id], lets)
            speeds = np.concatenate([constants[name] * curve[factors[name]] for name in factored])
            if not np.all(np.isfinite(speeds)):
                # A factor that cannot be calculated at a point of the curve.
                raise _SpeedUnavailable()
            return np.concatenate([row, speeds])

        def calculable(name: str) -> bool:
            try:
                values((name,), speeds=False)
            except UNCOMPUTABLE_ERRORS:
                return False
            return True

        # A formula names the values of every ifelse branch; one that cannot
        # be calculated at the fitted values shaped no point the Fit compared.
        names = rates + tuple(name for name in reads if calculable(name))
        try:
            base = values(names)
            block = np.empty((base.size, x.size))
            for index, step in enumerate(steps):
                block[:, index] = (
                    (values(names, index, step) - values(names, index, -step)) / (2.0 * step) if central[index]
                    else (4.0 * values(names, index, step) - 3.0 * base - values(names, index, 2.0 * step)) / (2.0 * step)
                )
        except (*UNCOMPUTABLE_ERRORS, _SpeedUnavailable):
            return False
        # Each change relative to its own value, so rows in any unit compare;
        # a value of zero keeps its change and must stay zero.
        block /= np.where(base != 0.0, np.abs(base), 1.0)[:, None]
        # A common factor moves each speed by its sign, as its row is
        # relative to its magnitude; a speed of zero stays zero.
        rows = np.arange(base.size)
        direction = np.where((rows < len(rates)) | (rows >= len(names)), np.sign(base), 0.0)
        blocks.append(block)
        # A time-X dataset, or one whose target reads time, sees the scale.
        scales.append(None if payload.x_name == "t" or "t" in roots else direction)
    free = [index for index, scale in enumerate(scales) if scale is not None]
    if not free:
        return False
    common = np.zeros((sum(block.shape[0] for block in blocks), len(free)))
    row = 0
    for index, block in enumerate(blocks):
        if scales[index] is not None:
            common[row:row + block.shape[0], free.index(index)] = scales[index]
        row += block.shape[0]
    fitted = np.vstack(blocks)

    def rank(matrix: np.ndarray) -> int:
        norms = np.linalg.norm(matrix, axis=0)
        scaled = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)
        singular = np.linalg.svd(scaled, compute_uv=False)
        return int(np.sum(singular > 1e-7 * singular[0])) if singular.size and singular[0] > 0.0 else 0

    # The fitted values reach a common factor when their columns and the
    # per-dataset factor columns share a direction.
    return rank(np.hstack([fitted, common])) < rank(fitted) + rank(common)


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
        # result and a diagnostic for a failed one.
        object.__setattr__(self, "method", _PROFILE_METHOD)
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class _SideEnd(Exception):
    """Ends one Profile side from inside SciPy's root finder."""

    def __init__(self, value: float | None, resolution: ProfileLikelihoodEndpoint) -> None:
        super().__init__()
        self.endpoint = (value, resolution)


def profile_likelihood_settlement_result(
    *,
    config: ProfileLikelihoodConfig,
    status: Literal["stopped", "failed"],
    cause: FactualCause | None,
) -> ProfileLikelihoodResult:
    """Construct a stopped/failed Profile result; only a failure has a cause."""

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
    objective: _fit_execution.GlobalFitObjective,
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
    planned_sides = 2 * len(config.parameter_names)
    completed_sides = 0
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
                completed_sides=completed_sides,
                planned_sides=planned_sides,
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
            nonlocal active_side, active_discovery, active_refinement
            active_side = side
            if bound == best:
                # The estimate sits on this side's bound, so the search range
                # leaves nothing to profile on this side.
                return None, ProfileLikelihoodEndpoint("bound", "search_limit_reached")
            inner_x = best
            inner_parameters = dict(initial)
            # The optimum is the first inner refit.
            inner_record: Mapping[str, object] = {"status": "success", "rise": 0.0, "converged": True}
            bracket: tuple[
                float, float, Mapping[str, object], Mapping[str, object], Mapping[str, float], Mapping[str, float]
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
                nonlocal active_discovery, active_refinement
                if cancellation_check():
                    raise FittingCancelled()
                fit = fit_parameters_with_fixed(
                    objective.evaluate_points,
                    dict(warm_start),
                    fixed_parameters={name: float(fixed)},
                    bounds=bound_map,
                    method=options.optimizer_method,
                    max_nfev=options.max_nfev_per_point,
                    ftol=ftol,
                    xtol=xtol,
                    cancellation_check=cancellation_check,
                )
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
                [evaluation_residuals] = objective.evaluate_scientific_residuals([
                    np.asarray(
                        [fit.parameters[item] for item in initial],
                        dtype=float,
                    ),
                ])
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
                        inner_record,
                        record,
                        dict(inner_parameters),
                        dict(fitted_parameters),
                    )
                    break
                inner_x = float(fixed)
                inner_record = record
                inner_parameters = dict(fitted_parameters or inner_parameters)
            if bracket is None:
                # The last discovery step refits at the bound itself.
                return None, ProfileLikelihoodEndpoint("bound", "search_limit_reached")
            left_x, right_x, left_record, right_record, left_parameters, right_parameters = bracket
            rise_tolerance = 1e-4 * threshold
            # SciPy's root finder evaluates both bracket ends first; they are
            # the refits discovery already made.
            known = {
                left_x: (left_record, left_parameters),
                right_x: (right_record, right_parameters),
            }
            refinement_step = 0
            warm_start = left_parameters

            def signed_rise(fixed: float) -> float:
                """√rise − √cut-off at one refit, nearly linear in the
                coordinate on a quadratic profile.  A refit that ends the side
                raises its endpoint."""

                nonlocal refinement_step, warm_start
                fixed = float(fixed)
                if fixed in known:
                    record, fitted_parameters = known.pop(fixed)
                else:
                    refinement_step += 1
                    record, fitted_parameters = append_fit(
                        fixed,
                        stage="refinement",
                        discovery_step=None,
                        refinement_step=refinement_step,
                        # The root finder keeps a sign change, so its latest
                        # inner refit is the bracket's inner end.
                        warm_start=warm_start,
                    )
                if record["status"] != "success":
                    raise _SideEnd(None, ProfileLikelihoodEndpoint("failed", "scientific_unavailable"))
                rise = float(record["rise"])
                if rise >= threshold - rise_tolerance and not record["converged"]:
                    raise _SideEnd(None, ProfileLikelihoodEndpoint("unavailable", "refit_not_converged"))
                if abs(rise - threshold) <= rise_tolerance:
                    raise _SideEnd(fixed, threshold_resolution(fitted_parameters))
                if rise < threshold:
                    warm_start = dict(fitted_parameters)
                return math.sqrt(max(rise, 0.0)) - math.sqrt(threshold)

            from scipy.optimize import brentq

            try:
                # A crossing within the rise tolerance ends the side; the
                # root finder's own result is never an endpoint.
                brentq(
                    signed_rise, left_x, right_x, xtol=math.ulp(0.0), maxiter=64,
                    full_output=True, disp=False,
                )
            except _SideEnd as end:
                return end.endpoint
            return None, ProfileLikelihoodEndpoint("unavailable", "threshold_not_located")

        side_results: dict[str, tuple[float | None, ProfileLikelihoodEndpoint]] = {}
        for side, bound in (("lower", low), ("upper", high)):
            try:
                side_results[side] = side_profile(side, bound)
                completed_sides += 1
                # The count shows now, not after the next side's first refit.
                emit_progress(stage="side", coordinate=active_coordinate, side=side)
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

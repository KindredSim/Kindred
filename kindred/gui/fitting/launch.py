"""Immutable Fit capture plus strict Fit request construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping, Sequence

from kindred.core.quantity_units import Unit

from kindred.core.analysis.fit_uncertainty import ProfileLikelihoodConfig, ProfileLikelihoodOptions
from kindred.core.analysis.global_fit_execution import (
    FitCoordinate,
    FitParameterLayout,
    build_parameter_layout_for_dataset_ids,
    parameter_group_variable_id,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
    FitDatasetVariableParamSpec,
    FitSharedVariableParamSpec,
)
from kindred.core.analysis.dataset_sampling import compute_sampled_indices
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    MechanismTabDocument,
    ProjectDocument,
    TabScientificSettings,
    canonical_initial_values,
)
from kindred.core.species import validate_concentration, validate_input_time_seconds
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig,
    build_solver_config,
)
from kindred.gui.parameter_enumeration import (
    admit_gui_parameter_value,
    enumerate_parameter_coordinates,
)
from kindred.gui.fitting.contracts import FitParametricSettings
from kindred.gui.fitting.ic_tolerance import FitInitialTolerance

__all__ = [
    "FitParameterConfiguration",
    "FitDatasetLocalSettings",
    "FitDatasetWeightMode",
    "FitSolver",
    "FitExecutionRequest",
    "FitPreparedDatasetConfig",
    "FitSharedInitialGroup",
    "FitInitialVariable",
    "FitMappedDatasetSnapshot",
    "FitObservationSnapshot",
    "FitOriginSnapshot",
    "FitParameterVariable",
    "FitProfileSetup",
    "FitSetupSnapshot",
    "build_fit_execution_request",
    "capture_fit_origin",
    "create_initial_fit_setup",
    "selected_observations",
    "fit_execution_span",
    "fit_dataset_time_defaults",
    "fit_coordinate_datasets_text",
    "fit_group_labels",
    "shared_initial_group_labels",
    "shared_initial_group_variable_id",
    "skipped_fit_analysis_coordinates",
    "fit_parameter_overall",
    "fit_uncertainty_coordinates",
    "fit_profile_config",
    "fit_profile_targets",
    "fit_setup_readiness",
]

FitDatasetWeightMode = Literal["implicit", "user"]
FitSolver = Literal["BDF", "Radau"]


def fit_execution_span(t_start: object, t_end: object) -> tuple[float, float]:
    """Validate and return a strictly positive Fit solver span."""
    if t_start is None or t_end is None:
        raise ValueError("Enter both Simulation start and Simulation end.")
    start = validate_input_time_seconds(t_start, field="Simulation start")
    end = validate_input_time_seconds(t_end, field="Simulation end")
    if end <= start:
        raise ValueError("Simulation end must be greater than Simulation start.")
    return start, end


@dataclass(frozen=True, slots=True)
class FitObservationSnapshot:
    raw_observation_id: str
    raw_column: str
    model_target: str
    unit: Unit | None
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FitMappedDatasetSnapshot:
    """One mapped dataset copied by ``capture_fit_origin`` from the admitted
    document: its identities, canonical initial values, uniquely mapped
    observations and resolved Run start."""

    dataset_id: str
    display_name: str
    time_column_name: str | None
    set_id: str
    set_name: str
    initial_values: tuple[tuple[str, float], ...]
    observations: tuple[FitObservationSnapshot, ...]
    resolved_t_start: float


@dataclass(frozen=True, slots=True)
class FitParameterConfiguration:
    """Complete typed configuration for one scalar scope.

    ``FitParameterVariable`` retains one value for each scope and exposes its
    selected configuration directly.  The carriers are deliberately narrow
    and immutable.  ``groups`` lists the Per-dataset parameter groups, each
    as its origin dataset followed by its other members in sorted order;
    every member stores the group's value in ``dataset_values``.
    ``fixed_dataset_ids`` lists the datasets of a varied Per-dataset
    parameter whose values are used as entered rather than fitted.
    """

    value: float
    minimum: float
    maximum: float
    fit: bool = True
    log10: bool = False
    dataset_values: tuple[tuple[str, float], ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    fixed_dataset_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum >= self.maximum:
            raise ValueError("Parameter requires minimum < maximum.")
        groups = tuple(
            (group[0], *sorted(group[1:])) if group else ()
            for group in self.groups
        )
        object.__setattr__(self, "groups", tuple(sorted(groups)))
        object.__setattr__(self, "fixed_dataset_ids", tuple(sorted(set(self.fixed_dataset_ids))))

    @property
    def varied_dataset_ids(self) -> frozenset[str]:
        """Datasets whose value this configuration fits."""

        if not self.fit:
            return frozenset()
        return frozenset(dataset_id for dataset_id, _value in self.dataset_values) - set(self.fixed_dataset_ids)


def _admit_parameter_groups(
    configuration: FitParameterConfiguration,
    dataset_values: Mapping[str, float],
    *,
    parameter_name: str,
) -> None:
    """Admit disjoint groups of at least two retained, equal-valued, varied members.

    Fixed datasets must be retained, and a varied parameter keeps at least one
    varied dataset; every dataset fixed is the parameter's Vary off.
    """

    fixed = set(configuration.fixed_dataset_ids)
    if fixed and (not configuration.fit or not fixed.issubset(dataset_values)):
        raise ValueError(f"{parameter_name} fixed datasets require Vary and retained datasets.")
    if fixed and fixed == set(dataset_values):
        raise ValueError(f"{parameter_name} with every dataset fixed is Vary off.")
    occupied: set[str] = set()
    for group in configuration.groups:
        if not configuration.fit or fixed.intersection(group):
            raise ValueError(f"{parameter_name} groups require Vary.")
        if len(group) < 2 or len(set(group)) != len(group):
            raise ValueError(f"{parameter_name} groups require at least two unique datasets.")
        if occupied.intersection(group):
            raise ValueError(f"{parameter_name} groups cannot share a dataset.")
        occupied.update(group)
        if any(member not in dataset_values for member in group):
            raise ValueError(f"{parameter_name} group members must be retained datasets.")
        if len({dataset_values[member] for member in group}) != 1:
            raise ValueError(f"{parameter_name} group members must hold one value.")


def _admit_fit_parameter_configuration(
    configuration: FitParameterConfiguration,
    *,
    parameter_name: str,
    role: str,
    unit: str,
    allow_exact_zero: bool,
    dataset_scoped: bool,
) -> FitParameterConfiguration:
    """Admit one complete scope candidate using its retained category."""

    def admit(value: object, field: str) -> float:
        return admit_gui_parameter_value(
            value,
            role=role,
            unit=unit,
            allow_exact_zero=allow_exact_zero,
            field=f"{parameter_name} {field}",
        )

    minimum = admit(configuration.minimum, "minimum")
    maximum = admit(configuration.maximum, "maximum")
    def admit_start(value: float, field: str) -> float:
        value = admit(value, field)
        if not minimum <= value <= maximum:
            raise ValueError(f"{parameter_name} {field} must lie within its bounds.")
        if configuration.log10 and value <= 0.0:
            raise ValueError("Log10 fitting requires positive values.")
        return value

    if configuration.log10 and minimum <= 0.0:
        raise ValueError("Log10 fitting requires positive bounds.")
    value = configuration.value
    dataset_values = configuration.dataset_values
    if dataset_scoped and dataset_values:
        # A group member holds its origin's start (group admission checks
        # that), so only origins and ungrouped datasets are checked and named.
        members = {member for group in configuration.groups for member in group[1:]}
        dataset_values = tuple(
            (dataset_id, item if dataset_id in members
             else admit_start(item, f"Initial guess for dataset {dataset_id}"))
            for dataset_id, item in dataset_values
        )
    else:
        value = admit_start(value, "Initial guess")
    _admit_parameter_groups(
        configuration,
        dict(dataset_values) if dataset_scoped else {},
        parameter_name=parameter_name,
    )
    return replace(
        configuration,
        value=value,
        minimum=minimum,
        maximum=maximum,
        dataset_values=dataset_values,
    )


@dataclass(frozen=True, slots=True)
class FitParameterVariable:
    """One Fit scalar; its values are admitted here, before any commit.

    Identity and category fields come from the accepted GUI coordinate and
    ``scope`` from the Scope choices.
    """

    name: str
    display_name: str
    scope: Literal["shared", "per_dataset"]
    shared_config: FitParameterConfiguration
    per_dataset_config: FitParameterConfiguration
    allow_exact_zero: bool
    family: str = ""
    role: str = ""
    unit: str = "1"

    def __post_init__(self) -> None:
        shared_config = _admit_fit_parameter_configuration(
            self.shared_config,
            parameter_name=self.name,
            role=self.role,
            unit=self.unit,
            allow_exact_zero=self.allow_exact_zero,
            dataset_scoped=False,
        )
        per_dataset_config = _admit_fit_parameter_configuration(
            self.per_dataset_config,
            parameter_name=self.name,
            role=self.role,
            unit=self.unit,
            allow_exact_zero=self.allow_exact_zero,
            dataset_scoped=True,
        )
        object.__setattr__(self, "shared_config", shared_config)
        object.__setattr__(self, "per_dataset_config", per_dataset_config)

    @property
    def selected_config(self) -> FitParameterConfiguration:
        """Return the one configuration selected for execution/presentation."""

        return (
            self.shared_config
            if self.scope == "shared"
            else self.per_dataset_config
        )


def fit_group_labels(
    parameter: FitParameterVariable,
    datasets: Sequence[FitDatasetLocalSettings],
    origin: FitOriginSnapshot,
) -> dict[str, tuple[int, str]]:
    """Number and label each group of a Per-dataset parameter, by origin dataset.

    Groups are numbered by the table position of the dataset each was created
    from, so joining or excluding never renumbers them.  The label names the
    included members in table order, or every member when none is included.
    This is the only owner of a group's number and name.
    """

    return _numbered_group_labels(
        ((group[0], group) for group in parameter.per_dataset_config.groups), datasets, origin,
    )


def _numbered_group_labels(
    groups: Iterable[tuple[str, Iterable[str]]],
    datasets: Sequence[FitDatasetLocalSettings],
    origin: FitOriginSnapshot,
) -> dict[str, tuple[int, str]]:
    """Number (origin dataset, members) groups and label them "Group n (members)".

    Groups are numbered by the table position of their origin dataset.  The
    label names the included members in table order, or every member when
    none is included.
    """

    order = tuple(item.dataset_id for item in datasets)
    included = {item.dataset_id for item in datasets if item.included}
    names = {item.dataset_id: item.display_name for item in origin.mapped_datasets}
    labels: dict[str, tuple[int, str]] = {}
    for number, (origin_dataset_id, members) in enumerate(
        sorted(groups, key=lambda item: order.index(item[0])), start=1,
    ):
        ordered = sorted(members, key=order.index)
        shown = [item for item in ordered if item in included] or ordered
        labels[origin_dataset_id] = (number, f"Group {number} ({', '.join(names[item] for item in shown)})")
    return labels


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean kept within the range of the values."""

    mean = math.fsum(value / len(values) for value in values)
    # A mean never leaves the reduced values; rounding must not either.
    return min(max(mean, min(values)), max(values))


def fit_parameter_overall(
    parameter: FitParameterVariable,
    datasets: Sequence[FitDatasetLocalSettings],
    origin: FitOriginSnapshot,
) -> tuple[tuple[tuple[str, float], ...], float] | None:
    """Return one Per-dataset parameter's fitted values in table order and their mean.

    Only included, varied datasets count: each group with an included member
    once, under its group label, and each other such dataset under its name.
    Every entry is (label, value).  The mean is
    arithmetic.  This is the only place an overall of a Per-dataset parameter
    is computed.
    """

    config = parameter.per_dataset_config
    values = dict(config.dataset_values)
    varied = config.varied_dataset_ids
    names = {item.dataset_id: item.display_name for item in origin.mapped_datasets}
    labels = fit_group_labels(parameter, datasets, origin)
    included = tuple(item.dataset_id for item in datasets if item.included and item.dataset_id in varied)
    group_of = {member: group for group in config.groups for member in group}
    entries: list[tuple[str, float]] = []
    for dataset_id in included:
        group = group_of.get(dataset_id)
        if group is None:
            entries.append((names[dataset_id], values[dataset_id]))
        elif dataset_id == next(item for item in included if item in group):
            entries.append((labels[group[0]][1], values[group[0]]))
    if not entries:
        return None
    return tuple(entries), _mean(tuple(value for _label, value in entries))


def convert_fit_parameter_scope(
    parameter: FitParameterVariable,
    scope: str,
    datasets: Sequence[FitDatasetLocalSettings],
    origin: FitOriginSnapshot,
) -> FitParameterVariable:
    """Convert accepted local values together with their active configuration.

    Switching to Shared seeds the value with the parameter's overall mean, over
    every varied dataset when none of them is included and over every value when
    none is varied; it dissolves the groups and clears the fixed datasets.
    Switching to Per-dataset gives every dataset the Shared value and Vary.
    """
    active = parameter.selected_config
    per_dataset = parameter.per_dataset_config
    if scope == "shared" and parameter.scope != "shared":
        overall = fit_parameter_overall(parameter, datasets, origin) or fit_parameter_overall(
            parameter, tuple(replace(item, included=True) for item in datasets), origin,
        )
        values = tuple(value for _dataset_id, value in active.dataset_values)
        value = overall[1] if overall is not None else _mean(values) if values else active.value
        active = replace(active, value=value, dataset_values=(), groups=(), fixed_dataset_ids=())
        per_dataset = replace(per_dataset, groups=(), fixed_dataset_ids=())
    elif scope != "shared" and parameter.scope == "shared":
        active = replace(
            active,
            dataset_values=tuple((item.dataset_id, active.value) for item in datasets),
        )
    configs = {"shared_config": parameter.shared_config, "per_dataset_config": per_dataset}
    configs["shared_config" if scope == "shared" else "per_dataset_config"] = active
    return replace(parameter, scope=scope, **configs)


def _replace_fit_parameter(
    parameter: FitParameterVariable,
    **changes: object,
) -> FitParameterVariable:
    """Replace fields of the selected scope configuration.

    The inactive scope configuration is retained byte-for-byte.  All
    production replacements of a scalar use this helper, so the replacement
    is admitted again by ``FitParameterVariable``.
    """

    selected = "shared_config" if parameter.scope == "shared" else "per_dataset_config"
    return replace(parameter, **{selected: replace(parameter.selected_config, **changes)})


def _complete_fit_parameter_scopes(
    parameter: FitParameterVariable,
    dataset_ids: Iterable[str],
) -> FitParameterVariable:
    """Seed the inactive Per-dataset config for a captured parameter."""

    retained_ids = tuple(dataset_ids)
    existing = dict(parameter.per_dataset_config.dataset_values)
    dataset_values = tuple(
        (dataset_id, existing.get(dataset_id, parameter.selected_config.value))
        for dataset_id in retained_ids
    )
    if parameter.scope == "per_dataset":
        return _replace_fit_parameter(parameter, dataset_values=dataset_values)
    per_dataset = FitParameterConfiguration(
        value=parameter.selected_config.value,
        minimum=parameter.selected_config.minimum,
        maximum=parameter.selected_config.maximum,
        fit=parameter.selected_config.fit,
        log10=parameter.selected_config.log10,
        dataset_values=dataset_values,
    )
    return replace(parameter, per_dataset_config=per_dataset)


@dataclass(frozen=True, slots=True)
class FitInitialVariable:
    set_id: str
    species: str
    initial: float
    tolerance: FitInitialTolerance = FitInitialTolerance()
    log10: bool = False

    def __post_init__(self) -> None:
        initial = validate_concentration(self.initial, field=f"{self.species} initial")
        minimum, _maximum = self.tolerance.bounds_for(initial)
        if self.log10 and not self.tolerance.enabled:
            raise ValueError("Log10 IC fitting requires an enabled tolerance.")
        if self.log10 and (minimum <= 0.0 or initial <= 0.0):
            raise ValueError(f"{self.species} log10 fitting requires positive values.")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "log10", bool(self.log10))

    @property
    def fit(self) -> bool:
        return self.tolerance.enabled

    @property
    def minimum(self) -> float:
        return self.tolerance.bounds_for(self.initial)[0]

    @property
    def maximum(self) -> float:
        return self.tolerance.bounds_for(self.initial)[1]


@dataclass(frozen=True, slots=True)
class FitSharedInitialGroup:
    """One immutable shared-initial fact authored by the Fit setup."""

    species: str
    origin_dataset_id: str
    member_dataset_ids: tuple[str, ...]
    initial: float
    tolerance: FitInitialTolerance = FitInitialTolerance()
    log10: bool = False

    def __post_init__(self) -> None:
        species = self.species
        origin = self.origin_dataset_id
        members = self.member_dataset_ids
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError("Shared initial groups require at least two unique datasets.")
        if origin not in members:
            raise ValueError("Shared initial group origin must be a member dataset.")
        initial = validate_concentration(self.initial, field=f"{species} initial")
        minimum, _maximum = self.tolerance.bounds_for(initial)
        if self.log10 and (not self.tolerance.enabled or minimum <= 0.0 or initial <= 0.0):
            raise ValueError(f"{species} log10 fitting requires positive varying bounds.")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "origin_dataset_id", origin)
        object.__setattr__(self, "member_dataset_ids", (origin, *tuple(item for item in sorted(members) if item != origin)))
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "log10", bool(self.log10))

    @property
    def fit(self) -> bool:
        return self.tolerance.enabled

    @property
    def minimum(self) -> float:
        return self.tolerance.bounds_for(self.initial)[0]

    @property
    def maximum(self) -> float:
        return self.tolerance.bounds_for(self.initial)[1]


def shared_initial_group_labels(
    groups: Iterable[FitSharedInitialGroup],
    species: str,
    datasets: Sequence[FitDatasetLocalSettings],
    origin: FitOriginSnapshot,
) -> dict[str, tuple[int, str]]:
    """Number and label one species' shared initial groups by origin dataset, Group 1 first.

    Groups are numbered by the table position of their origin dataset.  The
    label names the included members in table order, or every member when none
    is included, as fit_group_labels does for parameter groups.  This is the
    only owner of a shared initial group's number and name.
    """

    return _numbered_group_labels(
        ((group.origin_dataset_id, group.member_dataset_ids) for group in groups if group.species == species),
        datasets, origin,
    )


def shared_initial_group_variable_id(species: str, origin_dataset_id: str) -> str:
    """Stable coupled-variable ID of one fitted shared initial group."""

    return f"initial:{species}:{origin_dataset_id}"


@dataclass(frozen=True, slots=True)
class FitDatasetLocalSettings:
    dataset_id: str
    included: bool
    t_start: float | None
    simulation_end: float | None = None
    data_range_min: float = 0.0
    data_range_max: float = 0.0
    maximum_observations_per_target: int | None = None
    weight: float = 1.0
    target_species: tuple[str, ...] = ()
    target_weights: tuple[tuple[str, float], ...] = ()
    observation_targets: tuple[tuple[str, str], ...] = ()
    parametric: FitParametricSettings | None = None

    def __post_init__(self) -> None:
        if self.parametric is None and self.data_range_max < self.data_range_min:
            raise ValueError("Data Range maximum must be >= minimum.")
        maximum = (
            None
            if self.maximum_observations_per_target in (None, 0)
            else int(self.maximum_observations_per_target)
        )
        if maximum is not None and maximum < 2:
            raise ValueError(
                "Maximum observations per target must be All or at least 2."
            )
        object.__setattr__(self, "maximum_observations_per_target", maximum)


@dataclass(frozen=True, slots=True)
class FitOriginSnapshot:
    tab_id: str
    tab_name: str
    mechanism_source: MechanismAuthoringSource
    prepared_dsl: PreparedDslRequestSet
    scientific_settings: TabScientificSettings
    fit_defaults: FitDefaults
    mapped_datasets: tuple[FitMappedDatasetSnapshot, ...]
    parameter_defaults: tuple[FitParameterVariable, ...]
    def __post_init__(self) -> None:
        object.__setattr__(self, "mapped_datasets", tuple(self.mapped_datasets))
        object.__setattr__(self, "parameter_defaults", tuple(self.parameter_defaults))


@dataclass(frozen=True, slots=True)
class FitProfileSetup:
    """Profile likelihood in the Fit setup: on or off, the fitted values the
    user unticked, and its options.  It profiles every other fitted value, and
    keeps the unticked ones while it is off."""

    enabled: bool = False
    unticked: frozenset[FitCoordinate] = frozenset()
    options: ProfileLikelihoodOptions = ProfileLikelihoodOptions()


def fit_profile_targets(
    profile: FitProfileSetup,
    parameters: Iterable[FitParameterVariable],
    datasets: Iterable[FitDatasetLocalSettings],
) -> tuple[FitCoordinate, ...]:
    """The ticked fitted values, in roster order, excluded datasets' included."""

    return tuple(
        coordinate for coordinate in fit_uncertainty_coordinates(parameters, datasets)
        if coordinate not in profile.unticked
    )


def fit_profile_config(
    profile: FitProfileSetup,
    parameters: Iterable[FitParameterVariable],
    datasets: Iterable[FitDatasetLocalSettings],
) -> ProfileLikelihoodConfig | None:
    """The Profile run of a setup, or None when Profile is off or no ticked
    value has an included dataset."""

    if not profile.enabled:
        return None
    parameters = tuple(parameters)
    datasets = tuple(datasets)
    targets = fit_profile_targets(profile, parameters, datasets)
    skipped = skipped_fit_analysis_coordinates(
        targets,
        excluded_dataset_ids=(item.dataset_id for item in datasets if not item.included),
        parameters=parameters,
    )
    analyzed = tuple(coordinate for coordinate in targets if coordinate not in skipped)
    return ProfileLikelihoodConfig(analyzed, profile.options) if analyzed else None


def fit_uncertainty_coordinates(
    parameters: Iterable[FitParameterVariable],
    datasets: Iterable[FitDatasetLocalSettings],
) -> tuple[FitCoordinate, ...]:
    """Return the exact ordinary typed uncertainty roster.

    A Per-dataset parameter lists one target per group, at its first member,
    and one per ungrouped varied dataset, in import order.  Excluded datasets'
    coordinates are listed too; a fit analyses only the included ones.
    """
    parameter_items = tuple(parameters)
    dataset_ids = tuple(item.dataset_id for item in datasets)
    coordinates: list[FitCoordinate] = []
    for parameter in sorted(parameter_items, key=lambda item: item.name):
        if not parameter.selected_config.fit:
            continue
        if parameter.scope == "shared":
            coordinates.append(FitCoordinate("shared", parameter.name))
            continue
        varied = parameter.selected_config.varied_dataset_ids
        group_of = {
            member: group for group in parameter.selected_config.groups for member in group
        }
        for dataset_id in dataset_ids:
            group = group_of.get(dataset_id)
            if group is None and dataset_id in varied:
                coordinates.append(FitCoordinate("dataset", parameter.name, dataset_id))
            elif group is not None and dataset_id == next(
                item for item in dataset_ids if item in group
            ):
                coordinates.append(FitCoordinate("group", parameter.name, group[0]))
    return tuple(coordinates)


def fit_coordinate_datasets_text(
    coordinate: FitCoordinate,
    setup: FitSetupSnapshot,
) -> str | None:
    """Name the datasets of one uncertainty target, or None for a Shared one."""

    if coordinate.scope == "dataset":
        return next(
            item.display_name for item in setup.origin.mapped_datasets
            if item.dataset_id == coordinate.dataset_id
        )
    if coordinate.scope != "group":
        return None
    parameter = next(item for item in setup.parameters if item.name == coordinate.name)
    return fit_group_labels(parameter, setup.datasets, setup.origin)[coordinate.dataset_id][1]


def skipped_fit_analysis_coordinates(
    coordinates: Iterable[FitCoordinate],
    *,
    excluded_dataset_ids: Iterable[str],
    parameters: Iterable[FitParameterVariable],
) -> frozenset[FitCoordinate]:
    """The uncertainty targets a fit skips because their datasets are excluded.

    A Per-dataset target is skipped when its dataset is excluded, and a group
    target only when all its members are excluded.  This is the one rule for
    which selected targets a fit analyses.
    """

    excluded_ids = set(excluded_dataset_ids)
    if not excluded_ids:
        return frozenset()
    members = {
        FitCoordinate("group", parameter.name, group[0]): group
        for parameter in parameters
        for group in parameter.selected_config.groups
    }
    return frozenset(
        coordinate for coordinate in coordinates
        if coordinate.scope != "shared"
        and excluded_ids.issuperset(
            members[coordinate] if coordinate.scope == "group" else (coordinate.dataset_id,)
        )
    )


@dataclass(frozen=True, slots=True)
class FitPreparedDatasetConfig:
    """One admitted dataset/grid solver policy completed before dispatch.

    Its sole producer selects the observations and retains their sorted
    distinct admitted times.
    """

    dataset_id: str
    selected_observations: tuple[FitObservationSnapshot, ...]
    parametric_x_observation: FitObservationSnapshot | None
    retained_times: tuple[float, ...]
    execution_t_start: float
    execution_t_end: float
    solver_config: PreparedSolverConfig


@dataclass(frozen=True, slots=True)
class _FitDispatchCompilation:
    """Private immutable lowering retained only by an admitted Fit request."""

    included_dataset_ids: tuple[str, ...]
    prepared_datasets: tuple[FitPreparedDatasetConfig, ...]
    ordinary_shared_variables: tuple[FitSharedVariableParamSpec, ...]
    ordinary_coupled_variables: tuple[FitCoupledVariableParamSpec, ...]
    ordinary_dataset_overrides: tuple[FitDatasetParameterOverrides, ...]
    candidate_names: tuple[str, ...]
    ordinary_optimizer_layout: FitParameterLayout


@dataclass(frozen=True, slots=True)
class FitExecutionRequest:
    setup: FitSetupSnapshot
    dispatch: _FitDispatchCompilation


@dataclass(frozen=True, slots=True)
class FitSetupSnapshot:
    """One complete typed Fit setup at launch or terminal completion."""

    origin: FitOriginSnapshot
    solver: FitSolver
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
    defaults: FitDefaults
    dataset_weight_mode: FitDatasetWeightMode
    shared_initial_groups: tuple[FitSharedInitialGroup, ...] = ()
    profile: FitProfileSetup = FitProfileSetup()
    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        parameters = tuple(self.parameters)
        initials = tuple(self.initials)
        shared_initial_groups = tuple(self.shared_initial_groups)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initials", initials)
        object.__setattr__(self, "shared_initial_groups", tuple(sorted(shared_initial_groups, key=lambda item: (item.species, item.origin_dataset_id))))
        if not self.profile.unticked <= set(fit_uncertainty_coordinates(parameters, datasets)):
            raise ValueError("Profile setup references an unavailable fitted coordinate.")


def _scan_parameters(tab: MechanismTabDocument) -> tuple[FitParameterVariable, ...]:
    if not tab.mechanism_source.reactions_text.strip():
        return ()
    variables: list[FitParameterVariable] = []
    coordinates = enumerate_parameter_coordinates(tab.prepared_dsl)
    # Vary starts off; the user chooses what is fitted.
    for coordinate in coordinates:
        value = coordinate.value
        low, high = coordinate.fit_minimum, coordinate.fit_maximum
        variables.append(
            FitParameterVariable(
                name=coordinate.name,
                display_name=coordinate.display_name,
                scope="shared",
                shared_config=FitParameterConfiguration(
                    value=value,
                    minimum=low,
                    maximum=high,
                    fit=False,
                    log10=False,
                ),
                per_dataset_config=FitParameterConfiguration(
                    value=value,
                    minimum=low,
                    maximum=high,
                    fit=False,
                    log10=False,
                ),
                family=coordinate.family,
                role=coordinate.role,
                allow_exact_zero=coordinate.allow_exact_zero,
                unit=coordinate.unit,
            )
        )
    return tuple(variables)


def fit_dataset_time_defaults(
    mapped: FitMappedDatasetSnapshot,
) -> tuple[float | None, float | None, float, float]:
    """Return the initial simulation span and observation range for a dataset."""

    times = [value for observation in mapped.observations for value in observation.t]
    return (
        mapped.resolved_t_start if times else None,
        max(times, default=None),
        min(times, default=mapped.resolved_t_start),
        max(times, default=mapped.resolved_t_start),
    )


def create_initial_fit_setup(origin: FitOriginSnapshot) -> FitSetupSnapshot:
    """Build the sole typed Fit-local setup without consulting widget state."""

    datasets = []
    for mapped in origin.mapped_datasets:
        start, end, low, high = fit_dataset_time_defaults(mapped)
        datasets.append(FitDatasetLocalSettings(
            dataset_id=mapped.dataset_id,
            included=False,
            t_start=start,
            simulation_end=end,
            data_range_min=low,
            data_range_max=high,
            maximum_observations_per_target=None,
            weight=1.0,
            target_species=tuple(
                observation.model_target
                for observation in mapped.observations
            ),
            target_weights=tuple(
                (observation.model_target, 1.0)
                for observation in mapped.observations
            ),
            observation_targets=tuple(
                (
                    observation.raw_observation_id,
                    observation.model_target,
                )
                for observation in mapped.observations
            ),
        ))
    values_by_key = {
        (mapped.set_id, species): value
        for mapped in origin.mapped_datasets
        for species, value in mapped.initial_values
    }
    initials = tuple(
        FitInitialVariable(
            set_id=set_id,
            species=species,
            initial=value,
            tolerance=FitInitialTolerance(),
            log10=False,
        )
        for (set_id, species), value in values_by_key.items()
    )
    retained_dataset_ids = tuple(item.dataset_id for item in origin.mapped_datasets)
    parameters = tuple(
        _complete_fit_parameter_scopes(parameter, retained_dataset_ids)
        for parameter in origin.parameter_defaults
    )
    return FitSetupSnapshot(
        origin=origin,
        solver="BDF",
        datasets=tuple(datasets),
        parameters=parameters,
        initials=initials,
        defaults=origin.fit_defaults,
        dataset_weight_mode="implicit",
        shared_initial_groups=(),
    )


def capture_fit_origin(
    document: ProjectDocument,
    tab_id: str,
) -> FitOriginSnapshot:
    """Copy mapped dataset science from one exact tab."""
    tab = document.tab(tab_id)
    if tab.prepared_dsl is None:
        raise ValueError("Enter a valid mechanism before opening Global Fit.")
    mapped: list[FitMappedDatasetSnapshot] = []
    links = dict(tab.dataset_links)
    # Fit lists datasets in the project's import order, as the main window does.
    for dataset in document.datasets:
        link = links.get(dataset.dataset_id)
        if link is None or link.set_id is None:
            continue
        initial_set = tab.initial_condition_set(link.set_id)
        by_raw_id = {
            observation.raw_observation_id: observation
            for observation in dataset.observations
        }
        observations: list[FitObservationSnapshot] = []
        for raw_observation_id, model_target in link.observation_mapping:
            observation = by_raw_id[raw_observation_id]
            observations.append(
                FitObservationSnapshot(
                    raw_observation_id=observation.raw_observation_id,
                    raw_column=observation.column_name,
                    model_target=model_target,
                    unit=observation.unit,
                    t=observation.t,
                    y=observation.y,
                    source_rows=observation.source_rows,
                )
            )
        mapped.append(
            FitMappedDatasetSnapshot(
                dataset_id=dataset.dataset_id,
                display_name=dataset.display_name,
                time_column_name=dataset.time_column_name,
                set_id=initial_set.set_id,
                set_name=initial_set.name,
                initial_values=canonical_initial_values(tab, initial_set.set_id),
                observations=tuple(observations),
                resolved_t_start=tab.resolved_execution_settings(
                    initial_set.set_id
                ).t_start,
            )
        )
    return FitOriginSnapshot(
        tab_id=tab.tab_id,
        tab_name=tab.name,
        mechanism_source=tab.mechanism_source,
        prepared_dsl=tab.prepared_dsl,
        scientific_settings=tab.scientific_settings,
        fit_defaults=document.fit_defaults,
        mapped_datasets=tuple(mapped),
        parameter_defaults=_scan_parameters(tab),
    )


def selected_observations(
    mapped: FitMappedDatasetSnapshot,
    local: FitDatasetLocalSettings,
) -> tuple[tuple[FitObservationSnapshot, ...], FitObservationSnapshot | None]:
    """Select Y observations and their shared measured-X row capture."""
    target_by_raw_id = {
        raw_id: model_target
        for raw_id, model_target in local.observation_targets
    }
    wanted = set(local.target_species)
    observations_by_target: dict[str, FitObservationSnapshot] = {}
    for observation in mapped.observations:
        target = target_by_raw_id.get(observation.raw_observation_id)
        if target is None or target not in wanted:
            continue
        # The admitted mapping gives each target exactly one column.
        observations_by_target[target] = observation

    x_rows: set[int] | None = None
    x_observation: FitObservationSnapshot | None = None
    if local.parametric is not None:
        # The admitted mapping gives measured X exactly one column, never a Y column.
        x_raw_id = next(
            raw_id
            for raw_id, current_target in local.observation_targets
            if current_target == local.parametric.x_name
        )
        x_observation = next(
            observation
            for observation in mapped.observations
            if observation.raw_observation_id == x_raw_id
        )
        x_rows = set(x_observation.source_rows)

    selected: list[FitObservationSnapshot] = []
    for target in local.target_species:
        observation = observations_by_target[target]
        if x_rows is None and not observation.t:
            raise ValueError("No recorded time. Select a measured X quantity.")
        usable = tuple(
            index for index, row in enumerate(observation.source_rows)
            if (row in x_rows if x_rows is not None else (
                local.data_range_min <= observation.t[index] <= local.data_range_max
            ))
        )
        if not usable:
            selection = "Parametric-X row pairing" if x_rows is not None else "Data Range"
            raise ValueError(f"Target {target!r} has zero usable observations after {selection}.")
        maximum = local.maximum_observations_per_target
        indices = usable
        if maximum is not None:
            # Sampling selects observations, never supplies a numerical time;
            # time-X and measured-X pick the same rows for the same cap.
            indices = tuple(
                usable[index]
                for index in compute_sampled_indices(count=len(usable), n_points=maximum)
            )
        selected.append(
            FitObservationSnapshot(
                raw_observation_id=observation.raw_observation_id,
                raw_column=observation.raw_column,
                model_target=target,
                unit=observation.unit,
                t=tuple(float(observation.t[index]) for index in indices) if observation.t else (),
                y=tuple(float(observation.y[index]) for index in indices),
                source_rows=tuple(
                    int(observation.source_rows[index]) for index in indices
                ),
            )
        )
    if x_observation is None or not selected:
        return tuple(selected), None
    retained_rows = {
        row for observation in selected for row in observation.source_rows
    }
    indices = tuple(
        index for index, row in enumerate(x_observation.source_rows)
        if row in retained_rows
    )
    return tuple(selected), FitObservationSnapshot(
        raw_observation_id=x_observation.raw_observation_id,
        raw_column=x_observation.raw_column,
        model_target=local.parametric.x_name,
        unit=x_observation.unit,
        t=tuple(float(x_observation.t[index]) for index in indices) if x_observation.t else (),
        y=tuple(float(x_observation.y[index]) for index in indices),
        source_rows=tuple(x_observation.source_rows[index] for index in indices),
    )


def fit_setup_readiness(
    *,
    origin: FitOriginSnapshot,
    datasets: Iterable[FitDatasetLocalSettings],
    parameters: Iterable[FitParameterVariable],
    initials: Iterable[FitInitialVariable],
    shared_initial_groups: Iterable[FitSharedInitialGroup],
    profile: FitProfileSetup,
    check_observations: bool = True,
) -> str | None:
    """Return the shared cheap structural Run Fit readiness truth.

    This deliberately admits no science and does not replace the request
    builder's strict validation.  It only answers whether the captured setup
    has every included dataset producing observations and at least one eligible
    varying parameter or initial condition.
    """

    dataset_items = tuple(datasets)
    parameter_items = tuple(parameters)
    initial_items = tuple(initials)
    shared_initial_group_items = tuple(shared_initial_groups)
    mapped_by_id = {item.dataset_id: item for item in origin.mapped_datasets}
    included = tuple(item for item in dataset_items if item.included)
    if not included:
        return "Select at least one mapped dataset for fitting."
    for local in included:
        mapped = mapped_by_id[local.dataset_id]
        if not local.target_species:
            return f"Dataset {mapped.display_name!r} must select at least one target."
        if check_observations:
            try:
                observations, _ = selected_observations(mapped, local)
            except ValueError as exc:
                return f"Dataset {mapped.display_name!r}: {exc}"
            if not observations:
                return f"Dataset {mapped.display_name!r} has no selected observations."

    included_ids = {item.dataset_id for item in included}
    has_fitted_initial = any(
        item.fit
        and not any(
            item.species == group.species and item.set_id in {
                mapped.set_id
                for mapped in origin.mapped_datasets
                if mapped.dataset_id in group.member_dataset_ids
            }
            for group in shared_initial_group_items
        )
        and any(
            mapped.set_id == item.set_id and mapped.dataset_id in included_ids
            for mapped in origin.mapped_datasets
        )
        for item in initial_items
    ) or any(
        group.fit
        and any(
            mapped.dataset_id in included_ids and mapped.dataset_id in group.member_dataset_ids
            and any(species == group.species
                for species, _value in mapped.initial_values
            )
            for mapped in origin.mapped_datasets
        )
        for group in shared_initial_group_items
    )
    has_fitted_parameter = any(
        item.selected_config.fit
        if item.scope == "shared"
        else bool(item.selected_config.varied_dataset_ids & included_ids)
        for item in parameter_items
    )
    if not has_fitted_parameter and not has_fitted_initial:
        return "Select at least one parameter or initial concentration to fit."
    if profile.enabled and fit_profile_config(profile, parameter_items, dataset_items) is None:
        return "Profile Likelihood has no ticked value with an included dataset."
    return None


def _prepare_fit_datasets(
    *,
    setup: FitSetupSnapshot,
) -> tuple[FitPreparedDatasetConfig, ...]:
    """Perform deferred inclusion science and retain each selection once.

    The setup holds one local settings value per captured dataset, its
    readiness gate found an included dataset, and its captured initial values
    and observation targets come from the admitted document.
    """

    mapped_by_id = {item.dataset_id: item for item in setup.origin.mapped_datasets}
    included = tuple(item for item in setup.datasets if item.included)
    science = setup.origin.scientific_settings
    prepared: list[FitPreparedDatasetConfig] = []
    for local in included:
        mapped = mapped_by_id[local.dataset_id]
        selected, x_observation = selected_observations(mapped, local)
        retained_times = (() if local.parametric is not None else tuple(
            sorted({float(value) for item in selected for value in item.t})
        ))
        execution_t_start, execution_t_end = fit_execution_span(
            local.t_start, local.simulation_end
        )
        if any(
            value < execution_t_start or value > execution_t_end
            for value in retained_times
        ):
            raise ValueError(
                f"Dataset {mapped.display_name!r} has selected experimental times "
                "outside the Simulation interval. Adjust Simulation start, Simulation end, or Data Range."
            )
        prepared.append(
            FitPreparedDatasetConfig(
                dataset_id=local.dataset_id,
                selected_observations=selected,
                parametric_x_observation=x_observation,
                retained_times=retained_times,
                execution_t_start=execution_t_start,
                execution_t_end=execution_t_end,
                solver_config=build_solver_config(
                    solver=setup.solver,
                    rtol=science.rtol,
                    atol=science.atol,
                    grid={"N": 100},
                    use_analytic_jacobian=science.analytic_jacobian_policy,
                ),
            )
        )
    return tuple(prepared)


def _compile_fit_dispatch(
    *,
    setup: FitSetupSnapshot,
    prepared_datasets: tuple[FitPreparedDatasetConfig, ...],
) -> _FitDispatchCompilation:
    """Complete all static Fit facts before a worker can be constructed."""

    included = tuple(item for item in setup.datasets if item.included)
    included_ids = tuple(item.dataset_id for item in included)

    shared_variables: list[FitSharedVariableParamSpec] = []
    coupled_variables: list[FitCoupledVariableParamSpec] = []
    variable_by_dataset: dict[str, list[FitDatasetVariableParamSpec]] = {
        dataset_id: [] for dataset_id in included_ids
    }
    fixed_by_dataset: dict[str, dict[str, float]] = {
        dataset_id: {} for dataset_id in included_ids
    }
    for parameter in setup.parameters:
        config = parameter.selected_config
        dataset_values = dict(config.dataset_values)
        if config.fit and parameter.scope == "shared":
            shared_variables.append(
                FitSharedVariableParamSpec(
                    name=parameter.name,
                    initial=config.value,
                    minimum=config.minimum,
                    maximum=config.maximum,
                    log10=config.log10,
                )
            )
        elif config.fit:
            # A group with an included member is one coupled variable over its
            # included members; a group with none contributes nothing.
            grouped = {member for group in config.groups for member in group}
            fixed = set(config.fixed_dataset_ids)
            for group in config.groups:
                members = tuple(item for item in group if item in included_ids)
                if members:
                    coupled_variables.append(
                        FitCoupledVariableParamSpec(
                            variable_id=parameter_group_variable_id(parameter.name, group[0]),
                            parameter_name=parameter.name,
                            dataset_ids=members,
                            initial=dataset_values[group[0]],
                            minimum=config.minimum,
                            maximum=config.maximum,
                            log10=config.log10,
                        )
                    )
            for dataset_id in included_ids:
                if dataset_id in grouped:
                    continue
                if dataset_id in fixed:
                    fixed_by_dataset[dataset_id][parameter.name] = dataset_values[dataset_id]
                    continue
                variable_by_dataset[dataset_id].append(
                    FitDatasetVariableParamSpec(
                        name=parameter.name,
                        initial=dataset_values[dataset_id],
                        minimum=config.minimum,
                        maximum=config.maximum,
                        log10=config.log10,
                    )
                )
        elif parameter.scope == "per_dataset":
            # The setup retains one value for every captured dataset.
            for dataset_id in included_ids:
                fixed_by_dataset[dataset_id][parameter.name] = dataset_values[dataset_id]
        else:
            for dataset_id in included_ids:
                fixed_by_dataset[dataset_id][parameter.name] = config.value

    datasets_for_set: dict[str, tuple[str, ...]] = {}
    for mapped in setup.origin.mapped_datasets:
        if mapped.dataset_id not in included_ids:
            continue
        datasets_for_set[mapped.set_id] = (
            *datasets_for_set.get(mapped.set_id, ()),
            mapped.dataset_id,
        )
    grouped_dataset_species: set[tuple[str, str]] = set()
    for group in setup.shared_initial_groups:
        applicable_ids = tuple(
            dataset_id
            for dataset_id in group.member_dataset_ids
            if dataset_id in included_ids
        )
        for dataset_id in applicable_ids:
            grouped_dataset_species.add((dataset_id, group.species))
        if group.fit and applicable_ids:
            coupled_variables.append(
                FitCoupledVariableParamSpec(
                    variable_id=shared_initial_group_variable_id(group.species, group.origin_dataset_id),
                    parameter_name=f"init:{group.species}",
                    dataset_ids=applicable_ids,
                    initial=group.initial,
                    minimum=group.minimum,
                    maximum=group.maximum,
                    log10=group.log10,
                )
            )
        else:
            for dataset_id in applicable_ids:
                fixed_by_dataset[dataset_id][f"init:{group.species}"] = group.initial
    for initial in setup.initials:
        target_ids = datasets_for_set.get(initial.set_id, ())
        target_ids = tuple(
            dataset_id
            for dataset_id in target_ids
            if (dataset_id, initial.species) not in grouped_dataset_species
        )
        if initial.fit and target_ids:
            coupled_variables.append(
                FitCoupledVariableParamSpec(
                    variable_id=f"initial:{initial.set_id}:{initial.species}",
                    parameter_name=f"init:{initial.species}",
                    dataset_ids=target_ids,
                    initial=initial.initial,
                    minimum=initial.minimum,
                    maximum=initial.maximum,
                    log10=initial.log10,
                )
            )
        else:
            for dataset_id in target_ids:
                fixed_by_dataset[dataset_id][f"init:{initial.species}"] = initial.initial

    overrides = tuple(
        FitDatasetParameterOverrides(
            dataset_id=dataset_id,
            fixed_params=tuple(sorted(fixed_by_dataset[dataset_id].items())),
            variable_params=tuple(variable_by_dataset[dataset_id]),
        )
        for dataset_id in included_ids
    )
    layout = build_parameter_layout_for_dataset_ids(
        dataset_ids=included_ids,
        shared_params={item.name: item.initial for item in shared_variables},
        dataset_overrides=overrides,
        coupled_variables=tuple(coupled_variables),
        bounds={
            item.name: (item.minimum, item.maximum)
            for item in shared_variables
        },
        log10_params={item.name: item.log10 for item in shared_variables},
    )
    return _FitDispatchCompilation(
        included_dataset_ids=included_ids,
        prepared_datasets=tuple(prepared_datasets),
        ordinary_shared_variables=tuple(shared_variables),
        ordinary_coupled_variables=tuple(coupled_variables),
        ordinary_dataset_overrides=overrides,
        candidate_names=tuple(sorted(parameter.name for parameter in setup.parameters)),
        ordinary_optimizer_layout=layout,
    )


def build_fit_execution_request(
    *,
    setup: FitSetupSnapshot,
) -> FitExecutionRequest:
    """Compile one exact accepted setup into its immutable execution request."""
    readiness = fit_setup_readiness(
        origin=setup.origin,
        datasets=setup.datasets,
        parameters=setup.parameters,
        initials=setup.initials,
        shared_initial_groups=setup.shared_initial_groups,
        profile=setup.profile,
        check_observations=False,
    )
    if readiness is not None:
        raise ValueError(readiness)
    prepared_datasets = _prepare_fit_datasets(
        setup=setup,
    )
    dispatch = _compile_fit_dispatch(
        setup=setup,
        prepared_datasets=prepared_datasets,
    )
    return FitExecutionRequest(
        setup=setup,
        dispatch=dispatch,
    )

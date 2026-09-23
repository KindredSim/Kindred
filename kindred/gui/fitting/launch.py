"""Immutable Fit capture plus strict Fit request construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

import numpy as np
from kindred.core.quantity_units import Unit

from kindred.core.analysis.fit_uncertainty import (
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
)
from kindred.core.analysis.global_fit_execution import FitCoordinate
from kindred.core.analysis.global_fit_execution import (
    FitParameterLayout,
    build_parameter_layout_for_dataset_ids,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FIT_SCOPE_CENTRED_FIXED_EFFECTS,
    FitCoupledVariableParamSpec,
    FitCentredFixedEffectsParamSpec,
    FitDatasetParameterOverrides,
    FitDatasetVariableParamSpec,
    FitSharedVariableParamSpec,
)
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.core.analysis.dataset_sampling import compute_sampled_indices
from kindred.core.dataset_mapping import (
    resolve_fit_t_start,
)
from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.project_document import (
    FitDefaults,
    MechanismTabDocument,
    ProjectDocument,
    TabScientificSettings,
    canonical_initial_values,
    validate_fit_solver,
)
from kindred.core.species import validate_concentration, validate_input_time_seconds
from kindred.core.time_grid import build_time_grid
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.core.simulation_preparation_stages import (
    PreparedSolverConfig,
    build_solver_config,
)
from kindred.gui.parameter_enumeration import (
    GuiParameterScientificCategory,
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
    "FitSelectedObservation",
    "FitSetupSnapshot",
    "build_fit_execution_request",
    "capture_fit_origin",
    "create_initial_fit_setup",
    "selected_observations",
    "fit_execution_span",
    "fit_uncertainty_coordinates",
    "prune_fit_analysis_coordinates",
    "fit_setup_readiness",
]

FitDatasetWeightMode = Literal["implicit", "user"]
FitSolver = Literal["BDF", "Radau"]


def _required(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def _finite(value: object, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


def fit_execution_span(t_start: object, t_end: object) -> tuple[float, float]:
    """Validate and return a strictly positive Fit solver span."""
    if t_start is None or t_end is None:
        raise ValueError("Enter both Simulation start and Simulation end.")
    start = validate_input_time_seconds(t_start, field="Fit execution start")
    end = validate_input_time_seconds(t_end, field="Fit execution end")
    if end <= start:
        raise ValueError("Fit execution end must be greater than its start.")
    return start, end


def _validated_dataset_weight_mode(
    value: object,
) -> FitDatasetWeightMode:
    if value == "implicit":
        return "implicit"
    if value == "user":
        return "user"
    raise ValueError("Dataset weight mode must be implicit or user.")


@dataclass(frozen=True, slots=True)
class FitObservationSnapshot:
    raw_observation_id: str
    raw_column: str
    model_target: str
    unit: Unit
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_observation_id",
            _required(self.raw_observation_id, "raw observation ID"),
        )
        object.__setattr__(self, "raw_column", _required(self.raw_column, "raw column"))
        object.__setattr__(
            self,
            "model_target",
            _required(self.model_target, "mapped model target"),
        )


@dataclass(frozen=True, slots=True)
class FitMappedDatasetSnapshot:
    dataset_id: str
    display_name: str
    set_id: str
    set_name: str
    initial_values: tuple[tuple[str, float], ...]
    observations: tuple[FitObservationSnapshot, ...]
    resolved_t_start: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _required(self.dataset_id, "dataset ID"))
        object.__setattr__(self, "display_name", _required(self.display_name, "dataset name"))
        object.__setattr__(self, "set_id", _required(self.set_id, "set ID"))
        object.__setattr__(self, "set_name", _required(self.set_name, "set name"))
        names: set[str] = set()
        values: list[tuple[str, float]] = []
        for raw_name, raw_value in self.initial_values:
            name = _required(raw_name, "initial species")
            if name in names:
                raise ValueError(f"Duplicate initial species {name!r}.")
            names.add(name)
            value = validate_concentration(
                _finite(raw_value, f"initial value {name}"),
                field=f"initial value {name}",
            )
            values.append((name, value))
        object.__setattr__(self, "initial_values", tuple(values))
        observations = tuple(self.observations)
        if any(
            not isinstance(item, FitObservationSnapshot)
            for item in observations
        ):
            raise TypeError(
                "Mapped Fit observations must be FitObservationSnapshot values."
            )
        raw_ids = [item.raw_observation_id for item in observations]
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(
                "Mapped Fit dataset-local raw observation IDs must be unique."
            )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(
            self,
            "resolved_t_start",
            _finite(self.resolved_t_start, "resolved mapped start"),
        )


@dataclass(frozen=True, slots=True)
class FitParameterConfiguration:
    """Complete typed configuration for one scalar scope.

    ``FitParameterVariable`` retains one value for each scope and exposes its
    selected configuration directly.  The carriers are deliberately narrow
    and immutable.
    """

    value: float
    minimum: float
    maximum: float
    fit: bool = True
    log10: bool = False
    dataset_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        value = _finite(self.value, "parameter value")
        minimum = _finite(self.minimum, "parameter minimum")
        maximum = _finite(self.maximum, "parameter maximum")
        if minimum >= maximum:
            raise ValueError("Parameter requires minimum < maximum.")
        values = tuple(
            (_required(dataset_id, "dataset ID"), _finite(item, "dataset value"))
            for dataset_id, item in self.dataset_values
        )
        if len({dataset_id for dataset_id, _value in values}) != len(values):
            raise ValueError("Parameter dataset values repeat a dataset ID.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "fit", bool(self.fit))
        object.__setattr__(self, "log10", bool(self.log10))
        object.__setattr__(self, "dataset_values", values)


def _admit_fit_parameter_configuration(
    configuration: FitParameterConfiguration,
    *,
    parameter_name: str,
    role: str,
    unit: str,
    scientific_category: GuiParameterScientificCategory,
    allow_exact_zero: bool,
    dataset_scoped: bool,
) -> FitParameterConfiguration:
    """Admit one complete scope candidate using its retained category."""

    if not isinstance(configuration, FitParameterConfiguration):
        raise TypeError("Fit parameter configuration must be typed.")

    def admit(value: object, field: str) -> float:
        return admit_gui_parameter_value(
            value,
            scientific_category=scientific_category,
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
        dataset_values = tuple(
            (dataset_id, admit_start(item, f"value for dataset {dataset_id}"))
            for dataset_id, item in dataset_values
        )
    else:
        value = admit_start(value, "value")
    return replace(
        configuration,
        value=value,
        minimum=minimum,
        maximum=maximum,
        dataset_values=dataset_values,
    )


@dataclass(frozen=True, slots=True)
class FitParameterVariable:
    name: str
    display_name: str
    scope: str
    shared_config: FitParameterConfiguration
    per_dataset_config: FitParameterConfiguration
    scientific_category: GuiParameterScientificCategory
    allow_exact_zero: bool
    family: str = ""
    role: str = ""
    unit: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required(self.name, "parameter name"))
        object.__setattr__(
            self,
            "display_name",
            _required(self.display_name, "parameter display name"),
        )
        scope = str(self.scope or "").strip().lower()
        if scope not in {"shared", "per_dataset", FIT_SCOPE_CENTRED_FIXED_EFFECTS}:
            raise ValueError(
                "Scalar Fit scope must be Shared, Per-dataset, or centred fixed effects."
            )
        scientific_category = str(self.scientific_category or "").strip()
        if scientific_category not in {
            "rate",
            "equilibrium_constant",
            "finite_scalar",
        }:
            raise ValueError("Fit parameter scientific category is invalid.")
        if type(self.allow_exact_zero) is not bool:
            raise TypeError("Fit parameter exact-zero authority must be a bool.")
        shared_config = _admit_fit_parameter_configuration(
            self.shared_config,
            parameter_name=self.name,
            role=self.role,
            unit=self.unit,
            scientific_category=scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            dataset_scoped=False,
        )
        per_dataset_config = _admit_fit_parameter_configuration(
            self.per_dataset_config,
            parameter_name=self.name,
            role=self.role,
            unit=self.unit,
            scientific_category=scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            dataset_scoped=True,
        )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "shared_config", shared_config)
        object.__setattr__(self, "per_dataset_config", per_dataset_config)
        object.__setattr__(self, "family", str(self.family or "").strip())
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "scientific_category", scientific_category)
        object.__setattr__(self, "unit", str(self.unit or "1").strip() or "1")

    @property
    def selected_config(self) -> FitParameterConfiguration:
        """Return the one configuration selected for execution/presentation."""

        return (
            self.shared_config
            if self.scope == "shared"
            else self.per_dataset_config
        )


def convert_fit_parameter_scope(
    parameter: FitParameterVariable,
    scope: str,
    datasets: Sequence[FitDatasetLocalSettings],
    *,
    statistic: str = "arithmetic",
) -> FitParameterVariable:
    """Convert accepted local values together with their active configuration."""
    active = parameter.selected_config
    if scope == "shared" and parameter.scope != "shared":
        roster = tuple(item.dataset_id for item in datasets if item.included) or tuple(
            item.dataset_id for item in datasets
        )
        values = dict(active.dataset_values)
        selected = tuple(values[dataset_id] for dataset_id in roster)
        if not selected:
            value = active.value
        elif statistic == "geometric":
            if any(value <= 0.0 for value in selected):
                raise ValueError("Geometric Shared seed requires positive values.")
            value = math.exp(math.fsum(math.log(value) for value in selected) / len(selected))
        else:
            value = math.fsum(value / len(selected) for value in selected)
        active = replace(active, value=value, dataset_values=())
    elif scope != "shared" and parameter.scope == "shared":
        active = replace(
            active,
            dataset_values=tuple((item.dataset_id, active.value) for item in datasets),
        )
    return replace(
        parameter,
        scope=scope,
        **{"shared_config" if scope == "shared" else "per_dataset_config": active},
    )


def _replace_fit_parameter(
    parameter: FitParameterVariable,
    **changes: object,
) -> FitParameterVariable:
    """Replace fields in the selected scalar configuration.

    The inactive scope configuration is intentionally retained byte-for-byte;
    only the selected configuration is rebuilt from the requested active
    fields.  All production replacements of a scalar use this helper so the
    typed invariant is checked by ``FitParameterConfiguration`` itself.
    """

    if not isinstance(parameter, FitParameterVariable):
        raise TypeError("parameter must be FitParameterVariable.")
    allowed = {
        "initial",
        "minimum",
        "maximum",
        "fit",
        "log10",
        "scope",
        "dataset_values",
        "name",
        "display_name",
        "family",
        "role",
        "unit",
    }
    unknown = set(changes).difference(allowed)
    if unknown:
        raise TypeError("Unsupported Fit parameter replacement fields: " + ", ".join(sorted(unknown)))
    target_scope = str(changes.get("scope", parameter.scope)).strip().lower()
    if target_scope not in {"shared", "per_dataset", FIT_SCOPE_CENTRED_FIXED_EFFECTS}:
        raise ValueError(
            "Scalar Fit scope must be Shared, Per-dataset, or centred fixed effects."
        )
    selected = (
        parameter.shared_config
        if target_scope == "shared"
        else parameter.per_dataset_config
    )
    values = {
        "value": selected.value,
        "minimum": selected.minimum,
        "maximum": selected.maximum,
        "fit": selected.fit,
        "log10": selected.log10,
        "dataset_values": selected.dataset_values,
    }
    values.update(
        {
            ("value" if key == "initial" else key): value
            for key, value in changes.items()
            if key in {"initial", "minimum", "maximum", "fit", "log10", "dataset_values"}
        }
    )
    active = FitParameterConfiguration(**values)
    configs = {
        "shared_config": parameter.shared_config,
        "per_dataset_config": parameter.per_dataset_config,
    }
    configs[
        "shared_config" if target_scope == "shared" else "per_dataset_config"
    ] = active
    presentation = {
        key: value
        for key, value in changes.items()
        if key in {"name", "display_name", "family", "role", "unit"}
    }
    return replace(
        parameter,
        scope=target_scope,
        shared_config=configs["shared_config"],
        per_dataset_config=configs["per_dataset_config"],
        **presentation,
    )


def _complete_fit_parameter_scopes(
    parameter: FitParameterVariable,
    dataset_ids: Iterable[str],
) -> FitParameterVariable:
    """Seed the inactive Per-dataset config for a captured parameter."""

    retained_ids = tuple(_required(dataset_id, "dataset ID") for dataset_id in dataset_ids)
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
        object.__setattr__(self, "set_id", _required(self.set_id, "set ID"))
        object.__setattr__(self, "species", _required(self.species, "species"))
        initial = validate_concentration(
            _finite(self.initial, f"{self.species} initial"),
            field=f"{self.species} initial",
        )
        if not isinstance(self.tolerance, FitInitialTolerance):
            raise TypeError("Initial tolerance must be FitInitialTolerance.")
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
        species = _required(self.species, "initial species")
        origin = _required(self.origin_dataset_id, "initial group origin dataset ID")
        members = tuple(_required(item, "initial group member dataset ID") for item in self.member_dataset_ids)
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError("Shared initial groups require at least two unique datasets.")
        if origin not in members:
            raise ValueError("Shared initial group origin must be a member dataset.")
        initial = validate_concentration(
            _finite(self.initial, f"{species} initial"),
            field=f"{species} initial",
        )
        if not isinstance(self.tolerance, FitInitialTolerance):
            raise TypeError("Shared initial group tolerance must be FitInitialTolerance.")
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
        object.__setattr__(self, "dataset_id", _required(self.dataset_id, "dataset ID"))
        t_start = None if self.t_start is None else _finite(self.t_start, "Fit simulation start")
        if t_start is not None and t_start < 0.0:
            raise ValueError("Fit simulation start must be nonnegative.")
        simulation_end = None if self.simulation_end is None else _finite(self.simulation_end, "Simulation end")
        object.__setattr__(self, "simulation_end", simulation_end)
        low = _finite(self.data_range_min, "Data Range minimum")
        high = _finite(self.data_range_max, "Data Range maximum")
        if self.parametric is None and high < low:
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
        weight = _finite(self.weight, "dataset weight")
        if weight <= 0.0:
            raise ValueError("Dataset weight must be positive.")
        targets = tuple(_required(item, "target series") for item in self.target_species)
        if len(targets) != len(set(targets)):
            raise ValueError("Fit target series must be unique.")
        target_weights = tuple(
            (_required(name, "target series"), _finite(value, "target weight"))
            for name, value in self.target_weights
        )
        weight_roster = targets
        if self.parametric is not None:
            weight_roster = (*targets, self.parametric.x_name)
        weight_names = [name for name, _value in target_weights]
        if (
            len(weight_names) != len(set(weight_names))
            or set(weight_names) != set(weight_roster)
        ):
            raise ValueError(
                "Fit target weights must exactly match selected targets."
            )
        if any(value <= 0.0 for _name, value in target_weights):
            raise ValueError("Target weights must be positive.")
        mapping = tuple(
            (
                _required(raw_id, "raw observation ID"),
                _required(model, "model target"),
            )
            for raw_id, model in self.observation_targets
        )
        if len({raw_id for raw_id, _model in mapping}) != len(mapping):
            raise ValueError(
                "Observation target overrides must have unique dataset-local "
                "raw IDs."
            )
        mapped_targets = [model for _raw_id, model in mapping]
        if (
            len(mapped_targets) != len(set(mapped_targets))
            or not set(targets).issubset(set(mapped_targets))
        ):
            raise ValueError(
                "Observation target overrides must retain one unique mapping "
                "for every selected Fit target."
            )
        object.__setattr__(self, "t_start", t_start)
        object.__setattr__(self, "data_range_min", low)
        object.__setattr__(self, "data_range_max", high)
        object.__setattr__(self, "maximum_observations_per_target", maximum)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "target_species", targets)
        object.__setattr__(self, "target_weights", target_weights)
        object.__setattr__(self, "observation_targets", mapping)
        if self.parametric is not None and self.parametric.dataset_id != self.dataset_id:
            raise ValueError("Parametric settings changed stable dataset identity.")
        if self.parametric is not None and self.parametric.x_name in targets:
            raise ValueError(
                "The selected parametric X series cannot also be a Y target."
            )


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
        object.__setattr__(self, "tab_id", _required(self.tab_id, "tab ID"))
        object.__setattr__(self, "tab_name", _required(self.tab_name, "tab name"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be MechanismAuthoringSource.")
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
        if not source_matches_analysis(
            self.mechanism_source,
            self.prepared_dsl.analysis,
        ):
            raise ValueError(
                "Fit origin prepared DSL must be the retained analysis for "
                "the captured source."
            )
        if not isinstance(self.scientific_settings, TabScientificSettings):
            raise TypeError("scientific_settings must be TabScientificSettings.")
        if not isinstance(self.fit_defaults, FitDefaults):
            raise TypeError("fit_defaults must be FitDefaults.")
        mapped_datasets = tuple(self.mapped_datasets)
        parameter_defaults = tuple(self.parameter_defaults)
        if any(
            not isinstance(item, FitMappedDatasetSnapshot)
            for item in mapped_datasets
        ):
            raise TypeError(
                "Fit origin datasets must be FitMappedDatasetSnapshot values."
            )
        if any(
            not isinstance(item, FitParameterVariable)
            for item in parameter_defaults
        ):
            raise TypeError(
                "Fit origin parameters must be FitParameterVariable values."
            )
        dataset_ids = [item.dataset_id for item in mapped_datasets]
        set_ids = [item.set_id for item in mapped_datasets]
        parameter_names = [item.name for item in parameter_defaults]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Fit origin dataset IDs must be unique.")
        if len(set_ids) != len(set(set_ids)):
            raise ValueError(
                "Fit origin mapped set IDs must be unique across datasets."
            )
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Fit origin parameter names must be unique.")
        object.__setattr__(self, "mapped_datasets", mapped_datasets)
        object.__setattr__(self, "parameter_defaults", parameter_defaults)


def _validate_advanced_analysis_setup(
    *,
    datasets: tuple[FitDatasetLocalSettings, ...],
    parameters: tuple[FitParameterVariable, ...],
    profile_likelihood: ProfileLikelihoodConfig | None,
    residual_bootstrap: ResidualBootstrapConfig | None,
) -> None:
    if profile_likelihood is not None and not isinstance(
        profile_likelihood,
        ProfileLikelihoodConfig,
    ):
        raise TypeError("Profile likelihood configuration must be typed.")
    if residual_bootstrap is not None and not isinstance(
        residual_bootstrap,
        ResidualBootstrapConfig,
    ):
        raise TypeError("Residual bootstrap configuration must be typed.")
    allowed_coordinates = fit_uncertainty_coordinates(parameters, datasets)
    allowed_set = set(allowed_coordinates)
    allowed_display = {item.display_name for item in allowed_coordinates}
    for config in (profile_likelihood, residual_bootstrap):
        if config is None:
            continue
        if not config.parameter_names:
            raise ValueError(
                "Enabled uncertainty analysis requires at least one coordinate."
            )
        requested = tuple(config.parameter_names)
        resolved: list[FitCoordinate] = []
        for item in requested:
            if isinstance(item, FitCoordinate):
                coordinate = item
            else:
                display = str(item or "").strip()
                if display not in allowed_display:
                    raise ValueError(
                        "Uncertainty configuration references an unavailable "
                        "fitted coordinate."
                    )
                coordinate = next(
                    value for value in allowed_coordinates
                    if value.display_name == display
                )
            if coordinate not in allowed_set:
                raise ValueError(
                    "Uncertainty configuration references an unavailable fitted "
                    "coordinate."
                )
            if coordinate.scope in {"coupled", "centred_deviation"}:
                raise ValueError(
                    "Coupled and centred dataset-deviation coordinates cannot "
                    "be selected as uncertainty targets."
                )
            resolved.append(coordinate)
        if len(resolved) != len(set(resolved)):
            raise ValueError("Uncertainty coordinates repeat a typed identity.")


def fit_uncertainty_coordinates(
    parameters: Iterable[FitParameterVariable],
    datasets: Iterable[FitDatasetLocalSettings],
) -> tuple[FitCoordinate, ...]:
    """Return the exact ordinary typed uncertainty roster.

    A centred parameter exposes only its overall coordinate. Its dataset
    deviations are fitted nuisances and are never public uncertainty targets.
    """
    parameter_items = tuple(parameters)
    included_ids = tuple(sorted(item.dataset_id for item in datasets if item.included))
    coordinates: list[FitCoordinate] = []
    for parameter in sorted(parameter_items, key=lambda item: item.name):
        if not parameter.selected_config.fit:
            continue
        if parameter.scope == "shared":
            coordinates.append(FitCoordinate("shared", parameter.name))
        elif parameter.scope == FIT_SCOPE_CENTRED_FIXED_EFFECTS:
            coordinates.append(FitCoordinate("centred_overall", parameter.name))
        else:
            values = dict(parameter.selected_config.dataset_values)
            coordinates.extend(
                FitCoordinate("dataset", parameter.name, dataset_id)
                for dataset_id in included_ids
                if dataset_id in values
            )
    return tuple(coordinates)


def prune_fit_analysis_coordinates(
    configuration: ProfileLikelihoodConfig | ResidualBootstrapConfig | None,
    changed_parameter_names: Iterable[str],
) -> ProfileLikelihoodConfig | ResidualBootstrapConfig | None:
    """Drop stale uncertainty coordinates for changed Fit parameters.

    Fit-coordinate identities are admitted before this projection.  A scope
    or Vary change invalidates every coordinate for the changed parameter,
    while unrelated coordinates retain their original order and identity.
    """

    if configuration is None:
        return None
    changed_names = set(changed_parameter_names)
    remaining = tuple(
        coordinate
        for coordinate in configuration.parameter_names
        if coordinate.name not in changed_names
    )
    if not remaining:
        return None
    if remaining == configuration.parameter_names:
        return configuration
    return replace(configuration, parameter_names=remaining)


@dataclass(frozen=True, slots=True)
class FitPreparedDatasetConfig:
    """One admitted dataset/grid solver policy completed before dispatch."""

    dataset_id: str
    selected_observations: tuple[FitSelectedObservation, ...]
    parametric_x_observation: FitSelectedObservation | None
    retained_times: tuple[float, ...]
    execution_t_start: float
    execution_t_end: float
    fit_sampling_policy: str
    fit_sampling_identity: tuple[tuple[str, object], ...]
    solver_config: PreparedSolverConfig

    def __post_init__(self) -> None:
        dataset_id = _required(self.dataset_id, "prepared Fit dataset ID")
        selected = tuple(self.selected_observations)
        if not selected or any(
            not isinstance(item, FitSelectedObservation) for item in selected
        ):
            raise TypeError(
                "Prepared Fit dataset requires selected Fit observations."
            )
        if self.parametric_x_observation is not None and not isinstance(
            self.parametric_x_observation,
            FitSelectedObservation,
        ):
            raise TypeError(
                "Prepared Fit dataset Parametric-X observation must be typed."
            )
        retained_times = tuple(float(value) for value in self.retained_times)
        if (
            (not retained_times and self.parametric_x_observation is None)
            or any(not math.isfinite(value) for value in retained_times)
            or any(right <= left for left, right in zip(retained_times, retained_times[1:]))
        ):
            raise ValueError("Prepared Fit dataset times must be finite and ordered.")
        if not isinstance(self.solver_config, PreparedSolverConfig):
            raise TypeError("Prepared Fit dataset requires PreparedSolverConfig.")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "selected_observations", selected)
        object.__setattr__(self, "retained_times", retained_times)


@dataclass(frozen=True, slots=True)
class _FitDispatchCompilation:
    """Private immutable lowering retained only by an admitted Fit request."""

    included_dataset_ids: tuple[str, ...]
    prepared_datasets: tuple[FitPreparedDatasetConfig, ...]
    ordinary_shared_variables: tuple[FitSharedVariableParamSpec, ...]
    ordinary_coupled_variables: tuple[FitCoupledVariableParamSpec, ...]
    fixed_by_dataset: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    ordinary_dataset_overrides: tuple[FitDatasetParameterOverrides, ...]
    candidate_names: tuple[str, ...]
    ordinary_optimizer_layout: FitParameterLayout | None
    ordinary_centred_variables: tuple[FitCentredFixedEffectsParamSpec, ...]


@dataclass(frozen=True, slots=True)
class FitExecutionRequest:
    setup: FitSetupSnapshot
    dispatch: _FitDispatchCompilation

    def __post_init__(self) -> None:
        if not isinstance(self.setup, FitSetupSnapshot):
            raise TypeError("setup must be FitSetupSnapshot.")
        if not isinstance(self.dispatch, _FitDispatchCompilation):
            raise TypeError("Fit request dispatch must be compiled.")


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
    profile_likelihood: ProfileLikelihoodConfig | None = None
    residual_bootstrap: ResidualBootstrapConfig | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.origin, FitOriginSnapshot):
            raise TypeError("snapshot origin must be FitOriginSnapshot.")
        if not isinstance(self.defaults, FitDefaults):
            raise TypeError("snapshot defaults must be FitDefaults.")
        object.__setattr__(self, "solver", validate_fit_solver(self.solver))
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )
        datasets = tuple(self.datasets)
        parameters = tuple(self.parameters)
        initials = tuple(self.initials)
        shared_initial_groups = tuple(self.shared_initial_groups)
        if any(
            not isinstance(item, FitDatasetLocalSettings)
            for item in datasets
        ):
            raise TypeError(
                "Fit snapshot datasets must be FitDatasetLocalSettings values."
            )
        if any(
            not isinstance(item, FitParameterVariable)
            for item in parameters
        ):
            raise TypeError(
                "Fit snapshot parameters must be FitParameterVariable values."
            )
        if any(
            not isinstance(item, FitInitialVariable)
            for item in initials
        ):
            raise TypeError(
                "Fit snapshot initials must be FitInitialVariable values."
            )
        if any(not isinstance(item, FitSharedInitialGroup) for item in shared_initial_groups):
            raise TypeError("Fit snapshot shared initial groups must be typed.")
        dataset_ids = [item.dataset_id for item in datasets]
        parameter_names = [item.name for item in parameters]
        initial_keys = [(item.set_id, item.species) for item in initials]
        group_identities = [(item.species, item.origin_dataset_id) for item in shared_initial_groups]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Fit snapshot dataset identities repeat.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Fit snapshot parameter identities repeat.")
        if len(initial_keys) != len(set(initial_keys)):
            raise ValueError("Fit snapshot initial-condition identities repeat.")
        if len(group_identities) != len(set(group_identities)):
            raise ValueError("Fit snapshot shared initial group identities repeat.")
        retained_by_id = {item.dataset_id: item for item in datasets}
        initial_by_set_species = {(item.set_id, item.species) for item in initials}
        occupied: set[tuple[str, str]] = set()
        for group in shared_initial_groups:
            if group.origin_dataset_id not in retained_by_id:
                raise ValueError("Shared initial group origin is not retained.")
            origin = self.origin.mapped_datasets
            mapped_by_id = {item.dataset_id: item for item in origin}
            for dataset_id in group.member_dataset_ids:
                if dataset_id not in retained_by_id:
                    raise ValueError("Shared initial group member is not retained.")
                mapped = mapped_by_id.get(dataset_id)
                if mapped is None:
                    raise ValueError("Shared initial group member is not captured.")
                if not any(species == group.species for species, _value in mapped.initial_values):
                    raise ValueError("Shared initial group members must contain the species.")
                if (mapped.set_id, group.species) not in initial_by_set_species:
                    raise ValueError(
                        "Shared initial group members require compatible retained initials."
                    )
                key = (dataset_id, group.species)
                if key in occupied:
                    raise ValueError("Shared initial groups cannot overlap a dataset/species row.")
                occupied.add(key)
        retained_dataset_ids = set(dataset_ids)
        incomplete_parameter_maps = sorted(
            item.name
            for item in parameters
            if item.scope in {"per_dataset", FIT_SCOPE_CENTRED_FIXED_EFFECTS}
            and {
                dataset_id
                for dataset_id, _value in item.selected_config.dataset_values
            }
            != retained_dataset_ids
        )
        if incomplete_parameter_maps:
            raise ValueError(
                "Fit snapshot parameters require one retained value for "
                "every captured dataset: "
                + ", ".join(incomplete_parameter_maps)
                + "."
            )
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initials", initials)
        object.__setattr__(self, "shared_initial_groups", tuple(sorted(shared_initial_groups, key=lambda item: (item.species, item.origin_dataset_id))))
        _validate_advanced_analysis_setup(
            datasets=datasets,
            parameters=parameters,
            profile_likelihood=self.profile_likelihood,
            residual_bootstrap=self.residual_bootstrap,
        )

@dataclass(frozen=True, slots=True)
class FitSelectedObservation:
    raw_observation_id: str
    raw_column: str
    model_target: str
    unit: Unit
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]


def _scan_parameters(tab: MechanismTabDocument) -> tuple[FitParameterVariable, ...]:
    if not tab.mechanism_source.reactions_text.strip():
        return ()
    variables: list[FitParameterVariable] = []
    if not isinstance(tab.prepared_dsl, PreparedDslRequestSet):
        raise ValueError("Fit requires an accepted prepared mechanism.")
    coordinates = enumerate_parameter_coordinates(tab.prepared_dsl)
    for coordinate in coordinates:
        name = _required(coordinate.name, "parameter name")
        value = _finite(coordinate.value, f"parameter {name}")
        low = float(coordinate.minimum)
        high = float(coordinate.maximum)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(
                f"Accepted parameter {name!r} has invalid Fit bounds."
            )
        variables.append(
            FitParameterVariable(
                name=name,
                display_name=str(coordinate.display_name or name),
                scope="shared",
                shared_config=FitParameterConfiguration(
                    value=value,
                    minimum=low,
                    maximum=high,
                    fit=True,
                    log10=False,
                ),
                per_dataset_config=FitParameterConfiguration(
                    value=value,
                    minimum=low,
                    maximum=high,
                    fit=True,
                    log10=False,
                ),
                family=str(coordinate.family),
                role=str(coordinate.role),
                scientific_category=coordinate.scientific_category,
                allow_exact_zero=coordinate.allow_exact_zero,
                unit=str(coordinate.unit),
            )
        )
    return tuple(variables)


def create_initial_fit_setup(origin: FitOriginSnapshot) -> FitSetupSnapshot:
    """Build the sole typed Fit-local setup without consulting widget state."""

    if not isinstance(origin, FitOriginSnapshot):
        raise TypeError("origin must be FitOriginSnapshot.")
    datasets = tuple(
        FitDatasetLocalSettings(
            dataset_id=mapped.dataset_id,
            included=False,
            t_start=(
                mapped.resolved_t_start
                if any(observation.t for observation in mapped.observations)
                else None
            ),
            simulation_end=max(
                (value for observation in mapped.observations for value in observation.t),
                default=None,
            ),
            data_range_min=min(
                (
                    value
                    for observation in mapped.observations
                    for value in observation.t
                ),
                default=mapped.resolved_t_start,
            ),
            data_range_max=max(
                (
                    value
                    for observation in mapped.observations
                    for value in observation.t
                ),
                default=mapped.resolved_t_start,
            ),
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
        )
        for mapped in origin.mapped_datasets
    )
    values_by_key: dict[tuple[str, str], float] = {}
    for mapped in origin.mapped_datasets:
        for species, value in mapped.initial_values:
            key = (mapped.set_id, species)
            previous = values_by_key.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Mapped set {mapped.set_id!r} repeats initial {species!r} "
                    "with conflicting values."
                )
            values_by_key[key] = value
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
        datasets=datasets,
        parameters=parameters,
        initials=initials,
        defaults=origin.fit_defaults,
        dataset_weight_mode="implicit",
        shared_initial_groups=(),
    )


def capture_fit_origin(
    document: ProjectDocument,
    tab_id: str,
    *,
    dataset_ids: Iterable[str] | None = None,
) -> FitOriginSnapshot:
    """Copy mapped dataset science from one exact tab."""
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be ProjectDocument.")
    tab = document.tab(tab_id)
    if not isinstance(tab.prepared_dsl, PreparedDslRequestSet):
        raise ValueError("Fit requires an accepted visible mechanism.")
    selected_dataset_ids = (
        None
        if dataset_ids is None
        else frozenset(
            str(dataset_id)
            for dataset_id in dataset_ids
            if str(dataset_id).strip()
        )
    )
    mapped: list[FitMappedDatasetSnapshot] = []
    for dataset_id, link in tab.dataset_links:
        if link.set_id is None:
            continue
        if (
            selected_dataset_ids is not None
            and dataset_id not in selected_dataset_ids
        ):
            continue
        dataset = document.dataset(dataset_id)
        initial_set = tab.initial_condition_set(link.set_id)
        by_raw_id = {
            observation.raw_observation_id: observation
            for observation in dataset.observations
        }
        observations: list[FitObservationSnapshot] = []
        for raw_observation_id, model_target in link.observation_mapping:
            observation = by_raw_id.get(raw_observation_id)
            if observation is None:
                raise ValueError(
                    f"Dataset {dataset.display_name!r} mapping references "
                    f"missing raw observation ID {raw_observation_id!r}."
                )
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
                set_id=initial_set.set_id,
                set_name=initial_set.name,
                initial_values=canonical_initial_values(tab, initial_set.set_id),
                observations=tuple(observations),
                resolved_t_start=resolve_fit_t_start(
                    tab,
                    initial_set.set_id,
                    label=f"Fit snapshot for dataset {dataset.dataset_id!r}",
                ),
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
) -> tuple[FitSelectedObservation, ...]:
    """Materialize the retained observation projection for one dataset."""
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
        if target in observations_by_target:
            raise ValueError(
                f"Dataset {mapped.display_name!r} maps multiple selected "
                f"columns to target {target!r}."
            )
        observations_by_target[target] = observation
    missing = wanted - set(observations_by_target)
    if missing:
        raise ValueError(
            f"Dataset {mapped.display_name!r} has no selected observation "
            f"rows for: {', '.join(sorted(missing))}."
        )

    x_rows: set[int] | None = None
    if local.parametric is not None:
        x_raw_ids = tuple(
            raw_id
            for raw_id, current_target in local.observation_targets
            if current_target == local.parametric.x_name
        )
        if len(x_raw_ids) != 1:
            raise ValueError(
                f"Dataset {mapped.display_name!r} Parametric-X series "
                f"{local.parametric.x_name!r} must map to exactly one "
                "observation."
            )
        active_y_raw_ids = {
            observation.raw_observation_id
            for observation in observations_by_target.values()
        }
        x_raw_id = x_raw_ids[0]
        if x_raw_id in active_y_raw_ids:
            raise ValueError(
                f"Dataset {mapped.display_name!r} Parametric-X series "
                f"{local.parametric.x_name!r} cannot reuse a selected Y "
                "observation column."
            )
        x_observation = next(
            (
                observation
                for observation in mapped.observations
                if observation.raw_observation_id == x_raw_id
            ),
            None,
        )
        if x_observation is None:
            raise ValueError(
                f"Dataset {mapped.display_name!r} Parametric-X mapping "
                f"references missing raw observation {x_raw_id!r}."
            )
        x_rows = set(x_observation.source_rows)

    selected: list[FitSelectedObservation] = []
    for target in local.target_species:
        observation = observations_by_target[target]
        if x_rows is None and not observation.t:
            raise ValueError(
                f"Dataset {mapped.display_name!r} has no recorded time. Select a measured X quantity."
            )
        usable = tuple(
            index for index, row in enumerate(observation.source_rows)
            if (row in x_rows if x_rows is not None else (
                local.data_range_min <= observation.t[index] <= local.data_range_max
            ))
        )
        if not usable:
            selection = "Parametric-X physical-row pairing" if x_rows is not None else "Data Range"
            raise ValueError(
                f"Dataset {mapped.display_name!r} target {target!r} has "
                f"zero usable observations after {selection}."
            )
        maximum = local.maximum_observations_per_target
        indices = usable
        if maximum is not None:
            # Sampling selects observations, never supplies a numerical time.
            # A/Y selection is independent of any recorded time column.
            indices = tuple(usable[index] for index in np.unique(
                np.linspace(0, len(usable) - 1, min(maximum, len(usable)), dtype=int)
            )) if x_rows is not None else tuple(
                usable[index] for index in compute_sampled_indices(
                    t=np.asarray([observation.t[index] for index in usable]),
                    t_min=-math.inf, t_max=math.inf, n_points=maximum,
                )
            )
        selected.append(
            FitSelectedObservation(
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
    return tuple(selected)


def fit_setup_readiness(
    *,
    origin: FitOriginSnapshot,
    datasets: Iterable[FitDatasetLocalSettings],
    parameters: Iterable[FitParameterVariable],
    initials: Iterable[FitInitialVariable],
    shared_initial_groups: Iterable[FitSharedInitialGroup] = (),
    check_observations: bool = True,
) -> str | None:
    """Return the shared cheap structural Run Fit readiness truth.

    This deliberately admits no science and does not replace the request
    builder's strict validation.  It only answers whether the captured setup
    has every included dataset producing observations and at least one eligible
    varying parameter or initial condition.
    """

    if not isinstance(origin, FitOriginSnapshot):
        raise TypeError("Fit readiness requires FitOriginSnapshot.")
    dataset_items = tuple(datasets)
    parameter_items = tuple(parameters)
    initial_items = tuple(initials)
    shared_initial_group_items = tuple(shared_initial_groups)
    mapped_by_id = {item.dataset_id: item for item in origin.mapped_datasets}
    included = tuple(item for item in dataset_items if item.included)
    if not included:
        return "Select at least one mapped dataset for fitting."
    for local in included:
        mapped = mapped_by_id.get(local.dataset_id)
        if mapped is None:
            return f"Included dataset {local.dataset_id!r} is not captured."
        if not local.target_species:
            return f"Dataset {mapped.display_name!r} must select at least one target."
        if check_observations:
            try:
                observations = selected_observations(mapped, local)
            except (KeyError, TypeError, ValueError) as exc:
                return str(exc)
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
    if not any(item.selected_config.fit for item in parameter_items) and not has_fitted_initial:
        return "Select at least one parameter or initial concentration to fit."
    return None


def _selected_parametric_x_observation(
    mapped: FitMappedDatasetSnapshot,
    local: FitDatasetLocalSettings,
    selected: tuple[FitSelectedObservation, ...],
) -> FitSelectedObservation | None:
    """Materialize the one row-aligned Parametric-X capture with the Y rows."""

    parametric = local.parametric
    if parametric is None:
        return None
    x_raw_ids = tuple(
        raw_id
        for raw_id, target in local.observation_targets
        if target == parametric.x_name
    )
    if len(x_raw_ids) != 1:
        raise ValueError(
            f"Dataset {mapped.display_name!r} Parametric-X series "
            f"{parametric.x_name!r} must map to exactly one observation."
        )
    x_source = next(
        (
            observation
            for observation in mapped.observations
            if observation.raw_observation_id == x_raw_ids[0]
        ),
        None,
    )
    if x_source is None:
        raise ValueError(
            f"Dataset {mapped.display_name!r} Parametric-X mapping references "
            f"missing raw observation {x_raw_ids[0]!r}."
        )
    retained_rows = {
        int(row)
        for observation in selected
        for row in observation.source_rows
    }
    indices = tuple(
        index
        for index, row in enumerate(x_source.source_rows)
        if int(row) in retained_rows
    )
    if not indices:
        raise ValueError(
            f"Dataset {mapped.display_name!r} Parametric-X series "
            f"{parametric.x_name!r} has no physically paired observations."
        )
    return FitSelectedObservation(
        raw_observation_id=x_source.raw_observation_id,
        raw_column=x_source.raw_column,
        model_target=parametric.x_name,
        unit=x_source.unit,
        t=tuple(float(x_source.t[index]) for index in indices) if x_source.t else (),
        y=tuple(float(x_source.y[index]) for index in indices),
        source_rows=tuple(int(x_source.source_rows[index]) for index in indices),
    )


def _prepare_fit_datasets(
    *,
    setup: FitSetupSnapshot,
) -> tuple[FitPreparedDatasetConfig, ...]:
    """Perform deferred inclusion science and retain each selection once."""

    mapped_by_id = {item.dataset_id: item for item in setup.origin.mapped_datasets}
    setup_ids = tuple(item.dataset_id for item in setup.datasets)
    if len(setup_ids) != len(set(setup_ids)) or set(setup_ids) != set(mapped_by_id):
        raise ValueError("Fit request must contain each mapped dataset exactly once.")
    mechanism_species = set(setup.origin.prepared_dsl.analysis.species_names)
    allowed_targets = mechanism_species | set(
        setup.origin.prepared_dsl.analysis.observable_names
    )
    included = tuple(item for item in setup.datasets if item.included)
    if not included:
        raise ValueError("Select at least one mapped dataset for fitting.")
    science = setup.origin.scientific_settings
    initials_by_set: dict[str, set[str]] = {}
    for initial in setup.initials:
        initials_by_set.setdefault(initial.set_id, set()).add(initial.species)
    prepared: list[FitPreparedDatasetConfig] = []
    for local in included:
        mapped = mapped_by_id.get(local.dataset_id)
        if mapped is None:
            raise ValueError(
                f"Included dataset {local.dataset_id!r} is not captured."
            )
        if not math.isfinite(mapped.resolved_t_start) or mapped.resolved_t_start < 0.0:
            raise ValueError(
                f"Dataset {mapped.display_name!r} has no finite captured start."
            )
        captured_initials = {species for species, _value in mapped.initial_values}
        if not mechanism_species.issubset(captured_initials):
            raise ValueError(
                f"Dataset {mapped.display_name!r} captured initials do not "
                "cover the mechanism species."
            )
        if initials_by_set.get(mapped.set_id, set()) != captured_initials:
            raise ValueError(
                f"Dataset {mapped.display_name!r} mapped set identity is not "
                "retained by the accepted initial-condition rows."
            )
        captured_by_raw = {
            item.raw_observation_id: item for item in mapped.observations
        }
        for raw_id, model_target in local.observation_targets:
            if raw_id not in captured_by_raw:
                raise ValueError(
                    f"Dataset {mapped.display_name!r} mapping references "
                    f"uncaptured raw observation {raw_id!r}."
                )
            if model_target not in allowed_targets:
                raise ValueError(
                    f"Dataset {mapped.display_name!r} mapped target "
                    f"{model_target!r} is not a mechanism target or admitted "
                    "canonical Algebra target."
                )
        selected = selected_observations(mapped, local)
        x_observation = _selected_parametric_x_observation(mapped, local, selected)
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
                "outside the Simulation interval. Adjust the interval or Data Range."
            )
        grid_points = 100
        policy = (
            "parametric_regular_grid"
            if local.parametric is not None
            else "fit_internal_regular_grid_union_retained_endpoints"
        )
        identity = (
            ("solver", setup.solver),
            ("t_start", execution_t_start),
            ("t_end", execution_t_end),
            ("regular_grid_count", grid_points),
            ("retained_observation_times", retained_times),
            ("union_endpoints", (execution_t_start, execution_t_end)),
        )
        try:
            build_time_grid(execution_t_start, execution_t_end, {"N": grid_points})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Fit regular grid for dataset {local.dataset_id!r} cannot be "
                "realized on this time span."
            ) from exc
        prepared.append(
            FitPreparedDatasetConfig(
                dataset_id=local.dataset_id,
                selected_observations=selected,
                parametric_x_observation=x_observation,
                retained_times=retained_times,
                execution_t_start=execution_t_start,
                execution_t_end=execution_t_end,
                fit_sampling_policy=policy,
                fit_sampling_identity=identity,
                solver_config=build_solver_config(
                    solver_input=setup.solver,
                    rtol=science.rtol,
                    atol=science.atol,
                    grid={"N": grid_points},
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

    mapped_by_id = {item.dataset_id: item for item in setup.origin.mapped_datasets}
    request_ids = tuple(item.dataset_id for item in setup.datasets)
    if len(request_ids) != len(set(request_ids)) or set(request_ids) != set(mapped_by_id):
        raise ValueError("Fit dispatch dataset roster does not match its capture.")
    included = tuple(item for item in setup.datasets if item.included)
    included_ids = tuple(item.dataset_id for item in included)
    if not included_ids:
        raise ValueError("Fit dispatch requires at least one included dataset.")

    shared_variables: list[FitSharedVariableParamSpec] = []
    centred_variables: list[FitCentredFixedEffectsParamSpec] = []
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
        elif config.fit and parameter.scope == FIT_SCOPE_CENTRED_FIXED_EFFECTS:
            centred_variables.append(
                FitCentredFixedEffectsParamSpec(
                    parameter_name=parameter.name,
                    dataset_ids=tuple(included_ids),
                    initial_values=tuple(
                        (dataset_id, dataset_values[dataset_id])
                        for dataset_id in included_ids
                    ),
                    minimum=config.minimum,
                    maximum=config.maximum,
                    log10=config.log10,
                )
            )
        elif config.fit:
            for dataset_id in included_ids:
                if dataset_id not in dataset_values:
                    raise ValueError(
                        f"Fit dispatch parameter {parameter.name!r} omitted "
                        f"included dataset {dataset_id!r}."
                    )
                variable_by_dataset[dataset_id].append(
                    FitDatasetVariableParamSpec(
                        name=parameter.name,
                        initial=dataset_values[dataset_id],
                        minimum=config.minimum,
                        maximum=config.maximum,
                        log10=config.log10,
                    )
                )
        elif parameter.scope in {"per_dataset", FIT_SCOPE_CENTRED_FIXED_EFFECTS}:
            for dataset_id in included_ids:
                if dataset_id not in dataset_values:
                    raise ValueError(
                        f"Fit dispatch fixed parameter {parameter.name!r} omitted "
                        f"included dataset {dataset_id!r}."
                    )
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
                    variable_id=f"initial:{group.species}:{group.origin_dataset_id}",
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
        centred_variables=tuple(centred_variables),
        bounds={
            item.name: (item.minimum, item.maximum)
            for item in shared_variables
        },
        log10_params={item.name: item.log10 for item in shared_variables},
    )
    prepared_by_id = {item.dataset_id: item for item in prepared_datasets}
    if set(prepared_by_id) != set(included_ids):
        raise ValueError("Fit dispatch prepared dataset roster is incomplete.")
    return _FitDispatchCompilation(
        included_dataset_ids=included_ids,
        prepared_datasets=tuple(prepared_datasets),
        ordinary_shared_variables=tuple(shared_variables),
        ordinary_coupled_variables=tuple(coupled_variables),
        fixed_by_dataset=tuple(
            (item.dataset_id, item.fixed_params)
            for item in overrides
        ),
        ordinary_dataset_overrides=overrides,
        ordinary_centred_variables=tuple(centred_variables),
        candidate_names=tuple(
            sorted(
                parameter.name
                for parameter in setup.parameters
                if not parameter.name.startswith(INITIAL_PREFIX)
            )
        ),
        ordinary_optimizer_layout=layout,
    )


def build_fit_execution_request(
    *,
    setup: FitSetupSnapshot,
) -> FitExecutionRequest:
    """Compile one exact accepted setup into its immutable execution request."""
    if not isinstance(setup, FitSetupSnapshot):
        raise TypeError("setup must be FitSetupSnapshot.")
    readiness = fit_setup_readiness(
        origin=setup.origin,
        datasets=setup.datasets,
        parameters=setup.parameters,
        initials=setup.initials,
        shared_initial_groups=setup.shared_initial_groups,
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

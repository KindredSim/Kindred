"""Immutable Fit capture plus strict Fit-local request construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal

import numpy as np

from kindred.core.algebra.observable_introspection import (
    FitLocalObservable,
    fit_observable_names,
)
from kindred.core.analysis.dataset_sampling import compute_sampled_indices
from kindred.core.dataset_mapping import (
    resolve_fit_t_start,
)
from kindred.core.datasets import foundation as dataset_foundation
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    MechanismTabDocument,
    ProjectDocument,
    TabScientificSettings,
    canonical_initial_values,
    canonical_species_names,
    validate_fit_defaults,
    validate_fit_solver,
)
from kindred.core.simulator.dsl_parse import (
    DslDerivedObservableDefinition,
    DslDerivedScalarDefinition,
    PreparedDslRequestSet,
)
from kindred.core.simulator.wegscheider_symbolic import (
    UnresolvedWegscheiderCyclicityError,
)
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.fitting.contracts import FitParametricSettings, FitRunIdentity

__all__ = [
    "FitParameterConfiguration",
    "FitDatasetLocalSettings",
    "FitDatasetSetupEdit",
    "FitDatasetWeightMode",
    "FitSolver",
    "FitDatasetWeightModeSetupEdit",
    "FitDefaultsSetupEdit",
    "FitExecutionRequest",
    "FitInitialSetupEdit",
    "FitInitialGlobalVariable",
    "FitInitialScopeSetupEdit",
    "FitInitialVariable",
    "FitMappedDatasetSnapshot",
    "FitObservationSnapshot",
    "FitOriginSnapshot",
    "FitParameterVariable",
    "FitParameterSetupEdit",
    "FitParameterScopeSetupEdit",
    "FitSolverSetupEdit",
    "FitSelectedObservation",
    "FitSetupSnapshot",
    "build_fit_execution_request",
    "capture_fit_origin",
    "create_initial_fit_setup",
    "fit_origin_science_matches",
    "revalidate_fit_request_inclusions",
    "selected_observations",
    "fit_local_observable_definitions",
]

FitDatasetWeightMode = Literal["implicit", "user"]
FitSolver = Literal["BDF", "Radau"]


def fit_local_observable_definitions(
    observables: Iterable[FitLocalObservable],
    parameters: Iterable[FitParameterVariable],
) -> tuple[DslDerivedObservableDefinition, ...]:
    """Join observable scalar names to the canonical selected Fit configs."""

    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    definitions: list[DslDerivedObservableDefinition] = []
    for observable in observables:
        scalar_definitions: list[DslDerivedScalarDefinition] = []
        for scalar_name in observable.scalar_names:
            parameter = parameter_by_name.get(scalar_name)
            if parameter is None:
                raise ValueError(
                    f"Observable scalar {scalar_name!r} has no Fit parameter row."
                )
            config = parameter.selected_config
            scalar_definitions.append(
                DslDerivedScalarDefinition(
                    name=scalar_name,
                    value=config.value,
                    minimum=config.minimum,
                    maximum=config.maximum,
                    scope=parameter.scope,
                )
            )
        definitions.append(
            DslDerivedObservableDefinition(
                name=observable.name,
                expression=observable.expression,
                scalar_parameters=tuple(scalar_definitions),
            )
        )
    return tuple(definitions)


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
    model_species: str
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
            "model_species",
            _required(self.model_species, "mapped model series"),
        )
        times = tuple(_finite(value, "observation time") for value in self.t)
        values = tuple(_finite(value, "observation value") for value in self.y)
        if len(times) != len(values) or not times:
            raise ValueError("Fit observation requires equal nonempty t/y values.")
        rows = tuple(int(value) for value in self.source_rows)
        if (
            len(rows) != len(times)
            or any(row <= 0 for row in rows)
            or len(set(rows)) != len(rows)
        ):
            raise ValueError(
                "Fit observations require complete unique positive physical "
                "source-row identity."
            )
        object.__setattr__(self, "t", times)
        object.__setattr__(self, "y", values)
        object.__setattr__(self, "source_rows", rows)


@dataclass(frozen=True, slots=True)
class FitMappedDatasetSnapshot:
    dataset_id: str
    display_name: str
    set_id: str
    set_name: str
    initial_values: tuple[tuple[str, float], ...]
    observations: tuple[FitObservationSnapshot, ...]
    included_at_launch: bool
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
            value = _finite(raw_value, f"initial value {name}")
            if value < 0.0:
                raise ValueError(
                    f"Initial value {name!r} must be nonnegative."
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
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if math.isnan(minimum) or math.isnan(maximum):
            raise ValueError("Parameter bounds must not be NaN.")
        if minimum >= maximum or not minimum <= value <= maximum:
            raise ValueError("Parameter requires minimum <= value <= maximum.")
        if self.log10 and (minimum <= 0.0 or value <= 0.0):
            raise ValueError("Log10 fitting requires positive values.")
        values = tuple(
            (_required(dataset_id, "dataset ID"), _finite(item, "dataset value"))
            for dataset_id, item in self.dataset_values
        )
        if len({dataset_id for dataset_id, _value in values}) != len(values):
            raise ValueError("Parameter dataset values repeat a dataset ID.")
        if any(
            not minimum <= item <= maximum or (self.log10 and item <= 0.0)
            for _dataset_id, item in values
        ):
            raise ValueError("Parameter dataset values violate bounds or Log10 domain.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "fit", bool(self.fit))
        object.__setattr__(self, "log10", bool(self.log10))
        object.__setattr__(self, "dataset_values", values)


@dataclass(frozen=True, slots=True)
class FitParameterVariable:
    name: str
    display_name: str
    scope: str
    shared_config: FitParameterConfiguration
    per_dataset_config: FitParameterConfiguration
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
        if scope not in {"shared", "per_dataset"}:
            raise ValueError("Scalar Fit scope must be Shared or Per-dataset.")
        if not isinstance(self.shared_config, FitParameterConfiguration):
            raise TypeError("Shared scalar configuration must be typed.")
        if not isinstance(self.per_dataset_config, FitParameterConfiguration):
            raise TypeError("Per-dataset scalar configuration must be typed.")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "family", str(self.family or "").strip())
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "unit", str(self.unit or "1").strip() or "1")

    @property
    def selected_config(self) -> FitParameterConfiguration:
        """Return the one configuration selected for execution/presentation."""

        return (
            self.shared_config
            if self.scope == "shared"
            else self.per_dataset_config
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
    if target_scope not in {"shared", "per_dataset"}:
        raise ValueError("Scalar Fit scope must be Shared or Per-dataset.")
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
    configs["shared_config" if target_scope == "shared" else "per_dataset_config"] = active
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
    minimum: float
    maximum: float
    fit: bool = False
    log10: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", _required(self.set_id, "set ID"))
        object.__setattr__(self, "species", _required(self.species, "species"))
        initial = _finite(self.initial, f"{self.species} initial")
        minimum = _finite(self.minimum, f"{self.species} minimum")
        maximum = _finite(self.maximum, f"{self.species} maximum")
        if minimum >= maximum or not minimum <= initial <= maximum:
            raise ValueError(f"{self.species} requires minimum <= initial <= maximum.")
        if self.log10 and (minimum <= 0.0 or initial <= 0.0):
            raise ValueError(f"{self.species} log10 fitting requires positive values.")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        if minimum < 0.0:
            raise ValueError("Initial concentration minimum must be nonnegative.")


@dataclass(frozen=True, slots=True)
class FitInitialGlobalVariable:
    """One immutable species-level Global IC optimizer configuration."""

    species: str
    initial: float
    minimum: float
    maximum: float
    fit: bool = False
    log10: bool = False

    def __post_init__(self) -> None:
        species = _required(self.species, "initial species")
        initial = _finite(self.initial, f"{species} initial")
        minimum = _finite(self.minimum, f"{species} minimum")
        maximum = _finite(self.maximum, f"{species} maximum")
        if minimum < 0.0 or minimum >= maximum or not minimum <= initial <= maximum:
            raise ValueError(
                f"{species} requires nonnegative minimum <= initial <= maximum."
            )
        if self.log10 and (minimum <= 0.0 or initial <= 0.0):
            raise ValueError(f"{species} log10 fitting requires positive values.")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "fit", bool(self.fit))
        object.__setattr__(self, "log10", bool(self.log10))


@dataclass(frozen=True, slots=True)
class FitDatasetLocalSettings:
    dataset_id: str
    included: bool
    t_start: float
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
        t_start = _finite(self.t_start, "Fit simulation start")
        if t_start < 0.0:
            raise ValueError("Fit simulation start must be nonnegative.")
        low = _finite(self.data_range_min, "Data Range minimum")
        high = _finite(self.data_range_max, "Data Range maximum")
        if high < low:
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
        weight_names = [name for name, _value in target_weights]
        if (
            len(weight_names) != len(set(weight_names))
            or set(weight_names) != set(targets)
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
    capture_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required(self.tab_id, "tab ID"))
        object.__setattr__(self, "tab_name", _required(self.tab_name, "tab name"))
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be MechanismAuthoringSource.")
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
        if (
            self.prepared_dsl.analysis.normalized_text
            != self.mechanism_source.reactions_text.strip()
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
        capture_id = str(self.capture_id or "").strip()
        if not capture_id:
            payload = {
                "tab_id": self.tab_id,
            "mechanism_text": self.mechanism_source.reactions_text,
                "prepared_analysis_identity": self.prepared_dsl.analysis.identity,
                "scientific_settings": {
                    "temperature": self.scientific_settings.temperature,
                    "rtol": self.scientific_settings.rtol,
                    "atol": self.scientific_settings.atol,
                    "analytic_jacobian_policy": (
                        self.scientific_settings.analytic_jacobian_policy
                    ),
                    "wegscheider_policy": (
                        self.scientific_settings.wegscheider_policy
                    ),
                },
                "datasets": [
                    {
                        "dataset_id": item.dataset_id,
                        "set_id": item.set_id,
                        "initial_values": dict(item.initial_values),
                        "included_at_launch": item.included_at_launch,
                        "resolved_t_start": item.resolved_t_start,
                        "observations": [
                            {
                                "raw_observation_id": obs.raw_observation_id,
                                "model_species": obs.model_species,
                                "t": obs.t,
                                "y": obs.y,
                                "source_rows": obs.source_rows,
                            }
                            for obs in item.observations
                        ],
                    }
                    for item in sorted(
                        (
                            mapped
                            for mapped in self.mapped_datasets
                            if mapped.included_at_launch
                        ),
                        key=lambda value: value.dataset_id,
                    )
                ],
                "parameters": [
                    {
                        "name": item.name,
                        "display_name": item.display_name,
                        "initial": item.selected_config.value,
                        "family": item.family,
                        "role": item.role,
                        "unit": item.unit,
                    }
                    for item in self.parameter_defaults
                ],
            }
            capture_id = FitRunIdentity.build(payload).short_hash
        object.__setattr__(self, "capture_id", capture_id)


@dataclass(frozen=True, slots=True)
class FitExecutionRequest:
    origin: FitOriginSnapshot
    mechanism_source: MechanismAuthoringSource
    prepared_dsl: PreparedDslRequestSet
    solver: FitSolver
    local_observables: tuple[FitLocalObservable, ...]
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
    global_initials: tuple[FitInitialGlobalVariable, ...]
    initial_scopes: tuple[tuple[str, str], ...]
    defaults: FitDefaults
    dataset_weight_mode: FitDatasetWeightMode

    def __post_init__(self) -> None:
        if not isinstance(self.origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError("mechanism_source must be MechanismAuthoringSource.")
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
        if (
            self.prepared_dsl.analysis.normalized_text
            != self.mechanism_source.reactions_text.strip()
        ):
            raise ValueError(
                "Fit request prepared DSL must exactly match its effective source."
            )
        if not isinstance(self.defaults, FitDefaults):
            raise TypeError("defaults must be FitDefaults.")
        object.__setattr__(self, "solver", validate_fit_solver(self.solver))
        validate_fit_defaults(self.defaults)
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )
        local_observables = tuple(self.local_observables)
        raw_datasets = tuple(self.datasets)
        raw_parameters = tuple(self.parameters)
        raw_initials = tuple(self.initials)
        raw_global_initials = tuple(self.global_initials)
        if any(
            not isinstance(item, FitLocalObservable)
            for item in local_observables
        ):
            raise TypeError(
                "Fit request observables must be FitLocalObservable values."
            )
        if any(
            not isinstance(item, FitDatasetLocalSettings)
            for item in raw_datasets
        ):
            raise TypeError(
                "Fit request datasets must be FitDatasetLocalSettings values."
            )
        if any(
            not isinstance(item, FitParameterVariable)
            for item in raw_parameters
        ):
            raise TypeError(
                "Fit request parameters must be FitParameterVariable values."
            )
        if any(
            not isinstance(item, FitInitialVariable)
            for item in raw_initials
        ):
            raise TypeError(
                "Fit request initials must be FitInitialVariable values."
            )
        if any(
            not isinstance(item, FitInitialGlobalVariable)
            for item in raw_global_initials
        ):
            raise TypeError(
                "Fit request global initials must be FitInitialGlobalVariable values."
            )
        datasets = tuple(sorted(raw_datasets, key=lambda item: item.dataset_id))
        parameters = tuple(
            _replace_fit_parameter(
                parameter,
                dataset_values=tuple(sorted(parameter.selected_config.dataset_values)),
            )
            for parameter in raw_parameters
        )
        initials = tuple(
            sorted(raw_initials, key=lambda item: (item.set_id, item.species))
        )
        global_initials = tuple(sorted(raw_global_initials, key=lambda item: item.species))
        initial_scopes = tuple(
            (_required(species, "initial species"), str(scope).strip().lower())
            for species, scope in self.initial_scopes
        )
        initial_scopes = tuple(sorted(initial_scopes, key=lambda item: item[0]))
        if any(scope not in {"local", "global"} for _species, scope in initial_scopes):
            raise ValueError("Fit request initial scopes must be Local or Global.")
        scope_species = [species for species, _scope in initial_scopes]
        if len(scope_species) != len(set(scope_species)):
            raise ValueError("Fit request initial scope identities repeat.")
        global_species = [item.species for item in global_initials]
        if len(global_species) != len(set(global_species)):
            raise ValueError("Fit request Global initial identities repeat.")
        initial_species = {
            item.species
            for item in initials
        }
        if set(scope_species) != initial_species:
            raise ValueError(
                "Fit request must retain one scope for every initial species."
            )
        active_global_species = {
            species
            for species, scope in initial_scopes
            if scope == "global"
        }
        if set(item.species for item in global_initials) != active_global_species:
            raise ValueError(
                "Fit request Global configurations must match active Global scopes."
            )
        object.__setattr__(self, "local_observables", local_observables)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initials", initials)
        object.__setattr__(self, "global_initials", global_initials)
        object.__setattr__(self, "initial_scopes", initial_scopes)


@dataclass(frozen=True, slots=True)
class FitSetupSnapshot:
    """One complete typed Fit setup at launch or terminal completion."""

    origin_tab_id: str
    origin_capture_id: str
    mechanism_source: MechanismAuthoringSource
    prepared_dsl: PreparedDslRequestSet
    solver: FitSolver
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
    global_initials: tuple[FitInitialGlobalVariable, ...]
    initial_scopes: tuple[tuple[str, str], ...]
    defaults: FitDefaults
    local_observables: tuple[FitLocalObservable, ...]
    dataset_weight_mode: FitDatasetWeightMode

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin_tab_id",
            _required(self.origin_tab_id, "snapshot tab ID"),
        )
        object.__setattr__(
            self,
            "origin_capture_id",
            _required(self.origin_capture_id, "snapshot capture ID"),
        )
        if not isinstance(self.mechanism_source, MechanismAuthoringSource):
            raise TypeError(
                "snapshot mechanism_source must be MechanismAuthoringSource."
            )
        if not isinstance(self.prepared_dsl, PreparedDslRequestSet):
            raise TypeError(
                "snapshot prepared_dsl must be a PreparedDslRequestSet."
            )
        if (
            self.prepared_dsl.analysis.normalized_text
            != self.mechanism_source.reactions_text.strip()
        ):
            raise ValueError(
                "Fit snapshot prepared DSL must exactly match its source."
            )
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
        global_initials = tuple(self.global_initials)
        initial_scopes = tuple(
            (_required(species, "initial species"), str(scope).strip().lower())
            for species, scope in self.initial_scopes
        )
        local_observables = tuple(self.local_observables)
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
        if any(
            not isinstance(item, FitInitialGlobalVariable)
            for item in global_initials
        ):
            raise TypeError(
                "Fit snapshot global initials must be FitInitialGlobalVariable values."
            )
        if any(scope not in {"local", "global"} for _species, scope in initial_scopes):
            raise ValueError("Fit snapshot initial scopes must be Local or Global.")
        if len(initial_scopes) != len({species for species, _scope in initial_scopes}):
            raise ValueError("Fit snapshot initial scope identities repeat.")
        if any(
            not isinstance(item, FitLocalObservable)
            for item in local_observables
        ):
            raise TypeError(
                "Fit snapshot observables must be FitLocalObservable values."
            )
        dataset_ids = [item.dataset_id for item in datasets]
        parameter_names = [item.name for item in parameters]
        initial_keys = [(item.set_id, item.species) for item in initials]
        initial_species = {item.species for item in initials}
        global_species = [item.species for item in global_initials]
        scope_species = [species for species, _scope in initial_scopes]
        observable_names = [item.name for item in local_observables]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Fit snapshot dataset identities repeat.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Fit snapshot parameter identities repeat.")
        if len(initial_keys) != len(set(initial_keys)):
            raise ValueError("Fit snapshot initial-condition identities repeat.")
        if len(global_species) != len(set(global_species)):
            raise ValueError("Fit snapshot Global initial identities repeat.")
        if set(scope_species) != initial_species:
            raise ValueError(
                "Fit snapshot must retain one scope for every initial species."
            )
        active_global_species = {
            species
            for species, scope in initial_scopes
            if scope == "global"
        }
        if set(global_species) != active_global_species:
            raise ValueError(
                "Fit snapshot Global configurations must match active Global scopes."
            )
        if len(observable_names) != len(set(observable_names)):
            raise ValueError("Fit snapshot observable identities repeat.")
        retained_dataset_ids = set(dataset_ids)
        incomplete_parameter_maps = sorted(
            item.name
            for item in parameters
            if item.scope == "per_dataset"
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
        object.__setattr__(self, "global_initials", global_initials)
        object.__setattr__(self, "initial_scopes", initial_scopes)
        object.__setattr__(self, "local_observables", local_observables)

    @classmethod
    def from_request(
        cls,
        request: FitExecutionRequest,
    ) -> "FitSetupSnapshot":
        if not isinstance(request, FitExecutionRequest):
            raise TypeError("request must be FitExecutionRequest.")
        return cls(
            origin_tab_id=request.origin.tab_id,
            origin_capture_id=request.origin.capture_id,
            mechanism_source=request.mechanism_source,
            prepared_dsl=request.prepared_dsl,
            solver=request.solver,
            datasets=request.datasets,
            parameters=request.parameters,
            initials=request.initials,
            global_initials=request.global_initials,
            initial_scopes=request.initial_scopes,
            defaults=request.defaults,
            local_observables=request.local_observables,
            dataset_weight_mode=request.dataset_weight_mode,
        )


@dataclass(frozen=True, slots=True)
class FitSelectedObservation:
    raw_observation_id: str
    raw_column: str
    model_species: str
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FitDatasetSetupEdit:
    """Typed replacement intent for one stable dataset setup row."""

    origin_tab_id: str
    origin_capture_id: str
    dataset: FitDatasetLocalSettings
    raw_observation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        if not isinstance(self.dataset, FitDatasetLocalSettings):
            raise TypeError("dataset edit must carry FitDatasetLocalSettings.")
        if self.raw_observation_id is not None:
            object.__setattr__(
                self,
                "raw_observation_id",
                _required(self.raw_observation_id, "raw observation ID"),
            )


@dataclass(frozen=True, slots=True)
class FitDatasetWeightModeSetupEdit:
    """Typed replacement intent for the one run-wide dataset-weight mode."""

    origin_tab_id: str
    origin_capture_id: str
    dataset_weight_mode: FitDatasetWeightMode

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin_tab_id",
            _required(self.origin_tab_id, "origin tab ID"),
        )
        object.__setattr__(
            self,
            "origin_capture_id",
            _required(self.origin_capture_id, "origin capture ID"),
        )
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )


@dataclass(frozen=True, slots=True)
class FitParameterSetupEdit:
    """Typed replacement intent for one stable parameter setup row."""

    origin_tab_id: str
    origin_capture_id: str
    parameter: FitParameterVariable

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        if not isinstance(self.parameter, FitParameterVariable):
            raise TypeError("parameter edit must carry FitParameterVariable.")


@dataclass(frozen=True, slots=True)
class FitParameterScopeSetupEdit:
    """Selection-only owner intent for a scalar scope transition."""

    origin_tab_id: str
    origin_capture_id: str
    parameter_name: str
    scope: Literal["shared", "per_dataset"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        object.__setattr__(self, "parameter_name", _required(self.parameter_name, "parameter name"))
        if self.scope not in {"shared", "per_dataset"}:
            raise ValueError("Scalar scope transition must target Shared or Per-dataset.")


@dataclass(frozen=True, slots=True)
class FitSolverSetupEdit:
    """Owner intent for the one Fit-wide numerical solver policy."""

    origin_tab_id: str
    origin_capture_id: str
    solver: FitSolver

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        object.__setattr__(self, "solver", validate_fit_solver(self.solver))


@dataclass(frozen=True, slots=True)
class FitInitialSetupEdit:
    """Typed replacement intent for one stable set/species setup row."""

    origin_tab_id: str
    origin_capture_id: str
    initial: FitInitialVariable

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        if not isinstance(self.initial, FitInitialVariable):
            raise TypeError("initial edit must carry FitInitialVariable.")


@dataclass(frozen=True, slots=True)
class FitInitialScopeSetupEdit:
    """Atomic owner intent for one species' IC scope transition.

    A Global transition carries the freshly authored value and bounds in this
    same immutable intent.  Local transitions carry no Global configuration;
    the owner removes any previously active Global entry atomically.
    """

    origin_tab_id: str
    origin_capture_id: str
    species: str
    scope: Literal["global", "local"]
    global_initial: FitInitialGlobalVariable | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        object.__setattr__(self, "species", _required(self.species, "initial species"))
        scope = str(self.scope or "").strip().lower()
        if scope not in {"global", "local"}:
            raise ValueError("Initial scope transition must target Global or Local.")
        object.__setattr__(self, "scope", scope)
        if scope == "global":
            if not isinstance(self.global_initial, FitInitialGlobalVariable):
                raise ValueError(
                    "Global scope transition requires fresh value and bounds."
                )
            if self.global_initial.species != self.species:
                raise ValueError(
                    "Global scope transition species does not match its configuration."
                )
        elif self.global_initial is not None:
            raise ValueError("Local scope transition cannot carry Global state.")


@dataclass(frozen=True, slots=True)
class FitDefaultsSetupEdit:
    """Typed replacement intent for optimizer-only Fit defaults."""

    origin_tab_id: str
    origin_capture_id: str
    defaults: FitDefaults

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_tab_id", _required(self.origin_tab_id, "origin tab ID"))
        object.__setattr__(self, "origin_capture_id", _required(self.origin_capture_id, "origin capture ID"))
        if not isinstance(self.defaults, FitDefaults):
            raise TypeError("defaults edit must carry FitDefaults.")


def _parameter_bounds(value: float) -> tuple[float, float]:
    if value > 0.0:
        low = max(value * 0.01, 1e-12)
        high = value * 100.0
        if high > low:
            return low, high
    span = max(1.0, abs(value) * 10.0)
    return value - span, value + span


def _initial_bounds(value: float) -> tuple[float, float]:
    if value > 0.0:
        return 0.0, max(value * 10.0, value + 1.0)
    return 0.0, 1.0


def _scan_parameters(tab: MechanismTabDocument) -> tuple[FitParameterVariable, ...]:
    if not tab.mechanism_source.reactions_text.strip():
        return ()
    variables: list[FitParameterVariable] = []
    if not isinstance(tab.prepared_dsl, PreparedDslRequestSet):
        raise ValueError("Fit requires an accepted prepared mechanism.")
    try:
        coordinates = enumerate_parameter_coordinates(
            tab.prepared_dsl,
            wegscheider_cyclicity_enabled=bool(
                tab.scientific_settings.wegscheider_policy
            ),
        )
    except UnresolvedWegscheiderCyclicityError:
        # Fit's worker owns the guided correction handshake.  Keep the
        # accepted source and captured policy unchanged, but allow only this
        # typed unresolved-cycle admission to reach that existing path.
        coordinates = enumerate_parameter_coordinates(
            tab.prepared_dsl,
            wegscheider_cyclicity_enabled=False,
        )
    for coordinate in coordinates:
        name = _required(coordinate.name, "parameter name")
        value = _finite(coordinate.value, f"parameter {name}")
        low = float(coordinate.minimum)
        high = float(coordinate.maximum)
        if math.isnan(low) or math.isnan(high) or low >= high:
            low, high = _parameter_bounds(value)
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
            included=mapped.included_at_launch,
            t_start=mapped.resolved_t_start,
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
                observation.model_species
                for observation in mapped.observations
            ),
            target_weights=tuple(
                (observation.model_species, 1.0)
                for observation in mapped.observations
            ),
            observation_targets=tuple(
                (
                    observation.raw_observation_id,
                    observation.model_species,
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
            minimum=_initial_bounds(value)[0],
            maximum=_initial_bounds(value)[1],
            fit=False,
            log10=False,
        )
        for (set_id, species), value in values_by_key.items()
    )
    # Global is an explicit, active-only configuration. Initial capture owns
    # Local rows and scope selection only; no dormant scientific Global value
    # is invented before the user authors one.
    global_initials: tuple[FitInitialGlobalVariable, ...] = ()
    initial_scopes = tuple(
        (species, "local")
        for species in sorted({item.species for item in initials})
    )
    retained_dataset_ids = tuple(item.dataset_id for item in origin.mapped_datasets)
    parameters = tuple(
        _complete_fit_parameter_scopes(parameter, retained_dataset_ids)
        for parameter in origin.parameter_defaults
    )
    return FitSetupSnapshot(
        origin_tab_id=origin.tab_id,
        origin_capture_id=origin.capture_id,
        mechanism_source=origin.mechanism_source,
        prepared_dsl=origin.prepared_dsl,
        solver="BDF",
        datasets=datasets,
        parameters=parameters,
        initials=initials,
        global_initials=global_initials,
        initial_scopes=initial_scopes,
        defaults=origin.fit_defaults,
        local_observables=(),
        dataset_weight_mode="implicit",
    )


def fit_origin_science_matches(
    captured: FitOriginSnapshot,
    current: FitOriginSnapshot,
    *,
    included_dataset_ids: Iterable[str] | None = None,
) -> bool:
    """Compare only science that entered the active Fit dataset roster."""

    if not isinstance(captured, FitOriginSnapshot) or not isinstance(
        current,
        FitOriginSnapshot,
    ):
        raise TypeError("Fit origin comparison requires FitOriginSnapshot values.")

    active_ids = frozenset(
        str(dataset_id)
        for dataset_id in (
            included_dataset_ids
            if included_dataset_ids is not None
            else (
                item.dataset_id
                for item in captured.mapped_datasets
                if item.included_at_launch
            )
        )
        if str(dataset_id).strip()
    )

    def dataset_science(
        item: FitMappedDatasetSnapshot,
    ) -> tuple[object, ...]:
        return (
            item.dataset_id,
            item.set_id,
            item.initial_values,
            item.resolved_t_start,
            tuple(
                (
                    observation.raw_observation_id,
                    observation.model_species,
                    observation.t,
                    observation.y,
                    observation.source_rows,
                )
                for observation in item.observations
            ),
        )

    return bool(
        captured.tab_id == current.tab_id
        and captured.mechanism_source == current.mechanism_source
        and captured.prepared_dsl is current.prepared_dsl
        and captured.scientific_settings == current.scientific_settings
        and tuple(
            dataset_science(item)
            for item in sorted(
                captured.mapped_datasets,
                key=lambda value: value.dataset_id,
            )
            if item.dataset_id in active_ids
        )
        == tuple(
            dataset_science(item)
            for item in sorted(
                current.mapped_datasets,
                key=lambda value: value.dataset_id,
            )
            if item.dataset_id in active_ids
        )
        and active_ids.issubset(
            item.dataset_id for item in captured.mapped_datasets
        )
        and active_ids.issubset(
            item.dataset_id for item in current.mapped_datasets
        )
        and captured.parameter_defaults == current.parameter_defaults
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
    mechanism_species = canonical_species_names(tab)
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
        dataset = dataset_foundation.validate_raw_dataset_record(
            document.dataset(dataset_id)
        )
        initial_set = tab.initial_condition_set(link.set_id)
        validated = (
            dataset_foundation.validate_fit_inclusion(
                dataset=dataset,
                set_id=link.set_id,
                observation_mapping=link.observation_mapping,
                initial_values=initial_set.values_dict(),
                mechanism_species=mechanism_species,
                resolved_t_start=resolve_fit_t_start(
                    tab,
                    initial_set.set_id,
                    label=f"Fit snapshot for dataset {dataset.dataset_id!r}",
                ),
            )
            if link.include_in_fit
            else None
        )
        by_raw_id = {
            observation.raw_observation_id: observation
            for observation in dataset.observations
        }
        observations: list[FitObservationSnapshot] = []
        mapping = (
            tuple(
                (target.raw_observation_id, target.model_target)
                for target in validated.targets
            )
            if validated is not None
            else link.observation_mapping
        )
        for raw_observation_id, model_species in mapping:
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
                    model_species=model_species,
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
                included_at_launch=bool(link.include_in_fit),
                resolved_t_start=(
                    validated.resolved_t_start
                    if validated is not None
                    else resolve_fit_t_start(
                        tab,
                        initial_set.set_id,
                        label=(
                            f"Fit snapshot for dataset {dataset.dataset_id!r}"
                        ),
                    )
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
        usable = tuple(
            index
            for index, (time_value, source_row) in enumerate(
                zip(observation.t, observation.source_rows)
            )
            if (
                time_value >= local.t_start
                and local.data_range_min <= time_value <= local.data_range_max
                and (x_rows is None or source_row in x_rows)
            )
        )
        if not usable:
            pairing = (
                " and Parametric-X physical-row pairing"
                if x_rows is not None
                else ""
            )
            raise ValueError(
                f"Dataset {mapped.display_name!r} target {target!r} has "
                "zero usable observations after Data Range, Simulation "
                f"start{pairing}."
            )
        maximum = local.maximum_observations_per_target
        indices = usable
        if maximum is not None:
            indices = tuple(
                usable[index]
                for index in compute_sampled_indices(
                    t=np.asarray([observation.t[index] for index in usable]),
                    t_min=-math.inf,
                    t_max=math.inf,
                    n_points=maximum,
                )
            )
        selected.append(
            FitSelectedObservation(
                raw_observation_id=observation.raw_observation_id,
                raw_column=observation.raw_column,
                model_species=target,
                t=tuple(float(observation.t[index]) for index in indices),
                y=tuple(float(observation.y[index]) for index in indices),
                source_rows=tuple(
                    int(observation.source_rows[index]) for index in indices
                ),
            )
        )
    return tuple(selected)


def revalidate_fit_request_inclusions(
    *,
    document: ProjectDocument,
    request: FitExecutionRequest,
) -> None:
    """Revalidate every currently included dataset through Dataset authority."""

    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be ProjectDocument.")
    if not isinstance(request, FitExecutionRequest):
        raise TypeError("request must be FitExecutionRequest.")
    tab = document.tab(request.origin.tab_id)
    species = canonical_species_names(tab)
    mapped_by_id = {
        item.dataset_id: item for item in request.origin.mapped_datasets
    }
    allowed_observable_names = {
        item.name for item in request.local_observables
    } | set(fit_observable_names(request.prepared_dsl))
    for local in request.datasets:
        if not local.included:
            continue
        dataset = dataset_foundation.validate_raw_dataset_record(
            document.dataset(local.dataset_id)
        )
        link = tab.dataset_link(local.dataset_id)
        if link is None:
            raise ValueError(
                f"Dataset {local.dataset_id!r} is no longer linked in the active tab."
            )
        initial_set = tab.initial_condition_set(link.set_id)
        validated = dataset_foundation.validate_fit_inclusion(
            dataset=dataset,
            set_id=link.set_id,
            observation_mapping=link.observation_mapping,
            initial_values=initial_set.values_dict(),
            mechanism_species=species,
            resolved_t_start=resolve_fit_t_start(
                tab,
                initial_set.set_id,
                label=f"Fit inclusion for dataset {dataset.dataset_id!r}",
            ),
        )
        captured = mapped_by_id[local.dataset_id]
        if validated.set_id != captured.set_id:
            raise ValueError(
                f"Dataset {captured.display_name!r} changed mapped set."
            )
        validated_by_raw = {
            item.raw_observation_id: item for item in validated.targets
        }
        captured_by_raw = {
            item.raw_observation_id: item for item in captured.observations
        }
        for raw_id, model_target in local.observation_targets:
            authoritative = validated_by_raw.get(raw_id)
            snapshot = captured_by_raw.get(raw_id)
            if authoritative is None or snapshot is None:
                raise ValueError(
                    f"Dataset {captured.display_name!r} target "
                    f"raw observation {raw_id!r} is no longer eligible."
                )
            if (
                authoritative.raw_observation_id != raw_id
                or snapshot.raw_observation_id != raw_id
                or tuple(authoritative.t) != snapshot.t
                or tuple(authoritative.y) != snapshot.y
                or tuple(authoritative.source_rows) != snapshot.source_rows
            ):
                raise ValueError(
                    f"Dataset {captured.display_name!r} observation "
                    f"{raw_id!r} changed after Fit capture."
                )
            if (
                model_target != authoritative.model_target
                and model_target not in allowed_observable_names
            ):
                raise ValueError(
                    f"Dataset {captured.display_name!r} mapped target "
                    f"{model_target!r} is no longer valid."
                )


def build_fit_execution_request(
    *,
    origin: FitOriginSnapshot,
    mechanism_source: MechanismAuthoringSource,
    prepared_dsl: PreparedDslRequestSet,
    solver: FitSolver = "BDF",
    local_observables: Iterable[FitLocalObservable],
    datasets: Iterable[FitDatasetLocalSettings],
    parameters: Iterable[FitParameterVariable],
    initials: Iterable[FitInitialVariable],
    global_initials: Iterable[FitInitialGlobalVariable],
    initial_scopes: Iterable[tuple[str, str]],
    dataset_weight_mode: FitDatasetWeightMode,
    defaults: FitDefaults | None = None,
) -> FitExecutionRequest:
    """Validate a complete owner-built Fit request at the execution boundary."""
    request = FitExecutionRequest(
        origin=origin,
        mechanism_source=mechanism_source,
        prepared_dsl=prepared_dsl,
        solver=solver,
        local_observables=tuple(local_observables),
        datasets=tuple(datasets),
        parameters=tuple(parameters),
        initials=tuple(initials),
        global_initials=tuple(global_initials),
        initial_scopes=tuple(initial_scopes),
        defaults=defaults or origin.fit_defaults,
        dataset_weight_mode=dataset_weight_mode,
    )
    mapped_by_id = {item.dataset_id: item for item in origin.mapped_datasets}
    requested_ids = [item.dataset_id for item in request.datasets]
    if (
        len(requested_ids) != len(set(requested_ids))
        or set(requested_ids) != set(mapped_by_id)
    ):
        raise ValueError("Fit request must contain each mapped dataset exactly once.")
    included = {
        item.dataset_id
        for item in request.datasets
        if item.included
    }
    if not included:
        raise ValueError("Select at least one mapped dataset for fitting.")
    retained_dataset_ids = set(mapped_by_id)
    parameter_names = [item.name for item in request.parameters]
    expected_names = [item.name for item in origin.parameter_defaults]
    local_scalar_names = {
        scalar_name
        for observable in request.local_observables
        for scalar_name in observable.scalar_names
    }
    if (
        len(parameter_names) != len(set(parameter_names))
        or set(parameter_names) != set(expected_names) | local_scalar_names
    ):
        raise ValueError(
            "Fit request must contain each captured and Fit-local scalar exactly once."
        )
    for parameter in request.parameters:
        local_ids = {
        dataset_id for dataset_id, _value in parameter.selected_config.dataset_values
        }
        if parameter.scope == "per_dataset" and local_ids != retained_dataset_ids:
            raise ValueError(
                f"Parameter {parameter.name!r} requires one retained value "
                "for every captured dataset."
            )
    initial_keys = [(item.set_id, item.species) for item in request.initials]
    expected_initial_keys = {
        (mapped.set_id, species)
        for mapped in origin.mapped_datasets
        for species, _value in mapped.initial_values
    }
    if len(initial_keys) != len(set(initial_keys)) or set(initial_keys) != expected_initial_keys:
        raise ValueError(
            "Fit request must contain each mapped initial concentration exactly once."
        )
    initial_rows_by_species: dict[str, list[FitInitialVariable]] = {}
    for initial in request.initials:
        initial_rows_by_species.setdefault(initial.species, []).append(initial)
    global_by_species = {item.species: item for item in request.global_initials}
    scopes = dict(request.initial_scopes)
    expected_species = set(initial_rows_by_species)
    if set(scopes) != expected_species:
        raise ValueError(
            "Fit request must retain one scope per initial species."
        )
    active_global_species = {
        species for species, scope in scopes.items() if scope == "global"
    }
    if set(global_by_species) != active_global_species:
        raise ValueError(
            "Fit request Global configurations must match active Global scopes."
        )
    included_set_ids = {
        mapped.set_id
        for mapped in origin.mapped_datasets
        if mapped.dataset_id in included
    }
    for species, rows in initial_rows_by_species.items():
        active_rows = [
            item for item in rows if item.set_id in included_set_ids
        ]
        selected_scope = scopes[species]
        if selected_scope == "global":
            if not active_rows:
                continue
            global_by_species[species]  # accessed to enforce identity above
    for local in request.datasets:
        if local.included:
            observations = selected_observations(
                mapped_by_id[local.dataset_id],
                local,
            )
            if not observations:
                raise ValueError(
                    f"Dataset {mapped_by_id[local.dataset_id].display_name!r} "
                    "has no selected observations."
                )
            if local.data_range_max <= local.t_start:
                raise ValueError(
                    f"Dataset {mapped_by_id[local.dataset_id].display_name!r} "
                    "Fit interval requires Data Range maximum to be greater "
                    "than Simulation start."
                )
    has_fitted_initial = any(
        item.fit
        and scopes.get(item.species) == "local"
        and any(
            mapped.set_id == item.set_id and mapped.dataset_id in included
            for mapped in origin.mapped_datasets
        )
        for item in request.initials
    ) or any(
        item.fit
        and scopes.get(item.species) == "global"
        and any(
            mapped.dataset_id in included
            and any(
                species == item.species
                for species, _value in mapped.initial_values
            )
            for mapped in origin.mapped_datasets
        )
        for item in request.global_initials
    )
    if not any(item.selected_config.fit for item in request.parameters) and not has_fitted_initial:
        raise ValueError(
            "Select at least one parameter or initial concentration to fit."
        )
    return request

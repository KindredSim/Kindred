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
    validate_fit_inclusion,
    validate_raw_dataset_record,
)
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    MechanismTabDocument,
    ProjectDocument,
    TabScientificSettings,
    canonical_initial_values,
    canonical_species_names,
    default_project_document,
)
from kindred.core.simulator.dsl_parse import PreparedDslRequestSet
from kindred.gui.parameter_enumeration import enumerate_parameter_coordinates
from kindred.gui.fitting.contracts import FitParametricSettings, FitRunIdentity

__all__ = [
    "FitDatasetLocalSettings",
    "FitDatasetSetupEdit",
    "FitDatasetWeightMode",
    "FitDatasetWeightModeSetupEdit",
    "FitDefaultsSetupEdit",
    "FitExecutionRequest",
    "FitInitialSetupEdit",
    "FitInitialVariable",
    "FitMappedDatasetSnapshot",
    "FitObservationSnapshot",
    "FitOriginSnapshot",
    "FitParameterVariable",
    "FitParameterSetupEdit",
    "FitSelectedObservation",
    "FitSetupSnapshot",
    "build_fit_execution_request",
    "capture_fit_origin",
    "create_initial_fit_setup",
    "fit_origin_science_matches",
    "revalidate_fit_request_inclusions",
    "selected_observations",
]

FitDatasetWeightMode = Literal["implicit", "user"]


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
class FitParameterVariable:
    name: str
    display_name: str
    initial: float
    minimum: float
    maximum: float
    fit: bool = True
    log10: bool = False
    scope: str = "shared"
    dataset_values: tuple[tuple[str, float], ...] = ()
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
        initial = _finite(self.initial, f"{self.name} initial")
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if math.isnan(minimum) or math.isnan(maximum):
            raise ValueError(f"{self.name} bounds cannot be NaN.")
        if minimum >= maximum or not minimum <= initial <= maximum:
            raise ValueError(f"{self.name} requires minimum <= initial <= maximum.")
        if self.log10 and (minimum <= 0.0 or initial <= 0.0):
            raise ValueError(f"{self.name} log10 fitting requires positive values.")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        scope = str(self.scope or "").strip().lower()
        if scope not in {"shared", "per_dataset"}:
            raise ValueError("Scalar Fit scope must be Shared or Per-dataset.")
        dataset_values = tuple(
            (_required(dataset_id, "dataset ID"), _finite(value, self.name))
            for dataset_id, value in self.dataset_values
        )
        if len(dataset_values) != len({item[0] for item in dataset_values}):
            raise ValueError(f"{self.name} repeats a dataset-local value.")
        if any(
            not minimum <= value <= maximum
            or (self.log10 and value <= 0.0)
            for _dataset_id, value in dataset_values
        ):
            raise ValueError(
                f"{self.name} dataset values must satisfy its bounds and "
                "Log10 domain."
            )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "dataset_values", dataset_values)
        object.__setattr__(self, "family", str(self.family or "").strip())
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "unit", str(self.unit or "1").strip() or "1")


@dataclass(frozen=True, slots=True)
class FitInitialVariable:
    set_id: str
    species: str
    initial: float
    minimum: float
    maximum: float
    fit: bool = False
    log10: bool = False
    scope: str = "global"

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
        scope = str(self.scope or "").strip().lower()
        if scope not in {"global", "local"}:
            raise ValueError("Initial Fit scope must be Global or Local.")
        if minimum < 0.0:
            raise ValueError("Initial concentration minimum must be nonnegative.")
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True, slots=True)
class FitDatasetLocalSettings:
    dataset_id: str
    included: bool
    t_start: float
    solver: str = "BDF"
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
        solver = str(self.solver or "").strip()
        if solver not in {"BDF", "Radau"}:
            raise ValueError("Fit solver must be BDF or Radau.")
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
            or set(mapped_targets) != set(targets)
        ):
            raise ValueError(
                "Observation target overrides must exactly match selected "
                "Fit targets."
            )
        object.__setattr__(self, "t_start", t_start)
        object.__setattr__(self, "solver", solver)
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
                    "sparse_jacobian_policy": (
                        self.scientific_settings.sparse_jacobian_policy
                    ),
                    "wegscheider_policy": (
                        self.scientific_settings.wegscheider_policy
                    ),
                },
                "fit_defaults": {
                    "method": self.fit_defaults.method,
                    "max_nfev": self.fit_defaults.max_nfev,
                    "ftol": self.fit_defaults.ftol,
                    "xtol": self.fit_defaults.xtol,
                    "use_seed": self.fit_defaults.use_seed,
                    "seed": self.fit_defaults.seed,
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
                    for item in self.mapped_datasets
                ],
                "parameters": [
                    {
                        "name": item.name,
                        "display_name": item.display_name,
                        "initial": item.initial,
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
    local_observables: tuple[FitLocalObservable, ...]
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
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
        default_project_document(fit_defaults=self.defaults)
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )
        local_observables = tuple(self.local_observables)
        datasets = tuple(self.datasets)
        parameters = tuple(self.parameters)
        initials = tuple(self.initials)
        if any(
            not isinstance(item, FitLocalObservable)
            for item in local_observables
        ):
            raise TypeError(
                "Fit request observables must be FitLocalObservable values."
            )
        if any(
            not isinstance(item, FitDatasetLocalSettings)
            for item in datasets
        ):
            raise TypeError(
                "Fit request datasets must be FitDatasetLocalSettings values."
            )
        if any(
            not isinstance(item, FitParameterVariable)
            for item in parameters
        ):
            raise TypeError(
                "Fit request parameters must be FitParameterVariable values."
            )
        if any(
            not isinstance(item, FitInitialVariable)
            for item in initials
        ):
            raise TypeError(
                "Fit request initials must be FitInitialVariable values."
            )
        object.__setattr__(self, "local_observables", local_observables)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initials", initials)


@dataclass(frozen=True, slots=True)
class FitSetupSnapshot:
    """One complete typed Fit setup at launch or terminal completion."""

    origin_tab_id: str
    origin_capture_id: str
    mechanism_source: MechanismAuthoringSource
    prepared_dsl: PreparedDslRequestSet
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
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
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )
        datasets = tuple(self.datasets)
        parameters = tuple(self.parameters)
        initials = tuple(self.initials)
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
            not isinstance(item, FitLocalObservable)
            for item in local_observables
        ):
            raise TypeError(
                "Fit snapshot observables must be FitLocalObservable values."
            )
        dataset_ids = [item.dataset_id for item in datasets]
        parameter_names = [item.name for item in parameters]
        initial_keys = [(item.set_id, item.species) for item in initials]
        observable_names = [item.name for item in local_observables]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Fit snapshot dataset identities repeat.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Fit snapshot parameter identities repeat.")
        if len(initial_keys) != len(set(initial_keys)):
            raise ValueError("Fit snapshot initial-condition identities repeat.")
        if len(observable_names) != len(set(observable_names)):
            raise ValueError("Fit snapshot observable identities repeat.")
        retained_dataset_ids = set(dataset_ids)
        incomplete_parameter_maps = sorted(
            item.name
            for item in parameters
            if {
                dataset_id
                for dataset_id, _value in item.dataset_values
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
            datasets=request.datasets,
            parameters=request.parameters,
            initials=request.initials,
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
        high = min(value * 100.0, 1e300)
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
    coordinates = enumerate_parameter_coordinates(
        tab.prepared_dsl,
        wegscheider_cyclicity_enabled=bool(
            tab.scientific_settings.wegscheider_policy
        ),
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
                initial=value,
                minimum=low,
                maximum=high,
                fit=True,
                log10=bool(coordinate.log10_allowed and value > 0.0 and low > 0.0),
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
            solver="BDF",
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
            scope="local",
        )
        for (set_id, species), value in values_by_key.items()
    )
    retained_dataset_ids = tuple(item.dataset_id for item in datasets)
    parameters = tuple(
        replace(
            parameter,
            dataset_values=tuple(
                (dataset_id, parameter.initial)
                for dataset_id in retained_dataset_ids
            ),
        )
        for parameter in origin.parameter_defaults
    )
    return FitSetupSnapshot(
        origin_tab_id=origin.tab_id,
        origin_capture_id=origin.capture_id,
        mechanism_source=origin.mechanism_source,
        prepared_dsl=origin.prepared_dsl,
        datasets=datasets,
        parameters=parameters,
        initials=initials,
        defaults=origin.fit_defaults,
        local_observables=(),
        dataset_weight_mode="implicit",
    )


def fit_origin_science_matches(
    captured: FitOriginSnapshot,
    current: FitOriginSnapshot,
) -> bool:
    """Compare Fit admission science while excluding mutable display labels."""

    if not isinstance(captured, FitOriginSnapshot) or not isinstance(
        current,
        FitOriginSnapshot,
    ):
        raise TypeError("Fit origin comparison requires FitOriginSnapshot values.")

    def dataset_science(
        item: FitMappedDatasetSnapshot,
    ) -> tuple[object, ...]:
        return (
            item.dataset_id,
            item.set_id,
            item.initial_values,
            item.included_at_launch,
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
        and tuple(dataset_science(item) for item in captured.mapped_datasets)
        == tuple(dataset_science(item) for item in current.mapped_datasets)
        and captured.parameter_defaults == current.parameter_defaults
    )


def capture_fit_origin(document: ProjectDocument, tab_id: str) -> FitOriginSnapshot:
    """Copy mapped dataset science from one exact tab."""
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be ProjectDocument.")
    tab = document.tab(tab_id)
    if not isinstance(tab.prepared_dsl, PreparedDslRequestSet):
        raise ValueError("Fit requires an accepted visible mechanism.")
    mechanism_species = canonical_species_names(tab)
    mapped: list[FitMappedDatasetSnapshot] = []
    for dataset_id, link in tab.dataset_links:
        if link.set_id is None:
            continue
        dataset = validate_raw_dataset_record(document.dataset(dataset_id))
        initial_set = tab.initial_condition_set(link.set_id)
        validated = (
            validate_fit_inclusion(
                document=document,
                tab=tab,
                dataset_id=dataset_id,
                mechanism_species=mechanism_species,
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
                    validated.execution.t_start
                    if validated is not None
                    else tab.resolved_execution_settings(initial_set.set_id).t_start
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
        x_observations = tuple(
            observation
            for observation in mapped.observations
            if observation.model_species == local.parametric.x_name
        )
        if len(x_observations) != 1:
            raise ValueError(
                f"Dataset {mapped.display_name!r} Parametric-X series "
                f"{local.parametric.x_name!r} must map to exactly one "
                "observation."
            )
        x_rows = set(x_observations[0].source_rows)

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
        validated = validate_fit_inclusion(
            document=document,
            tab=tab,
            dataset_id=local.dataset_id,
            mechanism_species=species,
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
    local_observables: Iterable[FitLocalObservable],
    datasets: Iterable[FitDatasetLocalSettings],
    parameters: Iterable[FitParameterVariable],
    initials: Iterable[FitInitialVariable],
    dataset_weight_mode: FitDatasetWeightMode,
    defaults: FitDefaults | None = None,
) -> FitExecutionRequest:
    """Validate a complete owner-built Fit request at the execution boundary."""
    request = FitExecutionRequest(
        origin=origin,
        mechanism_source=mechanism_source,
        prepared_dsl=prepared_dsl,
        local_observables=tuple(local_observables),
        datasets=tuple(datasets),
        parameters=tuple(parameters),
        initials=tuple(initials),
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
        scalar.name
        for observable in request.local_observables
        for scalar in observable.scalar_parameters
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
            dataset_id for dataset_id, _value in parameter.dataset_values
        }
        if local_ids != retained_dataset_ids:
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
    included_set_ids = {
        mapped.set_id
        for mapped in origin.mapped_datasets
        if mapped.dataset_id in included
    }
    for species, rows in initial_rows_by_species.items():
        active_rows = [
            item for item in rows if item.set_id in included_set_ids
        ]
        scopes = {item.scope for item in active_rows}
        if len(scopes) != 1:
            raise ValueError(
                f"Included initial {species!r} cannot mix Global and Local scope."
            )
        if scopes == {"global"}:
            signatures = {
                (
                    item.initial,
                    item.minimum,
                    item.maximum,
                    item.fit,
                    item.log10,
                )
                for item in active_rows
            }
            if len(signatures) != 1:
                raise ValueError(
                    f"Global initial {species!r} requires one explicit "
                    "value, bounds, Fit selection, and Log10 selection."
                )
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
    if not any(item.fit for item in request.parameters) and not any(
        item.fit
        and any(
            mapped.set_id == item.set_id and mapped.dataset_id in included
            for mapped in origin.mapped_datasets
        )
        for item in request.initials
    ):
        raise ValueError(
            "Select at least one parameter or initial concentration to fit."
        )
    return request

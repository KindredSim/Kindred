"""Immutable Fit capture plus strict Fit-local request construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

import numpy as np

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
from kindred.core.algebra.observable_introspection import (
    AdmittedAlgebraProgram,
    FitLocalObservable,
    admit_fit_local_algebra,
    fit_observable_names,
)
from kindred.gui.fitting.constants import INITIAL_PREFIX
from kindred.core.analysis.dataset_sampling import compute_sampled_indices
from kindred.core.dataset_mapping import (
    resolve_fit_t_start,
)
from kindred.core.datasets import foundation as dataset_foundation
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
    canonical_observation_target_names,
    canonical_species_names,
    validate_fit_defaults,
    validate_fit_solver,
)
from kindred.core.species import TIME_MAX_S, validate_concentration
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
    "FitPhysicalCoordinatePolicy",
    "FitInitialGlobalVariable",
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
    "revalidate_fit_request_inclusions",
    "selected_observations",
    "fit_execution_span",
    "build_fit_local_admission",
    "fit_uncertainty_coordinates",
    "FitLocalScalarProvenance",
    "fit_setup_readiness",
]

FitDatasetWeightMode = Literal["implicit", "user"]
FitSolver = Literal["BDF", "Radau"]


def build_fit_local_admission(
    prepared_dsl: PreparedDslRequestSet,
    observables: Iterable[FitLocalObservable],
    parameters: Iterable[FitParameterVariable],
) -> AdmittedAlgebraProgram | None:
    """Admit the sole Fit-local Algebra program from selected rows."""

    observable_items = tuple(observables)
    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    scalar_names = tuple(
        dict.fromkeys(
            scalar_name
            for observable in observable_items
            for scalar_name in observable.scalar_names
        )
    )
    for scalar_name in scalar_names:
        parameter = parameter_by_name.get(scalar_name)
        if parameter is None:
            raise ValueError(
                f"Observable scalar {scalar_name!r} has no Fit parameter row."
            )
        if (
            parameter.scientific_category != "finite_scalar"
            or not parameter.allow_exact_zero
            or parameter.scope == FIT_SCOPE_CENTRED_FIXED_EFFECTS
        ):
            raise ValueError(
                f"Fit-local scalar {scalar_name!r} lost finite-scalar authority."
            )
    if not observable_items:
        return None
    return admit_fit_local_algebra(
        base=prepared_dsl,
        observables=observable_items,
        scalar_names=scalar_names,
    )


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
    """Return the canonical positive solver span for one Fit selection.

    A selected inclusive window may retain exactly one observation at its
    boundary (``t_start == t_end``).  That is valid data, but the simulation
    preparation contract requires a strictly positive span.  Widen only the
    execution span by one representable float; the selected observations and
    user-facing Data Range remain unchanged.  At the admitted upper endpoint,
    move the execution start back by one representable float instead.
    """

    start = _finite(t_start, "Fit execution start")
    end = _finite(t_end, "Fit execution end")
    if start < 0.0 or end < 0.0 or end > TIME_MAX_S or start > TIME_MAX_S:
        raise ValueError("Fit execution span must use admitted nonnegative seconds.")
    if end > start:
        return start, end
    if end < start:
        raise ValueError("Fit execution end must not precede its start.")
    widened_end = math.nextafter(start, math.inf)
    if widened_end <= TIME_MAX_S:
        return start, widened_end
    widened_start = math.nextafter(start, -math.inf)
    if widened_start < 0.0 or not widened_start < end:
        raise ValueError("Fit boundary has no representable positive execution span.")
    return widened_start, end


def _validated_dataset_weight_mode(
    value: object,
) -> FitDatasetWeightMode:
    if value == "implicit":
        return "implicit"
    if value == "user":
        return "user"
    raise ValueError("Dataset weight mode must be implicit or user.")


@dataclass(frozen=True, slots=True)
class FitLocalScalarProvenance:
    """Immutable admission/start provenance for a Fit-local scalar.

    The scalar is born inside one Fit capture, not in project Parameters.
    Keeping that fact on the typed parameter row prevents later reporting or
    Apply code from fabricating a project-captured default for it.
    """

    observable_name: str
    admission_scope: Literal["shared", "per_dataset"]
    admitted_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "observable_name", _required(self.observable_name, "Fit-local observable name"))
        scope = str(self.admission_scope or "").strip().lower()
        if scope not in {"shared", "per_dataset"}:
            raise ValueError("Fit-local scalar admission scope must be Shared or Per-dataset.")
        object.__setattr__(self, "admission_scope", scope)
        object.__setattr__(
            self,
            "admitted_value",
            admit_gui_parameter_value(
                self.admitted_value,
                scientific_category="finite_scalar",
                allow_exact_zero=True,
                field="Fit-local admitted value",
            ),
        )


@dataclass(frozen=True, slots=True)
class FitObservationSnapshot:
    raw_observation_id: str
    raw_column: str
    model_target: str
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


def _admit_fit_parameter_configuration(
    configuration: FitParameterConfiguration,
    *,
    parameter_name: str,
    scientific_category: GuiParameterScientificCategory,
    allow_exact_zero: bool,
) -> FitParameterConfiguration:
    """Admit one complete scope candidate using its retained category."""

    if not isinstance(configuration, FitParameterConfiguration):
        raise TypeError("Fit parameter configuration must be typed.")

    def admit(value: object, field: str) -> float:
        return admit_gui_parameter_value(
            value,
            scientific_category=scientific_category,
            allow_exact_zero=allow_exact_zero,
            field=f"{parameter_name} {field}",
        )

    value = admit(configuration.value, "value")
    minimum = admit(configuration.minimum, "minimum")
    maximum = admit(configuration.maximum, "maximum")
    dataset_values = tuple(
        (dataset_id, admit(item, f"value for dataset {dataset_id}"))
        for dataset_id, item in configuration.dataset_values
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
    fit_local_provenance: FitLocalScalarProvenance | None = None

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
            scientific_category=scientific_category,
            allow_exact_zero=self.allow_exact_zero,
        )
        per_dataset_config = _admit_fit_parameter_configuration(
            self.per_dataset_config,
            parameter_name=self.name,
            scientific_category=scientific_category,
            allow_exact_zero=self.allow_exact_zero,
        )
        if self.fit_local_provenance is not None and not isinstance(
            self.fit_local_provenance,
            FitLocalScalarProvenance,
        ):
            raise TypeError("Fit-local scalar provenance must be typed.")
        if self.fit_local_provenance is not None and (
            scientific_category != "finite_scalar"
            or not self.allow_exact_zero
        ):
            raise ValueError(
                "Fit-local scalars require arbitrary finite-scalar admission."
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
        initial = validate_concentration(
            _finite(self.initial, f"{species} initial"),
            field=f"{species} initial",
        )
        minimum = validate_concentration(
            _finite(self.minimum, f"{species} minimum"),
            field=f"{species} minimum",
        )
        maximum = validate_concentration(
            _finite(self.maximum, f"{species} maximum"),
            field=f"{species} maximum",
        )
        if minimum >= maximum or not minimum <= initial <= maximum:
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
        if not parameter.selected_config.fit or parameter.fit_local_provenance is not None:
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


@dataclass(frozen=True, slots=True)
class FitPreparedDatasetConfig:
    """One admitted dataset/grid solver policy completed before dispatch."""

    dataset_id: str
    retained_times: tuple[float, ...]
    execution_t_start: float
    execution_t_end: float
    fit_sampling_policy: str
    fit_sampling_identity: tuple[tuple[str, object], ...]
    solver_config: PreparedSolverConfig

    def __post_init__(self) -> None:
        dataset_id = _required(self.dataset_id, "prepared Fit dataset ID")
        retained_times = tuple(float(value) for value in self.retained_times)
        if (
            not retained_times
            or any(not math.isfinite(value) for value in retained_times)
            or any(right <= left for left, right in zip(retained_times, retained_times[1:]))
        ):
            raise ValueError("Prepared Fit dataset times must be finite and ordered.")
        if not isinstance(self.solver_config, PreparedSolverConfig):
            raise TypeError("Prepared Fit dataset requires PreparedSolverConfig.")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "retained_times", retained_times)


@dataclass(frozen=True, slots=True)
class FitPhysicalCoordinatePolicy:
    """Retained physical admission policy for one Fit scalar coordinate."""

    parameter_name: str
    dataset_id: str | None
    scientific_category: GuiParameterScientificCategory
    allow_exact_zero: bool
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_name", _required(self.parameter_name, "physical Fit coordinate"))
        if self.dataset_id is not None:
            object.__setattr__(self, "dataset_id", _required(self.dataset_id, "physical Fit dataset ID"))
        if self.scientific_category not in {"rate", "equilibrium_constant", "finite_scalar"}:
            raise ValueError("Physical Fit coordinate category is invalid.")
        admit_gui_parameter_value(
            self.minimum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            field=f"{self.parameter_name} retained minimum",
        )
        admit_gui_parameter_value(
            self.maximum,
            scientific_category=self.scientific_category,
            allow_exact_zero=self.allow_exact_zero,
            field=f"{self.parameter_name} retained maximum",
        )


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
    physical_coordinate_policies: tuple[FitPhysicalCoordinatePolicy, ...]
    ordinary_optimizer_layout: FitParameterLayout | None
    ordinary_centred_variables: tuple[FitCentredFixedEffectsParamSpec, ...]


@dataclass(frozen=True, slots=True)
class FitExecutionRequest:
    origin: FitOriginSnapshot
    solver: FitSolver
    fit_local_admission: AdmittedAlgebraProgram | None
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
    global_initials: tuple[FitInitialGlobalVariable, ...]
    initial_scopes: tuple[tuple[str, str], ...]
    defaults: FitDefaults
    dataset_weight_mode: FitDatasetWeightMode
    profile_likelihood: ProfileLikelihoodConfig | None = None
    residual_bootstrap: ResidualBootstrapConfig | None = None
    dispatch: _FitDispatchCompilation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if not isinstance(self.defaults, FitDefaults):
            raise TypeError("defaults must be FitDefaults.")
        object.__setattr__(self, "solver", validate_fit_solver(self.solver))
        validate_fit_defaults(self.defaults)
        object.__setattr__(
            self,
            "dataset_weight_mode",
            _validated_dataset_weight_mode(self.dataset_weight_mode),
        )
        admission = self.fit_local_admission
        if admission is not None:
            if not isinstance(admission, AdmittedAlgebraProgram):
                raise TypeError(
                    "Fit request local admission must be AdmittedAlgebraProgram."
                )
            if admission.base_analysis != self.origin.prepared_dsl.analysis:
                raise ValueError(
                    "Fit request local admission no longer matches base science."
                )
        raw_datasets = tuple(self.datasets)
        raw_parameters = tuple(self.parameters)
        raw_initials = tuple(self.initials)
        raw_global_initials = tuple(self.global_initials)
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
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initials", initials)
        object.__setattr__(self, "global_initials", global_initials)
        object.__setattr__(self, "initial_scopes", initial_scopes)
        local_scalar_names = set(() if admission is None else admission.scalar_names)
        parameter_by_name = {item.name: item for item in parameters}
        for scalar_name in local_scalar_names:
            parameter = parameter_by_name.get(scalar_name)
            if (
                parameter is None
                or parameter.fit_local_provenance is None
                or parameter.scientific_category != "finite_scalar"
                or not parameter.allow_exact_zero
            ):
                raise ValueError(
                    f"Fit-local scalar {scalar_name!r} lost its admitted row authority."
                )
        unexpected_local_rows = sorted(
            item.name
            for item in parameters
            if item.fit_local_provenance is not None
            and item.name not in local_scalar_names
        )
        if unexpected_local_rows:
            raise ValueError(
                "Fit request retained unowned Fit-local scalar row(s): "
                + ", ".join(unexpected_local_rows)
            )
        _validate_advanced_analysis_setup(
            datasets=datasets,
            parameters=parameters,
            profile_likelihood=self.profile_likelihood,
            residual_bootstrap=self.residual_bootstrap,
        )
        object.__setattr__(
            self,
            "dispatch",
            _compile_fit_dispatch(self),
        )


@dataclass(frozen=True, slots=True)
class FitSetupSnapshot:
    """One complete typed Fit setup at launch or terminal completion."""

    origin: FitOriginSnapshot
    solver: FitSolver
    datasets: tuple[FitDatasetLocalSettings, ...]
    parameters: tuple[FitParameterVariable, ...]
    initials: tuple[FitInitialVariable, ...]
    global_initials: tuple[FitInitialGlobalVariable, ...]
    initial_scopes: tuple[tuple[str, str], ...]
    defaults: FitDefaults
    local_observables: tuple[FitLocalObservable, ...]
    dataset_weight_mode: FitDatasetWeightMode
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
        object.__setattr__(self, "global_initials", global_initials)
        object.__setattr__(self, "initial_scopes", initial_scopes)
        object.__setattr__(self, "local_observables", local_observables)
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
        origin=origin,
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
    observation_targets = canonical_observation_target_names(tab)
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
                initial_values=dict(
                    canonical_initial_values(tab, initial_set.set_id)
                ),
                mechanism_species=mechanism_species,
                observation_targets=observation_targets,
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
        for raw_observation_id, model_target in mapping:
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
                model_target=target,
                t=tuple(float(observation.t[index]) for index in indices),
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
    global_initials: Iterable[FitInitialGlobalVariable],
    initial_scopes: Iterable[tuple[str, str]],
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
    global_initial_items = tuple(global_initials)
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
        try:
            observations = selected_observations(mapped, local)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
        if not observations:
            return f"Dataset {mapped.display_name!r} has no selected observations."

    scopes = dict(initial_scopes)
    included_ids = {item.dataset_id for item in included}
    has_fitted_initial = any(
        item.fit
        and scopes.get(item.species) == "local"
        and any(
            mapped.set_id == item.set_id and mapped.dataset_id in included_ids
            for mapped in origin.mapped_datasets
        )
        for item in initial_items
    ) or any(
        item.fit
        and scopes.get(item.species) == "global"
        and any(
            mapped.dataset_id in included_ids
            and any(
                species == item.species
                for species, _value in mapped.initial_values
            )
            for mapped in origin.mapped_datasets
        )
        for item in global_initial_items
    )
    if not any(item.selected_config.fit for item in parameter_items) and not has_fitted_initial:
        return "Select at least one parameter or initial concentration to fit."
    return None


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
    mechanism_species = canonical_species_names(tab)
    observation_targets = canonical_observation_target_names(tab)
    mapped_by_id = {
        item.dataset_id: item for item in request.origin.mapped_datasets
    }
    allowed_observable_names = {
        item.name
        for item in (
            ()
            if request.fit_local_admission is None
            else request.fit_local_admission.observables
        )
    } | set(fit_observable_names(request.origin.prepared_dsl))
    captured_link_facts = tuple(
        sorted(
            (item.dataset_id, bool(item.included_at_launch))
            for item in request.origin.mapped_datasets
        )
    )
    current_link_facts = tuple(
        sorted(
            (dataset_id, bool(link.include_in_fit))
            for dataset_id, link in tab.dataset_links
            if link.set_id is not None
        )
    )
    if current_link_facts != captured_link_facts:
        raise ValueError(
            "Fit dataset links or inclusion changed after capture."
        )
    for local in request.datasets:
        captured = mapped_by_id.get(local.dataset_id)
        if captured is None:
            raise ValueError(
                f"Dataset {local.dataset_id!r} is not part of the captured Fit roster."
            )
        link = tab.dataset_link(local.dataset_id)
        if link is None:
            raise ValueError(
                f"Dataset {local.dataset_id!r} is no longer linked in the active tab."
            )
        if bool(link.include_in_fit) != bool(captured.included_at_launch):
            raise ValueError(
                f"Dataset {captured.display_name!r} changed Fit inclusion after capture."
            )
        if not local.included:
            continue
        dataset = dataset_foundation.validate_raw_dataset_record(
            document.dataset(local.dataset_id)
        )
        initial_set = tab.initial_condition_set(link.set_id)
        validated = dataset_foundation.validate_fit_inclusion(
            dataset=dataset,
            set_id=link.set_id,
            observation_mapping=link.observation_mapping,
            initial_values=dict(
                canonical_initial_values(tab, initial_set.set_id)
            ),
            mechanism_species=mechanism_species,
            observation_targets=observation_targets,
            resolved_t_start=resolve_fit_t_start(
                tab,
                initial_set.set_id,
                label=f"Fit inclusion for dataset {dataset.dataset_id!r}",
            ),
        )
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


def _compile_fit_dispatch(
    request: FitExecutionRequest,
) -> _FitDispatchCompilation:
    """Complete all static Fit facts before a worker can be constructed."""

    mapped_by_id = {item.dataset_id: item for item in request.origin.mapped_datasets}
    request_ids = tuple(item.dataset_id for item in request.datasets)
    if len(request_ids) != len(set(request_ids)) or set(request_ids) != set(mapped_by_id):
        raise ValueError("Fit dispatch dataset roster does not match its capture.")
    included = tuple(item for item in request.datasets if item.included)
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
    physical_policies: list[FitPhysicalCoordinatePolicy] = []
    for parameter in request.parameters:
        config = parameter.selected_config
        dataset_values = dict(config.dataset_values)
        target_ids = (None,) if parameter.scope == "shared" else included_ids
        for dataset_id in target_ids:
            physical_policies.append(
                FitPhysicalCoordinatePolicy(
                    parameter_name=parameter.name,
                    dataset_id=dataset_id,
                    scientific_category=parameter.scientific_category,
                    allow_exact_zero=parameter.allow_exact_zero,
                    minimum=config.minimum,
                    maximum=config.maximum,
                )
            )
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
            if parameter.fit_local_provenance is not None:
                raise ValueError(
                    "Fit-local observable scalars cannot use centred fixed effects."
                )
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
    applicable_dataset_ids: dict[str, set[str]] = {}
    for mapped in request.origin.mapped_datasets:
        if mapped.dataset_id not in included_ids:
            continue
        datasets_for_set[mapped.set_id] = (
            *datasets_for_set.get(mapped.set_id, ()),
            mapped.dataset_id,
        )
        for species, _value in mapped.initial_values:
            applicable_dataset_ids.setdefault(species, set()).add(mapped.dataset_id)
    scopes = dict(request.initial_scopes)
    for initial in request.global_initials:
        if scopes.get(initial.species) != "global":
            continue
        applicable_ids = tuple(sorted(applicable_dataset_ids.get(initial.species, ())))
        if initial.fit and applicable_ids:
            coupled_variables.append(
                FitCoupledVariableParamSpec(
                    variable_id=f"initial:global:{initial.species}",
                    parameter_name=f"init:{initial.species}",
                    dataset_ids=applicable_ids,
                    initial=initial.initial,
                    minimum=initial.minimum,
                    maximum=initial.maximum,
                    log10=initial.log10,
                )
            )
        else:
            for dataset_id in applicable_ids:
                fixed_by_dataset[dataset_id][f"init:{initial.species}"] = initial.initial
    for initial in request.initials:
        if scopes.get(initial.species) != "local":
            continue
        target_ids = datasets_for_set.get(initial.set_id, ())
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
    science = request.origin.scientific_settings
    prepared_datasets: list[FitPreparedDatasetConfig] = []
    for local in included:
        selected = selected_observations(mapped_by_id[local.dataset_id], local)
        retained_times = tuple(
            sorted({float(value) for item in selected for value in item.t})
        )
        execution_t_start, execution_t_end = fit_execution_span(
            local.t_start, local.data_range_max
        )
        grid_points = max(100, len(retained_times)) if local.parametric is not None else 100
        policy = (
            "parametric_regular_grid_union_retained_endpoints"
            if local.parametric is not None
            else "fit_internal_regular_grid_union_retained_endpoints"
        )
        identity = (
            ("solver", request.solver),
            ("t_start", execution_t_start),
            ("t_end", execution_t_end),
            ("regular_grid_count", grid_points),
            ("retained_observation_times", retained_times),
            ("union_endpoints", (execution_t_start, execution_t_end)),
        )
        try:
            # Complete the regular-grid admission before Fit worker creation;
            # exact retained observations are unioned later without a row cap.
            build_time_grid(execution_t_start, execution_t_end, {"N": grid_points})
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Fit regular grid for dataset {local.dataset_id!r} "
                "cannot be realized on this time span."
            ) from exc
        prepared_datasets.append(
            FitPreparedDatasetConfig(
                dataset_id=local.dataset_id,
                retained_times=retained_times,
                execution_t_start=execution_t_start,
                execution_t_end=execution_t_end,
                fit_sampling_policy=policy,
                fit_sampling_identity=identity,
                solver_config=build_solver_config(
                    solver_input=request.solver,
                    rtol=science.rtol,
                    atol=science.atol,
                    grid={"N": grid_points},
                    use_analytic_jacobian=science.analytic_jacobian_policy,
                ),
            )
        )
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
                for parameter in request.parameters
                if not parameter.name.startswith(INITIAL_PREFIX)
            )
        ),
        physical_coordinate_policies=tuple(physical_policies),
        ordinary_optimizer_layout=layout,
    )


def build_fit_execution_request(
    *,
    origin: FitOriginSnapshot,
    solver: FitSolver = "BDF",
    local_observables: Iterable[FitLocalObservable],
    fit_local_admission: AdmittedAlgebraProgram | None = None,
    datasets: Iterable[FitDatasetLocalSettings],
    parameters: Iterable[FitParameterVariable],
    initials: Iterable[FitInitialVariable],
    global_initials: Iterable[FitInitialGlobalVariable],
    initial_scopes: Iterable[tuple[str, str]],
    dataset_weight_mode: FitDatasetWeightMode,
    defaults: FitDefaults | None = None,
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    residual_bootstrap: ResidualBootstrapConfig | None = None,
) -> FitExecutionRequest:
    """Validate a complete owner-built Fit request at the execution boundary."""
    observable_items = tuple(local_observables)
    dataset_items = tuple(datasets)
    parameter_items = tuple(parameters)
    initial_items = tuple(initials)
    global_initial_items = tuple(global_initials)
    initial_scope_items = tuple(initial_scopes)
    readiness = fit_setup_readiness(
        origin=origin,
        datasets=dataset_items,
        parameters=parameter_items,
        initials=initial_items,
        global_initials=global_initial_items,
        initial_scopes=initial_scope_items,
    )
    if readiness is not None:
        raise ValueError(readiness)
    admission = fit_local_admission
    if admission is None:
        admission = build_fit_local_admission(
            origin.prepared_dsl,
            observable_items,
            parameter_items,
        )
    elif not isinstance(admission, AdmittedAlgebraProgram):
        raise TypeError("fit_local_admission must be AdmittedAlgebraProgram or None.")
    if admission is not None and tuple(admission.observables) != observable_items:
        raise ValueError("Fit-local admission declarations do not match request.")
    request = FitExecutionRequest(
        origin=origin,
        solver=solver,
        fit_local_admission=admission,
        datasets=dataset_items,
        parameters=parameter_items,
        initials=initial_items,
        global_initials=global_initial_items,
        initial_scopes=initial_scope_items,
        defaults=defaults or origin.fit_defaults,
        dataset_weight_mode=dataset_weight_mode,
        profile_likelihood=profile_likelihood,
        residual_bootstrap=residual_bootstrap,
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
    retained_dataset_ids = set(mapped_by_id)
    parameter_names = [item.name for item in request.parameters]
    expected_names = [item.name for item in origin.parameter_defaults]
    local_scalar_names = {
        scalar_name
        for scalar_name in (
            ()
            if request.fit_local_admission is None
            else request.fit_local_admission.scalar_names
        )
    }
    if (
        len(parameter_names) != len(set(parameter_names))
        or set(parameter_names) != set(expected_names) | local_scalar_names
    ):
        raise ValueError(
            "Fit request must contain each captured and Fit-local scalar exactly once."
        )
    captured_by_name = {
        item.name: item for item in request.origin.parameter_defaults
    }
    for parameter in request.parameters:
        captured = captured_by_name.get(parameter.name)
        if captured is not None and (
            parameter.family != captured.family
            or parameter.role != captured.role
            or parameter.scientific_category != captured.scientific_category
            or parameter.allow_exact_zero != captured.allow_exact_zero
        ):
            raise ValueError(
                f"Parameter {parameter.name!r} changed its accepted scientific "
                "authority."
            )
        if captured is None and (
            parameter.name not in local_scalar_names
            or parameter.fit_local_provenance is None
            or parameter.scientific_category != "finite_scalar"
            or not parameter.allow_exact_zero
        ):
            raise ValueError(
                f"Fit-local scalar {parameter.name!r} lost finite-scalar authority."
            )
        local_ids = {
        dataset_id for dataset_id, _value in parameter.selected_config.dataset_values
        }
        if parameter.scope in {"per_dataset", FIT_SCOPE_CENTRED_FIXED_EFFECTS} and local_ids != retained_dataset_ids:
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
        if local.included and local.data_range_max <= local.t_start:
            raise ValueError(
                f"Dataset {mapped_by_id[local.dataset_id].display_name!r} "
                "Fit interval requires Data Range maximum to be greater "
                "than Simulation start."
            )
    return request

"""Immutable project science values and their canonical validation."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence, TypeAlias

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.dsl_parse import (
    DslAnalysis,
    DslInitialConditionAdmission,
    PreparedDslRequestSet,
    prepare_dsl_request_set,
)
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.units import UnitsModel

__all__ = [
    "DatasetLink",
    "FitDefaults",
    "FrozenJsonArray",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "InitialConditionSet",
    "MechanismSourceCollision",
    "MechanismSourceFailure",
    "MechanismSourceReconciliation",
    "MechanismTabDocument",
    "ObservationSeries",
    "ProjectDocument",
    "RawDatasetRecord",
    "SimulationExecutionSettings",
    "TabScientificSettings",
    "default_fit_defaults",
    "default_project_document",
    "default_simulation_execution_settings",
    "default_tab_scientific_settings",
    "freeze_json_value",
    "fresh_mechanism_tab",
    "new_dataset_id",
    "new_project_id",
    "new_set_id",
    "new_tab_id",
    "next_mechanism_name",
    "next_set_name",
    "reconcile_tab_mechanism_source",
    "resolve_simulation_execution_settings",
    "thaw_json_value",
    "validate_raw_observation_id",
    "validate_project_document",
    "validate_simulation_execution_settings",
    "validate_tab_scientific_settings",
]

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class FrozenJsonArray:
    items: tuple["FrozenJsonValue", ...]


@dataclass(frozen=True, slots=True)
class FrozenJsonObject:
    items: tuple[tuple[str, "FrozenJsonValue"], ...]


FrozenJsonValue: TypeAlias = JsonScalar | FrozenJsonArray | FrozenJsonObject

_SIMULATION_POINT_RANGE = (10, 100_000)
_TEMPERATURE_RANGE_K = (0.1, 10_000.0)
_SOLVERS = ("Radau", "BDF")
_FITTING_METHODS = ("trf", "dogbox", "differential_evolution")
_FITTING_MAX_NFEV_RANGE = (10, 10_000)
_FITTING_SEED_RANGE = (0, 999_999)
_SCIENTIFIC_VALUE_MAX = 1e300


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def new_tab_id() -> str:
    return _new_id("tab")


def new_set_id() -> str:
    return _new_id("set")


def new_dataset_id() -> str:
    return _new_id("dataset")


def new_project_id() -> str:
    return _new_id("project")


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank.")
    return normalized


def _require_id(value: object, label: str) -> str:
    normalized = _require_name(value, label)
    if normalized != value:
        raise ValueError(f"{label} must not contain edge whitespace.")
    return normalized


def validate_raw_observation_id(
    value: object,
    label: str = "raw observation ID",
) -> str:
    """Require one already-allocated opaque raw observation identity."""

    if type(value) is not str:
        raise TypeError(f"{label} must be a str.")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonblank and edge-normalized.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool.")
    return bool(value)


def _require_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{label} must be an int.")
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _require_finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{label} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} must be at most {maximum}.")
    return parsed


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Normalize a JSON-compatible value into an immutable representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return int(value)
    if type(value) is float:
        return _require_finite(value, "JSON number")
    if isinstance(value, Mapping):
        items: list[tuple[str, FrozenJsonValue]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            items.append((key, freeze_json_value(item)))
        items.sort(key=lambda item: item[0])
        return FrozenJsonObject(tuple(items))
    if isinstance(value, (list, tuple)):
        return FrozenJsonArray(tuple(freeze_json_value(item) for item in value))
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")


def thaw_json_value(value: FrozenJsonValue) -> object:
    if isinstance(value, FrozenJsonObject):
        return {str(key): thaw_json_value(item) for key, item in value.items}
    if isinstance(value, FrozenJsonArray):
        return [thaw_json_value(item) for item in value.items]
    return value


@dataclass(frozen=True, slots=True)
class SimulationExecutionSettings:
    """Complete per-target execution value shared by Run, Preview, and Fit."""

    solver: str = "BDF"
    t_start: float = 0.0
    t_end: float = 10.0
    integration_points: int = 100


@dataclass(frozen=True, slots=True)
class TabScientificSettings:
    """Scientific settings that are shared by every execution in one tab."""

    temperature: float = 298.15
    rtol: float = 1e-6
    atol: float = 1e-12
    sparse_jacobian_policy: bool = True
    wegscheider_policy: bool = True


@dataclass(frozen=True, slots=True)
class FitDefaults:
    method: str = "trf"
    max_nfev: int = 1000
    ftol: float = 1e-10
    xtol: float = 1e-10
    use_seed: bool = True
    seed: int = 42


@dataclass(frozen=True, slots=True)
class InitialConditionSet:
    set_id: str
    name: str
    values: tuple[tuple[str, float], ...] = ()

    def values_dict(self) -> dict[str, float]:
        return dict(self.values)

    def with_values(
        self,
        values: Mapping[str, Real] | Iterable[tuple[str, Real]],
    ) -> "InitialConditionSet":
        items = values.items() if isinstance(values, Mapping) else values
        return replace(
            self,
            values=tuple(
                (
                    _require_name(species, "initial-condition species name"),
                    _require_finite(
                        value,
                        f"initial value for {species}",
                        minimum=0.0,
                    ),
                )
                for species, value in items
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationSeries:
    raw_observation_id: str
    column_name: str
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_raw_observation_id(self.raw_observation_id)


@dataclass(frozen=True, slots=True)
class RawDatasetRecord:
    dataset_id: str
    display_name: str
    original_filename: str
    original_format: str
    exact_source_bytes: bytes
    source_metadata: FrozenJsonValue
    observations: tuple[ObservationSeries, ...]


@dataclass(frozen=True, slots=True)
class DatasetLink:
    set_id: str | None = None
    observation_mapping: tuple[tuple[str, str], ...] = ()
    include_in_fit: bool = False


@dataclass(frozen=True, slots=True)
class MechanismTabDocument:
    tab_id: str
    name: str
    mechanism_source: MechanismAuthoringSource
    notes: str
    initial_condition_sets: tuple[InitialConditionSet, ...]
    run_defaults: SimulationExecutionSettings
    scientific_settings: TabScientificSettings
    run_overrides_by_set_id: tuple[
        tuple[str, SimulationExecutionSettings], ...
    ] = ()
    dataset_links: tuple[tuple[str, DatasetLink], ...] = ()
    prepared_dsl: PreparedDslRequestSet | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def dsl_analysis(self) -> DslAnalysis | None:
        prepared = self.prepared_dsl
        return None if prepared is None else prepared.analysis

    def initial_condition_set(self, set_id: str) -> InitialConditionSet:
        target = str(set_id)
        for item in self.initial_condition_sets:
            if item.set_id == target:
                return item
        raise KeyError(f"Unknown initial-condition set ID: {target!r}.")

    def dataset_link(self, dataset_id: str) -> DatasetLink | None:
        target = str(dataset_id)
        for current_id, link in self.dataset_links:
            if current_id == target:
                return link
        return None

    def resolved_execution_settings(
        self,
        set_id: str,
    ) -> SimulationExecutionSettings:
        return resolve_simulation_execution_settings(self, set_id)

    def add_initial_condition_set(
        self,
        initial_set: InitialConditionSet,
    ) -> "MechanismTabDocument":
        if not isinstance(initial_set, InitialConditionSet):
            raise TypeError("initial_set must be an InitialConditionSet.")
        return self.with_initial_condition_sets(
            (*self.initial_condition_sets, initial_set)
        )

    def replace_initial_condition_set(
        self,
        initial_set: InitialConditionSet,
    ) -> "MechanismTabDocument":
        if not isinstance(initial_set, InitialConditionSet):
            raise TypeError("initial_set must be an InitialConditionSet.")
        if not any(
            item.set_id == initial_set.set_id
            for item in self.initial_condition_sets
        ):
            raise KeyError(
                f"Unknown initial-condition set ID: {initial_set.set_id!r}."
            )
        return self.with_initial_condition_sets(
            tuple(
                initial_set if item.set_id == initial_set.set_id else item
                for item in self.initial_condition_sets
            )
        )

    def rename_initial_condition_set(
        self,
        set_id: str,
        name: str,
    ) -> "MechanismTabDocument":
        initial_set = self.initial_condition_set(set_id)
        return self.replace_initial_condition_set(
            replace(
                initial_set,
                name=_require_name(name, "initial-condition set name"),
            )
        )

    def with_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> "MechanismTabDocument":
        validate_simulation_execution_settings(settings)
        return replace(self, run_defaults=settings)

    def with_run_override(
        self,
        set_id: str,
        settings: SimulationExecutionSettings | None,
    ) -> "MechanismTabDocument":
        target = self.initial_condition_set(set_id).set_id
        overrides = dict(self.run_overrides_by_set_id)
        if settings is None:
            overrides.pop(target, None)
        else:
            validate_simulation_execution_settings(settings)
            overrides[target] = settings
        return replace(
            self,
            run_overrides_by_set_id=tuple(overrides.items()),
        )

    def with_initial_condition_sets(
        self,
        sets: Sequence[InitialConditionSet],
    ) -> "MechanismTabDocument":
        candidate_sets = tuple(sets)
        retained_ids = {item.set_id for item in candidate_sets}
        run_overrides = tuple(
            (set_id, settings)
            for set_id, settings in self.run_overrides_by_set_id
            if set_id in retained_ids
        )
        links = tuple(
            (
                dataset_id,
                link
                if link.set_id in retained_ids or link.set_id is None
                else replace(link, set_id=None, include_in_fit=False),
            )
            for dataset_id, link in self.dataset_links
        )
        return replace(
            self,
            initial_condition_sets=candidate_sets,
            run_overrides_by_set_id=run_overrides,
            dataset_links=links,
        )

    def without_initial_condition_sets(
        self,
        set_ids: Iterable[str],
    ) -> "MechanismTabDocument":
        targets = {str(set_id) for set_id in set_ids}
        if not targets:
            return self
        retained = tuple(
            item
            for item in self.initial_condition_sets
            if item.set_id not in targets
        )
        if len(retained) == len(self.initial_condition_sets):
            return self
        if not retained:
            retained = (
                InitialConditionSet(
                    set_id=new_set_id(),
                    name="set1",
                    values=(),
                ),
            )
        return self.with_initial_condition_sets(retained)

    def with_dataset_link(
        self,
        dataset_id: str,
        link: DatasetLink,
        *,
        override_set_occupant: bool = False,
    ) -> "MechanismTabDocument":
        target_dataset_id = _require_id(dataset_id, "dataset ID")
        if not isinstance(link, DatasetLink):
            raise TypeError("link must be a DatasetLink.")
        links = dict(self.dataset_links)
        if link.set_id is not None:
            occupant = next(
                (
                    linked_dataset_id
                    for linked_dataset_id, existing in links.items()
                    if linked_dataset_id != target_dataset_id
                    and existing.set_id == link.set_id
                ),
                None,
            )
            if occupant is not None and not override_set_occupant:
                raise ValueError(
                    f"Initial-condition set {link.set_id!r} is already mapped "
                    f"to dataset {occupant!r}."
                )
            if occupant is not None:
                old = links[occupant]
                links[occupant] = replace(
                    old,
                    set_id=None,
                    include_in_fit=False,
                )
        links[target_dataset_id] = link
        return replace(self, dataset_links=tuple(links.items()))

    def without_dataset(self, dataset_id: str) -> "MechanismTabDocument":
        target = str(dataset_id)
        return replace(
            self,
            dataset_links=tuple(
                (current_id, link)
                for current_id, link in self.dataset_links
                if current_id != target
            ),
        )

    def has_canonical_content(
        self,
        *,
        fresh_run_defaults: SimulationExecutionSettings | None = None,
        fresh_scientific_settings: TabScientificSettings | None = None,
    ) -> bool:
        expected_run_defaults = (
            default_simulation_execution_settings()
            if fresh_run_defaults is None
            else fresh_run_defaults
        )
        expected_scientific_settings = (
            default_tab_scientific_settings()
            if fresh_scientific_settings is None
            else fresh_scientific_settings
        )
        if self.mechanism_source != MechanismAuthoringSource():
            return True
        if self.notes.strip():
            return True
        if (
            self.run_defaults != expected_run_defaults
            or self.scientific_settings != expected_scientific_settings
            or self.run_overrides_by_set_id
        ):
            return True
        if self.dataset_links:
            return True
        if len(self.initial_condition_sets) != 1:
            return True
        only = self.initial_condition_sets[0]
        return only.name != "set1" or bool(only.values)


@dataclass(frozen=True, slots=True)
class ProjectDocument:
    project_id: str
    datasets: tuple[RawDatasetRecord, ...]
    tabs: tuple[MechanismTabDocument, ...]
    fit_defaults: FitDefaults

    def tab(self, tab_id: str) -> MechanismTabDocument:
        target = str(tab_id)
        for item in self.tabs:
            if item.tab_id == target:
                return item
        raise KeyError(f"Unknown mechanism tab ID: {target!r}.")

    def dataset(self, dataset_id: str) -> RawDatasetRecord:
        target = str(dataset_id)
        for item in self.datasets:
            if item.dataset_id == target:
                return item
        raise KeyError(f"Unknown dataset ID: {target!r}.")

    def replace_tab(self, tab: MechanismTabDocument) -> "ProjectDocument":
        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("tab must be a MechanismTabDocument.")
        if not any(item.tab_id == tab.tab_id for item in self.tabs):
            raise KeyError(f"Unknown mechanism tab ID: {tab.tab_id!r}.")
        return validate_project_document(
            replace(
                self,
                tabs=tuple(
                    tab if item.tab_id == tab.tab_id else item
                    for item in self.tabs
                ),
            )
        )

    def add_tab(
        self,
        *,
        run_defaults: SimulationExecutionSettings | None = None,
        scientific_settings: TabScientificSettings | None = None,
        name: str | None = None,
    ) -> "ProjectDocument":
        new_tab = fresh_mechanism_tab(
            name=name or next_mechanism_name(item.name for item in self.tabs),
            run_defaults=run_defaults,
            scientific_settings=scientific_settings,
        )
        return validate_project_document(replace(self, tabs=(*self.tabs, new_tab)))

    def rename_tab(self, tab_id: str, name: str) -> "ProjectDocument":
        tab = self.tab(tab_id)
        return self.replace_tab(
            replace(tab, name=_require_name(name, "tab name"))
        )

    def remove_tab(self, tab_id: str) -> "ProjectDocument":
        if len(self.tabs) <= 1:
            raise ValueError("The last mechanism tab cannot be closed.")
        target = str(tab_id)
        tabs = tuple(item for item in self.tabs if item.tab_id != target)
        if len(tabs) == len(self.tabs):
            raise KeyError(f"Unknown mechanism tab ID: {target!r}.")
        return validate_project_document(replace(self, tabs=tabs))

    def add_datasets(
        self,
        records: tuple[RawDatasetRecord, ...],
    ) -> "ProjectDocument":
        """Validate one complete dataset batch as one atomic proposal."""

        if type(records) is not tuple or any(
            not isinstance(record, RawDatasetRecord) for record in records
        ):
            raise TypeError(
                "records must be an exact tuple of RawDatasetRecord values."
            )
        if not records:
            return self
        return validate_project_document(
            replace(self, datasets=(*self.datasets, *records))
        )

    def replace_dataset(self, record: RawDatasetRecord) -> "ProjectDocument":
        if not isinstance(record, RawDatasetRecord):
            raise TypeError("record must be a RawDatasetRecord.")
        if not any(
            item.dataset_id == record.dataset_id for item in self.datasets
        ):
            raise KeyError(f"Unknown dataset ID: {record.dataset_id!r}.")
        return validate_project_document(
            replace(
                self,
                datasets=tuple(
                    record if item.dataset_id == record.dataset_id else item
                    for item in self.datasets
                ),
            )
        )

    def rename_dataset(
        self,
        dataset_id: str,
        display_name: str,
    ) -> "ProjectDocument":
        record = self.dataset(dataset_id)
        return self.replace_dataset(
            replace(
                record,
                display_name=_require_name(
                    display_name,
                    "dataset display name",
                ),
            )
        )

    def remove_dataset(self, dataset_id: str) -> "ProjectDocument":
        target = str(dataset_id)
        if not any(item.dataset_id == target for item in self.datasets):
            raise KeyError(f"Unknown dataset ID: {target!r}.")
        return validate_project_document(
            replace(
                self,
                datasets=tuple(
                    item
                    for item in self.datasets
                    if item.dataset_id != target
                ),
                tabs=tuple(tab.without_dataset(target) for tab in self.tabs),
            )
        )


@dataclass(frozen=True, slots=True)
class MechanismSourceReconciliation:
    proposed_document: ProjectDocument
    introduced_species: tuple[str, ...]
    removed_species: tuple[str, ...]
    inline_initial_conditions: DslInitialConditionAdmission
    materialized_set_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MechanismSourceCollision:
    """A pure pre-mutation request for exact named-set replacement consent."""

    existing_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.existing_names) is not tuple
            or not self.existing_names
            or any(
                type(name) is not str or not name or name != name.strip()
                for name in self.existing_names
            )
        ):
            raise ValueError(
                "Mechanism source collision names must be nonempty text."
            )


@dataclass(frozen=True, slots=True)
class MechanismSourceFailure:
    message: str


def default_simulation_execution_settings() -> SimulationExecutionSettings:
    return SimulationExecutionSettings()


def default_tab_scientific_settings() -> TabScientificSettings:
    return TabScientificSettings()


def default_fit_defaults() -> FitDefaults:
    return FitDefaults()


def fresh_mechanism_tab(
    *,
    name: str = "Mechanism 1",
    run_defaults: SimulationExecutionSettings | None = None,
    scientific_settings: TabScientificSettings | None = None,
) -> MechanismTabDocument:
    baseline_set = InitialConditionSet(
        set_id=new_set_id(),
        name="set1",
        values=(),
    )
    return MechanismTabDocument(
        tab_id=new_tab_id(),
        name=_require_name(name, "tab name"),
        mechanism_source=MechanismAuthoringSource(),
        notes="",
        initial_condition_sets=(baseline_set,),
        run_defaults=(
            run_defaults or default_simulation_execution_settings()
        ),
        scientific_settings=(
            scientific_settings or default_tab_scientific_settings()
        ),
    )


def default_project_document(
    *,
    run_defaults: SimulationExecutionSettings | None = None,
    scientific_settings: TabScientificSettings | None = None,
    fit_defaults: FitDefaults | None = None,
) -> ProjectDocument:
    return validate_project_document(
        ProjectDocument(
            project_id=new_project_id(),
            datasets=(),
            tabs=(
                fresh_mechanism_tab(
                    run_defaults=run_defaults,
                    scientific_settings=scientific_settings,
                ),
            ),
            fit_defaults=fit_defaults or default_fit_defaults(),
        )
    )


def _next_sequential_name(
    existing_names: Iterable[str],
    prefix: str,
) -> str:
    occupied = {
        str(name).strip().casefold()
        for name in existing_names
    }
    number = 1
    while f"{prefix}{number}".casefold() in occupied:
        number += 1
    return f"{prefix}{number}"


def next_mechanism_name(existing_names: Iterable[str]) -> str:
    return _next_sequential_name(existing_names, "Mechanism ")


def next_set_name(existing_names: Iterable[str]) -> str:
    return _next_sequential_name(existing_names, "set")


def _validate_unique(
    values: Iterable[object],
    label: str,
    *,
    key: Callable[[object], object] | None = None,
) -> None:
    seen: set[object] = set()
    transform = key or (lambda value: value)
    for value in values:
        normalized = transform(value)
        if normalized in seen:
            raise ValueError(
                f"{label} must be unique; duplicate {value!r}."
            )
        seen.add(normalized)


def validate_simulation_execution_settings(
    settings: SimulationExecutionSettings,
    label: str = "execution settings",
) -> SimulationExecutionSettings:
    if not isinstance(settings, SimulationExecutionSettings):
        raise TypeError(f"{label} must be SimulationExecutionSettings.")
    try:
        normalized_solver = normalize_solver_name(settings.solver)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}.solver must be one of {_SOLVERS!r}."
        ) from exc
    if normalized_solver != settings.solver or normalized_solver not in _SOLVERS:
        raise ValueError(
            f"{label}.solver must use canonical supported spelling."
        )
    start = _require_finite(
        settings.t_start,
        f"{label}.t_start",
        minimum=0.0,
    )
    end = _require_finite(settings.t_end, f"{label}.t_end")
    if end <= start:
        raise ValueError(f"{label}.t_end must be greater than t_start.")
    _require_int(
        settings.integration_points,
        f"{label}.integration_points",
        *_SIMULATION_POINT_RANGE,
    )
    return settings


def validate_tab_scientific_settings(
    settings: TabScientificSettings,
    label: str = "scientific settings",
) -> TabScientificSettings:
    if not isinstance(settings, TabScientificSettings):
        raise TypeError(f"{label} must be TabScientificSettings.")
    _require_finite(
        settings.temperature,
        f"{label}.temperature",
        minimum=_TEMPERATURE_RANGE_K[0],
        maximum=_TEMPERATURE_RANGE_K[1],
    )
    _require_finite(
        settings.rtol,
        f"{label}.rtol",
        maximum=_SCIENTIFIC_VALUE_MAX,
        strictly_positive=True,
    )
    _require_finite(
        settings.atol,
        f"{label}.atol",
        maximum=_SCIENTIFIC_VALUE_MAX,
        strictly_positive=True,
    )
    _require_bool(
        settings.sparse_jacobian_policy,
        f"{label}.sparse_jacobian_policy",
    )
    _require_bool(
        settings.wegscheider_policy,
        f"{label}.wegscheider_policy",
    )
    return settings


def resolve_simulation_execution_settings(
    tab: MechanismTabDocument,
    set_id: str,
) -> SimulationExecutionSettings:
    """Resolve one exact set to either its complete override or live default."""
    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    target = tab.initial_condition_set(set_id).set_id
    settings = dict(tab.run_overrides_by_set_id).get(
        target,
        tab.run_defaults,
    )
    return validate_simulation_execution_settings(settings)


def _validate_fit_defaults(defaults: FitDefaults) -> None:
    if not isinstance(defaults, FitDefaults):
        raise TypeError("fit_defaults must be FitDefaults.")
    if defaults.method not in _FITTING_METHODS:
        raise ValueError(
            f"fit_defaults.method must be one of {_FITTING_METHODS!r}."
        )
    _require_int(
        defaults.max_nfev,
        "fit_defaults.max_nfev",
        *_FITTING_MAX_NFEV_RANGE,
    )
    for field_name in ("ftol", "xtol"):
        _require_finite(
            getattr(defaults, field_name),
            f"fit_defaults.{field_name}",
            maximum=_SCIENTIFIC_VALUE_MAX,
            strictly_positive=True,
        )
    _require_bool(defaults.use_seed, "fit_defaults.use_seed")
    _require_int(
        defaults.seed,
        "fit_defaults.seed",
        *_FITTING_SEED_RANGE,
    )


def _normalized_dsl_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _normalized_mechanism_source(
    source: MechanismAuthoringSource,
) -> MechanismAuthoringSource:
    return MechanismAuthoringSource.from_parts(
        reactions_text=_normalized_dsl_text(source.reactions_text),
        state_network_dsl=_normalized_dsl_text(source.state_network_dsl),
    )


def _prepare_source_dsl(
    source: MechanismAuthoringSource,
    scientific_settings: TabScientificSettings,
) -> PreparedDslRequestSet | None:
    if not source.full_dsl.strip():
        return None
    return prepare_dsl_request_set(
        source.full_dsl,
        initials={},
        units=UnitsModel(
            temperature_K=scientific_settings.temperature,
        )
    )


def _prepared_dsl_matches_tab(tab: MechanismTabDocument) -> bool:
    prepared = tab.prepared_dsl
    if prepared is None:
        return not tab.mechanism_source.full_dsl.strip()
    if type(prepared) is not PreparedDslRequestSet:
        return False
    analysis = prepared.analysis
    if analysis.normalized_text != _normalized_dsl_text(
        tab.mechanism_source.full_dsl
    ):
        return False
    analysis_temperature = analysis.structure.get("temperature_K")
    return (
        type(analysis_temperature) in {int, float}
        and float(analysis_temperature)
        == float(tab.scientific_settings.temperature)
    )


def reconcile_tab_mechanism_source(
    document: ProjectDocument,
    tab_id: str,
    visible_source: MechanismAuthoringSource,
    *,
    replace_initial_condition_names: Iterable[str] = (),
) -> (
    MechanismSourceReconciliation
    | MechanismSourceCollision
    | MechanismSourceFailure
):
    """Purely validate and reconcile visible source into one atomic proposal."""
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    if not isinstance(visible_source, MechanismAuthoringSource):
        raise TypeError("visible_source must be MechanismAuthoringSource.")
    try:
        tab = document.tab(tab_id)
        normalized_visible = _normalized_mechanism_source(visible_source)
        admitted_dsl = _prepare_source_dsl(
            normalized_visible,
            tab.scientific_settings,
        )
        admission = (
            DslInitialConditionAdmission(
                materialized_source=normalized_visible.full_dsl
            )
            if admitted_dsl is None
            else admitted_dsl.analysis.inline_initial_conditions
        )
        new_species = (
            ()
            if admitted_dsl is None
            else admitted_dsl.analysis.species_names
        )
        old_species = (
            ()
            if tab.dsl_analysis is None
            else tab.dsl_analysis.species_names
        )
        new_species_set = set(new_species)
        old_species_set = set(old_species)
        introduced = tuple(
            species
            for species in new_species
            if species not in old_species_set
        )
        removed = tuple(
            species
            for species in old_species
            if species not in new_species_set
        )

        reconciled_sets: list[InitialConditionSet] = []
        for initial_set in tab.initial_condition_sets:
            values = {
                species: value
                for species, value in initial_set.values
                if species in new_species_set
            }
            for species in introduced:
                values[species] = 0.0
            reconciled_sets.append(
                replace(initial_set, values=tuple(values.items()))
            )
        existing_by_name = {
            item.name.casefold(): item for item in reconciled_sets
        }
        authorized_names = {
            _require_name(
                name,
                "authorized initial-condition replacement name",
            ).casefold()
            for name in replace_initial_condition_names
        }
        declared_named = tuple(admission.named_declarations)
        collisions = tuple(
            existing_by_name[declaration.source_name.casefold()].name
            for declaration in declared_named
            if declaration.source_name is not None
            and declaration.source_name.casefold() in existing_by_name
            and declaration.source_name.casefold() not in authorized_names
        )
        if collisions:
            return MechanismSourceCollision(
                existing_names=tuple(dict.fromkeys(collisions))
            )
        unused_authorizations = authorized_names.difference(
            declaration.source_name.casefold()
            for declaration in declared_named
            if declaration.source_name is not None
            and declaration.source_name.casefold() in existing_by_name
        )
        if unused_authorizations:
            raise ValueError(
                "Initial-condition replacement authorization does not match "
                "the current typed admission: "
                + ", ".join(sorted(unused_authorizations))
            )

        anonymous_declarations = tuple(
            declaration
            for declaration in admission.declarations
            if declaration.source_kind == "anonymous"
        )
        anonymous_name: str | None = None
        anonymous_existing_id: str | None = None
        if anonymous_declarations:
            declared_name_keys = {
                declaration.source_name.casefold()
                for declaration in declared_named
                if declaration.source_name is not None
            }
            pristine_set1 = next(
                (
                    item
                    for item in reconciled_sets
                    if item.name == "set1"
                    and not any(value != 0.0 for _species, value in item.values)
                    and not any(
                        set_id == item.set_id
                        for set_id, _settings in tab.run_overrides_by_set_id
                    )
                    and not any(
                        link.set_id == item.set_id
                        for _dataset_id, link in tab.dataset_links
                    )
                ),
                None,
            )
            if pristine_set1 is not None and "set1" not in declared_name_keys:
                anonymous_name = pristine_set1.name
                anonymous_existing_id = pristine_set1.set_id
            else:
                occupied = {
                    item.name.casefold() for item in reconciled_sets
                } | declared_name_keys
                suffix = 1
                while f"set{suffix}".casefold() in occupied:
                    suffix += 1
                anonymous_name = f"set{suffix}"

        for declaration in admission.declarations:
            context_name = (
                declaration.source_name
                if declaration.source_kind == "named"
                else anonymous_name
            )
            if context_name is None:
                raise ValueError(
                    "Anonymous initial conditions have no materialization "
                    "target."
                )
            for assignment in declaration.assignments:
                if assignment.species not in new_species_set:
                    raise ValueError(
                        f"Inline initial-condition set {context_name!r} "
                        f"references species {assignment.species!r} absent "
                        "from the complete Reactions plus State Network model."
                    )

        mutable_sets = list(reconciled_sets)
        index_by_name = {
            item.name.casefold(): index
            for index, item in enumerate(mutable_sets)
        }
        materialized_ids: list[str] = []
        anonymous_target: InitialConditionSet | None = None
        for declaration in admission.declarations:
            if declaration.source_kind == "named":
                target_name = str(declaration.source_name)
                target_index = index_by_name.get(target_name.casefold())
                values = {species: 0.0 for species in new_species}
                values.update(
                    {
                        assignment.species: assignment.value
                        for assignment in declaration.assignments
                    }
                )
                if target_index is None:
                    target = InitialConditionSet(
                        set_id=new_set_id(),
                        name=target_name,
                        values=tuple(values.items()),
                    )
                    index_by_name[target_name.casefold()] = len(mutable_sets)
                    mutable_sets.append(target)
                else:
                    existing = mutable_sets[target_index]
                    target = replace(existing, values=tuple(values.items()))
                    mutable_sets[target_index] = target
                materialized_ids.append(target.set_id)
                continue

            if anonymous_name is None:
                raise ValueError(
                    "Anonymous initial conditions have no allocated set."
                )
            if anonymous_target is None:
                target_index = index_by_name.get(anonymous_name.casefold())
                if target_index is None:
                    anonymous_target = InitialConditionSet(
                        set_id=new_set_id(),
                        name=anonymous_name,
                        values=tuple(
                            (species, 0.0) for species in new_species
                        ),
                    )
                    index_by_name[anonymous_name.casefold()] = len(mutable_sets)
                    mutable_sets.append(anonymous_target)
                else:
                    anonymous_target = mutable_sets[target_index]
                    if (
                        anonymous_existing_id is not None
                        and anonymous_target.set_id != anonymous_existing_id
                    ):
                        raise ValueError(
                            "Anonymous initial-condition allocation changed "
                            "during admission."
                        )
                materialized_ids.append(anonymous_target.set_id)
            values = {
                species: float(
                    anonymous_target.values_dict().get(species, 0.0)
                )
                for species in new_species
            }
            for assignment in declaration.assignments:
                values[assignment.species] = assignment.value
            anonymous_target = replace(
                anonymous_target,
                values=tuple(values.items()),
            )
            mutable_sets[index_by_name[anonymous_name.casefold()]] = (
                anonymous_target
            )

        accepted_source = (
            normalized_visible
            if not admission.has_declarations
            else _normalized_mechanism_source(
                MechanismAuthoringSource.from_full_dsl_text(
                    admission.materialized_source
                )
            )
        )
        prepared_dsl = (
            _prepare_source_dsl(
                accepted_source,
                tab.scientific_settings,
            )
            if admission.has_declarations
            else admitted_dsl
        )
        if (
            prepared_dsl is not None
            and prepared_dsl.analysis.inline_initial_conditions.has_declarations
        ):
            raise ValueError(
                "Accepted source still contains executable inline initial "
                "conditions after materialization."
            )
        reconciled_links: list[tuple[str, DatasetLink]] = []
        for dataset_id, link in tab.dataset_links:
            observation_mapping = tuple(
                (raw_column, model_species)
                for raw_column, model_species in link.observation_mapping
                if model_species in new_species_set
            )
            reconciled_links.append(
                (
                    dataset_id,
                    replace(
                        link,
                        observation_mapping=observation_mapping,
                        include_in_fit=(
                            link.include_in_fit
                            and bool(observation_mapping)
                        ),
                    ),
                )
            )
        proposed_tab = replace(
            tab.with_initial_condition_sets(tuple(mutable_sets)),
            mechanism_source=accepted_source,
            dataset_links=tuple(reconciled_links),
            prepared_dsl=prepared_dsl,
        )
        proposed_document = replace(
            document,
            tabs=tuple(
                proposed_tab if item.tab_id == tab.tab_id else item
                for item in document.tabs
            ),
        )
        return MechanismSourceReconciliation(
            proposed_document=proposed_document,
            introduced_species=introduced,
            removed_species=removed,
            inline_initial_conditions=admission,
            materialized_set_ids=tuple(dict.fromkeys(materialized_ids)),
        )
    except (DSLError, KeyError, TypeError, ValueError) as exc:
        return MechanismSourceFailure(message=str(exc))


def validate_project_document(document: ProjectDocument) -> ProjectDocument:
    """Validate every canonical value and cross-reference."""
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    from kindred.core.dataset_mapping import (
        validate_fit_inclusion,
        validate_raw_dataset_record,
    )
    from kindred.core.datasets.observation_mapping import (
        normalize_observation_mapping,
    )

    if not document.tabs:
        raise ValueError("A project must contain at least one mechanism tab.")
    _validate_fit_defaults(document.fit_defaults)

    for dataset in document.datasets:
        validate_raw_dataset_record(dataset)
        raw_observation_ids = tuple(
            validate_raw_observation_id(
                series.raw_observation_id,
                (
                    f"Raw dataset {dataset.dataset_id!r} observation "
                    f"{series.column_name!r} raw observation ID"
                ),
            )
            for series in dataset.observations
        )
        _validate_unique(
            raw_observation_ids,
            f"raw observation IDs in dataset {dataset.dataset_id!r}",
        )
        normalized_metadata = freeze_json_value(
            thaw_json_value(dataset.source_metadata)
        )
        if normalized_metadata != dataset.source_metadata:
            raise TypeError(
                f"Raw dataset {dataset.dataset_id!r} source_metadata must "
                "be created with freeze_json_value()."
            )

    _validate_unique(
        (_require_id(tab.tab_id, "tab ID") for tab in document.tabs),
        "tab IDs",
    )
    _require_id(document.project_id, "project ID")
    _validate_unique(
        (_require_name(tab.name, "tab name") for tab in document.tabs),
        "tab names",
        key=lambda value: str(value).casefold(),
    )
    _validate_unique(
        (
            _require_id(item.dataset_id, "dataset ID")
            for item in document.datasets
        ),
        "dataset IDs",
    )
    _validate_unique(
        (
            _require_name(item.display_name, "dataset display name")
            for item in document.datasets
        ),
        "dataset display names",
        key=lambda value: str(value).casefold(),
    )

    dataset_by_id = {
        item.dataset_id: item
        for item in document.datasets
    }

    global_set_ids: list[str] = []
    mechanism_species_by_tab_id: dict[str, tuple[str, ...]] = {}
    hydrated_tabs: list[MechanismTabDocument] = []
    all_entity_ids = [document.project_id]
    all_entity_ids.extend(tab.tab_id for tab in document.tabs)
    all_entity_ids.extend(
        dataset.dataset_id
        for dataset in document.datasets
    )
    for tab in document.tabs:
        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("tabs must contain MechanismTabDocument values.")
        if not isinstance(tab.mechanism_source, MechanismAuthoringSource):
            raise TypeError(
                "mechanism_source must be MechanismAuthoringSource."
            )
        if not isinstance(tab.notes, str):
            raise TypeError("tab notes must be a str.")
        if not tab.initial_condition_sets:
            raise ValueError(
                f"Tab {tab.name!r} must contain at least one "
                "initial-condition set."
            )
        validate_simulation_execution_settings(
            tab.run_defaults,
            f"{tab.name}.run_defaults",
        )
        validate_tab_scientific_settings(
            tab.scientific_settings,
            f"{tab.name}.scientific_settings",
        )
        if (
            tab.prepared_dsl is not None
            and type(tab.prepared_dsl) is not PreparedDslRequestSet
        ):
            raise TypeError(
                f"Tab {tab.tab_id!r} ({tab.name!r}) prepared_dsl must "
                "be PreparedDslRequestSet or None."
            )
        try:
            if not _prepared_dsl_matches_tab(tab):
                tab = replace(
                    tab,
                    prepared_dsl=_prepare_source_dsl(
                        tab.mechanism_source,
                        tab.scientific_settings,
                    ),
                )
            analysis = tab.dsl_analysis
            if (
                analysis is not None
                and analysis.inline_initial_conditions.has_declarations
            ):
                raise ValueError(
                    "Executable inline initial conditions are forbidden after "
                    "typed canonical materialization."
                )
            mechanism_species = (
                ()
                if analysis is None
                else analysis.species_names
            )
        except (DSLError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Tab {tab.tab_id!r} ({tab.name!r}) contains an invalid "
                f"accepted mechanism: {exc}"
            ) from exc
        mechanism_species_by_tab_id[tab.tab_id] = mechanism_species
        mechanism_species_set = frozenset(mechanism_species)
        set_id_values = [
            _require_id(item.set_id, "initial-condition set ID")
            for item in tab.initial_condition_sets
        ]
        _validate_unique(
            set_id_values,
            f"set IDs in tab {tab.name!r}",
        )
        set_ids = set(set_id_values)
        global_set_ids.extend(set_id_values)
        _validate_unique(
            (
                _require_name(item.name, "initial-condition set name")
                for item in tab.initial_condition_sets
            ),
            f"set names in tab {tab.name!r}",
            key=lambda value: str(value).casefold(),
        )
        for item in tab.initial_condition_sets:
            if not isinstance(item, InitialConditionSet):
                raise TypeError(
                    "initial_condition_sets must contain "
                    "InitialConditionSet values."
                )
            _validate_unique(
                (
                    _require_name(
                        species,
                        "initial-condition species name",
                    )
                    for species, _ in item.values
                ),
                f"species in initial-condition set {item.set_id!r}",
            )
            for species, value in item.values:
                _require_finite(
                    value,
                    "initial concentration for "
                    f"tab {tab.tab_id!r} ({tab.name!r}), "
                    f"set {item.set_id!r} ({item.name!r}), "
                    f"species {species!r}",
                    minimum=0.0,
                )
                if species not in mechanism_species_set:
                    raise ValueError(
                        f"Initial-condition set {item.set_id!r} references "
                        f"species {species!r} absent from the mechanism."
                    )

        _validate_unique(
            (
                set_id
                for set_id, _ in tab.run_overrides_by_set_id
            ),
            f"run override set IDs in tab {tab.name!r}",
        )
        for set_id, settings in tab.run_overrides_by_set_id:
            if set_id not in set_ids:
                raise ValueError(
                    f"Run override references missing set ID {set_id!r}."
                )
            validate_simulation_execution_settings(
                settings,
                f"{tab.name}.run_override[{set_id}]",
            )

        _validate_unique(
            (
                dataset_id
                for dataset_id, _ in tab.dataset_links
            ),
            f"dataset link IDs in tab {tab.name!r}",
        )
        mapped_set_ids: list[str] = []
        for dataset_id, link in tab.dataset_links:
            if not isinstance(link, DatasetLink):
                raise TypeError(
                    "dataset_links must contain DatasetLink values."
                )
            _require_bool(
                link.include_in_fit,
                "DatasetLink.include_in_fit",
            )
            if link.set_id is not None:
                if link.set_id in set_ids:
                    mapped_set_ids.append(link.set_id)
                elif not link.include_in_fit:
                    raise ValueError(
                        f"Dataset {dataset_id!r} maps to missing set ID "
                        f"{link.set_id!r}."
                    )

            if link.include_in_fit:
                continue
            dataset = dataset_by_id.get(dataset_id)
            if dataset is None:
                raise ValueError(
                    f"Dataset link references missing dataset ID "
                    f"{dataset_id!r}."
                )
            normalize_observation_mapping(
                link.observation_mapping,
                raw_columns=tuple(
                    series.column_name
                    for series in dataset.observations
                ),
                model_species=mechanism_species,
            )
        _validate_unique(
            mapped_set_ids,
            f"mapped set IDs in tab {tab.name!r}",
        )
        hydrated_tabs.append(tab)

    _validate_unique(global_set_ids, "set IDs across the project")
    all_entity_ids.extend(global_set_ids)
    _validate_unique(all_entity_ids, "stable entity IDs across the project")

    hydrated_document = replace(
        document,
        tabs=tuple(hydrated_tabs),
    )
    for tab in hydrated_document.tabs:
        for dataset_id, link in tab.dataset_links:
            if not link.include_in_fit:
                continue
            dataset = dataset_by_id.get(dataset_id)
            initial_set = (
                None
                if link.set_id is None
                else next(
                    (
                        item
                        for item in tab.initial_condition_sets
                        if item.set_id == link.set_id
                    ),
                    None,
                )
            )
            dataset_label = (
                f"{dataset.display_name!r} (ID {dataset.dataset_id!r})"
                if dataset is not None
                else f"<missing> (ID {dataset_id!r})"
            )
            set_label = (
                f"{initial_set.name!r} (ID {initial_set.set_id!r})"
                if initial_set is not None
                else f"<missing> (ID {link.set_id!r})"
            )
            try:
                validate_fit_inclusion(
                    document=hydrated_document,
                    tab=tab,
                    dataset_id=dataset_id,
                    mechanism_species=mechanism_species_by_tab_id[
                        tab.tab_id
                    ],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid Fit inclusion for "
                    f"tab {tab.name!r} (ID {tab.tab_id!r}), "
                    f"dataset {dataset_label}, set {set_label}, "
                    f"mapping {link.observation_mapping!r}: {exc}"
                ) from exc
    return hydrated_document

"""Immutable project science values and their canonical validation."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence

from kindred.core.authored_source_errors import AuthoredSourceError
from kindred.core.datasets import foundation as dataset_foundation
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.fitting_method_policy import LEAST_SQUARES_TOLERANCE, validate_fit_method
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.dsl_parse import (
    DslAnalysis,
    DslInitialConditionAdmission,
    PreparedDslRequestSet,
    prepare_dsl_request_set,
)
from kindred.core.simulator.dsl_build_input import source_coordinate_capabilities
from kindred.core.simulator.dsl_derivation import (
    insert_scalar_source_declarations,
)
from kindred.core.simulator.exact_constraint_composition import (
    ExactConstraintComposition,
    compose_exact_constraints,
)
from kindred.core.simulator.source_correction import (
    CorrectionPlan,
    CorrectionPlanner,
    CorrectionTarget,
    build_correction_expressions,
)
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.scipy_integrate import validate_scipy_rtol
from kindred.core.units import UnitsModel
from kindred.core.species import (
    MAX_FIT_EVALUATIONS,
    MAX_GENERATED_TIME_ROWS,
    TEMPERATURE_MAX_K,
    TEMPERATURE_MIN_K,
    validate_concentration,
    validate_c0,
    validate_temperature,
    validate_input_time_seconds,
)

__all__ = [
    "DatasetLink",
    "FitDefaults",
    "InitialConditionSet",
    "MechanismSourceCollision",
    "MechanismSourceReconciliation",
    "ExactCorrectionAssessment",
    "MechanismTabDocument",
    "ProjectDocument",
    "SimulationExecutionSettings",
    "TabScientificSettings",
    "default_fit_defaults",
    "default_project_document",
    "default_simulation_execution_settings",
    "default_tab_scientific_settings",
    "FIT_MAX_NFEV_RANGE",
    "FIT_SEED_RANGE",
    "SIMULATION_POINT_RANGE",
    "TEMPERATURE_RANGE_K",
    "parse_points_text",
    "SUPPORTED_SOLVERS",
    "validate_fit_solver",
    "fresh_mechanism_tab",
    "new_dataset_id",
    "new_project_id",
    "new_set_id",
    "new_tab_id",
    "next_mechanism_name",
    "next_set_name",
    "canonical_initial_values",
    "canonical_observation_target_names",
    "canonical_species_names",
    "prepare_authored_source_admission",
    "reconcile_tab_mechanism_source",
    "assess_tab_exact_correction",
    "select_exact_correction_basis",
    "prepare_tab_scientific_settings",
    "resolve_simulation_execution_settings",
    "validate_fit_defaults",
    "validate_project_document",
    "validate_simulation_execution_settings",
    "validate_tab_scientific_settings",
]

_SIMULATION_POINT_RANGE = (10, MAX_GENERATED_TIME_ROWS)
_TEMPERATURE_RANGE_K = (TEMPERATURE_MIN_K, TEMPERATURE_MAX_K)
_SOLVERS = ("Radau", "BDF")
_FITTING_MAX_NFEV_RANGE = (10, MAX_FIT_EVALUATIONS)
_FITTING_SEED_RANGE = (0, 999_999)

# Public option policy owned by the canonical project model.  GUI controls
# project these immutable values; they must not restate or narrow them.
SUPPORTED_SOLVERS = _SOLVERS
FIT_MAX_NFEV_RANGE = _FITTING_MAX_NFEV_RANGE
FIT_SEED_RANGE = _FITTING_SEED_RANGE
SIMULATION_POINT_RANGE = _SIMULATION_POINT_RANGE
TEMPERATURE_RANGE_K = _TEMPERATURE_RANGE_K


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


def _validate_initial_condition_values(
    values: object,
    *,
    context: str,
) -> tuple[tuple[str, float], ...]:
    """Admit one set's species values: nonblank unique species, valid values."""

    validated: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, (species, value) in enumerate(values):
        _require_name(
            species,
            f"{context} values[{index}][0] species name",
        )
        if species in seen:
            raise ValueError(
                f"Duplicate species {species!r} in {context}."
            )
        parsed = validate_concentration(
            value,
            field=f"{context} values[{index}][1] initial concentration",
        )
        seen.add(species)
        validated.append((species, parsed))
    return tuple(validated)


def validate_fit_solver(value: object) -> str:
    """Validate and return one canonical solver name for Fit execution."""

    try:
        normalized = normalize_solver_name(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Fit solver must be one of {SUPPORTED_SOLVERS!r}."
        ) from exc
    if normalized not in SUPPORTED_SOLVERS:
        raise ValueError(
            f"Fit solver must be one of {SUPPORTED_SOLVERS!r}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SimulationExecutionSettings:
    """Complete per-target Run/Preview settings.

    Fit owns its simulation interval and does not carry this Run-owned
    aggregate.
    """

    solver: str = "BDF"
    t_start: float = 0.0
    t_end: float = 10.0
    integration_points: int = 100


@dataclass(frozen=True, slots=True)
class TabScientificSettings:
    """Mechanism settings shared within one tab; fitting always bypasses stops."""

    temperature: float = UnitsModel.temperature_K
    standard_conc_M: float = UnitsModel.standard_conc_M
    energy_unit: str = UnitsModel.energy_unit
    rtol: float = 1e-6
    atol: float = 1e-12
    analytic_jacobian_policy: bool = True
    use_authored_stops: bool = True

    @property
    def default_units(self) -> UnitsModel:
        return UnitsModel(
            temperature_K=self.temperature,
            standard_conc_M=self.standard_conc_M,
            energy_unit=self.energy_unit,
        )


@dataclass(frozen=True, slots=True)
class FitDefaults:
    method: str = "trf"
    max_nfev: int = 1000
    ftol: float = LEAST_SQUARES_TOLERANCE
    xtol: float = LEAST_SQUARES_TOLERANCE
    use_seed: bool = True
    seed: int = 42


@dataclass(frozen=True, slots=True)
class InitialConditionSet:
    set_id: str
    name: str
    values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Admit an immutable, duplicate-free carrier at construction."""

        _require_id(self.set_id, "initial-condition set ID")
        _require_name(self.name, "initial-condition set name")
        validated = _validate_initial_condition_values(
            self.values,
            context=f"initial-condition set {self.set_id!r}",
        )
        # Numeric admission canonicalizes Real values to the model's float
        # representation while retaining authored species order.
        object.__setattr__(self, "values", validated)

    def values_dict(self) -> dict[str, float]:
        return dict(self.values)

    def with_values(
        self,
        values: Mapping[str, Real] | Iterable[tuple[str, Real]],
    ) -> "InitialConditionSet":
        items = values.items() if isinstance(values, Mapping) else values
        # The constructor admits species names and concentrations once.
        return replace(self, values=tuple(items))


def _validate_initial_condition_set_carriers(
    sets: Iterable[InitialConditionSet],
    *,
    context: str,
) -> tuple[InitialConditionSet, ...]:
    """Reject duplicate set IDs or names before any set map is built."""

    validated: list[InitialConditionSet] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for item in sets:
        set_id, set_name = item.set_id, item.name
        if set_id in seen_ids:
            raise ValueError(
                f"Duplicate initial-condition set ID {set_id!r} in {context}."
            )
        name_key = set_name.casefold()
        if name_key in seen_names:
            raise ValueError(
                f"Duplicate initial-condition set name {item.name!r} "
                f"in {context}."
            )
        seen_ids.add(set_id)
        seen_names.add(name_key)
        validated.append(item)
    return tuple(validated)


@dataclass(frozen=True, slots=True)
class DatasetLink:
    set_id: str | None = None
    observation_mapping: tuple[tuple[str, str], ...] = ()


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
        return self.with_initial_condition_sets(
            (*self.initial_condition_sets, initial_set)
        )

    def replace_initial_condition_set(
        self,
        initial_set: InitialConditionSet,
    ) -> "MechanismTabDocument":
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

    def with_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> "MechanismTabDocument":
        # Run settings arrive admitted by their editor.
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
        _validate_initial_condition_set_carriers(
            candidate_sets,
            context="initial-condition sets",
        )
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
                else replace(link, set_id=None),
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
    ) -> "MechanismTabDocument":
        links = dict(self.dataset_links)
        links[dataset_id] = link
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
    datasets: tuple[dataset_foundation.RawDatasetRecord, ...]
    tabs: tuple[MechanismTabDocument, ...]
    fit_defaults: FitDefaults
    imported_sources: tuple[dataset_foundation.ImportedSource, ...] = ()

    def tab(self, tab_id: str) -> MechanismTabDocument:
        target = str(tab_id)
        for item in self.tabs:
            if item.tab_id == target:
                return item
        raise KeyError(f"Unknown mechanism tab ID: {target!r}.")

    def dataset(self, dataset_id: str) -> dataset_foundation.RawDatasetRecord:
        target = str(dataset_id)
        for item in self.datasets:
            if item.dataset_id == target:
                return item
        raise KeyError(f"Unknown dataset ID: {target!r}.")

    def imported_source(self, source_id: str) -> dataset_foundation.ImportedSource:
        target = str(source_id)
        for source in self.imported_sources:
            if source.source_id == target:
                return source
        raise KeyError(f"Unknown imported source ID: {target!r}.")

    def replace_tab(self, tab: MechanismTabDocument) -> "ProjectDocument":
        if not any(item.tab_id == tab.tab_id for item in self.tabs):
            raise KeyError(f"Unknown mechanism tab ID: {tab.tab_id!r}.")
        return replace(
            self,
            tabs=tuple(
                tab if item.tab_id == tab.tab_id else item
                for item in self.tabs
            ),
        )

    def add_tab(
        self,
        *,
        run_defaults: SimulationExecutionSettings | None = None,
        scientific_settings: TabScientificSettings | None = None,
    ) -> "ProjectDocument":
        new_tab = fresh_mechanism_tab(
            name=next_mechanism_name(item.name for item in self.tabs),
            run_defaults=run_defaults,
            scientific_settings=scientific_settings,
        )
        return replace(self, tabs=(*self.tabs, new_tab))

    def rename_tab(self, tab_id: str, name: str) -> "ProjectDocument":
        tab = self.tab(tab_id)
        candidate = self.replace_tab(
            replace(tab, name=_require_name(name, "tab name"))
        )
        _validate_unique(
            (item.name for item in candidate.tabs),
            "tab names",
            key=lambda value: str(value).casefold(),
        )
        return candidate

    def remove_tab(self, tab_id: str) -> "ProjectDocument":
        if len(self.tabs) <= 1:
            raise ValueError("The last mechanism tab cannot be closed.")
        target = str(tab_id)
        tabs = tuple(item for item in self.tabs if item.tab_id != target)
        if len(tabs) == len(self.tabs):
            raise KeyError(f"Unknown mechanism tab ID: {target!r}.")
        return replace(self, tabs=tabs)

    def add_datasets(
        self,
        records: tuple[dataset_foundation.RawDatasetRecord, ...],
        *,
        sources: tuple[dataset_foundation.ImportedSource, ...],
    ) -> "ProjectDocument":
        """Prepare one complete dataset batch as one atomic proposal."""

        return replace(
            self,
            datasets=(*self.datasets, *records),
            imported_sources=(*self.imported_sources, *sources),
        )

    def replace_dataset(
        self,
        record: dataset_foundation.RawDatasetRecord,
    ) -> "ProjectDocument":
        if not any(
            item.dataset_id == record.dataset_id for item in self.datasets
        ):
            raise KeyError(f"Unknown dataset ID: {record.dataset_id!r}.")
        return replace(
            self,
            datasets=tuple(
                record if item.dataset_id == record.dataset_id else item
                for item in self.datasets
            ),
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
                display_name=display_name,
            )
        )

    def remove_dataset(self, dataset_id: str) -> "ProjectDocument":
        target = str(dataset_id)
        if not any(item.dataset_id == target for item in self.datasets):
            raise KeyError(f"Unknown dataset ID: {target!r}.")
        datasets = tuple(
            item
            for item in self.datasets
            if item.dataset_id != target
        )
        referenced_source_ids = {item.source_id for item in datasets if item.source_id is not None}
        imported_sources = tuple(
            source
            for source in self.imported_sources
            if source.source_id in referenced_source_ids
        )
        return replace(
            self,
            datasets=datasets,
            imported_sources=imported_sources,
            tabs=tuple(tab.without_dataset(target) for tab in self.tabs),
        )


@dataclass(frozen=True, slots=True)
class MechanismSourceReconciliation:
    proposed_document: ProjectDocument
    materialized_set_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExactCorrectionAssessment:
    """One pure symbolic assessment of visible authored mechanism source.

    The assessment carries the complete typed composition and source-backed
    target roster used by the global correction planner.  ``proposal`` is
    source text only: it is deliberately not parsed or admitted here.  The
    ordinary source reconciliation boundary must admit an accepted proposal
    before any document publication.
    """

    composition: ExactConstraintComposition | None
    targets: tuple[CorrectionTarget, ...]
    plan: CorrectionPlan
    proposal: MechanismAuthoringSource | None = None
    unavailable_reason: str | None = None
    # The param lines the proposal adds, as name and expression.
    corrections: Mapping[str, str] = field(default_factory=dict)

    @property
    def requires_correction(self) -> bool:
        """Whether the composed symbolic constraints include an invalid row."""

        return bool(
            self.composition is not None
            and any(
                item.status in {"unsatisfied", "unresolved"}
                for item in self.composition.constraints
            )
        )


@dataclass(frozen=True, slots=True)
class MechanismSourceCollision:
    """A pure pre-mutation request for exact named-set replacement consent."""

    existing_names: tuple[str, ...]
    replacement: MechanismSourceReconciliation


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
        name=name,
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
    """Construct one default project candidate without admitting it."""

    return ProjectDocument(
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
                f"{label} must be unique. Duplicate: {value!r}."
            )
        seen.add(normalized)


def parse_points_text(text: str, *, field: str = "Points") -> int:
    """Admit authored Points text without clamping or reverting it."""
    minimum, maximum = _SIMULATION_POINT_RANGE
    message = f"{field} {text!r} must be an integer between {minimum} and {maximum}."
    try:
        value = int(str(text).strip())
    except ValueError as exc:
        raise ValueError(message) from exc
    if not minimum <= value <= maximum:
        raise ValueError(message)
    return value


def validate_simulation_execution_settings(
    settings: SimulationExecutionSettings,
    label: str | None = None,
) -> SimulationExecutionSettings:
    """Admit Run settings; messages name the on-screen fields."""
    prefix = "" if label is None else f"{label}: "
    normalized_solver = validate_fit_solver(settings.solver)
    if normalized_solver != settings.solver:
        raise ValueError(
            f"{prefix}Solver {settings.solver!r} must use canonical supported spelling."
        )
    start = validate_input_time_seconds(
        settings.t_start, field=f"{prefix}Start time {settings.t_start!r}"
    )
    end = validate_input_time_seconds(
        settings.t_end, field=f"{prefix}End time {settings.t_end!r}"
    )
    if end <= start:
        raise ValueError(
            f"{prefix}End time {end!r} must be greater than start time {start!r}."
        )
    _require_int(
        settings.integration_points,
        f"{prefix}Points {settings.integration_points!r}",
        *_SIMULATION_POINT_RANGE,
    )
    return settings


def validate_tab_scientific_settings(
    settings: TabScientificSettings,
    label: str | None = None,
) -> TabScientificSettings:
    """Admit scientific settings; messages name the on-screen fields."""
    prefix = "" if label is None else f"{label}: "
    validate_temperature(settings.temperature, field=f"{prefix}Temperature")
    validate_c0(settings.standard_conc_M, field=f"{prefix}Standard concentration")
    if settings.energy_unit not in {"J/mol", "kJ/mol", "kcal/mol"}:
        raise ValueError(f"{prefix}Energy units must be J/mol, kJ/mol, or kcal/mol.")
    validate_scipy_rtol(
        settings.rtol,
        field=f"{prefix}Relative tolerance",
    )
    _require_finite(
        settings.atol,
        f"{prefix}Absolute tolerance",
        strictly_positive=True,
    )
    return settings


def resolve_simulation_execution_settings(
    tab: MechanismTabDocument,
    set_id: str,
) -> SimulationExecutionSettings:
    """Resolve one exact set to either its complete override or live default."""
    target = tab.initial_condition_set(set_id).set_id
    return dict(tab.run_overrides_by_set_id).get(target, tab.run_defaults)


def validate_fit_defaults(defaults: FitDefaults) -> FitDefaults:
    """Validate the narrow Fit-default value without walking a project."""

    validate_fit_method(defaults.method)
    if defaults.method != str(defaults.method).strip():
        raise ValueError("fit_defaults.method must use canonical supported spelling.")
    _require_int(
        defaults.max_nfev,
        "fit_defaults.max_nfev",
        *FIT_MAX_NFEV_RANGE,
    )
    for field_name in ("ftol", "xtol"):
        _require_finite(
            getattr(defaults, field_name),
            f"fit_defaults.{field_name}",
            strictly_positive=True,
        )
    # With the zero-only gradient stop, SciPy requires at least one exposed
    # tolerance at or above binary64 epsilon. This is a pair restriction:
    # either tolerance may be smaller when the other satisfies the minimum.
    epsilon = math.ulp(1.0)
    if defaults.method in {"trf", "dogbox"} and max(defaults.ftol, defaults.xtol) < epsilon:
        raise ValueError(
            "TRF and Dogbox require at least one convergence tolerance "
            f"(ftol or xtol) to be at least {epsilon:.17g}."
        )
    _require_int(
        defaults.seed,
        "fit_defaults.seed",
        *FIT_SEED_RANGE,
    )
    return defaults


def prepare_authored_source_admission(
    source: MechanismAuthoringSource,
    scientific_settings: TabScientificSettings,
) -> PreparedDslRequestSet | None:
    """Admit user source through canonical parsing and exact consistency."""

    try:
        prepared_dsl = (
            None
            if not source.reactions_text.strip()
            else prepare_dsl_request_set(
                source.reactions_text,
                units=scientific_settings.default_units,
            )
        )
        _validate_exact_source_constraints(prepared_dsl)
    except AuthoredSourceError:
        raise
    except (DSLError, ValueError, OverflowError) as exc:
        raise AuthoredSourceError(str(exc)) from exc
    return prepared_dsl


def _validate_exact_source_constraints(
    prepared_dsl: PreparedDslRequestSet | None,
) -> None:
    """Reject every admitted reversible cycle that is not an exact identity."""

    if prepared_dsl is None:
        return
    composition = compose_exact_constraints(
        prepared_dsl.accepted_mechanism_for_structure(),
        parameter_algebra_spec=prepared_dsl.parameter_algebra_spec_for_execution(),
    )
    invalid = tuple(
        constraint
        for constraint in composition.constraints
        if constraint.status in {"unsatisfied", "unresolved"}
    )
    if invalid:
        raise AuthoredSourceError("\n\n".join(item.reason for item in invalid))


def canonical_species_names(tab: MechanismTabDocument) -> tuple[str, ...]:
    """Return the accepted mechanism species in canonical document order."""

    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        return ()
    return tuple(str(name) for name in prepared_dsl.analysis.species_names)


def canonical_observation_target_names(
    tab: MechanismTabDocument,
) -> tuple[str, ...]:
    """Return accepted species followed by accepted authored observables."""

    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        return ()
    analysis = prepared_dsl.analysis
    return (
        tuple(str(name) for name in analysis.species_names)
        + tuple(str(name) for name in analysis.observable_names)
    )


def canonical_initial_values(
    tab: MechanismTabDocument,
    set_id: str,
) -> tuple[tuple[str, float], ...]:
    """Project every accepted species, using exact zero for an absent value."""

    canonical = tab.initial_condition_set(set_id).values_dict()
    return tuple(
        (species, float(canonical.get(species, 0.0)))
        for species in canonical_species_names(tab)
    )


def reconcile_tab_mechanism_source(
    document: ProjectDocument,
    tab_id: str,
    visible_source: MechanismAuthoringSource,
) -> (
    MechanismSourceReconciliation
    | MechanismSourceCollision
):
    """Purely validate and reconcile visible source into one atomic proposal."""
    tab = document.tab(tab_id)
    admitted_dsl = prepare_authored_source_admission(
        visible_source,
        tab.scientific_settings,
    )
    admission = (
        DslInitialConditionAdmission(
            materialized_source=visible_source.reactions_text
        )
        if admitted_dsl is None
        else admitted_dsl.analysis.inline_initial_conditions
    )
    new_species = (
        ()
        if admitted_dsl is None
        else admitted_dsl.analysis.species_names
    )
    new_observation_targets = (
        ()
        if admitted_dsl is None
        else (
            *admitted_dsl.analysis.species_names,
            *admitted_dsl.analysis.observable_names,
        )
    )
    old_species = (
        ()
        if tab.dsl_analysis is None
        else tab.dsl_analysis.species_names
    )
    new_species_set = set(new_species)
    new_observation_targets_set = set(new_observation_targets)
    old_species_set = set(old_species)
    introduced = tuple(
        species
        for species in new_species
        if species not in old_species_set
    )
    reconciled_sets: list[InitialConditionSet] = []
    for initial_set in tab.initial_condition_sets:
        source_values = initial_set.values_dict()
        values = {
            species: value
            for species, value in source_values.items()
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
    declared_named = tuple(admission.named_declarations)
    collisions = tuple(
        existing_by_name[declaration.source_name.casefold()].name
        for declaration in declared_named
        if declaration.source_name is not None
        and declaration.source_name.casefold() in existing_by_name
    )

    anonymous_declarations = tuple(
        declaration
        for declaration in admission.declarations
        if declaration.source_kind == "anonymous"
    )
    anonymous_name: str | None = None
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
        for assignment in declaration.assignments:
            if assignment.species not in new_species_set:
                raise AuthoredSourceError(
                    f"Inline initial-condition set {context_name!r} "
                    f"references species {assignment.species!r} absent "
                    "from the complete Reactions model."
                )

    mutable_sets = list(reconciled_sets)
    materialized_ids: list[str] = []
    index_by_name = {
        item.name.casefold(): index
        for index, item in enumerate(mutable_sets)
    }
    anonymous_target: InitialConditionSet | None = None
    for declaration in admission.declarations:
        if declaration.source_kind == "named":
            target_name = str(declaration.source_name)
            target_index = index_by_name.get(target_name.casefold())
            values = {species: 0.0 for species in new_species}
            declaration_values = {
                assignment.species: assignment.value
                for assignment in declaration.assignments
            }
            values.update(declaration_values)
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
            materialized_ids.append(anonymous_target.set_id)
        anonymous_values = anonymous_target.values_dict()
        values = {
            species: float(anonymous_values.get(species, 0.0))
            for species in new_species
        }
        declaration_values = {
            assignment.species: assignment.value
            for assignment in declaration.assignments
        }
        values.update(declaration_values)
        anonymous_target = replace(
            anonymous_target,
            values=tuple(values.items()),
        )
        mutable_sets[index_by_name[anonymous_name.casefold()]] = (
            anonymous_target
        )

    accepted_source = (
        visible_source
        if not admission.has_declarations
        else MechanismAuthoringSource.from_parts(
            reactions_text=admission.materialized_source
        )
    )
    if admission.has_declarations:
        prepared_dsl = prepare_authored_source_admission(
            accepted_source,
            tab.scientific_settings,
        )
    else:
        prepared_dsl = admitted_dsl
    reconciled_links: list[tuple[str, DatasetLink]] = []
    for dataset_id, link in tab.dataset_links:
        observation_mapping = tuple(
            (raw_observation_id, model_target)
            for raw_observation_id, model_target
            in link.observation_mapping
            if model_target in new_observation_targets_set
        )
        reconciled_links.append(
            (
                dataset_id,
                replace(
                    link,
                    observation_mapping=observation_mapping,
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
    reconciliation = MechanismSourceReconciliation(
        proposed_document=proposed_document,
        materialized_set_ids=tuple(dict.fromkeys(materialized_ids)),
    )
    if collisions:
        return MechanismSourceCollision(
            existing_names=tuple(dict.fromkeys(collisions)),
            replacement=reconciliation,
        )
    return reconciliation


def _admitted_correction_targets(
    tab: MechanismTabDocument,
    composition: ExactConstraintComposition,
) -> tuple[CorrectionTarget, ...]:
    """List, in source order, the reaction-line values the open cycles involve.

    A value is listed when an open cycle's requirement reads it, and a param
    line written from other steps' values when it sets a step of an open
    cycle.  Calculated readouts and values outside the open cycles are not
    listed.
    """

    prepared = tab.prepared_dsl
    param_lines = prepared.parameter_algebra_spec_for_execution().parameter_dependencies
    open_cycles = tuple(
        item for item in composition.constraints if item.status != "resolved"
    )
    read = {name for item in open_cycles for name in item.monomial.names()}
    open_steps = {
        step for item in open_cycles for step, coefficient in item.step_coefficients if coefficient
    }
    context = {
        entry["step_index"]: entry["context"]
        for entry in prepared.accepted_mechanism_for_structure().metadata["step_index_map"]
    }
    targets: list[CorrectionTarget] = []
    for origin in prepared.analysis.origins:
        for coordinate in origin.coordinates:
            name = coordinate.canonical_name
            if not source_coordinate_capabilities(coordinate.role).correction_target:
                continue
            closure = name in composition.relations or name in composition.unresolved
            if name in read or (closure and origin.step_index in open_steps):
                targets.append(
                    CorrectionTarget(
                        name=name,
                        family=coordinate.role,
                        param_line=name in param_lines,
                        context=context[origin.step_index],
                    )
                )
    return tuple(targets)


def _correction_proposal(
    source: MechanismAuthoringSource,
    composition: ExactConstraintComposition,
    plan: CorrectionPlan,
) -> tuple[MechanismAuthoringSource, dict[str, str]]:
    corrections = build_correction_expressions(composition, plan)
    updated_source = insert_scalar_source_declarations(
        source.reactions_text,
        corrections,
    )
    return MechanismAuthoringSource.from_parts(reactions_text=updated_source), corrections


def select_exact_correction_basis(
    assessment: ExactCorrectionAssessment,
    visible_source: MechanismAuthoringSource,
    selected_basis: Sequence[str],
) -> ExactCorrectionAssessment:
    """Build a proposal from the original source for one exact selected basis."""

    composition = assessment.composition
    if composition is None:
        raise ValueError("Exact correction composition is unavailable.")
    plan = CorrectionPlanner(composition, assessment.targets).plan_selected_basis(
        selected_basis
    )
    if not plan.available:
        raise ValueError(plan.reason or "Selected correction basis is unavailable.")
    proposal, corrections = _correction_proposal(visible_source, composition, plan)
    return replace(
        assessment,
        plan=plan,
        proposal=proposal,
        unavailable_reason=None,
        corrections=corrections,
    )


def assess_tab_exact_correction(
    document: ProjectDocument,
    tab_id: str,
    visible_source: MechanismAuthoringSource | None = None,
) -> ExactCorrectionAssessment:
    """Assess visible source and derive one exact correction proposal.

    Parsing, symbolic composition, target derivation, and global basis
    planning happen once in this core owner.  The returned proposal retains
    authored source bytes but is not an admitted document; callers must pass
    it through :func:`reconcile_tab_mechanism_source` before publication.
    Invalid visible source and unavailable correction bases are represented
    by ``unavailable_reason`` so a modal caller can synchronously stop without
    creating a second scientific route.
    """

    tab = document.tab(tab_id)
    source = tab.mechanism_source if visible_source is None else visible_source

    empty_composition = ExactConstraintComposition(())
    if not source.reactions_text.strip():
        return ExactCorrectionAssessment(
            composition=empty_composition,
            targets=(),
            plan=CorrectionPlan((), (), True),
        )

    try:
        # This is intentionally parse-only.  Mandatory exact validation is
        # owned by ordinary source admission after a user accepts a proposal.
        prepared = prepare_dsl_request_set(
            source.reactions_text,
            units=tab.scientific_settings.default_units,
        )
        detached_tab = replace(
            tab,
            mechanism_source=source,
            prepared_dsl=prepared,
        )
        parameter_spec = prepared.parameter_algebra_spec_for_execution()
        composition = compose_exact_constraints(
            prepared.accepted_mechanism_for_structure(),
            parameter_algebra_spec=parameter_spec,
        )
        if not any(item.status in {"unsatisfied", "unresolved"}
                   for item in composition.constraints):
            return ExactCorrectionAssessment(
                composition=composition, targets=(), plan=CorrectionPlan((), (), True),
            )
        targets = _admitted_correction_targets(detached_tab, composition)
        plan = CorrectionPlanner(composition, targets).plan()
    except (AuthoredSourceError, DSLError, TypeError, ValueError, OverflowError) as exc:
        return ExactCorrectionAssessment(
            composition=None,
            targets=(),
            plan=CorrectionPlan((), (), False, str(exc)),
            unavailable_reason=str(exc),
        )

    if not plan.available:
        return ExactCorrectionAssessment(
            composition=composition,
            targets=targets,
            plan=plan,
            unavailable_reason=plan.reason or "Exact correction is unavailable.",
        )

    try:
        proposal, corrections = _correction_proposal(source, composition, plan)
    except (TypeError, ValueError, OverflowError) as exc:
        return ExactCorrectionAssessment(
            composition=composition,
            targets=targets,
            plan=plan,
            unavailable_reason=str(exc),
        )
    return ExactCorrectionAssessment(
        composition=composition,
        targets=targets,
        plan=plan,
        proposal=proposal,
        corrections=corrections,
    )


def prepare_tab_scientific_settings(
    tab: MechanismTabDocument,
    settings: TabScientificSettings,
) -> MechanismTabDocument:
    """Build one detached tab with settings-matching prepared science.

    A mechanism-default change reparses the unchanged canonical source under
    the new defaults, with explicit DSL values taking precedence. Other
    scientific setting changes retain the already admitted prepared object by
    identity.  No state is published here; the registry owns the one
    ``replace_document`` publication and its single result fence.
    """

    if settings.default_units == tab.scientific_settings.default_units:
        prepared = tab.prepared_dsl
    else:
        prepared = prepare_authored_source_admission(
            tab.mechanism_source,
            settings,
        )
    return replace(
        tab,
        scientific_settings=settings,
        prepared_dsl=prepared,
    )


def validate_project_document(document: ProjectDocument) -> ProjectDocument:
    """Validate every canonical value and cross-reference."""
    from kindred.core.datasets.observation_mapping import (
        normalize_observation_mapping,
    )

    # Settings, Fit defaults, imported sources and dataset records arrive
    # admitted by archive decoding, their editors or spreadsheet admission.
    if not document.tabs:
        raise ValueError("A project must contain at least one mechanism tab.")

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
    hydrated_tabs: list[MechanismTabDocument] = []
    all_entity_ids = [document.project_id]
    all_entity_ids.extend(source.source_id for source in document.imported_sources)
    all_entity_ids.extend(tab.tab_id for tab in document.tabs)
    all_entity_ids.extend(
        dataset.dataset_id
        for dataset in document.datasets
    )
    for tab in document.tabs:
        if not tab.initial_condition_sets:
            raise ValueError(
                f"Tab {tab.name!r} must contain at least one "
                "initial-condition set."
            )
        try:
            # Opened archives arrive unparsed; an empty tab stays unparsed.
            if tab.prepared_dsl is None and tab.mechanism_source.reactions_text.strip():
                prepared_dsl = prepare_authored_source_admission(
                    tab.mechanism_source,
                    tab.scientific_settings,
                )
                tab = replace(
                    tab,
                    prepared_dsl=prepared_dsl,
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
        except AuthoredSourceError as exc:
            raise AuthoredSourceError(
                f"Tab {tab.tab_id!r} ({tab.name!r}) contains invalid "
                f"authored Reactions: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Tab {tab.tab_id!r} ({tab.name!r}) contains an invalid "
                f"accepted mechanism: {exc}"
            ) from exc
        observation_targets = canonical_observation_target_names(tab)
        mechanism_species_set = frozenset(mechanism_species)
        admitted_initial_sets = _validate_initial_condition_set_carriers(
            tab.initial_condition_sets,
            context=(
                f"initial-condition sets in tab {tab.tab_id!r}"
            ),
        )
        set_id_values = [item.set_id for item in admitted_initial_sets]
        set_ids = set(set_id_values)
        global_set_ids.extend(set_id_values)
        for item in admitted_initial_sets:
            for species, _value in item.values:
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

        _validate_unique(
            (
                dataset_id
                for dataset_id, _ in tab.dataset_links
            ),
            f"dataset link IDs in tab {tab.name!r}",
        )
        mapped_set_ids: list[str] = []
        for dataset_id, link in tab.dataset_links:
            if link.set_id is not None:
                if link.set_id in set_ids:
                    mapped_set_ids.append(link.set_id)
                else:
                    raise ValueError(
                        f"Dataset {dataset_id!r} maps to missing set ID "
                        f"{link.set_id!r}."
                    )

            dataset = dataset_by_id.get(dataset_id)
            if dataset is None:
                raise ValueError(
                    f"Dataset link references missing dataset ID "
                    f"{dataset_id!r}."
                )
            normalized_observation_mapping = normalize_observation_mapping(
                link.observation_mapping,
                raw_observations=tuple(
                    (
                        series.raw_observation_id,
                        series.column_name,
                    )
                    for series in dataset.observations
                ),
                model_targets=observation_targets,
            )
            if normalized_observation_mapping != link.observation_mapping:
                raise ValueError(
                    f"Dataset {dataset_id!r} observation mapping must already "
                    "be canonical and ordered by raw observation identity."
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
    return hydrated_document

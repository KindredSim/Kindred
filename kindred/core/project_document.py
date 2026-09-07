"""Immutable project science values and their canonical validation."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Callable, Iterable, Mapping, Sequence

from kindred.core.authored_source_errors import AuthoredSourceError
from kindred.core.equilibrium_rate_authority import (
    normalize_existing_equilibrium_rate_authority,
)
from kindred.core.datasets import foundation as dataset_foundation
from kindred.core.mechanism_source import (
    MechanismAuthoringSource,
    source_matches_analysis,
)
from kindred.core.fitting_method_policy import validate_fit_method
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
from kindred.core.simulator.step_indexing import (
    iter_canonical_parameters,
    lookup_source_coordinate_target,
)
from kindred.core.simulator.wegscheider_symbolic import analyze_wegscheider_topology
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.scipy_integrate import validate_scipy_rtol
from kindred.core.units import UnitsModel
from kindred.core.species import (
    CONCENTRATION_MAX_M,
    CONCENTRATION_MIN_M,
    MAX_FIT_EVALUATIONS,
    MAX_GENERATED_TIME_ROWS,
    TIME_MAX_S,
    TIME_MIN_S,
    TEMPERATURE_MAX_K,
    TEMPERATURE_MIN_K,
    validate_concentration,
    validate_c0,
    validate_temperature,
    validate_time_seconds,
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
    "CONCENTRATION_RANGE_M",
    "TIME_RANGE_S",
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
    "retained_tab_mechanism_source",
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
CONCENTRATION_RANGE_M = (CONCENTRATION_MIN_M, CONCENTRATION_MAX_M)
TIME_RANGE_S = (TIME_MIN_S, TIME_MAX_S)


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


def _validate_initial_condition_values(
    values: object,
    *,
    context: str,
) -> tuple[tuple[str, float], ...]:
    """Validate one carrier before any species map can be materialized.

    The exact tuple/pair shape is part of the immutable project contract.  A
    forged instance can bypass a dataclass ``__post_init__`` with
    ``object.__new__``/``object.__setattr__``; callers that are about to build
    a map therefore repeat this small, centralized preflight.
    """

    if type(values) is not tuple:
        raise TypeError(f"{context} values must be an exact tuple.")
    validated: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, pair in enumerate(values):
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(
                f"{context} values[{index}] must be an exact species/value "
                "pair."
            )
        species, value = pair
        if type(species) is not str:
            raise TypeError(
                f"{context} values[{index}][0] species name must be a str."
            )
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

    Fit resolves only ``t_start`` at its inclusion boundary and does not carry
    this Run-owned aggregate.
    """

    solver: str = "BDF"
    t_start: float = 0.0
    t_end: float = 10.0
    integration_points: int = 100


@dataclass(frozen=True, slots=True)
class TabScientificSettings:
    """Scientific settings that are shared by every execution in one tab."""

    temperature: float = UnitsModel.temperature_K
    standard_conc_M: float = UnitsModel.standard_conc_M
    energy_unit: str = UnitsModel.energy_unit
    rtol: float = 1e-6
    atol: float = 1e-12
    analytic_jacobian_policy: bool = True

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
    ftol: float = 1e-10
    xtol: float = 1e-10
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
        return replace(
            self,
            values=tuple(
                (
                    _require_name(species, "initial-condition species name"),
                    validate_concentration(
                        value,
                        field=f"initial value for {species}",
                    ),
                )
                for species, value in items
            ),
        )


def _validate_initial_condition_set_carriers(
    sets: object,
    *,
    context: str,
) -> tuple[InitialConditionSet, ...]:
    """Preflight all sets before any set-name/index map can be built."""

    if type(sets) is not tuple:
        raise TypeError(f"{context} must be an exact tuple.")
    validated: list[InitialConditionSet] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(sets):
        if not isinstance(item, InitialConditionSet):
            raise TypeError(
                f"{context}[{index}] must be an InitialConditionSet."
            )
        set_id = _require_id(item.set_id, "initial-condition set ID")
        set_name = _require_name(
            item.name,
            "initial-condition set name",
        )
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
        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("tab must be a MechanismTabDocument.")
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
        name: str | None = None,
    ) -> "ProjectDocument":
        new_tab = fresh_mechanism_tab(
            name=name or next_mechanism_name(item.name for item in self.tabs),
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
            (_require_name(item.name, "tab name") for item in candidate.tabs),
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
        if not isinstance(record, dataset_foundation.RawDatasetRecord):
            raise TypeError("record must be a RawDatasetRecord.")
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
        datasets = tuple(
            item
            for item in self.datasets
            if item.dataset_id != target
        )
        referenced_source_ids = {item.source_id for item in datasets}
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
        if not isinstance(self.replacement, MechanismSourceReconciliation):
            raise TypeError(
                "Mechanism source collision replacement must be prepared."
            )


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
                f"{label} must be unique; duplicate {value!r}."
            )
        seen.add(normalized)


def validate_simulation_execution_settings(
    settings: SimulationExecutionSettings,
    label: str = "execution settings",
) -> SimulationExecutionSettings:
    if not isinstance(settings, SimulationExecutionSettings):
        raise TypeError(f"{label} must be SimulationExecutionSettings.")
    normalized_solver = validate_fit_solver(settings.solver)
    if normalized_solver != settings.solver:
        raise ValueError(
            f"{label}.solver must use canonical supported spelling."
        )
    start = validate_time_seconds(settings.t_start, field=f"{label}.t_start")
    end = validate_time_seconds(settings.t_end, field=f"{label}.t_end")
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
    validate_temperature(settings.temperature, field=f"{label}.temperature")
    validate_c0(settings.standard_conc_M, field=f"{label}.standard_conc_M")
    if settings.energy_unit not in {"J/mol", "kJ/mol", "kcal/mol"}:
        raise ValueError(f"{label}.energy_unit must be J/mol, kJ/mol, or kcal/mol.")
    validate_scipy_rtol(
        settings.rtol,
        field=f"{label}.rtol",
    )
    _require_finite(
        settings.atol,
        f"{label}.atol",
        strictly_positive=True,
    )
    _require_bool(
        settings.analytic_jacobian_policy,
        f"{label}.analytic_jacobian_policy",
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


def validate_fit_defaults(defaults: FitDefaults) -> FitDefaults:
    """Validate the narrow Fit-default value without walking a project."""

    if not isinstance(defaults, FitDefaults):
        raise TypeError("fit_defaults must be FitDefaults.")
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
    _require_bool(defaults.use_seed, "fit_defaults.use_seed")
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
                initials={},
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
    if not invalid:
        return
    details = "; ".join(
        f"{constraint.cycle_id}: {constraint.expression}"
        + (
            f" ({constraint.reason})"
            if constraint.reason
            else ""
        )
        for constraint in invalid
    )
    raise AuthoredSourceError(
        "Authored reversible-cycle consistency failed: " + details
    )


def canonical_species_names(tab: MechanismTabDocument) -> tuple[str, ...]:
    """Return the accepted mechanism species in canonical document order."""

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        return ()
    if type(prepared_dsl) is not PreparedDslRequestSet:
        raise TypeError("tab prepared_dsl must be PreparedDslRequestSet.")
    return tuple(str(name) for name in prepared_dsl.analysis.species_names)


def canonical_observation_target_names(
    tab: MechanismTabDocument,
) -> tuple[str, ...]:
    """Return accepted species followed by accepted authored observables."""

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        return ()
    if type(prepared_dsl) is not PreparedDslRequestSet:
        raise TypeError("tab prepared_dsl must be PreparedDslRequestSet.")
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


def _prepared_dsl_matches_tab(tab: MechanismTabDocument) -> bool:
    prepared = tab.prepared_dsl
    if prepared is None:
        return not tab.mechanism_source.reactions_text.strip()
    if type(prepared) is not PreparedDslRequestSet:
        return False
    analysis = prepared.analysis
    if not source_matches_analysis(tab.mechanism_source, analysis):
        return False
    return prepared.default_units == tab.scientific_settings.default_units


def retained_tab_mechanism_source(
    document: ProjectDocument,
    tab_id: str,
) -> MechanismSourceReconciliation:
    """Return one action admission from retained accepted mechanism truth.

    This is the no-draft half of the mechanism-dependent admission boundary.
    It never parses or rewrites source: the canonical tab already owns the
    exact source and its matching prepared DSL request set.
    """

    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    tab = document.tab(tab_id)
    if not _prepared_dsl_matches_tab(tab):
        raise ValueError(
            f"Mechanism tab {tab.tab_id!r} has no matching retained prepared DSL."
        )
    _validate_exact_source_constraints(tab.prepared_dsl)
    return MechanismSourceReconciliation(
        proposed_document=document,
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
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    if not isinstance(visible_source, MechanismAuthoringSource):
        raise TypeError("visible_source must be MechanismAuthoringSource.")
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
    admitted_initial_sets = _validate_initial_condition_set_carriers(
        tab.initial_condition_sets,
        context=f"initial-condition sets in tab {tab.tab_id!r}",
    )
    reconciled_sets: list[InitialConditionSet] = []
    for initial_set in admitted_initial_sets:
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
    composition: object,
) -> tuple[CorrectionTarget, ...]:
    """Derive every potential correction target and its owned availability."""

    prepared = tab.prepared_dsl
    if prepared is None:
        raise ValueError("Correction requires admitted prepared DSL.")
    if prepared.temperature_schedule_for_execution() is not None:
        raise ValueError(
            "Exact source correction is unavailable when temperature is scheduled."
        )
    mechanism = prepared.accepted_mechanism_for_structure()
    parameter_spec = prepared.parameter_algebra_spec_for_execution()
    parameter_dependencies = parameter_spec.parameter_dependencies
    assignment_kinds = {
        str(name): parameter_spec.parameter_assignment_kind(str(name))
        for name in parameter_dependencies
    }
    coordinates = tuple(
        coordinate
        for origin in prepared.analysis.origins
        for coordinate in origin.coordinates
    )
    powers: set[str] = set()
    for item in getattr(composition, "constraints", ()):
        powers.update(str(name) for name in item.monomial.coordinate_power_map())
    admitted: list[CorrectionTarget] = []
    admitted_names: set[str] = set()
    equilibria = tuple(getattr(mechanism, "equilibria", ()) or ())
    for coordinate in coordinates:
        name = str(coordinate.canonical_name)
        role = str(coordinate.role)
        capabilities = source_coordinate_capabilities(role)
        assignment_kind = assignment_kinds.get(name)
        if (
            not capabilities.correction_target
            or (
                name not in powers
                and assignment_kind not in {"derived scalar", "mechanism-bound"}
            )
        ):
            continue
        admitted_names.add(name)
        source_target = lookup_source_coordinate_target(mechanism, name)
        if source_target is None:
            admitted.append(
                CorrectionTarget(
                    name=name,
                    family=role,
                    source_backed=False,
                    editable=False,
                    derived=assignment_kind in {"derived scalar", "mechanism-bound"},
                )
            )
            continue
        kind, item_index, target_role, entry = source_target
        if str(target_role) != role:
            admitted.append(
                CorrectionTarget(
                    name=name,
                    family=role,
                    source_backed=False,
                    editable=False,
                    derived=assignment_kind in {"derived scalar", "mechanism-bound"},
                )
            )
            continue
        editable = not bool(entry.get("locked", False))
        derived = assignment_kind in {
            "derived scalar",
            "mechanism-bound",
        }
        if str(kind) == "equilibrium":
            authority = normalize_existing_equilibrium_rate_authority(
                equilibria[int(item_index)]
            )
            if capabilities.equilibrium_authority:
                editable = editable and authority.authored_role_is_editable(role)
                derived = derived or authority.authored_role_is_derived(role)
        admitted.append(
            CorrectionTarget(
                name=name,
                family=role,
                editable=editable,
                locked=bool(entry.get("locked", False)),
                derived=derived,
            )
        )
    cycle_steps = {
        step_index
        for cycle in analyze_wegscheider_topology(mechanism).cycles
        for step_index in cycle.step_indices
    }
    for name, entry, role in iter_canonical_parameters(mechanism):
        if role != "Keq" or name in admitted_names:
            continue
        step_index = int(entry["step_index"])
        if step_index not in cycle_steps:
            continue
        admitted.append(
            CorrectionTarget(
                name=name,
                family=role,
                source_backed=False,
                editable=False,
                derived=True,
            )
        )
    return tuple(admitted)


def _correction_proposal(
    source: MechanismAuthoringSource,
    composition: ExactConstraintComposition,
    plan: CorrectionPlan,
) -> MechanismAuthoringSource:
    corrections = build_correction_expressions(composition, plan)
    updated_source = insert_scalar_source_declarations(
        source.reactions_text,
        corrections,
    )
    return MechanismAuthoringSource.from_parts(reactions_text=updated_source)


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
    proposal = _correction_proposal(visible_source, composition, plan)
    return replace(
        assessment,
        plan=plan,
        proposal=proposal,
        unavailable_reason=None,
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

    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    tab = document.tab(tab_id)
    source = tab.mechanism_source if visible_source is None else visible_source
    if not isinstance(source, MechanismAuthoringSource):
        raise TypeError("visible_source must be MechanismAuthoringSource.")

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
            initials={},
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
        targets = _admitted_correction_targets(detached_tab, composition)
        plan = CorrectionPlanner(composition, targets).plan()
    except (AuthoredSourceError, DSLError, TypeError, ValueError, OverflowError) as exc:
        return ExactCorrectionAssessment(
            composition=None,
            targets=(),
            plan=CorrectionPlan((), (), False, str(exc)),
            unavailable_reason=str(exc),
        )

    if not any(
        item.status in {"unsatisfied", "unresolved"}
        for item in composition.constraints
    ):
        return ExactCorrectionAssessment(
            composition=composition,
            targets=targets,
            plan=plan,
        )
    if not plan.available:
        return ExactCorrectionAssessment(
            composition=composition,
            targets=targets,
            plan=plan,
            unavailable_reason=plan.reason or "Exact correction is unavailable.",
        )

    try:
        proposal = _correction_proposal(source, composition, plan)
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

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    validate_tab_scientific_settings(settings)
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
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    from kindred.core.datasets.observation_mapping import (
        normalize_observation_mapping,
    )

    if not document.tabs:
        raise ValueError("A project must contain at least one mechanism tab.")
    if type(document.datasets) is not tuple:
        raise TypeError("Project datasets must be an immutable tuple.")
    if type(document.imported_sources) is not tuple or any(
        not isinstance(source, dataset_foundation.ImportedSource)
        for source in document.imported_sources
    ):
        raise TypeError("Project imported_sources must be an exact ImportedSource tuple.")
    validate_fit_defaults(document.fit_defaults)

    for source in document.imported_sources:
        dataset_foundation.validate_imported_source(source)
    _validate_unique(
        (source.source_id for source in document.imported_sources),
        "imported source IDs",
    )
    referenced_source_ids: set[str] = set()
    for dataset in document.datasets:
        dataset_foundation.validate_raw_dataset_record(dataset)
        referenced_source_ids.add(dataset.source_id)
        if not any(source.source_id == dataset.source_id for source in document.imported_sources):
            raise ValueError(
                f"Dataset {dataset.dataset_id!r} references unknown imported source "
                f"{dataset.source_id!r}."
            )
        raw_observation_ids = tuple(
            dataset_foundation.validate_raw_observation_id(
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
        normalized_metadata = dataset_foundation.freeze_json_value(
            dataset_foundation.thaw_json_value(dataset.source_metadata)
        )
        if normalized_metadata != dataset.source_metadata:
            raise TypeError(
                f"Raw dataset {dataset.dataset_id!r} source_metadata must "
                "be created with freeze_json_value()."
            )
    source_ids = {source.source_id for source in document.imported_sources}
    if referenced_source_ids != source_ids:
        orphaned = tuple(sorted(source_ids - referenced_source_ids))
        raise ValueError(f"Project contains unreferenced imported sources: {orphaned!r}.")

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
    all_entity_ids.extend(source.source_id for source in document.imported_sources)
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
                prepared_dsl = prepare_authored_source_admission(
                    tab.mechanism_source,
                    tab.scientific_settings,
                )
                tab = replace(
                    tab,
                    prepared_dsl=prepared_dsl,
                )
            else:
                _validate_exact_source_constraints(tab.prepared_dsl)
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
        except (DSLError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Tab {tab.tab_id!r} ({tab.name!r}) contains an invalid "
                f"accepted mechanism: {exc}"
            ) from exc
        mechanism_species_by_tab_id[tab.tab_id] = mechanism_species
        observation_targets = canonical_observation_target_names(tab)
        mechanism_species_set = frozenset(mechanism_species)
        admitted_initial_sets = _validate_initial_condition_set_carriers(
            tab.initial_condition_sets,
            context=(
                f"initial-condition sets in tab {tab.tab_id!r}"
            ),
        )
        set_id_values = [
            _require_id(item.set_id, "initial-condition set ID")
            for item in admitted_initial_sets
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
                for item in admitted_initial_sets
            ),
            f"set names in tab {tab.name!r}",
            key=lambda value: str(value).casefold(),
        )
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
                if dataset is None:
                    raise ValueError(
                        f"Dataset link references missing dataset ID {dataset_id!r}."
                    )
                if initial_set is None:
                    raise ValueError(
                        f"Mapped initial-condition set {link.set_id!r} is no longer present."
                    )
                resolved_settings = tab.resolved_execution_settings(
                    initial_set.set_id
                )
                dataset_foundation.validate_fit_inclusion(
                    dataset=dataset,
                    set_id=initial_set.set_id,
                    observation_mapping=link.observation_mapping,
                    initial_values=dict(
                        canonical_initial_values(tab, initial_set.set_id)
                    ),
                    mechanism_species=mechanism_species_by_tab_id[tab.tab_id],
                    observation_targets=canonical_observation_target_names(tab),
                    resolved_t_start=resolved_settings.t_start,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid Fit inclusion for "
                    f"tab {tab.name!r} (ID {tab.tab_id!r}), "
                    f"dataset {dataset_label}, set {set_label}, "
                    f"mapping {link.observation_mapping!r}: {exc}"
                ) from exc
    return hydrated_document

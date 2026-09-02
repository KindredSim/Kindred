"""Pure plans for project-global datasets and per-tab dataset links."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping, Sequence
import uuid

from kindred.core.datasets.observation_mapping import (
    normalize_observation_mapping,
    seed_species_values_at_exact_zero,
)
from kindred.core.datasets import foundation as dataset_foundation
from kindred.core.project_document import (
    DatasetLink,
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    canonical_initial_values,
    canonical_species_names,
    new_set_id,
)

class DatasetMappingError(ValueError):
    """Raised when a proposed dataset mutation violates a real invariant."""


class DatasetMappingChoice(str, Enum):
    CREATE_SET = "create_set_from_t0"
    MAP_EXISTING = "map_existing"
    LEAVE_UNMAPPED = "leave_unmapped"


class OccupiedSetChoice(str, Enum):
    OVERRIDE = "override"
    CREATE_COPY = "create_copy"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class DatasetMappingProposal:
    """One complete, immutable mapping choice for one committed dataset."""

    dataset_id: str
    action: DatasetMappingChoice | str
    target_set_id: str | None = None
    occupied_choice: OccupiedSetChoice | str | None = None
    observation_mapping: tuple[tuple[str, str], ...] = ()
    proposed_set_name: str | None = None

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise DatasetMappingError("A mapping proposal requires a dataset ID.")
        try:
            action = DatasetMappingChoice(self.action)
        except (TypeError, ValueError) as exc:
            raise DatasetMappingError("A mapping proposal requires a valid action.") from exc
        target = None if self.target_set_id is None else str(self.target_set_id).strip()
        if target == "":
            target = None
        occupied = self.occupied_choice
        if occupied is not None:
            try:
                occupied = OccupiedSetChoice(occupied)
            except (TypeError, ValueError) as exc:
                raise DatasetMappingError("A mapping proposal requires a valid occupancy choice.") from exc
        mapping = tuple(
            (str(raw_id).strip(), str(species).strip())
            for raw_id, species in tuple(self.observation_mapping)
        )
        if any(not raw_id or not species for raw_id, species in mapping):
            raise DatasetMappingError("Observation mappings cannot contain blank identities.")
        proposed_name = None if self.proposed_set_name is None else str(self.proposed_set_name).strip()
        if proposed_name == "":
            proposed_name = None
        if action is DatasetMappingChoice.MAP_EXISTING and not target:
            raise DatasetMappingError("Map Existing requires a target set.")
        if action is DatasetMappingChoice.CREATE_SET and target is not None:
            raise DatasetMappingError("Create Set cannot target an existing set.")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "target_set_id", target)
        object.__setattr__(self, "occupied_choice", occupied)
        object.__setattr__(self, "observation_mapping", mapping)
        object.__setattr__(self, "proposed_set_name", proposed_name)


@dataclass(frozen=True, slots=True)
class DatasetMappingBatchProposal:
    """One immutable, ordered proposal covering every committed dataset."""

    tab_id: str
    proposals: tuple[DatasetMappingProposal, ...]

    def __post_init__(self) -> None:
        tab_id = str(self.tab_id).strip()
        proposals = tuple(self.proposals)
        if not tab_id:
            raise DatasetMappingError("A mapping batch requires an origin tab ID.")
        if not proposals or any(not isinstance(item, DatasetMappingProposal) for item in proposals):
            raise DatasetMappingError("A mapping batch requires one proposal per dataset.")
        dataset_ids = tuple(item.dataset_id for item in proposals)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise DatasetMappingError("A mapping batch cannot contain a dataset twice.")
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "proposals", proposals)


@dataclass(frozen=True)
class SetCopyPlan:
    source_set_id: str
    new_set_id: str
    new_name: str
    values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DatasetSetMappingPlan:
    links: tuple[tuple[str, DatasetLink], ...]
    copied_set: SetCopyPlan | None = None
    canceled: bool = False


@dataclass(frozen=True)
class GlobalDatasetDeletePlan:
    referenced: bool


def resolve_fit_t_start(
    tab: MechanismTabDocument,
    set_id: str,
    *,
    label: str = "Fit inclusion",
) -> float:
    """Resolve and validate only the start-time scalar owned by Fit."""

    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    target = tab.initial_condition_set(set_id).set_id
    settings = dict(tab.run_overrides_by_set_id).get(target, tab.run_defaults)
    raw_t_start = settings.t_start
    if type(raw_t_start) is bool:
        raise DatasetMappingError(
            f"{label} requires a numeric resolved start time."
        )
    try:
        resolved_t_start = float(raw_t_start)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetMappingError(
            f"{label} requires a numeric resolved start time."
        ) from exc
    if not math.isfinite(resolved_t_start) or resolved_t_start < 0.0:
        raise DatasetMappingError(
            f"{label} requires a finite nonnegative resolved start time."
        )
    return resolved_t_start


def export_original_source(
    document: ProjectDocument,
    record: dataset_foundation.RawDatasetRecord,
    destination: str | os.PathLike[str],
) -> Path:
    """Resolve and write a dataset's canonical source bytes to the destination.

    Passing a directory uses the original filename. Passing a file path writes
    to that exact path, allowing the integration-owned save dialog to control
    overwrite confirmation.
    """
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    try:
        validated_record = dataset_foundation.validate_raw_dataset_record(record)
        source = document.imported_source(validated_record.source_id)
        source = dataset_foundation.validate_imported_source(source)
    except dataset_foundation.RawDatasetValidationError as exc:
        raise DatasetMappingError(str(exc)) from exc
    original_filename = source.original_filename
    source_bytes = source.exact_source_bytes
    target = Path(destination)
    if target.is_dir():
        target = target / original_filename
    if not target.parent.is_dir():
        raise DatasetMappingError(
            f"Export directory does not exist: {target.parent}."
        )
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(source_bytes)
        try:
            existing_mode = stat.S_IMODE(target.stat().st_mode)
        except FileNotFoundError:
            pass
        else:
            os.chmod(temporary, existing_mode)
        os.replace(temporary, target)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def create_set_from_exact_zero(
    *,
    tab: MechanismTabDocument,
    dataset: dataset_foundation.RawDatasetRecord,
    set_id: str,
    set_name: str,
    model_targets: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
) -> MechanismTabDocument:
    """Create and map one complete IC set from confirmed exact-zero values.

    model_targets is the accepted observation-target roster.  The mechanism
    species roster remains canonical state owned by tab and is the only roster
    materialized into the new initial-condition set.
    """
    clean_set_id = str(set_id).strip()
    clean_name = str(set_name).strip()
    if not clean_set_id or not clean_name:
        raise DatasetMappingError("The new initial-condition set requires a stable ID and name.")
    if any(existing.set_id == clean_set_id for existing in tab.initial_condition_sets):
        raise DatasetMappingError(f"Initial-condition set ID '{clean_set_id}' already exists.")
    if any(existing.name.casefold() == clean_name.casefold() for existing in tab.initial_condition_sets):
        raise DatasetMappingError(f"An initial-condition set named '{clean_name}' already exists.")
    observations = {
        series.raw_observation_id: {
            "raw_column": series.column_name,
            "t": series.t,
            "y": series.y,
            "source_rows": series.source_rows,
        }
        for series in dataset.observations
    }
    normalized_mapping = normalize_observation_mapping(
        confirmed_mapping,
        raw_observations=tuple(
            (series.raw_observation_id, series.column_name)
            for series in dataset.observations
        ),
        model_targets=model_targets,
    )
    values = seed_species_values_at_exact_zero(
        observations=observations,
        mechanism_species=canonical_species_names(tab),
        model_targets=model_targets,
        confirmed_mapping=normalized_mapping,
    )
    created_set = InitialConditionSet(
        set_id=clean_set_id,
        name=clean_name,
        values=values,
    )
    with_set = tab.with_initial_condition_sets((*tab.initial_condition_sets, created_set))
    return with_set.with_dataset_link(
        dataset.dataset_id,
        DatasetLink(
            set_id=clean_set_id,
            observation_mapping=normalized_mapping,
            include_in_fit=False,
        ),
    )


def allocate_initial_condition_set_name(
    display_name: str,
    *,
    existing_names: Iterable[str],
) -> str:
    """Allocate a display-derived set name with extension-aware suffixes."""
    raw_name = str(display_name).strip()
    if not raw_name:
        raise DatasetMappingError("Create Set requires a nonblank dataset display name.")
    stem, dot, suffix = raw_name.rpartition(".")
    base = stem if dot and stem else raw_name
    extension = f".{suffix}" if dot and stem else ""
    occupied = {str(name).strip().casefold() for name in existing_names}
    candidate = raw_name
    counter = 1
    while candidate.casefold() in occupied:
        candidate = f"{base}_{counter}{extension}"
        counter += 1
    return candidate


def apply_dataset_mapping_batch(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    batch: DatasetMappingBatchProposal,
    model_targets: Sequence[str],
) -> MechanismTabDocument:
    """Validate and fold one complete mapping batch into one tab candidate.

    No ProjectDocument or live tab is mutated while the proposal is folded;
    callers publish the returned candidate exactly once through the registry.
    model_targets admits raw observation mappings; the tab's canonical
    mechanism species remain the sole initial-condition roster.
    """
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be a ProjectDocument.")
    if not isinstance(tab, MechanismTabDocument):
        raise TypeError("tab must be a MechanismTabDocument.")
    if not isinstance(batch, DatasetMappingBatchProposal):
        raise TypeError("batch must be a DatasetMappingBatchProposal.")
    if batch.tab_id != tab.tab_id:
        raise DatasetMappingError("Mapping batch origin does not match its tab.")
    mechanism_species = canonical_species_names(tab)
    observation_targets = tuple(str(item).strip() for item in model_targets)
    species = mechanism_species
    if not species or any(not item for item in species) or len(set(species)) != len(species):
        raise DatasetMappingError("Mapping requires a unique mechanism species roster.")
    if (
        not observation_targets
        or any(not item for item in observation_targets)
        or len(set(observation_targets)) != len(observation_targets)
    ):
        raise DatasetMappingError("Mapping requires a unique model target roster.")
    records = {record.dataset_id: record for record in document.datasets}
    if any(item.dataset_id not in records for item in batch.proposals):
        raise DatasetMappingError("Mapping batch references a dataset absent from the committed document.")
    current = dict(
        validate_partial_bijection(
            tab.dataset_links,
            known_dataset_ids=records,
            known_set_ids=(item.set_id for item in tab.initial_condition_sets),
        )
    )
    sets = list(tab.initial_condition_sets)
    used_names = {item.name.casefold() for item in sets}
    for item in batch.proposals:
        dataset = records[item.dataset_id]
        selected = current.get(item.dataset_id, DatasetLink())
        normalized_mapping = normalize_observation_mapping(
            item.observation_mapping,
            raw_observations=tuple(
                (series.raw_observation_id, series.column_name)
                for series in dataset.observations
            ),
            model_targets=observation_targets,
        )
        if item.action is DatasetMappingChoice.LEAVE_UNMAPPED:
            if item.target_set_id is not None or item.occupied_choice is not None:
                raise DatasetMappingError("Leave Unmapped cannot include a target or occupancy choice.")
            current[item.dataset_id] = DatasetLink(
                set_id=None,
                observation_mapping=normalized_mapping,
                include_in_fit=False,
            )
            continue
        if item.action is DatasetMappingChoice.CREATE_SET:
            if item.occupied_choice is not None:
                raise DatasetMappingError("Create Set cannot include an occupancy choice.")
            expected_name = allocate_initial_condition_set_name(
                dataset.display_name,
                existing_names=used_names,
            )
            if item.proposed_set_name != expected_name:
                raise DatasetMappingError(
                    "Create Set name is not the canonical display-derived name."
                )
            observations = {
                series.raw_observation_id: {
                    "raw_column": series.column_name,
                    "t": series.t,
                    "y": series.y,
                    "source_rows": series.source_rows,
                }
                for series in dataset.observations
            }
            values = seed_species_values_at_exact_zero(
                observations=observations,
                mechanism_species=species,
                model_targets=observation_targets,
                confirmed_mapping=normalized_mapping,
            )
            new_set = InitialConditionSet(
                set_id=new_set_id(),
                name=expected_name,
                values=values,
            )
            sets.append(new_set)
            used_names.add(expected_name.casefold())
            current[item.dataset_id] = DatasetLink(
                set_id=new_set.set_id,
                observation_mapping=normalized_mapping,
                include_in_fit=False,
            )
            continue
        target = item.target_set_id
        if target is None:
            raise DatasetMappingError("Map Existing requires a target set.")
        try:
            target_set = next(current_set for current_set in sets if current_set.set_id == target)
        except StopIteration as exc:
            raise DatasetMappingError(f"Unknown target set {target!r}.") from exc
        occupant = next(
            (dataset_id for dataset_id, link in current.items()
             if dataset_id != item.dataset_id and link.set_id == target),
            None,
        )
        destination = target
        if occupant is not None:
            if item.occupied_choice is OccupiedSetChoice.CANCEL:
                current[item.dataset_id] = DatasetLink(
                    set_id=None,
                    observation_mapping=normalized_mapping,
                    include_in_fit=False,
                )
                continue
            if item.occupied_choice is OccupiedSetChoice.OVERRIDE:
                old = current[occupant]
                current[occupant] = DatasetLink(
                    set_id=None,
                    observation_mapping=old.observation_mapping,
                    include_in_fit=False,
                )
            elif item.occupied_choice is OccupiedSetChoice.CREATE_COPY:
                expected_name = allocate_initial_condition_set_name(
                    target_set.name,
                    existing_names=used_names,
                )
                if item.proposed_set_name != expected_name:
                    raise DatasetMappingError(
                        "Copied set name is not the canonical collision-safe name."
                    )
                destination = new_set_id()
                sets.append(
                    InitialConditionSet(
                        set_id=destination,
                        name=expected_name,
                        values=target_set.values,
                    )
                )
                used_names.add(expected_name.casefold())
            else:
                raise DatasetMappingError("An occupied target requires an explicit disposition.")
        elif item.occupied_choice is not None:
            raise DatasetMappingError("An unoccupied target cannot include an occupancy choice.")
        current[item.dataset_id] = DatasetLink(
            set_id=destination,
            observation_mapping=normalized_mapping,
            include_in_fit=selected.include_in_fit,
        )
    candidate = replace(
        tab,
        initial_condition_sets=tuple(sets),
        dataset_links=validate_partial_bijection(
            current.items(),
            known_dataset_ids=records,
            known_set_ids=(item.set_id for item in sets),
        ),
    )
    return revalidate_fit_inclusions(
        document=document,
        tab=candidate,
        model_targets=observation_targets,
    )


def apply_dataset_set_mapping_plan(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    plan: DatasetSetMappingPlan,
    model_targets: Sequence[str],
) -> MechanismTabDocument:
    """Materialize a mapping plan and drop every newly invalid Fit inclusion.

    model_targets is the accepted observation-target roster; mechanism species
    are resolved from the canonical tab for Fit admission.
    """
    if plan.canceled:
        return tab
    sets = tab.initial_condition_sets
    if plan.copied_set is not None:
        copy = plan.copied_set
        if any(item.set_id == copy.new_set_id for item in sets):
            raise DatasetMappingError(f"Initial-condition set ID '{copy.new_set_id}' already exists.")
        if any(item.name.casefold() == copy.new_name.casefold() for item in sets):
            raise DatasetMappingError(f"An initial-condition set named '{copy.new_name}' already exists.")
        sets = (
            *sets,
            InitialConditionSet(
                set_id=copy.new_set_id,
                name=copy.new_name,
                values=copy.values,
            ),
        )
    candidate = replace(
        tab,
        initial_condition_sets=tuple(sets),
        dataset_links=plan.links,
    )
    return revalidate_fit_inclusions(
        document=document,
        tab=candidate,
        model_targets=model_targets,
    )


def with_confirmed_observation_mapping(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset: dataset_foundation.RawDatasetRecord,
    model_targets: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
) -> MechanismTabDocument:
    """Replace one active-tab mapping against the accepted target roster."""
    normalized = normalize_observation_mapping(
        confirmed_mapping,
        raw_observations=tuple(
            (series.raw_observation_id, series.column_name)
            for series in dataset.observations
        ),
        model_targets=model_targets,
    )
    current = tab.dataset_link(dataset.dataset_id)
    if current is None and not normalized:
        return tab
    current = current or DatasetLink()
    candidate = tab.with_dataset_link(
        dataset.dataset_id,
        DatasetLink(
            set_id=current.set_id,
            observation_mapping=normalized,
            include_in_fit=current.include_in_fit,
        ),
    )
    return revalidate_fit_inclusions(
        document=document,
        tab=candidate,
        model_targets=model_targets,
    )


def _validate_current_fit_inclusion(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset_id: str,
    model_targets: Sequence[str],
) -> dataset_foundation.FitInclusionValidation:
    """Resolve mapping-owned context before calling the lower pure validator."""
    try:
        dataset = document.dataset(dataset_id)
    except KeyError as exc:
        raise DatasetMappingError(
            f"Dataset {dataset_id!r} is no longer present."
        ) from exc
    current = tab.dataset_link(dataset.dataset_id)
    if current is None or current.set_id is None:
        raise DatasetMappingError(
            "Map the dataset to an initial-condition set before including it in Fit."
        )
    try:
        initial_set = tab.initial_condition_set(current.set_id)
    except KeyError as exc:
        raise DatasetMappingError(
            f"Mapped initial-condition set {current.set_id!r} is no longer present."
        ) from exc
    normalized_mapping = normalize_observation_mapping(
        current.observation_mapping,
        raw_observations=tuple(
            (series.raw_observation_id, series.column_name)
            for series in dataset.observations
        ),
        model_targets=model_targets,
    )
    if normalized_mapping != current.observation_mapping:
        raise DatasetMappingError(
            "Fit inclusion observation mapping is not canonical."
        )
    return dataset_foundation.validate_fit_inclusion(
        dataset=dataset,
        set_id=initial_set.set_id,
        observation_mapping=normalized_mapping,
        initial_values=dict(canonical_initial_values(tab, initial_set.set_id)),
        mechanism_species=canonical_species_names(tab),
        observation_targets=model_targets,
        resolved_t_start=resolve_fit_t_start(
            tab,
            initial_set.set_id,
            label=f"Fit inclusion for dataset {dataset.dataset_id!r}",
        ),
    )


def revalidate_fit_inclusions(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    model_targets: Sequence[str],
) -> MechanismTabDocument:
    """Remove invalid included links after any dataset/set prerequisite change."""
    candidate = tab
    for dataset_id, link in tab.dataset_links:
        if not link.include_in_fit:
            continue
        try:
            _validate_current_fit_inclusion(
                document=document,
                tab=candidate,
                dataset_id=dataset_id,
                model_targets=model_targets,
            )
        except (KeyError, TypeError, ValueError):
            current = candidate.dataset_link(dataset_id)
            if current is not None:
                candidate = candidate.with_dataset_link(
                    dataset_id,
                    replace(current, include_in_fit=False),
                )
    return candidate


def with_validated_fit_inclusion(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset_id: str,
    include: bool,
    model_targets: Sequence[str],
) -> MechanismTabDocument:
    """Replace inclusion only after complete current-state validation."""
    current = tab.dataset_link(dataset_id)
    if not include:
        if current is None:
            return tab
    else:
        _validate_current_fit_inclusion(
            document=document,
            tab=tab,
            dataset_id=dataset_id,
            model_targets=model_targets,
        )
    if current is None:
        raise DatasetMappingError(
            f"Dataset {dataset_id!r} has no active-tab link."
        )
    return tab.with_dataset_link(
        dataset_id,
        DatasetLink(
            set_id=current.set_id,
            observation_mapping=current.observation_mapping,
            include_in_fit=include,
        ),
    )


def validate_unique_dataset_name(
    candidate: str,
    *,
    existing: Iterable[tuple[str, str]],
    excluding_dataset_id: str | None = None,
) -> str:
    """Return a stripped globally unique name or raise without mutation."""
    clean = str(candidate).strip()
    if not clean:
        raise DatasetMappingError("Dataset display name cannot be blank.")
    folded = clean.casefold()
    for dataset_id, display_name in existing:
        if excluding_dataset_id is not None and str(dataset_id) == str(excluding_dataset_id):
            continue
        if str(display_name).strip().casefold() == folded:
            raise DatasetMappingError(f"A dataset named '{clean}' already exists.")
    return clean


def validate_partial_bijection(
    links: Iterable[tuple[str, DatasetLink]],
    *,
    known_dataset_ids: Iterable[str] | None = None,
    known_set_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, DatasetLink], ...]:
    """Validate one tab's dataset links and return deterministic ID order."""
    known_datasets = None if known_dataset_ids is None else {str(value) for value in known_dataset_ids}
    known_sets = None if known_set_ids is None else {str(value) for value in known_set_ids}
    by_dataset: dict[str, DatasetLink] = {}
    dataset_by_set: dict[str, str] = {}
    for raw_dataset_id, raw_link in links:
        dataset_id = str(raw_dataset_id).strip()
        if not dataset_id:
            raise DatasetMappingError("A dataset link requires a nonblank dataset ID.")
        if known_datasets is not None and dataset_id not in known_datasets:
            raise DatasetMappingError(f"Dataset link references unknown dataset '{dataset_id}'.")
        if dataset_id in by_dataset:
            raise DatasetMappingError(f"Dataset '{dataset_id}' has more than one link in this tab.")
        set_id = str(raw_link.set_id).strip() if raw_link.set_id is not None else None
        if set_id == "":
            set_id = None
        if set_id is not None:
            if known_sets is not None and set_id not in known_sets:
                raise DatasetMappingError(f"Dataset link references unknown set '{set_id}'.")
            occupying_dataset = dataset_by_set.get(set_id)
            if occupying_dataset is not None:
                raise DatasetMappingError(
                    f"Set '{set_id}' is mapped to both '{occupying_dataset}' and '{dataset_id}'."
                )
            dataset_by_set[set_id] = dataset_id
        elif raw_link.include_in_fit:
            raise DatasetMappingError("An unmapped dataset cannot be included in Fit.")
        link = DatasetLink(
            set_id=set_id,
            observation_mapping=tuple(raw_link.observation_mapping),
            include_in_fit=bool(raw_link.include_in_fit),
        )
        by_dataset[dataset_id] = link
    return tuple((key, by_dataset[key]) for key in sorted(by_dataset))


def plan_dataset_to_set_mapping(
    *,
    links: Iterable[tuple[str, DatasetLink]],
    dataset_id: str,
    target_set_id: str,
    occupied_choice: OccupiedSetChoice | None,
    copy_plan: SetCopyPlan | None = None,
) -> DatasetSetMappingPlan:
    """Plan Map Existing after the occupied-set choice has been presented."""
    target_dataset = str(dataset_id).strip()
    target_set = str(target_set_id).strip()
    if not target_dataset or not target_set:
        raise DatasetMappingError("Mapping requires a dataset ID and set ID.")
    current = dict(validate_partial_bijection(links))
    selected = current.get(target_dataset, DatasetLink())
    occupant = next(
        ((current_dataset_id, link) for current_dataset_id, link in current.items()
         if link.set_id == target_set and current_dataset_id != target_dataset),
        None,
    )
    if occupant is not None and occupied_choice == OccupiedSetChoice.CANCEL:
        return DatasetSetMappingPlan(
            links=tuple((key, current[key]) for key in sorted(current)),
            canceled=True,
        )
    destination_set = target_set
    copied_set: SetCopyPlan | None = None
    if occupant is not None:
        occupant_dataset_id, occupant_link = occupant
        if occupied_choice == OccupiedSetChoice.OVERRIDE:
            current[occupant_dataset_id] = DatasetLink(
                set_id=None,
                observation_mapping=occupant_link.observation_mapping,
                include_in_fit=False,
            )
        elif occupied_choice == OccupiedSetChoice.CREATE_COPY:
            if copy_plan is None or copy_plan.source_set_id != target_set:
                raise DatasetMappingError("Create Copy requires a complete copy plan for the occupied set.")
            destination_set = str(copy_plan.new_set_id).strip()
            if not destination_set or destination_set == target_set:
                raise DatasetMappingError("The copied set requires a new stable set ID.")
            copied_set = copy_plan
        else:
            raise DatasetMappingError("An occupied set requires Override, Create Copy, or Cancel.")
    current[target_dataset] = DatasetLink(
        set_id=destination_set,
        observation_mapping=selected.observation_mapping,
        include_in_fit=selected.include_in_fit,
    )
    return DatasetSetMappingPlan(
        links=validate_partial_bijection(current.items()),
        copied_set=copied_set,
    )


def plan_leave_unmapped(
    *,
    links: Iterable[tuple[str, DatasetLink]],
    dataset_id: str,
) -> DatasetSetMappingPlan:
    """Confirm that an already-unmapped raw record stays unmapped."""
    target = str(dataset_id)
    current = dict(validate_partial_bijection(links))
    selected = current.get(target)
    if selected is None:
        return DatasetSetMappingPlan(
            links=tuple((key, current[key]) for key in sorted(current))
        )
    if selected.set_id is not None:
        raise DatasetMappingError(
            "A mapped dataset cannot be left unmapped; remap it or delete it globally."
        )
    current[target] = DatasetLink(
        set_id=None,
        observation_mapping=selected.observation_mapping,
        include_in_fit=False,
    )
    return DatasetSetMappingPlan(links=validate_partial_bijection(current.items()))


def plan_global_dataset_delete(
    *,
    dataset_id: str,
    links_by_tab: Mapping[str, Sequence[tuple[str, DatasetLink]]],
    overlays_by_tab: Mapping[str, Sequence[tuple[str, Sequence[str]]]],
) -> GlobalDatasetDeletePlan:
    """Report whether direct global deletion needs user confirmation.

    ProjectDocument.remove_dataset and presentation reconciliation perform the
    actual atomic cleanup. This value carries only the fact its real caller
    uses; it does not duplicate the replacement document or presentation.
    """
    target = str(dataset_id).strip()
    if not target:
        raise DatasetMappingError("Dataset deletion requires a stable dataset ID.")
    tab_ids = tuple(dict.fromkeys((*links_by_tab.keys(), *overlays_by_tab.keys())))
    referenced = False
    for raw_tab_id in tab_ids:
        links = validate_partial_bijection(links_by_tab.get(raw_tab_id, ()))
        if any(current_dataset_id == target for current_dataset_id, _link in links):
            referenced = True
        if any(
            str(raw_dataset_id) == target
            for raw_dataset_id, _raw_columns in overlays_by_tab.get(raw_tab_id, ())
        ):
            referenced = True
    return GlobalDatasetDeletePlan(referenced=referenced)

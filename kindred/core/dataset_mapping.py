"""Pure plans for project-global datasets and per-tab dataset links."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping, Sequence
import uuid

from kindred.core.datasets.observation_mapping import (
    InitialConcentrationSource,
    extract_initial_concentrations,
    normalize_observation_mapping,
)
from kindred.core.datasets import foundation as dataset_foundation
from kindred.core.project_document import (
    DatasetLink,
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    canonical_species_names,
    new_set_id,
)

class DatasetMappingError(ValueError):
    """Raised when a proposed dataset mutation violates a real invariant."""


class DatasetMappingChoice(str, Enum):
    CREATE_SET = "create_set"
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
    action: DatasetMappingChoice
    target_set_id: str | None = None
    occupied_choice: OccupiedSetChoice | str | None = None
    observation_mapping: tuple[tuple[str, str], ...] = ()
    initial_concentration_source: InitialConcentrationSource | None = None

    def __post_init__(self) -> None:
        # Producers pass the occupancy choice as its string value.
        if self.occupied_choice is not None:
            object.__setattr__(
                self, "occupied_choice", OccupiedSetChoice(self.occupied_choice)
            )


@dataclass(frozen=True, slots=True)
class DatasetMappingBatchProposal:
    """One immutable, ordered proposal covering every committed dataset."""

    tab_id: str
    proposals: tuple[DatasetMappingProposal, ...]


@dataclass(frozen=True)
class GlobalDatasetDeletePlan:
    referenced: bool


def export_original_source(
    source: dataset_foundation.ImportedSource,
    destination: str | os.PathLike[str],
) -> Path:
    """Write an admitted source's exact bytes to the destination file path.

    The integration-owned save dialog chooses the path and controls overwrite
    confirmation.
    """
    source_bytes = source.exact_source_bytes
    target = Path(destination)
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
    mechanism_species = canonical_species_names(tab)
    if not mechanism_species:
        raise DatasetMappingError("Mapping requires a mechanism with at least one species.")
    observation_targets = tuple(model_targets)
    species = mechanism_species
    records = {record.dataset_id: record for record in document.datasets}
    current = dict(tab.dataset_links)
    sets = list(tab.initial_condition_sets)
    used_names = {item.name.casefold() for item in sets}
    for item in batch.proposals:
        dataset = records[item.dataset_id]
        normalized_mapping = normalize_observation_mapping(
            item.observation_mapping,
            raw_observations=tuple(
                (series.raw_observation_id, series.column_name)
                for series in dataset.observations
            ),
            model_targets=observation_targets,
        )
        if item.action is DatasetMappingChoice.LEAVE_UNMAPPED:
            current[item.dataset_id] = DatasetLink(
                set_id=None,
                observation_mapping=normalized_mapping,
            )
            continue
        if item.action is DatasetMappingChoice.CREATE_SET:
            expected_name = allocate_initial_condition_set_name(
                dataset.display_name,
                existing_names=used_names,
            )
            values = extract_initial_concentrations(
                observations=dataset.observations,
                mechanism_species=species,
                confirmed_mapping=normalized_mapping,
                source=item.initial_concentration_source,
            )
            new_set = InitialConditionSet(
                set_id=new_set_id(),
                name=expected_name,
                values=tuple((name, 0.0 if value is None else value) for name, value in values),
            )
            sets.append(new_set)
            used_names.add(expected_name.casefold())
            current[item.dataset_id] = DatasetLink(
                set_id=new_set.set_id,
                observation_mapping=normalized_mapping,
            )
            continue
        target = item.target_set_id
        if target is None:
            raise DatasetMappingError("Map Existing requires a target set.")
        target_set = next(current_set for current_set in sets if current_set.set_id == target)
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
                )
                continue
            if item.occupied_choice is OccupiedSetChoice.OVERRIDE:
                old = current[occupant]
                current[occupant] = DatasetLink(
                    set_id=None,
                    observation_mapping=old.observation_mapping,
                )
            elif item.occupied_choice is OccupiedSetChoice.CREATE_COPY:
                expected_name = allocate_initial_condition_set_name(
                    target_set.name,
                    existing_names=used_names,
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
        )
    candidate = replace(
        tab,
        initial_condition_sets=tuple(sets),
        dataset_links=tuple(sorted(current.items())),
    )
    return candidate


def with_confirmed_observation_mapping(
    *,
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
        ),
    )
    return candidate


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
        links = links_by_tab.get(raw_tab_id, ())
        if any(current_dataset_id == target for current_dataset_id, _link in links):
            referenced = True
        if any(
            str(raw_dataset_id) == target
            for raw_dataset_id, _raw_columns in overlays_by_tab.get(raw_tab_id, ())
        ):
            referenced = True
    return GlobalDatasetDeletePlan(referenced=referenced)

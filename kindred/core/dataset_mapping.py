"""Pure plans for project-global datasets and per-tab dataset links."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kindred.core.datasets.observation_mapping import (
    InitialConcentrationSource,
    ObservationMappingError,
    extract_initial_concentrations,
    initial_time,
    normalize_observation_mapping,
    supplies_concentration,
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
from kindred.io.paths import create_sibling_temporary, replace_destination

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
    # Manual species do not read the data; None supplies no concentration.
    manual_initial_concentrations: tuple[tuple[str, float | None], ...] = ()

    def __post_init__(self) -> None:
        # Producers pass the occupancy choice as its string value.
        if self.occupied_choice is not None:
            object.__setattr__(
                self, "occupied_choice", OccupiedSetChoice(self.occupied_choice)
            )


@dataclass(frozen=True, slots=True)
class DatasetMappingBatchProposal:
    """One immutable, ordered proposal covering every committed dataset."""

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
    descriptor: int | None
    descriptor, temporary = create_sibling_temporary(target)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(source_bytes)
        replace_destination(temporary, target)
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
    """Return ``display_name``, or when a set already uses it, regardless of
    case, the name followed by the first free ``_2``, ``_3`` and so on."""
    # Dataset display names and set names are admitted nonblank and
    # edge-normalized.
    occupied = {name.casefold() for name in existing_names}
    candidate = display_name
    counter = 2
    while candidate.casefold() in occupied:
        candidate = f"{display_name}_{counter}"
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
    overrides = dict(tab.run_overrides_by_set_id)
    defaults = tab.run_defaults
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
            manual = dict(item.manual_initial_concentrations)
            data_mapping = tuple(
                (raw_id, target) for raw_id, target in normalized_mapping
                if target not in manual
            )
            if item.initial_concentration_source is not None:
                values = extract_initial_concentrations(
                    observations=dataset.observations,
                    mechanism_species=species,
                    confirmed_mapping=data_mapping,
                    source=item.initial_concentration_source,
                )
            else:
                # Without a source choice no species may read the data.
                series_by_id = {series.raw_observation_id: series for series in dataset.observations}
                if any(
                    target in species and supplies_concentration(series_by_id[raw_id])
                    for raw_id, target in data_mapping
                ):
                    raise ObservationMappingError("Choose a source for the initial concentrations.")
                values = tuple((name, None) for name in species)
            values = tuple((name, manual.get(name, value)) for name, value in values)
            new_set = InitialConditionSet(
                set_id=new_set_id(),
                name=expected_name,
                values=tuple((name, 0.0 if value is None else value) for name, value in values),
            )
            sets.append(new_set)
            used_names.add(expected_name.casefold())
            # Measured values describe the state at their recorded time, so the
            # set's Run starts there; the end keeps the tab's, or its duration.
            start = None if item.initial_concentration_source is None else initial_time(
                dataset.observations, item.initial_concentration_source
            )
            if start is not None and start != defaults.t_start:
                end = defaults.t_end if defaults.t_end > start else start + defaults.t_end - defaults.t_start
                overrides[new_set.set_id] = replace(defaults, t_start=start, t_end=end)
            current[item.dataset_id] = DatasetLink(
                set_id=new_set.set_id,
                observation_mapping=normalized_mapping,
            )
            continue
        # Both mapping screens require a target set before Map Existing.
        target = item.target_set_id
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
                # The values describe the state at the original's Run start.
                if target in overrides:
                    overrides[destination] = overrides[target]
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
        run_overrides_by_set_id=tuple(overrides.items()),
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
    clean = candidate.strip()
    if not clean:
        raise DatasetMappingError("Dataset display name cannot be blank.")
    folded = clean.casefold()
    for dataset_id, display_name in existing:
        if dataset_id == excluding_dataset_id:
            continue
        if display_name.strip().casefold() == folded:
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
    # Both mappings cover the same tabs.
    target = dataset_id
    referenced = False
    for raw_tab_id, links in links_by_tab.items():
        if any(current_dataset_id == target for current_dataset_id, _link in links):
            referenced = True
        if any(
            raw_dataset_id == target
            for raw_dataset_id, _raw_columns in overlays_by_tab[raw_tab_id]
        ):
            referenced = True
    return GlobalDatasetDeletePlan(referenced=referenced)

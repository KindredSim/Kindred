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
from kindred.core.datasets.csv_import import (
    identify_physical_columns,
    validate_tabular_source_bytes,
)
from kindred.core.project_document import (
    DatasetLink,
    InitialConditionSet,
    MechanismTabDocument,
    ObservationSeries,
    ProjectDocument,
    RawDatasetRecord,
    SimulationExecutionSettings,
    thaw_json_value,
    validate_simulation_execution_settings,
)

_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_PHYSICAL_COLUMNS = 256
_MAX_RETAINED_OBSERVATION_CELLS = 1_000_000


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
    dataset_id: str
    referenced: bool


@dataclass(frozen=True, slots=True)
class DatasetMappedTarget:
    dataset_id: str
    raw_observation_id: str
    raw_column: str
    model_target: str
    t: tuple[float, ...]
    y: tuple[float, ...]
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if (
            not isinstance(self.raw_observation_id, str)
            or not self.raw_observation_id
            or self.raw_observation_id != self.raw_observation_id.strip()
        ):
            raise ValueError(
                "Mapped dataset target raw observation identity must be a "
                "nonblank edge-normalized string."
            )
        raw_observation_id = self.raw_observation_id
        raw_column = str(self.raw_column).strip()
        model_target = str(self.model_target).strip()
        if (
            not dataset_id
            or not raw_observation_id
            or not raw_column
            or not model_target
        ):
            raise ValueError("Mapped dataset target identities must be nonblank.")
        if (
            type(self.t) is not tuple
            or type(self.y) is not tuple
            or type(self.source_rows) is not tuple
            or not self.t
            or len(self.t) != len(self.y)
            or len(self.t) != len(self.source_rows)
        ):
            raise ValueError(
                "Mapped dataset target time, value, and source-row tuples "
                "must be immutable and have equal nonzero length."
            )
        if (
            any(
                type(value) is not float or not math.isfinite(value)
                for value in (*self.t, *self.y)
            )
            or any(
                type(row) is not int or row <= 0
                for row in self.source_rows
            )
            or len(set(self.source_rows)) != len(self.source_rows)
        ):
            raise ValueError(
                "Mapped dataset targets require finite values and unique "
                "positive physical source rows."
            )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(
            self,
            "raw_observation_id",
            raw_observation_id,
        )
        object.__setattr__(self, "raw_column", raw_column)
        object.__setattr__(self, "model_target", model_target)


@dataclass(frozen=True, slots=True)
class FitInclusionValidation:
    dataset_id: str
    set_id: str
    execution: SimulationExecutionSettings
    targets: tuple[DatasetMappedTarget, ...]

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not targets or any(
            target.dataset_id != self.dataset_id for target in targets
        ):
            raise ValueError(
                "Fit inclusion targets must be nonempty and share the stable "
                "dataset ID."
            )
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class FitInclusionRevalidation:
    tab: MechanismTabDocument
    excluded: tuple[tuple[str, str], ...]


def validate_raw_dataset_record(
    record: RawDatasetRecord,
) -> RawDatasetRecord:
    """Fail closed unless one immutable raw record satisfies Dataset law."""
    if not isinstance(record, RawDatasetRecord):
        raise TypeError("record must be a RawDatasetRecord.")
    if (
        not isinstance(record.dataset_id, str)
        or not record.dataset_id
        or record.dataset_id != record.dataset_id.strip()
    ):
        raise DatasetMappingError(
            "Raw dataset ID must be a nonblank edge-normalized string."
        )
    if (
        not isinstance(record.display_name, str)
        or not record.display_name
        or record.display_name != record.display_name.strip()
    ):
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} display name must be a "
            "nonblank edge-normalized string."
        )
    if record.original_format not in {"csv", "xlsx"}:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} original_format must be "
            "exactly 'csv' or 'xlsx'."
        )
    if (
        not isinstance(record.original_filename, str)
        or not record.original_filename
        or record.original_filename != record.original_filename.strip()
        or os.path.basename(record.original_filename) != record.original_filename
    ):
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} original filename must be a "
            "nonblank basename."
        )
    expected_suffix = f".{record.original_format}"
    if Path(record.original_filename).suffix.lower() != expected_suffix:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} original filename must end "
            f"in {expected_suffix!r}."
        )
    if type(record.exact_source_bytes) is not bytes:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} exact source must be immutable "
            "bytes."
        )
    if not record.exact_source_bytes:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} exact source bytes must be "
            "nonempty."
        )
    if len(record.exact_source_bytes) > _MAX_SOURCE_BYTES:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} exact source exceeds the "
            "32 MiB supported-size limit."
        )
    try:
        validate_tabular_source_bytes(
            record.exact_source_bytes,
            source_format=record.original_format,
            source_label=record.original_filename,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} exact source admission failed: "
            f"{exc}"
        ) from exc
    if type(record.observations) is not tuple:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} observations must be an "
            "immutable tuple."
        )
    if not record.observations:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} requires at least one "
            "observation series."
        )

    thawed_metadata = thaw_json_value(record.source_metadata)
    if not isinstance(thawed_metadata, dict):
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} source metadata must be an "
            "immutable JSON object."
        )
    physical_values = thawed_metadata.get("physical_columns")
    if not isinstance(physical_values, list) or not physical_values:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} must retain its physical "
            "source-column catalog."
        )
    if len(physical_values) > _MAX_PHYSICAL_COLUMNS:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} exceeds the 256 physical-"
            "column supported-size limit."
        )
    physical_rows: list[tuple[int, str, str]] = []
    for index, raw_physical in enumerate(physical_values):
        if (
            not isinstance(raw_physical, dict)
            or set(raw_physical)
            != {"source_index", "authored_name", "assigned_name"}
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} physical column {index} "
                "has an invalid schema."
            )
        source_index = raw_physical["source_index"]
        authored_name = raw_physical["authored_name"]
        assigned_name = raw_physical["assigned_name"]
        if (
            type(source_index) is not int
            or source_index != index
            or not isinstance(authored_name, str)
            or not authored_name
            or authored_name != authored_name.strip()
            or not isinstance(assigned_name, str)
            or not assigned_name
            or assigned_name != assigned_name.strip()
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} physical column {index} "
                "does not retain exact ordered source identity."
            )
        physical_rows.append(
            (source_index, authored_name, assigned_name)
        )
    expected_columns = identify_physical_columns(
        [authored_name for _index, authored_name, _assigned in physical_rows]
    )
    expected_rows = tuple(
        (
            column.source_index,
            column.authored_name,
            column.assigned_name,
        )
        for column in expected_columns
    )
    if tuple(physical_rows) != expected_rows:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} physical column names do not "
            "match the canonical case-sensitive duplicate allocation."
        )

    observation_names: list[str] = []
    raw_observation_ids: set[str] = set()
    retained_cells = 0
    for series in record.observations:
        if not isinstance(series, ObservationSeries):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observations must contain "
                "only ObservationSeries values."
            )
        name = series.column_name
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation names must be "
                "nonblank edge-normalized strings."
            )
        if name in observation_names:
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} contains duplicate "
                f"assigned observation name {name!r}."
            )
        raw_observation_id = series.raw_observation_id
        if (
            not isinstance(raw_observation_id, str)
            or not raw_observation_id
            or raw_observation_id != raw_observation_id.strip()
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation {name!r} "
                "requires a nonblank edge-normalized raw observation ID."
            )
        if raw_observation_id in raw_observation_ids:
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} contains duplicate raw "
                f"observation ID {raw_observation_id!r}."
            )
        raw_observation_ids.add(raw_observation_id)
        if (
            type(series.t) is not tuple
            or type(series.y) is not tuple
            or type(series.source_rows) is not tuple
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation {name!r} "
                "must own immutable t/y/source_rows tuples."
            )
        if (
            not series.t
            or len(series.t) != len(series.y)
            or len(series.t) != len(series.source_rows)
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (*series.t, *series.y)
            )
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation {name!r} "
                "must retain equal nonempty finite float time/value tuples "
                "aligned with source rows."
            )
        if any(
            type(row) is not int or row <= 0
            for row in series.source_rows
        ):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation {name!r} "
                "source rows must be positive integers."
            )
        if len(set(series.source_rows)) != len(series.source_rows):
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} observation {name!r} "
                "contains duplicate physical source-row identities."
            )
        retained_cells += len(series.t)
        if retained_cells > _MAX_RETAINED_OBSERVATION_CELLS:
            raise DatasetMappingError(
                f"Raw dataset {record.dataset_id!r} exceeds the 1000000 "
                "retained-observation-cell limit."
            )
        observation_names.append(name)
    source_observation_columns = thawed_metadata.get("observation_columns")
    if (
        not isinstance(source_observation_columns, list)
        or tuple(source_observation_columns) != tuple(observation_names)
    ):
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} observation-column metadata "
            "must exactly match its immutable ordered series labels."
        )
    physical_assigned = {
        assigned_name
        for _source_index, _authored_name, assigned_name in physical_rows
    }
    time_column = thawed_metadata.get("time_column")
    if (
        not isinstance(time_column, str)
        or time_column not in physical_assigned
        or time_column in observation_names
        or any(
            name not in physical_assigned
            for name in observation_names
        )
    ):
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} time/observation columns do "
            "not reference retained physical source columns."
        )
    if thawed_metadata.get("mapping_source") not in {"auto", "explicit"}:
        raise DatasetMappingError(
            f"Raw dataset {record.dataset_id!r} mapping source must be exactly "
            "'auto' or 'explicit'."
        )
    sheet_name = thawed_metadata.get("sheet_name")
    if record.original_format == "xlsx":
        if (
            not isinstance(sheet_name, str)
            or not sheet_name
            or not sheet_name.strip()
        ):
            raise DatasetMappingError(
                f"Raw XLSX dataset {record.dataset_id!r} requires its exact "
                "worksheet name."
            )
    elif sheet_name is not None:
        raise DatasetMappingError(
            f"Raw CSV dataset {record.dataset_id!r} cannot claim a worksheet."
        )
    return record


def export_original_source(
    record: RawDatasetRecord,
    destination: str | os.PathLike[str],
) -> Path:
    """Write a raw record's exact imported bytes to the chosen destination.

    Passing a directory uses the original filename. Passing a file path writes
    to that exact path, allowing the integration-owned save dialog to control
    overwrite confirmation.
    """
    validated_record = validate_raw_dataset_record(record)
    original_filename = validated_record.original_filename
    source_bytes = validated_record.exact_source_bytes
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
    dataset: RawDatasetRecord,
    set_id: str,
    set_name: str,
    model_species: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
    mapping_confirmed: bool,
) -> MechanismTabDocument:
    """Create and map one complete IC set from confirmed exact-zero values."""
    if not mapping_confirmed:
        raise DatasetMappingError(
            "Create Set From t=0 requires explicit confirmation of the observation mapping."
        )
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
        model_species=model_species,
    )
    values = seed_species_values_at_exact_zero(
        observations=observations,
        model_species=model_species,
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


def apply_dataset_set_mapping_plan(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    plan: DatasetSetMappingPlan,
    mechanism_species: Sequence[str],
) -> MechanismTabDocument:
    """Materialize a mapping plan and drop every newly invalid Fit inclusion."""
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
        mechanism_species=mechanism_species,
    ).tab


def with_confirmed_observation_mapping(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset: RawDatasetRecord,
    model_species: Sequence[str],
    confirmed_mapping: Iterable[tuple[str, str]],
) -> MechanismTabDocument:
    """Replace only one active-tab explicit observation mapping."""
    normalized = normalize_observation_mapping(
        confirmed_mapping,
        raw_observations=tuple(
            (series.raw_observation_id, series.column_name)
            for series in dataset.observations
        ),
        model_species=model_species,
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
        mechanism_species=model_species,
    ).tab


def validate_fit_inclusion(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset_id: str,
    mechanism_species: Sequence[str],
) -> FitInclusionValidation:
    """Fully validate one Dataset-owned Fit-inclusion boundary."""
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
    roster = tuple(str(species).strip() for species in mechanism_species)
    if not roster or any(not species for species in roster):
        raise DatasetMappingError(
            "Fit inclusion requires a nonempty valid mechanism species roster."
        )
    if len(set(roster)) != len(roster):
        raise DatasetMappingError("Mechanism species identities must be unique.")
    values = initial_set.values_dict()
    if set(values) != set(roster):
        missing = tuple(species for species in roster if species not in values)
        extra = tuple(species for species in values if species not in set(roster))
        raise DatasetMappingError(
            "Mapped initial-condition state is incomplete for the current "
            f"mechanism (missing={missing!r}, extra={extra!r})."
        )
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in values.values()
    ):
        raise DatasetMappingError(
            "Mapped initial-condition state must be finite and nonnegative."
        )
    raw_observations = tuple(
        (series.raw_observation_id, series.column_name)
        for series in dataset.observations
    )
    normalized_mapping = normalize_observation_mapping(
        current.observation_mapping,
        raw_observations=raw_observations,
        model_species=roster,
    )
    if normalized_mapping != current.observation_mapping:
        raise DatasetMappingError(
            "Fit inclusion observation mapping must already be canonical and "
            "ordered by raw observation identity."
        )
    if not normalized_mapping:
        raise DatasetMappingError(
            "Fit inclusion requires at least one valid mapped observation target."
        )
    series_by_raw_id = {
        series.raw_observation_id: series
        for series in dataset.observations
    }
    for raw_observation_id, _model_target in normalized_mapping:
        series = series_by_raw_id[raw_observation_id]
        if (
            not series.source_rows
            or len(series.source_rows) != len(series.t)
            or len(series.t) != len(series.y)
            or any(type(row) is not int or row <= 0 for row in series.source_rows)
        ):
            raise DatasetMappingError(
                f"Observation {series.column_name!r} (raw ID "
                f"{raw_observation_id!r}) does not retain complete positive "
                "physical source-row identity."
            )
    execution = validate_simulation_execution_settings(
        tab.resolved_execution_settings(initial_set.set_id),
        f"Fit inclusion for dataset {dataset.dataset_id!r}",
    )
    return FitInclusionValidation(
        dataset_id=dataset.dataset_id,
        set_id=initial_set.set_id,
        execution=execution,
        targets=tuple(
            DatasetMappedTarget(
                dataset_id=dataset.dataset_id,
                raw_observation_id=raw_observation_id,
                raw_column=series_by_raw_id[
                    raw_observation_id
                ].column_name,
                model_target=model_target,
                t=series_by_raw_id[raw_observation_id].t,
                y=series_by_raw_id[raw_observation_id].y,
                source_rows=series_by_raw_id[
                    raw_observation_id
                ].source_rows,
            )
            for raw_observation_id, model_target in normalized_mapping
        ),
    )


def revalidate_fit_inclusions(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    mechanism_species: Sequence[str],
) -> FitInclusionRevalidation:
    """Remove invalid included links after any dataset/set prerequisite change."""
    candidate = tab
    excluded: list[tuple[str, str]] = []
    for dataset_id, link in tab.dataset_links:
        if not link.include_in_fit:
            continue
        try:
            validate_fit_inclusion(
                document=document,
                tab=candidate,
                dataset_id=dataset_id,
                mechanism_species=mechanism_species,
            )
        except (KeyError, TypeError, ValueError) as exc:
            current = candidate.dataset_link(dataset_id)
            if current is not None:
                candidate = candidate.with_dataset_link(
                    dataset_id,
                    replace(current, include_in_fit=False),
                )
            excluded.append((dataset_id, str(exc)))
    return FitInclusionRevalidation(
        tab=candidate,
        excluded=tuple(excluded),
    )


def with_validated_fit_inclusion(
    *,
    document: ProjectDocument,
    tab: MechanismTabDocument,
    dataset_id: str,
    include: bool,
    mechanism_species: Sequence[str],
) -> MechanismTabDocument:
    """Replace inclusion only after complete current-state validation."""
    current = tab.dataset_link(dataset_id)
    if not include:
        if current is None:
            return tab
    else:
        validate_fit_inclusion(
            document=document,
            tab=tab,
            dataset_id=dataset_id,
            mechanism_species=mechanism_species,
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
    return GlobalDatasetDeletePlan(dataset_id=target, referenced=referenced)

"""Strict decoder and writer for current format 7 archives."""

from __future__ import annotations

import json
import itertools
import math
import os
import stat
import uuid
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    DatasetLink,
    FitDefaults,
    InitialConditionSet,
    MechanismTabDocument,
    ProjectDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
)
from kindred.core.datasets.foundation import (
    ImportedSource,
    ObservationSeries,
    RawDatasetRecord,
    freeze_json_value,
    thaw_json_value,
    validate_imported_source,
    validate_original_basename,
    validate_raw_dataset_record,
)
from kindred.core.datasets.csv_import import (
    parse_physical_rows,
    read_csv_rows_from_bytes,
)
from kindred.core.datasets.excel_import import ExcelWorkbookPair
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import parse_measurement_unit, parse_time_unit
from kindred.core.quantity_units import Unit, scientific_unit
from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    VesselVolumeAxisSelection,
    ProjectPresentationState,
    SimulationSeriesAxisSelection,
    WorkspacePresentationState,
    validate_workspace_presentation_state,
)

__all__ = [
    "ArchiveProject",
    "CURRENT_PROJECT_FORMAT",
    "KinArchiveStore",
]

CURRENT_PROJECT_FORMAT = "kindred-kin-7"
_PROJECT_MEMBER = "project.json"
_SOURCE_MEMBER_PREFIX = "sources/"
_CANONICAL_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_CANONICAL_ZIP_CREATE_SYSTEM = 3
_CANONICAL_ZIP_FILE_MODE = stat.S_IFREG | 0o644
_REPLAY_METADATA_FIELDS = frozenset(
    {
        "physical_columns",
        "time_column",
        "observation_columns",
        "original_time_unit",
        "original_concentration_units",
        "skip_unit_row",
    }
)


@dataclass(frozen=True, slots=True)
class ArchiveProject:
    document: ProjectDocument
    presentation: WorkspacePresentationState


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object field {key!r}.")
        result[key] = value
    return result


def _decode_json_bytes(data: bytes, *, label: str) -> object:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be strict UTF-8.") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{label} is not valid strict JSON.") from exc


def _canonical_zip_info(member_name: str) -> zipfile.ZipInfo:
    """Return host-independent metadata for one canonical archive member."""

    info = zipfile.ZipInfo(
        filename=member_name,
        date_time=_CANONICAL_ZIP_TIMESTAMP,
    )
    info.create_system = _CANONICAL_ZIP_CREATE_SYSTEM
    info.external_attr = _CANONICAL_ZIP_FILE_MODE << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def _finite(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: float | None = None,
) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{label} must be positive.")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{label} must be nonnegative.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} must be at most {maximum}.")
    return parsed


def _integer(value: object, label: str) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{label} must be an int.")
    return int(value)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool.")
    return bool(value)


def _string(value: object, label: str, *, nonblank: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str.")
    if nonblank and not value.strip():
        raise ValueError(f"{label} must not be blank.")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings.")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array.")
    return value


def _exact_fields(
    value: object,
    label: str,
    fields: set[str] | frozenset[str],
) -> Mapping[str, object]:
    data = _object(value, label)
    missing = sorted(fields - set(data))
    unknown = sorted(set(data) - fields)
    if missing:
        raise ValueError(
            f"{label} is missing field(s): {', '.join(missing)}."
        )
    if unknown:
        raise ValueError(
            f"{label} has unknown field(s): {', '.join(unknown)}."
        )
    return data


def _encode_execution(
    settings: SimulationExecutionSettings,
) -> dict[str, object]:
    return {
        "solver": settings.solver,
        "t_start": settings.t_start,
        "t_end": settings.t_end,
        "integration_points": settings.integration_points,
    }


def _decode_execution(
    value: object,
    label: str,
) -> SimulationExecutionSettings:
    data = _object(value, label)
    _exact_fields(
        data,
        label,
        {"solver", "t_start", "t_end", "integration_points"},
    )
    return SimulationExecutionSettings(
        solver=_string(
            data["solver"],
            f"{label}.solver",
            nonblank=True,
        ),
        t_start=_finite(data["t_start"], f"{label}.t_start"),
        t_end=_finite(data["t_end"], f"{label}.t_end"),
        integration_points=_integer(
            data["integration_points"],
            f"{label}.integration_points",
        ),
    )


def _encode_science(
    settings: TabScientificSettings,
) -> dict[str, object]:
    return {
        "temperature": settings.temperature,
        "standard_conc_M": settings.standard_conc_M,
        "energy_unit": settings.energy_unit,
        "rtol": settings.rtol,
        "atol": settings.atol,
        "analytic_jacobian_policy": settings.analytic_jacobian_policy,
        "use_authored_stops": settings.use_authored_stops,
    }


def _decode_science(
    value: object,
    label: str,
) -> TabScientificSettings:
    data = _exact_fields(
        value,
        label,
        {
            "temperature",
            "standard_conc_M",
            "energy_unit",
            "rtol",
            "atol",
            "analytic_jacobian_policy",
            "use_authored_stops",
        },
    )
    return TabScientificSettings(
        temperature=_finite(
            data["temperature"],
            f"{label}.temperature",
        ),
        rtol=_finite(data["rtol"], f"{label}.rtol"),
        standard_conc_M=_finite(data["standard_conc_M"], f"{label}.standard_conc_M"),
        energy_unit=_string(data["energy_unit"], f"{label}.energy_unit"),
        atol=_finite(data["atol"], f"{label}.atol"),
        analytic_jacobian_policy=_boolean(
            data["analytic_jacobian_policy"],
            f"{label}.analytic_jacobian_policy",
        ),
        use_authored_stops=_boolean(
            data["use_authored_stops"], f"{label}.use_authored_stops",
        ),
    )


def _encode_fit(defaults: FitDefaults) -> dict[str, object]:
    return {
        "method": defaults.method,
        "max_nfev": defaults.max_nfev,
        "ftol": defaults.ftol,
        "xtol": defaults.xtol,
        "use_seed": defaults.use_seed,
        "seed": defaults.seed,
    }


def _decode_fit(value: object, label: str) -> FitDefaults:
    data = _exact_fields(
        value,
        label,
        {"method", "max_nfev", "ftol", "xtol", "use_seed", "seed"},
    )
    return FitDefaults(
        method=_string(
            data["method"],
            f"{label}.method",
            nonblank=True,
        ),
        max_nfev=_integer(
            data["max_nfev"],
            f"{label}.max_nfev",
        ),
        ftol=_finite(data["ftol"], f"{label}.ftol"),
        xtol=_finite(data["xtol"], f"{label}.xtol"),
        use_seed=_boolean(
            data["use_seed"],
            f"{label}.use_seed",
        ),
        seed=_integer(data["seed"], f"{label}.seed"),
    )


def _encode_document(
    document: ProjectDocument,
) -> tuple[dict[str, object], tuple[tuple[str, bytes], ...]]:
    source_members: list[tuple[str, bytes]] = []
    source_payloads: list[dict[str, object]] = []
    for index, source in enumerate(document.imported_sources):
        source_member = (
            f"{_SOURCE_MEMBER_PREFIX}{index:08d}/{source.original_filename}"
        )
        source_members.append((source_member, source.exact_source_bytes))
        source_payloads.append(
            {
                "source_id": source.source_id,
                "original_filename": source.original_filename,
                "original_filepath": source.original_filepath,
                "original_format": source.original_format,
                "source_member": source_member,
            }
        )
    datasets: list[dict[str, object]] = []
    for dataset in document.datasets:
        datasets.append(
            {
                "dataset_id": dataset.dataset_id,
                "display_name": dataset.display_name,
                "time_column_name": dataset.time_column_name,
                "source_id": dataset.source_id,
                "source_metadata": thaw_json_value(
                    dataset.source_metadata
                ),
                "observations": [
                    {
                        "raw_observation_id": series.raw_observation_id,
                        "column_name": series.column_name,
                        "t": list(series.t),
                        "y": list(series.y),
                        **({"source_rows": list(series.source_rows)} if dataset.source_id is not None else {
                            "scientific_unit": (
                                None if series.unit is None
                                else scientific_unit(series.unit.dimensions).symbol
                            ),
                            "row_ids": list(series.source_rows),
                        }),
                    }
                    for series in dataset.observations
                ],
            }
        )
    payload = {
        "project_id": document.project_id,
        "sources": source_payloads,
        "datasets": datasets,
        "tabs": [
            {
                "tab_id": tab.tab_id,
                "name": tab.name,
                "mechanism_source": tab.mechanism_source.to_payload(),
                "notes": tab.notes,
                "initial_condition_sets": [
                    {
                        "set_id": initial_set.set_id,
                        "name": initial_set.name,
                        "values": [
                            [
                                species,
                                0.0 if value == 0.0 else value,
                            ]
                            for species, value in initial_set.values
                        ],
                    }
                    for initial_set in tab.initial_condition_sets
                ],
                "run_defaults": _encode_execution(tab.run_defaults),
                "scientific_settings": _encode_science(
                    tab.scientific_settings
                ),
                "run_overrides_by_set_id": [
                    {
                        "set_id": set_id,
                        "settings": _encode_execution(settings),
                    }
                    for set_id, settings
                    in tab.run_overrides_by_set_id
                ],
                "dataset_links": [
                    {
                        "dataset_id": dataset_id,
                        "set_id": link.set_id,
                        "observation_mapping": [
                            [raw_observation_id, model_target]
                            for raw_observation_id, model_target
                            in link.observation_mapping
                        ],
                    }
                    for dataset_id, link in tab.dataset_links
                ],
            }
            for tab in document.tabs
        ],
        "fit_defaults": _encode_fit(document.fit_defaults),
    }
    return payload, tuple(source_members)


def _decode_pair_array(
    value: object,
    label: str,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_array(value, label)):
        pair = _array(item, f"{label}[{index}]")
        if len(pair) != 2:
            raise ValueError(
                f"{label}[{index}] must contain exactly two values."
            )
        pairs.append(
            (
                _string(
                    pair[0],
                    f"{label}[{index}][0]",
                    nonblank=True,
                ),
                _string(
                    pair[1],
                    f"{label}[{index}][1]",
                    nonblank=True,
                ),
            )
        )
    return tuple(pairs)


def _replay_imported_dataset(
    dataset: RawDatasetRecord,
    source: ImportedSource,
    *,
    label: str,
    time_unit: Unit,
    observation_units: Mapping[str, Unit],
    workbooks: ExcelWorkbookPair | None = None,
) -> None:
    """Reparse retained source bytes and require exact observation replay."""
    metadata = thaw_json_value(dataset.source_metadata)
    raw_columns = metadata["physical_columns"]
    if not isinstance(raw_columns, list):
        raise ValueError(f"{label}.source_metadata.physical_columns must be an array.")
    columns = tuple(
        PhysicalColumn(
            source_index=_integer(item.get("source_index"), f"{label}.physical_columns[{index}].source_index"),
            authored_name=_string(item.get("authored_name"), f"{label}.physical_columns[{index}].authored_name", nonblank=True),
            assigned_name=_string(item.get("assigned_name"), f"{label}.physical_columns[{index}].assigned_name", nonblank=True),
        )
        for index, item in enumerate(raw_columns)
        if isinstance(item, Mapping)
    )
    if len(columns) != len(raw_columns):
        raise ValueError(f"{label}.source_metadata.physical_columns contains invalid entries.")
    observation_names = metadata["observation_columns"]
    if (
        not isinstance(observation_names, list)
        or any(not isinstance(name, str) or not name.strip() for name in observation_names)
    ):
        raise ValueError(f"{label}.source_metadata.observation_columns must be an array of names.")
    by_name = {column.assigned_name: column for column in columns}
    try:
        time_column = (
            None if metadata["time_column"] is None
            else by_name[str(metadata["time_column"])]
        )
        species_columns = tuple(by_name[str(name)] for name in observation_names)
    except KeyError as exc:
        raise ValueError(f"{label}.source_metadata references an unknown physical column.") from exc
    skip_unit_row = _boolean(
        metadata["skip_unit_row"], f"{label}.source_metadata.skip_unit_row",
    )

    def _compare(payload) -> None:  # noqa: ANN001
        expected = tuple(
            (series.column_name, series.unit, series.t, series.y, series.source_rows)
            for series in dataset.observations
        )
        actual = tuple(
            (series.column_name, series.unit, series.t, series.y, series.source_rows)
            for series in payload.observations
        )
        if actual != expected:
            raise ValueError(
                f"{label} observations do not replay exactly from retained source bytes."
            )

    if source.original_format == "csv":
        columns_from_source, rows = read_csv_rows_from_bytes(
            source.exact_source_bytes,
            source_label=source.original_filename,
        )
        try:
            if columns_from_source != columns:
                raise ValueError(f"{label} physical columns do not replay exactly.")
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"{label} source contains no data rows.")
            rows_to_parse = (
                row_iter
                if skip_unit_row
                else itertools.chain((first_row,), row_iter)
            )
            payload = parse_physical_rows(
                rows_to_parse,
                columns=columns,
                time_column=time_column,
                species_columns=species_columns,
                source_label=source.original_filename,
                source_row_start=3 if skip_unit_row else 2,
                time_unit=time_unit,
                observation_units=observation_units,
                metadata_extras={
                    "skip_unit_row": skip_unit_row,
                },
            )
            _compare(payload)
        finally:
            rows.close()
        return

    if source.original_format != "xlsx":
        raise ValueError(f"{label} has unsupported source format {source.original_format!r}.")
    sheet_name = metadata.get("sheet_name")
    if not isinstance(sheet_name, str) or not sheet_name:
        raise ValueError(f"{label}.source_metadata.sheet_name is required for XLSX replay.")
    workbook_context = (
        nullcontext(workbooks)
        if workbooks is not None
        else ExcelWorkbookPair(
            source.exact_source_bytes,
            source_label=source.original_filename,
        )
    )
    with workbook_context as active_workbooks:
        rows = active_workbooks.read_sheet_rows(sheet_name)
        try:
            row_iter = iter(rows)
            first_row = next(row_iter, None)
            if first_row is None:
                raise ValueError(f"{label} worksheet contains no data rows.")
            columns_from_source = tuple(first_row)
            if columns_from_source != columns:
                raise ValueError(f"{label} physical columns do not replay exactly.")
            rows_to_parse = (
                row_iter
                if skip_unit_row
                else itertools.chain((first_row,), row_iter)
            )
            payload = parse_physical_rows(
                rows_to_parse,
                columns=columns,
                time_column=time_column,
                species_columns=species_columns,
                source_label=source.original_filename,
                sheet_name=sheet_name,
                source_row_start=3 if skip_unit_row else 2,
                time_unit=time_unit,
                observation_units=observation_units,
                metadata_extras={
                    "skip_unit_row": skip_unit_row,
                },
            )
            _compare(payload)
        finally:
            rows.close()


def _require_replay_metadata(
    metadata: object,
    source: ImportedSource,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label}.source_metadata must be a JSON object.")
    missing = sorted(_REPLAY_METADATA_FIELDS - set(metadata))
    if missing:
        raise ValueError(
            f"{label}.source_metadata is missing replay field(s): "
            f"{', '.join(missing)}."
        )
    if source.original_format == "xlsx":
        sheet_name = metadata.get("sheet_name")
        if not isinstance(sheet_name, str) or not sheet_name:
            raise ValueError(
                f"{label}.source_metadata.sheet_name is required for XLSX replay."
            )
    return metadata


def _decode_document(
    value: object,
    source_bytes_by_member: Mapping[str, bytes],
) -> ProjectDocument:
    data = _exact_fields(
        value,
        "document",
        {"project_id", "sources", "datasets", "tabs", "fit_defaults"},
    )
    sources: list[ImportedSource] = []
    source_ids: set[str] = set()
    sources_by_id: dict[str, ImportedSource] = {}
    source_members: set[str] = set()
    for index, raw_source in enumerate(_array(data["sources"], "document.sources")):
        label = f"document.sources[{index}]"
        item = _exact_fields(
            raw_source,
            label,
            {"source_id", "original_filename", "original_filepath", "original_format", "source_member"},
        )
        original_filename = _string(
            item["original_filename"], f"{label}.original_filename", nonblank=True
        )
        try:
            validate_original_basename(original_filename, label=f"{label}.original_filename")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        original_format = _string(
            item["original_format"], f"{label}.original_format", nonblank=True
        )
        expected_member = (
            f"{_SOURCE_MEMBER_PREFIX}{index:08d}/{original_filename}"
        )
        source_member = _string(item["source_member"], f"{label}.source_member", nonblank=True)
        if source_member != expected_member:
            raise ValueError(f"{label}.source_member must be {expected_member!r}.")
        if source_member not in source_bytes_by_member:
            raise ValueError(f"Archive is missing source member {source_member!r}.")
        source_id = _string(item["source_id"], f"{label}.source_id", nonblank=True)
        if source_id in source_ids:
            raise ValueError(f"Duplicate imported source ID {source_id!r}.")
        source_ids.add(source_id)
        source_members.add(source_member)
        sources.append(
            ImportedSource(
                source_id=source_id,
                original_filename=original_filename,
                original_filepath=_string(item["original_filepath"], f"{label}.original_filepath", nonblank=True),
                original_format=original_format,
                exact_source_bytes=source_bytes_by_member[source_member],
            )
        )
        validate_imported_source(sources[-1])
        sources_by_id[source_id] = sources[-1]
    datasets: list[RawDatasetRecord] = []
    datasets_by_source: dict[str, list[tuple[RawDatasetRecord, str, Unit, dict[str, Unit]]]] = {}
    referenced_source_ids: set[str] = set()
    for index, raw_dataset in enumerate(
        _array(data["datasets"], "document.datasets")
    ):
        label = f"document.datasets[{index}]"
        item = _exact_fields(
            raw_dataset,
            label,
            {
                "dataset_id",
                "display_name",
                "time_column_name",
                "source_id",
                "source_metadata",
                "observations",
            },
        )
        created = item["source_id"] is None
        if created:
            if item["source_metadata"] is not None:
                raise ValueError(f"{label}: created datasets require null capture fields.")
            source_id, metadata = None, None
        else:
            source_id = _string(item["source_id"], f"{label}.source_id", nonblank=True)
            if source_id not in source_ids:
                raise ValueError(f"{label}.source_id references an unknown imported source.")
            referenced_source_ids.add(source_id)
            metadata = _require_replay_metadata(item["source_metadata"], sources_by_id[source_id], label=label)
            time_unit = parse_time_unit(_string(
                metadata["original_time_unit"], f"{label}.source_metadata.original_time_unit", nonblank=True,
            ))
            declared_units = metadata["original_concentration_units"]
            if not isinstance(declared_units, Mapping):
                raise ValueError(f"{label}.source_metadata.original_concentration_units must be an object.")
        observation_units: dict[str, Unit] = {}
        observations: list[ObservationSeries] = []
        for series_index, raw_series in enumerate(
            _array(item["observations"], f"{label}.observations")
        ):
            series_label = f"{label}.observations[{series_index}]"
            series = _exact_fields(
                raw_series,
                series_label,
                {
                    "raw_observation_id",
                    "column_name",
                    "t",
                    "y",
                } | ({"scientific_unit", "row_ids"} if created else {"source_rows"}),
            )
            column_name = _string(series["column_name"], f"{series_label}.column_name", nonblank=True)
            if created:
                unit = None if series["scientific_unit"] is None else parse_measurement_unit(_string(
                    series["scientific_unit"], f"{series_label}.scientific_unit", nonblank=True,
                ))
                if unit is not None and unit != scientific_unit(unit.dimensions):
                    raise ValueError(f"{series_label}: stored units must use scientific coordinates.")
            else:
                if column_name not in declared_units:
                    raise ValueError(f"{series_label} requires an input unit declaration.")
                observation_units[column_name] = parse_measurement_unit(_string(
                    declared_units[column_name],
                    f"{label}.source_metadata.original_concentration_units.{column_name}",
                    nonblank=True,
                ))
                unit = scientific_unit(observation_units[column_name].dimensions)
            row_key = "row_ids" if created else "source_rows"
            observations.append(
                ObservationSeries(
                    raw_observation_id=_string(
                        series["raw_observation_id"],
                        f"{series_label}.raw_observation_id",
                        nonblank=True,
                    ),
                    column_name=column_name,
                    unit=unit,
                    t=tuple(
                        _finite(
                            item_value,
                            f"{series_label}.t[{value_index}]",
                        )
                        for value_index, item_value in enumerate(
                            _array(series["t"], f"{series_label}.t")
                        )
                    ),
                    y=tuple(
                        _finite(
                            item_value,
                            f"{series_label}.y[{value_index}]",
                        )
                        for value_index, item_value in enumerate(
                            _array(series["y"], f"{series_label}.y")
                        )
                    ),
                    source_rows=tuple(
                        _integer(
                            item_value,
                            f"{series_label}.{row_key}[{value_index}]",
                        )
                        for value_index, item_value in enumerate(
                            _array(
                                series[row_key],
                                f"{series_label}.{row_key}",
                            )
                        )
                    ),
                )
            )
        dataset = RawDatasetRecord(
            dataset_id=_string(
                item["dataset_id"],
                f"{label}.dataset_id",
                nonblank=True,
            ),
            display_name=_string(
                item["display_name"],
                f"{label}.display_name",
                nonblank=True,
            ),
            time_column_name=(
                None if item["time_column_name"] is None
                else _string(item["time_column_name"], f"{label}.time_column_name", nonblank=True)
            ),
            source_id=source_id,
            source_metadata=freeze_json_value(metadata),
            observations=tuple(observations),
        )
        validate_raw_dataset_record(dataset)
        datasets.append(dataset)
        if not created:
            datasets_by_source.setdefault(source_id, []).append((dataset, label, time_unit, observation_units))
    for source_id, source_datasets in datasets_by_source.items():
        source = sources_by_id[source_id]
        if source.original_format == "xlsx":
            with ExcelWorkbookPair(
                source.exact_source_bytes,
                source_label=source.original_filename,
            ) as workbooks:
                for dataset, label, time_unit, observation_units in source_datasets:
                    _replay_imported_dataset(
                        dataset,
                        source,
                        label=label,
                        time_unit=time_unit,
                        observation_units=observation_units,
                        workbooks=workbooks,
                    )
        else:
            for dataset, label, time_unit, observation_units in source_datasets:
                _replay_imported_dataset(
                    dataset, source, label=label, time_unit=time_unit, observation_units=observation_units,
                )
    if referenced_source_ids != source_ids:
        raise ValueError("Archive contains an unreferenced imported source.")
    if source_members != set(source_bytes_by_member):
        raise ValueError("Archive contains an unreferenced or unknown source member.")

    tabs: list[MechanismTabDocument] = []
    for index, raw_tab in enumerate(
        _array(data["tabs"], "document.tabs")
    ):
        label = f"document.tabs[{index}]"
        item = _exact_fields(
            raw_tab,
            label,
            {
                "tab_id",
                "name",
                "mechanism_source",
                "notes",
                "initial_condition_sets",
                "run_defaults",
                "scientific_settings",
                "run_overrides_by_set_id",
                "dataset_links",
            },
        )
        tab_id = _string(
            item["tab_id"],
            f"{label}.tab_id",
            nonblank=True,
        )
        tab_name = _string(
            item["name"],
            f"{label}.name",
            nonblank=True,
        )
        initial_sets: list[InitialConditionSet] = []
        for set_index, raw_set in enumerate(
            _array(
                item["initial_condition_sets"],
                f"{label}.initial_condition_sets",
            )
        ):
            set_label = f"{label}.initial_condition_sets[{set_index}]"
            set_data = _exact_fields(
                raw_set,
                set_label,
                {"set_id", "name", "values"},
            )
            set_id = _string(
                set_data["set_id"],
                f"{set_label}.set_id",
                nonblank=True,
            )
            set_name = _string(
                set_data["name"],
                f"{set_label}.name",
                nonblank=True,
            )
            value_pairs: list[tuple[str, float]] = []
            for pair_index, raw_pair in enumerate(
                _array(set_data["values"], f"{set_label}.values")
            ):
                pair = _array(
                    raw_pair,
                    f"{set_label}.values[{pair_index}]",
                )
                if len(pair) != 2:
                    raise ValueError(
                        f"{set_label}.values[{pair_index}] must contain "
                        "exactly two values."
                    )
                species = _string(
                    pair[0],
                    f"{set_label}.values[{pair_index}][0]",
                    nonblank=True,
                )
                concentration = _finite(
                    pair[1],
                    "initial concentration for "
                    f"tab {tab_id!r} ({tab_name!r}), "
                    f"set {set_id!r} ({set_name!r}), "
                    f"species {species!r}",
                    nonnegative=True,
                )
                value_pairs.append(
                    (
                        species,
                        0.0 if concentration == 0.0 else concentration,
                    )
                )
            initial_sets.append(
                InitialConditionSet(
                    set_id=set_id,
                    name=set_name,
                    values=tuple(value_pairs),
                )
            )

        overrides: list[
            tuple[str, SimulationExecutionSettings]
        ] = []
        for override_index, raw_override in enumerate(
            _array(
                item["run_overrides_by_set_id"],
                f"{label}.run_overrides_by_set_id",
            )
        ):
            override_label = (
                f"{label}.run_overrides_by_set_id[{override_index}]"
            )
            override = _exact_fields(
                raw_override,
                override_label,
                {"set_id", "settings"},
            )
            overrides.append(
                (
                    _string(
                        override["set_id"],
                        f"{override_label}.set_id",
                        nonblank=True,
                    ),
                    _decode_execution(
                        override["settings"],
                        f"{override_label}.settings",
                    ),
                )
            )

        links: list[tuple[str, DatasetLink]] = []
        for link_index, raw_link in enumerate(
            _array(item["dataset_links"], f"{label}.dataset_links")
        ):
            link_label = f"{label}.dataset_links[{link_index}]"
            link = _exact_fields(
                raw_link,
                link_label,
                {
                    "dataset_id",
                    "set_id",
                    "observation_mapping",
                },
            )
            raw_set_id = link["set_id"]
            if raw_set_id is not None and not isinstance(raw_set_id, str):
                raise TypeError(
                    f"{link_label}.set_id must be a str or null."
                )
            links.append(
                (
                    _string(
                        link["dataset_id"],
                        f"{link_label}.dataset_id",
                        nonblank=True,
                    ),
                    DatasetLink(
                        set_id=(
                            None
                            if raw_set_id is None
                            else _string(
                                raw_set_id,
                                f"{link_label}.set_id",
                                nonblank=True,
                            )
                        ),
                        observation_mapping=_decode_pair_array(
                            link["observation_mapping"],
                            f"{link_label}.observation_mapping",
                        ),
                    ),
                )
            )
        tabs.append(
            MechanismTabDocument(
                tab_id=tab_id,
                name=tab_name,
                mechanism_source=MechanismAuthoringSource.from_payload(
                    _object(
                        item["mechanism_source"],
                        f"{label}.mechanism_source",
                    )
                ),
                notes=_string(item["notes"], f"{label}.notes"),
                initial_condition_sets=tuple(initial_sets),
                run_defaults=_decode_execution(
                    item["run_defaults"],
                    f"{label}.run_defaults",
                ),
                scientific_settings=_decode_science(
                    item["scientific_settings"],
                    f"{label}.scientific_settings",
                ),
                run_overrides_by_set_id=tuple(overrides),
                dataset_links=tuple(links),
            )
        )
    return ProjectDocument(
        project_id=_string(
            data["project_id"],
            "document.project_id",
            nonblank=True,
        ),
        imported_sources=tuple(sources),
        datasets=tuple(datasets),
        tabs=tuple(tabs),
        fit_defaults=_decode_fit(
            data["fit_defaults"],
            "document.fit_defaults",
        ),
    )


def _encode_presentation(
    state: ProjectPresentationState,
) -> dict[str, object]:
    return {
        "current_set_id": state.current_set_id,
        "plot_set_ids": list(state.plot_set_ids),
        "pinned_slider_set_ids": list(state.pinned_slider_set_ids),
        "overlay_raw_observation_ids_by_dataset": [
            {
                "dataset_id": dataset_id,
                "raw_observation_ids": list(raw_observation_ids),
            }
            for dataset_id, raw_observation_ids
            in state.overlay_raw_observation_ids_by_dataset
        ],
        "selected_x_series": _encode_axis_selection(state.selected_x_series),
        "selected_y_series": [
            _encode_axis_selection(selection)
            for selection in state.selected_y_series
        ],
        "selected_y_uses_default_all": state.selected_y_uses_default_all,
        "reference_lines_visible": state.reference_lines_visible,
        "intervention_annotations_visible": (
            state.intervention_annotations_visible
        ),
        "intervention_labels_visible": state.intervention_labels_visible,
        "log_x": state.log_x,
        "log_y": state.log_y,
        "legend_visible": state.legend_visible,
        "x_inverted": state.x_inverted,
        "y_inverted": state.y_inverted,
        "grid_x": state.grid_x,
        "grid_y": state.grid_y,
        "grid_opacity": state.grid_opacity,
        "plot_transform": state.plot_transform,
        "simulation_markers": state.simulation_markers,
        "simulation_opacity": state.simulation_opacity,
    }


def _encode_axis_selection(
    selection: AxisSeriesSelection,
) -> dict[str, object]:
    if isinstance(selection, VesselVolumeAxisSelection):
        return {"kind": "vessel_volume"}
    if isinstance(selection, SimulationSeriesAxisSelection):
        return {
            "kind": "simulation",
            "series_name": selection.series_name,
        }
    if isinstance(selection, ExperimentalAxisSelection):
        return {
            "kind": "experimental",
            "dataset_id": selection.dataset_id,
            "raw_observation_id": selection.raw_observation_id,
        }
    raise TypeError("Axis selection has an unsupported type.")


def _decode_axis_selection(
    value: object,
    label: str,
) -> AxisSeriesSelection:
    data = _object(value, label)
    kind = _string(data.get("kind"), f"{label}.kind", nonblank=True)
    if kind == "vessel_volume":
        _exact_fields(data, label, {"kind"})
        return VesselVolumeAxisSelection()
    if kind == "simulation":
        fields = _exact_fields(
            data,
            label,
            {"kind", "series_name"},
        )
        return SimulationSeriesAxisSelection(
            series_name=_string(
                fields["series_name"],
                f"{label}.series_name",
                nonblank=True,
            )
        )
    if kind == "experimental":
        fields = _exact_fields(
            data,
            label,
            {"kind", "dataset_id", "raw_observation_id"},
        )
        return ExperimentalAxisSelection(
            dataset_id=_string(
                fields["dataset_id"],
                f"{label}.dataset_id",
                nonblank=True,
            ),
            raw_observation_id=_string(
                fields["raw_observation_id"],
                f"{label}.raw_observation_id",
                nonblank=True,
            ),
        )
    raise ValueError(
        f"{label}.kind must be 'simulation', 'experimental', or 'vessel_volume'."
    )


def _optional_string(
    value: object,
    label: str,
) -> str | None:
    if value is None:
        return None
    return _string(value, label, nonblank=True)


def _decode_presentation(
    value: object,
    label: str,
) -> ProjectPresentationState:
    data = _exact_fields(
        value,
        label,
        {
            "current_set_id",
            "plot_set_ids",
            "pinned_slider_set_ids",
            "overlay_raw_observation_ids_by_dataset",
            "selected_x_series",
            "selected_y_series",
            "selected_y_uses_default_all",
            "reference_lines_visible",
            "intervention_annotations_visible",
            "intervention_labels_visible",
            "log_x",
            "log_y",
            "legend_visible",
            "x_inverted",
            "y_inverted",
            "grid_x",
            "grid_y",
            "grid_opacity",
            "plot_transform",
            "simulation_markers",
            "simulation_opacity",
        },
    )
    overlays: list[tuple[str, tuple[str, ...]]] = []
    for index, raw_overlay in enumerate(
        _array(
            data["overlay_raw_observation_ids_by_dataset"],
            f"{label}.overlay_raw_observation_ids_by_dataset",
        )
    ):
        overlay_label = (
            f"{label}.overlay_raw_observation_ids_by_dataset[{index}]"
        )
        overlay = _exact_fields(
            raw_overlay,
            overlay_label,
            {"dataset_id", "raw_observation_ids"},
        )
        overlays.append(
            (
                _string(
                    overlay["dataset_id"],
                    f"{overlay_label}.dataset_id",
                    nonblank=True,
                ),
                tuple(
                    _string(
                        raw_observation_id,
                        f"{overlay_label}.raw_observation_ids["
                        f"{raw_observation_index}]",
                        nonblank=True,
                    )
                    for raw_observation_index, raw_observation_id in enumerate(
                        _array(
                            overlay["raw_observation_ids"],
                            f"{overlay_label}.raw_observation_ids",
                        )
                    )
                ),
            )
        )
    return ProjectPresentationState(
        current_set_id=_optional_string(
            data["current_set_id"],
            f"{label}.current_set_id",
        ),
        plot_set_ids=tuple(
            _string(
                item,
                f"{label}.plot_set_ids[{index}]",
                nonblank=True,
            )
            for index, item in enumerate(
                _array(
                    data["plot_set_ids"],
                    f"{label}.plot_set_ids",
                )
            )
        ),
        pinned_slider_set_ids=tuple(
            _string(
                item,
                f"{label}.pinned_slider_set_ids[{index}]",
                nonblank=True,
            )
            for index, item in enumerate(
                _array(
                    data["pinned_slider_set_ids"],
                    f"{label}.pinned_slider_set_ids",
                )
            )
        ),
        overlay_raw_observation_ids_by_dataset=tuple(overlays),
        selected_x_series=_decode_axis_selection(
            data["selected_x_series"],
            f"{label}.selected_x_series",
        ),
        selected_y_series=tuple(
            _decode_axis_selection(
                item,
                f"{label}.selected_y_series[{index}]",
            )
            for index, item in enumerate(
                _array(
                    data["selected_y_series"],
                    f"{label}.selected_y_series",
                )
            )
        ),
        selected_y_uses_default_all=_boolean(
            data["selected_y_uses_default_all"],
            f"{label}.selected_y_uses_default_all",
        ),
        reference_lines_visible=_boolean(
            data["reference_lines_visible"],
            f"{label}.reference_lines_visible",
        ),
        intervention_annotations_visible=_boolean(
            data["intervention_annotations_visible"],
            f"{label}.intervention_annotations_visible",
        ),
        intervention_labels_visible=_boolean(
            data["intervention_labels_visible"],
            f"{label}.intervention_labels_visible",
        ),
        log_x=_boolean(data["log_x"], f"{label}.log_x"),
        log_y=_boolean(data["log_y"], f"{label}.log_y"),
        legend_visible=_boolean(
            data["legend_visible"],
            f"{label}.legend_visible",
        ),
        x_inverted=_boolean(
            data["x_inverted"],
            f"{label}.x_inverted",
        ),
        y_inverted=_boolean(
            data["y_inverted"],
            f"{label}.y_inverted",
        ),
        grid_x=_boolean(data["grid_x"], f"{label}.grid_x"),
        grid_y=_boolean(data["grid_y"], f"{label}.grid_y"),
        grid_opacity=_finite(data["grid_opacity"], f"{label}.grid_opacity"),
        plot_transform=_string(data["plot_transform"], f"{label}.plot_transform"),
        simulation_markers=_boolean(data["simulation_markers"], f"{label}.simulation_markers"),
        simulation_opacity=_finite(data["simulation_opacity"], f"{label}.simulation_opacity"),
    )


def _encode_project(
    document: ProjectDocument,
    presentation: WorkspacePresentationState,
) -> tuple[dict[str, object], tuple[tuple[str, bytes], ...]]:
    if not isinstance(document, ProjectDocument):
        raise TypeError("document must be ProjectDocument.")
    workspace_presentation = validate_workspace_presentation_state(
        document,
        presentation,
    )
    encoded_document, source_members = _encode_document(document)
    return (
        {
            "format": CURRENT_PROJECT_FORMAT,
            "document": encoded_document,
            "workspace": {
                "active_tab_id": workspace_presentation.active_tab_id,
                "presentation_by_tab_id": [
                    {
                        "tab_id": tab_id,
                        "state": _encode_presentation(state),
                    }
                    for tab_id, state
                    in workspace_presentation.presentation_by_tab_id
                ],
            },
        },
        source_members,
    )


def _decode_project(
    value: object,
    source_bytes_by_member: Mapping[str, bytes],
) -> ArchiveProject:
    root = _exact_fields(
        value,
        "project",
        {"format", "document", "workspace"},
    )
    project_format = _string(
        root["format"],
        "project.format",
        nonblank=True,
    )
    if project_format != CURRENT_PROJECT_FORMAT:
        raise ValueError(
            f"Unsupported project format {project_format!r}; "
            f"expected {CURRENT_PROJECT_FORMAT!r}."
        )
    document = _decode_document(
        root["document"], source_bytes_by_member,
    )
    workspace = _exact_fields(
        root["workspace"],
        "workspace",
        {"active_tab_id", "presentation_by_tab_id"},
    )
    presentations: list[
        tuple[str, ProjectPresentationState]
    ] = []
    for index, raw_entry in enumerate(
        _array(
            workspace["presentation_by_tab_id"],
            "workspace.presentation_by_tab_id",
        )
    ):
        label = f"workspace.presentation_by_tab_id[{index}]"
        entry = _exact_fields(
            raw_entry,
            label,
            {"tab_id", "state"},
        )
        presentations.append(
            (
                _string(
                    entry["tab_id"],
                    f"{label}.tab_id",
                    nonblank=True,
                ),
                _decode_presentation(
                    entry["state"],
                    f"{label}.state",
                ),
            )
        )
    workspace_presentation = WorkspacePresentationState(
        active_tab_id=_string(
            workspace["active_tab_id"],
            "workspace.active_tab_id",
            nonblank=True,
        ),
        presentation_by_tab_id=tuple(presentations),
    )
    return ArchiveProject(
        document=document,
        presentation=workspace_presentation,
    )


class KinArchiveStore:
    """Decode and save current format 7 archives strictly."""

    def load(self, path: str | os.PathLike[str]) -> ArchiveProject:
        """Decode one archive; final document admission belongs to its caller."""

        archive_path = os.path.abspath(os.fspath(path))
        try:
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if len(names) != len(set(names)):
                    raise ValueError(
                        "Current project archive has duplicate members."
                    )
                if _PROJECT_MEMBER not in names:
                    raise ValueError(
                        f"Current project archive is missing "
                        f"{_PROJECT_MEMBER!r}."
                    )
                if any(member.is_dir() for member in members):
                    raise ValueError(
                        "Current project archive must contain files only."
                    )
                if any(
                    member.compress_type != zipfile.ZIP_STORED
                    for member in members
                ):
                    raise ValueError(
                        "Current project archive members must be stored "
                        "without compression."
                    )
                project_bytes = archive.read(_PROJECT_MEMBER)
                source_bytes = {
                    name: archive.read(name)
                    for name in names
                    if name != _PROJECT_MEMBER
                }
        except zipfile.BadZipFile as exc:
            raise ValueError(
                "This is not a current Kindred project archive."
            ) from exc
        return _decode_project(
            _decode_json_bytes(project_bytes, label=_PROJECT_MEMBER),
            source_bytes,
        )

    def save(
        self,
        path: str | os.PathLike[str],
        document: ProjectDocument,
        presentation: WorkspacePresentationState,
    ) -> None:
        archive_path = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(archive_path) or os.curdir
        if not os.path.isdir(parent):
            raise FileNotFoundError(
                f"Project directory does not exist: {parent}"
            )
        payload, source_members = _encode_project(
            document,
            presentation,
        )
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        descriptor: int | None = None
        temporary_path = ""
        for _attempt in range(100):
            temporary_path = os.path.join(
                parent,
                f".{os.path.basename(archive_path)}.{uuid.uuid4().hex}.tmp",
            )
            try:
                descriptor = os.open(
                    temporary_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
            except FileExistsError:
                continue
            break
        if descriptor is None:
            raise FileExistsError(
                f"Could not allocate a sibling temporary archive in {parent}."
            )
        try:
            with os.fdopen(descriptor, mode="w+b") as handle:
                with zipfile.ZipFile(
                    handle,
                    mode="w",
                    compression=zipfile.ZIP_STORED,
                ) as archive:
                    members = (
                        (_PROJECT_MEMBER, encoded),
                        *source_members,
                    )
                    for member_name, member_bytes in sorted(members):
                        archive.writestr(
                            _canonical_zip_info(member_name),
                            member_bytes,
                        )
                handle.flush()
            # Inspect the destination entry itself.  In particular, a
            # dangling or live symlink is replaced as an entry by os.replace;
            # following it here would either mutate the target or make a
            # missing target appear to be a missing destination.
            try:
                existing_stat = os.lstat(archive_path)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISREG(existing_stat.st_mode):
                    os.chmod(
                        temporary_path,
                        stat.S_IMODE(existing_stat.st_mode),
                    )
            os.replace(temporary_path, archive_path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

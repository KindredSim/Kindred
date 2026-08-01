"""Stateless composition and export for the shared workspace result surface."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Protocol, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter

from kindred.core.result_repository import ResultRepository, SimulationResultValue
from kindred.gui.dataset_payload_snapshot import ExperimentalLayerInput

__all__ = [
    "ResultDataBlock",
    "ResultLayerKind",
    "VisibleResultSet",
    "WorkspacePlotLayer",
    "WorkspaceResultPresenter",
    "WorkspaceResultProjection",
    "XlsxExportCompletion",
]

_MAX_COPY_CELLS = 1_000_000
_MAX_COPY_CHARACTERS = 64 * 1024 * 1024
_MAX_XLSX_CELLS = 5_000_000
_MAX_XLSX_BYTES = 64 * 1024 * 1024


class _BoundedXlsxDestination:
    def __init__(
        self,
        destination: BinaryIO,
        *,
        cancellation_check: Callable[[], None] | None,
    ) -> None:
        if not callable(getattr(destination, "write", None)):
            raise TypeError("XLSX destination must be a writable binary stream.")
        if cancellation_check is not None and not callable(cancellation_check):
            raise TypeError("cancellation_check must be callable.")
        self._destination = destination
        self._cancellation_check = cancellation_check
        self._maximum_size = 0

    def write(self, data: object) -> int:
        if self._cancellation_check is not None:
            self._cancellation_check()
        payload = bytes(data)
        end = int(self.tell()) + len(payload)
        if max(self._maximum_size, end) > _MAX_XLSX_BYTES:
            raise ValueError(
                "Generated XLSX exceeds the bounded output-size limit "
                f"({_MAX_XLSX_BYTES // (1024 * 1024)} MiB)."
            )
        written = int(self._destination.write(payload))
        if written != len(payload):
            raise OSError(
                "XLSX destination accepted only "
                f"{written} of {len(payload)} bytes."
            )
        self._maximum_size = max(self._maximum_size, int(self.tell()))
        return written

    def __getattr__(self, name: str) -> object:
        return getattr(self._destination, name)

    @property
    def bytes_written(self) -> int:
        return self._maximum_size


@dataclass(frozen=True, slots=True)
class XlsxExportCompletion:
    worksheet_count: int
    data_row_count: int
    cell_count: int
    bytes_written: int


def _cleanup_write_only_workbook(workbook: Workbook) -> list[BaseException]:
    """Close writer streams and remove every owned worksheet tempfile."""

    failures: list[BaseException] = []
    for sheet in workbook.worksheets:
        rows = getattr(sheet, "_rows", None)
        if rows is not None:
            try:
                rows.close()
            except (StopIteration, RuntimeError, ValueError):
                pass
            except BaseException as exc:
                failures.append(exc)
        writer = getattr(sheet, "_writer", None)
        if writer is None:
            continue
        try:
            writer.close()
        except (StopIteration, RuntimeError, ValueError):
            pass
        except BaseException as exc:
            failures.append(exc)
        try:
            writer.cleanup()
        except (FileNotFoundError, ValueError):
            pass
        except BaseException as exc:
            failures.append(exc)
    try:
        workbook.close()
    except BaseException as exc:
        failures.append(exc)
    return failures


def _required_id(value: object, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be nonblank.")
    return normalized


def _owned_vector(value: object, *, field: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=float).reshape(-1)
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(float)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{field} must contain finite values.")
    array.setflags(write=False)
    return array


def _owned_identity_vector(
    value: object,
    *,
    field: str,
    size: int,
) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.int64).reshape(-1)
    array = (
        value
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(np.int64)
            and value.ndim == 1
            and not value.flags.writeable
        )
        else candidate.copy()
    )
    if array.size != int(size):
        raise ValueError(f"{field} must align with layer time.")
    if np.any(array < 0) or np.any(np.diff(array) < 0):
        raise ValueError(f"{field} must be ordered and nonnegative.")
    array.setflags(write=False)
    return array


def _segment_keys(
    values: Sequence[str],
    *,
    size: int,
) -> tuple[str, ...]:
    keys = tuple(_required_id(value, "segment key") for value in values)
    if len(keys) != int(size):
        raise ValueError("segment keys must align with layer time.")
    return keys


def _series_tuple(
    result: SimulationResultValue,
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple((name, values) for name, values in result.series.items())


@dataclass(frozen=True, slots=True)
class VisibleResultSet:
    set_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))
        object.__setattr__(self, "label", _required_id(self.label, "label"))


class ResultLayerKind(Enum):
    EXPLICIT = "explicit"
    PREVIEW = "preview"
    REFERENCE = "reference"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class WorkspacePlotLayer:
    layer_id: str
    source_id: str
    label: str
    kind: ResultLayerKind
    t: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...] = ()
    color_by_series: tuple[tuple[str, str], ...] = ()
    visible: bool = True
    dashed: bool = False
    opacity: float = 1.0
    markers: bool = False
    point_size: float = 6.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResultLayerKind):
            raise TypeError("kind must be a ResultLayerKind.")
        time = _owned_vector(self.t, field="plot time")
        series: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for raw_name, raw_values in self.series:
            name = _required_id(raw_name, "plot series name")
            if name in names:
                raise ValueError(f"plot layer contains duplicate series {name!r}.")
            values = _owned_vector(raw_values, field=f"plot series {name!r}")
            if values.size != time.size:
                raise ValueError(f"plot series {name!r} length does not match time.")
            names.add(name)
            series.append((name, values))
        if not series:
            raise ValueError("plot layer requires at least one series.")
        segment_ids = _owned_identity_vector(
            self.segment_ids,
            field="plot segment identity",
            size=time.size,
        )
        segment_keys = _segment_keys(self.segment_keys, size=time.size)
        discontinuity_ids = _owned_identity_vector(
            self.discontinuity_ids,
            field="plot discontinuity identity",
            size=time.size,
        )
        opacity = float(self.opacity)
        point_size = float(self.point_size)
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between zero and one.")
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_size must be positive.")
        colors = tuple(
            (_required_id(name, "color series name"), str(color))
            for name, color in self.color_by_series
            if str(name) in names
        )
        provenance: list[tuple[str, Mapping[str, object]]] = []
        provenance_names: set[str] = set()
        for raw_name, raw_payload in self.series_provenance:
            name = _required_id(raw_name, "series provenance name")
            if name not in names:
                raise ValueError(
                    "series provenance must reference a plot series."
                )
            if name in provenance_names:
                raise ValueError(
                    f"plot series provenance duplicates {name!r}."
                )
            if not isinstance(raw_payload, Mapping):
                raise TypeError("plot series provenance must be a mapping.")
            provenance_names.add(name)
            provenance.append(
                (
                    name,
                    MappingProxyType(
                        {
                            str(key): (
                                tuple(value)
                                if isinstance(value, (list, tuple))
                                else value
                            )
                            for key, value in raw_payload.items()
                        }
                    ),
                )
            )
        object.__setattr__(self, "layer_id", _required_id(self.layer_id, "layer_id"))
        object.__setattr__(self, "source_id", _required_id(self.source_id, "source_id"))
        object.__setattr__(self, "label", _required_id(self.label, "label"))
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", tuple(series))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "series_provenance", tuple(provenance))
        object.__setattr__(self, "color_by_series", colors)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "dashed", bool(self.dashed))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "markers", bool(self.markers))
        object.__setattr__(self, "point_size", point_size)


@dataclass(frozen=True, slots=True)
class ResultDataBlock:
    block_id: str
    label: str
    t: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    x_name: str = "t"
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...] = ()
    x_provenance: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        time = _owned_vector(self.t, field="data time")
        normalized: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for raw_name, raw_values in self.series:
            name = _required_id(raw_name, "data series name")
            if name in names:
                raise ValueError(
                    f"data block contains duplicate series {name!r}."
                )
            values = _owned_vector(raw_values, field=f"data series {name!r}")
            if values.size != time.size:
                raise ValueError(f"data series {name!r} length does not match time.")
            names.add(name)
            normalized.append((name, values))
        if not normalized:
            raise ValueError("data block requires at least one series.")
        segment_ids = _owned_identity_vector(
            self.segment_ids,
            field="data segment identity",
            size=time.size,
        )
        segment_keys = _segment_keys(self.segment_keys, size=time.size)
        discontinuity_ids = _owned_identity_vector(
            self.discontinuity_ids,
            field="data discontinuity identity",
            size=time.size,
        )
        provenance: list[tuple[str, Mapping[str, object]]] = []
        provenance_names: set[str] = set()
        for raw_name, raw_payload in self.series_provenance:
            name = _required_id(raw_name, "data series provenance name")
            if name not in names:
                raise ValueError(
                    "data series provenance must reference a data series."
                )
            if name in provenance_names:
                raise ValueError(
                    f"data series provenance duplicates {name!r}."
                )
            if not isinstance(raw_payload, Mapping):
                raise TypeError("data series provenance must be a mapping.")
            provenance_names.add(name)
            provenance.append(
                (
                    name,
                    MappingProxyType(
                        {
                            str(key): (
                                tuple(value)
                                if isinstance(value, (list, tuple))
                                else value
                            )
                            for key, value in raw_payload.items()
                        }
                    ),
                )
            )
        if not isinstance(self.x_provenance, Mapping):
            raise TypeError("x_provenance must be a mapping.")
        object.__setattr__(self, "block_id", _required_id(self.block_id, "block_id"))
        object.__setattr__(self, "label", _required_id(self.label, "label"))
        object.__setattr__(self, "x_name", _required_id(self.x_name, "x_name"))
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", tuple(normalized))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "series_provenance", tuple(provenance))
        object.__setattr__(
            self,
            "x_provenance",
            MappingProxyType(
                {
                    str(key): (
                        tuple(value)
                        if isinstance(value, (list, tuple))
                        else value
                    )
                    for key, value in self.x_provenance.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceResultProjection:
    """One immutable semantic projection used by plot, analysis, Copy, and XLSX."""

    tab_id: str
    current_set_id: str
    primary_layer_id: str
    plot_layers: tuple[WorkspacePlotLayer, ...]
    parameters: tuple[tuple[str, float, str], ...]
    scalar_values: tuple[tuple[str, float], ...]
    primary_statistics: ResultDataBlock | None
    copy_all_blocks: tuple[ResultDataBlock, ...]
    selected_x_series: str = "t"
    selected_y_series: tuple[str, ...] = ()
    selected_y_uses_default_all: bool = True
    available_y_series: tuple[str, ...] = ()
    parametric_mode: bool = False
    reference_lines_visible: bool = True
    intervention_annotations: tuple[Mapping[str, object], ...] = ()
    intervention_annotations_visible: bool = True

    def __post_init__(self) -> None:
        tab_id = _required_id(self.tab_id, "tab_id")
        layers = tuple(self.plot_layers)
        if any(not isinstance(layer, WorkspacePlotLayer) for layer in layers):
            raise TypeError(
                "plot_layers must contain only WorkspacePlotLayer values."
            )
        layer_ids = {layer.layer_id for layer in layers}
        if len(layer_ids) != len(layers):
            raise ValueError("plot layer IDs must be unique.")
        primary = str(self.primary_layer_id or "")
        if primary and primary not in layer_ids:
            raise ValueError("primary_layer_id must reference one plot layer.")
        current_set_id = str(self.current_set_id or "")
        if primary:
            primary_layer = next(layer for layer in layers if layer.layer_id == primary)
            if (
                not current_set_id
                or primary_layer.source_id != current_set_id
                or primary_layer.kind
                not in {ResultLayerKind.EXPLICIT, ResultLayerKind.PREVIEW}
            ):
                raise ValueError(
                    "primary must be the explicit or Preview layer for current_set_id."
                )
        elif self.parameters or self.scalar_values or self.primary_statistics is not None:
            raise ValueError("primary analysis truth requires an eligible primary layer.")
        parameters: list[tuple[str, float, str]] = []
        parameter_names: set[str] = set()
        for raw_name, raw_value, raw_unit in self.parameters:
            name = _required_id(raw_name, "projection parameter name")
            if name in parameter_names:
                raise ValueError(
                    f"projection parameters contain duplicate {name!r}."
                )
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(
                    f"projection parameter {name!r} must be finite."
                )
            parameter_names.add(name)
            parameters.append((name, value, str(raw_unit or "")))
        scalar_values: list[tuple[str, float]] = []
        scalar_names: set[str] = set()
        for raw_name, raw_value in self.scalar_values:
            name = _required_id(raw_name, "projection scalar name")
            if name in scalar_names:
                raise ValueError(
                    f"projection scalars contain duplicate {name!r}."
                )
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(
                    f"projection scalar {name!r} must be finite."
                )
            scalar_names.add(name)
            scalar_values.append((name, value))
        primary_statistics = self.primary_statistics
        if (
            primary_statistics is not None
            and not isinstance(primary_statistics, ResultDataBlock)
        ):
            raise TypeError(
                "primary_statistics must be a ResultDataBlock or None."
            )
        copy_blocks = tuple(self.copy_all_blocks)
        if any(not isinstance(block, ResultDataBlock) for block in copy_blocks):
            raise TypeError(
                "copy_all_blocks must contain only ResultDataBlock values."
            )
        selected_y = tuple(
            _required_id(name, "selected_y_series")
            for name in self.selected_y_series
        )
        if len(selected_y) != len(set(selected_y)):
            raise ValueError("selected_y_series must be unique.")
        available_y = tuple(
            _required_id(name, "available_y_series")
            for name in self.available_y_series
        )
        if len(available_y) != len(set(available_y)):
            raise ValueError("available_y_series must be unique.")
        annotations = tuple(self.intervention_annotations)
        if any(not isinstance(item, Mapping) for item in annotations):
            raise TypeError(
                "intervention_annotations must contain only mappings."
            )
        object.__setattr__(self, "tab_id", tab_id)
        object.__setattr__(self, "current_set_id", current_set_id)
        object.__setattr__(self, "primary_layer_id", primary)
        object.__setattr__(self, "plot_layers", layers)
        object.__setattr__(self, "parameters", tuple(parameters))
        object.__setattr__(self, "scalar_values", tuple(scalar_values))
        object.__setattr__(self, "primary_statistics", primary_statistics)
        object.__setattr__(self, "copy_all_blocks", copy_blocks)
        object.__setattr__(
            self,
            "selected_x_series",
            _required_id(self.selected_x_series or "t", "selected_x_series"),
        )
        object.__setattr__(
            self,
            "selected_y_series",
            selected_y,
        )
        object.__setattr__(
            self,
            "selected_y_uses_default_all",
            bool(self.selected_y_uses_default_all),
        )
        object.__setattr__(
            self,
            "available_y_series",
            available_y,
        )
        object.__setattr__(self, "parametric_mode", bool(self.parametric_mode))
        object.__setattr__(
            self,
            "reference_lines_visible",
            bool(self.reference_lines_visible),
        )
        object.__setattr__(
            self,
            "intervention_annotations",
            tuple(MappingProxyType(dict(item)) for item in annotations),
        )
        object.__setattr__(
            self,
            "intervention_annotations_visible",
            bool(self.intervention_annotations_visible),
        )


class WorkspaceResultSurface(Protocol):
    def apply_workspace_result_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None: ...


def _result_layer(
    *,
    set_id: str,
    label: str,
    result: SimulationResultValue,
    kind: ResultLayerKind,
    species_colors: Mapping[str, str],
) -> WorkspacePlotLayer:
    if kind is ResultLayerKind.PREVIEW:
        display_label = f"{label} [Preview]"
    elif kind is ResultLayerKind.REFERENCE:
        display_label = f"{label} [Reference]"
    else:
        display_label = label
    return WorkspacePlotLayer(
        layer_id=f"{kind.value}:{set_id}",
        source_id=set_id,
        label=display_label,
        kind=kind,
        t=result.t,
        series=_series_tuple(result),
        segment_ids=result.segment_ids,
        segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids,
        color_by_series=tuple(
            (name, str(species_colors[name]))
            for name in result.series
            if name in species_colors
        ),
        dashed=kind is ResultLayerKind.REFERENCE,
        opacity=0.42 if kind is ResultLayerKind.REFERENCE else 1.0,
    )


def _data_block(
    layer: WorkspacePlotLayer,
    *,
    x_name: str = "t",
    y_names: set[str] | None = None,
) -> ResultDataBlock | None:
    series_map = dict(layer.series)
    if x_name == "t":
        x_values = layer.t
    else:
        x_values = series_map.get(x_name)
        if x_values is None:
            return None
    selected = tuple(
        (name, values)
        for name, values in layer.series
        if name != x_name and (y_names is None or name in y_names)
    )
    if not selected:
        return None
    return ResultDataBlock(
        block_id=layer.layer_id,
        label=layer.label,
        t=x_values,
        series=selected,
        segment_ids=layer.segment_ids,
        segment_keys=layer.segment_keys,
        discontinuity_ids=layer.discontinuity_ids,
        x_name=x_name,
        series_provenance=tuple(
            (name, payload)
            for name, payload in layer.series_provenance
            if any(selected_name == name for selected_name, _values in selected)
        ),
    )


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if np.isfinite(parsed) else None


def _export_column_header(
    block: ResultDataBlock,
    name: str,
    provenance: Mapping[str, object],
) -> str:
    header = f"{block.label}::{name}"
    dataset_id = str(provenance.get("dataset_id") or "").strip()
    raw_observation_id = str(
        provenance.get("raw_observation_id") or ""
    ).strip()
    if dataset_id and raw_observation_id:
        return (
            f"{header} [dataset_id={dataset_id};"
            f"raw_observation_id={raw_observation_id}]"
        )
    return header


def _same_source_row_values(
    x_source: ExperimentalLayerInput,
    y_source: ExperimentalLayerInput,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]] | None:
    """Pair experimental X/Y only by their immutable physical source row."""

    if x_source.dataset_id != y_source.dataset_id:
        return None
    x_by_row = {
        int(row): float(value)
        for row, value in zip(x_source.source_rows, x_source.y, strict=True)
    }
    x_values: list[float] = []
    y_values: list[float] = []
    paired_rows: list[int] = []
    for row, value in zip(y_source.source_rows, y_source.y, strict=True):
        source_row = int(row)
        if source_row not in x_by_row:
            continue
        x_values.append(x_by_row[source_row])
        y_values.append(float(value))
        paired_rows.append(source_row)
    if not x_values:
        return None
    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        tuple(paired_rows),
    )


def _annotations_from_execution(
    result: SimulationResultValue,
) -> tuple[Mapping[str, object], ...]:
    return result.intervention_execution.annotation_projection(
        execution_start=float(result.t[0]),
        execution_end=float(result.t[-1]),
    )


class WorkspaceResultPresenter:
    """Pure composition plus the two Copy payloads and generated XLSX."""

    @staticmethod
    def compose(
        *,
        tab_id: str,
        current_set_id: str | None,
        visible_sets: Sequence[VisibleResultSet],
        repository: ResultRepository,
        reference_lines_visible: bool,
        intervention_annotations_visible: bool,
        experimental_layers: Sequence[ExperimentalLayerInput] = (),
        species_colors: Mapping[str, str] = MappingProxyType({}),
        experimental_point_size: float = 6.0,
        experimental_point_opacity: float = 0.85,
        selected_x_series: str = "t",
        selected_y_series: Sequence[str] = (),
        selected_y_uses_default_all: bool = True,
        parametric_mode: bool = False,
    ) -> WorkspaceResultProjection:
        tab_id = _required_id(tab_id, "tab_id")
        if not isinstance(repository, ResultRepository):
            raise TypeError("repository must be a ResultRepository.")
        if repository.tab_id != tab_id:
            raise ValueError("repository does not belong to the requested tab.")
        current_id = str(current_set_id or "")
        colors = {str(name): str(color) for name, color in species_colors.items()}
        selected_x = _required_id(
            selected_x_series or "t",
            "selected_x_series",
        )
        x_name = selected_x if bool(parametric_mode) else "t"
        selected_y_ordered = tuple(
            str(name) for name in selected_y_series if str(name)
        )
        selected_y = set(selected_y_ordered)
        wanted_y = (
            None
            if bool(selected_y_uses_default_all)
            else selected_y
        )

        layers: list[WorkspacePlotLayer] = []
        blocks: list[ResultDataBlock] = []
        available_y: list[str] = []
        available_y_seen: set[str] = set()
        primary_layer_id = ""
        primary_result: SimulationResultValue | None = None
        primary_block: ResultDataBlock | None = None
        annotations: list[Mapping[str, object]] = []
        seen_set_ids: set[str] = set()

        for visible in visible_sets:
            if not isinstance(visible, VisibleResultSet):
                raise TypeError("visible_sets must contain VisibleResultSet values.")
            if visible.set_id in seen_set_ids:
                raise ValueError(f"visible set {visible.set_id!r} is duplicated.")
            seen_set_ids.add(visible.set_id)

            preview = repository.preview_entry(visible.set_id)
            explicit = repository.explicit_entry(visible.set_id)
            foreground_result: SimulationResultValue | None = None
            if preview is not None:
                foreground_result = preview.result
                foreground = _result_layer(
                    set_id=visible.set_id,
                    label=visible.label,
                    result=preview.result,
                    kind=ResultLayerKind.PREVIEW,
                    species_colors=colors,
                )
                layers.append(foreground)
                for name, _values in foreground.series:
                    if name not in available_y_seen:
                        available_y_seen.add(name)
                        available_y.append(name)
                block = _data_block(
                    foreground,
                    x_name=x_name,
                    y_names=wanted_y,
                )
                if block is not None:
                    blocks.append(block)
            elif explicit is not None:
                foreground_result = explicit.result
                foreground = _result_layer(
                    set_id=visible.set_id,
                    label=visible.label,
                    result=explicit.result,
                    kind=ResultLayerKind.EXPLICIT,
                    species_colors=colors,
                )
                layers.append(foreground)
                for name, _values in foreground.series:
                    if name not in available_y_seen:
                        available_y_seen.add(name)
                        available_y.append(name)
                block = _data_block(
                    foreground,
                    x_name=x_name,
                    y_names=wanted_y,
                )
                if block is not None:
                    blocks.append(block)
            else:
                foreground = None

            if bool(reference_lines_visible):
                reference = repository.reference_entry(visible.set_id)
                if reference is not None:
                    reference_layer = _result_layer(
                        set_id=visible.set_id,
                        label=visible.label,
                        result=reference.result,
                        kind=ResultLayerKind.REFERENCE,
                        species_colors=colors,
                    )
                    layers.append(reference_layer)
                    for name, _values in reference_layer.series:
                        if name not in available_y_seen:
                            available_y_seen.add(name)
                            available_y.append(name)
                    block = _data_block(
                        reference_layer,
                        x_name=x_name,
                        y_names=wanted_y,
                    )
                    if block is not None:
                        blocks.append(block)
                    annotations.extend(
                        _annotations_from_execution(reference.result)
                    )

            if foreground is not None:
                annotations.extend(
                    _annotations_from_execution(foreground_result)
                )
            if visible.set_id == current_id and foreground is not None:
                primary_layer_id = foreground.layer_id
                primary_result = foreground_result
                primary_block = _data_block(
                    foreground,
                    x_name=x_name,
                    y_names=wanted_y,
                )

        point_size = float(experimental_point_size)
        point_opacity = float(experimental_point_opacity)
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("experimental_point_size must be positive.")
        if not np.isfinite(point_opacity) or not 0.0 <= point_opacity <= 1.0:
            raise ValueError(
                "experimental_point_opacity must be between zero and one."
            )
        experimental_inputs = tuple(experimental_layers)
        experimental_x_by_dataset: dict[str, ExperimentalLayerInput] = {}
        if bool(parametric_mode) and selected_x != "t":
            for candidate in experimental_inputs:
                if not isinstance(candidate, ExperimentalLayerInput):
                    continue
                if (
                    candidate.raw_column == selected_x
                    or candidate.presented_species == selected_x
                ):
                    experimental_x_by_dataset.setdefault(
                        candidate.dataset_id,
                        candidate,
                    )

        for experimental in experimental_inputs:
            if not isinstance(experimental, ExperimentalLayerInput):
                raise TypeError(
                    "experimental_layers must contain ExperimentalLayerInput values."
                )
            presented_species = experimental.presented_species
            experimental_x = np.asarray(experimental.t, dtype=float)
            experimental_y = np.asarray(experimental.y, dtype=float)
            experimental_x_name = "t"
            experimental_x_provenance: Mapping[str, object] = (
                MappingProxyType({})
            )
            selected_source_rows = tuple(experimental.source_rows)
            if bool(parametric_mode) and selected_x != "t":
                x_source = experimental_x_by_dataset.get(experimental.dataset_id)
                if (
                    x_source is None
                    or x_source.layer_id == experimental.layer_id
                ):
                    # Never draw experimental Y against a simulation-owned X.
                    # Exact paired experimental X data is required.
                    continue
                paired = _same_source_row_values(x_source, experimental)
                if paired is None:
                    continue
                (
                    experimental_x,
                    experimental_y,
                    selected_source_rows,
                ) = paired
                experimental_x_name = selected_x
                experimental_x_provenance = {
                    "dataset_id": x_source.dataset_id,
                    "raw_observation_id": x_source.raw_observation_id,
                    "raw_column": x_source.raw_column,
                    "source_rows": selected_source_rows,
                }
            label = f"{experimental.dataset_name}: {experimental.raw_column}"
            if experimental.model_species:
                label = f"{label} → {experimental.model_species}"
            layer = WorkspacePlotLayer(
                layer_id=f"experimental:{experimental.layer_id}",
                source_id=experimental.layer_id,
                label=label,
                kind=ResultLayerKind.EXPERIMENTAL,
                t=experimental_x,
                series=(
                    (
                        experimental.raw_column,
                        experimental_y,
                    ),
                ),
                segment_ids=np.zeros(experimental_x.size, dtype=np.int64),
                segment_keys=tuple(
                    f"experimental:{experimental.layer_id}"
                    for _ in experimental_x
                ),
                discontinuity_ids=np.zeros(
                    experimental_x.size,
                    dtype=np.int64,
                ),
                series_provenance=(
                    (
                        experimental.raw_column,
                        {
                            "dataset_id": experimental.dataset_id,
                            "raw_observation_id": (
                                experimental.raw_observation_id
                            ),
                            "raw_column": experimental.raw_column,
                            "model_species": experimental.model_species,
                            "source_rows": selected_source_rows,
                        },
                    ),
                ),
                color_by_series=(
                    (
                        experimental.raw_column,
                        colors.get(presented_species, "#4c78a8"),
                    ),
                ),
                opacity=point_opacity,
                markers=True,
                point_size=point_size,
            )
            layers.append(layer)
            for name, _values in layer.series:
                if name not in available_y_seen:
                    available_y_seen.add(name)
                    available_y.append(name)
            if wanted_y is None or (
                experimental.raw_column in wanted_y
                or presented_species in wanted_y
            ):
                blocks.append(
                    ResultDataBlock(
                        block_id=layer.layer_id,
                        label=layer.label,
                        t=experimental_x,
                        series=layer.series,
                        segment_ids=layer.segment_ids,
                        segment_keys=layer.segment_keys,
                        discontinuity_ids=layer.discontinuity_ids,
                        x_name=experimental_x_name,
                        series_provenance=layer.series_provenance,
                        x_provenance=experimental_x_provenance,
                    )
                )

        unique_annotations: list[Mapping[str, object]] = []
        annotation_signatures: set[tuple[tuple[str, str], ...]] = set()
        for annotation in annotations:
            signature = tuple(
                sorted((str(name), repr(value)) for name, value in annotation.items())
            )
            if signature in annotation_signatures:
                continue
            annotation_signatures.add(signature)
            unique_annotations.append(annotation)
        if bool(parametric_mode) and selected_x != "t":
            unique_annotations.clear()

        return WorkspaceResultProjection(
            tab_id=tab_id,
            current_set_id=current_id,
            primary_layer_id=primary_layer_id,
            plot_layers=tuple(layers),
            parameters=tuple(primary_result.parameters) if primary_result else (),
            scalar_values=(
                tuple(primary_result.scalars.items()) if primary_result else ()
            ),
            primary_statistics=primary_block,
            copy_all_blocks=tuple(blocks),
            selected_x_series=selected_x,
            selected_y_series=selected_y_ordered,
            selected_y_uses_default_all=bool(
                selected_y_uses_default_all
            ),
            available_y_series=tuple(available_y),
            parametric_mode=bool(parametric_mode),
            reference_lines_visible=bool(reference_lines_visible),
            intervention_annotations=tuple(unique_annotations),
            intervention_annotations_visible=bool(
                intervention_annotations_visible
                and not (bool(parametric_mode) and selected_x != "t")
            ),
        )

    @staticmethod
    def apply(
        surface: WorkspaceResultSurface,
        projection: WorkspaceResultProjection,
    ) -> None:
        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        surface.apply_workspace_result_projection(projection)

    @staticmethod
    def iter_blocks_tsv_chunks(
        blocks: Sequence[ResultDataBlock],
        *,
        rows_per_chunk: int = 2048,
    ) -> Iterator[str]:
        columns: list[tuple[str, Sequence[object], str]] = []
        for block in blocks:
            if not isinstance(block, ResultDataBlock):
                raise TypeError("Copy data must contain ResultDataBlock values.")
            columns.append(
                (
                    _export_column_header(
                        block,
                        block.x_name,
                        block.x_provenance,
                    ),
                    block.t,
                    "float",
                )
            )
            columns.extend(
                (
                    (f"{block.label} — segment_id", block.segment_ids, "int"),
                    (f"{block.label} — segment_key", block.segment_keys, "text"),
                    (
                        f"{block.label} — discontinuity_id",
                        block.discontinuity_ids,
                        "int",
                    ),
                )
            )
            provenance_by_name = dict(block.series_provenance)
            columns.extend(
                (
                    _export_column_header(
                        block,
                        name,
                        provenance_by_name.get(name, MappingProxyType({})),
                    ),
                    values,
                    "float",
                )
                for name, values in block.series
            )
        if not columns:
            raise ValueError("No visible result, Reference, or experimental data.")
        row_count = max(len(values) for _header, values, _kind in columns)
        cell_count = (row_count + 1) * len(columns)
        if cell_count > _MAX_COPY_CELLS:
            raise ValueError(
                "Copy payload exceeds the bounded full-resolution cell limit "
                f"({_MAX_COPY_CELLS:,})."
            )
        chunk_size = int(rows_per_chunk)
        if chunk_size < 1:
            raise ValueError("rows_per_chunk must be positive.")
        total_characters = 0
        header = (
            "\t".join(header for header, _values, _kind in columns)
            + "\n"
        )
        total_characters += len(header)
        yield header
        for start in range(0, row_count, chunk_size):
            output = io.StringIO()
            for index in range(start, min(start + chunk_size, row_count)):
                output.write(
                    "\t".join(
                        (
                            f"{float(values[index]):.17g}"
                            if kind == "float"
                            else (
                                str(int(values[index]))
                                if kind == "int"
                                else str(values[index])
                            )
                        )
                        if index < len(values)
                        else ""
                        for _header, values, kind in columns
                    )
                )
                output.write("\n")
            chunk = output.getvalue()
            total_characters += len(chunk)
            if total_characters > _MAX_COPY_CHARACTERS:
                raise ValueError(
                    "Copy payload exceeds the bounded output-size limit "
                    f"({_MAX_COPY_CHARACTERS // (1024 * 1024)} MiB)."
                )
            yield chunk

    @staticmethod
    def write_xlsx(
        projection: WorkspaceResultProjection,
        destination: BinaryIO,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> XlsxExportCompletion:
        """Write full-resolution worksheets directly to a bounded destination."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if not projection.copy_all_blocks:
            raise ValueError("No visible result, Reference, or experimental data.")
        total_cells = sum(
            (int(block.t.size) + 1) * (4 + len(block.series))
            for block in projection.copy_all_blocks
        )
        if total_cells > _MAX_XLSX_CELLS:
            raise ValueError(
                "XLSX payload exceeds the bounded full-resolution cell limit "
                f"({_MAX_XLSX_CELLS:,})."
            )
        output = _BoundedXlsxDestination(
            destination,
            cancellation_check=cancellation_check,
        )
        workbook = Workbook(write_only=True)
        used: set[str] = set()
        lineage_rows: list[tuple[str, str, Mapping[str, object]]] = []
        data_rows = 0
        archive: ZipFile | None = None
        original_failure: BaseException | None = None
        try:
            for block in projection.copy_all_blocks:
                if cancellation_check is not None:
                    cancellation_check()
                title = WorkspaceResultPresenter._worksheet_title(block.label, used)
                sheet = workbook.create_sheet(title=title)
                sheet.append(
                    [
                        block.x_name,
                        "segment_id",
                        "segment_key",
                        "discontinuity_id",
                        *(name for name, _values in block.series),
                    ]
                )
                for index in range(block.t.size):
                    if cancellation_check is not None and index % 256 == 0:
                        cancellation_check()
                    sheet.append(
                        [
                            float(block.t[index]),
                            int(block.segment_ids[index]),
                            block.segment_keys[index],
                            int(block.discontinuity_ids[index]),
                            *(
                                float(values[index])
                                for _name, values in block.series
                            ),
                        ]
                    )
                    data_rows += 1
                if block.x_provenance:
                    lineage_rows.append(
                        (title, block.x_name, block.x_provenance)
                    )
                lineage_rows.extend(
                    (title, name, provenance)
                    for name, provenance in block.series_provenance
                    if provenance
                )
            if lineage_rows:
                if cancellation_check is not None:
                    cancellation_check()
                lineage_sheet = workbook.create_sheet(
                    title=WorkspaceResultPresenter._worksheet_title(
                        "_Kindred Lineage",
                        used,
                    )
                )
                lineage_sheet.append(
                    [
                        "worksheet",
                        "series",
                        "dataset_id",
                        "raw_observation_id",
                        "raw_column",
                        "model_species",
                        "source_rows",
                    ]
                )
                for title, name, provenance in lineage_rows:
                    if cancellation_check is not None:
                        cancellation_check()
                    lineage_sheet.append(
                        [
                            title,
                            name,
                            str(provenance.get("dataset_id") or ""),
                            str(
                                provenance.get("raw_observation_id") or ""
                            ),
                            str(provenance.get("raw_column") or ""),
                            str(provenance.get("model_species") or ""),
                            json.dumps(
                                tuple(provenance.get("source_rows") or ()),
                                separators=(",", ":"),
                            ),
                        ]
                    )
                lineage_sheet.sheet_state = "hidden"
            if cancellation_check is not None:
                cancellation_check()
            archive = ZipFile(
                output,
                "w",
                ZIP_DEFLATED,
                allowZip64=True,
            )
            ExcelWriter(workbook, archive).save()
            return XlsxExportCompletion(
                worksheet_count=len(used),
                data_row_count=data_rows,
                cell_count=total_cells,
                bytes_written=output.bytes_written,
            )
        except BaseException as exc:
            original_failure = exc
            raise
        finally:
            cleanup_failures: list[BaseException] = []
            if archive is not None:
                try:
                    archive.close()
                except BaseException as exc:
                    cleanup_failures.append(exc)
            cleanup_failures.extend(
                _cleanup_write_only_workbook(workbook)
            )
            if original_failure is None and cleanup_failures:
                raise cleanup_failures[0]

    @staticmethod
    def _worksheet_title(label: str, used: set[str]) -> str:
        invalid = set("[]:*?/\\")
        base = "".join(
            "_"
            if char in invalid or ord(char) < 32
            else char
            for char in str(label)
        ).strip()
        base = base.strip("'") or "Result"
        base = base[:31]
        candidate = base
        suffix_index = 2
        while candidate.casefold() in used:
            suffix = f" ({suffix_index})"
            candidate = f"{base[: 31 - len(suffix)]}{suffix}"
            suffix_index += 1
        used.add(candidate.casefold())
        return candidate

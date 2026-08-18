"""Stateless composition and export for one retained workspace result page."""

from __future__ import annotations

import io
import csv
import json
import unicodedata
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from openpyxl import Workbook
from openpyxl.cell.cell import WriteOnlyCell
from openpyxl.writer.excel import ExcelWriter

from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import (
    ResultRepository,
    SimulationResultValue,
)
from kindred.core.simulation_failure import FactualCause
from kindred.core.simulator.solver_types import SimulationPresentationBoundary
from kindred.gui.dataset_payload_snapshot import (
    ExperimentalLayerIdentity,
    ExperimentalLayerInput,
)
from kindred.gui.plot_config import AxisSelectionCandidate

__all__ = [
    "InteractiveDisplayCurve",
    "InteractiveDisplayProjection",
    "ResultDataBlock",
    "ResultLayerKind",
    "ResultRenderInput",
    "VisibleResultSet",
    "XlsxExportCompletion",
    "compose_interactive_display",
    "compose_interactive_display_batch",
    "compose_result_input",
    "encode_clipboard_text",
    "encode_tsv_row",
    "iter_blocks_tsv_chunks",
    "validate_copy",
    "validate_xlsx_export",
    "write_xlsx",
]

_MAX_COPY_CELLS = 1_000_000
_MAX_COPY_CHARACTERS = 64 * 1024 * 1024
_MAX_XLSX_STRING_LENGTH = 32_767
_EXCEL_MAX_ROWS = 1_048_576
_EXCEL_MAX_COLUMNS = 16_384


@dataclass(frozen=True, slots=True)
class InteractiveDisplayCurve:
    """One immutable, transient physical curve; never Copy/export authority."""

    key: tuple[str, ...]
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    dashed: bool = False
    markers: bool = False
    point_size: float = 6.0
    opacity: float = 1.0
    # These arrays are display topology, not result authority.  Keeping them
    # with the transient curve prevents the preview renderer from connecting
    # points across an intervention/discontinuity while it is cross-fading.
    segment_ids: np.ndarray | None = None
    segment_keys: tuple[str, ...] | None = None
    discontinuity_ids: np.ndarray | None = None
    intervention_connectors: tuple[tuple[int, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        key = tuple(_required_id(value, "interactive curve key") for value in self.key)
        if not key:
            raise ValueError("interactive curve key must not be empty.")
        x = _owned_vector(self.x, field="interactive curve X")
        y = _owned_vector(self.y, field="interactive curve Y")
        if x.size != y.size:
            raise ValueError("interactive curve X and Y must have equal length.")
        point_size = float(self.point_size)
        opacity = float(self.opacity)
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("interactive curve point size must be positive.")
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("interactive curve opacity must be between zero and one.")
        segment_ids = None
        segment_keys = None
        discontinuity_ids = None
        if self.segment_ids is not None:
            segment_ids = _owned_identity_vector(
                self.segment_ids,
                field="interactive curve segment identity",
                size=x.size,
            )
            if self.segment_keys is None or self.discontinuity_ids is None:
                raise ValueError(
                    "interactive curve topology requires segment keys and "
                    "discontinuity identity."
                )
            segment_keys = tuple(
                _required_id(value, "interactive curve segment key")
                for value in self.segment_keys
            )
            if len(segment_keys) != x.size:
                raise ValueError(
                    "interactive curve segment keys must align with X values."
                )
            discontinuity_ids = _owned_identity_vector(
                self.discontinuity_ids,
                field="interactive curve discontinuity identity",
                size=x.size,
            )
        elif self.segment_keys is not None or self.discontinuity_ids is not None:
            raise ValueError(
                "interactive curve topology must provide segment IDs, keys, "
                "and discontinuity IDs together."
            )
        connectors = tuple(
            (
                int(boundary_index),
                float(boundary_time),
                float(pre_value),
                float(post_value),
            )
            for boundary_index, boundary_time, pre_value, post_value in
            self.intervention_connectors
        )
        if any(
            not np.all(np.isfinite(values[1:]))
            for values in connectors
        ):
            raise ValueError("interactive intervention connectors must be finite.")
        color = _required_id(self.color, "interactive curve color")
        object.__setattr__(self, "key", key)
        object.__setattr__(
            self, "label", _required_id(self.label, "interactive curve label")
        )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "dashed", bool(self.dashed))
        object.__setattr__(self, "markers", bool(self.markers))
        object.__setattr__(self, "point_size", point_size)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "intervention_connectors", connectors)


@dataclass(frozen=True, slots=True)
class InteractiveDisplayProjection:
    """Transient physical roster composed independently of result authority."""

    tab_id: str
    set_id: str
    generation: int
    x_label: str
    y_label: str
    curves: tuple[InteractiveDisplayCurve, ...]
    target_set_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        generation = self.generation
        if type(generation) is not int or generation <= 0:
            raise ValueError("interactive display generation must be positive.")
        curves = tuple(self.curves)
        if any(not isinstance(curve, InteractiveDisplayCurve) for curve in curves):
            raise TypeError("curves must contain InteractiveDisplayCurve values.")
        keys = tuple(curve.key for curve in curves)
        if len(keys) != len(set(keys)):
            raise ValueError("interactive display curve keys must be unique.")
        set_id = _required_id(self.set_id, "set_id")
        target_set_ids = tuple(
            _required_id(value, "target_set_id") for value in self.target_set_ids
        )
        if not target_set_ids:
            raise ValueError("interactive display target roster is required.")
        if len(set(target_set_ids)) != len(target_set_ids):
            raise ValueError("interactive display target roster must be unique.")
        if set_id not in target_set_ids:
            raise ValueError("set_id must belong to the interactive target roster.")
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", set_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "x_label", _required_id(self.x_label, "x_label"))
        object.__setattr__(self, "y_label", _required_id(self.y_label, "y_label"))
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "target_set_ids", target_set_ids)


class _XlsxDestination:
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
        written = int(self._destination.write(payload))
        if written != len(payload):
            raise OSError(
                f"XLSX destination accepted only {written} of {len(payload)} bytes."
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


def _axis_selection(value: object, *, field: str) -> AxisSeriesSelection:
    if not isinstance(
        value,
        (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
    ):
        raise TypeError(
            f"{field} must be a typed simulation or experimental axis selection."
        )
    return value


def _is_time_selection(value: AxisSeriesSelection) -> bool:
    return isinstance(value, SimulationSeriesAxisSelection) and value.series_name == "t"


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_safe(getattr(value, item.name)) for item in fields(value)
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def encode_tsv_row(values: Sequence[object]) -> str:
    """Encode one RFC-style quoted tab-separated row with a final newline."""

    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        dialect="excel-tab",
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    writer.writerow(tuple("" if value is None else str(value) for value in values))
    return output.getvalue()


def encode_clipboard_text(value: str) -> str:
    """Protect one dynamic clipboard text cell from spreadsheet formulas."""

    if not isinstance(value, str):
        raise TypeError("clipboard text must be a string.")
    index = 0
    while index < len(value):
        character = value[index]
        if character.isspace() or unicodedata.category(character).startswith("C"):
            index += 1
            continue
        if character in "=+-@":
            return "'" + value
        break
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _owned_vector(value: object, *, field: str) -> np.ndarray:
    candidate = np.asarray(value, dtype=float)
    if candidate.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional.")
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
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{field} must be a one-dimensional integer array.")
    candidate = raw.astype(np.int64, copy=False)
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


def _owned_source_rows(
    value: object,
    *,
    size: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("source_row must be a one-dimensional integer array.")
    candidate = raw.astype(np.int64, copy=False)
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
        raise ValueError("source_row must align with layer time.")
    if np.any(array <= 0) or np.unique(array).size != array.size:
        raise ValueError("source_row must contain unique positive row numbers.")
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


def _result_warning_details(result: SimulationResultValue | None) -> tuple[str, ...]:
    if not isinstance(result, SimulationResultValue):
        raise TypeError("warning details require a canonical simulation result.")
    details = [f"Preparation: {warning}" for warning in result.preparation_warnings]
    if result.failed_algebra_outputs:
        details.append(
            "Algebra failed: " + ", ".join(result.failed_algebra_outputs)
        )
    for cause in result.algebra_errors:
        if not isinstance(cause, FactualCause):
            raise TypeError("algebra errors must contain FactualCause values.")
        raw_name = cause.details.get("name")
        observable_name = str(raw_name or "<unknown observable>")
        code = cause.code if cause.code is not None else "<none>"
        details.append(
            "Algebra error for "
            f"{observable_name}: type={cause.type}; code={code}; "
            f"message={cause.message}"
        )
    if result.omitted_algebra_outputs:
        details.append(
            "Algebra omitted: " + ", ".join(result.omitted_algebra_outputs)
        )
    return tuple(details)


def _result_provenance_payload(result: SimulationResultValue) -> Mapping[str, object]:
    """Serialize readable provenance directly from the canonical result."""

    if not isinstance(result, SimulationResultValue):
        raise TypeError("result metadata requires a canonical simulation result.")
    return MappingProxyType(
        {
            "provenance": MappingProxyType(
                {
                    "solver": result.solver_provenance,
                    "preparation_warnings": result.preparation_warnings,
                    "algebra": MappingProxyType(
                        {
                            "requested": result.requested_algebra_outputs,
                            "succeeded": result.succeeded_algebra_outputs,
                            "failed": result.failed_algebra_outputs,
                            "omitted": result.omitted_algebra_outputs,
                            "errors": tuple(
                                cause.to_payload()
                                for cause in result.algebra_errors
                            ),
                        }
                    ),
                }
            ),
        }
    )


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
class ResultDataBlock:
    """One immutable full-resolution result/data table and its render style."""

    block_id: str | ExperimentalLayerIdentity
    label: str
    kind: ResultLayerKind
    t: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    series_axes: tuple[tuple[str, AxisSeriesSelection], ...]
    x_axis: AxisSeriesSelection
    result: SimulationResultValue | None
    segment_ids: np.ndarray | None
    segment_keys: tuple[str, ...] | None
    discontinuity_ids: np.ndarray | None
    source_row: np.ndarray | None
    x_name: str = "t"
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...] = ()
    x_provenance: Mapping[str, object] = MappingProxyType({})
    color_by_series: tuple[tuple[str, str], ...] = ()
    visible: bool = True
    dashed: bool = False
    opacity: float = 1.0
    markers: bool = False
    point_size: float = 6.0
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResultLayerKind):
            raise TypeError("kind must be a ResultLayerKind.")
        block_id = self.block_id
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if not isinstance(block_id, ExperimentalLayerIdentity):
                raise TypeError("experimental data blocks require an experimental identity.")
        elif type(block_id) is not str or not block_id.strip():
            raise TypeError("simulation data blocks require a nonblank set ID.")
        time = _owned_vector(self.t, field="data time")
        normalized: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for raw_name, raw_values in self.series:
            name = _required_id(raw_name, "data series name")
            if name in names:
                raise ValueError(f"data block contains duplicate series {name!r}.")
            values = _owned_vector(raw_values, field=f"data series {name!r}")
            if values.size != time.size:
                raise ValueError(f"data series {name!r} length does not match time.")
            names.add(name)
            normalized.append((name, values))
        if not normalized:
            raise ValueError("data block requires at least one series.")
        series_axes = tuple(
            (_required_id(name, "data axis series name"), _axis_selection(selection, field="data series axis"))
            for name, selection in self.series_axes
        )
        if tuple(name for name, _ in series_axes) != tuple(name for name, _ in normalized):
            raise ValueError("data series axes must align exactly with data series.")
        if len({selection for _, selection in series_axes}) != len(series_axes):
            raise ValueError("data series axis identities must be unique.")
        x_axis = _axis_selection(self.x_axis, field="data X axis")
        result = self.result
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if result is not None:
                raise ValueError("experimental data blocks prohibit repository entries.")
            if any(value is not None for value in (self.segment_ids, self.segment_keys, self.discontinuity_ids)):
                raise ValueError("experimental data blocks prohibit solver identity.")
            source_row = _owned_source_rows(self.source_row, size=time.size)
            segment_ids = segment_keys = discontinuity_ids = None
            boundaries = ()
        else:
            if not isinstance(x_axis, SimulationSeriesAxisSelection):
                raise TypeError("simulation data blocks require a simulation X axis.")
            if not isinstance(result, SimulationResultValue):
                raise TypeError("simulation data blocks require a canonical result.")
            if self.source_row is not None:
                raise ValueError("simulation data blocks prohibit observation source_row.")
            if self.segment_ids is None or self.segment_keys is None or self.discontinuity_ids is None:
                raise ValueError("simulation data blocks require complete executor identity.")
            segment_ids = _owned_identity_vector(self.segment_ids, field="data segment identity", size=time.size)
            segment_keys = _segment_keys(self.segment_keys, size=time.size)
            discontinuity_ids = _owned_identity_vector(self.discontinuity_ids, field="data discontinuity identity", size=time.size)
            source_row = None
            boundaries = tuple(self.presentation_boundaries)
            if boundaries is not result.presentation_boundaries:
                raise ValueError("simulation data blocks must retain canonical presentation boundaries.")
        provenance: list[tuple[str, Mapping[str, object]]] = []
        provenance_names: set[str] = set()
        for raw_name, raw_payload in self.series_provenance:
            name = _required_id(raw_name, "data series provenance name")
            if name not in names or name in provenance_names:
                raise ValueError("data series provenance must uniquely reference data.")
            if not isinstance(raw_payload, Mapping):
                raise TypeError("data series provenance must be a mapping.")
            provenance_names.add(name)
            provenance.append((name, MappingProxyType({
                str(key): tuple(value) if isinstance(value, (list, tuple)) else value
                for key, value in raw_payload.items()
            })))
        if not isinstance(self.x_provenance, Mapping):
            raise TypeError("x_provenance must be a mapping.")
        provenance_by_name = dict(provenance)
        for name, selection in series_axes:
            if self.kind is ResultLayerKind.EXPERIMENTAL:
                if not isinstance(selection, ExperimentalAxisSelection):
                    raise TypeError("experimental data series require experimental axis identity.")
                payload = provenance_by_name.get(name)
                if payload is None or payload.get("dataset_id") != selection.dataset_id or payload.get("raw_observation_id") != selection.raw_observation_id:
                    raise ValueError("experimental data identity must match raw-ID provenance.")
            elif not isinstance(selection, SimulationSeriesAxisSelection):
                raise TypeError("simulation data series require simulation axis identity.")
        if isinstance(x_axis, ExperimentalAxisSelection) and (
            self.x_provenance.get("dataset_id") != x_axis.dataset_id
            or self.x_provenance.get("raw_observation_id") != x_axis.raw_observation_id
        ):
            raise ValueError("experimental X identity must match raw-ID provenance.")
        colors = tuple(
            (_required_id(name, "color series name"), _required_id(color, "data series color"))
            for name, color in self.color_by_series
        )
        if {name for name, _color in colors} != names:
            raise ValueError("data series colors must cover every data series.")
        if len(colors) != len({name for name, _color in colors}):
            raise ValueError("data series colors must be unique by series.")
        opacity = float(self.opacity)
        point_size = float(self.point_size)
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between zero and one.")
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_size must be positive.")
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "label", _required_id(self.label, "label"))
        object.__setattr__(self, "x_name", _required_id(self.x_name, "x_name"))
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", tuple(normalized))
        object.__setattr__(self, "series_axes", series_axes)
        object.__setattr__(self, "x_axis", x_axis)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "source_row", source_row)
        object.__setattr__(self, "series_provenance", tuple(provenance))
        object.__setattr__(self, "x_provenance", MappingProxyType({
            str(key): tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in self.x_provenance.items()
        }))
        object.__setattr__(self, "color_by_series", colors)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "dashed", bool(self.dashed))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "markers", bool(self.markers))
        object.__setattr__(self, "point_size", point_size)
        object.__setattr__(self, "presentation_boundaries", boundaries)


@dataclass(frozen=True, slots=True)
class ResultRenderInput:
    """Call-local render facts; never durable result or export authority."""

    tab_id: str
    current_set_id: str
    blocks: tuple[ResultDataBlock, ...]
    statistics_series_names: tuple[str, ...] = ()
    selected_x_series: AxisSeriesSelection = SimulationSeriesAxisSelection("t")
    selected_y_series: tuple[AxisSeriesSelection, ...] = ()
    mapped_overlay_locks: tuple[tuple[ExperimentalAxisSelection, str], ...] = ()
    selected_y_uses_default_all: bool = True
    x_label: str = "Time (s)"
    y_label: str = "Value"
    axis_candidates: tuple[AxisSelectionCandidate, ...] = ()
    reference_lines_visible: bool = True
    intervention_annotations: tuple[Mapping[str, object], ...] = ()
    intervention_annotations_visible: bool = True
    unavailable_x_diagnostic: str | None = None
    completion_warning_details: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "current_set_id", str(self.current_set_id or ""))
        blocks = tuple(self.blocks)
        if any(not isinstance(block, ResultDataBlock) for block in blocks):
            raise TypeError("blocks must contain ResultDataBlock values.")
        identities = tuple((block.kind, block.block_id) for block in blocks)
        if len(identities) != len(set(identities)):
            raise ValueError("render blocks must have unique semantic identities.")
        object.__setattr__(self, "blocks", blocks)
        names = tuple(
            _required_id(name, "statistics series name")
            for name in self.statistics_series_names
        )
        if len(names) != len(set(names)):
            raise ValueError("statistics series names must be unique.")
        object.__setattr__(self, "statistics_series_names", names)
        selected_x = _axis_selection(self.selected_x_series, field="selected_x_series")
        selected_y = tuple(_axis_selection(item, field="selected_y_series") for item in self.selected_y_series)
        if len(selected_y) != len(set(selected_y)):
            raise ValueError("selected_y_series must be unique.")
        overlay_locks = tuple(
            (axis, _required_id(model_target, "mapped overlay model target"))
            for axis, model_target in self.mapped_overlay_locks
        )
        if any(not isinstance(axis, ExperimentalAxisSelection) for axis, _ in overlay_locks):
            raise TypeError("mapped overlay locks must contain experimental axes.")
        if len({axis for axis, _ in overlay_locks}) != len(overlay_locks):
            raise ValueError("mapped overlay locks must be unique by axis.")
        candidates = tuple(self.axis_candidates)
        if any(not isinstance(item, AxisSelectionCandidate) for item in candidates):
            raise TypeError("axis candidates must contain AxisSelectionCandidate values.")
        candidate_ids = tuple(item.selection for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)) or selected_x not in candidate_ids:
            raise ValueError("axis candidates must be unique and include selected X.")
        if any(axis not in candidate_ids for axis, _ in overlay_locks):
            raise ValueError("mapped overlay locks must be axis candidates.")
        annotations = tuple(self.intervention_annotations)
        if any(not isinstance(item, Mapping) for item in annotations):
            raise TypeError("intervention annotations must be mappings.")
        diagnostic = self.unavailable_x_diagnostic
        if diagnostic is not None and (type(diagnostic) is not str or not diagnostic.strip()):
            raise ValueError("unavailable X diagnostic must be nonblank.")
        warnings = tuple(
            (
                _required_id(label, "completion warning label"),
                tuple(str(detail) for detail in details),
            )
            for label, details in self.completion_warning_details
        )
        if len({label for label, _details in warnings}) != len(warnings):
            raise ValueError("completion warning labels must be unique.")
        object.__setattr__(self, "selected_x_series", selected_x)
        object.__setattr__(self, "selected_y_series", selected_y)
        object.__setattr__(self, "mapped_overlay_locks", overlay_locks)
        object.__setattr__(self, "selected_y_uses_default_all", bool(self.selected_y_uses_default_all))
        object.__setattr__(self, "x_label", str(self.x_label))
        object.__setattr__(self, "y_label", str(self.y_label))
        object.__setattr__(self, "axis_candidates", candidates)
        object.__setattr__(self, "reference_lines_visible", bool(self.reference_lines_visible))
        object.__setattr__(self, "intervention_annotations", tuple(MappingProxyType(dict(item)) for item in annotations))
        object.__setattr__(self, "intervention_annotations_visible", bool(self.intervention_annotations_visible))
        object.__setattr__(self, "unavailable_x_diagnostic", diagnostic)
        object.__setattr__(self, "completion_warning_details", warnings)

    def _analysis_block(self) -> ResultDataBlock | None:
        candidates = tuple(
            block
            for block in self.blocks
            if block.block_id == self.current_set_id
            and block.kind in {
                ResultLayerKind.EXPLICIT,
                ResultLayerKind.REFERENCE,
            }
            and isinstance(block.result, SimulationResultValue)
        )
        return next(
            (item for item in candidates if item.kind is ResultLayerKind.EXPLICIT),
            next(
                (item for item in candidates if item.kind is ResultLayerKind.REFERENCE),
                None,
            ),
        )

    @property
    def statistics_result(self) -> SimulationResultValue | None:
        block = self._analysis_block()
        return block.result if block is not None else None

    @property
    def parameters(self) -> tuple[tuple[str, float, str], ...]:
        block = self._analysis_block()
        result = block.result if block is not None else None
        return tuple(result.parameters) if isinstance(result, SimulationResultValue) else ()

    @property
    def scalar_values(self) -> tuple[tuple[str, float], ...]:
        block = self._analysis_block()
        result = block.result if block is not None else None
        return tuple(result.scalars.items()) if isinstance(result, SimulationResultValue) else ()

    @property
    def completion_warning_tooltip(self) -> str:
        return "\n\n".join(
            f"{label}:\n" + "\n".join(f"  • {detail}" for detail in details)
            for label, details in self.completion_warning_details
        )



def _result_block(
    *,
    set_id: str,
    label: str,
    result: SimulationResultValue,
    kind: ResultLayerKind,
    color_by_axis_selection: Mapping[AxisSeriesSelection, str],
    visible: bool = True,
) -> ResultDataBlock:
    if not isinstance(result, SimulationResultValue):
        raise TypeError("result block requires a canonical simulation result.")
    display_label = (
        f"{label} [Preview]"
        if kind is ResultLayerKind.PREVIEW
        else f"{label} [Reference]"
        if kind is ResultLayerKind.REFERENCE
        else label
    )
    series_axes = tuple((name, SimulationSeriesAxisSelection(name)) for name in result.series)
    return ResultDataBlock(
        block_id=set_id, label=display_label, kind=kind, t=result.t,
        series=_series_tuple(result), series_axes=series_axes,
        x_axis=SimulationSeriesAxisSelection("t"), result=result,
        segment_ids=result.segment_ids, segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids, source_row=None,
        color_by_series=tuple(
            (name, str(color_by_axis_selection[selection]))
            for name, selection in series_axes
        ),
        visible=visible, dashed=kind is ResultLayerKind.REFERENCE,
        opacity=0.42 if kind is ResultLayerKind.REFERENCE else 1.0,
        presentation_boundaries=result.presentation_boundaries,
    )


def _data_block(
    block: ResultDataBlock,
    *,
    x_axis: AxisSeriesSelection,
    y_axes: set[AxisSeriesSelection] | None = None,
    excluded_y_axes: set[AxisSeriesSelection] | None = None,
) -> ResultDataBlock | None:
    series_axis_map = dict(block.series_axes)
    if _is_time_selection(x_axis):
        x_values, x_name = block.t, "t"
        x_provenance: Mapping[str, object] = MappingProxyType({})
    elif isinstance(x_axis, SimulationSeriesAxisSelection) and block.kind is not ResultLayerKind.EXPERIMENTAL:
        x_name = next((name for name, selection in block.series_axes if selection == x_axis), "")
        if not x_name:
            return None
        x_values = dict(block.series)[x_name]
        x_provenance = dict(block.series_provenance).get(x_name, MappingProxyType({}))
    else:
        return None
    selected = tuple(
        (name, values)
        for name, values in block.series
        if series_axis_map[name] != x_axis
        and (excluded_y_axes is None or series_axis_map[name] not in excluded_y_axes)
        and (y_axes is None or series_axis_map[name] in y_axes)
    )
    if not selected:
        return None
    names = {name for name, _ in selected}
    return ResultDataBlock(
        block_id=block.block_id, label=block.label, kind=block.kind,
        t=x_values, series=selected,
        series_axes=tuple((name, selection) for name, selection in block.series_axes if name in names),
        x_axis=x_axis, result=block.result,
        segment_ids=block.segment_ids, segment_keys=block.segment_keys,
        discontinuity_ids=block.discontinuity_ids, source_row=block.source_row,
        x_name=x_name,
        series_provenance=tuple((name, payload) for name, payload in block.series_provenance if name in names),
        x_provenance=x_provenance,
        color_by_series=tuple((name, color) for name, color in block.color_by_series if name in names),
        visible=block.visible, dashed=block.dashed, opacity=block.opacity,
        markers=block.markers, point_size=block.point_size,
        presentation_boundaries=block.presentation_boundaries,
    )


def _has_selected_y_series(
    block: ResultDataBlock,
    *,
    y_axes: set[AxisSeriesSelection] | None,
    excluded_y_axes: set[AxisSeriesSelection] | None,
) -> bool:
    return any(
        (excluded_y_axes is None or selection not in excluded_y_axes)
        and (y_axes is None or selection in y_axes)
        for _name, selection in block.series_axes
    )


def _statistics_series_names(
    block: ResultDataBlock,
    *,
    y_axes: set[AxisSeriesSelection] | None,
    excluded_y_axes: set[AxisSeriesSelection] | None = None,
) -> tuple[str, ...]:
    if not isinstance(block.result, SimulationResultValue):
        raise TypeError("statistics require a canonical simulation result.")
    series_axes = dict(block.series_axes)
    succeeded = set(block.result.succeeded_algebra_outputs)
    return tuple(
        name for name in block.result.series
        if name not in succeeded
        and (excluded_y_axes is None or series_axes[name] not in excluded_y_axes)
        and (y_axes is None or series_axes[name] in y_axes)
    )


def _selected_x_unavailable_diagnostic(
    selected_x: AxisSeriesSelection,
    unavailable_labels: Sequence[str],
) -> str | None:
    if _is_time_selection(selected_x) or not unavailable_labels:
        return None
    labels = ", ".join(repr(str(label)) for label in unavailable_labels)
    selected_identity = (
        f"series {selected_x.series_name!r}"
        if isinstance(selected_x, SimulationSeriesAxisSelection)
        else f"raw observation {selected_x.raw_observation_id!r} in dataset {selected_x.dataset_id!r}"
    )
    return (
        f"Selected X {selected_identity} is unavailable for result layer(s) {labels}. "
        "The semantic result layer(s) are retained, but plot, Copy, and XLSX rows "
        "cannot be produced without X values."
    )


def _export_column_header(
    block: ResultDataBlock,
    name: str,
    provenance: Mapping[str, object],
) -> str:
    header = f"{block.label}::{name}"
    identity = {
        key: str(provenance[key])
        for key in (
            "dataset_id",
            "raw_observation_id",
            "dataset_name",
            "raw_column",
            "model_target",
            "set_id",
        )
        if provenance.get(key) not in {None, ""}
    }
    if identity:
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        header = f"{header} [provenance={encoded}]"
    return encode_clipboard_text(header)


def _unique_result_entries(
    blocks: Sequence[ResultDataBlock],
) -> tuple[SimulationResultValue, ...]:
    seen: set[int] = set()
    ordered: list[SimulationResultValue] = []
    for block in blocks:
        entry = block.result
        if entry is None:
            continue
        entry_identity = id(entry)
        if entry_identity in seen:
            continue
        seen.add(entry_identity)
        ordered.append(entry)
    return tuple(ordered)


def _xlsx_literal_cell(sheet: object, value: object) -> object:
    """Single XLSX cell boundary: strings are always literal, never formulas."""

    if not isinstance(value, str):
        return value
    length = len(value)
    if length > _MAX_XLSX_STRING_LENGTH:
        raise ValueError(
            "XLSX string cell exceeds Excel's physical limit of "
            f"{_MAX_XLSX_STRING_LENGTH:,} characters (got {length:,})."
        )
    cell = WriteOnlyCell(sheet, value=value)
    cell.data_type = "s"
    return cell


def _append_xlsx_row(sheet: object, values: Sequence[object]) -> None:
    append = getattr(sheet, "append", None)
    if not callable(append):
        raise TypeError("XLSX worksheet must provide append().")
    append([_xlsx_literal_cell(sheet, value) for value in values])


def _validate_xlsx_worksheet_dimensions(
    dimensions: Sequence[tuple[str, int, int]],
) -> None:
    """Validate every planned worksheet against Excel's physical limits."""

    for title, rows, columns in dimensions:
        if rows > _EXCEL_MAX_ROWS:
            raise ValueError(
                f"XLSX worksheet {title!r} requires {rows:,} rows; "
                f"Excel permits at most {_EXCEL_MAX_ROWS:,}."
            )
        if columns > _EXCEL_MAX_COLUMNS:
            raise ValueError(
                f"XLSX worksheet {title!r} requires {columns:,} columns; "
                f"Excel permits at most {_EXCEL_MAX_COLUMNS:,}."
            )


def _planned_xlsx_worksheet_dimensions(
    blocks: Sequence[ResultDataBlock],
    *,
    lineage_row_count: int,
    result_entry_count: int,
) -> tuple[tuple[str, int, int], ...]:
    """Plan data and support worksheet dimensions without allocating them."""

    used: set[str] = set()
    dimensions: list[tuple[str, int, int]] = []
    for block in blocks:
        title = _worksheet_title(block.label, used)
        identity_column_count = 2 if block.kind is ResultLayerKind.EXPERIMENTAL else 4
        dimensions.append(
            (
                title,
                int(block.t.size) + 1,
                identity_column_count + len(block.series),
            )
        )
    if lineage_row_count:
        dimensions.append(
            (
                _worksheet_title(
                    "_Kindred Lineage",
                    used,
                ),
                lineage_row_count + 1,
                8,
            )
        )
    if result_entry_count:
        dimensions.append(
            (
                _worksheet_title(
                    "_Kindred Result Metadata",
                    used,
                ),
                result_entry_count + 1,
                2,
            )
        )
    return tuple(dimensions)


def _observation_provenance(
    source: ExperimentalLayerInput,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "dataset_id": source.dataset_id,
        "raw_observation_id": source.raw_observation_id,
        "dataset_name": source.dataset_name,
        "raw_column": source.raw_column,
    }
    if source.model_target is not None:
        payload["model_target"] = source.model_target
    if source.set_id is not None:
        payload["set_id"] = source.set_id
    return MappingProxyType(payload)


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


def _frame_topology(
    frame: object,
    *,
    size: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Read executor-issued topology carried by the transient frame.

    The protocol owner is required to populate all three fields for an admitted
    preview result.  Silently treating a missing topology as one continuous
    curve would reintroduce the old boundary-loss failure.
    """

    segment_ids = getattr(frame, "segment_ids", None)
    segment_keys = getattr(frame, "segment_keys", None)
    discontinuity_ids = getattr(frame, "discontinuity_ids", None)
    if segment_ids is None:
        raise ValueError(
            "Interactive Preview frame is missing executor-issued topology."
        )
    if segment_keys is None or discontinuity_ids is None:
        raise ValueError(
            "Interactive Preview topology must include segment IDs, keys, "
            "and discontinuity IDs."
        )
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)
    keys = tuple(_required_id(value, "interactive frame segment key") for value in segment_keys)
    if segments.size != size or discontinuities.size != size or len(keys) != size:
        raise ValueError("Interactive Preview topology must align with frame values.")
    return segments, keys, discontinuities


def _frame_connectors(
    frame: object,
    *,
    selected_x: AxisSeriesSelection,
    series_names: set[str],
) -> tuple[tuple[int, float, float, float, str], ...]:
    """Project optional worker-issued intervention endpoint facts."""

    if not (
        isinstance(selected_x, SimulationSeriesAxisSelection)
        and selected_x.series_name == "t"
    ):
        return ()
    if not hasattr(frame, "presentation_boundaries"):
        raise ValueError(
            "Interactive Preview frame is missing presentation boundaries."
        )
    boundaries = getattr(frame, "presentation_boundaries")
    entries: list[tuple[int, float, float, float, str]] = []
    for boundary_index, boundary in enumerate(tuple(boundaries or ())):
        pre_by_name = dict(getattr(boundary, "pre_state", ()))
        post_by_name = dict(getattr(boundary, "post_state", ()))
        for name, pre_value in pre_by_name.items():
            if name not in series_names:
                continue
            post_value = float(post_by_name[name])
            pre_value = float(pre_value)
            if pre_value == post_value:
                continue
            entries.append(
                (
                    boundary_index,
                    float(boundary.time),
                    pre_value,
                    post_value,
                    str(name),
                )
            )
    return tuple(entries)



def compose_interactive_display(
    frame: object,
    *,
    render_input: ResultRenderInput,
) -> InteractiveDisplayProjection:
    from kindred.gui.interactive_preview import InteractivePreviewFrame
    if not isinstance(frame, InteractivePreviewFrame):
        raise TypeError("frame must be an InteractivePreviewFrame.")
    if not isinstance(render_input, ResultRenderInput):
        raise TypeError("render_input must be a ResultRenderInput.")
    if frame.tab_id != render_input.tab_id:
        raise ValueError("interactive frame belongs to a different tab.")
    frame_series = {series.series_id: series for series in frame.series}
    selected_x = render_input.selected_x_series

    def empty() -> InteractiveDisplayProjection:
        return InteractiveDisplayProjection(
            tab_id=frame.tab_id, set_id=frame.set_id, generation=frame.generation,
            x_label=render_input.x_label, y_label=render_input.y_label, curves=(),
            target_set_ids=frame.target_set_ids,
        )

    if isinstance(selected_x, ExperimentalAxisSelection):
        return empty()
    if selected_x.series_name == "t":
        x_values = np.asarray(frame.x_values, dtype=float)
    else:
        selected_series = frame_series.get(selected_x.series_name)
        if selected_series is None:
            return empty()
        x_values = np.asarray(selected_series.values, dtype=float)
    wanted_y = (
        tuple(name for name in frame_series if name != selected_x.series_name)
        if render_input.selected_y_uses_default_all
        else tuple(
            selection.series_name
            for selection in render_input.selected_y_series
            if isinstance(selection, SimulationSeriesAxisSelection)
            and selection.series_name != selected_x.series_name
        )
    )
    base_block = next(
        (
            block for block in render_input.blocks
            if block.block_id == frame.set_id
            and block.kind in {ResultLayerKind.EXPLICIT, ResultLayerKind.PREVIEW}
            and block.visible
        ),
        None,
    )
    if base_block is None:
        return empty()
    topology = _frame_topology(frame, size=x_values.size)
    connectors = _frame_connectors(
        frame, selected_x=selected_x, series_names=set(frame_series)
    )
    base_colors = dict(base_block.color_by_series)
    plotted_series = {name for name, _values in base_block.series}
    curves = tuple(
        InteractiveDisplayCurve(
            key=("simulation", "foreground", frame.set_id, series_id),
            label=f"{base_block.label}: {series.label}",
            x=x_values,
            y=np.asarray(series.values, dtype=float),
            color=series.color or base_colors.get(series_id) or "#3b82f6",
            segment_ids=topology[0], segment_keys=topology[1],
            discontinuity_ids=topology[2],
            intervention_connectors=tuple(
                (item[0], item[1], item[2], item[3])
                for item in connectors if item[4] == series_id
            ),
        )
        for series_id in wanted_y
        if series_id in plotted_series
        and (series := frame_series.get(series_id)) is not None
    )
    return InteractiveDisplayProjection(
        tab_id=frame.tab_id, set_id=frame.set_id, generation=frame.generation,
        x_label=render_input.x_label, y_label=render_input.y_label,
        curves=curves, target_set_ids=frame.target_set_ids,
    )

def compose_interactive_display_batch(
    frames: Sequence[object],
    *,
    render_input: ResultRenderInput,
) -> InteractiveDisplayProjection:
    captured = tuple(frames)
    if not captured:
        raise ValueError("interactive display batch requires at least one frame.")
    composed = tuple(
        compose_interactive_display(
            frame, render_input=render_input
        )
        for frame in captured
    )
    first = composed[0]
    if any(
        item.target_set_ids != first.target_set_ids
        or item.generation != first.generation
        or item.tab_id != first.tab_id
        for item in composed[1:]
    ):
        raise ValueError("interactive display batch frames must share generation and roster.")
    return InteractiveDisplayProjection(
        tab_id=composed[-1].tab_id, set_id=composed[-1].set_id,
        generation=first.generation, x_label=composed[-1].x_label,
        y_label=composed[-1].y_label,
        curves=tuple(curve for item in composed for curve in item.curves),
        target_set_ids=first.target_set_ids,
    )

def compose_result_input(
    *,
    tab_id: str,
    current_set_id: str | None,
    visible_sets: Sequence[VisibleResultSet],
    repository: ResultRepository,
    reference_lines_visible: bool,
    intervention_annotations_visible: bool,
    mechanism_species: Sequence[str],
    active_overlay_layers: Sequence[ExperimentalLayerInput] = (),
    color_by_axis_selection: Mapping[AxisSeriesSelection, str] = MappingProxyType({}),
    experimental_point_size: float = 6.0,
    experimental_point_opacity: float = 0.85,
    selected_x_series: AxisSeriesSelection = SimulationSeriesAxisSelection("t"),
    selected_y_series: Sequence[AxisSeriesSelection] = (),
    selected_y_uses_default_all: bool = True,
    preview_set_ids: Sequence[str] = (),
) -> ResultRenderInput:
    tab_id = _required_id(tab_id, "tab_id")
    if not isinstance(repository, ResultRepository):
        raise TypeError("repository must be a ResultRepository.")
    if repository.tab_id != tab_id:
        raise ValueError("repository does not belong to requested tab.")
    mechanism_species_set = {
        _required_id(name, "mechanism species") for name in mechanism_species
    }
    current_id = str(current_set_id or "")
    preview_ids = {_required_id(value, "preview set_id") for value in preview_set_ids}
    colors = {
        _axis_selection(selection, field="configured color axis"):
        _required_id(color, "configured axis color")
        for selection, color in color_by_axis_selection.items()
    }
    selected_x = _axis_selection(selected_x_series, field="selected_x_series")
    selected_y = tuple(_axis_selection(item, field="selected_y_series") for item in selected_y_series)
    if len(selected_y) != len(set(selected_y)):
        raise ValueError("selected_y_series must be unique.")
    point_size = float(experimental_point_size)
    point_opacity = float(experimental_point_opacity)
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("experimental_point_size must be positive.")
    if not np.isfinite(point_opacity) or not 0 <= point_opacity <= 1:
        raise ValueError("experimental_point_opacity must be between zero and one.")
    active = tuple(active_overlay_layers)
    if any(not isinstance(item, ExperimentalLayerInput) for item in active):
        raise TypeError("active_overlay_layers must contain ExperimentalLayerInput values.")
    experimental_by_axis = {
        ExperimentalAxisSelection(item.dataset_id, item.raw_observation_id): item
        for item in active
    }
    if len(experimental_by_axis) != len(active):
        raise ValueError("active overlay layers duplicate dataset/raw-observation identity.")
    selected_model_y = {
        item.series_name for item in selected_y
        if isinstance(item, SimulationSeriesAxisSelection)
    }
    mapped_overlay_locks = tuple(
        (
            ExperimentalAxisSelection(item.dataset_id, item.raw_observation_id),
            str(item.model_target),
        )
        for item in active
        if item.model_target is not None
        and (bool(selected_y_uses_default_all) or item.model_target in selected_model_y)
        and ExperimentalAxisSelection(item.dataset_id, item.raw_observation_id) not in selected_y
    )
    effective_y = tuple(
        dict.fromkeys(
            (
                *selected_y,
                *(axis for axis, _model_target in mapped_overlay_locks),
            )
        )
    )
    wanted_y = None if bool(selected_y_uses_default_all) else set(effective_y)
    excluded_simulation_y: set[AxisSeriesSelection] = set()
    excluded_experimental_y: set[AxisSeriesSelection] = set()
    selected_x_source: ExperimentalLayerInput | None = None
    if isinstance(selected_x, SimulationSeriesAxisSelection):
        excluded_simulation_y.add(selected_x)
        excluded_experimental_y.update(
            ExperimentalAxisSelection(item.dataset_id, item.raw_observation_id)
            for item in active if item.model_target == selected_x.series_name
        )
    elif isinstance(selected_x, ExperimentalAxisSelection):
        selected_x_source = experimental_by_axis.get(selected_x)
        excluded_experimental_y.add(selected_x)
        if selected_x_source is not None and selected_x_source.model_target:
            excluded_simulation_y.add(SimulationSeriesAxisSelection(selected_x_source.model_target))
    simulation_x = (
        SimulationSeriesAxisSelection(selected_x_source.model_target)
        if isinstance(selected_x, ExperimentalAxisSelection)
        and selected_x_source is not None
        and selected_x_source.model_target is not None
        else selected_x
    )

    blocks: list[ResultDataBlock] = []
    unavailable_labels: list[str] = []
    candidates: list[AxisSelectionCandidate] = []
    candidate_ids: set[AxisSeriesSelection] = set()

    def add_candidate(selection: AxisSeriesSelection, label: str) -> None:
        if selection not in candidate_ids:
            candidate_ids.add(selection)
            candidates.append(AxisSelectionCandidate(selection, label))

    add_candidate(SimulationSeriesAxisSelection("t"), "t")
    for selection in colors:
        if isinstance(selection, SimulationSeriesAxisSelection):
            add_candidate(selection, selection.series_name)
    statistics_names: tuple[str, ...] = ()
    annotations: list[Mapping[str, object]] = []
    warning_facts: list[tuple[str, tuple[str, ...]]] = []
    warning_ids: set[int] = set()
    seen_sets: set[str] = set()

    def add_warning(label: str, result: SimulationResultValue, visible: bool) -> None:
        if visible and result.completed_with_warnings and id(result) not in warning_ids:
            warning_ids.add(id(result))
            warning_facts.append((label, tuple(_result_warning_details(result))))

    for visible in visible_sets:
        if not isinstance(visible, VisibleResultSet):
            raise TypeError("visible_sets must contain VisibleResultSet values.")
        if visible.set_id in seen_sets:
            raise ValueError(f"visible set {visible.set_id!r} is duplicated.")
        seen_sets.add(visible.set_id)
        foreground = repository.current_result(visible.set_id)
        foreground_kind = ResultLayerKind.PREVIEW if visible.set_id in preview_ids else ResultLayerKind.EXPLICIT
        semantic: ResultDataBlock | None = None
        physical: ResultDataBlock | None = None
        if foreground is not None:
            semantic = _result_block(
                set_id=visible.set_id, label=visible.label, result=foreground,
                kind=foreground_kind, color_by_axis_selection=colors,
            )
            for name, selection in semantic.series_axes:
                add_candidate(selection, name)
            physical = _data_block(
                semantic, x_axis=simulation_x, y_axes=wanted_y,
                excluded_y_axes=excluded_simulation_y,
            )
            if physical is not None:
                blocks.append(physical)
            else:
                if _has_selected_y_series(
                    semantic,
                    y_axes=wanted_y,
                    excluded_y_axes=excluded_simulation_y,
                ):
                    unavailable_labels.append(semantic.label)
                if visible.set_id == current_id:
                    blocks.append(replace(semantic, visible=False))
            add_warning(semantic.label, foreground, True)
            annotations.extend(_annotations_from_execution(foreground))
        reference = repository.reference_result(visible.set_id)
        reference_semantic: ResultDataBlock | None = None
        if reference is not None:
            reference_semantic = _result_block(
                set_id=visible.set_id, label=visible.label, result=reference,
                kind=ResultLayerKind.REFERENCE,
                color_by_axis_selection=colors, visible=bool(reference_lines_visible),
            )
            for name, selection in reference_semantic.series_axes:
                add_candidate(selection, name)
            if reference_lines_visible:
                reference_physical = _data_block(
                    reference_semantic, x_axis=simulation_x, y_axes=wanted_y,
                    excluded_y_axes=excluded_simulation_y,
                )
                if reference_physical is not None:
                    blocks.append(reference_physical)
                else:
                    if _has_selected_y_series(
                        reference_semantic,
                        y_axes=wanted_y,
                        excluded_y_axes=excluded_simulation_y,
                    ):
                        unavailable_labels.append(reference_semantic.label)
                    if visible.set_id == current_id:
                        blocks.append(replace(reference_semantic, visible=False))
                add_warning(reference_semantic.label, reference, True)
            elif visible.set_id == current_id:
                blocks.append(reference_semantic)
            annotations.extend(_annotations_from_execution(reference))
        if visible.set_id == current_id:
            foreground_names = (
                _statistics_series_names(
                    semantic, y_axes=wanted_y, excluded_y_axes=excluded_simulation_y
                )
                if foreground is not None
                and foreground_kind is ResultLayerKind.EXPLICIT
                and semantic is not None
                else ()
            )
            if semantic is not None and foreground is not None:
                statistics_names = foreground_names
            elif reference is not None and reference_semantic is not None:
                statistics_names = _statistics_series_names(
                    reference_semantic, y_axes=wanted_y, excluded_y_axes=excluded_simulation_y
                )

    for experimental in active:
        y_axis = ExperimentalAxisSelection(experimental.dataset_id, experimental.raw_observation_id)
        label = f"{experimental.dataset_name}: {experimental.raw_column}"
        if experimental.model_target:
            label = f"{label} → {experimental.model_target}"
        add_candidate(y_axis, label)
        x_values = np.asarray(experimental.t, dtype=float)
        y_values = np.asarray(experimental.y, dtype=float)
        x_name = "t"
        x_axis: AxisSeriesSelection = SimulationSeriesAxisSelection("t")
        x_provenance: Mapping[str, object] = MappingProxyType({})
        source_rows = tuple(experimental.source_rows)
        unavailable = False
        if not _is_time_selection(selected_x):
            x_source = (
                next(
                    (
                        item for item in active
                        if isinstance(selected_x, SimulationSeriesAxisSelection)
                        and item.dataset_id == experimental.dataset_id
                        and item.model_target == selected_x.series_name
                    ),
                    None,
                )
                if isinstance(selected_x, SimulationSeriesAxisSelection)
                else selected_x_source
            )
            if x_source is None or x_source.dataset_id != experimental.dataset_id or y_axis in excluded_experimental_y:
                if y_axis in excluded_experimental_y:
                    continue
                unavailable = True
            else:
                paired = _same_source_row_values(x_source, experimental)
                if paired is None:
                    unavailable = True
                else:
                    x_values, y_values, source_rows = paired
                    x_name = x_source.raw_column
                    x_axis = selected_x
                    x_provenance = _observation_provenance(x_source)
        if unavailable:
            if wanted_y is None or y_axis in wanted_y:
                unavailable_labels.append(label)
            continue
        if wanted_y is not None and y_axis not in wanted_y:
            continue
        blocks.append(
            ResultDataBlock(
                block_id=experimental.layer_id, label=label,
                kind=ResultLayerKind.EXPERIMENTAL, t=x_values,
                series=((experimental.raw_column, y_values),),
                series_axes=((experimental.raw_column, y_axis),),
                x_axis=x_axis, result=None,
                segment_ids=None, segment_keys=None, discontinuity_ids=None,
                source_row=np.asarray(source_rows, dtype=np.int64),
                x_name=x_name,
                series_provenance=((experimental.raw_column, _observation_provenance(experimental)),),
                x_provenance=x_provenance,
                color_by_series=((experimental.raw_column, colors[y_axis]),),
                opacity=point_opacity, markers=True, point_size=point_size,
            )
        )

    unique_annotations: list[Mapping[str, object]] = []
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for annotation in annotations:
        signature = tuple(sorted((str(name), repr(value)) for name, value in annotation.items()))
        if signature not in signatures:
            signatures.add(signature)
            unique_annotations.append(annotation)
    if not _is_time_selection(selected_x):
        unique_annotations.clear()
    if selected_x not in candidate_ids:
        add_candidate(
            selected_x,
            selected_x.series_name if isinstance(selected_x, SimulationSeriesAxisSelection)
            else f"{selected_x.dataset_id}: {selected_x.raw_observation_id}",
        )
    selected_candidate = next(item for item in candidates if item.selection == selected_x)
    y_axes = tuple(selection for block in blocks for _name, selection in block.series_axes)

    def is_concentration(selection: AxisSeriesSelection) -> bool:
        if isinstance(selection, ExperimentalAxisSelection):
            item = experimental_by_axis.get(selection)
            return item is not None and item.model_target in mechanism_species_set
        return isinstance(selection, SimulationSeriesAxisSelection) and selection.series_name in mechanism_species_set

    return ResultRenderInput(
        tab_id=tab_id, current_set_id=current_id, blocks=tuple(blocks),
        statistics_series_names=statistics_names,
        selected_x_series=selected_x, selected_y_series=selected_y,
        mapped_overlay_locks=mapped_overlay_locks,
        selected_y_uses_default_all=bool(selected_y_uses_default_all),
        x_label="Time (s)" if _is_time_selection(selected_x) else selected_candidate.label,
        y_label="Concentration (M)" if y_axes and all(is_concentration(item) for item in y_axes) else "Value",
        axis_candidates=tuple(candidates),
        reference_lines_visible=bool(reference_lines_visible),
        intervention_annotations=tuple(unique_annotations),
        intervention_annotations_visible=bool(intervention_annotations_visible and _is_time_selection(selected_x)),
        unavailable_x_diagnostic=_selected_x_unavailable_diagnostic(selected_x, unavailable_labels),
        completion_warning_details=tuple(warning_facts),
    )

def validate_copy(
    render_input: ResultRenderInput,
) -> None:
    """Validate Copy against one call-local render input."""

    if not isinstance(render_input, ResultRenderInput):
        raise TypeError("render_input must be a ResultRenderInput.")
    if render_input.unavailable_x_diagnostic is not None:
        raise ValueError(render_input.unavailable_x_diagnostic)
    if not any(block.visible for block in render_input.blocks):
        raise ValueError("No visible result, Reference, or experimental data.")

def iter_blocks_tsv_chunks(
    blocks: Sequence[ResultDataBlock],
    *,
    rows_per_chunk: int = 2048,
) -> Iterator[str]:
    block_values = tuple(blocks)
    columns: list[tuple[str, Sequence[object], str]] = []
    for block in block_values:
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
        if block.kind is ResultLayerKind.EXPERIMENTAL:
            if block.source_row is None:
                raise ValueError("experimental Copy block lacks source_row.")
            columns.append(
                (
                    encode_clipboard_text(f"{block.label} — source_row"),
                    block.source_row,
                    "int",
                )
            )
        else:
            if (
                block.segment_ids is None
                or block.segment_keys is None
                or block.discontinuity_ids is None
            ):
                raise ValueError("simulation Copy block lacks executor identity.")
            columns.extend(
                (
                    (
                        encode_clipboard_text(f"{block.label} — segment_id"),
                        block.segment_ids,
                        "int",
                    ),
                    (
                        encode_clipboard_text(f"{block.label} — segment_key"),
                        block.segment_keys,
                        "text",
                    ),
                    (
                        encode_clipboard_text(f"{block.label} — discontinuity_id"),
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
    result_entries = _unique_result_entries(block_values)
    cell_count = (row_count + 1) * len(columns) + 2 * len(result_entries)
    if cell_count > _MAX_COPY_CELLS:
        raise ValueError(
            "Copy payload exceeds the bounded full-resolution cell limit "
            f"({_MAX_COPY_CELLS:,})."
        )
    chunk_size = int(rows_per_chunk)
    if chunk_size < 1:
        raise ValueError("rows_per_chunk must be positive.")
    total_characters = 0
    for entry in result_entries:
        metadata_line = encode_tsv_row(
            (
                "# Kindred canonical result",
                encode_clipboard_text(
                        _canonical_json(_result_provenance_payload(entry))
                ),
            )
        )
        total_characters += len(metadata_line)
        if total_characters > _MAX_COPY_CHARACTERS:
            raise ValueError(
                "Copy payload exceeds the bounded output-size limit "
                f"({_MAX_COPY_CHARACTERS // (1024 * 1024)} MiB)."
            )
        yield metadata_line
    header = encode_tsv_row(tuple(header for header, _values, _kind in columns))
    total_characters += len(header)
    if total_characters > _MAX_COPY_CHARACTERS:
        raise ValueError(
            "Copy payload exceeds the bounded output-size limit "
            f"({_MAX_COPY_CHARACTERS // (1024 * 1024)} MiB)."
        )
    yield header
    for start in range(0, row_count, chunk_size):
        output = io.StringIO()
        for index in range(start, min(start + chunk_size, row_count)):
            output.write(
                encode_tsv_row(
                    tuple(
                        (
                            f"{float(values[index]):.17g}"
                            if kind == "float"
                            else (
                                str(int(values[index]))
                                if kind == "int"
                                else encode_clipboard_text(str(values[index]))
                            )
                        )
                        if index < len(values)
                        else ""
                        for _header, values, kind in columns
                    )
                )
            )
        chunk = output.getvalue()
        total_characters += len(chunk)
        if total_characters > _MAX_COPY_CHARACTERS:
            raise ValueError(
                "Copy payload exceeds the bounded output-size limit "
                f"({_MAX_COPY_CHARACTERS // (1024 * 1024)} MiB)."
            )
        yield chunk

def validate_xlsx_export(
    blocks: Sequence[ResultDataBlock],
) -> None:
    """Validate the immutable visible block roster before writing."""

    block_values = tuple(blocks)
    if not block_values:
        raise ValueError("No visible result, Reference, or experimental data.")
    if any(not isinstance(block, ResultDataBlock) for block in block_values):
        raise TypeError(
            "XLSX export blocks must contain only ResultDataBlock values."
        )
    if any(not block.visible for block in block_values):
        raise ValueError("XLSX export blocks must all be visible.")
    lineage_row_count = sum(
        int(bool(block.x_provenance))
        + sum(bool(provenance) for _, provenance in block.series_provenance)
        for block in block_values
    )
    result_entry_count = len(_unique_result_entries(block_values))
    dimensions = _planned_xlsx_worksheet_dimensions(
        block_values,
        lineage_row_count=lineage_row_count,
        result_entry_count=result_entry_count,
    )
    _validate_xlsx_worksheet_dimensions(dimensions)

def write_xlsx(
    blocks: Sequence[ResultDataBlock],
    destination: BinaryIO,
    *,
    cancellation_check: Callable[[], None] | None = None,
) -> XlsxExportCompletion:
    """Write full-resolution worksheets directly to the destination."""

    block_values = tuple(blocks)
    validate_xlsx_export(block_values)
    output = _XlsxDestination(
        destination,
        cancellation_check=cancellation_check,
    )
    workbook = Workbook(write_only=True)
    used: set[str] = set()
    lineage_rows: list[tuple[str, str, Mapping[str, object]]] = []
    metadata_rows: list[tuple[str, SimulationResultValue]] = []
    metadata_ids: set[int] = set()
    data_rows = 0
    archive: ZipFile | None = None
    original_failure: BaseException | None = None
    try:
        for block in block_values:
            if cancellation_check is not None:
                cancellation_check()
            title = _worksheet_title(block.label, used)
            sheet = workbook.create_sheet(title=title)
            if block.kind is ResultLayerKind.EXPERIMENTAL:
                if block.source_row is None:
                    raise ValueError("experimental XLSX block lacks source_row.")
                identity_headers = ["source_row"]
            else:
                if (
                    block.segment_ids is None
                    or block.segment_keys is None
                    or block.discontinuity_ids is None
                ):
                    raise ValueError(
                        "simulation XLSX block lacks executor identity."
                    )
                identity_headers = [
                    "segment_id",
                    "segment_key",
                    "discontinuity_id",
                ]
            _append_xlsx_row(
                sheet,
                (
                    block.x_name,
                    *identity_headers,
                    *(name for name, _values in block.series),
                ),
            )
            for index in range(block.t.size):
                if cancellation_check is not None and index % 256 == 0:
                    cancellation_check()
                if block.kind is ResultLayerKind.EXPERIMENTAL:
                    identity_values: list[object] = [int(block.source_row[index])]
                else:
                    identity_values = [
                        int(block.segment_ids[index]),
                        block.segment_keys[index],
                        int(block.discontinuity_ids[index]),
                    ]
                _append_xlsx_row(
                    sheet,
                    (
                        float(block.t[index]),
                        *identity_values,
                        *(float(values[index]) for _name, values in block.series),
                    ),
                )
                data_rows += 1
            if block.x_provenance:
                lineage_rows.append((title, block.x_name, block.x_provenance))
            lineage_rows.extend(
                (title, name, provenance)
                for name, provenance in block.series_provenance
                if provenance
            )
            entry = block.result
            if entry is not None and id(entry) not in metadata_ids:
                metadata_ids.add(id(entry))
                metadata_rows.append((title, entry))
        if lineage_rows:
            if cancellation_check is not None:
                cancellation_check()
            lineage_sheet = workbook.create_sheet(
                    title=_worksheet_title(
                    "_Kindred Lineage",
                    used,
                )
            )
            _append_xlsx_row(
                lineage_sheet,
                (
                    "worksheet",
                    "series",
                    "dataset_id",
                    "raw_observation_id",
                    "dataset_name",
                    "raw_column",
                    "model_target",
                    "set_id",
                ),
            )
            for title, name, provenance in lineage_rows:
                if cancellation_check is not None:
                    cancellation_check()
                _append_xlsx_row(
                    lineage_sheet,
                    (
                        title,
                        name,
                        str(provenance.get("dataset_id") or ""),
                        str(provenance.get("raw_observation_id") or ""),
                        str(provenance.get("dataset_name") or ""),
                        str(provenance.get("raw_column") or ""),
                        str(provenance.get("model_target") or ""),
                        str(provenance.get("set_id") or ""),
                    ),
                )
            lineage_sheet.sheet_state = "hidden"
        if metadata_rows:
            if cancellation_check is not None:
                cancellation_check()
            metadata_sheet = workbook.create_sheet(
                    title=_worksheet_title(
                    "_Kindred Result Metadata",
                    used,
                )
            )
            _append_xlsx_row(
                metadata_sheet,
                ("worksheet", "repository_entry"),
            )
            for title, entry in metadata_rows:
                if cancellation_check is not None:
                    cancellation_check()
                _append_xlsx_row(
                    metadata_sheet,
                    (
                        title,
                        _canonical_json(_result_provenance_payload(entry)),
                    ),
                )
            metadata_sheet.sheet_state = "hidden"
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
        cleanup_failures.extend(_cleanup_write_only_workbook(workbook))
        if original_failure is None and cleanup_failures:
            raise cleanup_failures[0]

def _worksheet_title(label: str, used: set[str]) -> str:
    invalid = set("[]:*?/\\")
    base = "".join(
        "_" if char in invalid or ord(char) < 32 else char for char in str(label)
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

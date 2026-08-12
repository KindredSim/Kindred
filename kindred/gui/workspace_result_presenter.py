"""Stateless composition and export for one retained workspace result page."""

from __future__ import annotations

import io
import csv
import json
import unicodedata
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence, TypeAlias
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
    ExplicitResultEntry,
    PreviewResultEntry,
    ResultEntry,
    ResultRepository,
    SimulationResultValue,
)
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
    "SimulationResultLayerIdentity",
    "VisibleResultSet",
    "WorkspaceLayerIdentity",
    "WorkspacePlotLayer",
    "WorkspaceResultPresenter",
    "WorkspaceResultProjection",
    "XlsxExportCompletion",
    "encode_clipboard_text",
    "encode_tsv_row",
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
    target_set_ids: tuple[str, ...] = ()
    batch_id: str = ""

    def __post_init__(self) -> None:
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("interactive display generation must be nonnegative.")
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
            target_set_ids = (set_id,)
        if len(set(target_set_ids)) != len(target_set_ids):
            raise ValueError("interactive display target roster must be unique.")
        if set_id not in target_set_ids:
            raise ValueError("set_id must belong to the interactive target roster.")
        batch_id = str(self.batch_id or "").strip()
        if not batch_id:
            batch_id = f"{self.tab_id}:{generation}:{','.join(target_set_ids)}"
        object.__setattr__(self, "tab_id", _required_id(self.tab_id, "tab_id"))
        object.__setattr__(self, "set_id", set_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "x_label", _required_id(self.x_label, "x_label"))
        object.__setattr__(self, "y_label", _required_id(self.y_label, "y_label"))
        object.__setattr__(self, "curves", curves)
        object.__setattr__(self, "target_set_ids", target_set_ids)
        object.__setattr__(self, "batch_id", _required_id(batch_id, "batch_id"))


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
class SimulationResultLayerIdentity:
    """Tagged identity for one semantic result role and initial set."""

    kind: ResultLayerKind
    set_id: str

    def __post_init__(self) -> None:
        if self.kind not in {
            ResultLayerKind.EXPLICIT,
            ResultLayerKind.PREVIEW,
            ResultLayerKind.REFERENCE,
        }:
            raise ValueError(
                "Simulation result layer identity requires an explicit, "
                "Preview, or Reference kind."
            )
        object.__setattr__(self, "set_id", _required_id(self.set_id, "set_id"))


WorkspaceLayerIdentity: TypeAlias = (
    SimulationResultLayerIdentity | ExperimentalLayerIdentity
)


def _result_warning_details(entry: ResultEntry | None) -> tuple[str, ...]:
    if not isinstance(entry, (ExplicitResultEntry, PreviewResultEntry)):
        raise TypeError("warning details require a repository result entry.")
    provenance = entry.result.provenance
    details = [f"Preparation: {warning}" for warning in provenance.preparation_warnings]
    if provenance.failed_algebra_outputs:
        details.append(
            "Algebra failed: " + ", ".join(provenance.failed_algebra_outputs)
        )
    if provenance.omitted_algebra_outputs:
        details.append(
            "Algebra omitted: " + ", ".join(provenance.omitted_algebra_outputs)
        )
    return tuple(details)


def _result_entry_payload(entry: ResultEntry) -> Mapping[str, object]:
    """Serialize the exact retained entry key and result-owned provenance."""

    if not isinstance(entry, (ExplicitResultEntry, PreviewResultEntry)):
        raise TypeError("result metadata requires a repository result entry.")
    provenance = entry.result.provenance
    return MappingProxyType(
        {
            "entry_type": (
                "preview" if isinstance(entry, PreviewResultEntry) else "explicit"
            ),
            "key": entry.key,
            "provenance": MappingProxyType(
                {
                    "solver": provenance.solver,
                    "preparation_warnings": provenance.preparation_warnings,
                    "algebra": MappingProxyType(
                        {
                            "requested": provenance.requested_algebra_outputs,
                            "succeeded": provenance.succeeded_algebra_outputs,
                            "failed": provenance.failed_algebra_outputs,
                            "omitted": provenance.omitted_algebra_outputs,
                            "errors": provenance.algebra_errors,
                        }
                    ),
                }
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class WorkspacePlotLayer:
    layer_id: WorkspaceLayerIdentity
    label: str
    kind: ResultLayerKind
    t: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    series_axes: tuple[tuple[str, AxisSeriesSelection], ...]
    result_entry: ResultEntry | None
    segment_ids: np.ndarray | None
    segment_keys: tuple[str, ...] | None
    discontinuity_ids: np.ndarray | None
    source_row: np.ndarray | None
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...] = ()
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
        layer_id = self.layer_id
        if not isinstance(
            layer_id,
            (SimulationResultLayerIdentity, ExperimentalLayerIdentity),
        ):
            raise TypeError("layer_id must be a tagged workspace layer identity.")
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if not isinstance(layer_id, ExperimentalLayerIdentity):
                raise TypeError(
                    "experimental plot layers require an experimental layer identity."
                )
        elif (
            not isinstance(layer_id, SimulationResultLayerIdentity)
            or layer_id.kind is not self.kind
        ):
            raise ValueError(
                "simulation plot layer identity must match its semantic kind."
            )
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
        series_axes = tuple(
            (
                _required_id(raw_name, "plot axis series name"),
                _axis_selection(selection, field="plot series axis"),
            )
            for raw_name, selection in self.series_axes
        )
        if tuple(name for name, _ in series_axes) != tuple(name for name, _ in series):
            raise ValueError("plot series axes must align exactly with plot series.")
        axis_identities = tuple(selection for _, selection in series_axes)
        if len(axis_identities) != len(set(axis_identities)):
            raise ValueError("plot series axis identities must be unique.")
        result_entry = self.result_entry
        presentation_boundaries = tuple(self.presentation_boundaries)
        if any(
            not isinstance(boundary, SimulationPresentationBoundary)
            for boundary in presentation_boundaries
        ):
            raise TypeError(
                "presentation_boundaries must contain only "
                "SimulationPresentationBoundary values."
            )
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if result_entry is not None:
                raise ValueError(
                    "experimental plot layers prohibit repository entries."
                )
            if any(
                value is not None
                for value in (
                    self.segment_ids,
                    self.segment_keys,
                    self.discontinuity_ids,
                )
            ):
                raise ValueError("experimental plot layers prohibit solver identity.")
            source_row = _owned_source_rows(self.source_row, size=time.size)
            segment_ids = None
            segment_keys = None
            discontinuity_ids = None
            if presentation_boundaries:
                raise ValueError(
                    "experimental plot layers prohibit presentation boundaries."
                )
        else:
            if not isinstance(
                result_entry,
                (ExplicitResultEntry, PreviewResultEntry),
            ):
                raise TypeError("simulation plot layers require a repository entry.")
            if (
                self.kind is ResultLayerKind.PREVIEW
                and not isinstance(result_entry, PreviewResultEntry)
            ) or (
                self.kind
                in {
                    ResultLayerKind.EXPLICIT,
                    ResultLayerKind.REFERENCE,
                }
                and not isinstance(result_entry, ExplicitResultEntry)
            ):
                raise TypeError(
                    "plot layer kind conflicts with its repository entry type."
                )
            assert isinstance(layer_id, SimulationResultLayerIdentity)
            if result_entry.key.set_id != layer_id.set_id:
                raise ValueError(
                    "plot layer source must match its repository entry key."
                )
            entry_result = result_entry.result
            if presentation_boundaries is not entry_result.presentation_boundaries:
                raise ValueError(
                    "simulation plot layers must retain their repository "
                    "entry's exact presentation boundaries."
                )
            if (
                time is not entry_result.t
                or tuple(name for name, _values in series) != tuple(entry_result.series)
                or any(
                    values is not entry_result.series[name] for name, values in series
                )
            ):
                raise ValueError(
                    "simulation plot layers must retain their repository "
                    "entry's exact result arrays."
                )
            if self.source_row is not None:
                raise ValueError(
                    "simulation plot layers prohibit observation source_row."
                )
            if (
                self.segment_ids is None
                or self.segment_keys is None
                or self.discontinuity_ids is None
            ):
                raise ValueError(
                    "simulation plot layers require complete executor identity."
                )
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
            if (
                segment_ids is not entry_result.segment_ids
                or segment_keys != entry_result.segment_keys
                or discontinuity_ids is not entry_result.discontinuity_ids
            ):
                raise ValueError(
                    "simulation plot layers must retain their repository "
                    "entry's exact executor identity."
                )
            source_row = None
        opacity = float(self.opacity)
        point_size = float(self.point_size)
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between zero and one.")
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("point_size must be positive.")
        colors = tuple(
            (
                _required_id(name, "color series name"),
                _required_id(color, "plot series color"),
            )
            for name, color in self.color_by_series
        )
        if any(name not in names for name, _color in colors):
            raise ValueError("plot series colors must reference plot series.")
        if len(colors) != len({name for name, _ in colors}):
            raise ValueError("plot series colors must be unique by series.")
        if {name for name, _color in colors} != names:
            raise ValueError("plot series colors must cover every plot series.")
        provenance: list[tuple[str, Mapping[str, object]]] = []
        provenance_names: set[str] = set()
        for raw_name, raw_payload in self.series_provenance:
            name = _required_id(raw_name, "series provenance name")
            if name not in names:
                raise ValueError("series provenance must reference a plot series.")
            if name in provenance_names:
                raise ValueError(f"plot series provenance duplicates {name!r}.")
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
        provenance_by_name = dict(provenance)
        for name, selection in series_axes:
            if self.kind is ResultLayerKind.EXPERIMENTAL:
                if not isinstance(selection, ExperimentalAxisSelection):
                    raise TypeError(
                        "experimental plot series require experimental axis identity."
                    )
                payload = provenance_by_name.get(name)
                if (
                    payload is None
                    or payload.get("dataset_id") != selection.dataset_id
                    or payload.get("raw_observation_id") != selection.raw_observation_id
                ):
                    raise ValueError(
                        "experimental plot identity must match its raw-ID provenance."
                    )
            elif not isinstance(selection, SimulationSeriesAxisSelection):
                raise TypeError(
                    "simulation plot series require simulation axis identity."
                )
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "label", _required_id(self.label, "label"))
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", tuple(series))
        object.__setattr__(self, "series_axes", series_axes)
        object.__setattr__(self, "result_entry", result_entry)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "source_row", source_row)
        object.__setattr__(self, "series_provenance", tuple(provenance))
        object.__setattr__(self, "color_by_series", colors)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "dashed", bool(self.dashed))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "markers", bool(self.markers))
        object.__setattr__(self, "point_size", point_size)
        object.__setattr__(
            self,
            "presentation_boundaries",
            presentation_boundaries,
        )


@dataclass(frozen=True, slots=True)
class ResultDataBlock:
    block_id: WorkspaceLayerIdentity
    label: str
    kind: ResultLayerKind
    t: np.ndarray
    series: tuple[tuple[str, np.ndarray], ...]
    series_axes: tuple[tuple[str, AxisSeriesSelection], ...]
    x_axis: AxisSeriesSelection
    result_entry: ResultEntry | None
    segment_ids: np.ndarray | None
    segment_keys: tuple[str, ...] | None
    discontinuity_ids: np.ndarray | None
    source_row: np.ndarray | None
    x_name: str = "t"
    series_provenance: tuple[tuple[str, Mapping[str, object]], ...] = ()
    x_provenance: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ResultLayerKind):
            raise TypeError("kind must be a ResultLayerKind.")
        block_id = self.block_id
        if not isinstance(
            block_id,
            (SimulationResultLayerIdentity, ExperimentalLayerIdentity),
        ):
            raise TypeError("block_id must be a tagged workspace layer identity.")
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if not isinstance(block_id, ExperimentalLayerIdentity):
                raise TypeError(
                    "experimental data blocks require an experimental layer identity."
                )
        elif (
            not isinstance(block_id, SimulationResultLayerIdentity)
            or block_id.kind is not self.kind
        ):
            raise ValueError(
                "simulation data block identity must match its semantic kind."
            )
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
            (
                _required_id(raw_name, "data axis series name"),
                _axis_selection(selection, field="data series axis"),
            )
            for raw_name, selection in self.series_axes
        )
        if tuple(name for name, _ in series_axes) != tuple(
            name for name, _ in normalized
        ):
            raise ValueError("data series axes must align exactly with data series.")
        axis_identities = tuple(selection for _, selection in series_axes)
        if len(axis_identities) != len(set(axis_identities)):
            raise ValueError("data series axis identities must be unique.")
        x_axis = _axis_selection(self.x_axis, field="data X axis")
        result_entry = self.result_entry
        if self.kind is ResultLayerKind.EXPERIMENTAL:
            if result_entry is not None:
                raise ValueError(
                    "experimental data blocks prohibit repository entries."
                )
            if any(
                value is not None
                for value in (
                    self.segment_ids,
                    self.segment_keys,
                    self.discontinuity_ids,
                )
            ):
                raise ValueError("experimental data blocks prohibit solver identity.")
            source_row = _owned_source_rows(self.source_row, size=time.size)
            segment_ids = None
            segment_keys = None
            discontinuity_ids = None
        else:
            if not isinstance(x_axis, SimulationSeriesAxisSelection):
                raise TypeError("simulation data blocks require a simulation X axis.")
            if not isinstance(
                result_entry,
                (ExplicitResultEntry, PreviewResultEntry),
            ):
                raise TypeError("simulation data blocks require a repository entry.")
            if (
                self.kind is ResultLayerKind.PREVIEW
                and not isinstance(result_entry, PreviewResultEntry)
            ) or (
                self.kind
                in {
                    ResultLayerKind.EXPLICIT,
                    ResultLayerKind.REFERENCE,
                }
                and not isinstance(result_entry, ExplicitResultEntry)
            ):
                raise TypeError(
                    "data block kind conflicts with its repository entry type."
                )
            if self.source_row is not None:
                raise ValueError(
                    "simulation data blocks prohibit observation source_row."
                )
            if (
                self.segment_ids is None
                or self.segment_keys is None
                or self.discontinuity_ids is None
            ):
                raise ValueError(
                    "simulation data blocks require complete executor identity."
                )
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
            source_row = None
        provenance: list[tuple[str, Mapping[str, object]]] = []
        provenance_names: set[str] = set()
        for raw_name, raw_payload in self.series_provenance:
            name = _required_id(raw_name, "data series provenance name")
            if name not in names:
                raise ValueError("data series provenance must reference a data series.")
            if name in provenance_names:
                raise ValueError(f"data series provenance duplicates {name!r}.")
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
        provenance_by_name = dict(provenance)
        for name, selection in series_axes:
            if self.kind is ResultLayerKind.EXPERIMENTAL:
                if not isinstance(selection, ExperimentalAxisSelection):
                    raise TypeError(
                        "experimental data series require experimental axis identity."
                    )
                payload = provenance_by_name.get(name)
                if (
                    payload is None
                    or payload.get("dataset_id") != selection.dataset_id
                    or payload.get("raw_observation_id") != selection.raw_observation_id
                ):
                    raise ValueError(
                        "experimental data identity must match its raw-ID provenance."
                    )
            elif not isinstance(selection, SimulationSeriesAxisSelection):
                raise TypeError(
                    "simulation data series require simulation axis identity."
                )
        if isinstance(x_axis, ExperimentalAxisSelection) and (
            self.x_provenance.get("dataset_id") != x_axis.dataset_id
            or self.x_provenance.get("raw_observation_id") != x_axis.raw_observation_id
        ):
            raise ValueError(
                "experimental X identity must match its raw-ID provenance."
            )
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "label", _required_id(self.label, "label"))
        object.__setattr__(self, "x_name", _required_id(self.x_name, "x_name"))
        object.__setattr__(self, "t", time)
        object.__setattr__(self, "series", tuple(normalized))
        object.__setattr__(self, "series_axes", series_axes)
        object.__setattr__(self, "x_axis", x_axis)
        object.__setattr__(self, "result_entry", result_entry)
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
        object.__setattr__(self, "source_row", source_row)
        object.__setattr__(self, "series_provenance", tuple(provenance))
        object.__setattr__(
            self,
            "x_provenance",
            MappingProxyType(
                {
                    str(key): (
                        tuple(value) if isinstance(value, (list, tuple)) else value
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
    primary_layer_id: SimulationResultLayerIdentity | None
    plot_layers: tuple[WorkspacePlotLayer, ...]
    parameters: tuple[tuple[str, float, str], ...]
    scalar_values: tuple[tuple[str, float], ...]
    primary_statistics: ResultDataBlock | None
    copy_all_blocks: tuple[ResultDataBlock, ...]
    selected_x_series: AxisSeriesSelection = SimulationSeriesAxisSelection("t")
    selected_y_series: tuple[AxisSeriesSelection, ...] = ()
    derived_overlay_y_series: tuple[ExperimentalAxisSelection, ...] = ()
    selected_y_uses_default_all: bool = True
    x_label: str = "Time (s)"
    y_label: str = "Value"
    axis_candidates: tuple[AxisSelectionCandidate, ...] = ()
    reference_lines_visible: bool = True
    intervention_annotations: tuple[Mapping[str, object], ...] = ()
    intervention_annotations_visible: bool = True
    unavailable_x_diagnostic: str | None = None

    def __post_init__(self) -> None:
        tab_id = _required_id(self.tab_id, "tab_id")
        layers = tuple(self.plot_layers)
        if any(not isinstance(layer, WorkspacePlotLayer) for layer in layers):
            raise TypeError("plot_layers must contain only WorkspacePlotLayer values.")
        if any(
            layer.result_entry is not None and layer.result_entry.key.tab_id != tab_id
            for layer in layers
        ):
            raise ValueError(
                "projected repository entries must belong to projection tab_id."
            )
        layer_ids = {layer.layer_id for layer in layers}
        if len(layer_ids) != len(layers):
            raise ValueError("plot layer IDs must be unique.")
        primary = self.primary_layer_id
        if primary is not None and not isinstance(
            primary,
            SimulationResultLayerIdentity,
        ):
            raise TypeError(
                "primary_layer_id must be a simulation result layer identity or None."
            )
        if primary is not None and primary not in layer_ids:
            raise ValueError("primary_layer_id must reference one plot layer.")
        current_set_id = str(self.current_set_id or "")
        if primary is not None:
            primary_layer = next(layer for layer in layers if layer.layer_id == primary)
            if (
                not current_set_id
                or primary.set_id != current_set_id
                or primary_layer.kind
                not in {
                    ResultLayerKind.EXPLICIT,
                    ResultLayerKind.PREVIEW,
                    ResultLayerKind.REFERENCE,
                }
            ):
                raise ValueError(
                    "primary must be an accepted explicit, Preview, or Reference "
                    "layer for current_set_id."
                )
        elif (
            self.parameters or self.scalar_values or self.primary_statistics is not None
        ):
            raise ValueError(
                "primary analysis truth requires an eligible primary layer."
            )
        parameters: list[tuple[str, float, str]] = []
        parameter_names: set[str] = set()
        for raw_name, raw_value, raw_unit in self.parameters:
            name = _required_id(raw_name, "projection parameter name")
            if name in parameter_names:
                raise ValueError(f"projection parameters contain duplicate {name!r}.")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"projection parameter {name!r} must be finite.")
            parameter_names.add(name)
            parameters.append((name, value, str(raw_unit or "")))
        scalar_values: list[tuple[str, float]] = []
        scalar_names: set[str] = set()
        for raw_name, raw_value in self.scalar_values:
            name = _required_id(raw_name, "projection scalar name")
            if name in scalar_names:
                raise ValueError(f"projection scalars contain duplicate {name!r}.")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"projection scalar {name!r} must be finite.")
            scalar_names.add(name)
            scalar_values.append((name, value))
        primary_statistics = self.primary_statistics
        if primary_statistics is not None and not isinstance(
            primary_statistics, ResultDataBlock
        ):
            raise TypeError("primary_statistics must be a ResultDataBlock or None.")
        if primary_statistics is not None:
            primary_layer = next(layer for layer in layers if layer.layer_id == primary)
            if (
                primary_statistics.block_id != primary
                or not _is_time_selection(primary_statistics.x_axis)
                or primary_statistics.x_name != "t"
                or not np.array_equal(
                    primary_statistics.t,
                    primary_layer.t,
                )
                or primary_statistics.result_entry is not primary_layer.result_entry
            ):
                raise ValueError(
                    "primary statistics must use the primary result's "
                    "canonical time and repository entry."
                )
        copy_blocks = tuple(self.copy_all_blocks)
        if any(not isinstance(block, ResultDataBlock) for block in copy_blocks):
            raise TypeError("copy_all_blocks must contain only ResultDataBlock values.")
        layer_kind_by_id = {layer.layer_id: layer.kind for layer in layers}
        for block in copy_blocks:
            if layer_kind_by_id.get(block.block_id) is not block.kind:
                raise ValueError(
                    "copy blocks must match one projected layer and its kind."
                )
            layer = next(
                candidate
                for candidate in layers
                if candidate.layer_id == block.block_id
            )
            if block.result_entry is not layer.result_entry:
                raise ValueError(
                    "copy block must retain the exact projected repository entry."
                )
            if not layer.visible:
                raise ValueError(
                    "copy blocks must not physically project a hidden result layer."
                )
        selected_x = _axis_selection(
            self.selected_x_series,
            field="selected_x_series",
        )
        selected_y = tuple(
            _axis_selection(selection, field="selected_y_series")
            for selection in self.selected_y_series
        )
        if len(selected_y) != len(set(selected_y)):
            raise ValueError("selected_y_series must be unique.")
        derived_overlay_y = tuple(
            selection
            for selection in self.derived_overlay_y_series
            if isinstance(selection, ExperimentalAxisSelection)
        )
        if len(derived_overlay_y) != len(self.derived_overlay_y_series):
            raise TypeError(
                "derived_overlay_y_series must contain only experimental "
                "axis selections."
            )
        if len(derived_overlay_y) != len(set(derived_overlay_y)):
            raise ValueError("derived_overlay_y_series must be unique.")
        if set(derived_overlay_y).intersection(selected_y):
            raise ValueError(
                "derived_overlay_y_series must be distinct from authored "
                "selected_y_series."
            )
        if type(self.x_label) is not str:
            raise TypeError("x_label must be an exact string.")
        if type(self.y_label) is not str:
            raise TypeError("y_label must be an exact string.")
        candidates = tuple(self.axis_candidates)
        if any(not isinstance(item, AxisSelectionCandidate) for item in candidates):
            raise TypeError(
                "axis_candidates must contain AxisSelectionCandidate values."
            )
        candidate_ids = tuple(item.selection for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("axis candidate identities must be unique.")
        if selected_x not in candidate_ids:
            raise ValueError(
                "selected_x_series must exist in the reconciled axis candidate roster."
            )
        if any(selection not in candidate_ids for selection in derived_overlay_y):
            raise ValueError(
                "derived_overlay_y_series must exist in the active axis "
                "candidate roster."
            )
        annotations = tuple(self.intervention_annotations)
        if any(not isinstance(item, Mapping) for item in annotations):
            raise TypeError("intervention_annotations must contain only mappings.")
        unavailable_x_diagnostic = self.unavailable_x_diagnostic
        if unavailable_x_diagnostic is not None:
            if type(unavailable_x_diagnostic) is not str:
                raise TypeError(
                    "unavailable_x_diagnostic must be an exact string or None."
                )
            if not unavailable_x_diagnostic.strip():
                raise ValueError(
                    "unavailable_x_diagnostic must be nonblank when present."
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
            selected_x,
        )
        object.__setattr__(
            self,
            "selected_y_series",
            selected_y,
        )
        object.__setattr__(
            self,
            "derived_overlay_y_series",
            derived_overlay_y,
        )
        object.__setattr__(
            self,
            "selected_y_uses_default_all",
            bool(self.selected_y_uses_default_all),
        )
        object.__setattr__(self, "x_label", self.x_label)
        object.__setattr__(self, "y_label", self.y_label)
        object.__setattr__(
            self,
            "axis_candidates",
            candidates,
        )
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
        object.__setattr__(
            self,
            "unavailable_x_diagnostic",
            unavailable_x_diagnostic,
        )

    @property
    def completion_warnings(self) -> tuple[WorkspacePlotLayer, ...]:
        """Visible result layers requiring a prominent completion warning."""

        seen: set[int] = set()
        warnings: list[WorkspacePlotLayer] = []
        visible_layer_ids = {block.block_id for block in self.copy_all_blocks}
        for layer in self.plot_layers:
            entry = layer.result_entry
            if (
                layer.layer_id not in visible_layer_ids
                or entry is None
                or not entry.result.provenance.completed_with_warnings
                or id(entry) in seen
            ):
                continue
            seen.add(id(entry))
            warnings.append(layer)
        return tuple(warnings)

    @property
    def completion_warning_tooltip(self) -> str:
        groups = []
        for layer in self.completion_warnings:
            details = "\n".join(
                f"  • {detail}"
                for detail in _result_warning_details(layer.result_entry)
            )
            groups.append(f"{layer.label}:\n{details}")
        return "\n\n".join(groups)


def _result_layer(
    *,
    set_id: str,
    label: str,
    entry: ResultEntry,
    kind: ResultLayerKind,
    color_by_axis_selection: Mapping[AxisSeriesSelection, str],
    visible: bool = True,
) -> WorkspacePlotLayer:
    if not isinstance(entry, (ExplicitResultEntry, PreviewResultEntry)):
        raise TypeError("result layer requires a repository result entry.")
    result = entry.result
    if kind is ResultLayerKind.PREVIEW:
        display_label = f"{label} [Preview]"
    elif kind is ResultLayerKind.REFERENCE:
        display_label = f"{label} [Reference]"
    else:
        display_label = label
    layer_id = SimulationResultLayerIdentity(kind, set_id)
    series_axes = tuple(
        (name, SimulationSeriesAxisSelection(name)) for name in result.series
    )
    return WorkspacePlotLayer(
        layer_id=layer_id,
        label=display_label,
        kind=kind,
        t=result.t,
        series=_series_tuple(result),
        series_axes=series_axes,
        result_entry=entry,
        segment_ids=result.segment_ids,
        segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids,
        source_row=None,
        color_by_series=tuple(
            (name, str(color_by_axis_selection[selection]))
            for name, selection in series_axes
        ),
        visible=visible,
        dashed=kind is ResultLayerKind.REFERENCE,
        opacity=0.42 if kind is ResultLayerKind.REFERENCE else 1.0,
        presentation_boundaries=result.presentation_boundaries,
    )


def _data_block(
    layer: WorkspacePlotLayer,
    *,
    x_axis: AxisSeriesSelection,
    y_axes: set[AxisSeriesSelection] | None = None,
    excluded_y_axes: set[AxisSeriesSelection] | None = None,
) -> ResultDataBlock | None:
    series_map = dict(layer.series)
    series_axis_map = dict(layer.series_axes)
    if _is_time_selection(x_axis):
        x_values = layer.t
        x_name = "t"
        x_provenance: Mapping[str, object] = MappingProxyType({})
    elif (
        isinstance(x_axis, SimulationSeriesAxisSelection)
        and layer.kind is not ResultLayerKind.EXPERIMENTAL
    ):
        x_name = next(
            (name for name, selection in layer.series_axes if selection == x_axis),
            "",
        )
        if not x_name:
            return None
        x_values = series_map[x_name]
        x_provenance = dict(layer.series_provenance).get(
            x_name,
            MappingProxyType({}),
        )
    else:
        return None
    selected = tuple(
        (name, values)
        for name, values in layer.series
        if series_axis_map[name] != x_axis
        and (excluded_y_axes is None or series_axis_map[name] not in excluded_y_axes)
        and (y_axes is None or series_axis_map[name] in y_axes)
    )
    if not selected:
        return None
    selected_names = {name for name, _ in selected}
    return ResultDataBlock(
        block_id=layer.layer_id,
        label=layer.label,
        kind=layer.kind,
        t=x_values,
        series=selected,
        series_axes=tuple(
            (name, selection)
            for name, selection in layer.series_axes
            if name in selected_names
        ),
        x_axis=x_axis,
        result_entry=layer.result_entry,
        segment_ids=layer.segment_ids,
        segment_keys=layer.segment_keys,
        discontinuity_ids=layer.discontinuity_ids,
        source_row=layer.source_row,
        x_name=x_name,
        series_provenance=tuple(
            (name, payload)
            for name, payload in layer.series_provenance
            if name in selected_names
        ),
        x_provenance=x_provenance,
    )


def _has_selected_y_series(
    layer: WorkspacePlotLayer,
    *,
    y_axes: set[AxisSeriesSelection] | None,
    excluded_y_axes: set[AxisSeriesSelection] | None,
) -> bool:
    """Whether a semantic layer would contribute at least one selected Y."""

    series_names = {name for name, _values in layer.series}
    return any(
        excluded_y_axes is None or selection not in excluded_y_axes
        for name, selection in layer.series_axes
        if name in series_names
        and (y_axes is None or selection in y_axes)
    )


def _statistics_block(
    layer: WorkspacePlotLayer,
    *,
    y_axes: set[AxisSeriesSelection] | None,
    excluded_y_axes: set[AxisSeriesSelection] | None = None,
) -> ResultDataBlock | None:
    """Build selected-Y statistics against canonical simulation time."""

    entry = layer.result_entry
    if not isinstance(entry, (ExplicitResultEntry, PreviewResultEntry)):
        raise TypeError("statistics require a repository result entry.")
    result = entry.result
    series_axis_map = dict(layer.series_axes)
    succeeded_algebra_outputs = set(result.provenance.succeeded_algebra_outputs)
    selected = tuple(
        (name, values)
        for name, values in result.series.items()
        if (
            name not in succeeded_algebra_outputs
            and (
                excluded_y_axes is None or series_axis_map[name] not in excluded_y_axes
            )
            and (y_axes is None or series_axis_map[name] in y_axes)
        )
    )
    if not selected:
        return None
    selected_names = {name for name, _ in selected}
    return ResultDataBlock(
        block_id=layer.layer_id,
        label=layer.label,
        kind=layer.kind,
        t=result.t,
        series=selected,
        series_axes=tuple(
            (name, selection)
            for name, selection in layer.series_axes
            if name in selected_names
        ),
        x_axis=SimulationSeriesAxisSelection("t"),
        result_entry=entry,
        segment_ids=result.segment_ids,
        segment_keys=result.segment_keys,
        discontinuity_ids=result.discontinuity_ids,
        source_row=None,
        x_name="t",
    )


def _selected_x_unavailable_diagnostic(
    selected_x: AxisSeriesSelection,
    unavailable_layers: Sequence[WorkspacePlotLayer],
) -> str | None:
    """Return one deterministic diagnostic for visible semantic layers lacking X."""

    if _is_time_selection(selected_x):
        return None
    unavailable = tuple(unavailable_layers)
    if not unavailable:
        return None
    labels = ", ".join(repr(layer.label) for layer in unavailable)
    selected_identity = (
        f"series {selected_x.series_name!r}"
        if isinstance(selected_x, SimulationSeriesAxisSelection)
        else (
            f"raw observation {selected_x.raw_observation_id!r} "
            f"in dataset {selected_x.dataset_id!r}"
        )
    )
    return (
        f"Selected X {selected_identity} is unavailable for "
        f"result layer(s) {labels}. The semantic result layer(s) are retained, "
        "but plot, Copy, and XLSX rows cannot be produced without X values."
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
            "model_species",
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
) -> tuple[ResultEntry, ...]:
    seen: set[int] = set()
    ordered: list[ResultEntry] = []
    for block in blocks:
        entry = block.result_entry
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
        title = WorkspaceResultPresenter._worksheet_title(block.label, used)
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
                WorkspaceResultPresenter._worksheet_title(
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
                WorkspaceResultPresenter._worksheet_title(
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
    if source.model_species is not None:
        payload["model_species"] = source.model_species
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


def _frame_batch_id(frame: object) -> str:
    batch_id = str(getattr(frame, "batch_id", "") or "").strip()
    if not batch_id:
        raise ValueError("Interactive Preview frame requires an explicit batch_id.")
    return _required_id(batch_id, "interactive frame batch_id")


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


class WorkspaceResultPresenter:
    """Pure composition plus the two Copy payloads and generated XLSX."""

    @staticmethod
    def compose_interactive_display(
        frame: object,
        *,
        projection: WorkspaceResultProjection,
    ) -> InteractiveDisplayProjection:
        """Replace one foreground visually while retaining completed authority.

        ``InteractivePreviewFrame`` stays owned by the execution protocol.  This
        method consumes it only to form a physical roster; the returned value has
        no repository entry, statistics, Copy blocks, or export route.
        """

        from kindred.gui.interactive_preview import InteractivePreviewFrame

        if not isinstance(frame, InteractivePreviewFrame):
            raise TypeError("frame must be an InteractivePreviewFrame.")
        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if frame.tab_id != projection.tab_id:
            raise ValueError("interactive frame belongs to a different tab.")

        frame_series = {series.series_id: series for series in frame.series}
        selected_x = projection.selected_x_series
        if isinstance(selected_x, ExperimentalAxisSelection):
            # A transient simulation has no raw-observation row identity.  Keep
            # the completed experimental-X scene visible and publish no overlay.
            return InteractiveDisplayProjection(
                tab_id=frame.tab_id,
                set_id=frame.set_id,
                generation=frame.generation,
                x_label=projection.x_label,
                y_label=projection.y_label,
                curves=(),
                target_set_ids=frame.target_set_ids,
                batch_id=_frame_batch_id(frame),
            )
        if selected_x.series_name == "t":
            x_values = np.asarray(frame.x_values, dtype=float)
        else:
            try:
                x_values = np.asarray(
                    frame_series[selected_x.series_name].values,
                    dtype=float,
                )
            except KeyError:
                return InteractiveDisplayProjection(
                    tab_id=frame.tab_id,
                    set_id=frame.set_id,
                    generation=frame.generation,
                    x_label=projection.x_label,
                    y_label=projection.y_label,
                    curves=(),
                    target_set_ids=frame.target_set_ids,
                    batch_id=_frame_batch_id(frame),
                )

        if projection.selected_y_uses_default_all:
            wanted_y = tuple(
                name for name in frame_series if name != selected_x.series_name
            )
        else:
            wanted_y = tuple(
                selection.series_name
                for selection in projection.selected_y_series
                if isinstance(selection, SimulationSeriesAxisSelection)
                and selection.series_name != selected_x.series_name
            )

        base_layer = next(
            (
                layer
                for layer in projection.plot_layers
                if isinstance(layer.layer_id, SimulationResultLayerIdentity)
                and layer.layer_id.set_id == frame.set_id
                and layer.kind in {ResultLayerKind.EXPLICIT, ResultLayerKind.PREVIEW}
            ),
            None,
        )
        if base_layer is None or not base_layer.visible:
            return InteractiveDisplayProjection(
                tab_id=frame.tab_id,
                set_id=frame.set_id,
                generation=frame.generation,
                x_label=projection.x_label,
                y_label=projection.y_label,
                curves=(),
                target_set_ids=frame.target_set_ids,
                batch_id=_frame_batch_id(frame),
            )
        base_colors = dict(base_layer.color_by_series) if base_layer is not None else {}
        base_label = base_layer.label if base_layer is not None else frame.set_id
        plotted_series = {name for name, _values in base_layer.series}
        topology = _frame_topology(frame, size=x_values.size)
        connectors = _frame_connectors(
            frame,
            selected_x=selected_x,
            series_names=set(frame_series),
        )
        curves: list[InteractiveDisplayCurve] = []
        for series_id in wanted_y:
            if series_id not in plotted_series:
                continue
            series = frame_series.get(series_id)
            if series is None:
                continue
            curves.append(
                InteractiveDisplayCurve(
                    key=(
                        "simulation",
                        "foreground",
                        frame.set_id,
                        series_id,
                    ),
                    label=f"{base_label}: {series.label}",
                    x=x_values,
                    y=np.asarray(series.values, dtype=float),
                    color=(
                        series.color
                        or base_colors.get(series_id)
                        or "#3b82f6"
                    ),
                    segment_ids=topology[0],
                    segment_keys=topology[1],
                    discontinuity_ids=topology[2],
                    intervention_connectors=tuple(
                        (item[0], item[1], item[2], item[3])
                        for item in connectors
                        if item[4] == series_id
                    ),
                )
            )
        return InteractiveDisplayProjection(
            tab_id=frame.tab_id,
            set_id=frame.set_id,
            generation=frame.generation,
            x_label=projection.x_label,
            y_label=projection.y_label,
            curves=tuple(curves),
            target_set_ids=frame.target_set_ids,
            batch_id=_frame_batch_id(frame),
        )

    @staticmethod
    def compose_interactive_display_batch(
        frames: Sequence[object],
        *,
        projection: WorkspaceResultProjection,
    ) -> InteractiveDisplayProjection:
        """Compose the latest transient frame for each target into one roster.

        The returned projection remains physical-only.  Completed curves come
        from the supplied workspace projection; only foreground curves for a
        target with a successful transient frame are replaced.  This lets a
        later frame update one target without restoring an earlier target's
        completed baseline.
        """

        captured = tuple(frames)
        if not captured:
            raise ValueError("interactive display batch requires at least one frame.")
        composed = tuple(
            WorkspaceResultPresenter.compose_interactive_display(
                frame,
                projection=projection,
            )
            for frame in captured
        )
        roster = composed[0].target_set_ids
        generation = composed[0].generation
        batch_id = composed[0].batch_id
        if any(
            item.target_set_ids != roster
            or item.generation != generation
            or item.batch_id != batch_id
            for item in composed[1:]
        ):
            raise ValueError(
                "interactive display batch frames must share generation and roster."
            )
        # This projection is an overlay only.  The completed renderer keeps all
        # unchanged explicit/reference/experimental items visible and owns the
        # legend, range, Copy, and XLSX scene.  Re-composing those blocks here
        # would duplicate large unchanged rosters at preview cadence.
        curves: list[InteractiveDisplayCurve] = []
        for item in composed:
            curves.extend(item.curves)

        return InteractiveDisplayProjection(
            tab_id=composed[-1].tab_id,
            set_id=composed[-1].set_id,
            generation=generation,
            x_label=composed[-1].x_label,
            y_label=composed[-1].y_label,
            curves=tuple(curves),
            target_set_ids=roster,
            batch_id=batch_id,
        )

    @staticmethod
    def compose(
        *,
        tab_id: str,
        current_set_id: str | None,
        visible_sets: Sequence[VisibleResultSet],
        repository: ResultRepository,
        reference_lines_visible: bool,
        intervention_annotations_visible: bool,
        active_overlay_layers: Sequence[ExperimentalLayerInput] = (),
        color_by_axis_selection: Mapping[
            AxisSeriesSelection,
            str,
        ] = MappingProxyType({}),
        experimental_point_size: float = 6.0,
        experimental_point_opacity: float = 0.85,
        selected_x_series: AxisSeriesSelection = SimulationSeriesAxisSelection("t"),
        selected_y_series: Sequence[AxisSeriesSelection] = (),
        selected_y_uses_default_all: bool = True,
    ) -> WorkspaceResultProjection:
        tab_id = _required_id(tab_id, "tab_id")
        if not isinstance(repository, ResultRepository):
            raise TypeError("repository must be a ResultRepository.")
        if repository.tab_id != tab_id:
            raise ValueError("repository does not belong to the requested tab.")
        current_id = str(current_set_id or "")
        colors = {
            _axis_selection(
                selection,
                field="configured color axis",
            ): _required_id(color, "configured axis color")
            for selection, color in color_by_axis_selection.items()
        }
        selected_x = _axis_selection(
            selected_x_series,
            field="selected_x_series",
        )
        selected_y_ordered = tuple(
            _axis_selection(selection, field="selected_y_series")
            for selection in selected_y_series
        )
        if len(selected_y_ordered) != len(set(selected_y_ordered)):
            raise ValueError("selected_y_series must be unique.")
        point_size = float(experimental_point_size)
        point_opacity = float(experimental_point_opacity)
        if not np.isfinite(point_size) or point_size <= 0.0:
            raise ValueError("experimental_point_size must be positive.")
        if not np.isfinite(point_opacity) or not 0.0 <= point_opacity <= 1.0:
            raise ValueError("experimental_point_opacity must be between zero and one.")
        active_overlay_inputs = tuple(active_overlay_layers)
        if any(
            not isinstance(candidate, ExperimentalLayerInput)
            for candidate in active_overlay_inputs
        ):
            raise TypeError(
                "active_overlay_layers must contain ExperimentalLayerInput values."
            )
        experimental_by_axis = {
            ExperimentalAxisSelection(
                candidate.dataset_id,
                candidate.raw_observation_id,
            ): candidate
            for candidate in active_overlay_inputs
        }
        if len(experimental_by_axis) != len(active_overlay_inputs):
            raise ValueError(
                "Active overlay layers duplicate a dataset/raw-observation identity."
            )
        selected_model_y = {
            selection.series_name
            for selection in selected_y_ordered
            if isinstance(selection, SimulationSeriesAxisSelection)
        }
        derived_overlay_y_values: list[ExperimentalAxisSelection] = []
        for candidate in active_overlay_inputs:
            if candidate.model_species is None or not (
                bool(selected_y_uses_default_all)
                or candidate.model_species in selected_model_y
            ):
                continue
            selection = ExperimentalAxisSelection(
                candidate.dataset_id,
                candidate.raw_observation_id,
            )
            if selection not in selected_y_ordered:
                derived_overlay_y_values.append(selection)
        derived_overlay_y = tuple(derived_overlay_y_values)
        effective_y_ordered = tuple(
            dict.fromkeys((*selected_y_ordered, *derived_overlay_y))
        )
        wanted_y = (
            None
            if bool(selected_y_uses_default_all)
            else set(effective_y_ordered)
        )
        excluded_simulation_y: set[AxisSeriesSelection] = set()
        excluded_experimental_y: set[AxisSeriesSelection] = set()
        selected_x_source: ExperimentalLayerInput | None = None
        if isinstance(selected_x, SimulationSeriesAxisSelection):
            excluded_simulation_y.add(selected_x)
            excluded_experimental_y.update(
                ExperimentalAxisSelection(
                    candidate.dataset_id,
                    candidate.raw_observation_id,
                )
                for candidate in active_overlay_inputs
                if candidate.model_species == selected_x.series_name
            )
        elif isinstance(selected_x, ExperimentalAxisSelection):
            selected_x_source = experimental_by_axis.get(selected_x)
            excluded_experimental_y.add(selected_x)
            if selected_x_source is not None and selected_x_source.model_species:
                excluded_simulation_y.add(
                    SimulationSeriesAxisSelection(selected_x_source.model_species)
                )
        simulation_x_axis: AxisSeriesSelection = selected_x
        if (
            isinstance(selected_x, ExperimentalAxisSelection)
            and selected_x_source is not None
            and selected_x_source.model_species is not None
        ):
            # Experimental X remains the selected/source-row identity for raw
            # observations, while simulated and Reference layers consume its
            # current mapped model-series identity.
            simulation_x_axis = SimulationSeriesAxisSelection(
                selected_x_source.model_species
            )

        layers: list[WorkspacePlotLayer] = []
        blocks: list[ResultDataBlock] = []
        axis_candidates: list[AxisSelectionCandidate] = []
        axis_candidate_ids: set[AxisSeriesSelection] = set()
        unavailable_layers: list[WorkspacePlotLayer] = []

        def add_axis_candidate(
            selection: AxisSeriesSelection,
            label: str,
        ) -> None:
            if selection in axis_candidate_ids:
                return
            axis_candidate_ids.add(selection)
            axis_candidates.append(AxisSelectionCandidate(selection, label))

        add_axis_candidate(SimulationSeriesAxisSelection("t"), "t")
        for selection in colors:
            if isinstance(selection, SimulationSeriesAxisSelection):
                add_axis_candidate(selection, selection.series_name)
        primary_layer_id: SimulationResultLayerIdentity | None = None
        primary_entry: ResultEntry | None = None
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
            foreground_entry = preview if preview is not None else explicit
            foreground_kind = (
                ResultLayerKind.PREVIEW
                if preview is not None
                else ResultLayerKind.EXPLICIT
            )
            if foreground_entry is not None:
                foreground = _result_layer(
                    set_id=visible.set_id,
                    label=visible.label,
                    entry=foreground_entry,
                    kind=foreground_kind,
                    color_by_axis_selection=colors,
                )
                layers.append(foreground)
                for name, selection in foreground.series_axes:
                    add_axis_candidate(selection, name)
                block = _data_block(
                    foreground,
                    x_axis=simulation_x_axis,
                    y_axes=wanted_y,
                    excluded_y_axes=excluded_simulation_y,
                )
                if block is not None:
                    blocks.append(block)
                elif (
                    simulation_x_axis not in dict(foreground.series_axes).values()
                    and _has_selected_y_series(
                        foreground,
                        y_axes=wanted_y,
                        excluded_y_axes=excluded_simulation_y,
                    )
                ):
                    unavailable_layers.append(foreground)
            else:
                foreground = None

            reference = repository.reference_entry(visible.set_id)
            reference_layer: WorkspacePlotLayer | None = None
            if reference is not None:
                reference_layer = _result_layer(
                    set_id=visible.set_id,
                    label=visible.label,
                    entry=reference,
                    kind=ResultLayerKind.REFERENCE,
                    color_by_axis_selection=colors,
                    visible=bool(reference_lines_visible),
                )
                layers.append(reference_layer)
                for name, selection in reference_layer.series_axes:
                    add_axis_candidate(selection, name)
                if bool(reference_lines_visible):
                    block = _data_block(
                        reference_layer,
                        x_axis=simulation_x_axis,
                        y_axes=wanted_y,
                        excluded_y_axes=excluded_simulation_y,
                    )
                    if block is not None:
                        blocks.append(block)
                    elif (
                        simulation_x_axis
                        not in dict(reference_layer.series_axes).values()
                        and _has_selected_y_series(
                            reference_layer,
                            y_axes=wanted_y,
                            excluded_y_axes=excluded_simulation_y,
                        )
                    ):
                        # Reference is semantic analysis truth even when its
                        # selected-X physical rows cannot be projected. Keep
                        # the layer visible and account for it in the single
                        # deterministic unavailable-X diagnostic.
                        unavailable_layers.append(reference_layer)
                    annotations.extend(_annotations_from_execution(reference.result))

            if foreground is not None:
                assert foreground.result_entry is not None
                annotations.extend(
                    _annotations_from_execution(foreground.result_entry.result)
                )
            if visible.set_id == current_id:
                foreground_block = (
                    None
                    if foreground is None
                    or isinstance(foreground.result_entry, PreviewResultEntry)
                    else _statistics_block(
                        foreground,
                        y_axes=wanted_y,
                        excluded_y_axes=excluded_simulation_y,
                    )
                )
                if (
                    foreground is not None
                    and not isinstance(
                        foreground.result_entry,
                        PreviewResultEntry,
                    )
                    and (
                        foreground_block is not None
                        or reference_layer is None
                    )
                ):
                    primary_layer_id = foreground.layer_id
                    primary_entry = foreground.result_entry
                    primary_block = foreground_block
                elif reference_layer is not None:
                    # Pending, cancelled, or failed Preview has no replacement
                    # statistics.  The retained accepted Reference remains the
                    # analysis owner even when no foreground layer is present.
                    primary_layer_id = reference_layer.layer_id
                    primary_entry = reference_layer.result_entry
                    primary_block = _statistics_block(
                        reference_layer,
                        y_axes=wanted_y,
                        excluded_y_axes=excluded_simulation_y,
                    )

        for experimental in active_overlay_inputs:
            y_axis = ExperimentalAxisSelection(
                experimental.dataset_id,
                experimental.raw_observation_id,
            )
            candidate_label = f"{experimental.dataset_name}: {experimental.raw_column}"
            if experimental.model_species:
                candidate_label = f"{candidate_label} → {experimental.model_species}"
            add_axis_candidate(y_axis, candidate_label)
            experimental_x = np.asarray(experimental.t, dtype=float)
            experimental_y = np.asarray(experimental.y, dtype=float)
            experimental_x_name = "t"
            experimental_x_axis: AxisSeriesSelection = SimulationSeriesAxisSelection(
                "t"
            )
            experimental_x_provenance: Mapping[str, object] = MappingProxyType({})
            selected_source_rows = tuple(experimental.source_rows)
            unavailable_physical_projection = False
            if not _is_time_selection(selected_x):
                if isinstance(selected_x, SimulationSeriesAxisSelection):
                    x_source = next(
                        (
                            candidate
                            for candidate in active_overlay_inputs
                            if candidate.dataset_id == experimental.dataset_id
                            and candidate.model_species == selected_x.series_name
                        ),
                        None,
                    )
                else:
                    x_source = selected_x_source
                if (
                    x_source is None
                    or x_source.dataset_id != experimental.dataset_id
                    or y_axis in excluded_experimental_y
                ):
                    if y_axis in excluded_experimental_y:
                        continue
                    unavailable_physical_projection = True
                else:
                    paired = _same_source_row_values(x_source, experimental)
                    if paired is None:
                        unavailable_physical_projection = True
                    else:
                        (
                            experimental_x,
                            experimental_y,
                            selected_source_rows,
                        ) = paired
                        experimental_x_name = x_source.raw_column
                        experimental_x_axis = selected_x
                        experimental_x_provenance = _observation_provenance(x_source)
            label = candidate_label
            layer = WorkspacePlotLayer(
                layer_id=experimental.layer_id,
                label=label,
                kind=ResultLayerKind.EXPERIMENTAL,
                t=experimental_x,
                series=(
                    (
                        experimental.raw_column,
                        experimental_y,
                    ),
                ),
                series_axes=((experimental.raw_column, y_axis),),
                result_entry=None,
                segment_ids=None,
                segment_keys=None,
                discontinuity_ids=None,
                source_row=np.asarray(selected_source_rows, dtype=np.int64),
                series_provenance=(
                    (
                        experimental.raw_column,
                        _observation_provenance(experimental),
                    ),
                ),
                color_by_series=((experimental.raw_column, colors[y_axis]),),
                opacity=point_opacity,
                markers=True,
                point_size=point_size,
            )
            layers.append(layer)
            if unavailable_physical_projection:
                if wanted_y is None or y_axis in wanted_y:
                    unavailable_layers.append(layer)
            elif wanted_y is None or y_axis in wanted_y:
                blocks.append(
                    ResultDataBlock(
                        block_id=layer.layer_id,
                        label=layer.label,
                        kind=layer.kind,
                        t=experimental_x,
                        series=layer.series,
                        series_axes=layer.series_axes,
                        x_axis=experimental_x_axis,
                        result_entry=None,
                        segment_ids=layer.segment_ids,
                        segment_keys=layer.segment_keys,
                        discontinuity_ids=layer.discontinuity_ids,
                        source_row=layer.source_row,
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
        if not _is_time_selection(selected_x):
            unique_annotations.clear()

        if selected_x not in axis_candidate_ids:
            add_axis_candidate(
                selected_x,
                (
                    selected_x.series_name
                    if isinstance(selected_x, SimulationSeriesAxisSelection)
                    else (
                        f"{selected_x.dataset_id}: "
                        f"{selected_x.raw_observation_id}"
                    )
                ),
            )
        selected_x_candidate = next(
            candidate
            for candidate in axis_candidates
            if candidate.selection == selected_x
        )
        x_label = (
            "Time (s)" if _is_time_selection(selected_x) else selected_x_candidate.label
        )

        effective_y_axes = tuple(
            selection for block in blocks for _name, selection in block.series_axes
        )

        def is_concentration(selection: AxisSeriesSelection) -> bool:
            if isinstance(selection, ExperimentalAxisSelection):
                source = experimental_by_axis.get(selection)
                return source is not None and source.model_species is not None
            for layer in layers:
                if layer.kind is ResultLayerKind.EXPERIMENTAL:
                    continue
                if selection not in dict(layer.series_axes).values():
                    continue
                if layer.result_entry is None:
                    continue
                return (
                    selection.series_name
                    not in layer.result_entry.result.provenance.requested_algebra_outputs
                )
            return False

        y_label = (
            "Concentration (M)"
            if effective_y_axes
            and all(is_concentration(selection) for selection in effective_y_axes)
            else "Value"
        )

        return WorkspaceResultProjection(
            tab_id=tab_id,
            current_set_id=current_id,
            primary_layer_id=primary_layer_id,
            plot_layers=tuple(layers),
            parameters=(
                tuple(primary_entry.result.parameters)
                if primary_entry is not None
                else ()
            ),
            scalar_values=(
                tuple(primary_entry.result.scalars.items())
                if primary_entry is not None
                else ()
            ),
            primary_statistics=primary_block,
            copy_all_blocks=tuple(blocks),
            selected_x_series=selected_x,
            selected_y_series=selected_y_ordered,
            derived_overlay_y_series=derived_overlay_y,
            selected_y_uses_default_all=bool(selected_y_uses_default_all),
            x_label=x_label,
            y_label=y_label,
            axis_candidates=tuple(axis_candidates),
            reference_lines_visible=bool(reference_lines_visible),
            intervention_annotations=tuple(unique_annotations),
            intervention_annotations_visible=bool(
                intervention_annotations_visible and _is_time_selection(selected_x)
            ),
            unavailable_x_diagnostic=_selected_x_unavailable_diagnostic(
                selected_x,
                unavailable_layers,
            ),
        )

    @staticmethod
    def validate_copy(
        projection: WorkspaceResultProjection,
    ) -> None:
        """Validate complete Copy availability at the result projection boundary."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if projection.unavailable_x_diagnostic is not None:
            raise ValueError(projection.unavailable_x_diagnostic)
        if not projection.copy_all_blocks:
            raise ValueError("No visible result, Reference, or experimental data.")

    @staticmethod
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
                    "# Kindred repository entry",
                    encode_clipboard_text(
                        _canonical_json(_result_entry_payload(entry))
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

    @staticmethod
    def validate_xlsx_export(
        projection: WorkspaceResultProjection,
    ) -> None:
        """Validate XLSX worksheet planning before destination creation."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if projection.unavailable_x_diagnostic is not None:
            raise ValueError(projection.unavailable_x_diagnostic)
        blocks = tuple(projection.copy_all_blocks)
        if not blocks:
            raise ValueError("No visible result, Reference, or experimental data.")
        if any(not isinstance(block, ResultDataBlock) for block in blocks):
            raise TypeError(
                "XLSX export blocks must contain only ResultDataBlock values."
            )
        lineage_row_count = sum(
            int(bool(block.x_provenance))
            + sum(bool(provenance) for _, provenance in block.series_provenance)
            for block in blocks
        )
        result_entry_count = len(_unique_result_entries(blocks))
        dimensions = _planned_xlsx_worksheet_dimensions(
            blocks,
            lineage_row_count=lineage_row_count,
            result_entry_count=result_entry_count,
        )
        _validate_xlsx_worksheet_dimensions(dimensions)

    @staticmethod
    def write_xlsx(
        projection: WorkspaceResultProjection,
        destination: BinaryIO,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> XlsxExportCompletion:
        """Write full-resolution worksheets directly to the destination."""

        WorkspaceResultPresenter.validate_xlsx_export(projection)
        blocks = projection.copy_all_blocks
        output = _XlsxDestination(
            destination,
            cancellation_check=cancellation_check,
        )
        workbook = Workbook(write_only=True)
        used: set[str] = set()
        lineage_rows: list[tuple[str, str, Mapping[str, object]]] = []
        metadata_rows: list[tuple[str, ResultEntry]] = []
        metadata_ids: set[int] = set()
        data_rows = 0
        archive: ZipFile | None = None
        original_failure: BaseException | None = None
        try:
            for block in blocks:
                if cancellation_check is not None:
                    cancellation_check()
                title = WorkspaceResultPresenter._worksheet_title(block.label, used)
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
                entry = block.result_entry
                if entry is not None and id(entry) not in metadata_ids:
                    metadata_ids.add(id(entry))
                    metadata_rows.append((title, entry))
            if lineage_rows:
                if cancellation_check is not None:
                    cancellation_check()
                lineage_sheet = workbook.create_sheet(
                    title=WorkspaceResultPresenter._worksheet_title(
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
                        "model_species",
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
                            str(provenance.get("model_species") or ""),
                            str(provenance.get("set_id") or ""),
                        ),
                    )
                lineage_sheet.sheet_state = "hidden"
            if metadata_rows:
                if cancellation_check is not None:
                    cancellation_check()
                metadata_sheet = workbook.create_sheet(
                    title=WorkspaceResultPresenter._worksheet_title(
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
                            _canonical_json(_result_entry_payload(entry)),
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

    @staticmethod
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

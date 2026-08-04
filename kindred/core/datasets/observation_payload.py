from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Dict, Mapping
import uuid

import numpy as np

from kindred.core.project_document import (
    FrozenJsonValue,
    ObservationSeries,
    freeze_json_value,
)

_MAX_RETAINED_OBSERVATION_CELLS = 1_000_000


@dataclass(frozen=True, slots=True)
class DatasetObservationPayload:
    """One deeply immutable parsed payload ready for project admission."""

    observations: tuple[ObservationSeries, ...]
    source_metadata: FrozenJsonValue

    def __post_init__(self) -> None:
        observations = self.observations
        if type(observations) is not tuple or not observations:
            raise ValueError(
                "Dataset payloads require at least one immutable observation "
                "series."
            )
        names: set[str] = set()
        identities: set[str] = set()
        retained_cells = 0
        for series in observations:
            if not isinstance(series, ObservationSeries):
                raise TypeError(
                    "Dataset payload observations must be ObservationSeries "
                    "values."
                )
            if series.raw_observation_id in identities:
                raise ValueError(
                    "Raw observation identity "
                    f"{series.raw_observation_id!r} is duplicated within the "
                    "dataset."
                )
            if series.column_name in names:
                raise ValueError(
                    f"Observation series name {series.column_name!r} is duplicated."
                )
            retained_cells += len(series.t)
            if retained_cells > _MAX_RETAINED_OBSERVATION_CELLS:
                raise ValueError(
                    "Dataset payload exceeds the 1000000 retained-observation-"
                    "cell limit."
                )
            names.add(series.column_name)
            identities.add(series.raw_observation_id)


def new_raw_observation_id() -> str:
    """Allocate one opaque dataset-local raw observation identity."""
    return uuid.uuid4().hex


def normalize_raw_observation_id(value: object) -> str:
    """Validate one already-allocated opaque raw observation identity."""
    if type(value) is not str:
        raise ValueError("Raw observation identity must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(
            "Raw observation identity must be nonblank and edge-normalized."
        )
    return normalized


def _as_finite_float_tuple(
    values: object,
    *,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, np.ndarray):
        source = values.reshape(-1)
    else:
        try:
            source = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{label} must be a one-dimensional sequence.") from exc
    converted: list[float] = []
    for value in source:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must contain only finite numeric values.")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must contain only finite numeric values.")
        converted.append(parsed)
    return tuple(converted)


def _as_source_row_tuple(values: object) -> tuple[int, ...]:
    if isinstance(values, np.ndarray):
        source = values.reshape(-1)
    else:
        try:
            source = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "Source row identities must be a one-dimensional sequence."
            ) from exc
    rows: list[int] = []
    for value in source:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Source row identities must be positive integers.")
        try:
            row = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Source row identities must be positive integers."
            ) from exc
        if not math.isfinite(numeric) or numeric != row or row <= 0:
            raise ValueError("Source row identities must be positive integers.")
        rows.append(row)
    return tuple(rows)


def immutable_dataset_payload(
    observations: Mapping[str, object] | None,
    *,
    metadata: Mapping[str, object] | None = None,
) -> DatasetObservationPayload:
    """Validate and materialize parser output exactly once into final tuples."""

    if not isinstance(observations, Mapping) or not observations:
        raise ValueError(
            "Dataset payloads require at least one observation series."
        )
    materialized: list[ObservationSeries] = []
    names: set[str] = set()
    identities: set[str] = set()
    retained_cells = 0
    for raw_name, raw_spec in observations.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Observation series names must be nonblank.")
        if name in names:
            raise ValueError(
                f"Observation series name {name!r} is duplicated after trimming."
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Observation series {name!r} must be a mapping.")
        raw_observation_id = normalize_raw_observation_id(
            raw_spec.get("raw_observation_id")
        )
        if raw_observation_id in identities:
            raise ValueError(
                "Raw observation identity "
                f"{raw_observation_id!r} is duplicated within the dataset."
            )
        times = _as_finite_float_tuple(
            raw_spec.get("t", ()),
            label=f"Observation series {name!r} times",
        )
        values = _as_finite_float_tuple(
            raw_spec.get("y", ()),
            label=f"Observation series {name!r} values",
        )
        rows = _as_source_row_tuple(raw_spec.get("source_rows", ()))
        if not times or len(times) != len(values) or len(times) != len(rows):
            raise ValueError(
                f"Observation series {name!r} requires equal nonempty time, "
                "value, and source-row sequences."
            )
        retained_cells += len(times)
        if retained_cells > _MAX_RETAINED_OBSERVATION_CELLS:
            raise ValueError(
                "Dataset payload exceeds the 1000000 retained-observation-cell "
                "limit."
            )
        if len(set(rows)) != len(rows):
            raise ValueError(
                f"Observation series {name!r} contains duplicate physical "
                "source-row identities."
            )
        materialized.append(
            ObservationSeries(
                raw_observation_id=raw_observation_id,
                column_name=name,
                t=times,
                y=values,
                source_rows=rows,
            )
        )
        names.add(name)
        identities.add(raw_observation_id)
    return DatasetObservationPayload(
        observations=tuple(materialized),
        source_metadata=freeze_json_value(dict(metadata or {})),
    )


def _as_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1).copy()


def _as_source_row_array(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size == 0:
        return np.asarray([], dtype=np.int64)
    flattened = raw.reshape(-1)
    rows: list[int] = []
    for value in flattened:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Source row identities must be positive integers.")
        try:
            row = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Source row identities must be positive integers.") from exc
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Source row identities must be positive integers.") from exc
        if not np.isfinite(numeric) or numeric != row or row <= 0:
            raise ValueError("Source row identities must be positive integers.")
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def copy_observations_map(
    observations: Mapping[str, object] | None,
) -> Dict[str, Dict[str, object]]:
    if not isinstance(observations, Mapping) or not observations:
        raise ValueError(
            "Dataset payloads require at least one observation series."
        )
    copied: Dict[str, Dict[str, object]] = {}
    raw_observation_ids: set[str] = set()
    for raw_name, raw_spec in observations.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Observation series names must be nonblank.")
        if name in copied:
            raise ValueError(
                f"Observation series name {name!r} is duplicated after trimming."
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"Observation series {name!r} must be a mapping."
            )
        raw_observation_id = normalize_raw_observation_id(
            raw_spec.get("raw_observation_id")
        )
        if raw_observation_id in raw_observation_ids:
            raise ValueError(
                "Raw observation identity "
                f"{raw_observation_id!r} is duplicated within the dataset."
            )
        raw_observation_ids.add(raw_observation_id)
        t_values = _as_float_array(raw_spec.get("t", []))
        y_values = _as_float_array(raw_spec.get("y", []))
        if t_values.size == 0 or t_values.size != y_values.size:
            raise ValueError(
                f"Observation series {name!r} requires equal nonempty time "
                f"and value arrays; received {t_values.size} and "
                f"{y_values.size}."
            )
        if not np.all(np.isfinite(t_values)) or not np.all(np.isfinite(y_values)):
            raise ValueError(f"Observation series '{name}' contains non-finite values.")
        source_rows = _as_source_row_array(raw_spec.get("source_rows", ()))
        if source_rows.size != t_values.size:
            raise ValueError(
                f"Observation series '{name}' has {t_values.size} observations but "
                f"{source_rows.size} source row identities."
            )
        if len(set(int(value) for value in source_rows)) != source_rows.size:
            raise ValueError(
                f"Observation series {name!r} contains duplicate physical "
                "source-row identities."
            )
        copied[name] = {
            "raw_observation_id": raw_observation_id,
            "t": t_values,
            "y": y_values,
            "source_rows": source_rows,
        }
    return copied

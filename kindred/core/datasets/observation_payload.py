from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from collections.abc import Iterable
from typing import Mapping
import uuid

import numpy as np

from kindred.core.datasets.foundation import (
    FrozenJsonValue,
    ObservationSeries,
    freeze_json_value,
    validate_raw_observation_id,
)

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
            names.add(series.column_name)
            identities.add(series.raw_observation_id)


def new_raw_observation_id() -> str:
    """Allocate one opaque dataset-local raw observation identity."""
    return uuid.uuid4().hex


def _as_finite_float_tuple(
    values: object,
    *,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, np.ndarray):
        source = values.reshape(-1)
    else:
        if not isinstance(values, Iterable):
            raise ValueError(f"{label} must be a one-dimensional sequence.")
        source = tuple(values)
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
        if not isinstance(values, Iterable):
            raise ValueError(
                "Source row identities must be a one-dimensional sequence."
            )
        source = tuple(values)
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
        try:
            raw_observation_id = validate_raw_observation_id(
                raw_spec.get("raw_observation_id"),
                "raw observation identity",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
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

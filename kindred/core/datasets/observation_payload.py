from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import uuid

from kindred.core.quantity_units import SECOND, Unit, conversion_scale, scientific_unit
from kindred.core.species import validate_observation

from kindred.core.datasets.foundation import (
    FrozenJsonValue,
    ObservationSeries,
)


@dataclass(frozen=True, slots=True)
class DatasetObservationPayload:
    """One deeply immutable parsed payload ready for project admission."""

    observations: tuple[ObservationSeries, ...]
    source_metadata: FrozenJsonValue


def new_raw_observation_id() -> str:
    """Allocate one opaque dataset-local raw observation identity."""
    return uuid.uuid4().hex


def created_dataset_payload(
    columns: Sequence[tuple[str, Unit | None, Sequence[float | None]]],
    *,
    time: tuple[Unit, Sequence[float | None]] | None = None,
) -> DatasetObservationPayload:
    """Admit numeric columns using retained known or accepted unknown units."""
    names = [name.strip() for name, _unit, _values in columns]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Measurement column names must be nonempty and unique.")
    retained_rows = sorted({
        row for _name, _unit, values in columns
        for row, value in enumerate(values, 1) if value is not None
    })
    recorded: dict[int, float] = {}
    if time is not None:
        time_unit, values = time
        scale = conversion_scale(time_unit, SECOND)
        for row in retained_rows:
            if values[row - 1] is None:
                raise ValueError(f"Row {row}: a measurement requires recorded time.")
            recorded[row] = scale.apply(validate_observation(values[row - 1], field=f"Row {row} time"))
    observations = []
    for name, (_old_name, unit, values) in zip(names, columns, strict=True):
        scientific = None if unit is None else scientific_unit(unit.dimensions)
        scale = None if unit is None else conversion_scale(unit, scientific)
        rows, measurements = [], []
        for row, value in enumerate(values, 1):
            if value is not None:
                label = f"{name}, row {row}"
                admitted = validate_observation(value, field=label)
                measurements.append(admitted if scale is None else scale.apply(admitted))
                rows.append(row)
        if measurements:
            observations.append(ObservationSeries(
                raw_observation_id=new_raw_observation_id(), column_name=name,
                unit=scientific, t=tuple(recorded[row] for row in rows) if time is not None else (),
                y=tuple(measurements), source_rows=tuple(rows),
            ))
    if not observations:
        raise ValueError("There are no measurements to create a dataset from.")
    return DatasetObservationPayload(observations=tuple(observations), source_metadata=None)

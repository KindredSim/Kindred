"""Immutable, row-aligned Fit dataset captures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

import numpy as np

__all__ = [
    "FitDatasetSpec",
    "FitObservationSeries",
    "normalize_dataset_target_weights",
]


def _readonly_float_vector(values: object, field: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FitObservationSeries:
    """One visible Y series with its dataset row identity."""

    raw_observation_id: str
    raw_column: str
    t: np.ndarray
    y: np.ndarray
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.raw_observation_id) is not str
            or not self.raw_observation_id
            or self.raw_observation_id != self.raw_observation_id.strip()
        ):
            raise ValueError(
                "Fit observation raw identity must be a nonblank "
                "dataset-local opaque string."
            )
        if (
            type(self.raw_column) is not str
            or not self.raw_column
            or self.raw_column != self.raw_column.strip()
        ):
            raise ValueError("Fit observation raw column must be nonblank.")
        t = _readonly_float_vector(self.t, "observation time")
        y = _readonly_float_vector(self.y, "observation value")
        if type(self.source_rows) is not tuple or any(
            type(value) is not int for value in self.source_rows
        ):
            raise ValueError(
                "Fit observation source rows must be an exact integer tuple."
            )
        rows = self.source_rows
        if y.size == 0 or (t.size and t.size != y.size) or len(rows) != y.size:
            raise ValueError(
                "Fit observation time, value, and source-row arrays must have "
                "the same nonzero length."
            )
        if len(rows) != len(set(rows)) or any(value <= 0 for value in rows):
            raise ValueError(
                "Fit observation source rows must be unique and positive."
            )
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "source_rows", rows)


def normalize_dataset_target_weights(
    *,
    dataset_id: str,
    selected_targets: Sequence[str],
    target_weights: Optional[Mapping[str, object]],
) -> dict[str, float]:
    selected = tuple(str(name).strip() for name in selected_targets if str(name).strip())
    raw = dict(target_weights or {})
    unknown = set(raw) - set(selected)
    if unknown:
        raise ValueError(
            f"Dataset {dataset_id!r} has weights for unselected targets: "
            f"{', '.join(sorted(unknown))}."
        )
    normalized: dict[str, float] = {}
    for name in selected:
        value = float(raw.get(name, 1.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Dataset {dataset_id!r} target weight for {name!r} must be "
                "finite and positive."
            )
        normalized[name] = value
    return normalized


@dataclass(frozen=True, slots=True)
class FitDatasetSpec:
    """One captured dataset; no live dataset lookup is permitted after capture."""

    dataset_id: str
    species_list: tuple[str, ...]
    observations: Mapping[str, FitObservationSeries]
    target_weights: Mapping[str, float] = field(default_factory=dict)
    x_name: str = "t"
    # One immutable X capture. Its `y` member stores captured X values; using
    # the same row-aligned DTO keeps raw identity and dataset row
    # custody identical for Y and X.
    x_observations: FitObservationSeries | None = None

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id or "").strip()
        if not dataset_id:
            raise ValueError("Fit dataset ID must be nonblank.")
        species = tuple(
            str(name).strip() for name in self.species_list if str(name).strip()
        )
        if not species or len(species) != len(set(species)):
            raise ValueError("Fit targets must be nonempty and unique.")
        observation_map = dict(self.observations)
        if set(observation_map) != set(species):
            raise ValueError(
                f"Dataset {dataset_id!r} observations must exactly match selected targets."
            )
        for name, series in observation_map.items():
            if not isinstance(series, FitObservationSeries):
                raise TypeError(
                    f"Dataset {dataset_id!r} observation {name!r} must be "
                    "FitObservationSeries."
                )
        raw_observation_ids = tuple(
            series.raw_observation_id
            for series in observation_map.values()
        )
        if len(raw_observation_ids) != len(set(raw_observation_ids)):
            raise ValueError(
                f"Dataset {dataset_id!r} Fit targets must use unique stable "
                "raw observation IDs."
            )
        x_name = str(self.x_name or "t").strip() or "t"
        x_series: FitObservationSeries | None = None
        if x_name == "t":
            if any(series.t.size == 0 for series in observation_map.values()):
                raise ValueError("Time-X fitting requires recorded experimental times.")
            if self.x_observations is not None:
                raise ValueError("Time-X Fit captures must not contain an X block.")
        else:
            if x_name in species:
                raise ValueError(
                    f"Dataset {dataset_id!r} cannot fit its parametric X "
                    "series as a Y target."
                )
            x_series = self.x_observations
            if not isinstance(x_series, FitObservationSeries):
                raise ValueError(
                    f"Dataset {dataset_id!r} parametric-X captures require one X block."
                )
            if x_series.raw_observation_id in raw_observation_ids:
                raise ValueError(
                    f"Dataset {dataset_id!r} X raw observation ID must be unique from Y."
                )
            x_rows = set(x_series.source_rows)
            for name, series in observation_map.items():
                if not set(series.source_rows).issubset(x_rows):
                    raise ValueError(
                        f"Dataset {dataset_id!r} target {name!r} references an X source row "
                        "that is absent from the captured X block."
                    )
            if not x_rows:
                raise ValueError(f"Dataset {dataset_id!r} X block must not be empty.")

        weight_roster = species if x_name == "t" else (*species, x_name)
        weights = normalize_dataset_target_weights(
            dataset_id=dataset_id,
            selected_targets=weight_roster,
            target_weights=self.target_weights,
        )
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "species_list", species)
        object.__setattr__(
            self,
            "observations",
            MappingProxyType(dict(observation_map)),
        )
        object.__setattr__(self, "target_weights", MappingProxyType(weights))
        object.__setattr__(self, "x_name", x_name)
        object.__setattr__(self, "x_observations", x_series)

    @property
    def point_count(self) -> int:
        y_count = sum(
            int(series.y.size)
            for series in self.observations.values()
        )
        if self.x_name == "t" or self.x_observations is None:
            return y_count
        retained_x_rows = {
            int(row)
            for series in self.observations.values()
            for row in series.source_rows
        }
        return y_count + len(retained_x_rows)

    @property
    def target_roster(self) -> tuple[str, ...]:
        """Exact weighted objective roster, including canonical X."""

        return self.species_list if self.x_name == "t" else (*self.species_list, self.x_name)

    @property
    def x_source_rows(self) -> tuple[int, ...]:
        """Ordered union of dataset X rows referenced by retained Y rows."""

        if self.x_name == "t" or self.x_observations is None:
            return ()
        retained = {
            int(row)
            for series in self.observations.values()
            for row in series.source_rows
        }
        return tuple(row for row in self.x_observations.source_rows if row in retained)

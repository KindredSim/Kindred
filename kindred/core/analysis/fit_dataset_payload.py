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
    """One visible Y series with its physical source-row identity."""

    raw_observation_id: str
    t: np.ndarray
    y: np.ndarray
    source_rows: tuple[int, ...]
    pre_boundary_segment_keys_by_source_row: tuple[
        tuple[int, str],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        raw_observation_id = str(self.raw_observation_id)
        if (
            not raw_observation_id
            or raw_observation_id != raw_observation_id.strip()
        ):
            raise ValueError(
                "Fit observation raw identity must be a nonblank "
                "dataset-local opaque string."
            )
        t = _readonly_float_vector(self.t, "observation time")
        y = _readonly_float_vector(self.y, "observation value")
        rows = tuple(int(value) for value in self.source_rows)
        if t.size == 0 or t.size != y.size or len(rows) != y.size:
            raise ValueError(
                "Fit observation time, value, and source-row arrays must have "
                "the same nonzero length."
            )
        if len(rows) != len(set(rows)) or any(value <= 0 for value in rows):
            raise ValueError(
                "Fit observation source rows must be unique and positive."
            )
        pre_boundary_keys = tuple(
            (int(row), str(key or "").strip())
            for row, key in self.pre_boundary_segment_keys_by_source_row
        )
        keyed_rows = [row for row, _key in pre_boundary_keys]
        if (
            len(keyed_rows) != len(set(keyed_rows))
            or any(row not in rows for row in keyed_rows)
            or any(not key for _row, key in pre_boundary_keys)
        ):
            raise ValueError(
                "Explicit pre-boundary Fit identity must contain unique "
                "captured source rows and nonblank exact segment keys."
            )
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "raw_observation_id", raw_observation_id)
        object.__setattr__(self, "source_rows", rows)
        object.__setattr__(
            self,
            "pre_boundary_segment_keys_by_source_row",
            pre_boundary_keys,
        )


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
    x_mode: str = "monotone"
    x_obs_by_species: Mapping[str, np.ndarray] = field(default_factory=dict)

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
        x_name = str(self.x_name or "t").strip() or "t"
        x_mode = str(self.x_mode or "").strip().lower()
        axes: dict[str, np.ndarray] = {}
        if x_name == "t":
            x_mode = "monotone"
            if self.x_obs_by_species:
                raise ValueError("Time-X Fit captures must not contain parametric axes.")
        else:
            if x_name in species:
                raise ValueError(
                    f"Dataset {dataset_id!r} cannot fit its parametric X "
                    "series as a Y target."
                )
            if x_mode not in {"monotone", "time_guided"}:
                raise ValueError(
                    f"Dataset {dataset_id!r} requires a resolved parametric-X "
                    "mode of 'monotone' or 'time_guided'."
                )
            raw_axes = dict(self.x_obs_by_species)
            if set(raw_axes) != set(species):
                raise ValueError(
                    f"Dataset {dataset_id!r} parametric-X axes must exactly match targets."
                )
            for name in species:
                axis = _readonly_float_vector(
                    raw_axes[name],
                    f"{dataset_id}:{name} observed X",
                )
                if axis.size != observation_map[name].y.size:
                    raise ValueError(
                        f"Dataset {dataset_id!r} target {name!r} has misaligned X/Y rows."
                    )
                axes[name] = axis

        weights = normalize_dataset_target_weights(
            dataset_id=dataset_id,
            selected_targets=species,
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
        object.__setattr__(self, "x_mode", x_mode)
        object.__setattr__(self, "x_obs_by_species", MappingProxyType(axes))

    @property
    def point_count(self) -> int:
        return sum(
            int(series.y.size)
            for series in self.observations.values()
        )

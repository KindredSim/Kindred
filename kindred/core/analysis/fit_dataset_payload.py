"""Immutable, row-aligned Fit dataset captures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

__all__ = [
    "FitDatasetSpec",
    "FitObservationSeries",
]


def _readonly_float_vector(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FitObservationSeries:
    """One visible Y series with its dataset row identity.

    Its values are admitted dataset observations with their unique positive
    source rows.
    """

    raw_observation_id: str
    raw_column: str
    t: np.ndarray
    y: np.ndarray
    source_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _readonly_float_vector(self.t))
        object.__setattr__(self, "y", _readonly_float_vector(self.y))


@dataclass(frozen=True, slots=True)
class FitDatasetSpec:
    """One captured dataset; no live dataset lookup is permitted after capture.

    The Fit request selects one uniquely mapped observation per target and,
    for Parametric-X, the X rows paired with them; target weights are the
    admitted dataset settings.
    """

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
        weight_roster = (
            self.species_list if self.x_name == "t" else (*self.species_list, self.x_name)
        )
        # Order the admitted target weights by the weighted roster.
        weights = {name: float(self.target_weights[name]) for name in weight_roster}
        object.__setattr__(
            self,
            "observations",
            MappingProxyType(dict(self.observations)),
        )
        object.__setattr__(self, "target_weights", MappingProxyType(weights))

    @property
    def point_count(self) -> int:
        y_count = sum(
            int(series.y.size)
            for series in self.observations.values()
        )
        if self.x_name == "t":
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

        if self.x_name == "t":
            return ()
        retained = {
            int(row)
            for series in self.observations.values()
            for row in series.source_rows
        }
        return tuple(row for row in self.x_observations.source_rows if row in retained)

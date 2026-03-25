from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class SimulationSeriesPayload(Mapping[str, Any]):
    """Typed payload for fitting-oriented simulation series output."""

    t: np.ndarray
    species: Dict[str, np.ndarray]
    algebra_scalars: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "t":
            return self.t
        if key == "species":
            return self.species
        if key == "algebra_scalars":
            return self.algebra_scalars
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "t"
        yield "species"
        if self.algebra_scalars:
            yield "algebra_scalars"

    def __len__(self) -> int:
        return 2 + int(bool(self.algebra_scalars))

    def to_legacy_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t": self.t,
            "species": self.species,
        }
        if self.algebra_scalars:
            payload["algebra_scalars"] = dict(self.algebra_scalars)
        return payload


def coerce_simulation_series_payload(simulation: object) -> SimulationSeriesPayload:
    if isinstance(simulation, SimulationSeriesPayload):
        return simulation
    if not isinstance(simulation, Mapping):
        raise ValueError("Simulation function must return a dictionary.")

    t_sim = simulation.get("t")
    if t_sim is not None:
        t_sim = np.asarray(t_sim, dtype=float).reshape(-1)

    species_map = simulation.get("species")
    if species_map is not None:
        normalized_species = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in species_map.items()
        }
    else:
        normalized_species = {}
        for key, values in simulation.items():
            if key in {"t", "algebra_scalars"}:
                continue
            try:
                arr = np.asarray(values, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                arr = None
            if arr is None:
                continue
            normalized_species[str(key)] = arr
        if not normalized_species:
            raise ValueError("Simulation result contains no species data.")

    raw_scalars = simulation.get("algebra_scalars")
    normalized_scalars: Dict[str, float] = {}
    if isinstance(raw_scalars, Mapping):
        for name, value in raw_scalars.items():
            try:
                normalized_scalars[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

    return SimulationSeriesPayload(
        t=np.asarray(t_sim, dtype=float).reshape(-1) if t_sim is not None else np.asarray([], dtype=float),
        species=normalized_species,
        algebra_scalars=normalized_scalars,
    )

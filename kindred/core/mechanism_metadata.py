from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

from kindred.core.temperature import TemperatureSchedule

__all__ = [
    "MechanismMetadataKeys",
    "EquilibriumMetadataKeys",
    "MechanismMetadataView",
    "EquilibriumMetadataView",
]


class MechanismMetadataKeys:
    TEMPERATURE_K = "temperature_K"
    STANDARD_CONC_M = "standard_conc_M"
    ENERGY_UNIT = "energy_unit"
    TEMPERATURE_SCHEDULE = "temperature_schedule"


class EquilibriumMetadataKeys:
    DG_EQ_J_PER_MOL = "dG_eq_J_per_mol"
    KEQ_INPUT = "Keq_input"
    STANDARD_CONC_M = "standard_conc_M"


# Temperature and standard concentrations arrive admitted by DSL parsing.
@dataclass(frozen=True)
class MechanismMetadataView:
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0
    energy_unit: str = "kJ/mol"
    temperature_schedule: Optional[TemperatureSchedule] = None

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any]) -> "MechanismMetadataView":
        return cls(
            temperature_K=meta[MechanismMetadataKeys.TEMPERATURE_K],
            standard_conc_M=meta[MechanismMetadataKeys.STANDARD_CONC_M],
            energy_unit=meta[MechanismMetadataKeys.ENERGY_UNIT],
            temperature_schedule=meta.get(MechanismMetadataKeys.TEMPERATURE_SCHEDULE),
        )

    def apply_to_metadata(self, meta: MutableMapping[str, Any]) -> None:
        meta[MechanismMetadataKeys.TEMPERATURE_K] = float(self.temperature_K)
        meta[MechanismMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        meta[MechanismMetadataKeys.ENERGY_UNIT] = str(self.energy_unit)
        if self.temperature_schedule is not None:
            meta[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = self.temperature_schedule


@dataclass(frozen=True)
class EquilibriumMetadataView:
    dG_eq_J_per_mol: Optional[float] = None
    Keq_input: Optional[object] = None
    standard_conc_M: Optional[float] = None

    def to_metadata(self) -> dict[str, object]:
        out: dict[str, object] = {
        }
        if self.dG_eq_J_per_mol is not None:
            out[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        if self.Keq_input is not None:
            out[EquilibriumMetadataKeys.KEQ_INPUT] = self.Keq_input
        if self.standard_conc_M is not None:
            out[EquilibriumMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        return out

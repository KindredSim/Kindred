from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

from kindred.core.simulator.common import normalize_energy_unit
from kindred.core.species import validate_c0
from kindred.core.temperature import (
    TemperatureScheduleProtocol,
    coerce_temperature_kelvin,
    coerce_temperature_schedule,
)

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
    FAST_EQUILIBRIUM = "fast_equilibrium"
    USER_PROVIDED_KF = "user_provided_kf"
    USER_PROVIDED_KR = "user_provided_kr"
    DG_EQ_J_PER_MOL = "dG_eq_J_per_mol"
    KEQ_INPUT = "Keq_input"
    FORWARD_MODEL = "forward_model"
    STANDARD_CONC_M = "standard_conc_M"
    KEQ_UNIT = "Keq_unit"


@dataclass(frozen=True)
class MechanismMetadataView:
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0
    energy_unit: str = "kJ/mol"
    temperature_schedule: Optional[TemperatureScheduleProtocol] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature_K",
            coerce_temperature_kelvin(self.temperature_K, field=MechanismMetadataKeys.TEMPERATURE_K),
        )
        standard = validate_c0(
            self.standard_conc_M,
            field=MechanismMetadataKeys.STANDARD_CONC_M,
        )
        object.__setattr__(self, "standard_conc_M", standard)

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any] | None) -> "MechanismMetadataView":
        if not isinstance(meta, Mapping):
            meta = {}
        return cls(
            temperature_K=coerce_temperature_kelvin(
                meta.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15),
                field=MechanismMetadataKeys.TEMPERATURE_K,
            ),
            standard_conc_M=meta.get(MechanismMetadataKeys.STANDARD_CONC_M, 1.0),
            energy_unit=normalize_energy_unit(
                meta.get(MechanismMetadataKeys.ENERGY_UNIT, "kJ/mol"),
                default="kJ/mol",
            ),
            temperature_schedule=coerce_temperature_schedule(
                meta.get(MechanismMetadataKeys.TEMPERATURE_SCHEDULE)
            ),
        )

    def apply_to_metadata(self, meta: MutableMapping[str, Any]) -> None:
        meta[MechanismMetadataKeys.TEMPERATURE_K] = float(self.temperature_K)
        meta[MechanismMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        meta[MechanismMetadataKeys.ENERGY_UNIT] = str(self.energy_unit)
        if self.temperature_schedule is not None:
            meta[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = self.temperature_schedule


@dataclass(frozen=True)
class EquilibriumMetadataView:
    fast_equilibrium: bool
    user_provided_kf: bool
    user_provided_kr: bool
    dG_eq_J_per_mol: Optional[float] = None
    Keq_input: Optional[object] = None
    forward_model: Optional[dict[str, object]] = None
    standard_conc_M: Optional[float] = None
    Keq_unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.standard_conc_M is not None:
            object.__setattr__(
                self,
                "standard_conc_M",
                validate_c0(
                    self.standard_conc_M,
                    field=EquilibriumMetadataKeys.STANDARD_CONC_M,
                ),
            )

    def to_metadata(self) -> dict[str, object]:
        out: dict[str, object] = {
            EquilibriumMetadataKeys.FAST_EQUILIBRIUM: bool(self.fast_equilibrium),
            EquilibriumMetadataKeys.USER_PROVIDED_KF: bool(self.user_provided_kf),
            EquilibriumMetadataKeys.USER_PROVIDED_KR: bool(self.user_provided_kr),
        }
        if self.dG_eq_J_per_mol is not None:
            out[EquilibriumMetadataKeys.DG_EQ_J_PER_MOL] = float(self.dG_eq_J_per_mol)
        if self.Keq_input is not None:
            out[EquilibriumMetadataKeys.KEQ_INPUT] = self.Keq_input
        if self.forward_model is not None:
            out[EquilibriumMetadataKeys.FORWARD_MODEL] = dict(self.forward_model)
        if self.standard_conc_M is not None:
            out[EquilibriumMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        if self.Keq_unit is not None:
            out[EquilibriumMetadataKeys.KEQ_UNIT] = str(self.Keq_unit)
        return out

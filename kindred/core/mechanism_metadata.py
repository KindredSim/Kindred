from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, MutableMapping, Optional

from kindred.core.simulator.common import normalize_energy_unit
from kindred.core.temperature import TemperatureScheduleProtocol, coerce_temperature_schedule
from kindred.core.intervention_schedule import InterventionSchedule, coerce_intervention_schedule

__all__ = [
    "MechanismMetadataKeys",
    "EquilibriumMetadataKeys",
    "MechanismMetadataView",
    "EquilibriumMetadataView",
]


class MechanismMetadataKeys:
    TEMPERATURE_K = "temperature_K"
    STANDARD_CONC_M = "standard_conc_M"
    KAPPA_GLOBAL = "kappa_global"
    ENERGY_UNIT = "energy_unit"
    STEP_INDEX_POLICY = "step_index_policy"
    ALGEBRA_TEXT = "algebra_text"
    TEMPERATURE_SCHEDULE = "temperature_schedule"
    INTERVENTION_SCHEDULE = "intervention_schedule"
    WEGSCHEIDER_CYCLICITY_ENABLED = "wegscheider_cyclicity_enabled"
    STATE_NETWORK = "state_network"


class EquilibriumMetadataKeys:
    FAST_EQUILIBRIUM = "fast_equilibrium"
    USER_PROVIDED_KF = "user_provided_kf"
    USER_PROVIDED_KR = "user_provided_kr"
    DG_EQ_J_PER_MOL = "dG_eq_J_per_mol"
    KEQ_INPUT = "Keq_input"
    EXPLICIT_RATES = "explicit_rates"
    FORWARD_MODEL = "forward_model"
    STANDARD_CONC_M = "standard_conc_M"


def _as_float(value: object, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return bool(int(value))
        except Exception:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none", "null"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _optional_finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


@dataclass(frozen=True)
class MechanismMetadataView:
    temperature_K: float = 298.15
    standard_conc_M: float = 1.0
    kappa_global: float = 1.0
    energy_unit: str = "kJ/mol"
    step_index_policy: str = "dsl_only"
    algebra_text: Optional[str] = None
    temperature_schedule: Optional[TemperatureScheduleProtocol] = None
    intervention_schedule: Optional[InterventionSchedule] = None

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any] | None) -> "MechanismMetadataView":
        if not isinstance(meta, Mapping):
            meta = {}
        return cls(
            temperature_K=_as_float(
                meta.get(MechanismMetadataKeys.TEMPERATURE_K, 298.15),
                default=298.15,
            ),
            standard_conc_M=_as_float(
                meta.get(MechanismMetadataKeys.STANDARD_CONC_M, 1.0),
                default=1.0,
            ),
            kappa_global=_as_float(
                meta.get(MechanismMetadataKeys.KAPPA_GLOBAL, 1.0),
                default=1.0,
            ),
            energy_unit=normalize_energy_unit(
                meta.get(MechanismMetadataKeys.ENERGY_UNIT, "kJ/mol"),
                default="kJ/mol",
            ),
            step_index_policy=str(
                meta.get(MechanismMetadataKeys.STEP_INDEX_POLICY, "dsl_only")
            ),
            algebra_text=(
                str(meta.get(MechanismMetadataKeys.ALGEBRA_TEXT))
                if meta.get(MechanismMetadataKeys.ALGEBRA_TEXT) is not None
                else None
            ),
            temperature_schedule=coerce_temperature_schedule(
                meta.get(MechanismMetadataKeys.TEMPERATURE_SCHEDULE)
            ),
            intervention_schedule=coerce_intervention_schedule(
                meta.get(MechanismMetadataKeys.INTERVENTION_SCHEDULE)
            ),
        )

    def apply_to_metadata(self, meta: MutableMapping[str, Any]) -> None:
        meta[MechanismMetadataKeys.TEMPERATURE_K] = float(self.temperature_K)
        meta[MechanismMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        meta[MechanismMetadataKeys.KAPPA_GLOBAL] = float(self.kappa_global)
        meta[MechanismMetadataKeys.ENERGY_UNIT] = str(self.energy_unit)
        meta[MechanismMetadataKeys.STEP_INDEX_POLICY] = str(self.step_index_policy)
        if self.algebra_text is not None:
            meta[MechanismMetadataKeys.ALGEBRA_TEXT] = str(self.algebra_text)
        if self.temperature_schedule is not None:
            meta[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = self.temperature_schedule
        if self.intervention_schedule is not None:
            meta[MechanismMetadataKeys.INTERVENTION_SCHEDULE] = self.intervention_schedule


@dataclass(frozen=True)
class EquilibriumMetadataView:
    fast_equilibrium: bool
    user_provided_kf: bool
    user_provided_kr: bool
    dG_eq_J_per_mol: Optional[float] = None
    Keq_input: Optional[object] = None
    explicit_rates: tuple[float, ...] = ()
    forward_model: Optional[dict[str, object]] = None
    standard_conc_M: Optional[float] = None

    @classmethod
    def from_metadata(
        cls,
        meta: Mapping[str, Any] | None,
        *,
        default_fast: bool = False,
    ) -> "EquilibriumMetadataView":
        if not isinstance(meta, Mapping):
            meta = {}
        explicit_rates = meta.get(EquilibriumMetadataKeys.EXPLICIT_RATES, ())
        if not isinstance(explicit_rates, (list, tuple)):
            explicit_rates = ()
        explicit_rates_f = []
        for x in explicit_rates:
            coerced = _optional_finite_float(x)
            if coerced is not None:
                explicit_rates_f.append(coerced)
        forward_model = meta.get(EquilibriumMetadataKeys.FORWARD_MODEL)
        if isinstance(forward_model, Mapping):
            forward_model = dict(forward_model)
        else:
            forward_model = None

        dG = meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
        dG_f = _optional_finite_float(dG) if dG is not None else None

        std = meta.get(EquilibriumMetadataKeys.STANDARD_CONC_M)
        std_f = _optional_finite_float(std) if std is not None else None

        return cls(
            fast_equilibrium=_as_bool(
                meta.get(EquilibriumMetadataKeys.FAST_EQUILIBRIUM, default_fast)
            ),
            user_provided_kf=_as_bool(meta.get(EquilibriumMetadataKeys.USER_PROVIDED_KF)),
            user_provided_kr=_as_bool(meta.get(EquilibriumMetadataKeys.USER_PROVIDED_KR)),
            dG_eq_J_per_mol=dG_f,
            Keq_input=meta.get(EquilibriumMetadataKeys.KEQ_INPUT),
            explicit_rates=tuple(explicit_rates_f),
            forward_model=forward_model,
            standard_conc_M=std_f,
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
        if self.explicit_rates:
            out[EquilibriumMetadataKeys.EXPLICIT_RATES] = list(self.explicit_rates)
        if self.forward_model is not None:
            out[EquilibriumMetadataKeys.FORWARD_MODEL] = dict(self.forward_model)
        if self.standard_conc_M is not None:
            out[EquilibriumMetadataKeys.STANDARD_CONC_M] = float(self.standard_conc_M)
        return out

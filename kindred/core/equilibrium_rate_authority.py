from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from kindred.core.mechanism_metadata import EquilibriumMetadataKeys

AUTHORITY_ERROR = "equilibrium requires kf and exactly one of kr or Keq/dG_eq"
GENERATED_RATE_MODEL_ERROR = "generated transition-state equilibrium requires complete generated_rate_model"
GENERATED_DIRECT_MODEL_ERROR = "generated direct thermodynamic equilibrium requires complete generated_rate_model"
GENERATED_FAST_EQUILIBRIUM_RUNTIME_PARAMETER_PREFIX = "dG_eq_fast__"
_AUTHORITY_SOURCE_METADATA_KEY = "authority_source"
_GENERATED_SCALAR_MIRROR_METADATA_KEYS = frozenset({"kf", "kr", "Keq"})


class EquilibriumRateAuthorityKind:
    KR = "kr_authoritative"
    KEQ = "Keq_authoritative"


class EquilibriumRateInputContext(str, Enum):
    PUBLIC = "public"
    NORMALIZED_PUBLIC = "normalized_public"
    GENERATED_COMPUTATIONAL_MODE = "generated_computational_mode"


class GeneratedEquilibriumAuthorityKind:
    TRANSITION_STATE_BARRIER_RATES = "generated_transition_state_barrier_rates"
    DIRECT_THERMODYNAMIC_EQUILIBRIUM = "generated_direct_thermodynamic_equilibrium"


@dataclass(frozen=True, slots=True)
class EquilibriumRateAuthority:
    kind: str
    kf: object
    kr: object | None
    Keq: object | None
    derived_role: str
    source: str = ""
    dG_eq_J_per_mol: object | None = None
    reverse_std_ratio: object = 1.0
    has_explicit_keq_param: bool = False
    has_thermo_param: bool = False
    kf_editable: bool = True
    kr_editable: bool = True

    @property
    def consumes_keq(self) -> bool:
        return self.kind == EquilibriumRateAuthorityKind.KEQ

    def effective_reverse_rate(self, Keq: object | None = None, kf: object | None = None) -> object:
        if self.kind == EquilibriumRateAuthorityKind.KR:
            if self.kr is None:
                raise ValueError(AUTHORITY_ERROR)
            return self.kr
        keq_value = self.Keq if Keq is None else Keq
        if keq_value is None:
            raise ValueError(AUTHORITY_ERROR)
        return effective_reverse_rate_from_keq(self.kf if kf is None else kf, keq_value, self.reverse_std_ratio)

    def authored_role_is_editable(self, role: str) -> bool:
        role = str(role)
        if role == "kf":
            return bool(self.kf_editable)
        if role == "kr":
            return self.kind == EquilibriumRateAuthorityKind.KR and bool(self.kr_editable)
        if role == "Keq":
            return bool(self.has_explicit_keq_param)
        if role == "dG_eq":
            return self.kind == EquilibriumRateAuthorityKind.KEQ and self.dG_eq_J_per_mol is not None
        return False

    def authored_role_is_derived(self, role: str) -> bool:
        role = str(role)
        if role == "kr":
            return self.kind == EquilibriumRateAuthorityKind.KEQ
        if role == "Keq":
            return not bool(self.has_explicit_keq_param)
        return False

    def step_map_fields(self) -> dict[str, object]:
        return {
            "equilibrium_authority": {
                "kind": self.kind,
                "source": self.source,
                "derived_role": self.derived_role,
                "reverse_std_ratio": self.reverse_std_ratio,
                "has_explicit_keq_param": bool(self.has_explicit_keq_param),
                "has_thermo_param": bool(self.has_thermo_param),
                "editable": {
                    "kf": self.authored_role_is_editable("kf"),
                    "kr": self.authored_role_is_editable("kr"),
                    "Keq": self.authored_role_is_editable("Keq"),
                    "dG_eq": self.authored_role_is_editable("dG_eq"),
                },
                "derived": {
                    "kf": self.authored_role_is_derived("kf"),
                    "kr": self.authored_role_is_derived("kr"),
                    "Keq": self.authored_role_is_derived("Keq"),
                    "dG_eq": self.authored_role_is_derived("dG_eq"),
                },
            },
        }

    def identity_items(self) -> tuple[tuple[str, object], ...]:
        items: list[tuple[str, object]] = [
            ("kind", self.kind),
            ("source", self.source),
            ("kf", self.kf),
            ("reverse_std_ratio", self.reverse_std_ratio if self.consumes_keq else None),
        ]
        if self.kind == EquilibriumRateAuthorityKind.KR:
            items.append(("kr", self.kr))
        else:
            items.append(("Keq", self.Keq))
            items.append(("dG_eq_J_per_mol", self.dG_eq_J_per_mol))
        return tuple(items)

    def authored_symbolic_roles(self) -> tuple[str, ...]:
        roles = ["kf"]
        if self.kind == EquilibriumRateAuthorityKind.KR:
            roles.append("kr")
        elif self.has_explicit_keq_param:
            roles.append("Keq")
        return tuple(roles)


@dataclass(frozen=True, slots=True)
class EquilibriumAuthorityReadoutUpdate:
    name: str
    value: float
    editable: bool = False
    derived: bool = True


def effective_reverse_rate_from_keq(kf: object, Keq: object, reverse_std_ratio: object = 1.0) -> object:
    """Return the effective reverse rate for thermodynamic equilibrium authority."""
    return kf / (Keq * reverse_std_ratio)  # type: ignore[operator]


def json_safe_equilibrium_authority_payload(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): json_safe_equilibrium_authority_payload(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [json_safe_equilibrium_authority_payload(item) for item in value]
    if isinstance(value, set):
        return [json_safe_equilibrium_authority_payload(item) for item in sorted(value, key=lambda item: str(item))]
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    if math.isfinite(numeric):
        return numeric
    return str(value)


def _finite_float_or_none(value: object, *, temperature_K: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        if callable(value):
            if temperature_K is not None and bool(getattr(value, "accepts_runtime_temperature", False)):
                raw = value(float(temperature_K))
            else:
                raw = value()
        else:
            raw = value
        out = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _metadata_bool(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none", "null"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def generated_equilibrium_authority_kind(metadata: Mapping[str, Any] | None) -> str:
    meta = metadata or {}
    kind = str(meta.get(EquilibriumMetadataKeys.GENERATED_AUTHORITY) or "")
    if kind in {
        GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES,
        GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM,
    }:
        return kind
    return ""


def _generated_rate_model(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    meta = metadata or {}
    model = meta.get(EquilibriumMetadataKeys.GENERATED_RATE_MODEL)
    return model if isinstance(model, Mapping) else {}


def _required_positive_model_float(
    model: Mapping[str, Any],
    key: str,
    *,
    temperature_K: float | None = None,
) -> float | None:
    if key not in model:
        return None
    value = _finite_float_or_none(model.get(key), temperature_K=temperature_K)
    if value is None or value <= 0.0:
        return None
    return float(value)


def _generated_model_temperature(model: Mapping[str, Any]) -> float | None:
    temperature = _finite_float_or_none(model.get("temperature_K"))
    if temperature is None or temperature <= 0.0:
        return None
    return float(temperature)


def generated_transition_state_rate_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
    direction: str,
) -> float | None:
    meta = metadata or {}
    if generated_equilibrium_authority_kind(meta) != GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES:
        return None
    model = _generated_rate_model(meta)
    if str(model.get("type") or "") != GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES:
        return None
    if direction == "forward":
        dG_key = "dG_act_fwd_J_per_mol"
        degeneracy_key = "degeneracy_ratio_fwd"
        std_ratio_key = "std_ratio_fwd"
    elif direction == "reverse":
        dG_key = "dG_act_rev_J_per_mol"
        degeneracy_key = "degeneracy_ratio_rev"
        std_ratio_key = "std_ratio_rev"
    else:
        return None
    dG = _finite_float_or_none(model.get(dG_key), temperature_K=temperature_K)
    if dG is None:
        return None
    T = float(temperature_K)
    if not (math.isfinite(T) and T > 0.0):
        return None
    kappa = _required_positive_model_float(model, "kappa", temperature_K=temperature_K)
    degeneracy_ratio = _required_positive_model_float(model, degeneracy_key, temperature_K=temperature_K)
    std_ratio = _required_positive_model_float(model, std_ratio_key, temperature_K=temperature_K)
    if kappa is None or degeneracy_ratio is None or std_ratio is None:
        return None

    from kindred.core.constants import R, h, k_B

    return float(
        kappa
        * float(k_B)
        * T
        / float(h)
        * degeneracy_ratio
        * math.exp(-float(dG) / (float(R) * T))
        * std_ratio
    )


def generated_direct_forward_rate_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
) -> float | None:
    _ = temperature_K
    meta = metadata or {}
    if generated_equilibrium_authority_kind(meta) != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return None
    model = _generated_rate_model(meta)
    if str(model.get("type") or "") != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return None
    return _required_positive_model_float(model, "kf")


def generated_direct_keq_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
) -> float | None:
    meta = metadata or {}
    if generated_equilibrium_authority_kind(meta) != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return None
    model = _generated_rate_model(meta)
    if str(model.get("type") or "") != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return None
    dG = _finite_float_or_none(model.get("dG_eq_J_per_mol"), temperature_K=temperature_K)
    std_ratio = _required_positive_model_float(model, "std_ratio", temperature_K=temperature_K)
    T = float(temperature_K)
    if dG is None or std_ratio is None or not (math.isfinite(T) and T > 0.0):
        return None
    from kindred.core.kinetics import K_from_deltaG_eq

    return float(K_from_deltaG_eq(dG, T))


def generated_direct_reverse_rate_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
) -> float | None:
    meta = metadata or {}
    model = _generated_rate_model(meta)
    kf = generated_direct_forward_rate_from_metadata(meta, temperature_K=temperature_K)
    keq = generated_direct_keq_from_metadata(meta, temperature_K=temperature_K)
    std_ratio = _required_positive_model_float(model, "std_ratio", temperature_K=temperature_K)
    if kf is None or keq is None or std_ratio is None or abs(float(keq)) <= 1e-30:
        return None
    return float(effective_reverse_rate_from_keq(kf, keq, std_ratio))


def generated_equilibrium_forward_rate_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
) -> float | None:
    kind = generated_equilibrium_authority_kind(metadata)
    if kind == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return generated_direct_forward_rate_from_metadata(metadata, temperature_K=temperature_K)
    return generated_transition_state_rate_from_metadata(
        metadata,
        temperature_K=temperature_K,
        direction="forward",
    )


def generated_equilibrium_reverse_rate_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    temperature_K: float,
) -> float | None:
    kind = generated_equilibrium_authority_kind(metadata)
    if kind == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return generated_direct_reverse_rate_from_metadata(metadata, temperature_K=temperature_K)
    return generated_transition_state_rate_from_metadata(
        metadata,
        temperature_K=temperature_K,
        direction="reverse",
    )


def generated_transition_state_rate_model_is_complete(metadata: Mapping[str, Any] | None) -> bool:
    meta = metadata or {}
    if generated_equilibrium_authority_kind(meta) != GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES:
        return False
    model = _generated_rate_model(meta)
    temperature = _finite_float_or_none(model.get("temperature_K"))
    if temperature is None or temperature <= 0.0:
        return False
    return (
        generated_equilibrium_forward_rate_from_metadata(meta, temperature_K=temperature) is not None
        and generated_equilibrium_reverse_rate_from_metadata(meta, temperature_K=temperature) is not None
    )


def generated_direct_thermodynamic_equilibrium_model_is_complete(metadata: Mapping[str, Any] | None) -> bool:
    meta = metadata or {}
    if generated_equilibrium_authority_kind(meta) != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return False
    model = _generated_rate_model(meta)
    if str(model.get("type") or "") != GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return False
    temperature = _generated_model_temperature(model)
    if temperature is None:
        return False
    return (
        generated_direct_forward_rate_from_metadata(meta, temperature_K=temperature) is not None
        and generated_direct_keq_from_metadata(meta, temperature_K=temperature) is not None
        and generated_direct_reverse_rate_from_metadata(meta, temperature_K=temperature) is not None
    )


def _generated_model_error_for_kind(kind: str) -> str:
    if kind == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        return GENERATED_DIRECT_MODEL_ERROR
    return GENERATED_RATE_MODEL_ERROR


def plain_finite_float_or_none(value: object) -> float | None:
    """Return a finite float only for already-scalar values.

    Unlike `_finite_float_or_none`, this deliberately refuses callables and
    binding-like objects so construction code can avoid publishing stale derived
    raw fields for values that must remain runtime-resolved.
    """
    if value is None or callable(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def effective_equilibrium_forward_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    temperature = float(temperature_K)
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        generated_kind = generated_equilibrium_authority_kind(meta)
        if generated_kind:
            generated_forward = generated_equilibrium_forward_rate_from_metadata(
                meta,
                temperature_K=temperature,
            )
            generated_reverse = generated_equilibrium_reverse_rate_from_metadata(
                meta,
                temperature_K=temperature,
            )
            if generated_forward is None or generated_reverse is None:
                raise ValueError(_generated_model_error_for_kind(generated_kind))
            return float(generated_forward)
    return _finite_float_or_none(getattr(eq, "kf", None), temperature_K=temperature)


def effective_equilibrium_keq(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = float(temperature_K)
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        generated_kind = generated_equilibrium_authority_kind(meta)
        if generated_kind == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
            generated_keq = generated_direct_keq_from_metadata(meta, temperature_K=temperature)
            generated_forward = generated_equilibrium_forward_rate_from_metadata(meta, temperature_K=temperature)
            generated_reverse = generated_equilibrium_reverse_rate_from_metadata(meta, temperature_K=temperature)
            if generated_keq is None or generated_forward is None or generated_reverse is None:
                raise ValueError(GENERATED_DIRECT_MODEL_ERROR)
            return float(generated_keq)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        if (
            isinstance(meta, Mapping)
            and generated_equilibrium_authority_kind(meta)
            == GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES
        ):
            kf = generated_equilibrium_forward_rate_from_metadata(meta, temperature_K=temperature)
            kr = generated_equilibrium_reverse_rate_from_metadata(meta, temperature_K=temperature)
            if kf is None or kr is None or abs(kr) <= 1e-30:
                raise ValueError(GENERATED_RATE_MODEL_ERROR)
            return float(kf) / float(kr)
        kf = None
        kr = None
        if isinstance(meta, Mapping):
            kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
            kr = generated_equilibrium_reverse_rate_from_metadata(meta, temperature_K=temperature)
        if kr is None:
            kf = _finite_float_or_none(authority.kf, temperature_K=temperature)
            kr = _finite_float_or_none(authority.kr, temperature_K=temperature)
        if kf is not None and kr is not None and abs(kr) > 1e-30:
            return float(kf) / float(kr)
        return None

    if isinstance(meta, Mapping):
        if (
            authority.kind == EquilibriumRateAuthorityKind.KEQ
            and authority.dG_eq_J_per_mol is not None
            and not authority.has_explicit_keq_param
        ):
            dg_value = _finite_float_or_none(authority.dG_eq_J_per_mol, temperature_K=temperature)
            if dg_value is not None:
                from kindred.core.kinetics import K_from_deltaG_eq

                return float(K_from_deltaG_eq(dg_value, temperature))
        metadata_keq_source = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        if callable(metadata_keq_source):
            metadata_keq = _finite_float_or_none(metadata_keq_source, temperature_K=temperature)
            if metadata_keq is not None:
                return float(metadata_keq)
        dg_value = _finite_float_or_none(meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL), temperature_K=temperature)
        if dg_value is not None:
            from kindred.core.kinetics import K_from_deltaG_eq

            return float(K_from_deltaG_eq(dg_value, temperature))
        metadata_keq = _finite_float_or_none(metadata_keq_source, temperature_K=temperature)
        if metadata_keq is not None:
            return float(metadata_keq)
    direct_keq = _finite_float_or_none(getattr(eq, "Keq", None), temperature_K=temperature)
    if direct_keq is not None:
        return float(direct_keq)
    kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
    kr = _finite_float_or_none(getattr(eq, "kr", None), temperature_K=temperature)
    if kf is not None and kr is not None and abs(kr) > 1e-30:
        return float(kf) / float(kr)
    return None


def effective_equilibrium_reverse_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = float(temperature_K)
    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        generated_kind = generated_equilibrium_authority_kind(meta)
        generated_rate = generated_equilibrium_reverse_rate_from_metadata(
            meta,
            temperature_K=temperature,
        )
        if generated_rate is not None:
            return float(generated_rate)
        if generated_kind:
            raise ValueError(_generated_model_error_for_kind(generated_kind))
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        return _finite_float_or_none(authority.kr, temperature_K=temperature)
    kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
    keq = effective_equilibrium_keq(eq, temperature_K=temperature)
    if kf is None or keq is None or abs(keq) <= 1e-30:
        return None
    value = authority.effective_reverse_rate(Keq=keq, kf=kf)
    return _finite_float_or_none(value, temperature_K=temperature)


def authority_readout_updates_from_step_entry(
    entry: Mapping[str, Any],
    current_values: Mapping[str, object],
) -> tuple[EquilibriumAuthorityReadoutUpdate, ...]:
    if str(entry.get("kind") or "") != "equilibrium":
        return ()
    try:
        step_index = int(entry.get("step_index"))
    except (TypeError, ValueError):
        return ()
    authority = authority_fields_from_step_entry(entry)
    if not authority:
        return ()
    editable = authority.get("editable")
    if not isinstance(editable, Mapping):
        editable = {}
    derived_role = str(authority.get("derived_role") or "")
    try:
        reverse_std_ratio = float(authority.get("reverse_std_ratio") or 1.0)
    except (TypeError, ValueError, OverflowError):
        reverse_std_ratio = 1.0
    if not math.isfinite(reverse_std_ratio) or reverse_std_ratio <= 0.0:
        reverse_std_ratio = 1.0

    kf_key = f"kf{step_index}"
    kr_key = f"kr{step_index}"
    keq_key = f"Keq{step_index}"

    def _current(name: str) -> float | None:
        return _finite_float_or_none(current_values.get(name))

    def _readout(name: str, value: float) -> EquilibriumAuthorityReadoutUpdate:
        return EquilibriumAuthorityReadoutUpdate(
            name=name,
            value=float(value),
            editable=False,
            derived=True,
        )

    updates: list[EquilibriumAuthorityReadoutUpdate] = []
    keq_editable = bool(editable.get("Keq"))
    k_val = _current(keq_key) if (keq_editable or bool(authority.get("has_thermo_param"))) else None
    if derived_role == "Keq" or (k_val is None and not keq_editable):
        kf_val = _current(kf_key)
        kr_val = _current(kr_key)
        if kf_val is None or kr_val is None or abs(kr_val) <= 1e-30:
            return ()
        k_val = float(kf_val) / float(kr_val)
        updates.append(_readout(keq_key, k_val))
        return tuple(updates)
    if k_val is None or abs(k_val) <= 1e-30:
        return tuple(updates)
    if not keq_editable:
        updates.append(_readout(keq_key, k_val))

    if derived_role == "kf":
        kr_val = _current(kr_key)
        if kr_val is not None:
            updates.append(_readout(kf_key, float(kr_val) * float(k_val) * reverse_std_ratio))
    elif derived_role == "kr":
        kf_val = _current(kf_key)
        if kf_val is not None:
            updates.append(_readout(kr_key, effective_reverse_rate_from_keq(kf_val, k_val, reverse_std_ratio)))

    return tuple(updates)


def _metadata_has_thermo_authority(metadata: Mapping[str, Any] | None) -> bool:
    meta = metadata or {}
    return (
        meta.get(EquilibriumMetadataKeys.KEQ_INPUT) is not None
        or meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) is not None
    )


def _is_generated_equilibrium_metadata(metadata: Mapping[str, Any] | None) -> bool:
    return bool(generated_equilibrium_authority_kind(metadata))


def _coerce_context(context: EquilibriumRateInputContext | str | None) -> EquilibriumRateInputContext:
    if context is None:
        return EquilibriumRateInputContext.PUBLIC
    if isinstance(context, EquilibriumRateInputContext):
        return context
    return EquilibriumRateInputContext(str(context))


def _metadata_dg_eq(metadata: Mapping[str, Any] | None) -> object | None:
    meta = metadata or {}
    return meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)


def _metadata_keq_input(metadata: Mapping[str, Any] | None) -> object | None:
    meta = metadata or {}
    return meta.get(EquilibriumMetadataKeys.KEQ_INPUT)


def _metadata_reverse_std_ratio(metadata: Mapping[str, Any] | None) -> object:
    meta = metadata or {}
    value = meta.get("std_ratio")
    if value is None:
        return 1.0
    return value


def _metadata_user_flags(metadata: Mapping[str, Any] | None) -> tuple[bool, bool]:
    meta = metadata or {}
    return (
        bool(meta.get(EquilibriumMetadataKeys.USER_PROVIDED_KF)),
        bool(meta.get(EquilibriumMetadataKeys.USER_PROVIDED_KR)),
    )


def validate_equilibrium_rate_authority_flags(
    *,
    has_kf: bool,
    has_kr: bool,
    has_Keq: bool,
    has_dG_eq: bool,
) -> None:
    reverse_authority_count = int(bool(has_kr)) + int(bool(has_Keq)) + int(bool(has_dG_eq))
    if not bool(has_kf) or reverse_authority_count != 1:
        raise ValueError(AUTHORITY_ERROR)


def validate_equilibrium_rate_authority_values(
    *,
    kf: object | None,
    kr: object | None,
    Keq: object | None,
    metadata: Mapping[str, Any] | None = None,
    context: EquilibriumRateInputContext | str | None = None,
) -> None:
    normalize_equilibrium_rate_authority(kf=kf, kr=kr, Keq=Keq, metadata=metadata, context=context)


def normalize_equilibrium_rate_authority(
    *,
    kf: object | None,
    kr: object | None,
    Keq: object | None,
    metadata: Mapping[str, Any] | None = None,
    context: EquilibriumRateInputContext | str | None = None,
) -> EquilibriumRateAuthority:
    ctx = _coerce_context(context)
    meta = metadata or {}
    metadata_keq = _metadata_keq_input(meta)
    metadata_dg = _metadata_dg_eq(meta)
    user_kf, user_kr = _metadata_user_flags(meta)
    thermo_count = int(Keq is not None or metadata_keq is not None) + int(metadata_dg is not None)
    has_thermo = thermo_count > 0
    generated_kind = generated_equilibrium_authority_kind(meta)
    if generated_kind == GeneratedEquilibriumAuthorityKind.TRANSITION_STATE_BARRIER_RATES:
        if not generated_transition_state_rate_model_is_complete(meta):
            raise ValueError(GENERATED_RATE_MODEL_ERROR)
        model = _generated_rate_model(meta)
        model_temperature = _generated_model_temperature(model)
        if model_temperature is None:
            raise ValueError(GENERATED_RATE_MODEL_ERROR)
        model_kf = generated_equilibrium_forward_rate_from_metadata(meta, temperature_K=float(model_temperature))
        model_kr = generated_equilibrium_reverse_rate_from_metadata(meta, temperature_K=float(model_temperature))
        if model_kf is None or model_kr is None:
            raise ValueError(GENERATED_RATE_MODEL_ERROR)
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR,
            kf=float(model_kf),
            kr=float(model_kr),
            Keq=None,
            derived_role="Keq",
            source=generated_kind,
            dG_eq_J_per_mol=metadata_dg,
            reverse_std_ratio=_metadata_reverse_std_ratio(meta),
            has_thermo_param=bool(has_thermo),
            kf_editable=False,
            kr_editable=False,
        )
    if generated_kind == GeneratedEquilibriumAuthorityKind.DIRECT_THERMODYNAMIC_EQUILIBRIUM:
        if not generated_direct_thermodynamic_equilibrium_model_is_complete(meta):
            raise ValueError(GENERATED_DIRECT_MODEL_ERROR)
        model = _generated_rate_model(meta)
        model_temperature = _generated_model_temperature(model)
        model_kf = generated_direct_forward_rate_from_metadata(
            meta,
            temperature_K=float(model_temperature if model_temperature is not None else 298.15),
        )
        model_keq = generated_direct_keq_from_metadata(
            meta,
            temperature_K=float(model_temperature if model_temperature is not None else 298.15),
        )
        model_dg = _finite_float_or_none(model.get("dG_eq_J_per_mol"))
        model_std_ratio = _required_positive_model_float(model, "std_ratio")
        if model_kf is None or model_keq is None or model_dg is None or model_std_ratio is None:
            raise ValueError(GENERATED_DIRECT_MODEL_ERROR)
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KEQ,
            kf=float(model_kf),
            kr=kr,
            Keq=float(model_keq),
            derived_role="kr",
            source=generated_kind,
            dG_eq_J_per_mol=float(model_dg),
            reverse_std_ratio=float(model_std_ratio),
            has_explicit_keq_param=False,
            has_thermo_param=True,
            kf_editable=False,
            kr_editable=False,
        )
    has_kf = kf is not None
    has_kr = kr is not None
    if kf is None:
        raise ValueError(AUTHORITY_ERROR)
    if thermo_count > 1 and ctx == EquilibriumRateInputContext.PUBLIC:
        raise ValueError(AUTHORITY_ERROR)

    if ctx == EquilibriumRateInputContext.PUBLIC:
        validate_equilibrium_rate_authority_flags(
            has_kf=has_kf,
            has_kr=has_kr,
            has_Keq=bool(Keq is not None or metadata_keq is not None),
            has_dG_eq=bool(metadata_dg is not None),
        )
    elif ctx == EquilibriumRateInputContext.NORMALIZED_PUBLIC:
        if not user_kf or user_kr or not has_thermo:
            validate_equilibrium_rate_authority_flags(
                has_kf=has_kf,
                has_kr=has_kr,
                has_Keq=bool(Keq is not None or metadata_keq is not None),
                has_dG_eq=bool(metadata_dg is not None),
            )
    elif ctx == EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE:
        if not has_thermo:
            raise ValueError(AUTHORITY_ERROR)
    else:
        raise ValueError(AUTHORITY_ERROR)

    if has_kr and not has_thermo:
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR,
            kf=kf,
            kr=kr,
            Keq=None,
            derived_role="Keq",
            source=ctx.value,
        )

    if has_kr and has_thermo and ctx == EquilibriumRateInputContext.PUBLIC:
        raise ValueError(AUTHORITY_ERROR)
    if has_kr and has_thermo and ctx == EquilibriumRateInputContext.NORMALIZED_PUBLIC and user_kr:
        raise ValueError(AUTHORITY_ERROR)

    keq_source = metadata_keq if metadata_keq is not None else Keq
    if keq_source is None:
        keq_source = Keq
    if keq_source is None and metadata_dg is None:
        raise ValueError(AUTHORITY_ERROR)
    return EquilibriumRateAuthority(
        kind=EquilibriumRateAuthorityKind.KEQ,
        kf=kf,
        kr=kr,
        Keq=keq_source,
        derived_role="kr",
        source=ctx.value,
        dG_eq_J_per_mol=metadata_dg,
        reverse_std_ratio=_metadata_reverse_std_ratio(meta),
        has_explicit_keq_param=bool(metadata_keq is not None or (Keq is not None and metadata_dg is None)),
        has_thermo_param=bool(has_thermo),
        kf_editable=ctx != EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE,
        kr_editable=ctx != EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE,
    )


def normalize_existing_equilibrium_rate_authority(eq: object) -> EquilibriumRateAuthority:
    meta = getattr(eq, "metadata", {}) or {}
    context: EquilibriumRateInputContext | str = EquilibriumRateInputContext.PUBLIC
    if isinstance(meta, Mapping):
        if str(meta.get(_AUTHORITY_SOURCE_METADATA_KEY) or "") == EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE.value:
            context = EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE
        elif (
            (_metadata_has_thermo_authority(meta) or getattr(eq, "Keq", None) is not None)
            and not bool(meta.get(EquilibriumMetadataKeys.USER_PROVIDED_KR))
        ):
            context = EquilibriumRateInputContext.NORMALIZED_PUBLIC
    return normalize_equilibrium_rate_authority(
        kf=getattr(eq, "kf", None),
        kr=getattr(eq, "kr", None),
        Keq=getattr(eq, "Keq", None),
        metadata=meta if isinstance(meta, Mapping) else {},
        context=context,
    )


def _source_metadata_mapping(source: object) -> Mapping[str, Any]:
    if isinstance(source, Mapping) and "equilibrium_authority" not in source:
        return source
    meta = getattr(source, "metadata", {}) or {}
    return meta if isinstance(meta, Mapping) else {}


def _source_authority_fields(source: object) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        fields = authority_fields_from_step_entry(source)
        if fields:
            return fields
    return {}


def _generated_fast_runtime_source(source: object, metadata: Mapping[str, Any]) -> str:
    fields = _source_authority_fields(source)
    if fields:
        return str(fields.get("source") or "")
    if not isinstance(source, Mapping):
        try:
            return str(normalize_existing_equilibrium_rate_authority(source).source)
        except ValueError:
            return ""
    return str(metadata.get(_AUTHORITY_SOURCE_METADATA_KEY) or "")


def _generated_fast_runtime_has_thermo_authority(source: object, metadata: Mapping[str, Any]) -> bool:
    fields = _source_authority_fields(source)
    if fields:
        return bool(fields.get("has_thermo_param"))
    if not isinstance(source, Mapping):
        try:
            authority = normalize_existing_equilibrium_rate_authority(source)
        except ValueError:
            return False
        return authority.kind == EquilibriumRateAuthorityKind.KEQ and bool(authority.has_thermo_param)
    return metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) is not None


def generated_fast_equilibrium_runtime_parameter_name(source: object) -> str | None:
    metadata = _source_metadata_mapping(source)
    if _generated_fast_runtime_source(source, metadata) != EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE.value:
        return None
    if not _generated_fast_runtime_has_thermo_authority(source, metadata):
        return None
    cm_id_source: object | None = None
    if isinstance(source, Mapping):
        cm_id_source = source.get("cm_id")
    if cm_id_source is None:
        cm_id_source = metadata.get("cm_id")
    cm_id = str(cm_id_source or "").strip()
    if not cm_id:
        return None
    return f"{GENERATED_FAST_EQUILIBRIUM_RUNTIME_PARAMETER_PREFIX}{cm_id}"


def equilibrium_is_generated_authority_output(source: object) -> bool:
    metadata = _source_metadata_mapping(source)
    return bool(
        generated_equilibrium_authority_kind(metadata)
        or generated_fast_equilibrium_runtime_parameter_name(source) is not None
    )


def json_safe_equilibrium_metadata_payload(source: object) -> dict[str, object]:
    payload = json_safe_equilibrium_authority_payload(_source_metadata_mapping(source))
    if not isinstance(payload, Mapping):
        return {}
    out = {str(key): payload[key] for key in payload}
    if equilibrium_is_generated_authority_output(source):
        return {
            key: value
            for key, value in out.items()
            if key not in _GENERATED_SCALAR_MIRROR_METADATA_KEYS
        }
    return out


def authority_fields_from_step_entry(entry: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        return {}
    value = entry.get("equilibrium_authority")
    return value if isinstance(value, Mapping) else {}


def step_entry_authored_role_is_editable(entry: Mapping[str, Any] | None, role: str) -> bool | None:
    fields = authority_fields_from_step_entry(entry)
    editable = fields.get("editable")
    if isinstance(editable, Mapping) and role in editable:
        return bool(editable[role])
    return None


def require_step_entry_authored_role_is_editable(
    entry: Mapping[str, Any] | None,
    role: str,
    *,
    parameter_name: str | None = None,
) -> None:
    editable = step_entry_authored_role_is_editable(entry, role)
    name = str(parameter_name or role)
    if editable is True:
        return
    if editable is False:
        raise ValueError(f"{name} is not editable under normalized equilibrium authority.")
    raise ValueError(f"{name} is missing normalized equilibrium_authority editability metadata.")


def step_entry_authored_role_is_derived(entry: Mapping[str, Any] | None, role: str) -> bool | None:
    fields = authority_fields_from_step_entry(entry)
    derived = fields.get("derived")
    if isinstance(derived, Mapping) and role in derived:
        return bool(derived[role])
    return None


def public_text_equilibrium_role_editable(
    *,
    has_kr: bool,
    has_Keq: bool,
    has_dG_eq: bool,
    role: str,
) -> bool:
    """Return editability for a public DSL equilibrium line already parsed as text tokens."""
    role = str(role)
    has_thermo = bool(has_Keq or has_dG_eq)
    if role == "kf":
        return True
    if role == "kr":
        return bool(has_kr and not has_thermo)
    if role == "Keq":
        return bool(has_Keq)
    if role == "dG_eq":
        return bool(has_dG_eq)
    return False

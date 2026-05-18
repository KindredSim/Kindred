from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from kindred.core.mechanism_metadata import EquilibriumMetadataKeys

AUTHORITY_ERROR = "equilibrium requires kf and exactly one of kr or Keq/dG_eq"


class EquilibriumRateAuthorityKind:
    KR = "kr_authoritative"
    KEQ = "Keq_authoritative"


class EquilibriumRateInputContext(str, Enum):
    PUBLIC = "public"
    NORMALIZED_PUBLIC = "normalized_public"
    GENERATED_COMPUTATIONAL_MODE = "generated_computational_mode"
    GENERATED_STATE_NETWORK = "generated_state_network"


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

    def role_editability(self, role: str) -> bool:
        role = str(role)
        if role == "kf":
            return True
        if role == "kr":
            return self.kind == EquilibriumRateAuthorityKind.KR
        if role == "Keq":
            return bool(self.has_explicit_keq_param)
        if role == "dG_eq":
            return self.kind == EquilibriumRateAuthorityKind.KEQ and self.dG_eq_J_per_mol is not None
        return False

    def role_derived(self, role: str) -> bool:
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
                    "kf": self.role_editability("kf"),
                    "kr": self.role_editability("kr"),
                    "Keq": self.role_editability("Keq"),
                    "dG_eq": self.role_editability("dG_eq"),
                },
                "derived": {
                    "kf": self.role_derived("kf"),
                    "kr": self.role_derived("kr"),
                    "Keq": self.role_derived("Keq"),
                    "dG_eq": self.role_derived("dG_eq"),
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


@dataclass(frozen=True, slots=True)
class EquilibriumAuthorityReadoutUpdate:
    name: str
    value: float
    editable: bool = False
    derived: bool = True


def effective_reverse_rate_from_keq(kf: object, Keq: object, reverse_std_ratio: object = 1.0) -> object:
    """Return the effective reverse rate for thermodynamic equilibrium authority."""
    return kf / (Keq * reverse_std_ratio)  # type: ignore[operator]


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        raw = value() if callable(value) else value
        out = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


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


def effective_equilibrium_keq(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        kf = _finite_float_or_none(authority.kf)
        kr = _finite_float_or_none(authority.kr)
        if kf is not None and kr is not None and abs(kr) > 1e-30:
            return float(kf) / float(kr)
        return None

    meta = getattr(eq, "metadata", {}) or {}
    if isinstance(meta, Mapping):
        dg_value = _finite_float_or_none(meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL))
        if dg_value is not None:
            from kindred.core.kinetics import K_from_deltaG_eq

            return float(K_from_deltaG_eq(dg_value, float(temperature_K)))
        metadata_keq = _finite_float_or_none(meta.get(EquilibriumMetadataKeys.KEQ_INPUT))
        if metadata_keq is not None:
            return float(metadata_keq)
    direct_keq = _finite_float_or_none(getattr(eq, "Keq", None))
    if direct_keq is not None:
        return float(direct_keq)
    kf = _finite_float_or_none(getattr(eq, "kf", None))
    kr = _finite_float_or_none(getattr(eq, "kr", None))
    if kf is not None and kr is not None and abs(kr) > 1e-30:
        return float(kf) / float(kr)
    return None


def effective_equilibrium_reverse_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        return _finite_float_or_none(authority.kr)
    kf = _finite_float_or_none(getattr(eq, "kf", None))
    keq = effective_equilibrium_keq(eq, temperature_K=float(temperature_K))
    if kf is None or keq is None or abs(keq) <= 1e-30:
        return None
    value = authority.effective_reverse_rate(Keq=keq, kf=kf)
    return _finite_float_or_none(value)


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
    meta = metadata or {}
    source = str(meta.get("source") or "")
    return source in {"state_network", "state_network_direct"}


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


def validate_generated_computational_legacy_kr(
    *,
    kf: float,
    kr: float,
    Keq: float,
    reverse_std_ratio: float,
) -> None:
    derived_kr = float(effective_reverse_rate_from_keq(float(kf), float(Keq), float(reverse_std_ratio)))
    if not math.isclose(float(kr), derived_kr, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("legacy Computational Mode kr is inconsistent with kf, dG_eq, and cm_std_ratio")


def derive_generated_computational_std_ratio_from_legacy_kr(
    *,
    kf: float,
    kr: float,
    Keq: float,
) -> float:
    value = float(kf) / (float(kr) * float(Keq))
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError("legacy Computational Mode kr is inconsistent with kf and dG_eq")
    return float(value)


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
    has_kf = kf is not None
    has_kr = kr is not None
    if kf is None:
        raise ValueError(AUTHORITY_ERROR)

    thermo_count = int(Keq is not None or metadata_keq is not None) + int(metadata_dg is not None)
    has_thermo = thermo_count > 0
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
    elif ctx == EquilibriumRateInputContext.GENERATED_STATE_NETWORK:
        if not has_kr and not has_thermo:
            raise ValueError(AUTHORITY_ERROR)
    else:
        raise ValueError(AUTHORITY_ERROR)

    if ctx == EquilibriumRateInputContext.GENERATED_STATE_NETWORK and has_kr and user_kr and not has_thermo:
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR,
            kf=kf,
            kr=kr,
            Keq=None,
            derived_role="Keq",
            source=ctx.value,
        )

    if has_kr and not has_thermo:
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR,
            kf=kf,
            kr=kr,
            Keq=None,
            derived_role="Keq",
            source=ctx.value,
        )

    if has_kr and has_thermo and ctx == EquilibriumRateInputContext.GENERATED_STATE_NETWORK and user_kr:
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR,
            kf=kf,
            kr=kr,
            Keq=None,
            derived_role="Keq",
            source=ctx.value,
            dG_eq_J_per_mol=metadata_dg,
            reverse_std_ratio=_metadata_reverse_std_ratio(meta),
            has_thermo_param=bool(has_thermo),
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
    )


def normalize_existing_equilibrium_rate_authority(eq: object) -> EquilibriumRateAuthority:
    meta = getattr(eq, "metadata", {}) or {}
    context: EquilibriumRateInputContext | str = EquilibriumRateInputContext.PUBLIC
    if isinstance(meta, Mapping):
        if str(meta.get("source") or "") == "state_network" or _is_generated_equilibrium_metadata(meta):
            context = EquilibriumRateInputContext.GENERATED_STATE_NETWORK
        elif str(meta.get("authority_source") or "") == EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE.value:
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


def authority_fields_from_step_entry(entry: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        return {}
    value = entry.get("equilibrium_authority")
    return value if isinstance(value, Mapping) else {}


def step_entry_role_editable(entry: Mapping[str, Any] | None, role: str) -> bool | None:
    fields = authority_fields_from_step_entry(entry)
    editable = fields.get("editable")
    if isinstance(editable, Mapping) and role in editable:
        return bool(editable[role])
    return None


def require_step_entry_role_editable(
    entry: Mapping[str, Any] | None,
    role: str,
    *,
    parameter_name: str | None = None,
) -> None:
    editable = step_entry_role_editable(entry, role)
    name = str(parameter_name or role)
    if editable is True:
        return
    if editable is False:
        raise ValueError(f"{name} is not editable under normalized equilibrium authority.")
    raise ValueError(f"{name} is missing normalized equilibrium_authority editability metadata.")


def step_entry_role_derived(entry: Mapping[str, Any] | None, role: str) -> bool | None:
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

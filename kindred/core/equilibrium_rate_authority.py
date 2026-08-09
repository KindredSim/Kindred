from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Iterable, Mapping

from kindred.core.constants import R
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.thermodynamics import K_from_deltaG_eq
from kindred.core.simulator.errors import DSLError

AUTHORITY_ERROR = "equilibrium requires kf and exactly one of kr or Keq/dG_eq"


def _positive_finite(value: object, *, label: str) -> float:
    if type(value) is bool:
        raise DSLError(f"{label} must be positive and finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise DSLError(f"{label} must be positive and finite") from exc
    if not number > 0.0 or not math.isfinite(number):
        raise DSLError(f"{label} must be positive and finite")
    return number


def choose_fast_equilibrium_rate(
    explicit_rates: Iterable[float] | None = None,
) -> float:
    """Return the authored forward coefficient without inventing rate meaning.

    The equilibrium parser supplies authored rates in directional order, so the
    first value is the explicit forward coefficient.  Reverse coefficients do
    not participate in choosing or rescaling it.
    """

    rates = tuple(explicit_rates or ())
    if not rates:
        raise DSLError("equilibrium requires an explicit positive finite kf")
    return _positive_finite(rates[0], label="kf")


@dataclass(frozen=True, slots=True)
class FastEquilibriumRates:
    kf: float
    kr: float
    Keq: float
    k_fast: float
    source: str


def derive_fast_equilibrium_rates(
    *,
    Keq: float | None = None,
    dG_eq_J_per_mol: float | None = None,
    temperature_K: float = 298.15,
    explicit_rates: Iterable[float] | None = None,
) -> FastEquilibriumRates:
    """Derive one fast-equilibrium rate pair."""

    if Keq is None and dG_eq_J_per_mol is None:
        raise DSLError(
            "derive_fast_equilibrium_rates requires Keq or "
            "dG_eq_J_per_mol"
        )
    if Keq is None:
        try:
            delta_g = float(dG_eq_J_per_mol)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DSLError("dG_eq_J_per_mol must be finite") from exc
        if not math.isfinite(delta_g):
            raise DSLError("dG_eq_J_per_mol must be finite")
        equilibrium_constant = K_from_deltaG_eq(
            delta_g,
            _positive_finite(temperature_K, label="temperature_K"),
        )
        source = "derived(ΔG°)"
    else:
        equilibrium_constant = _positive_finite(Keq, label="Keq")
        source = "derived(Keq)"
    forward = choose_fast_equilibrium_rate(explicit_rates)
    reverse = float(
        effective_reverse_rate_from_keq(
            forward,
            equilibrium_constant,
        )
    )
    return FastEquilibriumRates(
        kf=forward,
        kr=reverse,
        Keq=equilibrium_constant,
        k_fast=forward,
        source=source,
    )


class EquilibriumRateAuthorityKind:
    KR = "kr_authoritative"
    KEQ = "Keq_authoritative"


class EquilibriumRateInputContext(str, Enum):
    PUBLIC = "public"
    NORMALIZED_PUBLIC = "normalized_public"


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

@dataclass(frozen=True, slots=True)
class EquilibriumRateSymbolicRelations:
    """Canonical public relations for one accepted reversible step."""

    kf: object
    kr: object
    Keq: object


def equilibrium_rate_symbolic_relations(
    equilibrium: object,
    *,
    forward_relation: object,
    temperature: object,
    standard_conc_M: float,
    literal: Callable[[float], object],
    exponential: Callable[[object], object],
) -> EquilibriumRateSymbolicRelations:
    """Project accepted equilibrium authority without choosing a backend.

    Explicit ``kf + kr`` is standard-state inert.  Thermodynamic authority
    consumes the accepted standard concentration only at the directional
    exponent that owns the missing reverse coefficient.
    """

    if not callable(literal):
        raise TypeError("literal must be callable.")
    if not callable(exponential):
        raise TypeError("exponential must be callable.")

    def finite_scalar(
        value: object,
        *,
        label: str,
        positive: bool,
    ) -> float:
        try:
            raw = value() if callable(value) else value
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            requirement = "positive and finite" if positive else "finite"
            raise ValueError(f"{label} must be {requirement}.") from exc
        if not math.isfinite(number) or (positive and number <= 0.0):
            requirement = "positive and finite" if positive else "finite"
            raise ValueError(f"{label} must be {requirement}.")
        return float(number)

    authority = normalize_existing_equilibrium_rate_authority(equilibrium)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        reverse_value = finite_scalar(
            authority.kr,
            label="equilibrium kr",
            positive=True,
        )
        reverse_relation = literal(reverse_value)
        return EquilibriumRateSymbolicRelations(
            kf=forward_relation,
            kr=reverse_relation,
            Keq=forward_relation / reverse_relation,  # type: ignore[operator]
        )

    if authority.has_explicit_keq_param:
        keq_relation = literal(
            finite_scalar(
                authority.Keq,
                label="equilibrium Keq",
                positive=True,
            )
        )
    elif authority.dG_eq_J_per_mol is not None:
        delta_g = finite_scalar(
            authority.dG_eq_J_per_mol,
            label="equilibrium dG_eq_J_per_mol",
            positive=False,
        )
        keq_relation = exponential(
            -literal(delta_g)
            / (literal(float(R)) * temperature)  # type: ignore[operator]
        )
    else:
        keq_relation = literal(
            finite_scalar(
                authority.Keq,
                label="equilibrium Keq",
                positive=True,
            )
        )

    standard = finite_scalar(
        standard_conc_M,
        label="equilibrium standard_conc_M",
        positive=True,
    )
    forward_order = math.fsum(
        float(value)
        for value in dict(
            getattr(equilibrium, "stoich_forward", {}) or {}
        ).values()
    )
    reverse_order = math.fsum(
        float(value)
        for value in dict(
            getattr(equilibrium, "stoich_back", {}) or {}
        ).values()
    )
    standard_exponent = float(forward_order - reverse_order)
    if not math.isfinite(standard_exponent):
        raise ValueError(
            "equilibrium standard-state exponent must be finite."
        )
    reverse_relation = forward_relation / keq_relation  # type: ignore[operator]
    if standard_exponent != 0.0:
        reverse_relation *= literal(standard) ** literal(  # type: ignore[operator]
            standard_exponent
        )
    return EquilibriumRateSymbolicRelations(
        kf=forward_relation,
        kr=reverse_relation,
        Keq=keq_relation,
    )


def effective_reverse_rate_from_keq(kf: object, Keq: object, reverse_std_ratio: object = 1.0) -> object:
    """Return the missing reverse coefficient from thermodynamic authority.

    Concrete coefficient inputs are required to be positive finite reals. The
    symbolic path remains available for the algebra layer, whose values are not
    concrete coefficients at this boundary.
    """

    def concrete(value: object, *, label: str) -> float | None:
        if type(value) is bool or isinstance(value, str):
            raise ValueError(f"{label} must be a positive finite real")
        if callable(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label} must be a positive finite real")
        return float(number)

    kf_value = concrete(kf, label="kf")
    keq_value = concrete(Keq, label="Keq")
    standard_ratio = concrete(reverse_std_ratio, label="reverse_std_ratio")
    if kf_value is None or keq_value is None or standard_ratio is None:
        return kf / (Keq * reverse_std_ratio)  # type: ignore[operator]
    try:
        # Combine binary exponents separately so a representable reverse rate
        # is not lost to overflow/underflow in the intermediate Keq*C0 term.
        kf_mantissa, kf_exponent = math.frexp(kf_value)
        keq_mantissa, keq_exponent = math.frexp(keq_value)
        ratio_mantissa, ratio_exponent = math.frexp(standard_ratio)
        scaled_mantissa = (
            kf_mantissa / keq_mantissa / ratio_mantissa
        )
        reverse = math.ldexp(
            scaled_mantissa,
            kf_exponent - keq_exponent - ratio_exponent,
        )
    except OverflowError as exc:
        raise ValueError("thermodynamic reverse coefficient overflows the finite real domain") from exc
    if not math.isfinite(reverse) or reverse <= 0.0:
        raise ValueError("thermodynamic reverse coefficient must be a positive finite real")
    return float(reverse)


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


def plain_finite_float_or_none(value: object) -> float | None:
    """Return a finite float only for already-scalar values.

    Unlike `_finite_float_or_none`, this deliberately refuses callables and
    binding-like objects so construction code can avoid publishing stale derived
    raw fields for values that must remain runtime-resolved.
    """
    if value is None or callable(value) or type(value) is bool:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def effective_equilibrium_forward_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    return _finite_float_or_none(
        getattr(eq, "kf", None),
        temperature_K=float(temperature_K),
    )


def effective_equilibrium_keq(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = float(temperature_K)
    meta = getattr(eq, "metadata", {}) or {}
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        # Explicit kf+kr has no writable Keq authority, but its ratio remains
        # a useful derived readout.  It is deliberately independent of C0:
        # the explicit-rate basis owns both directional coefficients unchanged.
        kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
        kr = _finite_float_or_none(authority.kr, temperature_K=temperature)
        if kf is None or kr is None or kf <= 0.0 or kr <= 0.0:
            return None
        try:
            derived = float(kf) / float(kr)
        except (OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(derived) or derived <= 0.0:
            return None
        return derived

    if isinstance(meta, Mapping):
        if (
            authority.kind == EquilibriumRateAuthorityKind.KEQ
            and authority.dG_eq_J_per_mol is not None
            and not authority.has_explicit_keq_param
        ):
            dg_value = _finite_float_or_none(authority.dG_eq_J_per_mol, temperature_K=temperature)
            if dg_value is not None:
                return float(K_from_deltaG_eq(dg_value, temperature))
        metadata_keq_source = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
        if callable(metadata_keq_source):
            metadata_keq = _finite_float_or_none(metadata_keq_source, temperature_K=temperature)
            if metadata_keq is not None:
                return float(metadata_keq)
        dg_value = _finite_float_or_none(meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL), temperature_K=temperature)
        if dg_value is not None:
            return float(K_from_deltaG_eq(dg_value, temperature))
        metadata_keq = _finite_float_or_none(metadata_keq_source, temperature_K=temperature)
        if metadata_keq is not None:
            return float(metadata_keq)
    direct_keq = _finite_float_or_none(getattr(eq, "Keq", None), temperature_K=temperature)
    if direct_keq is not None:
        return float(direct_keq)
    kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
    kr = _finite_float_or_none(getattr(eq, "kr", None), temperature_K=temperature)
    if kf is not None and kr is not None and kf > 0.0 and kr > 0.0:
        try:
            derived = float(kf) / float(kr)
        except (OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(derived) and derived > 0.0:
            return derived
    return None

def effective_equilibrium_reverse_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = float(temperature_K)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        return _finite_float_or_none(authority.kr, temperature_K=temperature)
    kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
    keq = effective_equilibrium_keq(eq, temperature_K=temperature)
    if kf is None or keq is None or kf <= 0.0 or keq <= 0.0:
        return None
    value = authority.effective_reverse_rate(Keq=keq, kf=kf)
    return _finite_float_or_none(value, temperature_K=temperature)


def _metadata_has_thermo_authority(metadata: Mapping[str, Any] | None) -> bool:
    meta = metadata or {}
    return (
        meta.get(EquilibriumMetadataKeys.KEQ_INPUT) is not None
        or meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL) is not None
    )


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


def _validate_concrete_positive_coefficient(value: object | None, *, label: str) -> None:
    """Reject invalid concrete coefficients while preserving symbolic authority."""
    if value is None or callable(value):
        return
    if type(value) is bool or isinstance(value, str):
        raise ValueError(f"{label} must be a positive finite real")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite real")


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
    _validate_concrete_positive_coefficient(kf, label="kf")
    _validate_concrete_positive_coefficient(kr, label="kr")
    _validate_concrete_positive_coefficient(Keq, label="Keq")
    _validate_concrete_positive_coefficient(metadata_keq, label="Keq_input")
    user_kf, user_kr = _metadata_user_flags(meta)
    thermo_count = int(Keq is not None or metadata_keq is not None) + int(metadata_dg is not None)
    has_thermo = thermo_count > 0
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
            # Explicit concentration-form kf+kr is independent of C0.  Keep
            # the retained ratio at the multiplicative identity even if an
            # internal caller supplied stale derived metadata.
            reverse_std_ratio=1.0,
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
    _validate_concrete_positive_coefficient(
        _metadata_reverse_std_ratio(meta),
        label="reverse_std_ratio",
    )
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
        kf_editable=True,
        kr_editable=True,
    )


def normalize_existing_equilibrium_rate_authority(eq: object) -> EquilibriumRateAuthority:
    meta = getattr(eq, "metadata", {}) or {}
    context: EquilibriumRateInputContext | str = EquilibriumRateInputContext.PUBLIC
    if isinstance(meta, Mapping):
        if (
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


def json_safe_equilibrium_metadata_payload(source: object) -> dict[str, object]:
    if isinstance(source, Mapping) and "equilibrium_authority" not in source:
        metadata = source
    else:
        metadata = getattr(source, "metadata", {}) or {}
    payload = json_safe_equilibrium_authority_payload(metadata)
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): payload[key] for key in payload}


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

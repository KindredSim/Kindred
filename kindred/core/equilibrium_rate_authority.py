from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Iterable, Mapping

from kindred.core.constants import R
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.species import validate_c0, validate_temperature
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.thermodynamics import (
    concentration_Keq_from_deltaG_eq,
)
from kindred.core.units import (
    admit_keq,
    admit_rate,
    equilibrium_constant_unit,
    rate_constant_unit,
)
from kindred.core.simulator.errors import DSLError

AUTHORITY_ERROR = "equilibrium requires kf and exactly one of kr or Keq/dG_eq"

__all__ = [
    "equilibrium_constant_unit",
    "rate_constant_unit",
    "choose_fast_equilibrium_rate",
    "admit_explicit_directional_rate_pair",
    "derived_equilibrium_ratio_from_directional_rates",
    "FastEquilibriumRates",
    "derive_fast_equilibrium_rates",
    "EquilibriumRateAuthorityKind",
    "EquilibriumRateInputContext",
    "EquilibriumRateAuthority",
    "EquilibriumRateSymbolicRelations",
    "equilibrium_rate_symbolic_relations",
    "effective_reverse_rate_from_keq",
    "json_safe_equilibrium_authority_payload",
    "plain_finite_float_or_none",
    "effective_equilibrium_forward_rate",
    "effective_equilibrium_keq",
    "effective_equilibrium_reverse_rate",
    "validate_equilibrium_rate_authority_flags",
    "normalize_equilibrium_rate_authority",
    "normalize_existing_equilibrium_rate_authority",
    "authority_fields_from_step_entry",
    "step_entry_authored_role_is_editable",
    "require_step_entry_authored_role_is_editable",
    "step_entry_authored_role_is_derived",
]


def _positive_finite(value: object, *, label: str, allow_zero: bool = False) -> float:
    if type(value) is bool:
        raise DSLError(f"{label} must be positive and finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise DSLError(f"{label} must be positive and finite") from exc
    try:
        return admit_rate(number, field=label, allow_zero=allow_zero)
    except ValueError as exc:
        raise DSLError(str(exc)) from exc


def choose_fast_equilibrium_rate(
    explicit_rates: Iterable[float] | None = None,
    *,
    allow_zero: bool = False,
) -> float:
    """Return the authored forward coefficient without inventing rate meaning.

    The equilibrium parser supplies authored rates in directional order, so the
    first value is the explicit forward coefficient.  Reverse coefficients do
    not participate in choosing or rescaling it.
    """

    rates = tuple(explicit_rates or ())
    if not rates:
        raise DSLError("equilibrium requires an explicit positive finite kf")
    return _positive_finite(rates[0], label="kf", allow_zero=allow_zero)


def admit_explicit_directional_rate_pair(
    kf: object,
    kr: object,
) -> tuple[float, float]:
    """Admit the two independent coefficients of an authored reversible pair."""

    try:
        forward = admit_rate(kf, field="kf")
        reverse = admit_rate(kr, field="kr")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return float(forward), float(reverse)


def derived_equilibrium_ratio_from_directional_rates(
    kf: object,
    kr: object,
) -> float:
    """Return the finite positive readout derived from an admitted ``kf+kr`` pair.

    This ratio is not authored concentration-form ``Keq`` authority.  In
    particular, it is not constrained by the authored-``Keq`` envelope.
    """

    forward, reverse = admit_explicit_directional_rate_pair(kf, kr)
    try:
        ratio = forward / reverse
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "explicit kf+kr must derive a finite positive equilibrium ratio"
        ) from exc
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            "explicit kf+kr must derive a finite positive equilibrium ratio"
        )
    return float(ratio)


@dataclass(frozen=True, slots=True)
class FastEquilibriumRates:
    kf: float
    kr: float
    Keq: float
    source: str


def derive_fast_equilibrium_rates(
    *,
    Keq: float | None = None,
    dG_eq_J_per_mol: float | None = None,
    temperature_K: float = 298.15,
    explicit_rates: Iterable[float] | None = None,
    standard_conc_M: float = 1.0,
    delta_n: float = 0.0,
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
        equilibrium_constant = concentration_Keq_from_deltaG_eq(
            delta_g,
            validate_temperature(temperature_K, field="temperature_K"),
            standard_conc_M,
            delta_n,
        )
        source = "derived(dG_eq)"
    else:
        try:
            equilibrium_constant = admit_keq(Keq, field="Keq")
        except ValueError as exc:
            raise DSLError(str(exc)) from exc
        source = "derived(Keq)"
    forward = choose_fast_equilibrium_rate(
        explicit_rates,
        allow_zero=bool(Keq is not None or dG_eq_J_per_mol is not None),
    )
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
    has_explicit_keq_param: bool = False
    has_thermo_param: bool = False
    kf_editable: bool = True
    kr_editable: bool = True

    def effective_reverse_rate(self, Keq: object | None = None, kf: object | None = None) -> object:
        if self.kind == EquilibriumRateAuthorityKind.KR:
            if self.kr is None:
                raise ValueError(AUTHORITY_ERROR)
            return self.kr
        keq_value = self.Keq if Keq is None else Keq
        if keq_value is None:
            raise ValueError(AUTHORITY_ERROR)
        return effective_reverse_rate_from_keq(self.kf if kf is None else kf, keq_value)

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
    literal: Callable[[object], object],
    exponential: Callable[[object], object],
    source_coordinate: Callable[[str], object] | None = None,
    standard_concentration: object | None = None,
) -> EquilibriumRateSymbolicRelations:
    """Project accepted equilibrium authority without choosing a backend.

    Explicit ``kf + kr`` is standard-state inert.  Thermodynamic authority
    projects its accepted free-energy relation from the normalized authority;
    ``standard_concentration`` is the exact backend expression used for the
    concentration-form normalization.
    """

    if not callable(literal):
        raise TypeError("literal must be callable.")
    if not callable(exponential):
        raise TypeError("exponential must be callable.")
    if source_coordinate is not None and not callable(source_coordinate):
        raise TypeError("source_coordinate must be callable when provided.")

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
        reverse_relation = (
            source_coordinate("kr")
            if source_coordinate is not None
            else literal(reverse_value)
        )
        return EquilibriumRateSymbolicRelations(
            kf=forward_relation,
            kr=reverse_relation,
            Keq=forward_relation / reverse_relation,  # type: ignore[operator]
        )

    if authority.has_explicit_keq_param:
        keq_relation = (
            source_coordinate("Keq")
            if source_coordinate is not None
            else literal(
                finite_scalar(
                    authority.Keq,
                    label="equilibrium Keq",
                    positive=True,
                )
            )
        )
    elif authority.dG_eq_J_per_mol is not None:
        delta_g = finite_scalar(
            authority.dG_eq_J_per_mol,
            label="equilibrium dG_eq_J_per_mol",
            positive=False,
        )
        keq_relation = exponential(
            -(source_coordinate("dG_eq") if source_coordinate is not None else literal(delta_g))
            / (literal(float(R)) * temperature)  # type: ignore[operator]
        )
        forward_order = directional_order_fraction(
            getattr(equilibrium, "stoich_forward", {}) or {},
        )
        reverse_order = directional_order_fraction(
            getattr(equilibrium, "stoich_back", {}) or {},
        )
        delta_n = reverse_order - forward_order
        if delta_n:
            if standard_concentration is None:
                metadata = getattr(equilibrium, "metadata", {}) or {}
                raw_standard = (
                    metadata.get(EquilibriumMetadataKeys.STANDARD_CONC_M, 1.0)
                    if isinstance(metadata, Mapping)
                    else 1.0
                )
                try:
                    standard = validate_c0(
                        raw_standard,
                        field="equilibrium standard_conc_M",
                    )
                except ValueError as exc:
                    raise ValueError(
                        "equilibrium standard concentration must be strictly positive and admitted"
                    ) from exc
                standard_concentration = literal(float(standard))
            keq_relation *= standard_concentration ** literal(delta_n)  # type: ignore[operator]
    else:
        keq_relation = (
            source_coordinate("Keq")
            if source_coordinate is not None
            else literal(
                finite_scalar(
                    authority.Keq,
                    label="equilibrium Keq",
                    positive=True,
                )
            )
        )

    reverse_relation = forward_relation / keq_relation  # type: ignore[operator]
    return EquilibriumRateSymbolicRelations(
        kf=forward_relation,
        kr=reverse_relation,
        Keq=keq_relation,
    )


def effective_reverse_rate_from_keq(kf: object, Keq: object) -> object:
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
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label} must be a finite real")
        if label == "Keq" and number == 0.0:
            raise ValueError(f"{label} must be positive and finite")
        if label == "kf":
            try:
                admit_rate(number, field=label, allow_zero=True)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        elif label == "Keq":
            try:
                admit_keq(number, field=label)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        elif number == 0.0:
            raise ValueError(f"{label} must be positive and finite")
        else:
            try:
                admit_rate(number, field=label)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        return float(number)

    kf_value = concrete(kf, label="kf")
    keq_value = concrete(Keq, label="Keq")
    if kf_value is None or keq_value is None:
        return kf / Keq  # type: ignore[operator]
    if kf_value == 0.0:
        return 0.0
    try:
        kf_mantissa, kf_exponent = math.frexp(kf_value)
        keq_mantissa, keq_exponent = math.frexp(keq_value)
        scaled_mantissa = kf_mantissa / keq_mantissa
        reverse = math.ldexp(
            scaled_mantissa,
            kf_exponent - keq_exponent,
        )
    except OverflowError as exc:
        raise ValueError("thermodynamic reverse coefficient overflows the finite real domain") from exc
    if not math.isfinite(reverse) or reverse < 0.0:
        raise ValueError("thermodynamic reverse coefficient must be finite")
    if reverse == 0.0:
        raise ValueError(
            "thermodynamic reverse coefficient underflows the finite real domain"
        )
    try:
        admit_rate(reverse, field="thermodynamic reverse rate")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
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
    try:
        temperature = validate_temperature(temperature_K, field="temperature_K")
        metadata = getattr(eq, "metadata", {}) or {}
        standard_conc = (
            metadata.get(EquilibriumMetadataKeys.STANDARD_CONC_M, 1.0)
            if isinstance(metadata, Mapping)
            else 1.0
        )
        from kindred.core.kinetics import (
            forward_rate_at_temperature,
            forward_rate_model_for_equilibrium,
        )

        model = forward_rate_model_for_equilibrium(
            eq,
            temperature_K=temperature,
            default_standard_conc_M=float(standard_conc),
        )
        value = forward_rate_at_temperature(model, temperature)
        return admit_rate(value, field="equilibrium kf", allow_zero=True)
    except (TypeError, ValueError, OverflowError):
        return None


def effective_equilibrium_keq(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = validate_temperature(temperature_K, field="temperature_K")
    meta = getattr(eq, "metadata", {}) or {}
    forward_order = directional_order_fraction(
        getattr(eq, "stoich_forward", {}) or {},
    )
    reverse_order = directional_order_fraction(
        getattr(eq, "stoich_back", {}) or {},
    )
    delta_n = float(reverse_order - forward_order)
    standard_conc = 1.0
    if isinstance(meta, Mapping):
        standard_conc = meta.get(
            EquilibriumMetadataKeys.STANDARD_CONC_M,
            1.0,
        )
    try:
        standard_conc = float(standard_conc)
    except (TypeError, ValueError, OverflowError):
        return None
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        # Explicit kf+kr has no writable Keq authority, but its ratio remains
        # a useful derived readout.  It is deliberately independent of C0:
        # the explicit-rate basis owns both directional coefficients unchanged.
        kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
        kr = _finite_float_or_none(authority.kr, temperature_K=temperature)
        if kf is None or kr is None or kf <= 0.0 or kr <= 0.0:
            return None
        try:
            return derived_equilibrium_ratio_from_directional_rates(kf, kr)
        except ValueError:
            return None

    if (
        authority.kind == EquilibriumRateAuthorityKind.KEQ
        and authority.dG_eq_J_per_mol is not None
        and not authority.has_explicit_keq_param
    ):
        dg_value = _finite_float_or_none(
            authority.dG_eq_J_per_mol,
            temperature_K=temperature,
        )
        if dg_value is not None:
            return float(
                concentration_Keq_from_deltaG_eq(
                    dg_value,
                    temperature,
                    standard_conc,
                    delta_n,
                )
            )

    authoritative_keq = _finite_float_or_none(
        authority.Keq,
        temperature_K=temperature,
    )
    if authoritative_keq is not None:
        try:
            return admit_keq(authoritative_keq, field="Keq")
        except ValueError:
            return None
    return None

def effective_equilibrium_reverse_rate(eq: object, *, temperature_K: float = 298.15) -> float | None:
    authority = normalize_existing_equilibrium_rate_authority(eq)
    temperature = float(temperature_K)
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        explicit = _finite_float_or_none(authority.kr, temperature_K=temperature)
        if explicit is None:
            return None
        try:
            return admit_rate(explicit, field="equilibrium kr")
        except ValueError:
            return None
    kf = effective_equilibrium_forward_rate(eq, temperature_K=temperature)
    keq = effective_equilibrium_keq(eq, temperature_K=temperature)
    if kf is None or keq is None or kf < 0.0 or keq <= 0.0:
        return None
    value = authority.effective_reverse_rate(Keq=keq, kf=kf)
    resolved = _finite_float_or_none(value, temperature_K=temperature)
    if resolved is None:
        return None
    try:
        return admit_rate(
            resolved,
            field="equilibrium kr",
            allow_zero=True,
        )
    except ValueError:
        return None


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


def _validate_concrete_positive_coefficient(
    value: object | None,
    *,
    label: str,
    allow_zero: bool = False,
) -> None:
    """Reject invalid concrete coefficients while preserving symbolic authority."""
    if value is None:
        return
    if type(value) is bool or isinstance(value, str):
        raise ValueError(f"{label} must be a positive finite real")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return
    if label in {"kf", "kr"}:
        admit_rate(number, field=label, allow_zero=bool(allow_zero))
        return
    if label in {"Keq", "Keq_input"}:
        admit_keq(number, field=label)
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
    thermo_input = bool(
        Keq is not None or metadata_keq is not None or metadata_dg is not None
    )
    user_kf, user_kr = _metadata_user_flags(meta)
    # A zero forward coefficient is meaningful only when the authored reverse
    # authority is thermodynamic (it disables both directions).  In
    # particular, an explicit kf+kr pair remains strictly positive.
    _validate_concrete_positive_coefficient(
        kf,
        label="kf",
        # Normalized mechanisms retain their derived ``kr`` readout on the
        # object, so authored reverse-role metadata—not the presence of that
        # derived field—decides whether thermodynamic zero is admissible.
        allow_zero=bool(thermo_input and not user_kr),
    )
    _validate_concrete_positive_coefficient(
        kr,
        label="kr",
        # A normalized thermodynamic step retains its derived ``kr`` field.
        # Permit that field to be zero only when the authored forward
        # authority is the thermodynamic zero; an authored explicit pair still
        # rejects zero in either direction.
        allow_zero=bool(
            thermo_input
            and not user_kr
            and _finite_float_or_none(kf) == 0.0
        ),
    )
    _validate_concrete_positive_coefficient(Keq, label="Keq")
    _validate_concrete_positive_coefficient(metadata_keq, label="Keq_input")
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
        kf_numeric = plain_finite_float_or_none(kf)
        kr_numeric = plain_finite_float_or_none(kr)
        if kf_numeric is not None and kr_numeric is not None:
            derived_equilibrium_ratio_from_directional_rates(
                kf_numeric,
                kr_numeric,
            )
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

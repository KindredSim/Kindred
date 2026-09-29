from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from kindred.core.constants import R
from kindred.core.kinetics import ConstantRateModel, ForwardRateModel, forward_rate_model_payload
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.species import validate_c0
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import (
    admit_energy_j_per_mol,
    admit_keq,
    admit_rate,
)

AUTHORITY_ERROR = "equilibrium requires kf and exactly one of kr or Keq/dG_eq"

__all__ = [
    "admit_explicit_directional_rate_pair",
    "derived_equilibrium_ratio_from_directional_rates",
    "EquilibriumRateAuthorityKind",
    "EquilibriumRateAuthority",
    "EquilibriumRateSymbolicRelations",
    "equilibrium_rate_symbolic_relations",
    "effective_reverse_rate_from_keq",
    "json_safe_equilibrium_authority_payload",
    "validate_equilibrium_rate_authority_flags",
    "normalize_equilibrium_rate_authority",
]


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


class EquilibriumRateAuthorityKind:
    KR = "kr_authoritative"
    KEQ = "Keq_authoritative"


@dataclass(frozen=True, slots=True)
class EquilibriumRateAuthority:
    kind: str
    forward_model: ForwardRateModel
    kr: object | None
    Keq: object | None
    source: str = ""
    dG_eq_J_per_mol: object | None = None
    has_explicit_keq_param: bool = False

    def authored_role_is_editable(self, role: str) -> bool:
        """Whether an indexed kf/kr/Keq coordinate is an authored input."""
        if role == "kf":
            return isinstance(self.forward_model, ConstantRateModel)
        if role == "kr":
            return self.kind == EquilibriumRateAuthorityKind.KR
        return self.has_explicit_keq_param

    def identity_items(self) -> tuple[tuple[str, object], ...]:
        items: list[tuple[str, object]] = [
            ("kind", self.kind),
            ("source", self.source),
            ("forward_model", forward_rate_model_payload(self.forward_model)),
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
    standard_concentration: object,
    constant: Callable[[str], object] | None = None,
) -> EquilibriumRateSymbolicRelations:
    """Project accepted equilibrium authority without choosing a backend.

    Explicit ``kf + kr`` is standard-state inert.  Thermodynamic authority
    projects its accepted free-energy relation from the normalized authority;
    ``standard_concentration`` is the exact backend expression used for the
    concentration-form normalization.
    ``constant`` optionally projects the named fixed gas constant instead of its
    owned numeric literal, without changing thermodynamic or runtime authority.
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

    authority = equilibrium.authority
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
            / ((constant("R") if constant is not None else literal(float(R))) * temperature)  # type: ignore[operator]
        )
        forward_order = directional_order_fraction(
            equilibrium.stoich_forward,
        )
        reverse_order = directional_order_fraction(
            equilibrium.stoich_back,
        )
        delta_n = reverse_order - forward_order
        if delta_n:
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


def normalize_equilibrium_rate_authority(
    *,
    forward_model: ForwardRateModel,
    kr: float | None,
    metadata: Mapping[str, Any] | None = None,
) -> EquilibriumRateAuthority:
    """Admit authored reverse authority without materializing derived rates."""
    meta = metadata or {}
    keq = meta.get(EquilibriumMetadataKeys.KEQ_INPUT)
    delta_g = meta.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
    validate_equilibrium_rate_authority_flags(
        has_kf=True, has_kr=kr is not None,
        has_Keq=keq is not None, has_dG_eq=delta_g is not None,
    )
    if isinstance(forward_model, ConstantRateModel):
        admit_rate(forward_model.k, field="kf", allow_zero=kr is None)
    if kr is not None:
        reverse = admit_rate(kr, field="kr")
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR, forward_model=forward_model,
            kr=reverse, Keq=None, source="explicit_kr",
        )
    if keq is not None:
        keq = admit_keq(keq, field="Keq")
    if delta_g is not None:
        delta_g = admit_energy_j_per_mol(delta_g, field="dG_eq_J_per_mol")
        validate_c0(meta.get(EquilibriumMetadataKeys.STANDARD_CONC_M, 1.0),
                    field="equilibrium standard_conc_M")
    return EquilibriumRateAuthority(
        kind=EquilibriumRateAuthorityKind.KEQ, forward_model=forward_model,
        kr=None, Keq=keq,
        source="explicit_Keq" if keq is not None else "dG_eq",
        dG_eq_J_per_mol=delta_g, has_explicit_keq_param=keq is not None,
    )

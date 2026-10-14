from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from kindred.core.constants import R
from kindred.core.exceptions import ValueDomainError
from kindred.core.kinetics import ForwardRateModel, forward_rate_model_payload
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
from kindred.core.stoichiometry import directional_order_fraction
from kindred.core.units import admit_rate

AUTHORITY_ERROR = "A reversible step needs a forward rate and exactly one of kr, Keq, or dG_eq."

__all__ = [
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


def derived_equilibrium_ratio_from_directional_rates(kf: float, kr: float) -> float:
    """Return the finite positive readout derived from an admitted ``kf+kr`` pair.

    This ratio is not authored concentration-form ``Keq`` authority.  In
    particular, it is not constrained by the authored-``Keq`` envelope.
    """

    try:
        ratio = kf / kr
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueDomainError(
            "Keq", "explicit kf+kr must derive a finite positive equilibrium ratio"
        ) from exc
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueDomainError(
            "Keq", "explicit kf+kr must derive a finite positive equilibrium ratio"
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

    # The authority's kr, Keq and dG_eq arrive admitted by DSL parsing or the
    # grouped coordinate owner.
    authority = equilibrium.authority
    if authority.kind == EquilibriumRateAuthorityKind.KR:
        reverse_relation = (
            source_coordinate("kr")
            if source_coordinate is not None
            else literal(float(authority.kr))
        )
        return EquilibriumRateSymbolicRelations(
            kr=reverse_relation,
            Keq=forward_relation / reverse_relation,  # type: ignore[operator]
        )

    if authority.has_explicit_keq_param:
        keq_relation = (
            source_coordinate("Keq")
            if source_coordinate is not None
            else literal(float(authority.Keq))
        )
    elif authority.dG_eq_J_per_mol is not None:
        keq_relation = exponential(
            -(source_coordinate("dG_eq") if source_coordinate is not None
              else literal(float(authority.dG_eq_J_per_mol)))
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
            else literal(float(authority.Keq))
        )

    reverse_relation = forward_relation / keq_relation  # type: ignore[operator]
    return EquilibriumRateSymbolicRelations(
        kr=reverse_relation,
        Keq=keq_relation,
    )


def effective_reverse_rate_from_keq(kf: float, Keq: float) -> float:
    """Return the reverse coefficient of thermodynamic authority, admitted as a rate."""

    return admit_rate(kf / Keq, field="The calculated kr (kf/Keq)", allow_zero=kf == 0.0)


def json_safe_equilibrium_authority_payload(value: object) -> object:
    # identity_items holds only str, finite float, int, None and mappings.
    if isinstance(value, Mapping):
        return {
            str(key): json_safe_equilibrium_authority_payload(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    return value


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
    metadata: Mapping[str, Any],
) -> EquilibriumRateAuthority:
    """Admit authored reverse authority without materializing derived rates."""
    # The parser admitted exactly one reverse authority, and coordinate writes
    # never add or remove one.
    keq = metadata.get(EquilibriumMetadataKeys.KEQ_INPUT)
    delta_g = metadata.get(EquilibriumMetadataKeys.DG_EQ_J_PER_MOL)
    # Rate and energy values arrive admitted by DSL parsing or the grouped
    # coordinate owner, and the standard concentration by DSL parsing.
    if kr is not None:
        return EquilibriumRateAuthority(
            kind=EquilibriumRateAuthorityKind.KR, forward_model=forward_model,
            kr=kr, Keq=None, source="explicit_kr",
        )
    return EquilibriumRateAuthority(
        kind=EquilibriumRateAuthorityKind.KEQ, forward_model=forward_model,
        kr=None, Keq=keq,
        source="explicit_Keq" if keq is not None else "dG_eq",
        dG_eq_J_per_mol=delta_g, has_explicit_keq_param=keq is not None,
    )

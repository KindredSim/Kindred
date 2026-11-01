"""
Core kinetics formulas shared across Kindred.

This module contains the fundamental Arrhenius/Eyring rate mappings so both
the simulator layer and the ODE builder can depend on a single source of truth
without creating cross-layer imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, TypeAlias

from .constants import R, h, k_B
from .exceptions import ValueDomainError
from .units import admit_rate

__all__ = [
    "arrhenius_rate",
    "eyring_rate",
    "ConstantRateModel",
    "ArrheniusRateModel",
    "EyringRateModel",
    "ForwardRateModel",
    "forward_rate_at_temperature",
    "forward_rate_family",
    "forward_rate_model_payload",
    "forward_rate_symbolic_relation",
]


# k, A, Ea, dG_act and kappa arrive admitted against their role domains: authored
# fields by DSL parsing, coordinate writes by the grouped coordinate owner.
# DSL parsing also admits Eyring's standard concentration (C0), molecularity
# (the summed positive reactant coefficients) and positive integer degeneracy.
# Temperatures arrive admitted by the DSL T directive or its admitted schedule.
@dataclass(frozen=True, slots=True)
class ConstantRateModel:
    k: float


@dataclass(frozen=True, slots=True)
class ArrheniusRateModel:
    A: float
    Ea_J_per_mol: float


@dataclass(frozen=True, slots=True)
class EyringRateModel:
    dG_act_J_per_mol: float
    kappa: float
    standard_conc_M: float
    molecularity: float
    degeneracy: int = 1


ForwardRateModel: TypeAlias = (
    ConstantRateModel | ArrheniusRateModel | EyringRateModel
)


def forward_rate_family(model: ForwardRateModel) -> str:
    """Return the stable authored family name for one typed model."""
    if isinstance(model, ConstantRateModel):
        return "Constant"
    if isinstance(model, ArrheniusRateModel):
        return "Arrhenius"
    return "Eyring"


def forward_rate_model_payload(model: ForwardRateModel) -> dict[str, object]:
    family = forward_rate_family(model)
    if isinstance(model, ConstantRateModel):
        fields: dict[str, object] = {"k": float(model.k)}
    elif isinstance(model, ArrheniusRateModel):
        fields = {
            "A": float(model.A),
            "Ea_J_per_mol": float(model.Ea_J_per_mol),
        }
    else:
        fields = {
            "dG_act_J_per_mol": float(model.dG_act_J_per_mol),
            "kappa": float(model.kappa),
            "standard_conc_M": float(model.standard_conc_M),
            "molecularity": float(model.molecularity),
            "degeneracy": int(model.degeneracy),
        }
    return {"family": family, **fields}


def forward_rate_at_temperature(
    model: ForwardRateModel,
    temperature_K: float,
) -> float:
    """Evaluate exactly one typed forward-rate family."""
    if isinstance(model, ConstantRateModel):
        return model.k
    if isinstance(model, ArrheniusRateModel):
        return arrhenius_rate(model.A, model.Ea_J_per_mol, temperature_K)
    return eyring_rate(
        model.dG_act_J_per_mol,
        temperature_K,
        kappa=model.kappa,
        standard_conc_M=model.standard_conc_M,
        molecularity=model.molecularity,
        degeneracy=model.degeneracy,
    )


def forward_rate_symbolic_relation(
    model: ForwardRateModel,
    *,
    temperature: object,
    literal: Callable[[object], object],
    exponential: Callable[[object], object],
    source_coordinate: Callable[[str], object | None],
    standard_concentration: object | None = None,
    constant: Callable[[str], object],
) -> object:
    """Project one accepted typed rate model onto a symbolic backend.

    Chemistry owns the relation and all canonical-unit inputs.  The caller
    supplies only the backend's symbolic temperature, exact-literal constructor,
    and exponential operation; no symbolic backend is imported here.  The
    source-coordinate callback returns ``None`` only for an omitted Eyring
    kappa or degeneracy, which then keeps its literal default.  The optional
    ``standard_concentration`` is a backend-owned exact expression used for
    Eyring's concentration normalization; when omitted, the accepted model's
    numeric standard concentration is used.  ``constant`` projects named
    fixed physical constants.
    """

    def exact(field: str) -> object:
        return constant(field)

    def coordinate(role: str, value: float) -> object:
        """Use a canonical authored coordinate, or the admitted literal."""

        resolved = source_coordinate(role)
        return literal(float(value)) if resolved is None else resolved

    # The typed models carry admitted fields.
    if isinstance(model, ConstantRateModel):
        return coordinate("kf", float(model.k))
    if isinstance(model, ArrheniusRateModel):
        return coordinate("A", float(model.A)) * exponential(  # type: ignore[operator]
            -coordinate("Ea", float(model.Ea_J_per_mol))
            / (exact("R") * temperature)  # type: ignore[operator]
        )
    degeneracy_relation = coordinate("degeneracy", float(model.degeneracy))
    kappa_relation = coordinate("kappa", float(model.kappa))
    concentration_relation = (
        literal(float(model.standard_conc_M))
        if standard_concentration is None
        else standard_concentration
    )
    relation = (
        degeneracy_relation
        * kappa_relation
        * exact("k_B")
        * temperature
        / exact("h")
        * exponential(
            -coordinate("dG_act", float(model.dG_act_J_per_mol))
            / (exact("R") * temperature)  # type: ignore[operator]
        )
    )
    standard_exponent = Fraction(1) - Fraction(str(model.molecularity))
    if standard_exponent:
        relation *= concentration_relation ** literal(standard_exponent)  # type: ignore[operator]
    return relation


def eyring_rate(
    dG_act_J_per_mol: float,
    T: float,
    *,
    kappa: float,
    molecularity: float,
    standard_conc_M: float,
    degeneracy: int,
) -> float:
    """
    Compute Eyring rate constant.

    - Unimolecular: g * kappa * (k_B T / h) * exp(-dG_act/(R T))  [1/s]
    - n-molecular (n >= 2): multiply by 1/(C0^(n-1))        [1/(M^(n-1)*s)]
    """
    exponent = -dG_act_J_per_mol / (R * T)
    try:
        k = (
            degeneracy
            * kappa
            * (k_B * T / h)
            * math.exp(exponent)
            * math.pow(standard_conc_M, 1.0 - molecularity)
        )
    except OverflowError as exc:
        raise ValueDomainError("Eyring rate overflows the finite real domain") from exc
    return admit_rate(k, field="Eyring rate")


def arrhenius_rate(A: float, Ea_J_per_mol: float, T: float) -> float:
    """
    Arrhenius rate constant:

        k = A * exp(-Ea/(R T))

    Notes
    -----
    - A should already carry the correct order unit (1/s or 1/(M^(n-1)*s)).
    - This function does not attempt to infer molecularity from A.
    """
    exponent = -Ea_J_per_mol / (R * T)
    try:
        rate = A * math.exp(exponent)
    except OverflowError as exc:
        raise ValueDomainError("Arrhenius rate overflows the finite real domain") from exc
    return admit_rate(rate, field="Arrhenius rate")

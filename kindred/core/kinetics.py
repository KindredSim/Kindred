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
from .species import validate_c0, validate_temperature
from .units import (
    admit_arrhenius_prefactor,
    admit_energy_j_per_mol,
    admit_rate,
)

__all__ = [
    "arrhenius_rate",
    "eyring_prefactor",
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


@dataclass(frozen=True, slots=True)
class ConstantRateModel:
    k: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "k",
            admit_rate(self.k, field="Constant k", allow_zero=True),
        )


@dataclass(frozen=True, slots=True)
class ArrheniusRateModel:
    A: float
    Ea_J_per_mol: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "A",
            admit_arrhenius_prefactor(self.A, field="Arrhenius A"),
        )
        object.__setattr__(
            self,
            "Ea_J_per_mol",
            admit_energy_j_per_mol(
                self.Ea_J_per_mol,
                field="Arrhenius Ea_J_per_mol",
            ),
        )


@dataclass(frozen=True, slots=True)
class EyringRateModel:
    dG_act_J_per_mol: float
    kappa: float
    standard_conc_M: float
    molecularity: float
    degeneracy: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dG_act_J_per_mol",
            admit_energy_j_per_mol(
                self.dG_act_J_per_mol,
                field="Eyring dG_act_J_per_mol",
            ),
        )
        kappa = _numeric_float(self.kappa, field="Eyring kappa")
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("Eyring kappa must be positive and finite.")
        object.__setattr__(self, "kappa", kappa)
        standard = validate_c0(
            self.standard_conc_M,
            field="Eyring standard_conc_M",
        )
        object.__setattr__(self, "standard_conc_M", standard)
        _positive_integer(self.degeneracy, field="Eyring degeneracy")
        object.__setattr__(self, "molecularity", _directional_order(self.molecularity))


ForwardRateModel: TypeAlias = (
    ConstantRateModel | ArrheniusRateModel | EyringRateModel
)


def _numeric_float(value: object, *, field: str) -> float:
    if type(value) is bool:
        raise TypeError(f"{field} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc


def _positive_integer(value: object, *, field: str) -> int:
    """Admit only a positive, non-boolean builtin integer coordinate."""
    if type(value) is not int:
        raise TypeError(f"{field} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _directional_order(value: object, *, field: str = "molecularity") -> float:
    order = _numeric_float(value, field=field)
    if not math.isfinite(order) or order < 0.0:
        raise ValueError(f"{field} must be a finite nonnegative real.")
    return float(order)


def _finite_result(value: float, *, field: str) -> float:
    try:
        return admit_rate(value, field=field)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def forward_rate_family(model: ForwardRateModel) -> str:
    """Return the stable authored family name for one typed model."""
    if isinstance(model, ConstantRateModel):
        return "Constant"
    if isinstance(model, ArrheniusRateModel):
        return "Arrhenius"
    if isinstance(model, EyringRateModel):
        return "Eyring"
    raise TypeError("model must be one typed forward-rate model.")


def forward_rate_model_payload(model: ForwardRateModel) -> dict[str, object]:
    family = forward_rate_family(model)
    if isinstance(model, ConstantRateModel):
        fields: dict[str, object] = {"k": float(model.k)}
    elif isinstance(model, ArrheniusRateModel):
        fields = {
            "A": float(model.A),
            "Ea_J_per_mol": float(model.Ea_J_per_mol),
        }
    elif isinstance(model, EyringRateModel):
        fields = {
            "dG_act_J_per_mol": float(model.dG_act_J_per_mol),
            "kappa": float(model.kappa),
            "standard_conc_M": float(model.standard_conc_M),
            "molecularity": float(model.molecularity),
            "degeneracy": int(model.degeneracy),
        }
    else:
        raise TypeError("Unsupported typed forward-rate model.")
    return {"family": family, **fields}


def forward_rate_at_temperature(
    model: ForwardRateModel,
    temperature_K: float,
) -> float:
    """Evaluate exactly one typed forward-rate family."""
    if isinstance(model, ConstantRateModel):
        value = _numeric_float(model.k, field="Constant k")
        admit_rate(value, field="Constant k", allow_zero=True)
        return value
    if isinstance(model, ArrheniusRateModel):
        return arrhenius_rate(model.A, model.Ea_J_per_mol, temperature_K)
    if isinstance(model, EyringRateModel):
        return eyring_rate(
            model.dG_act_J_per_mol,
            temperature_K,
            kappa=model.kappa,
            standard_conc_M=model.standard_conc_M,
            molecularity=model.molecularity,
            degeneracy=_positive_integer(
                model.degeneracy,
                field="Eyring degeneracy",
            ),
        )
    raise TypeError("model must be one typed forward-rate model.")


def forward_rate_symbolic_relation(
    model: ForwardRateModel,
    *,
    temperature: object,
    literal: Callable[[object], object],
    exponential: Callable[[object], object],
    source_coordinate: Callable[[str], object | None] | None = None,
    standard_concentration: object | None = None,
    constant: Callable[[str], object] | None = None,
) -> object:
    """Project one accepted typed rate model onto a symbolic backend.

    Chemistry owns the relation and all canonical-unit inputs.  The caller
    supplies only the backend's symbolic temperature, exact-literal constructor,
    and exponential operation; no symbolic backend is imported here.  When a
    source-coordinate callback is supplied, it may return ``None`` for omitted
    Eyring coordinates; those roles then retain their literal defaults.  The
    optional ``standard_concentration`` is a backend-owned exact expression
    used for Eyring's concentration normalization.  If omitted, the accepted
    model's numeric standard concentration is used for ordinary projections.
    ``constant`` optionally projects named fixed physical constants; omitting it
    preserves the ordinary owned numeric constants without changing the formula.
    """

    def exact(value: object, *, field: str) -> object:
        return constant(field) if constant is not None else literal(float(value))

    def coordinate(role: str, value: float) -> object:
        """Use a canonical authored coordinate, or the admitted literal."""

        if source_coordinate is not None:
            resolved = source_coordinate(role)
            # A source-coordinate owner may omit a role whose scientific
            # value is a model default.  Omitted Eyring kappa and degeneracy
            # remain their literal defaults, while authored values are
            # source-backed coordinates.
            if resolved is not None:
                return resolved
            if role not in {"kappa", "degeneracy"}:
                raise ValueError(
                    f"missing canonical source coordinate for {role!r}"
                )
        return literal(float(value))

    # The typed models admitted every field at construction.
    if isinstance(model, ConstantRateModel):
        return coordinate("kf", float(model.k))
    if isinstance(model, ArrheniusRateModel):
        return coordinate("A", float(model.A)) * exponential(  # type: ignore[operator]
            -coordinate("Ea", float(model.Ea_J_per_mol))
            / (exact(R, field="R") * temperature)  # type: ignore[operator]
        )
    if isinstance(model, EyringRateModel):
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
            * exact(k_B, field="k_B")
            * temperature
            / exact(h, field="h")
            * exponential(
                -coordinate("dG_act", float(model.dG_act_J_per_mol))
                / (exact(R, field="R") * temperature)  # type: ignore[operator]
            )
        )
        standard_exponent = Fraction(1) - Fraction(str(model.molecularity))
        if standard_exponent:
            relation *= concentration_relation ** literal(standard_exponent)  # type: ignore[operator]
        return relation
    raise TypeError("model must be one typed forward-rate model.")


def eyring_prefactor(T: float) -> float:
    """(k_B T / h) in 1/s."""
    temperature = validate_temperature(T, field="T")
    return _finite_result((k_B * temperature) / h, field="Eyring prefactor")


def eyring_rate(
    dG_act_J_per_mol: float,
    T: float,
    *,
    kappa: float = 1.0,
    molecularity: float = 1.0,
    standard_conc_M: float = 1.0,
    degeneracy: int = 1,
) -> float:
    """
    Compute Eyring rate constant.

    - Unimolecular: g * kappa * (k_B T / h) * exp(-dG_act/(R T))  [1/s]
    - n-molecular (n >= 2): multiply by 1/(C0^(n-1))        [1/(M^(n-1)*s)]
    """
    T = validate_temperature(T, field="T")
    standard = _numeric_float(
        standard_conc_M,
        field="standard_conc_M",
    )
    standard = validate_c0(standard, field="standard_conc_M")
    n = _directional_order(molecularity)
    degeneracy = _positive_integer(degeneracy, field="Eyring degeneracy")

    transmission = _numeric_float(kappa, field="kappa")
    if not math.isfinite(transmission) or transmission <= 0.0:
        raise ValueError("kappa must be positive and finite")

    dG = _numeric_float(
        dG_act_J_per_mol,
        field="dG_act_J_per_mol",
    )
    dG = admit_energy_j_per_mol(dG, field="dG_act_J_per_mol")
    exponent = -dG / (R * T)
    try:
        k = (
            degeneracy
            * transmission
            * eyring_prefactor(T)
            * math.exp(exponent)
            * math.pow(standard, 1.0 - n)
        )
    except OverflowError as exc:
        raise ValueError("Eyring rate overflows the finite real domain") from exc
    return _finite_result(k, field="Eyring rate")


def arrhenius_rate(A: float, Ea_J_per_mol: float, T: float) -> float:
    """
    Arrhenius rate constant:

        k = A * exp(-Ea/(R T))

    Notes
    -----
    - A should already carry the correct order unit (1/s or 1/(M^(n-1)*s)).
    - This function does not attempt to infer molecularity from A.
    """
    T = validate_temperature(T, field="T")
    prefactor = admit_arrhenius_prefactor(A, field="A")
    activation = admit_energy_j_per_mol(Ea_J_per_mol, field="Ea_J_per_mol")
    exponent = -activation / (R * T)
    try:
        rate = prefactor * math.exp(exponent)
    except OverflowError as exc:
        raise ValueError("Arrhenius rate overflows the finite real domain") from exc
    return _finite_result(rate, field="Arrhenius rate")

"""
Core kinetics formulas shared across Kindred.

This module intentionally contains the fundamental Arrhenius/Eyring and
equilibrium-constant mappings so both the simulator layer and the ODE builder
can depend on a single source of truth without creating cross-layer imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, TypeAlias

from .constants import R, h, k_B

__all__ = [
    "arrhenius_rate",
    "eyring_prefactor",
    "eyring_rate",
    "K_from_deltaG_eq",
    "ConstantRateModel",
    "ArrheniusRateModel",
    "EyringRateModel",
    "ForwardRateModel",
    "forward_rate_at_temperature",
    "forward_rate_family",
    "forward_rate_model_for_equilibrium",
    "forward_rate_model_for_reaction",
    "write_reference_rate",
]


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
    molecularity: int


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


def forward_rate_family(model: ForwardRateModel) -> str:
    """Return the stable authored family name for one typed model."""
    if isinstance(model, ConstantRateModel):
        return "Constant"
    if isinstance(model, ArrheniusRateModel):
        return "Arrhenius"
    if isinstance(model, EyringRateModel):
        return "Eyring"
    raise TypeError("model must be one typed forward-rate model.")


def _decode_forward_rate_model(
    payload: Mapping[str, object],
    *,
    family: str,
    constant_rate: object | None,
    molecularity: int,
    default_kappa: float = 1.0,
    default_standard_conc_M: float = 1.0,
) -> ForwardRateModel:
    """Decode one final Constant/Arrhenius/Eyring authority.

    ``constant_rate`` is used only for the Constant family. Energy-authored
    families must carry their complete authored coordinates in ``payload``;
    they never fall back to a scalar rate evaluated at one temperature.
    """

    fields = dict(payload)
    if type(molecularity) is not int:
        raise TypeError("molecularity must be an int.")
    order = molecularity
    if order < 1:
        raise ValueError("molecularity must be at least one.")
    allowed_fields = {
        "Constant": frozenset({"k"}),
        "Arrhenius": frozenset({"A", "Ea_J_per_mol"}),
        "Eyring": frozenset(
            {
                "dG_act_J_per_mol",
                "kappa",
                "standard_conc_M",
            }
        ),
    }
    if family not in allowed_fields:
        raise ValueError(f"Unsupported forward-rate family {family!r}.")
    unexpected = sorted(set(fields).difference(allowed_fields[family]))
    if unexpected:
        raise ValueError(
            f"{family} forward-rate authority has unsupported field(s): "
            + ", ".join(unexpected)
        )

    if family == "Constant":
        raw_rate = fields.get("k")
        if raw_rate is None:
            raw_rate = constant_rate
        if raw_rate is None:
            raise ValueError("Constant forward-rate authority requires k.")
        model: ForwardRateModel = ConstantRateModel(
            k=_numeric_float(raw_rate, field="Constant k")
        )
    elif family == "Arrhenius":
        if fields.get("A") is None:
            raise ValueError("Arrhenius forward-rate authority requires A.")
        raw_ea = fields.get("Ea_J_per_mol")
        if raw_ea is None:
            raise ValueError(
                "Arrhenius forward-rate authority requires Ea_J_per_mol."
            )
        model = ArrheniusRateModel(
            A=_numeric_float(fields["A"], field="Arrhenius A"),
            Ea_J_per_mol=_numeric_float(
                raw_ea,
                field="Arrhenius Ea_J_per_mol",
            ),
        )
    elif family == "Eyring":
        raw_dg = fields.get("dG_act_J_per_mol")
        if raw_dg is None:
            raise ValueError(
                "Eyring forward-rate authority requires dG_act_J_per_mol."
            )
        model = EyringRateModel(
            dG_act_J_per_mol=_numeric_float(
                raw_dg,
                field="Eyring dG_act_J_per_mol",
            ),
            kappa=_numeric_float(
                fields.get("kappa", default_kappa),
                field="Eyring kappa",
            ),
            standard_conc_M=_numeric_float(
                fields.get(
                    "standard_conc_M",
                    default_standard_conc_M,
                ),
                field="Eyring standard_conc_M",
            ),
            molecularity=order,
        )
    # Validate authored coordinates without evaluating at an arbitrary
    # temperature (extreme, otherwise-valid barriers can be temperature-safe
    # only over their declared execution range).
    if isinstance(model, ConstantRateModel):
        if not math.isfinite(model.k) or model.k < 0.0:
            raise ValueError("Constant k must be finite and nonnegative.")
    elif isinstance(model, ArrheniusRateModel):
        if not math.isfinite(model.A) or model.A <= 0.0:
            raise ValueError("Arrhenius A must be positive and finite.")
        if not math.isfinite(model.Ea_J_per_mol):
            raise ValueError("Arrhenius Ea_J_per_mol must be finite.")
    elif isinstance(model, EyringRateModel):
        if not math.isfinite(model.dG_act_J_per_mol):
            raise ValueError("Eyring dG_act_J_per_mol must be finite.")
        if not math.isfinite(model.kappa) or model.kappa <= 0.0:
            raise ValueError("Eyring kappa must be positive and finite.")
        if (
            not math.isfinite(model.standard_conc_M)
            or model.standard_conc_M <= 0.0
        ):
            raise ValueError(
                "Eyring standard concentration must be positive and finite."
            )
    return model


def forward_rate_model_for_reaction(
    reaction: object,
    *,
    default_kappa: float = 1.0,
    default_standard_conc_M: float = 1.0,
) -> ForwardRateModel:
    """Project one mechanism reaction onto the final typed rate model."""

    overrides = getattr(reaction, "overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise TypeError("reaction overrides must be a mapping.")
    family = str(overrides.get("model") or "Constant")
    fields = {
        str(name): value
        for name, value in overrides.items()
        if str(name) != "model"
    }
    return _decode_forward_rate_model(
        fields,
        family=family,
        constant_rate=getattr(reaction, "rate", None),
        molecularity=getattr(reaction, "order", 1),
        default_kappa=default_kappa,
        default_standard_conc_M=default_standard_conc_M,
    )


def forward_rate_model_for_equilibrium(
    equilibrium: object,
    *,
    temperature_K: float,
    default_kappa: float = 1.0,
    default_standard_conc_M: float = 1.0,
) -> ForwardRateModel:
    """Project one equilibrium's forward side onto the same typed model."""

    metadata = getattr(equilibrium, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("equilibrium metadata must be a mapping.")
    raw_model = metadata.get("forward_model")
    if raw_model is not None and not isinstance(raw_model, Mapping):
        raise TypeError("equilibrium forward_model must be a mapping.")
    model_payload = raw_model if isinstance(raw_model, Mapping) else None
    from .equilibrium_rate_authority import effective_equilibrium_forward_rate

    constant_rate = effective_equilibrium_forward_rate(
        equilibrium,
        temperature_K=_numeric_float(
            temperature_K,
            field="temperature_K",
        ),
    )
    molecularity = max(
        1,
        int(
            round(
                sum(
                    float(value)
                    for value in dict(
                        getattr(equilibrium, "stoich_forward", {}) or {}
                    ).values()
                )
            )
        ),
    )
    raw_fields = dict(model_payload or {})
    family = str(raw_fields.pop("type", "Constant") or "Constant")
    return _decode_forward_rate_model(
        raw_fields,
        family=family,
        constant_rate=constant_rate,
        molecularity=molecularity,
        default_kappa=default_kappa,
        default_standard_conc_M=default_standard_conc_M,
    )


def forward_rate_at_temperature(
    model: ForwardRateModel,
    temperature_K: float,
) -> float:
    """Evaluate exactly one typed forward-rate family."""
    if isinstance(model, ConstantRateModel):
        value = _numeric_float(model.k, field="Constant k")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Constant k must be finite and nonnegative.")
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
        )
    raise TypeError("model must be one typed forward-rate model.")


def write_reference_rate(
    model: ForwardRateModel,
    *,
    reference_rate: float,
    reference_temperature_K: float,
) -> ForwardRateModel:
    """Write coupled k@Tref without changing family or creating scalar k."""
    rate = _numeric_float(reference_rate, field="reference_rate")
    temperature = _numeric_float(
        reference_temperature_K,
        field="reference_temperature_K",
    )
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("reference_rate must be finite and nonnegative.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "reference_temperature_K must be positive and finite."
        )
    if isinstance(model, ConstantRateModel):
        return ConstantRateModel(k=rate)
    if rate == 0.0:
        raise ValueError(
            "Energy-authored reference_rate must be positive and finite."
        )
    if isinstance(model, ArrheniusRateModel):
        A = _numeric_float(model.A, field="Arrhenius A")
        if not math.isfinite(A) or A <= 0.0:
            raise ValueError("Arrhenius A must be positive and finite.")
        Ea = -R * temperature * (math.log(rate) - math.log(A))
        if not math.isfinite(Ea):
            raise ValueError("The requested Arrhenius k@Tref is not finite.")
        return ArrheniusRateModel(A=A, Ea_J_per_mol=Ea)
    if isinstance(model, EyringRateModel):
        kappa = _numeric_float(model.kappa, field="Eyring kappa")
        standard = _numeric_float(
            model.standard_conc_M,
            field="Eyring standard_conc_M",
        )
        if type(model.molecularity) is not int:
            raise TypeError("Eyring molecularity must be an int.")
        molecularity = model.molecularity
        if kappa <= 0.0 or not math.isfinite(kappa):
            raise ValueError("Eyring kappa must be positive and finite.")
        if standard <= 0.0 or not math.isfinite(standard):
            raise ValueError(
                "Eyring standard concentration must be positive and finite."
            )
        if molecularity < 1:
            raise ValueError("Eyring molecularity must be at least one.")
        log_ratio = (
            math.log(rate)
            + max(0, molecularity - 1) * math.log(standard)
            - math.log(kappa)
            - math.log(eyring_prefactor(temperature))
        )
        dG = -R * temperature * log_ratio
        if not math.isfinite(dG):
            raise ValueError("The requested Eyring k@Tref is not finite.")
        return EyringRateModel(
            dG_act_J_per_mol=dG,
            kappa=kappa,
            standard_conc_M=standard,
            molecularity=molecularity,
        )
    raise TypeError("model must be one typed forward-rate model.")


def eyring_prefactor(T: float) -> float:
    """(k_B T / h) in 1/s."""
    temperature = _numeric_float(T, field="T")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("T must be positive and finite")
    return (k_B * temperature) / h


def eyring_rate(
    dG_act_J_per_mol: float,
    T: float,
    *,
    kappa: float = 1.0,
    molecularity: int = 1,
    standard_conc_M: float = 1.0,
) -> float:
    """
    Compute Eyring rate constant.

    - Unimolecular: κ * (k_B T / h) * exp(-ΔG‡/(R T))      [1/s]
    - n-molecular (n >= 2): multiply by 1/(C°^(n-1))        [1/(M^(n-1)*s)]
    """
    T = _numeric_float(T, field="T")
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    standard = _numeric_float(
        standard_conc_M,
        field="standard_conc_M",
    )
    if standard <= 0.0 or not math.isfinite(standard):
        raise ValueError("standard_conc_M must be positive and finite")
    if type(molecularity) is not int:
        raise TypeError("molecularity must be an int")
    n = molecularity
    if n < 1:
        raise ValueError("molecularity must be >= 1")

    transmission = _numeric_float(kappa, field="kappa")
    if not math.isfinite(transmission) or transmission <= 0.0:
        raise ValueError("kappa must be positive and finite")

    dG = _numeric_float(
        dG_act_J_per_mol,
        field="dG_act_J_per_mol",
    )
    if not math.isfinite(dG):
        raise ValueError("dG_act_J_per_mol must be finite")
    exponent = -dG / (R * T)
    k = transmission * eyring_prefactor(T) * math.exp(exponent)
    if n >= 2:
        k /= standard ** (n - 1)
    return float(k)


def arrhenius_rate(A: float, Ea_J_per_mol: float, T: float) -> float:
    """
    Arrhenius rate constant:

        k = A * exp(-Ea/(R T))

    Notes
    -----
    - A should already carry the correct order unit (1/s or 1/(M^(n-1)*s)).
    - This function does not attempt to infer molecularity from A.
    """
    T = _numeric_float(T, field="T")
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    prefactor = _numeric_float(A, field="A")
    activation = _numeric_float(Ea_J_per_mol, field="Ea_J_per_mol")
    if not math.isfinite(prefactor) or prefactor <= 0.0:
        raise ValueError("A must be positive and finite")
    if not math.isfinite(activation):
        raise ValueError("Ea_J_per_mol must be finite")
    exponent = -activation / (R * T)
    return float(prefactor * math.exp(exponent))


def K_from_deltaG_eq(dG_eq_J_per_mol: float, T: float) -> float:
    """
    Equilibrium constant from standard Gibbs free energy change:

        K = exp(-ΔG° / (R T))
    """
    T = _numeric_float(T, field="T")
    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("T must be positive and finite")
    dG = _numeric_float(dG_eq_J_per_mol, field="dG_eq_J_per_mol")
    if not math.isfinite(dG):
        raise ValueError("dG_eq_J_per_mol must be finite")
    return float(math.exp(-dG / (R * T)))

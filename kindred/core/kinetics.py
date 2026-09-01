"""
Core kinetics formulas shared across Kindred.

This module intentionally contains the fundamental Arrhenius/Eyring and
equilibrium-constant mappings so both the simulator layer and the ODE builder
can depend on a single source of truth without creating cross-layer imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping, TypeAlias

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
    "forward_rate_model_for_equilibrium",
    "forward_rate_model_for_reaction",
    "forward_rate_symbolic_relation",
    "write_reference_rate",
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
        raise ValueError(f"{field} must be a finite non-negative real.")
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


def _decode_forward_rate_model(
    payload: Mapping[str, object],
    *,
    family: str,
    constant_rate: object | None,
    molecularity: float,
    default_standard_conc_M: float = 1.0,
) -> ForwardRateModel:
    """Decode one final Constant/Arrhenius/Eyring authority.

    ``constant_rate`` is used only for the Constant family. Energy-authored
    families must carry their complete authored coordinates in ``payload``;
    they never fall back to a scalar rate evaluated at one temperature.
    """

    fields = dict(payload)
    order = _directional_order(molecularity)
    allowed_fields = {
        "Constant": frozenset({"k"}),
        "Arrhenius": frozenset({"A", "Ea_J_per_mol"}),
        "Eyring": frozenset(
            {
                "dG_act_J_per_mol",
                "kappa",
                "standard_conc_M",
                "degeneracy",
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
                fields.get("kappa", 1.0),
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
            degeneracy=_positive_integer(
                fields.get("degeneracy", 1),
                field="Eyring degeneracy",
            ),
        )
    # Validate authored coordinates without evaluating at an arbitrary
    # temperature (extreme, otherwise-valid barriers can be temperature-safe
    # only over their declared execution range).
    if isinstance(model, ConstantRateModel):
        admit_rate(model.k, field="Constant k", allow_zero=True)
    elif isinstance(model, ArrheniusRateModel):
        admit_arrhenius_prefactor(model.A, field="Arrhenius A")
        admit_energy_j_per_mol(model.Ea_J_per_mol, field="Arrhenius Ea_J_per_mol")
    elif isinstance(model, EyringRateModel):
        admit_energy_j_per_mol(
            model.dG_act_J_per_mol,
            field="Eyring dG_act_J_per_mol",
        )
        if not math.isfinite(model.kappa) or model.kappa <= 0.0:
            raise ValueError("Eyring kappa must be positive and finite.")
    return model


def forward_rate_model_for_reaction(
    reaction: object,
    *,
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
    constant_rate = getattr(reaction, "rate", None)
    bound_model = getattr(constant_rate, "model", None)
    if isinstance(
        bound_model,
        (ConstantRateModel, ArrheniusRateModel, EyringRateModel),
    ):
        if forward_rate_family(bound_model) != family:
            raise ValueError(
                "Bound reaction rate family conflicts with accepted overrides."
            )
        return bound_model
    if family == "Constant" and callable(constant_rate):
        if bool(
            getattr(
                constant_rate,
                "accepts_runtime_temperature",
                False,
            )
        ):
            raise ValueError(
                "Constant symbolic rate authority cannot depend on runtime temperature."
            )
        constant_rate = constant_rate()
    return _decode_forward_rate_model(
        fields,
        family=family,
        constant_rate=constant_rate,
        molecularity=getattr(reaction, "order", 1),
        default_standard_conc_M=default_standard_conc_M,
    )


def forward_rate_model_for_equilibrium(
    equilibrium: object,
    *,
    temperature_K: float | None,
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

    bound_model = getattr(getattr(equilibrium, "kf", None), "model", None)
    if isinstance(
        bound_model,
        (ConstantRateModel, ArrheniusRateModel, EyringRateModel),
    ):
        raw_family = str(
            dict(model_payload or {}).get("type", "Constant")
            or "Constant"
        )
        if forward_rate_family(bound_model) != raw_family:
            raise ValueError(
                "Bound equilibrium rate family conflicts with accepted metadata."
            )
        return bound_model

    if temperature_K is None:
        constant_rate = getattr(equilibrium, "kf", None)
        if callable(constant_rate):
            if bool(
                getattr(
                    constant_rate,
                    "accepts_runtime_temperature",
                    False,
                )
            ):
                constant_rate = None
            else:
                constant_rate = constant_rate()
    else:
        constant_rate = effective_equilibrium_forward_rate(
            equilibrium,
            temperature_K=_numeric_float(
                temperature_K,
                field="temperature_K",
            ),
        )
    molecularity = _directional_order(
        math.fsum(
            float(value)
            for value in dict(
                getattr(equilibrium, "stoich_forward", {}) or {}
            ).values()
        ),
        field="equilibrium forward directional order",
    )
    raw_fields = dict(model_payload or {})
    family = str(raw_fields.pop("type", "Constant") or "Constant")
    if family == "Constant":
        # ``equilibrium.kf`` is the live accepted coordinate.  The metadata
        # copy records authored family shape and must not become a stale mirror
        # after canonical parameter rebinding.
        raw_fields.pop("k", None)
    return _decode_forward_rate_model(
        raw_fields,
        family=family,
        constant_rate=constant_rate,
        molecularity=molecularity,
        default_standard_conc_M=default_standard_conc_M,
    )


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
    """

    if not callable(literal):
        raise TypeError("literal must be callable.")
    if not callable(exponential):
        raise TypeError("exponential must be callable.")
    if source_coordinate is not None and not callable(source_coordinate):
        raise TypeError("source_coordinate must be callable when provided.")

    def exact(value: object, *, field: str) -> object:
        number = _numeric_float(value, field=field)
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite.")
        return literal(float(number))

    def coordinate(role: str, value: float) -> object:
        """Use a canonical authored coordinate, or the admitted literal."""

        if source_coordinate is not None:
            resolved = source_coordinate(role)
            # A source-coordinate owner may omit a role whose scientific
            # value is a model default.  Omitted Eyring κ and degeneracy
            # remain their literal defaults, while authored values are
            # source-backed coordinates.
            if resolved is not None:
                return resolved
            if role not in {"kappa", "degeneracy"}:
                raise ValueError(
                    f"missing canonical source coordinate for {role!r}"
                )
        return literal(float(value))

    if isinstance(model, ConstantRateModel):
        value = _numeric_float(model.k, field="Constant k")
        admit_rate(value, field="Constant k", allow_zero=True)
        return coordinate("kf", value)
    if isinstance(model, ArrheniusRateModel):
        prefactor = _numeric_float(model.A, field="Arrhenius A")
        activation = _numeric_float(
            model.Ea_J_per_mol,
            field="Arrhenius Ea_J_per_mol",
        )
        prefactor = admit_arrhenius_prefactor(prefactor, field="Arrhenius A")
        activation = admit_energy_j_per_mol(
            activation,
            field="Arrhenius Ea_J_per_mol",
        )
        return coordinate("A", prefactor) * exponential(  # type: ignore[operator]
            -coordinate("Ea", activation)
            / (exact(R, field="R") * temperature)  # type: ignore[operator]
        )
    if isinstance(model, EyringRateModel):
        barrier = _numeric_float(
            model.dG_act_J_per_mol,
            field="Eyring dG_act_J_per_mol",
        )
        transmission = _numeric_float(model.kappa, field="Eyring kappa")
        barrier = admit_energy_j_per_mol(
            barrier,
            field="Eyring dG_act_J_per_mol",
        )
        if not math.isfinite(transmission) or transmission <= 0.0:
            raise ValueError("Eyring kappa must be positive and finite.")
        standard = _numeric_float(
            model.standard_conc_M,
            field="Eyring standard_conc_M",
        )
        standard = validate_c0(standard, field="Eyring standard_conc_M")
        order = _directional_order(
            model.molecularity,
            field="Eyring molecularity",
        )
        degeneracy = _positive_integer(
            model.degeneracy,
            field="Eyring degeneracy",
        )
        degeneracy_relation = coordinate("degeneracy", float(degeneracy))
        kappa_relation = coordinate("kappa", transmission)
        concentration_relation = (
            literal(float(standard))
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
                -coordinate("dG_act", barrier)
                / (exact(R, field="R") * temperature)  # type: ignore[operator]
            )
        )
        standard_exponent = Fraction(1) - Fraction(str(order))
        if standard_exponent:
            relation *= concentration_relation ** literal(standard_exponent)  # type: ignore[operator]
        return relation
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
    rate = admit_rate(rate, field="reference_rate")
    temperature = validate_temperature(
        temperature,
        field="reference_temperature_K",
    )
    if isinstance(model, ConstantRateModel):
        return ConstantRateModel(k=rate)
    if isinstance(model, ArrheniusRateModel):
        A = _numeric_float(model.A, field="Arrhenius A")
        A = admit_arrhenius_prefactor(A, field="Arrhenius A")
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
        molecularity = _directional_order(
            model.molecularity,
            field="Eyring molecularity",
        )
        degeneracy = _positive_integer(
            model.degeneracy,
            field="Eyring degeneracy",
        )
        if kappa <= 0.0 or not math.isfinite(kappa):
            raise ValueError("Eyring kappa must be positive and finite.")
        standard = validate_c0(
            standard,
            field="Eyring standard_conc_M",
        )
        log_ratio = (
            math.log(rate)
            - math.log(degeneracy)
            - (1.0 - molecularity) * math.log(standard)
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
            degeneracy=degeneracy,
        )
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

    - Unimolecular: g * κ * (k_B T / h) * exp(-ΔG‡/(R T))  [1/s]
    - n-molecular (n >= 2): multiply by 1/(C°^(n-1))        [1/(M^(n-1)*s)]
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

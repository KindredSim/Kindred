# kindred/core/ode_builder.py
"""
ODE system builder from Mechanism objects.

This module ties Mechanism data to kinetics and solver plumbing. It constructs
temperature-aware dy/dt = f(t, y, T=...) callables from mechanism stoichiometry
plus Arrhenius/Eyring models in kindred.core.kinetics. Equilibria stay reversible
and are evaluated as net forward minus reverse rates (no reaction duplication).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np

from .equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    GENERATED_RATE_MODEL_ERROR,
    effective_equilibrium_forward_rate,
    effective_equilibrium_keq,
    effective_equilibrium_reverse_rate,
    generated_equilibrium_authority_kind,
    json_safe_equilibrium_authority_payload,
    normalize_existing_equilibrium_rate_authority,
)
from .mechanism import Equilibrium, FrozenDict, Mechanism, Reaction
from .rate_binding import ForwardRateBinding, RateBinding
from .kinetics import (
    ArrheniusRateModel,
    ConstantRateModel,
    EyringRateModel,
    ForwardRateModel,
    forward_rate_at_temperature,
    forward_rate_family,
    forward_rate_model_for_equilibrium,
    forward_rate_model_for_reaction,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EffectiveRateCoefficients",
    "EffectiveRateCoefficientEvaluator",
    "RateBinding",
    "build_effective_rate_coefficient_evaluator",
    "build_ode_rhs_from_mechanism",
]


def _stoichiometric_flux_product(
    stoichiometry: np.ndarray,
    rates: np.ndarray,
    *,
    workspace: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute species fluxes without entering the BLAS matrix-vector path."""
    stoich_arr = np.asarray(stoichiometry, dtype=float)
    rate_arr = np.asarray(rates, dtype=float).reshape(-1)
    if workspace is None:
        products = np.multiply(stoich_arr, rate_arr)
    else:
        products = workspace
        np.multiply(stoich_arr, rate_arr, out=products)
    if out is None:
        return np.sum(products, axis=1, dtype=float)
    np.sum(products, axis=1, out=out)
    return out.copy()


def _evaluate_rate_binding(binding: RateBinding, temperature_K: float) -> float:
    if bool(getattr(binding, "accepts_runtime_temperature", False)):
        return float(binding(float(temperature_K)))  # type: ignore[call-arg]
    return float(binding())


def _forward_rate_model_payload(model: ForwardRateModel) -> dict[str, object]:
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
            "molecularity": int(model.molecularity),
        }
    else:
        raise TypeError("Unsupported typed forward-rate model.")
    return {"family": family, **fields}


@dataclass(frozen=True, slots=True)
class EffectiveRateCoefficients:
    """Immutable effective forward/reverse coefficients for one temperature."""

    temperature_K: float
    forward: tuple[float, ...]
    reverse: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EffectiveRateCoefficientEvaluator:
    """One typed per-mechanism runtime coefficient authority."""

    nominal_temperature_K: float
    step_kinds: tuple[str, ...]
    authority_snapshot: tuple[Mapping[str, object], ...]
    _evaluators: tuple[Callable[[float], tuple[float, float]], ...]

    def __post_init__(self) -> None:
        frozen_snapshot: list[Mapping[str, object]] = []
        for item in self.authority_snapshot:
            if not isinstance(item, Mapping):
                raise TypeError(
                    "Effective rate authority snapshots must be mappings."
                )
            frozen_snapshot.append(
                item if isinstance(item, FrozenDict) else FrozenDict(item)
            )
        object.__setattr__(
            self,
            "authority_snapshot",
            tuple(frozen_snapshot),
        )
        object.__setattr__(self, "step_kinds", tuple(self.step_kinds))
        object.__setattr__(self, "_evaluators", tuple(self._evaluators))

    def __call__(self, temperature_K: float) -> EffectiveRateCoefficients:
        temperature = float(temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Runtime temperature must be positive and finite.")
        pairs = tuple(evaluator(temperature) for evaluator in self._evaluators)
        forward = tuple(float(pair[0]) for pair in pairs)
        reverse = tuple(float(pair[1]) for pair in pairs)
        values = (*forward, *reverse)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Effective rate coefficients must be finite and nonnegative."
            )
        return EffectiveRateCoefficients(
            temperature_K=temperature,
            forward=forward,
            reverse=reverse,
        )


def build_effective_rate_coefficient_evaluator(
    mechanism: Mechanism,
) -> EffectiveRateCoefficientEvaluator:
    """Build the sole effective-rate authority consumed by RHS and Jacobian."""

    from .mechanism_metadata import MechanismMetadataView

    mech_meta = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    nominal_temperature = float(mech_meta.temperature_K)
    default_standard_conc = float(mech_meta.standard_conc_M)
    default_kappa = float(mech_meta.kappa_global)
    evaluators: list[Callable[[float], tuple[float, float]]] = []
    step_kinds: list[str] = []
    snapshots: list[Mapping[str, object]] = []

    for reaction in mechanism.reactions:
        binding = (
            reaction.rate
            if isinstance(reaction.rate, RateBinding)
            else None
        )
        if binding is not None:
            if isinstance(binding, ForwardRateBinding):
                snapshot = {
                    "kind": "reaction",
                    "binding_name": str(binding.name),
                    "forward_model": _forward_rate_model_payload(
                        binding.model
                    ),
                }
            else:
                snapshot = {
                    "kind": "reaction",
                    "binding_name": str(binding.name),
                    "binding_value": float(binding()),
                }

            def reaction_coefficients(
                temperature_K: float,
                *,
                _binding: RateBinding = binding,
            ) -> tuple[float, float]:
                return (
                    _evaluate_rate_binding(_binding, temperature_K),
                    0.0,
                )

        else:
            model = forward_rate_model_for_reaction(
                reaction,
                default_kappa=default_kappa,
                default_standard_conc_M=default_standard_conc,
            )
            snapshot = {
                "kind": "reaction",
                "forward_model": _forward_rate_model_payload(model),
            }

            def reaction_coefficients(
                temperature_K: float,
                *,
                _model: ForwardRateModel = model,
            ) -> tuple[float, float]:
                return (
                    forward_rate_at_temperature(_model, temperature_K),
                    0.0,
                )

        evaluators.append(reaction_coefficients)
        step_kinds.append("reaction")
        snapshots.append(snapshot)

    for equilibrium in mechanism.equilibria:
        metadata = getattr(equilibrium, "metadata", {}) or {}
        generated_kind = generated_equilibrium_authority_kind(metadata)
        if generated_kind:
            snapshot = {
                "kind": "equilibrium",
                "generated_authority": str(generated_kind),
                "authority": json_safe_equilibrium_authority_payload(
                    metadata
                ),
            }

            def equilibrium_coefficients(
                temperature_K: float,
                *,
                _equilibrium: Equilibrium = equilibrium,
            ) -> tuple[float, float]:
                forward = effective_equilibrium_forward_rate(
                    _equilibrium,
                    temperature_K=temperature_K,
                )
                reverse = effective_equilibrium_reverse_rate(
                    _equilibrium,
                    temperature_K=temperature_K,
                )
                if forward is None or reverse is None:
                    raise ValueError(GENERATED_RATE_MODEL_ERROR)
                return float(forward), float(reverse)

        else:
            authority = normalize_existing_equilibrium_rate_authority(
                equilibrium
            )
            forward_binding = (
                equilibrium.kf
                if isinstance(equilibrium.kf, RateBinding)
                else None
            )
            if isinstance(forward_binding, ForwardRateBinding):
                forward_model = forward_binding.model
            else:
                forward_model = forward_rate_model_for_equilibrium(
                    equilibrium,
                    temperature_K=nominal_temperature,
                    default_kappa=default_kappa,
                    default_standard_conc_M=default_standard_conc,
                )
            snapshot = {
                "kind": "equilibrium",
                "forward_model": _forward_rate_model_payload(forward_model),
                "forward_binding_name": (
                    str(forward_binding.name)
                    if isinstance(forward_binding, RateBinding)
                    else None
                ),
                "authority": json_safe_equilibrium_authority_payload(
                    dict(authority.identity_items())
                ),
                "reverse_binding_name": (
                    str(authority.kr.name)
                    if isinstance(authority.kr, RateBinding)
                    else None
                ),
                "reverse_model": (
                    _forward_rate_model_payload(authority.kr.model)
                    if isinstance(authority.kr, ForwardRateBinding)
                    else None
                ),
            }

            def equilibrium_coefficients(
                temperature_K: float,
                *,
                _equilibrium: Equilibrium = equilibrium,
                _authority=authority,
                _forward_binding: RateBinding | None = forward_binding,
                _forward_model: ForwardRateModel = forward_model,
            ) -> tuple[float, float]:
                forward = (
                    _evaluate_rate_binding(
                        _forward_binding,
                        temperature_K,
                    )
                    if _forward_binding is not None
                    else forward_rate_at_temperature(
                        _forward_model,
                        temperature_K,
                    )
                )
                if _authority.kind == EquilibriumRateAuthorityKind.KR:
                    reverse = _authority.effective_reverse_rate()
                    if isinstance(reverse, RateBinding):
                        reverse = _evaluate_rate_binding(
                            reverse,
                            temperature_K,
                        )
                    elif callable(reverse):
                        reverse = float(reverse())
                else:
                    equilibrium_constant = effective_equilibrium_keq(
                        _equilibrium,
                        temperature_K=temperature_K,
                    )
                    if equilibrium_constant is None:
                        raise ValueError(
                            "Equilibrium missing usable Keq/dG_eq authority."
                        )
                    reverse = _authority.effective_reverse_rate(
                        Keq=equilibrium_constant,
                        kf=forward,
                    )
                return float(forward), float(reverse)

        evaluators.append(equilibrium_coefficients)
        step_kinds.append("equilibrium")
        snapshots.append(snapshot)

    return EffectiveRateCoefficientEvaluator(
        nominal_temperature_K=nominal_temperature,
        step_kinds=tuple(step_kinds),
        authority_snapshot=tuple(snapshots),
        _evaluators=tuple(evaluators),
    )


def build_ode_rhs_from_mechanism(
    mechanism: Mechanism
) -> Callable[..., np.ndarray]:
    """
    Build ODE right-hand side function from a Mechanism.

    Parameters
    ----------
    mechanism : Mechanism
        Mechanism with species, reactions, and equilibria

    Returns
    -------
    callable
        Function f(t, y, T=...) -> dy/dt where y is concentration vector
        in mechanism.species_names() order

    Notes
    -----
    - Rates come from Mechanism.rate bindings or Arrhenius/Eyring overrides in
      kinetics.py; T defaults to mechanism.metadata["temperature_K"] if omitted.
    - Reactions contribute: rate = k * ∏[reactants]^stoich
    - Equilibria remain a single reversible step with rate = v_fwd - v_rev
      (stoichiometry is not duplicated into two irreversible columns).
    - Stoichiometry determines species changes via a BLAS-free reduction.
    """
    species_names = mechanism.species_names()
    n_species = len(species_names)
    species_index = {name: idx for idx, name in enumerate(species_names)}
    coefficient_evaluator = build_effective_rate_coefficient_evaluator(
        mechanism
    )
    default_temperature = float(
        coefficient_evaluator.nominal_temperature_K
    )

    # Build list of all steps (irreversible reactions + equilibria)
    steps: List[Tuple[str, Reaction | Equilibrium]] = []
    for rxn in mechanism.reactions:
        steps.append(("reaction", rxn))
    for eq in mechanism.equilibria:
        steps.append(("equilibrium", eq))

    n_steps = len(steps)
    S = np.zeros((n_species, n_steps))
    exp_forward = np.zeros((n_steps, n_species), dtype=float)
    exp_reverse = np.zeros((n_steps, n_species), dtype=float)

    for i_step, (step_type, step_obj) in enumerate(steps):
        if step_type == "reaction":
            rxn = step_obj  # type: ignore[assignment]
            vec = rxn.net_stoich_vector(species_names)
            S[:, i_step] = vec

            for sp_name, order in rxn.rate_orders.items():
                idx = species_index[sp_name]
                exp_forward[i_step, idx] = float(order)

        elif step_type == "equilibrium":
            eq = step_obj  # type: ignore[assignment]
            fwd_vec = np.array(eq.forward_vector(species_names))
            back_vec = np.array(eq.back_vector(species_names))
            S[:, i_step] = back_vec - fwd_vec

            for sp_name, stoich_coef in eq.stoich_forward.items():
                exp_forward[i_step, species_index[sp_name]] = stoich_coef
            for sp_name, stoich_coef in eq.stoich_back.items():
                exp_reverse[i_step, species_index[sp_name]] = stoich_coef
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    forward_mask = exp_forward != 0.0
    reverse_mask = exp_reverse != 0.0
    forward_buffer = np.ones_like(exp_forward)
    reverse_buffer = np.ones_like(exp_reverse)

    k_forward_dyn = np.zeros(n_steps, dtype=float)
    k_reverse_dyn = np.zeros(n_steps, dtype=float)
    has_reverse_terms = bool(mechanism.equilibria)

    # Buffers for stable reversible net rate evaluation (avoids overflow and inf-inf cancellation).
    log_y = np.empty(n_species, dtype=float)
    log_k_forward_dyn = np.empty(n_steps, dtype=float)
    log_k_reverse_dyn = np.empty(n_steps, dtype=float)
    log_prod_forward = np.empty(n_steps, dtype=float)
    log_prod_reverse = np.empty(n_steps, dtype=float)
    log_rate_forward = np.empty(n_steps, dtype=float)
    log_rate_reverse = np.empty(n_steps, dtype=float)
    net_rates = np.empty(n_steps, dtype=float)
    stoich_rate_buffer = np.empty_like(S)
    stoich_flux = np.empty(n_species, dtype=float)

    def _stable_diff_exp_logs(log_f: np.ndarray, log_r: np.ndarray, out: np.ndarray) -> None:
        """
        Compute exp(log_f) - exp(log_r) stably from logs.

        This avoids `inf - inf -> nan` when both sides overflow, and reduces
        catastrophic cancellation when rates are nearly equal.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out.fill(0.0)
            max_lr = np.maximum(log_f, log_r)
            min_lr = np.minimum(log_f, log_r)
            mask = np.isfinite(max_lr)
            if not np.any(mask):
                return
            d = min_lr[mask] - max_lr[mask]  # <= 0
            small = -np.expm1(d)  # 1 - exp(d), in [0, 1]
            log_mag = max_lr[mask] + np.log(small)  # log(0) -> -inf => mag 0
            mag = np.exp(log_mag)
            sign = np.where(log_f[mask] >= log_r[mask], 1.0, -1.0)
            out[mask] = sign * mag

    def ode_rhs(t: float, y: np.ndarray, *, T: Optional[float] = None) -> np.ndarray:
        """ODE right-hand side from stoichiometry and reaction rates."""
        y_arr = y
        T_eval = default_temperature if T is None else float(T)
        coefficients = coefficient_evaluator(T_eval)
        k_forward_dyn[:] = coefficients.forward
        k_reverse_dyn[:] = coefficients.reverse

        if not has_reverse_terms:
            forward_buffer.fill(1.0)
            np.power(y_arr, exp_forward, out=forward_buffer, where=forward_mask)
            rates_forward = k_forward_dyn * forward_buffer.prod(axis=1)
            return _stoichiometric_flux_product(
                S,
                rates_forward,
                workspace=stoich_rate_buffer,
                out=stoich_flux,
            )

        # Reversible/equilibrium path: work in log-space and combine stably.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_y.fill(-np.inf)
            pos = y_arr > 0.0
            log_y[pos] = np.log(y_arr[pos])

            log_k_forward_dyn.fill(-np.inf)
            np.log(k_forward_dyn, out=log_k_forward_dyn, where=(k_forward_dyn > 0.0))
            log_k_reverse_dyn.fill(-np.inf)
            np.log(k_reverse_dyn, out=log_k_reverse_dyn, where=(k_reverse_dyn > 0.0))

            forward_buffer.fill(0.0)
            np.multiply(exp_forward, log_y, out=forward_buffer, where=forward_mask)
            np.sum(forward_buffer, axis=1, out=log_prod_forward)
            np.add(log_k_forward_dyn, log_prod_forward, out=log_rate_forward)

            reverse_buffer.fill(0.0)
            np.multiply(exp_reverse, log_y, out=reverse_buffer, where=reverse_mask)
            np.sum(reverse_buffer, axis=1, out=log_prod_reverse)
            np.add(log_k_reverse_dyn, log_prod_reverse, out=log_rate_reverse)

        _stable_diff_exp_logs(log_rate_forward, log_rate_reverse, net_rates)
        return _stoichiometric_flux_product(
            S,
            net_rates,
            workspace=stoich_rate_buffer,
            out=stoich_flux,
        )

    return ode_rhs

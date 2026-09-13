# kindred/core/ode_builder.py
"""
ODE system builder from Mechanism objects.

This module ties Mechanism data to kinetics and solver plumbing. It constructs
temperature-aware dy/dt = f(t, y, T=...) callables from mechanism stoichiometry
plus Arrhenius/Eyring models in kindred.core.kinetics. Equilibria stay reversible
and are evaluated as net forward minus reverse rates (no reaction duplication).
"""

from __future__ import annotations

from kindred.core.kinetics import forward_rate_model_payload

import math
import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from .equilibrium_rate_authority import (
    json_safe_equilibrium_authority_payload,
    normalize_existing_equilibrium_rate_authority,
)
from .exceptions import KineticRuntimeDomainError
from .mechanism import Equilibrium, FrozenDict, Mechanism, Reaction
from .species import validate_temperature
from .simulator.solver_types import (
    SimulationScientificUnavailability,
    _SimulationScientificUnavailabilitySignal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CompiledOdeTopology",
    "EffectiveRateCoefficients",
    "EffectiveRateCoefficientEvaluator",
    "build_effective_rate_coefficient_evaluator",
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
    parameter_values: Callable[[float, tuple[str, ...]], dict[str, float]]
    _parameter_names: tuple[tuple[str, str | None], ...]

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

    def __call__(self, temperature_K: float) -> EffectiveRateCoefficients:
        def failure_details(index: int | None = None) -> dict[str, object]:
            details: dict[str, object] = {}
            if index is not None:
                details["step_index"] = int(index)
                if index < len(self.step_kinds):
                    details["step_kind"] = str(self.step_kinds[index])
                snapshot = (
                    self.authority_snapshot[index]
                    if index < len(self.authority_snapshot)
                    else {}
                )
                names = tuple(
                    sorted(
                        {
                            str(snapshot.get(key)).strip()
                            for key in (
                                "binding_name",
                                "forward_binding_name",
                                "reverse_binding_name",
                            )
                            if snapshot.get(key) is not None
                            and str(snapshot.get(key)).strip()
                        }
                    )
                )
                details["authority_names"] = names
                details["parameter_names"] = names
                details["value_dependencies"] = names
            return details

        try:
            temperature = validate_temperature(
                temperature_K,
                field="Runtime temperature",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise KineticRuntimeDomainError(
                f"Runtime temperature is outside the admitted domain: {exc}",
                stage="scheduled_temperature",
                details=failure_details(),
            ) from exc
        roots = tuple(name for pair in self._parameter_names for name in pair if name is not None)
        values = self.parameter_values(temperature, roots)
        forward = tuple(float(values[name]) for name, _reverse in self._parameter_names)
        reverse = tuple(float(values[name]) if name else 0.0 for _forward, name in self._parameter_names)
        if not (
            len(self.step_kinds)
            == len(forward)
            == len(reverse)
        ):
            raise ValueError(
                "Effective rate coefficient shape conflicts with mechanism steps."
            )
        for index, (kind, forward_value, reverse_value) in enumerate(zip(
            self.step_kinds,
            forward,
            reverse,
            strict=True,
        )):
            if not math.isfinite(forward_value) or forward_value < 0.0:
                raise KineticRuntimeDomainError(
                    "Effective forward rate coefficients must be finite and nonnegative.",
                    parameter_names=tuple(failure_details(index).get("parameter_names", ())),
                    details=failure_details(index),
                )
            if kind == "equilibrium":
                if not math.isfinite(reverse_value) or reverse_value < 0.0:
                    raise KineticRuntimeDomainError(
                        "Effective reverse rate coefficients must be finite and nonnegative.",
                        parameter_names=tuple(failure_details(index).get("parameter_names", ())),
                        details=failure_details(index),
                    )
            elif kind == "reaction":
                if reverse_value != 0.0:
                    raise ValueError(
                        "Irreversible reaction coefficient rows cannot carry a reverse rate."
                    )
            else:
                raise ValueError(
                    f"Unsupported effective-rate step kind {kind!r}."
                )
        return EffectiveRateCoefficients(
            temperature_K=temperature,
            forward=forward,
            reverse=reverse,
        )

def build_effective_rate_coefficient_evaluator(
    mechanism: Mechanism,
    *,
    base_values: Mapping[str, float] | None = None,
) -> EffectiveRateCoefficientEvaluator:
    """Build the sole effective-rate authority consumed by RHS and Jacobian."""

    from .mechanism_metadata import MechanismMetadataView

    mech_meta = MechanismMetadataView.from_metadata(
        getattr(mechanism, "metadata", {}) or {}
    )
    nominal_temperature = float(mech_meta.temperature_K)
    from .simulator.parameter_algebra_eval import bind_parameter_evaluation
    from .simulator.parameter_algebra_spec import admit_parameter_evaluation_graph, parse_parameter_algebra_spec_from_dsl_text
    from .simulator.parameter_namespace import build_namespace_from_mechanism
    from .simulator.step_indexing import get_step_index_map

    spec = mechanism.metadata.get("parameter_algebra_spec")
    if spec is None:
        spec = admit_parameter_evaluation_graph(parse_parameter_algebra_spec_from_dsl_text(
            "", mechanism_namespace=build_namespace_from_mechanism(mechanism)), mechanism)
    evaluate = bind_parameter_evaluation(spec, mechanism, base_values=base_values)
    entries = {(entry["kind"], int(entry["reaction_index" if entry["kind"] == "reaction" else "equilibrium_index"])): entry
               for entry in get_step_index_map(mechanism)}
    step_kinds = []
    snapshots = []
    names = []
    for kind, steps in (("reaction", mechanism.reactions), ("equilibrium", mechanism.equilibria)):
        for index, step in enumerate(steps):
            number = entries[kind, index]["step_index"]
            forward_name = f"k{number}" if kind == "reaction" else f"kf{number}"
            reverse_name = None if kind == "reaction" else f"kr{number}"
            model = (step.forward_model
                     if kind == "reaction" else step.forward_model)
            snapshot = {"kind": kind, "forward_model": forward_rate_model_payload(model),
                        "binding_name" if kind == "reaction" else "forward_binding_name": forward_name,
                        "value_dependencies": tuple(spec.evaluation_dependencies.get(forward_name, ())),
                        "temperature_dependent": forward_name in spec.temperature_dependent_names}
            if kind == "equilibrium":
                authority = normalize_existing_equilibrium_rate_authority(step)
                snapshot.update(authority=json_safe_equilibrium_authority_payload(dict(authority.identity_items())), reverse_binding_name=reverse_name)
            names.append((forward_name, reverse_name))
            step_kinds.append(kind)
            snapshots.append(snapshot)
    return EffectiveRateCoefficientEvaluator(
        nominal_temperature_K=nominal_temperature, step_kinds=tuple(step_kinds),
        authority_snapshot=tuple(snapshots), parameter_values=evaluate, _parameter_names=tuple(names),
    )


@dataclass(frozen=True, slots=True)
class CompiledOdeTopology:
    """Mechanism-invariant numerical lowering shared by execution targets.

    The arrays in this value are structural and read-only.  ``bind_target``
    creates a target-local coefficient evaluator and RHS closure; none of the
    mutable solver workspace or rate bindings is retained here.
    """

    species_names: tuple[str, ...]
    step_kinds: tuple[str, ...]
    stoichiometry: np.ndarray
    forward_exponents: np.ndarray
    reverse_exponents: np.ndarray
    forward_mask: np.ndarray
    reverse_mask: np.ndarray
    fractional_forward_mask: np.ndarray
    fractional_reverse_mask: np.ndarray
    reaction_count: int

    @classmethod
    def from_mechanism(cls, mechanism: Mechanism) -> "CompiledOdeTopology":
        species_names = tuple(mechanism.species_names())
        n_species = len(species_names)
        species_index = {name: idx for idx, name in enumerate(species_names)}
        steps: list[tuple[str, Reaction | Equilibrium]] = [
            *(("reaction", rxn) for rxn in mechanism.reactions),
            *(("equilibrium", eq) for eq in mechanism.equilibria),
        ]
        n_steps = len(steps)
        stoichiometry = np.zeros((n_species, n_steps), dtype=float)
        forward_exponents = np.zeros((n_steps, n_species), dtype=float)
        reverse_exponents = np.zeros((n_steps, n_species), dtype=float)
        step_kinds: list[str] = []
        for index, (kind, step) in enumerate(steps):
            step_kinds.append(kind)
            if kind == "reaction":
                reaction = step
                stoichiometry[:, index] = reaction.net_stoich_vector(species_names)
                for name, order in reaction.rate_orders.items():
                    forward_exponents[index, species_index[name]] = float(order)
            else:
                equilibrium = step
                stoichiometry[:, index] = np.asarray(
                    equilibrium.back_vector(species_names), dtype=float
                ) - np.asarray(equilibrium.forward_vector(species_names), dtype=float)
                for name, coefficient in equilibrium.stoich_forward.items():
                    forward_exponents[index, species_index[name]] = float(coefficient)
                for name, coefficient in equilibrium.stoich_back.items():
                    reverse_exponents[index, species_index[name]] = float(coefficient)
        forward_mask = forward_exponents != 0.0
        reverse_mask = reverse_exponents != 0.0
        fractional_forward_mask = forward_mask & (
            forward_exponents != np.rint(forward_exponents)
        )
        fractional_reverse_mask = reverse_mask & (
            reverse_exponents != np.rint(reverse_exponents)
        )
        arrays = (
            stoichiometry,
            forward_exponents,
            reverse_exponents,
            forward_mask,
            reverse_mask,
            fractional_forward_mask,
            fractional_reverse_mask,
        )
        for array in arrays:
            array.setflags(write=False)
        return cls(
            species_names=species_names,
            step_kinds=tuple(step_kinds),
            stoichiometry=stoichiometry,
            forward_exponents=forward_exponents,
            reverse_exponents=reverse_exponents,
            forward_mask=forward_mask,
            reverse_mask=reverse_mask,
            fractional_forward_mask=fractional_forward_mask,
            fractional_reverse_mask=fractional_reverse_mask,
            reaction_count=len(mechanism.reactions),
        )

    def bind_target(
        self,
        mechanism: Mechanism,
        *,
        base_values: Mapping[str, float] | None = None,
    ) -> tuple[EffectiveRateCoefficientEvaluator, Callable[..., np.ndarray]]:
        """Bind one target's rate authority and mutable RHS workspace."""

        if tuple(mechanism.species_names()) != self.species_names:
            raise ValueError("Target mechanism species order conflicts with compiled ODE topology.")
        if len(mechanism.reactions) + len(mechanism.equilibria) != len(self.step_kinds):
            raise ValueError("Target mechanism step order conflicts with compiled ODE topology.")
        coefficient_evaluator = build_effective_rate_coefficient_evaluator(
            mechanism, base_values=base_values
        )
        return coefficient_evaluator, _build_bound_ode_rhs(
            self,
            mechanism,
            coefficient_evaluator,
        )


def _build_bound_ode_rhs(
    topology: CompiledOdeTopology,
    mechanism: Mechanism,
    coefficient_evaluator: EffectiveRateCoefficientEvaluator,
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
    - Rates come from the admitted typed forward model and reverse authority;
      T defaults to the declared reference temperature if omitted.
    - Reactions contribute: rate = k * ∏[reactants]^stoich
    - Equilibria remain a single reversible step with rate = v_fwd - v_rev
      (stoichiometry is not duplicated into two irreversible columns).
    - Stoichiometry determines species changes via a BLAS-free reduction.
    """
    species_names = topology.species_names
    n_species = len(species_names)
    default_temperature = float(
        coefficient_evaluator.nominal_temperature_K
    )

    n_steps = len(topology.step_kinds)
    n_reactions = topology.reaction_count
    S = topology.stoichiometry
    exp_forward = topology.forward_exponents
    exp_reverse = topology.reverse_exponents
    forward_mask = topology.forward_mask
    reverse_mask = topology.reverse_mask
    fractional_forward_mask = topology.fractional_forward_mask
    fractional_reverse_mask = topology.fractional_reverse_mask
    forward_buffer = np.ones_like(exp_forward)
    forward_power_bases = np.empty_like(exp_forward)
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

    def _runtime_dependency_details(indices: object = ()) -> dict[str, object]:
        selected = np.asarray(indices, dtype=int).reshape(-1)
        names: list[str] = []
        step_names: list[str] = []
        for index in selected:
            idx = int(index)
            if 0 <= idx < len(topology.step_kinds):
                step_type = topology.step_kinds[idx]
                step_names.append(f"{step_type}{idx}")
                names.extend(name for name in coefficient_evaluator._parameter_names[idx]
                             if name is not None)
        payload = {
            "step_names": tuple(step_names),
            "step_indices": tuple(int(index) for index in selected),
            "authority_names": tuple(sorted(set(names))),
            "parameter_names": tuple(sorted(set(names))),
            "value_dependencies": tuple(sorted(set(names))),
        }
        return payload

    def _stable_diff_exp_logs(log_f: np.ndarray, log_r: np.ndarray, out: np.ndarray) -> None:
        """
        Compute exp(log_f) - exp(log_r) stably from logs.

        This avoids `inf - inf -> nan` when both sides overflow, and reduces
        catastrophic cancellation when rates are nearly equal.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out.fill(0.0)
            if np.any(np.isnan(log_f)) or np.any(np.isnan(log_r)):
                raise _SimulationScientificUnavailabilitySignal(
                    SimulationScientificUnavailability(
                        producer="ode_builder",
                        stage="reversible_rate_logarithm",
                        details=_runtime_dependency_details(
                            np.flatnonzero(np.isnan(log_f) | np.isnan(log_r))
                        ),
                    )
                )
            max_lr = np.maximum(log_f, log_r)
            min_lr = np.minimum(log_f, log_r)
            finite_max = np.isfinite(max_lr)
            unrepresentable = ~finite_max & (max_lr > 0.0)
            if np.any(unrepresentable):
                raise _SimulationScientificUnavailabilitySignal(
                    SimulationScientificUnavailability(
                        producer="ode_builder",
                        stage="reversible_rate_difference",
                        details=_runtime_dependency_details(
                            np.flatnonzero(unrepresentable)
                        ),
                    )
                )
            if not np.any(finite_max):
                # Both rates are exactly zero in log space (-inf - -inf).
                return
            indices = np.flatnonzero(finite_max)
            d = min_lr[indices] - max_lr[indices]  # <= 0
            small = -np.expm1(d)  # 1 - exp(d), in [0, 1]
            log_mag = max_lr[indices] + np.log(small)
            mag = np.exp(log_mag)
            if np.any(~np.isfinite(mag)):
                raise _SimulationScientificUnavailabilitySignal(
                    SimulationScientificUnavailability(
                        producer="ode_builder",
                        stage="reversible_rate_difference",
                        details=_runtime_dependency_details(
                            indices
                        ),
                    )
                )
            sign = np.where(log_f[indices] >= log_r[indices], 1.0, -1.0)
            out[indices] = sign * mag

    def ode_rhs(t: float, y: np.ndarray, *, T: Optional[float] = None) -> np.ndarray:
        """ODE right-hand side from stoichiometry and reaction rates."""
        y_arr = y
        T_eval = default_temperature if T is None else float(T)
        coefficients = coefficient_evaluator(T_eval)
        k_forward_dyn[:] = coefficients.forward
        k_reverse_dyn[:] = coefficients.reverse

        if not has_reverse_terms:
            forward_buffer.fill(1.0)
            if np.any(
                fractional_forward_mask & (y_arr[np.newaxis, :] < 0.0)
            ):
                # Implicit solvers may briefly trial a negative internal state.
                # Restrict only non-integral power evaluation to its physical
                # domain; this leaves both the solver state and integer powers
                # unchanged and does not alter published concentration truth.
                forward_power_bases[:] = y_arr
                np.maximum(
                    forward_power_bases,
                    0.0,
                    out=forward_power_bases,
                    where=fractional_forward_mask,
                )
                np.power(
                    forward_power_bases,
                    exp_forward,
                    out=forward_buffer,
                    where=forward_mask,
                )
            else:
                np.power(
                    y_arr,
                    exp_forward,
                    out=forward_buffer,
                    where=forward_mask,
                )
            rates_forward = k_forward_dyn * forward_buffer.prod(axis=1)
            return _stoichiometric_flux_product(
                S,
                rates_forward,
                workspace=stoich_rate_buffer,
                out=stoich_flux,
            )

        if np.any(y_arr < 0.0):
            # The log-domain path below cannot retain the sign of integer
            # powers at a solver's negative trial state.  Evaluate those
            # powers directly while limiting only non-integral powers to a
            # scratch physical-domain base; leave y_arr itself untouched.
            forward_buffer.fill(1.0)
            forward_power_bases[:] = y_arr
            np.maximum(
                forward_power_bases,
                0.0,
                out=forward_power_bases,
                where=fractional_forward_mask,
            )
            np.power(
                forward_power_bases,
                exp_forward,
                out=forward_buffer,
                where=forward_mask,
            )
            np.prod(forward_buffer, axis=1, out=log_prod_forward)
            np.multiply(k_forward_dyn, log_prod_forward, out=net_rates)

            reverse_buffer.fill(1.0)
            forward_power_bases[:] = y_arr
            np.maximum(
                forward_power_bases,
                0.0,
                out=forward_power_bases,
                where=fractional_reverse_mask,
            )
            np.power(
                forward_power_bases,
                exp_reverse,
                out=reverse_buffer,
                where=reverse_mask,
            )
            np.prod(reverse_buffer, axis=1, out=log_prod_reverse)
            np.multiply(k_reverse_dyn, log_prod_reverse, out=log_rate_reverse)
            np.subtract(net_rates, log_rate_reverse, out=net_rates)
            return _stoichiometric_flux_product(
                S,
                net_rates,
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
        if n_reactions:
            forward_buffer[:n_reactions].fill(1.0)
            reaction_fractional_mask = fractional_forward_mask[:n_reactions]
            if np.any(
                reaction_fractional_mask & (y_arr[np.newaxis, :] < 0.0)
            ):
                # Keep irreversible integer-power trial evaluations signed even
                # when equilibria require the log-domain calculation below.
                forward_power_bases[:n_reactions] = y_arr
                np.maximum(
                    forward_power_bases[:n_reactions],
                    0.0,
                    out=forward_power_bases[:n_reactions],
                    where=reaction_fractional_mask,
                )
                np.power(
                    forward_power_bases[:n_reactions],
                    exp_forward[:n_reactions],
                    out=forward_buffer[:n_reactions],
                    where=forward_mask[:n_reactions],
                )
            else:
                np.power(
                    y_arr,
                    exp_forward[:n_reactions],
                    out=forward_buffer[:n_reactions],
                    where=forward_mask[:n_reactions],
                )
            net_rates[:n_reactions] = (
                k_forward_dyn[:n_reactions]
                * forward_buffer[:n_reactions].prod(axis=1)
            )
        return _stoichiometric_flux_product(
            S,
            net_rates,
            workspace=stoich_rate_buffer,
            out=stoich_flux,
        )

    return ode_rhs

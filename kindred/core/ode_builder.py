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

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .equilibrium_rate_authority import (
    json_safe_equilibrium_authority_payload,
)
from .mechanism import Equilibrium, FrozenDict, Mechanism, Reaction

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
    workspace: np.ndarray,
) -> np.ndarray:
    """Compute fresh species fluxes without entering the BLAS matrix-vector path."""
    np.multiply(stoichiometry, rates, out=workspace)
    return np.sum(workspace, axis=1)


@dataclass(frozen=True, slots=True)
class EffectiveRateCoefficients:
    """Immutable effective forward/reverse coefficients for one temperature."""

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

    def __call__(self, temperature: float) -> EffectiveRateCoefficients:
        # The temperature arrives admitted by the DSL T directive or its schedule.
        # Parameter evaluation admits every coefficient as a rate: coordinate
        # writes and Algebra targets as their role, derived readouts as computed.
        roots = tuple(name for pair in self._parameter_names for name in pair if name is not None)
        values = self.parameter_values(temperature, roots)
        return EffectiveRateCoefficients(
            forward=tuple(float(values[name]) for name, _reverse in self._parameter_names),
            reverse=tuple(float(values[name]) if name else 0.0 for _forward, name in self._parameter_names),
        )

def build_effective_rate_coefficient_evaluator(
    mechanism: Mechanism,
) -> EffectiveRateCoefficientEvaluator:
    """Build the sole effective-rate authority consumed by RHS and Jacobian."""

    from .mechanism_metadata import MechanismMetadataView

    mech_meta = MechanismMetadataView.from_metadata(mechanism.metadata)
    nominal_temperature = float(mech_meta.temperature_K)
    from .simulator.parameter_algebra_eval import bind_parameter_evaluation
    from .simulator.step_indexing import get_step_index_map

    # Admission always retains the spec; its absence is a producer error.
    spec = mechanism.metadata["parameter_algebra_spec"]
    evaluate = bind_parameter_evaluation(spec, mechanism)
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
            model = step.forward_model
            snapshot = {"kind": kind, "forward_model": forward_rate_model_payload(model),
                        "binding_name" if kind == "reaction" else "forward_binding_name": forward_name,
                        "value_dependencies": tuple(spec.evaluation_dependencies[forward_name]),
                        "temperature_dependent": forward_name in spec.temperature_dependent_names}
            if kind == "equilibrium":
                snapshot.update(authority=json_safe_equilibrium_authority_payload(dict(step.authority.identity_items())), reverse_binding_name=reverse_name)
            names.append((forward_name, reverse_name))
            step_kinds.append(kind)
            snapshots.append(FrozenDict(snapshot))
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

    step_kinds: tuple[str, ...]
    stoichiometry: np.ndarray
    # One row per forward term, then, when the mechanism has equilibria, one
    # per reverse term in the same step order.
    exponents: np.ndarray
    mask: np.ndarray
    fractional_mask: np.ndarray

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
                stoichiometry[:, index] = equilibrium.net_stoich_vector(species_names)
                for name, coefficient in equilibrium.stoich_forward.items():
                    forward_exponents[index, species_index[name]] = float(coefficient)
                for name, coefficient in equilibrium.stoich_back.items():
                    reverse_exponents[index, species_index[name]] = float(coefficient)
        exponents = (
            np.vstack((forward_exponents, reverse_exponents))
            if mechanism.equilibria
            else forward_exponents
        )
        mask = exponents != 0.0
        fractional_mask = mask & (exponents != np.rint(exponents))
        for array in (stoichiometry, exponents, mask, fractional_mask):
            array.setflags(write=False)
        return cls(
            step_kinds=tuple(step_kinds),
            stoichiometry=stoichiometry,
            exponents=exponents,
            mask=mask,
            fractional_mask=fractional_mask,
        )

    def bind_target(
        self,
        mechanism: Mechanism,
    ) -> tuple[EffectiveRateCoefficientEvaluator, Callable[..., np.ndarray]]:
        """Bind one target's rate authority and mutable RHS workspace.

        The target is a shell of the prepared owner that compiled this topology,
        so its species and step order are this topology's.
        """

        coefficient_evaluator = build_effective_rate_coefficient_evaluator(mechanism)
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
    - Rates come from the admitted typed forward model and reverse authority
      at the temperature T supplied by every caller.
    - Reactions contribute: rate = k * ∏[reactants]^stoich
    - Equilibria remain a single reversible step with rate = v_fwd - v_rev
      (stoichiometry is not duplicated into two irreversible columns).
    - Stoichiometry determines species changes via a BLAS-free reduction.
    """
    S = topology.stoichiometry
    exponents = topology.exponents
    mask = topology.mask
    fractional_mask = topology.fractional_mask
    n_steps = len(topology.step_kinds)
    has_reverse_terms = bool(mechanism.equilibria)
    has_fractional_orders = bool(fractional_mask.any())
    # Unmasked entries stay 1.0, so each row's product runs over its orders.
    terms_buffer = np.ones_like(exponents)
    power_bases = np.empty_like(exponents)
    stoich_rate_buffer = np.empty_like(S)
    coefficients = np.zeros(exponents.shape[0], dtype=float)
    # The coefficients are a function of temperature alone, so they are
    # refilled only when it changes; set after a successful evaluation.
    coefficients_T: float | None = None

    def ode_rhs(t: float, y: np.ndarray, *, T: float) -> np.ndarray:
        """ODE right-hand side from stoichiometry and reaction rates."""
        nonlocal coefficients_T
        if T != coefficients_T:
            values = coefficient_evaluator(float(T))
            coefficients[:n_steps] = values.forward
            if has_reverse_terms:
                coefficients[n_steps:] = values.reverse
            coefficients_T = T
        bases = y
        if has_fractional_orders:
            # Implicit solvers may briefly trial a negative internal state.
            # Restrict only non-integral power evaluation to its physical
            # domain; this leaves both the solver state and integer powers
            # unchanged and does not alter published concentration truth.
            power_bases[:] = y
            np.maximum(power_bases, 0.0, out=power_bases, where=fractional_mask)
            bases = power_bases
        np.power(bases, exponents, out=terms_buffer, where=mask)
        terms = coefficients * terms_buffer.prod(axis=1)
        rates = terms[:n_steps] - terms[n_steps:] if has_reverse_terms else terms
        return _stoichiometric_flux_product(S, rates, workspace=stoich_rate_buffer)

    return ode_rhs

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
from kindred.core.mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .mechanism import Equilibrium, FrozenDict, Mechanism, Reaction
from kindred.core.simulator.step_indexing import get_step_index_map
from kindred.core.simulator.parameter_algebra_eval import bind_parameter_evaluation
from kindred.core.algebra.errors import AlgebraError
from kindred.core.algebra.evaluator import PreparedAlgebraPointEvaluator
from kindred.core.algebra.simulation_series import failing_let_facts, observable_species_refs
from kindred.core.exceptions import ValueDomainError

__all__ = [
    "CompiledOdeTopology",
    "EffectiveRateCoefficients",
    "EffectiveRateCoefficientEvaluator",
    "rate_authority_snapshot",
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
    """One typed per-run runtime coefficient authority."""

    nominal_temperature_K: float
    step_kinds: tuple[str, ...]
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

def _step_coefficient_names(
    mechanism: Mechanism,
) -> tuple[tuple[str, Reaction | Equilibrium, Mapping[str, object], str, str | None], ...]:
    """Each step's kind, the step, its step-map entry, and its forward and
    reverse coefficient names, reactions first, as the topology orders them."""

    entries = {(entry["kind"], entry["reaction_index" if entry["kind"] == "reaction" else "equilibrium_index"]): entry
               for entry in get_step_index_map(mechanism)}
    steps = []
    for kind, members in (("reaction", mechanism.reactions), ("equilibrium", mechanism.equilibria)):
        for index, step in enumerate(members):
            entry = entries[kind, index]
            number = entry["step_index"]
            if kind == "reaction":
                steps.append((kind, step, entry, f"k{number}", None))
            else:
                steps.append((kind, step, entry, f"kf{number}", f"kr{number}"))
    return tuple(steps)


def rate_authority_snapshot(mechanism: Mechanism) -> tuple[Mapping[str, object], ...]:
    """Each step's rate model and the values its coefficient reads, in
    topology order, without calculating any value."""

    # Admission always retains the spec; its absence is a producer error.
    spec = mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC]
    snapshots = []
    for kind, step, _entry, forward_name, reverse_name in _step_coefficient_names(mechanism):
        snapshot = {"kind": kind, "forward_model": forward_rate_model_payload(step.forward_model),
                    "binding_name" if kind == "reaction" else "forward_binding_name": forward_name,
                    "value_dependencies": tuple(spec.evaluation_dependencies[forward_name]),
                    "temperature_dependent": forward_name in spec.temperature_dependent_names}
        if kind == "equilibrium":
            snapshot.update(authority=dict(step.authority.identity_items()), reverse_binding_name=reverse_name)
        snapshots.append(FrozenDict(snapshot))
    return tuple(snapshots)


def _build_effective_rate_coefficient_evaluator(
    mechanism: Mechanism,
    initials: Mapping[str, float],
) -> EffectiveRateCoefficientEvaluator:
    """Build the sole effective-rate authority consumed by RHS and Jacobian,
    for one run's starting concentrations."""

    mech_meta = MechanismMetadataView.from_metadata(mechanism.metadata)
    # Admission always retains the spec; its absence is a producer error.
    spec = mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC]
    steps = _step_coefficient_names(mechanism)
    return EffectiveRateCoefficientEvaluator(
        nominal_temperature_K=float(mech_meta.temperature_K),
        step_kinds=tuple(kind for kind, _step, _entry, _forward, _reverse in steps),
        parameter_values=bind_parameter_evaluation(spec, mechanism, initials),
        _parameter_names=tuple((forward, reverse) for _kind, _step, _entry, forward, reverse in steps),
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
        initials: Mapping[str, float],
    ) -> tuple[EffectiveRateCoefficientEvaluator, Callable[..., np.ndarray]]:
        """Bind one target's rate authority and mutable RHS workspace.

        The target is a shell of the prepared owner that compiled this topology,
        so its species and step order are this topology's. ``initials`` are
        the run's starting concentrations, which params may read.
        """

        coefficient_evaluator = _build_effective_rate_coefficient_evaluator(mechanism, initials)
        return coefficient_evaluator, _build_bound_ode_rhs(
            self,
            mechanism,
            coefficient_evaluator,
            bind_rate_factors(mechanism, initials, coefficient_evaluator.parameter_values),
        )


def factored_steps(mechanism: Mechanism) -> tuple[tuple[int, str, Mapping[str, object]], ...]:
    """Each step with a factor: its position in topology order, the let it
    names and its step-map entry."""
    return tuple((position, step.factor, entry)
                 for position, (_kind, step, entry, _forward, _reverse) in enumerate(_step_coefficient_names(mechanism))
                 if step.factor is not None)


def bind_rate_factors(
    mechanism: Mechanism,
    initials: Mapping[str, float],
    parameter_values: Callable[[float, tuple[str, ...]], dict[str, float]],
) -> tuple[np.ndarray, Callable[[float, np.ndarray, float], np.ndarray]] | None:
    """One run's rate factors: the topology positions of the factored steps
    and their values at a time, state and temperature, or None without factors.

    Each factor is its let calculated by the Algebra evaluator on the state
    with negative concentrations read as zero, as fractional orders are.
    """

    steps = factored_steps(mechanism)
    if not steps:
        return None

    spec = mechanism.metadata[MechanismMetadataKeys.PARAMETER_ALGEBRA_SPEC]
    block = spec.observable_block
    species_names = tuple(mechanism.species_names())
    reads = {
        name: np.fromiter((index for index, species in enumerate(species_names)
                           if species in observable_species_refs(block, name, "now")), dtype=np.int64)
        for _position, name, _entry in steps
    }
    evaluator = PreparedAlgebraPointEvaluator(
        block, species_names=species_names, initials=initials,
        symbol_resolver=None, runtime_symbol_names=set(),
    )
    positions = np.fromiter((position for position, _name, _entry in steps), dtype=np.int64)
    values = np.empty(len(steps), dtype=float)

    def evaluate(t: float, y: np.ndarray, T: float) -> np.ndarray:
        state = np.maximum(y, 0.0)

        def resolve(name: str) -> float | None:
            return parameter_values(T, (name,))[name] if name in spec.evaluation_dependencies else None

        for index, (_position, name, entry) in enumerate(steps):
            try:
                values[index] = evaluator.evaluate(name, t, state, resolve)
            except (AlgebraError, ValueDomainError) as exc:
                if not np.all(np.isfinite(y[reads[name]])):
                    # A solver trial with concentrations that are not finite,
                    # which the solver refuses as it refuses such rates.
                    values[index] = np.nan
                    continue
                # The let that failed names its inputs, and a param it reads
                # that failed names its own.
                if "assignment" not in exc.details:
                    exc.annotate(**failing_let_facts(spec, exc))
                exc.annotate(
                    prefix=f"The rate factor {name!r} of line {entry['line']} cannot be calculated at t = {float(t):g} s: ",
                    factor=name, time=float(t),
                )
                raise
        return values

    return positions, evaluate


def _build_bound_ode_rhs(
    topology: CompiledOdeTopology,
    mechanism: Mechanism,
    coefficient_evaluator: EffectiveRateCoefficientEvaluator,
    rate_factors: tuple[np.ndarray, Callable[[float, np.ndarray, float], np.ndarray]] | None,
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
    - A step with a factor has its net rate multiplied by it, so forward and
      reverse equally.
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
    factor_positions, factor_values = rate_factors if rate_factors is not None else (None, None)

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
        if factor_positions is not None:
            rates[factor_positions] *= factor_values(t, y, T)
        return _stoichiometric_flux_product(S, rates, workspace=stoich_rate_buffer)

    return ode_rhs

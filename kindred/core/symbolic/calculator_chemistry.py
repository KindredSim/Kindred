"""Calculator-only symbolic chemistry, projected from admitted source coordinates."""
from __future__ import annotations

from fractions import Fraction
from collections.abc import Mapping
from typing import Any

from kindred.core.equilibrium_rate_authority import equilibrium_rate_symbolic_relations
from kindred.core.kinetics import forward_rate_symbolic_relation
from kindred.core.mechanism import Mechanism
from kindred.core.mechanism_metadata import EquilibriumMetadataKeys, MechanismMetadataView
from kindred.core.simulator.kinetics import normalize_energy_to_J_per_mol
from kindred.core.simulator.parameter_algebra_spec import ParameterAlgebraSpec
from kindred.core.simulator.step_indexing import get_step_index_map

from .backend import require_sympy
from .errors import UnsupportedSymbolicExpressionError


def build_calculator_chemistry_relations(
    mechanism: Mechanism, *, parameter_spec: ParameterAlgebraSpec,
    symbols: Mapping[str, Any],
) -> dict[str, Any]:
    """Lower chemistry readouts without evaluating Algebra or independent inputs."""

    sympy = require_sympy()
    metadata = MechanismMetadataView.from_metadata(mechanism.metadata)

    def exact_literal(value: object) -> Any:
        if isinstance(value, Fraction):
            return sympy.Rational(value.numerator, value.denominator)
        return sympy.Rational(str(value))

    energy_factor = exact_literal(
        normalize_energy_to_J_per_mol(1, metadata.energy_unit)
    )
    temperature = symbols["T"]
    readouts_by_source: dict[int, dict[str, str]] = {}
    for item in parameter_spec.mechanism_namespace.ordered_items:
        readouts_by_source.setdefault(item.source_index, {})[
            item.info.role
        ] = item.canonical_name

    derived = parameter_spec.chemistry_dependencies
    chemistry_relations: dict[str, Any] = {}
    for source_index, entry in enumerate(get_step_index_map(mechanism)):
        reaction = entry["kind"] == "reaction"
        step = (
            mechanism.reactions[entry["reaction_index"]]
            if reaction
            else mechanism.equilibria[entry["equilibrium_index"]]
        )
        source_coordinates = entry["source_coordinates"]
        readouts = readouts_by_source[source_index]
        forward_name = readouts["k" if reaction else "kf"]

        def source_coordinate(role: str) -> Any:
            # Constant's forward helper uses kf for both kinds of step.
            source_role = "k" if reaction and role == "kf" else role
            name = source_coordinates.get(source_role)
            if name is None:
                if role in {"kappa", "degeneracy"}:
                    return None
                raise ValueError(f"Missing admitted source coordinate for {source_role!r}.")
            symbol = symbols[name]
            if role in {"Ea", "dG_act", "dG_eq"}:
                return energy_factor * symbol
            return symbol

        try:
            forward = forward_rate_symbolic_relation(
                step.forward_model,
                temperature=temperature,
                literal=exact_literal,
                exponential=sympy.exp,
                source_coordinate=source_coordinate,
                constant=symbols.__getitem__,
            )
            if forward_name in derived:
                chemistry_relations[forward_name] = forward
            if reaction:
                continue
            standard_concentration = step.metadata.get(
                EquilibriumMetadataKeys.STANDARD_CONC_M, metadata.standard_conc_M
            )
            if standard_concentration is None:
                standard_concentration = metadata.standard_conc_M
            relations = equilibrium_rate_symbolic_relations(
                step,
                forward_relation=symbols[forward_name],
                temperature=temperature,
                literal=exact_literal,
                exponential=sympy.exp,
                source_coordinate=source_coordinate,
                standard_concentration=exact_literal(float(standard_concentration)),
                constant=symbols.__getitem__,
            )
            for role, expression in (("kr", relations.kr), ("Keq", relations.Keq)):
                name = readouts[role]
                if name in derived:
                    chemistry_relations[name] = expression
        except (TypeError, ValueError) as exc:
            raise UnsupportedSymbolicExpressionError(
                f"Unsupported symbolic chemistry for step {entry['step_index']}: {exc}"
            ) from exc

    if set(chemistry_relations) != set(derived):
        raise UnsupportedSymbolicExpressionError(
            "Admitted chemistry dependency projection is incomplete."
        )
    return chemistry_relations

"""
Mechanism construction from one admitted ``DslMechanismBuildInput``.

``dsl_parse`` owns source parsing and analysis; this module only consumes the
immutable builder input and constructs the typed mechanism authority.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..mechanism import Mechanism
from ..mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView, EquilibriumMetadataView
from .dsl_format import format_stoichiometry_side as _fmt_side
from .dsl_build_input import DslMechanismBuildInput

logger = logging.getLogger(__name__)

__all__ = ["build_mechanism_from_ir"]


def build_mechanism_from_ir(
    build_input: DslMechanismBuildInput,
) -> Mechanism:
    """Build a mechanism from the admitted immutable builder input."""

    energy_unit = build_input.energy_unit
    T = build_input.temperature_K
    C0 = build_input.standard_conc_M
    steps = build_input.steps
    temperature_schedule = build_input.temperature_schedule

    mechanism = Mechanism()
    MechanismMetadataView(
        temperature_K=float(T),
        standard_conc_M=float(C0),
        energy_unit=str(energy_unit),
    ).apply_to_metadata(mechanism.metadata)

    # Canonical step-index mapping (global DSL step order).
    # Policy: includes every reaction/equilibrium line from the DSL text.
    step_index_map: List[Dict[str, object]] = []

    # Collect all species
    all_species = set()
    for step in steps:
        all_species.update(dict(step.reactants).keys())
        all_species.update(dict(step.products).keys())

    # Add species to mechanism
    for sp_name in sorted(all_species):
        mechanism.add_species(sp_name)

    authored_step_no = 0
    for _source_step_no, step in enumerate(steps, start=1):
        reactants = dict(step.reactants)
        products = dict(step.products)
        is_equilibrium_step = step.reversible

        if is_equilibrium_step:
            eq_metadata = EquilibriumMetadataView(
                dG_eq_J_per_mol=step.dG_eq_J_per_mol,
                Keq_input=step.Keq_input,
                standard_conc_M=step.standard_conc_M,
            ).to_metadata()

            eq_index = len(mechanism.equilibria)
            mechanism.add_equilibrium(
                stoich_forward=reactants,
                stoich_back=products,
                forward_model=step.forward_model,
                kr=step.kr,
                metadata=eq_metadata,
            )
        else:
            rxn_index = len(mechanism.reactions)
            mechanism.add_reaction(
                reactants=reactants,
                products=products,
                forward_model=step.forward_model,
            )

        # Record canonical authored step-index mapping for downstream layers (GUI/algebra/fitting).
        authored_step_no += 1
        arrow = "<->" if is_equilibrium_step else "->"
        context = f"{_fmt_side(reactants)} {arrow} {_fmt_side(products)}"
        entry: Dict[str, object] = {
            "step_index": int(authored_step_no),
            "kind": "equilibrium" if is_equilibrium_step else "reaction",
            "context": context,
            "line": step.line,
            "line_text": step.line_text,
        }
        # Project only the parser-admitted physical source roles into the
        # established metadata map.  Canonical indexed rate coordinates stay
        # in the same namespace; their authored spans remain on DslAnalysis.
        source_coordinates: dict[str, str] = {
            str(coordinate.role): str(coordinate.canonical_name)
            for coordinate in step.source_coordinates
        }
        if is_equilibrium_step:
            entry["equilibrium_index"] = int(eq_index)
        else:
            entry["reaction_index"] = int(rxn_index)
        entry["source_coordinates"] = source_coordinates
        step_index_map.append(entry)

    mechanism.metadata["step_index_map"] = step_index_map

    logger.debug(
        "Built mechanism from DSL reactions: %s species, %s reactions, %s equilibria",
        len(mechanism.species),
        len(mechanism.reactions),
        len(mechanism.equilibria),
    )

    if temperature_schedule is not None:
        mechanism.metadata[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = temperature_schedule
    return mechanism

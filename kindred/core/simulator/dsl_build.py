"""
Mechanism construction from one admitted ``DslMechanismBuildInput``.

``dsl_parse`` owns source parsing and analysis; this module only consumes the
exact immutable builder input and constructs the typed mechanism authority.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from ..equilibrium_rate_authority import (
    normalize_equilibrium_rate_authority,
)
from ..mechanism import Mechanism
from ..mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView, EquilibriumMetadataView
from ..thermodynamics import K_from_deltaG_eq
from .dsl_format import format_stoichiometry_side as _fmt_side
from .dsl_build_input import DslMechanismBuildInput
from .parameter_namespace import _namespace_policy_from_step

logger = logging.getLogger(__name__)

__all__ = ["build_mechanism_from_ir"]


def build_mechanism_from_ir(
    build_input: DslMechanismBuildInput,
) -> Mechanism:
    """Build a mechanism from the exact admitted immutable builder input."""

    if type(build_input) is not DslMechanismBuildInput:
        raise TypeError(
            "build_mechanism_from_ir requires exact DslMechanismBuildInput."
        )

    initials_merged = dict(build_input.initials)
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
        init_conc = float(initials_merged.get(sp_name, 0.0))
        mechanism.add_species(sp_name, init_conc)

    authored_step_no = 0
    for _source_step_no, step in enumerate(steps, start=1):
        reactants = dict(step.reactants)
        products = dict(step.products)
        model = step.model

        rxn_overrides: Dict[str, object] = {}
        if model == "Constant":
            rxn_overrides["model"] = "Constant"
        elif model == "Arrhenius":
            rxn_overrides["model"] = "Arrhenius"
            A = step.arrhenius_A
            EaJ = step.arrhenius_Ea_J_per_mol
            if A is not None:
                rxn_overrides["A"] = float(A)
            if EaJ is not None:
                rxn_overrides["Ea_J_per_mol"] = float(EaJ)
        elif model == "Eyring":
            rxn_overrides["model"] = "Eyring"
            rxn_overrides["degeneracy"] = int(step.degeneracy)
            dGJ = step.eyring_dG_act_J_per_mol
            if dGJ is not None:
                rxn_overrides["dG_act_J_per_mol"] = float(dGJ)
            kappa = step.kappa
            rxn_overrides["kappa"] = float(kappa)
            sc = step.standard_conc_M
            if sc is not None:
                rxn_overrides["standard_conc_M"] = float(sc)

        is_equilibrium = step.is_equilibrium
        namespace_policy = _namespace_policy_from_step(step)
        kr_attr = step.kr
        is_equilibrium_step = namespace_policy.step_kind == "equilibrium"

        if is_equilibrium_step:
            kf_val = step.kf
            kr_val = float(kr_attr) if kr_attr is not None else None
            Keq_input = step.Keq_input
            user_kf_explicit = step.user_kf_explicit
            user_kr_explicit = step.user_kr_explicit

            dG_eq_J = step.dG_eq_J_per_mol
            if Keq_input is not None:
                Keq_model_value = Keq_input
            elif dG_eq_J is not None:
                Keq_model_value = float(
                    K_from_deltaG_eq(float(dG_eq_J), T)
                )
            else:
                Keq_model_value = None

            forward_model: Optional[Dict[str, object]] = None
            if (
                model == "Arrhenius"
                and step.arrhenius_A is not None
                and step.arrhenius_Ea_J_per_mol is not None
            ):
                forward_model = {
                    "type": "Arrhenius",
                    "A": step.arrhenius_A,
                    "Ea_J_per_mol": step.arrhenius_Ea_J_per_mol,
                }
            elif (
                model == "Eyring"
                and step.eyring_dG_act_J_per_mol is not None
            ):
                forward_model = {
                    "type": "Eyring",
                    "dG_act_J_per_mol": step.eyring_dG_act_J_per_mol,
                    "degeneracy": int(step.degeneracy),
                    "kappa": float(step.kappa),
                    "standard_conc_M": (
                        step.standard_conc_M
                        if step.standard_conc_M is not None
                        else C0
                    ),
                }
            elif model == "Constant":
                forward_model = {
                    "type": "Constant",
                    "k": float(kf_val),
                }

            eq_metadata = EquilibriumMetadataView(
                fast_equilibrium=bool(is_equilibrium),
                user_provided_kf=bool(user_kf_explicit),
                user_provided_kr=bool(user_kr_explicit),
                dG_eq_J_per_mol=step.dG_eq_J_per_mol
                if step.dG_eq_J_per_mol is not None
                else None,
                Keq_input=Keq_input,
                forward_model=forward_model,
                standard_conc_M=step.standard_conc_M
                if step.standard_conc_M is not None
                else None,
            ).to_metadata()
            if Keq_input is not None or dG_eq_J is not None:
                nf = math.fsum(float(value) for value in reactants.values())
                nr = math.fsum(float(value) for value in products.values())
                eq_metadata["std_ratio"] = float(C0) ** (nr - nf)
            authority_context = "normalized_public"
            authority = normalize_equilibrium_rate_authority(
                kf=kf_val,
                kr=kr_val,
                Keq=Keq_model_value,
                metadata=eq_metadata,
                context=authority_context,
            )

            eq_index = len(mechanism.equilibria)
            mechanism._add_equilibrium_with_authority_context(
                stoich_forward=reactants,
                stoich_back=products,
                Keq=Keq_model_value,
                kf=kf_val,
                kr=kr_val,
                fast=is_equilibrium,  # Mark "equilibrium:" lines as fast
                metadata=eq_metadata,
                authority_context=authority_context,
            )
        else:
            rxn_index = len(mechanism.reactions)
            mechanism.add_reaction(
                reactants=reactants,
                products=products,
                rate=step.kf,
                overrides=rxn_overrides or None,
            )

        # Record canonical authored step-index mapping for downstream layers (GUI/algebra/fitting).
        authored_step_no += 1
        arrow = "<->" if is_equilibrium_step else "->"
        context = f"{_fmt_side(reactants)} {arrow} {_fmt_side(products)}"
        entry: Dict[str, object] = {
            "step_index": int(authored_step_no),
            "kind": namespace_policy.step_kind,
            "context": context,
        }
        if is_equilibrium_step:
            entry["equilibrium_index"] = int(eq_index)
            entry.update(authority.step_map_fields())
            entry["user_provided_kf"] = step.user_kf_explicit
            entry["user_provided_kr"] = step.user_kr_explicit
        else:
            entry["reaction_index"] = int(rxn_index)
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

"""
Mechanism construction from the simulator DSL intermediate representation (IR).

This module is intentionally separate from `dsl.py` so that importing the DSL parsing
utilities does not eagerly import Mechanism/kinetics/state-network construction code.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Dict, List, Optional

from ..equilibrium_rate_authority import (
    EquilibriumRateInputContext,
    normalize_equilibrium_rate_authority,
)
from ..mechanism import Mechanism
from ..mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView, EquilibriumMetadataView
from .dsl_format import format_stoichiometry_side as _fmt_side
from .errors import DSLError
from .common import K_from_deltaG_eq
from .parameter_namespace import _namespace_policy_from_step

logger = logging.getLogger(__name__)

__all__ = ["build_mechanism_from_ir"]


def build_mechanism_from_ir(
    ir: object,
    *,
    initials: Optional[Dict[str, float]] = None,
) -> Mechanism:
    """
    Build a `Mechanism` from a parsed DSL IR.

    Parameters
    ----------
    ir
        DSLIR-like object produced by `kindred.core.simulator.dsl._parse_dsl_ir`.
    initials
        Optional initial concentrations mapping. Treated as input-only.

    Returns
    -------
    Mechanism
    """
    initials_in = dict(initials or {})
    initials_merged = dict(initials_in)
    initials_merged.update(dict(getattr(ir, "initials_from_dsl", {}) or {}))

    energy_unit = str(getattr(ir, "energy_unit"))
    T = float(getattr(ir, "temperature_K"))
    C0 = float(getattr(ir, "standard_conc_M"))
    kappa_global = float(getattr(ir, "kappa_global"))
    net = getattr(ir, "state_network")
    steps = list(getattr(ir, "steps", []) or [])
    algebra_lines = list(getattr(ir, "algebra_lines", []) or [])
    temperature_schedule = getattr(ir, "temperature_schedule", None)
    intervention_schedule = getattr(ir, "intervention_schedule", None)

    mechanism = Mechanism()
    MechanismMetadataView(
        temperature_K=float(T),
        standard_conc_M=float(C0),
        kappa_global=float(kappa_global),
        energy_unit=str(energy_unit),
        step_index_policy="dsl_only",
    ).apply_to_metadata(mechanism.metadata)

    if getattr(net, "states")() or getattr(net, "edges")():
        mechanism.metadata["state_network"] = net.to_serializable()

    # Canonical step-index mapping (global DSL step order).
    # Policy: includes only explicit reaction/equilibrium lines from the DSL text.
    # State-network generated steps (added later) do NOT consume indices and are excluded.
    step_index_map: List[Dict[str, object]] = []

    # Collect all species
    all_species = set()
    for step in steps:
        all_species.update(dict(getattr(step, "reactants", {}) or {}).keys())
        all_species.update(dict(getattr(step, "products", {}) or {}).keys())

    # Add species to mechanism
    for sp_name in sorted(all_species):
        init_conc = float(initials_merged.get(sp_name, 0.0))
        mechanism.add_species(sp_name, init_conc)

    # Add reactions/equilibria to mechanism (preserve DSL order for step indexing)
    for step_no, step in enumerate(steps, start=1):
        reactants = dict(getattr(step, "reactants", {}) or {})
        products = dict(getattr(step, "products", {}) or {})

        model = str(getattr(step, "model", "Eyring") or "Eyring")

        rxn_overrides: Dict[str, object] = {}
        if model == "Arrhenius":
            rxn_overrides["model"] = "Arrhenius"
            A = getattr(step, "arrhenius_A", None)
            EaJ = getattr(step, "arrhenius_Ea_J_per_mol", None)
            if A is not None:
                rxn_overrides["A"] = float(A)
            if EaJ is not None:
                rxn_overrides["Ea_J_per_mol"] = float(EaJ)
        elif model == "Eyring":
            rxn_overrides["model"] = "Eyring"
            dGJ = getattr(step, "eyring_dG_act_J_per_mol", None)
            if dGJ is not None:
                rxn_overrides["dG_act_J_per_mol"] = float(dGJ)
            kappa = getattr(step, "kappa", None)
            if kappa is not None:
                rxn_overrides["kappa"] = float(kappa)
            sc = getattr(step, "standard_conc_M", None)
            if sc is not None:
                rxn_overrides["standard_conc_M"] = float(sc)

        is_equilibrium = bool(getattr(step, "is_equilibrium", False))
        namespace_policy = _namespace_policy_from_step(step)
        kr_attr = getattr(step, "kr", None)
        is_equilibrium_step = namespace_policy.step_kind == "equilibrium"

        if is_equilibrium_step:
            kf_val = float(getattr(step, "kf"))
            kr_val = float(kr_attr) if kr_attr is not None else None
            Keq_input = getattr(step, "Keq_input", None)
            user_kf_explicit = bool(getattr(step, "user_kf_explicit", False))
            user_kr_explicit = bool(getattr(step, "user_kr_explicit", False))

            if Keq_input is not None and kr_val is not None and user_kf_explicit and not user_kr_explicit:
                try:
                    Keq_in = float(Keq_input)
                except Exception:
                    Keq_in = float("nan")
                if math.isfinite(Keq_in) and abs(Keq_in) > 1e-30:
                    kr_val = kf_val / Keq_in

            dG_eq_J = getattr(step, "dG_eq_J_per_mol", None)
            if Keq_input is not None:
                Keq_model_value = Keq_input
            elif dG_eq_J is not None:
                Keq_model_value = float(K_from_deltaG_eq(float(dG_eq_J), float(getattr(ir, "temperature_K"))))
            else:
                Keq_model_value = None

            forward_model: Optional[Dict[str, object]] = None
            if model == "Arrhenius" and getattr(step, "arrhenius_A", None) is not None and getattr(
                step, "arrhenius_Ea_J_per_mol", None
            ) is not None:
                forward_model = {
                    "type": "Arrhenius",
                    "A": float(getattr(step, "arrhenius_A")),
                    "Ea_J_per_mol": float(getattr(step, "arrhenius_Ea_J_per_mol")),
                }
            elif model == "Eyring" and getattr(step, "eyring_dG_act_J_per_mol", None) is not None:
                forward_model = {
                    "type": "Eyring",
                    "dG_act_J_per_mol": float(getattr(step, "eyring_dG_act_J_per_mol")),
                    "kappa": float(getattr(step, "kappa", kappa_global)),
                    "standard_conc_M": float(getattr(step, "standard_conc_M", C0) or C0),
                }

            eq_metadata = EquilibriumMetadataView(
                fast_equilibrium=bool(is_equilibrium),
                user_provided_kf=bool(user_kf_explicit),
                user_provided_kr=bool(user_kr_explicit),
                dG_eq_J_per_mol=float(getattr(step, "dG_eq_J_per_mol"))
                if getattr(step, "dG_eq_J_per_mol", None) is not None
                else None,
                Keq_input=Keq_input,
                explicit_rates=tuple(float(x) for x in (getattr(step, "explicit_rates", []) or [])),
                forward_model=forward_model,
                standard_conc_M=float(getattr(step, "standard_conc_M"))
                if getattr(step, "standard_conc_M", None) is not None
                else None,
            ).to_metadata()
            if getattr(step, "cm_id", None) is not None:
                eq_metadata["cm_id"] = str(getattr(step, "cm_id"))
                if bool(getattr(step, "generated_computational_mode", False)):
                    eq_metadata["authority_source"] = EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE.value
            if getattr(step, "cm_std_ratio", None) is not None:
                eq_metadata["std_ratio"] = float(getattr(step, "cm_std_ratio"))
            authority_context = (
                EquilibriumRateInputContext.GENERATED_COMPUTATIONAL_MODE
                if bool(getattr(step, "generated_computational_mode", False))
                else EquilibriumRateInputContext.NORMALIZED_PUBLIC
            )
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
                rate=float(getattr(step, "kf")),
                overrides=rxn_overrides or None,
            )

        # Record canonical step-index mapping for downstream layers (GUI/algebra/fitting).
        arrow = "<->" if is_equilibrium_step else "->"
        context = f"{_fmt_side(reactants)} {arrow} {_fmt_side(products)}"
        entry: Dict[str, object] = {
            "step_index": int(step_no),
            "kind": namespace_policy.step_kind,
            "context": context,
        }
        if is_equilibrium_step:
            entry["equilibrium_index"] = int(eq_index)
            entry.update(authority.step_map_fields())
            entry["user_provided_kf"] = bool(getattr(step, "user_kf_explicit", False))
            entry["user_provided_kr"] = bool(getattr(step, "user_kr_explicit", False))
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

    # Convert state network to additional reactions if present
    if getattr(net, "states")() or getattr(net, "edges")():
        logger.debug("Converting state network to reactions...")
        from .state_network_converter import convert_state_network_to_mechanism

        rxn_start = len(mechanism.reactions)
        eq_start = len(mechanism.equilibria)

        state_mechanism = convert_state_network_to_mechanism(
            net,
            initials=initials_merged,
            temperature_K=T,
            kappa=kappa_global,
            C0_M=C0,
        )

        for sp_name in state_mechanism.species_names():
            if sp_name not in mechanism.species:
                init_conc = float(initials_merged.get(sp_name, 0.0))
                mechanism.add_species(sp_name, init_conc)

        for rxn in state_mechanism.reactions:
            mechanism.add_reaction(
                reactants=rxn.reactants,
                products=rxn.products,
                rate=rxn.rate,
                rate_orders=rxn.rate_orders,
                overrides=rxn.overrides,
                record_step_index=False,
            )

        for eq in state_mechanism.equilibria:
            mechanism._add_equilibrium_with_authority_context(
                stoich_forward=eq.stoich_forward,
                stoich_back=eq.stoich_back,
                Keq=eq.Keq,
                kf=eq.kf,
                kr=eq.kr,
                fast=eq.fast,
                metadata=getattr(eq, "metadata", None) or None,
                record_step_index=False,
                authority_context=EquilibriumRateInputContext.GENERATED_STATE_NETWORK,
            )

        # Safety guard: state-network generated steps do not participate in canonical step indexing
        from kindred.core.simulator.parameter_namespace import is_protected_indexed_identifier

        def _fail(reason: str) -> None:
            raise DSLError(
                "State-network generated steps currently do not participate in step indexing; "
                "they cannot introduce adjustable parameters. Convert them to explicit DSL reactions/equilibria "
                "if you need fit/slider parameters.\n\n" + reason,
                line_content="# State Network",
            )

        def _check_value(val: object, *, where: str) -> None:
            if val is None:
                return
            nm = getattr(val, "name", None)
            if nm is not None and is_protected_indexed_identifier(str(nm)):
                _fail(f"{where} introduced a canonical-looking parameter name {nm!r}.")
            if callable(val):
                _fail(f"{where} introduced a non-numeric binding/callable ({type(val).__name__}).")
            if not isinstance(val, numbers.Real):
                _fail(f"{where} introduced a non-numeric parameter value ({type(val).__name__}).")
            try:
                if not math.isfinite(float(val)):
                    _fail(f"{where} introduced a non-finite numeric value ({val!r}).")
            except Exception:
                _fail(f"{where} introduced a non-numeric parameter value ({type(val).__name__}).")

        for i, rxn in enumerate((getattr(mechanism, "reactions", []) or [])[rxn_start:], start=rxn_start + 1):
            _check_value(getattr(rxn, "rate", None), where=f"State-network reaction[{i}].rate")

        for i, eq in enumerate((getattr(mechanism, "equilibria", []) or [])[eq_start:], start=eq_start + 1):
            _check_value(getattr(eq, "kf", None), where=f"State-network equilibrium[{i}].kf")
            _check_value(getattr(eq, "kr", None), where=f"State-network equilibrium[{i}].kr")
            _check_value(getattr(eq, "Keq", None), where=f"State-network equilibrium[{i}].Keq")

        logger.debug(
            "After state network integration: %s species, %s reactions, %s equilibria",
            len(mechanism.species),
            len(mechanism.reactions),
            len(mechanism.equilibria),
        )

    if algebra_lines:
        algebra_text = "\n".join(algebra_lines)
        from kindred.core.algebra.simulation_series import compile_algebra_observables
        from kindred.core.simulator.parameter_algebra import (
            mechanism_parameter_namespace,
            parse_parameter_algebra_spec_from_dsl_text,
        )

        mechanism_namespace = mechanism_parameter_namespace(mechanism)
        parse_parameter_algebra_spec_from_dsl_text(
            algebra_text,
            mechanism_namespace=mechanism_namespace,
        )
        try:
            compile_algebra_observables(algebra_text, mechanism_namespace=mechanism_namespace)
        except ValueError as exc:
            raise DSLError(str(exc)) from exc
        mechanism.metadata[MechanismMetadataKeys.ALGEBRA_TEXT] = algebra_text

    if temperature_schedule is not None:
        mechanism.metadata[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = temperature_schedule
    if intervention_schedule is not None:
        mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE] = intervention_schedule

    return mechanism

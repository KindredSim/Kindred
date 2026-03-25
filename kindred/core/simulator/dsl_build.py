"""
Mechanism construction from the simulator DSL intermediate representation (IR).

This module is intentionally separate from `dsl.py` so that importing the DSL parsing
utilities does not eagerly import Mechanism/kinetics/state-network construction code.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
from typing import Dict, List, Optional

from ..mechanism import Mechanism
from ..mechanism_metadata import MechanismMetadataKeys, MechanismMetadataView, EquilibriumMetadataView
from .dsl_format import format_stoichiometry_side as _fmt_side
from .errors import DSLError

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

        # Build stoichiometry (products - reactants)
        stoich: Dict[str, float] = {}
        for sp, coef in reactants.items():
            stoich[sp] = -float(coef)
        for sp, coef in products.items():
            stoich[sp] = stoich.get(sp, 0.0) + float(coef)

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

        reversible = bool(getattr(step, "reversible", False))
        is_equilibrium = bool(getattr(step, "is_equilibrium", False))
        kr_attr = getattr(step, "kr", None)
        is_equilibrium_step = bool(is_equilibrium or (reversible and kr_attr is not None))

        if is_equilibrium_step:
            kf_val = float(getattr(step, "kf"))
            kr_val = float(kr_attr) if kr_attr is not None else None
            K_input = getattr(step, "K_input", None)
            user_kf_explicit = bool(getattr(step, "user_kf_explicit", False))
            user_kr_explicit = bool(getattr(step, "user_kr_explicit", False))

            if K_input is not None and kr_val is not None and (user_kr_explicit ^ user_kf_explicit):
                try:
                    K_in = float(K_input)
                except Exception:
                    K_in = float("nan")
                if math.isfinite(K_in) and abs(K_in) > 1e-30:
                    # Deterministic policy so explicit K always has semantics:
                    # - if only kr was explicitly provided, derive kf from kr and K
                    # - otherwise derive kr from kf and K
                    if user_kr_explicit and not user_kf_explicit:
                        kf_val = kr_val * K_in
                    else:
                        kr_val = kf_val / K_in

            K = kf_val / kr_val if kr_val and kr_val != 0 else None

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
                K_input=K_input,
                explicit_rates=tuple(float(x) for x in (getattr(step, "explicit_rates", []) or [])),
                forward_model=forward_model,
                standard_conc_M=float(getattr(step, "standard_conc_M"))
                if getattr(step, "standard_conc_M", None) is not None
                else None,
            ).to_metadata()

            eq_index = len(mechanism.equilibria)
            mechanism.add_equilibrium(
                stoich_forward=reactants,
                stoich_back=products,
                K=K,
                kf=kf_val,
                kr=kr_val,
                fast=is_equilibrium,  # Mark "equilibrium:" lines as fast
                metadata=eq_metadata,
            )
        else:
            rxn_index = len(mechanism.reactions)
            mechanism.add_reaction(stoich, rate=float(getattr(step, "kf")), overrides=rxn_overrides or None)

        # Record canonical step-index mapping for downstream layers (GUI/algebra/fitting).
        arrow = "<->" if is_equilibrium_step else "->"
        context = f"{_fmt_side(reactants)} {arrow} {_fmt_side(products)}"
        has_K_param = bool(getattr(step, "K_input", None) is not None)
        derive_rate = None
        if is_equilibrium_step and has_K_param:
            user_kf_explicit = bool(getattr(step, "user_kf_explicit", False))
            user_kr_explicit = bool(getattr(step, "user_kr_explicit", False))
            if user_kr_explicit and not user_kf_explicit:
                derive_rate = "kf"
            else:
                derive_rate = "kr"

        entry: Dict[str, object] = {
            "step_index": int(step_no),
            "kind": "equilibrium" if is_equilibrium_step else "reaction",
            "context": context,
        }
        if is_equilibrium_step:
            entry["equilibrium_index"] = int(eq_index)
            entry["has_K_param"] = bool(has_K_param)
            entry["derive_rate"] = derive_rate
            entry["user_provided_kf"] = bool(getattr(step, "user_kf_explicit", False))
            entry["user_provided_kr"] = bool(getattr(step, "user_kr_explicit", False))
        else:
            entry["reaction_index"] = int(rxn_index)
        step_index_map.append(entry)

    mechanism.metadata["step_index_map"] = step_index_map

    logger.info(
        "Built mechanism from DSL reactions: %s species, %s reactions, %s equilibria",
        len(mechanism.species),
        len(mechanism.reactions),
        len(mechanism.equilibria),
    )

    # Convert state network to additional reactions if present
    if getattr(net, "states")() or getattr(net, "edges")():
        logger.info("Converting state network to reactions...")
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
            mechanism.add_reaction(rxn.stoich, rate=rxn.rate)

        for eq in state_mechanism.equilibria:
            mechanism.add_equilibrium(
                stoich_forward=eq.stoich_forward,
                stoich_back=eq.stoich_back,
                K=eq.K,
                kf=eq.kf,
                kr=eq.kr,
                fast=eq.fast,
                metadata=getattr(eq, "metadata", None) or None,
            )

        # Safety guard: state-network generated steps do not participate in canonical step indexing
        _CANON = re.compile(r"^(k|kf|kr|K)\d+$")

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
            if nm is not None and _CANON.match(str(nm)):
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
            _check_value(getattr(eq, "K", None), where=f"State-network equilibrium[{i}].K")

        logger.info(
            "After state network integration: %s species, %s reactions, %s equilibria",
            len(mechanism.species),
            len(mechanism.reactions),
            len(mechanism.equilibria),
        )

    if algebra_lines:
        mechanism.metadata[MechanismMetadataKeys.ALGEBRA_TEXT] = "\n".join(algebra_lines)

    if temperature_schedule is not None:
        mechanism.metadata[MechanismMetadataKeys.TEMPERATURE_SCHEDULE] = temperature_schedule

    return mechanism

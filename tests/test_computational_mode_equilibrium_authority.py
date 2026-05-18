from __future__ import annotations

import math

import numpy as np
import pytest

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.equilibrium_rate_authority import (
    EquilibriumRateAuthorityKind,
    authority_readout_updates_from_step_entry,
    effective_equilibrium_keq,
    normalize_existing_equilibrium_rate_authority,
)
from kindred.core.simulator.computational_mode import CompReaction, CompSpec, CompSpecies, compile_comp_spec
from kindred.core.simulator.computational_mode import GENERATED_BLOCK_END, GENERATED_BLOCK_START
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import read_mechanism_parameter_values
from kindred.core.simulator.state_model import StateNetwork, StateType
from kindred.core.simulator.state_network_converter import StateNetworkConverter


pytestmark = pytest.mark.unit


def test_computational_mode_fast_equilibrium_generates_kf_dg_authority_without_kr():
    spec = CompSpec(
        temperature_K=298.15,
        energy_unit="kJ/mol",
        std_default_M=1.0,
        species={
            "A": CompSpecies(name="A", kind="GS", G_value=0.0, std_M=1.0, cref_M=1.0),
            "B": CompSpecies(name="B", kind="GS", G_value=1.0, std_M=1.0, cref_M=1.0),
        },
        reactions=[
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=10.0),
        ],
    )

    compiled = compile_comp_spec(spec, output_energy_unit="kJ/mol")

    assert "equilibrium: A <-> B" in compiled.generated_reaction_dsl
    assert "; kf=10" in compiled.generated_reaction_dsl
    assert "; dG_eq=" in compiled.generated_reaction_dsl
    assert "; cm_std_ratio=" in compiled.generated_reaction_dsl
    assert "; kr=" not in compiled.generated_reaction_dsl


def test_computational_mode_fast_equilibrium_parses_to_effective_reverse_rate():
    spec = CompSpec(
        temperature_K=298.15,
        energy_unit="kJ/mol",
        std_default_M=1.0,
        species={
            "A": CompSpecies(name="A", kind="GS", G_value=0.0, std_M=1.0, cref_M=1.0),
            "B": CompSpecies(name="B", kind="GS", G_value=1.0, std_M=1.0, cref_M=1.0),
        },
        reactions=[
            CompReaction(reactants={"A": 1.0}, products={"B": 1.0}, fast_k=10.0),
        ],
    )
    compiled = compile_comp_spec(spec, output_energy_unit="kJ/mol")
    mechanism = parse_dsl_to_mechanism(
        f"{GENERATED_BLOCK_START}\n{compiled.generated_reaction_dsl}\n{GENERATED_BLOCK_END}\ninit: A=1.0, B=0.0",
        initials={},
    )
    rhs = build_ode_rhs_from_mechanism(mechanism)

    eq = mechanism.equilibria[0]
    expected_keq = math.exp(-1000.0 / (8.31446261815324 * 298.15))
    expected_kr = 10.0 / expected_keq

    assert float(eq.kf) == pytest.approx(10.0)
    assert float(eq.kr) == pytest.approx(expected_kr)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [expected_kr, -expected_kr])


def test_state_network_direct_fast_equilibrium_uses_thermodynamic_authority():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_edge("A", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(network, initials={"A": 0.0, "B": 1.0})
    rhs = build_ode_rhs_from_mechanism(mechanism)

    eq = mechanism.equilibria[0]
    authority = normalize_existing_equilibrium_rate_authority(eq)
    expected_keq = math.exp(-1000.0 / (8.31446261815324 * 298.15))
    expected_std_ratio = 0.5
    expected_kr = float(eq.kf) / (expected_keq * expected_std_ratio)

    assert authority.kind == EquilibriumRateAuthorityKind.KEQ
    assert eq.Keq == pytest.approx(expected_keq)
    assert eq.kr == pytest.approx(expected_kr)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [expected_kr, -expected_kr])


def test_state_network_transition_state_readback_uses_explicit_rate_authority_for_keq():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_state("TS", kind=StateType.TS, energy=(50.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_edge("A", "TS")
    network.add_edge("TS", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(
        network,
        initials={"A": 0.0, "B": 1.0},
    )

    eq = mechanism.equilibria[0]
    authority = normalize_existing_equilibrium_rate_authority(eq)
    expected_keq = float(eq.kf) / float(eq.kr)
    algebra_values = read_mechanism_parameter_values(mechanism, names={"Keq1"})

    assert authority.kind == EquilibriumRateAuthorityKind.KR
    assert eq.metadata["Keq"] != pytest.approx(expected_keq)
    assert effective_equilibrium_keq(eq) == pytest.approx(expected_keq)
    assert algebra_values["Keq1"] == pytest.approx(expected_keq)


def test_state_network_transition_state_readout_refresh_derives_keq_from_current_rates():
    network = StateNetwork()
    network.add_state("A", kind=StateType.GS, energy=(0.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_state("B", kind=StateType.GS, energy=(1.0, "kJ/mol"), std_conc_product_M=0.5)
    network.add_state("TS", kind=StateType.TS, energy=(50.0, "kJ/mol"), std_conc_product_M=1.0)
    network.add_edge("A", "TS")
    network.add_edge("TS", "B")
    mechanism = StateNetworkConverter(temperature_K=298.15, C0_M=1.0).convert(
        network,
        initials={"A": 0.0, "B": 1.0},
    )
    entry = mechanism.metadata["step_index_map"][0]

    updates = {
        update.name: update.value
        for update in authority_readout_updates_from_step_entry(
            entry,
            {"kf1": 4.0, "kr1": 1.0, "Keq1": 2.0},
        )
    }

    assert updates["Keq1"] == pytest.approx(4.0)


def test_legacy_computational_mode_payload_validates_kr_as_derived_authority():
    legacy = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=legacy\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )
    mechanism = parse_dsl_to_mechanism(legacy, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)

    eq = mechanism.equilibria[0]
    authority = normalize_existing_equilibrium_rate_authority(eq)

    assert authority.kind == EquilibriumRateAuthorityKind.KEQ
    assert eq.metadata["dG_eq_J_per_mol"] == pytest.approx(1000.0)
    assert eq.metadata["std_ratio"] == pytest.approx(float(eq.kf) / (123.0 * float(eq.Keq)))
    assert float(eq.kr) == pytest.approx(123.0)
    np.testing.assert_allclose(rhs(0.0, np.asarray([0.0, 1.0], dtype=float)), [123.0, -123.0])


def test_legacy_computational_mode_payload_rejects_inconsistent_derived_kr():
    legacy = (
        "energy=kJ/mol\n"
        f"{GENERATED_BLOCK_START}\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=legacy; cm_std_ratio=1.0\n"
        f"{GENERATED_BLOCK_END}\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="legacy Computational Mode kr is inconsistent"):
        parse_dsl_to_mechanism(legacy, initials={})


def test_public_dsl_cannot_self_declare_computational_mode_legacy_authority():
    public_spoof = (
        "energy=kJ/mol\n"
        "equilibrium: A <-> B; kf=10.0; kr=123.0; dG_eq=1.0; cm_id=spoof\n"
        "init: A=0.0, B=1.0"
    )

    with pytest.raises(DSLError, match="generated equilibrium fields"):
        parse_dsl_to_mechanism(public_spoof, initials={})

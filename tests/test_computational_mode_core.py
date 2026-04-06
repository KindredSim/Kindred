import math

import pytest

import scipy.constants

from kindred.core.constants import R, k_B, h
from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _wrap_blocks(*, comp_body: str, generated_body: str) -> str:
    from kindred.core.simulator.computational_mode import (
        COMP_BLOCK_END,
        COMP_BLOCK_START,
        GENERATED_BLOCK_END,
        GENERATED_BLOCK_START,
    )

    comp_body = (comp_body or "").strip("\n")
    generated_body = (generated_body or "").strip("\n")
    return (
        f"{COMP_BLOCK_START}\n"
        f"{comp_body}\n"
        f"{COMP_BLOCK_END}\n\n"
        f"{GENERATED_BLOCK_START}\n"
        f"{generated_body}\n"
        f"{GENERATED_BLOCK_END}\n"
    )


@pytest.mark.unit
def test_hartree_to_jmol_matches_scipy_constants():
    from kindred.core.simulator.computational_mode import hartree_to_jmol

    expected = scipy.constants.value("Hartree energy") * scipy.constants.N_A
    assert float(hartree_to_jmol(1.0)) == pytest.approx(float(expected), rel=0, abs=1e-9)


@pytest.mark.unit
def test_parse_comp_block_rejects_short_form_energy_unit_aliases():
    from kindred.core.simulator.computational_mode import parse_comp_block

    with pytest.raises(ValueError, match="unsupported energy_unit"):
        parse_comp_block(
            "\n".join(
                [
                    "comp: T = 298.15 K",
                    "comp: pressure = 1 atm",
                    "comp: energy_unit = kj",
                    "comp: std_default = 1 M",
                    "comp: species A type=GS G=0.0 degeneracy=1",
                    "comp: species B type=GS G=1.0 degeneracy=1",
                    "comp: rxn A <-> B",
                ]
            )
        )


@pytest.mark.unit
def test_species_standard_state_correction_uses_gas_cref_when_omitted_and_std_changes_relative_G():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    T = 320.0
    P_atm = 1.0
    comp_body = "\n".join(
        [
            "comp: T = 320 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: species A type=GS G=0.0 degeneracy=1",
            "comp: species S type=GS G=0.0 std=19.2 M degeneracy=1",
            "comp: rxn A <-> S",
        ]
    )
    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)

    P_Pa = float(P_atm) * float(scipy.constants.atm)
    c_gas_M = P_Pa / (R * float(T)) / 1000.0
    expected_A = R * float(T) * math.log(1.0 / c_gas_M)
    expected_S = R * float(T) * math.log(19.2 / c_gas_M)

    assert compiled.species_G_std_J_per_mol["A"] == pytest.approx(expected_A, rel=0, abs=5e-6)
    assert compiled.species_G_std_J_per_mol["S"] == pytest.approx(expected_S, rel=0, abs=5e-6)
    assert (compiled.species_G_std_J_per_mol["S"] - compiled.species_G_std_J_per_mol["A"]) == pytest.approx(
        R * float(T) * math.log(19.2),
        rel=0,
        abs=5e-6,
    )


@pytest.mark.unit
def test_ts_channel_uses_per_species_std_factors_and_degeneracy_and_stoichiometry():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    T = 300.0
    hartree_to_jmol = scipy.constants.value("Hartree energy") * scipy.constants.N_A

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            # Choose cref=std so G_std == G_input for each species.
            "comp: species A type=GS G=0.0 std=2.0 M cref=2.0 M degeneracy=1",
            "comp: species B type=GS G=0.0 std=2.0 M cref=2.0 M degeneracy=1",
            "comp: species C type=GS G=-0.01 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species TS1 type=TS G=0.02 std=1.0 M cref=1.0 M degeneracy=2",
            "comp: channel A + B <-> C via TS1",
        ]
    )
    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)
    full_dsl = _wrap_blocks(comp_body=comp_body, generated_body=compiled.generated_reaction_dsl)

    mech = parse_dsl_to_mechanism(full_dsl, initials={})
    assert mech.metadata.get("state_network"), "Expected energy-mode state_network metadata"

    eqs = [eq for eq in mech.equilibria if (getattr(eq, "metadata", {}) or {}).get("source") == "state_network"]
    assert len(eqs) == 1
    eq = eqs[0]
    meta = getattr(eq, "metadata", {}) or {}

    assert eq.stoich_forward == {"A": 1.0, "B": 1.0}
    assert eq.stoich_back == {"C": 1.0}

    dG_act_fwd_J = float(meta["dG_act_fwd_J_per_mol"])
    dG_eq_J = float(meta["dG_eq_J_per_mol"])
    assert dG_act_fwd_J == pytest.approx(0.02 * hartree_to_jmol, rel=0, abs=1e-6)
    assert dG_eq_J == pytest.approx(-0.01 * hartree_to_jmol, rel=0, abs=1e-6)

    prefactor = (k_B * float(T)) / h
    deg_ratio_fwd = float(meta.get("degeneracy_ratio_fwd") or 1.0)
    deg_ratio_rev = float(meta.get("degeneracy_ratio_rev") or 1.0)
    assert deg_ratio_fwd == pytest.approx(2.0, rel=0, abs=0.0)
    assert deg_ratio_rev == pytest.approx(2.0, rel=0, abs=0.0)

    # Per-species std factors: std_TS / (std_A * std_B) = 1 / (2*2) = 0.25
    kf_expected = prefactor * math.exp(-dG_act_fwd_J / (R * float(T))) * deg_ratio_fwd * (1.0 / (2.0 * 2.0))
    dG_act_rev_J = dG_act_fwd_J - dG_eq_J
    kr_expected = prefactor * math.exp(-dG_act_rev_J / (R * float(T))) * deg_ratio_rev * (1.0 / 1.0)
    assert float(eq.kf) == pytest.approx(kf_expected, rel=1e-12)
    assert float(eq.kr) == pytest.approx(kr_expected, rel=1e-12)

    K_expected = math.exp(-dG_eq_J / (R * float(T)))
    assert float(eq.K) == pytest.approx(K_expected, rel=1e-12)

    # Detailed balance in activity convention: kf/kr = K * (std_prod/std_react)
    std_react = 2.0 * 2.0
    std_prod = 1.0
    assert float(eq.kf) / float(eq.kr) == pytest.approx(K_expected * (std_prod / std_react), rel=1e-12)


@pytest.mark.unit
def test_fast_equilibrium_uses_kfast_by_order_and_detailed_balance_with_std_powers():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    T = 300.0
    hartree_to_jmol = scipy.constants.value("Hartree energy") * scipy.constants.N_A
    dG_eq_J = -0.01 * hartree_to_jmol
    K_expected = math.exp(-dG_eq_J / (R * float(T)))

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            "comp: kfast_2 = 1e10",
            "comp: species A type=GS G=0.0 std=2.0 M cref=2.0 M degeneracy=1",
            "comp: species B type=GS G=0.0 std=2.0 M cref=2.0 M degeneracy=1",
            "comp: species C type=GS G=-0.01 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn A + B <-> C",
        ]
    )
    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)
    full_dsl = _wrap_blocks(comp_body=comp_body, generated_body=compiled.generated_reaction_dsl)

    mech = parse_dsl_to_mechanism(full_dsl, initials={})

    # The generated fast equilibrium should appear as an explicit equilibrium line (not state_network-derived).
    assert len(mech.equilibria) == 1
    eq = mech.equilibria[0]
    assert eq.stoich_forward == {"A": 1.0, "B": 1.0}
    assert eq.stoich_back == {"C": 1.0}

    kf = float(eq.kf)
    kr = float(eq.kr)

    # Order m=2 uses kfast_2 override.
    assert kf == pytest.approx(1e10, rel=0, abs=0.0)

    # Detailed balance with std powers: Kc = K * (std_prod/std_react) = K * (1 / (2*2))
    std_react = 2.0 * 2.0
    std_prod = 1.0
    Kc = K_expected * (std_prod / std_react)
    assert kf / kr == pytest.approx(Kc, rel=1e-12)


@pytest.mark.unit
def test_comp_mode_accepts_new_bidirectional_arrow_syntax():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species B type=GS G=-0.01 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn A <=> B",
        ]
    )
    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)
    assert "equilibrium:" in compiled.generated_reaction_dsl


@pytest.mark.unit
def test_comp_mode_parses_integer_star_and_float_stoichiometry_terms():
    from kindred.core.simulator.computational_mode import parse_comp_block

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species B type=GS G=-0.01 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn 2A <=> B",
            "comp: rxn 2 * A <=> B",
            "comp: rxn 1.5A <=> B",
        ]
    )

    spec = parse_comp_block(comp_body)
    assert len(spec.reactions) == 3

    assert spec.reactions[0].reactants == {"A": 2.0}
    assert spec.reactions[0].products == {"B": 1.0}

    assert spec.reactions[1].reactants == {"A": 2.0}
    assert spec.reactions[1].products == {"B": 1.0}

    assert spec.reactions[2].reactants == {"A": 1.5}
    assert spec.reactions[2].products == {"B": 1.0}


@pytest.mark.unit
def test_comp_mode_formats_integer_coefficients_without_zero_stripping():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species B type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn 10A <=> B",
            "comp: rxn 20B <=> A",
        ]
    )

    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)

    eq_lines = [
        ln.strip()
        for ln in compiled.generated_reaction_dsl.splitlines()
        if ln.strip().lower().startswith("equilibrium:")
    ]
    assert any("equilibrium: 10A <-> B" in ln for ln in eq_lines)
    assert any("equilibrium: 20B <-> A" in ln for ln in eq_lines)
    assert "feq__10A__B" in compiled.generated_reaction_dsl
    assert "feq__20B__A" in compiled.generated_reaction_dsl


@pytest.mark.unit
def test_comp_mode_rejects_malformed_spacing_not_masked_by_fallback():
    from kindred.core.simulator.computational_mode import parse_comp_block

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species C type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn 2 * A B <=> C",
        ]
    )

    with pytest.raises(ValueError, match="invalid term"):
        parse_comp_block(comp_body)


@pytest.mark.unit
def test_compiled_fast_equilibrium_includes_cm_id_and_dG_eq_tokens_for_energy_mode_sliders():
    from kindred.core.simulator.computational_mode import parse_comp_block, compile_comp_spec

    hartree_to_jmol = scipy.constants.value("Hartree energy") * scipy.constants.N_A
    expected_dG_eq_kj = (-0.01 * float(hartree_to_jmol)) / 1000.0

    comp_body = "\n".join(
        [
            "comp: T = 300 K",
            "comp: pressure = 1.0 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1.0 M",
            "comp: kfast_default = 1e9",
            # Choose cref=std so G_std == G_input for each species.
            "comp: species A type=GS G=0.0 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: species B type=GS G=-0.01 std=1.0 M cref=1.0 M degeneracy=1",
            "comp: rxn A <-> B",
        ]
    )
    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)

    eq_lines = [ln.strip() for ln in compiled.generated_reaction_dsl.splitlines() if ln.strip().lower().startswith("equilibrium:")]
    assert len(eq_lines) == 1
    line = eq_lines[0]

    assert "cm_id=" in line, "Expected Computational Mode fast-equilibrium id tag for GUI mapping"
    assert "dG_eq=" in line, "Expected ΔG° token for energy-mode fast-equilibrium slider"

    tokens = {}
    for part in [p.strip() for p in line.split(";")[1:] if p.strip()]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        tokens[k.strip()] = v.strip()
    assert "dG_eq" in tokens
    assert float(tokens["dG_eq"]) == pytest.approx(float(expected_dG_eq_kj), rel=0, abs=1e-6)

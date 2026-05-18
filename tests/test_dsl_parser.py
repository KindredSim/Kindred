"""
Tests for DSL parser functionality.

Tests cover:
- Basic reaction parsing
- Equilibrium parsing
- State network parsing
- Parameter extraction
- Error handling with enhanced messages
"""

import logging

import pytest
from kindred.core.simulator.dsl import (
    parse_dsl_to_mechanism,
    parse_dsl,
    extract_parameters_from_dsl,
    DSLError,
)

pytestmark = pytest.mark.unit



class TestBasicReactions:
    """Test basic reaction parsing."""

    def test_simple_reaction(self):
        """Test simple irreversible reaction."""
        dsl = """
        reaction: A -> B; k=1.5
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1
        assert "A" in mechanism.species
        assert "B" in mechanism.species
        assert mechanism.species["A"].initial_conc == 1.0
        assert mechanism.species["B"].initial_conc == 0.0

    def test_successful_mechanism_build_is_not_logged_at_info(self, caplog):
        """Routine parsing should not spam user-facing INFO logs."""
        dsl = """
        reaction: A -> B; k=1.5
        [A] = 1.0
        [B] = 0.0
        """

        with caplog.at_level(logging.INFO, logger="kindred.core.simulator.dsl_build"):
            parse_dsl_to_mechanism(dsl, initials={})

        assert all(
            "Built mechanism from DSL reactions" not in record.getMessage()
            for record in caplog.records
        )

    def test_bimolecular_reaction(self):
        """Test bimolecular reaction."""
        dsl = """
        reaction: A + B -> C; k=0.5
        [A] = 1.0
        [B] = 1.0
        [C] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1
        rxn = mechanism.reactions[0]
        assert rxn.reactants == {"A": 1.0, "B": 1.0}
        assert rxn.products == {"C": 1.0}
        assert rxn.rate_orders == {"A": 1.0, "B": 1.0}
        assert rxn.net_stoich == {"A": -1.0, "B": -1.0, "C": 1.0}

    def test_stoichiometric_coefficients(self):
        """Test non-unity stoichiometric coefficients."""
        dsl = """
        reaction: 2*A -> B; k=1.0
        [A] = 2.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        rxn = mechanism.reactions[0]
        assert rxn.reactants == {"A": 2.0}
        assert rxn.products == {"B": 1.0}
        assert rxn.rate_orders == {"A": 2.0}
        assert rxn.net_stoich == {"A": -2.0, "B": 1.0}

    def test_optional_asterisk_and_alternative_arrows(self):
        dsl = """
        reaction: 2A + B <=> C; kf=1.0, kr=0.1
        initial: A=1.0
        initial: B=1.0
        initial: C=0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1
        eq = mechanism.equilibria[0]

        assert float(eq.kf) == 1.0
        assert float(eq.kr) == 0.1
        assert eq.stoich_forward["A"] == 2.0
        assert eq.stoich_forward["B"] == 1.0
        assert eq.stoich_back["C"] == 1.0

        net = {}
        for sp, coef in eq.stoich_forward.items():
            net[sp] = net.get(sp, 0.0) - float(coef)
        for sp, coef in eq.stoich_back.items():
            net[sp] = net.get(sp, 0.0) + float(coef)
        assert net["A"] == -2.0
        assert net["B"] == -1.0
        assert net["C"] == 1.0

    def test_shorthand_syntax(self):
        """Test shorthand syntax without 'reaction:' prefix."""
        dsl = """
        A -> B; k=1.5
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1
        assert "A" in mechanism.species
        assert "B" in mechanism.species


class TestEquilibria:
    """Test equilibrium parsing."""

    def test_equilibrium_with_K(self):
        """Test equilibrium with equilibrium constant."""
        dsl = """
        equilibrium: A <-> B; kf=1.0; K=2.0
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1
        eq = mechanism.equilibria[0]
        assert eq.Keq is not None

    def test_equilibrium_with_rates(self):
        """Test equilibrium with explicit rate constants."""
        dsl = """
        equilibrium: A <-> B; kf=1.5; kr=0.75
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1
        eq = mechanism.equilibria[0]
        assert eq.kf is not None
        assert eq.kr is not None

    def test_equilibrium_with_thermodynamics(self):
        """Test equilibrium with thermodynamic parameters."""
        dsl = """
        T=310.0
        energy=kJ/mol
        equilibrium: A <-> B; kf=1.0; dG_eq=-8.5
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1

    @pytest.mark.parametrize(
        "line",
        [
            "equilibrium: A <-> B; K=2.0",
            "equilibrium: A <-> B; kr=0.5; K=2.0",
            "equilibrium: A <-> B; kr=0.5; dG_eq=-1.7",
            "equilibrium: A <-> B; kf=1.0; kr=0.5; K=2.0",
            "equilibrium: A <-> B; kf=1.0; kr=0.5; dG_eq=-1.7",
            "equilibrium: A <-> B; kf=1.0; K=2.0; dG_eq=-1.7",
        ],
    )
    def test_equilibrium_requires_kf_and_exactly_one_reverse_authority(self, line):
        with pytest.raises(DSLError, match="kf.*exactly one"):
            parse_dsl_to_mechanism(f"{line}\n[A] = 1.0\n[B] = 0.0", initials={})

    @pytest.mark.parametrize(
        "line",
        [
            "reaction: A <-> B; kr=0.5; K=2.0",
            "reaction: A <-> B; kf=1.0; kr=0.5; K=2.0",
            "reaction: A <-> B; kf=1.0; kr=0.5; dG_eq=-1.7",
            "A <-> B; kr=0.5; K=2.0",
            "A <-> B; kf=1.0; kr=0.5; K=2.0",
        ],
    )
    def test_reversible_reaction_requires_kf_and_exactly_one_reverse_authority(self, line):
        with pytest.raises(DSLError, match="kf.*exactly one"):
            parse_dsl_to_mechanism(f"{line}\n[A] = 1.0\n[B] = 0.0", initials={})


class TestThermodynamicParameters:
    """Test thermodynamic parameter parsing."""

    def test_eyring_reaction(self):
        """Test Eyring equation with dG_act."""
        dsl = """
        T=310.0
        energy=kJ/mol
        reaction: A -> B; dG_act=75.5
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1
        # Rate constant should be calculated from thermodynamics

    def test_arrhenius_reaction(self):
        """Test Arrhenius equation."""
        dsl = """
        T=298.15
        energy=kJ/mol
        reaction: A -> B; A=1.5e10; Ea=65.0
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1

    def test_per_reaction_temperature_rejected(self):
        """Per-reaction T= override is rejected (use global T= instead)."""
        dsl = """
        T=298.15
        energy=kJ/mol
        reaction: A -> B; dG_act=75.5; T=350.0
        [A] = 1.0
        [B] = 0.0
        """
        with pytest.raises(DSLError, match="Per-reaction T="):
            parse_dsl_to_mechanism(dsl, initials={})


class TestGlobalSettings:
    """Test global parameter settings."""

    def test_global_temperature(self):
        """Test global temperature setting."""
        dsl = """
        T=310.0
        reaction: A -> B; k=1.0
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1

    def test_global_energy_unit(self):
        """Test global energy unit setting."""
        dsl = """
        energy=kcal/mol
        T=298.15
        reaction: A -> B; dG_act=18.0
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 1


class TestErrorHandling:
    """Test enhanced error messages."""

    def test_missing_arrow_error(self):
        """Test error for missing reaction arrow."""
        dsl = """
        reaction: A B; k=1.0
        """
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})

        error_msg = str(exc_info.value)
        assert "must contain" in error_msg.lower()
        assert "example" in error_msg.lower()

    def test_invalid_temperature_error(self):
        """Test error for invalid temperature."""
        dsl = """
        T=-10.0
        """
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})

        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()

    def test_invalid_energy_unit_error(self):
        """Test error for invalid energy unit."""
        dsl = """
        energy=eV
        """
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})

        error_msg = str(exc_info.value)
        assert "kJ/mol" in error_msg or "kcal/mol" in error_msg

    def test_invalid_species_term_error(self):
        """Test error for invalid species term."""
        dsl = """
        reaction: @A -> B; k=1.0
        """
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})

        error_msg = str(exc_info.value)
        assert "invalid species term" in error_msg.lower()

    def test_invalid_shorthand_reaction_reports_error(self):
        """Bare arrow syntax with malformed stoichiometry should raise."""
        dsl = "A -> 2**B"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})
        error_msg = str(exc_info.value).lower()
        assert "invalid reaction" in error_msg or "invalid species term" in error_msg

    def test_unrecognized_directive_raises_error(self):
        """Unknown non-comment lines must not be silently ignored."""
        dsl = """
        nonsense directive
        reaction: A -> B; k=0.1
        """
        with pytest.raises(DSLError) as exc_info:
            parse_dsl_to_mechanism(dsl, initials={})
        assert "unrecognized line" in str(exc_info.value).lower()

    @pytest.mark.parametrize("parser", [parse_dsl, parse_dsl_to_mechanism])
    def test_reversible_eyring_without_equilibrium_data_reports_line(self, parser):
        """Reversible Eyring without reverse authority should surface DSLError with context, not ValueError."""
        dsl = "reaction: A <-> B; dG_act=10"
        with pytest.raises(DSLError) as exc_info:
            parser(dsl)

        err = exc_info.value
        assert err.line_number == 1
        assert "reaction: A <-> B" in (err.line_content or "")
        assert "exactly one of kr or Keq/dG_eq" in str(err)

    def test_empty_reactants_allowed(self):
        """Test that empty reactants are allowed (e.g., photochemical reactions)."""
        dsl = """
        reaction: -> B; k=1.0
        """
        # Should NOT raise - this is valid for source terms / photochemistry
        mech = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mech.reactions) == 1
        assert 'B' in mech.species
        assert mech.reactions[0].order == 0  # Zero-order reaction


class TestParameterExtraction:
    """Test parameter extraction from DSL."""

    def test_extract_rate_constants(self):
        """Test extracting rate constants."""
        dsl = """
        reaction: A -> B; k=1.5
        reaction: B -> C; k=0.8
        """
        params = extract_parameters_from_dsl(dsl)

        assert len(params) >= 2
        param_names = [p.name for p in params]
        assert "k1" in param_names
        assert "k2" in param_names
        assert "k" not in param_names

    def test_extract_thermodynamic_params(self):
        """Test extracting thermodynamic parameters."""
        dsl = """
        T=310.0
        reaction: A -> B; dG_act=75.5
        """
        params = extract_parameters_from_dsl(dsl)

        # dG_act is step-local input syntax; the public parameter identity is canonical k1.
        assert [p.name for p in params] == ["k1"]

    def test_extract_parameters_from_dsl_returns_canonical_equilibrium_names(self):
        dsl = """
        equilibrium: A <-> B; kf=1.0; kr=0.25
        """
        params = extract_parameters_from_dsl(dsl)

        assert [p.name for p in params] == ["kf1", "kr1", "Keq1"]
        assert [p.editable for p in params] == [True, True, False]

    def test_extract_parameters_from_dsl_rejects_duplicate_equilibrium_aliases(self):
        dsl = """
        equilibrium: A <-> B; kf=1.0; Keq=2.0; K_eq=3.0
        """

        with pytest.raises(DSLError, match="Duplicate parameter"):
            extract_parameters_from_dsl(dsl)


class TestComplexMechanisms:
    """Test complex multi-step mechanisms."""

    def test_consecutive_reactions(self):
        """Test A → B → C mechanism."""
        dsl = """
        reaction: A -> B; k=1.0
        reaction: B -> C; k=0.5
        [A] = 1.0
        [B] = 0.0
        [C] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 2
        assert len(mechanism.species) == 3

    def test_parallel_reactions(self):
        """Test parallel reactions A → B and A → C."""
        dsl = """
        reaction: A -> B; k=0.7
        reaction: A -> C; k=0.3
        [A] = 1.0
        [B] = 0.0
        [C] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.reactions) == 2
        assert "A" in mechanism.species
        assert "B" in mechanism.species
        assert "C" in mechanism.species

    def test_reversible_and_irreversible(self):
        """Test mix of reversible and irreversible reactions."""
        dsl = """
        equilibrium: A <-> B; kf=1.0; K=2.0
        reaction: B -> C; k=0.5
        [A] = 1.0
        [B] = 0.0
        [C] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1
        assert len(mechanism.reactions) == 1


class TestPublicAliasAndStateMemberParsing:
    """Public parser coverage for helper-backed DSL contracts."""

    @pytest.mark.parametrize(("directive", "expected"), [("C0=1.0", 1.0), ("c0=2.5", 2.5), ("c°=0.75", 0.75)])
    def test_global_standard_concentration_aliases_parse_through_public_dsl(self, directive, expected):
        result = parse_dsl(f"{directive}\nreaction: A -> B; k=1")

        assert result.ir.standard_conc_M == pytest.approx(expected)

    @pytest.mark.parametrize(("directive", "expected"), [("kappa=0.8", 0.8), ("κ=1.25", 1.25)])
    def test_global_kappa_aliases_parse_through_public_dsl(self, directive, expected):
        result = parse_dsl(f"{directive}\nreaction: A -> B; k=1")

        assert result.ir.kappa_global == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("directive", "expected_message"),
        [
            pytest.param("C0=0", "C0 must be positive", id="standard-concentration-zero"),
            pytest.param("c0=-1", "C0 must be positive", id="standard-concentration-negative"),
            pytest.param("c°=nope", "Invalid number", id="standard-concentration-invalid"),
            pytest.param("kappa=0", "κ must be positive", id="kappa-zero"),
            pytest.param("κ=-1", "κ must be positive", id="kappa-negative"),
            pytest.param("kappa=nope", "Invalid number", id="kappa-invalid"),
        ],
    )
    def test_invalid_global_alias_directives_fail_through_public_dsl(self, directive, expected_message):
        with pytest.raises(DSLError, match=expected_message):
            parse_dsl(f"{directive}\nreaction: A -> B; k=1")

    def test_mixed_case_rate_aliases_parse_through_public_dsl(self):
        mechanism = parse_dsl_to_mechanism("reaction: A <-> B; Kf=1.5; kR=0.25", initials={})

        eq = mechanism.equilibria[0]
        assert float(eq.kf) == pytest.approx(1.5)
        assert float(eq.kr) == pytest.approx(0.25)

    def test_equilibrium_K_alias_remains_distinct_from_irreversible_k(self):
        mechanism = parse_dsl_to_mechanism("reaction: A <-> B; k=1.0; K=2.0", initials={})

        eq = mechanism.equilibria[0]
        assert float(eq.kf) == pytest.approx(1.0)
        assert float(eq.kr) == pytest.approx(0.5)
        assert float(eq.Keq) == pytest.approx(2.0)

    def test_malformed_key_value_chunk_reports_public_dsl_error(self):
        with pytest.raises(DSLError, match=r"Expected key=value pair"):
            parse_dsl("reaction: A -> B; k=1.0; nope; kr=2.0")

    def test_state_members_parse_coefficients_through_public_dsl(self):
        result = parse_dsl("state: AB, kind=GS, energy=0, members=2A+B\nreaction: A -> B; k=1")

        assert result.ir.state_network.get("AB").members == ("A", "A", "B")

    def test_state_network_direct_equilibrium_builds_through_public_dsl(self):
        mechanism = parse_dsl_to_mechanism(
            "\n".join(
                [
                    "energy=kJ/mol",
                    "state: A, kind=GS, energy=0",
                    "state: B, kind=GS, energy=1",
                    "edge: A,B",
                    "init: A=1.0, B=0.0",
                ]
            ),
            initials={},
        )

        assert len(mechanism.equilibria) == 1
        assert mechanism.equilibria[0].metadata["source"] == "state_network_direct"

    @pytest.mark.parametrize(
        ("members", "message"),
        [
            ("0A+B", "positive integers"),
            ("2*A+B", "invalid members term"),
            ("A-B", "invalid members term"),
        ],
    )
    def test_invalid_state_members_report_public_dsl_error(self, members, message):
        with pytest.raises(DSLError, match=message):
            parse_dsl(f"state: AB, kind=GS, energy=0, members={members}\nreaction: A -> B; k=1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

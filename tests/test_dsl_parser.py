"""
Tests for DSL parser functionality.

Tests cover:
- Basic reaction parsing
- Equilibrium parsing
- State network parsing
- Parameter extraction
- Error handling with enhanced messages
"""

import pytest
from kindred.core.simulator.dsl import (
    parse_dsl_to_mechanism,
    parse_dsl,
    extract_parameters_from_dsl,
    DSLError,
    _bool_from_str,
    _parse_kappa_directive,
    _parse_keyvals,
    _parse_members_expr,
    _parse_standard_conc_directive,
)


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
        assert rxn.stoich["A"] == -1.0
        assert rxn.stoich["B"] == -1.0
        assert rxn.stoich["C"] == 1.0

    def test_stoichiometric_coefficients(self):
        """Test non-unity stoichiometric coefficients."""
        dsl = """
        reaction: 2*A -> B; k=1.0
        [A] = 2.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        rxn = mechanism.reactions[0]
        assert rxn.stoich["A"] == -2.0
        assert rxn.stoich["B"] == 1.0

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
        equilibrium: A <-> B; dG_eq=-8.5
        [A] = 1.0
        [B] = 0.0
        """
        mechanism = parse_dsl_to_mechanism(dsl, initials={})

        assert len(mechanism.equilibria) == 1


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
        """Reversible Eyring without K/dG_eq should surface DSLError with context, not ValueError."""
        dsl = "reaction: A <-> B; dG_act=10"
        with pytest.raises(DSLError) as exc_info:
            parser(dsl)

        err = exc_info.value
        assert err.line_number == 1
        assert "reaction: A <-> B" in (err.line_content or "")
        assert "requires K" in str(err)

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
        assert "k" in param_names or any("k" in name for name in param_names)

    def test_extract_thermodynamic_params(self):
        """Test extracting thermodynamic parameters."""
        dsl = """
        T=310.0
        reaction: A -> B; dG_act=75.5
        """
        params = extract_parameters_from_dsl(dsl)

        # Should extract dG_act parameter
        assert len(params) > 0

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


class TestExtractedHelperFunctions:
    """Direct tests for extracted helper functions in kindred.core.simulator.dsl."""

    @pytest.mark.parametrize(
        ("s", "expected"),
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("  TRUE  ", True),
            ("Yes", True),
            ("oN", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("  False  ", False),
            ("NO", False),
            ("OfF", False),
        ],
    )
    def test_bool_from_str_permutations(self, s, expected):
        assert _bool_from_str(s) is expected

    @pytest.mark.parametrize("s", ["", "2", "truthy", "y", "n", "enable", "disable"])
    def test_bool_from_str_invalid_inputs_raise_dslerror(self, s):
        with pytest.raises(DSLError, match=r"Invalid boolean"):
            _bool_from_str(s)

    def test_parse_keyvals_empty_returns_empty_dict(self):
        assert _parse_keyvals("") == {}
        assert _parse_keyvals("   ") == {}

    def test_parse_keyvals_splits_on_commas_and_semicolons_and_strips_values(self):
        kv = _parse_keyvals("kf=1.5; kr = 0.25,  energy=kJ/mol  , ,")
        assert kv["kf"] == "1.5"
        assert kv["kr"] == "0.25"
        assert kv["energy"] == "kJ/mol"

    def test_parse_keyvals_preserves_equilibrium_K_case(self):
        kv = _parse_keyvals("K=2.0, k=1.0")
        assert kv["Keq"] == "2.0"
        assert kv["k"] == "1.0"

    def test_parse_keyvals_preserves_irreversible_k_distinct_from_mixed_case_keq_alias(self):
        kv = _parse_keyvals("k=1.0, kEq=2.0")
        assert kv["k"] == "1.0"
        assert kv["Keq"] == "2.0"

    def test_parse_keyvals_normalizes_mixed_case_rate_aliases(self):
        kv = _parse_keyvals("Kf=1.5, kR=0.25")
        assert kv["kf"] == "1.5"
        assert kv["kr"] == "0.25"

    def test_parse_keyvals_normalizes_common_aliases(self):
        kv = _parse_keyvals("t=298.15, c0=1.0, kappa=0.8, dg_act=75.5")
        assert "T" in kv
        assert "C0" in kv
        assert "κ" in kv
        assert "dG_act" in kv

    def test_parse_keyvals_allows_equals_in_value(self):
        kv = _parse_keyvals("source=a=b=c")
        assert kv["source"] == "a=b=c"

    def test_parse_keyvals_rejects_malformed_chunks_without_equals(self):
        with pytest.raises(DSLError, match=r"Expected key=value pair"):
            _parse_keyvals("k=1.0, nope, kr=2.0")

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("A+B", ("A", "B")),
            ("2A+B", ("A", "A", "B")),
            ("2 A + B", ("A", "A", "B")),
            ("A+2B+C", ("A", "B", "B", "C")),
            ("A++B", ("A", "B")),
        ],
    )
    def test_parse_members_expr_valid(self, expr, expected):
        assert _parse_members_expr(expr) == expected

    @pytest.mark.parametrize("expr", ["", "   ", "+", " + + "])
    def test_parse_members_expr_empty_raises_valueerror(self, expr):
        with pytest.raises(ValueError, match=r"members cannot be empty"):
            _parse_members_expr(expr)

    @pytest.mark.parametrize("expr", ["0A+B", "0A", "0A+0B"])
    def test_parse_members_expr_nonpositive_coeff_raises_valueerror(self, expr):
        with pytest.raises(ValueError, match=r"positive integers"):
            _parse_members_expr(expr)

    @pytest.mark.parametrize("expr", ["2*A+B", "A*", "A-B", "A,B"])
    def test_parse_members_expr_invalid_term_raises_valueerror(self, expr):
        with pytest.raises(ValueError, match=r"invalid members term"):
            _parse_members_expr(expr)

    @pytest.mark.parametrize(("line", "expected"), [("C0=1.0", 1.0), ("c0 = 2.5", 2.5), ("c°=0.75", 0.75)])
    def test_parse_standard_conc_directive_valid(self, line, expected):
        assert _parse_standard_conc_directive(line) == expected

    @pytest.mark.parametrize("line", ["C0=0", "c0=-1", "c°=0"])
    def test_parse_standard_conc_directive_nonpositive_raises(self, line):
        with pytest.raises(DSLError, match=r"C0 must be positive"):
            _parse_standard_conc_directive(line)

    @pytest.mark.parametrize("line", ["C0=", "c0=  "])
    def test_parse_standard_conc_directive_empty_raises(self, line):
        with pytest.raises(DSLError, match=r"requires a numeric value"):
            _parse_standard_conc_directive(line)

    def test_parse_standard_conc_directive_invalid_number_raises(self):
        with pytest.raises(DSLError, match=r"Invalid number"):
            _parse_standard_conc_directive("c°=nope")

    @pytest.mark.parametrize(("line", "expected"), [("kappa=0.8", 0.8), ("κ=1.25", 1.25)])
    def test_parse_kappa_directive_valid(self, line, expected):
        assert _parse_kappa_directive(line) == expected

    @pytest.mark.parametrize("line", ["kappa=0", "κ=-1"])
    def test_parse_kappa_directive_nonpositive_raises(self, line):
        with pytest.raises(DSLError, match=r"κ must be positive"):
            _parse_kappa_directive(line)

    @pytest.mark.parametrize("line", ["kappa=", "κ=  "])
    def test_parse_kappa_directive_empty_raises(self, line):
        with pytest.raises(DSLError, match=r"requires a numeric value"):
            _parse_kappa_directive(line)

    def test_parse_kappa_directive_invalid_number_raises(self):
        with pytest.raises(DSLError, match=r"Invalid number"):
            _parse_kappa_directive("kappa=nope")

    def test_parse_kappa_directive_k_is_not_a_kappa_alias(self):
        with pytest.raises(DSLError):
            _parse_kappa_directive("k=0.8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

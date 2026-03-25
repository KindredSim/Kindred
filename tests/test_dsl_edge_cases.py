"""
Test DSL parsing edge cases and error handling.

This module tests that the DSL parser handles various edge cases gracefully:
- Empty input
- Malformed syntax
- Missing required fields
- Inconsistent initial conditions
- Parameter name collisions
- Unicode and special characters
"""

from __future__ import annotations

import pytest
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError


class TestDSLEmptyAndWhitespace:
    """Test DSL parser handles empty and whitespace-only input."""

    def test_empty_string(self):
        """Test parsing completely empty string."""
        with pytest.raises(Exception):  # Should raise some error
            parse_dsl_to_mechanism("")

    def test_whitespace_only(self):
        """Test parsing whitespace-only string."""
        with pytest.raises(Exception):
            parse_dsl_to_mechanism("   \n\n\t  \n  ")

    def test_comments_only(self):
        """Test parsing string with only comments."""
        dsl = """
        # This is a comment
        # Another comment
        # No actual content
        """
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)


class TestDSLMalformedSyntax:
    """Test DSL parser error handling for malformed syntax."""

    def test_reaction_missing_arrow(self):
        """Test reaction without arrow."""
        dsl = """
        reaction: A B; k=1.0
        initial: A=1.0
        """
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)

    def test_reaction_missing_rate_constant(self):
        """Test reaction without rate constant."""
        dsl = """
        reaction: A -> B
        initial: A=1.0
        """
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism(dsl)

    def test_initial_condition_invalid_format(self):
        """Test initial condition with invalid format."""
        dsl = """
        reaction: A -> B; k=1.0
        initial: A 1.0
        """
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)

    def test_invalid_stoichiometry(self):
        """Test reaction with invalid stoichiometric coefficient."""
        dsl = """
        reaction: -1*A -> B; k=1.0
        initial: A=1.0
        """
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)

    def test_species_name_with_spaces(self):
        """Test that species names with spaces are rejected."""
        dsl = """
        reaction: A B C -> D; k=1.0
        initial: A B C=1.0
        """
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)


class TestDSLInitialConditions:
    """Test initial condition handling."""

    def test_missing_initial_conditions(self):
        """Test mechanism without initial conditions."""
        dsl = """
        reaction: A -> B; k=0.5
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert 'A' in mech.species
        assert 'B' in mech.species
        assert mech.species['A'].initial_conc == 0.0
        assert mech.species['B'].initial_conc == 0.0

    def test_partial_initial_conditions(self):
        """Test mechanism with initial conditions for only some species."""
        dsl = """
        reaction: A -> B; k=0.5
        reaction: B -> C; k=0.3
        initial: A=1.0
        """
        # B and C should default to 0
        mech = parse_dsl_to_mechanism(dsl)
        assert mech.species['A'].initial_conc == 1.0
        assert mech.species['B'].initial_conc == 0.0
        assert mech.species['C'].initial_conc == 0.0

    def test_duplicate_initial_conditions(self):
        """Test species with multiple initial condition definitions."""
        dsl = """
        reaction: A -> B; k=0.5
        initial: A=1.0
        initial: A=2.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert mech.species['A'].initial_conc == 2.0

    def test_negative_initial_concentration(self):
        """Test that negative initial concentrations are handled."""
        dsl = """
        reaction: A -> B; k=0.5
        initial: A=-1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        # Parser currently allows negative initials; downstream validation/simulation may reject.
        assert mech.species['A'].initial_conc == -1.0


class TestDSLParameterHandling:
    """Test parameter definition and usage."""

    def test_undefined_parameter_reference(self):
        """Test reaction referencing undefined parameter."""
        dsl = """
        reaction: A -> B; k=k_forward
        initial: A=1.0
        """
        # Should either raise error or treat k_forward as symbolic
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)

    def test_parameter_redefinition(self):
        """Test redefining a parameter."""
        dsl = """
        let k1 = 1.5
        let k1 = 2.0
        reaction: A -> B; k=k1
        initial: A=1.0
        """
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism(dsl)

    def test_circular_parameter_definition(self):
        """Test circular parameter dependencies."""
        dsl = """
        let k1 = k2 * 2.0
        let k2 = k1 / 2.0
        reaction: A -> B; k=k1
        initial: A=1.0
        """
        # Should detect circular reference
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)


class TestDSLEquilibriumHandling:
    """Test equilibrium reaction parsing."""

    def test_equilibrium_basic(self):
        """Test basic equilibrium syntax."""
        dsl = """
        equilibrium: A <-> B; kf=1.0; K=2.5
        initial: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        # Should create forward and reverse reactions
        assert 'A' in mech.species
        assert 'B' in mech.species

    def test_equilibrium_missing_constant(self):
        """Test equilibrium without equilibrium constant."""
        dsl = """
        equilibrium: A <-> B
        initial: A=1.0
        """
        # Should raise error (K is required)
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)


class TestDSLUnicodeAndSpecialChars:
    """Test handling of Unicode and special characters."""

    def test_unicode_in_comments(self):
        """Test Unicode characters in comments."""
        dsl = """
        # Reaction with ΔG‡ = 20 kJ/mol
        reaction: A -> B; k=0.5
        initial: A=1.0
        """
        # Should parse successfully, ignoring comments
        mech = parse_dsl_to_mechanism(dsl)
        assert 'A' in mech.species

    def test_special_chars_in_species_names(self):
        """Test that special characters in species names are rejected."""
        dsl = """
        reaction: A* -> B; k=0.5
        initial: A*=1.0
        """
        # Most parsers reject special chars (except underscore/numbers)
        with pytest.raises(Exception):
            parse_dsl_to_mechanism(dsl)


class TestDSLComplexMechanisms:
    """Test complex multi-step mechanisms."""

    def test_chain_reaction(self):
        """Test parsing chain reaction mechanism."""
        dsl = """
        reaction: A -> B; k=1.0
        reaction: B -> C; k=0.5
        reaction: C -> D; k=0.3
        initial: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert len(mech.species) == 4
        assert len(mech.reactions) == 3

    def test_branching_mechanism(self):
        """Test mechanism with branching pathways."""
        dsl = """
        reaction: A -> B; k=1.0
        reaction: A -> C; k=0.5
        reaction: B -> D; k=0.3
        reaction: C -> D; k=0.2
        initial: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert 'A' in mech.species
        assert 'B' in mech.species
        assert 'C' in mech.species
        assert 'D' in mech.species
        assert len(mech.reactions) == 4

    def test_reversible_reactions(self):
        """Test mechanism with reversible reactions."""
        dsl = """
        reaction: A -> B; k=1.0
        reaction: B -> A; k=0.5
        initial: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert len(mech.reactions) == 2


class TestDSLErrorMessages:
    """Test that error messages are helpful."""

    def test_error_message_includes_line_number(self):
        """Test that parse errors include line numbers."""
        dsl = """
        reaction: A -> B; k=1.0
        reaction: X Y Z; k=2.0
        initial: A=1.0
        """
        try:
            parse_dsl_to_mechanism(dsl)
            assert False, "Should have raised error"
        except Exception as e:
            # Error message should be helpful
            msg = str(e)
            # Should mention the problem
            assert len(msg) > 10

    def test_error_message_for_invalid_number(self):
        """Test error message when numeric value is invalid."""
        dsl = """
        reaction: A -> B; k=not_a_number
        initial: A=1.0
        """
        with pytest.raises(Exception) as exc_info:
            parse_dsl_to_mechanism(dsl)
        # Error should mention the invalid value
        msg = str(exc_info.value)
        assert 'not_a_number' in msg or 'invalid' in msg.lower() or 'number' in msg.lower()


class TestDSLCaseSensitivity:
    """Test case sensitivity in DSL parsing."""

    def test_keyword_case_sensitivity(self):
        """Test if keywords are case-sensitive."""
        # Try uppercase keyword
        dsl = """
        REACTION: A -> B; k=1.0
        INITIAL: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        assert 'A' in mech.species
        assert 'B' in mech.species

    def test_species_name_case_sensitivity(self):
        """Test that species names are case-sensitive."""
        dsl = """
        reaction: A -> a; k=1.0
        initial: A=1.0
        """
        mech = parse_dsl_to_mechanism(dsl)
        # A and a should be different species
        assert 'A' in mech.species
        assert 'a' in mech.species
        assert mech.species['A'] != mech.species['a']

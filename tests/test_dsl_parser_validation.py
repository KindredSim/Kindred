"""
Regression tests for DSL parser validation hardening.

Each test must fail BEFORE the corresponding fix is applied and pass AFTER.
"""

from __future__ import annotations

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism, parse_dsl, DSLError
from kindred.core.simulator.dsl_parse import _KEY_ALIASES


# ---------------------------------------------------------------------------
# Fix 1: Empty directives must raise DSLError, not KeyError or crash
# ---------------------------------------------------------------------------


class TestEmptyDirectives:
    def test_energy_directive_no_value_raises(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=")

    def test_temperature_directive_no_value_raises(self):
        with pytest.raises(DSLError):
            parse_dsl("t=\nA -> B; k=1")

    def test_c0_directive_no_value_raises(self):
        with pytest.raises(DSLError):
            parse_dsl("c0=\nA -> B; k=1")

    def test_kappa_directive_no_value_raises(self):
        with pytest.raises(DSLError):
            parse_dsl("kappa=\nA -> B; k=1")


# ---------------------------------------------------------------------------
# Fix 2: Negative / inf / NaN rate constants rejected (zero allowed)
# ---------------------------------------------------------------------------


class TestRateConstantValidation:
    def test_negative_rate_constant_rejected(self):
        with pytest.raises(DSLError, match="non-negative"):
            parse_dsl_to_mechanism("A -> B; k=-1.0", initials={})

    def test_inf_rate_constant_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism("A -> B; k=inf", initials={})

    def test_nan_rate_constant_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism("A -> B; k=nan", initials={})

    def test_negative_inf_rate_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism("A -> B; k=-inf", initials={})

    def test_zero_rate_constant_allowed(self):
        result = parse_dsl("A -> B; k=0")
        assert result.ir is not None

    def test_negative_kf_rejected_equilibrium(self):
        with pytest.raises(DSLError, match="non-negative"):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=-1; kr=1", initials={}
            )

    def test_negative_kr_rejected_equilibrium(self):
        with pytest.raises(DSLError, match="non-negative"):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=1; kr=-1", initials={}
            )

    def test_negative_K_rejected_equilibrium(self):
        with pytest.raises(DSLError, match="non-negative"):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=1; K=-2", initials={}
            )

    def test_inf_kf_rejected_equilibrium(self):
        with pytest.raises(DSLError):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=inf; kr=1", initials={}
            )

    def test_zero_kf_equilibrium_rejected(self):
        """Zero kf in equilibrium implies K=0, which is physically invalid."""
        with pytest.raises(DSLError):
            parse_dsl("equilibrium: A <=> B; kf=0; kr=1")


# ---------------------------------------------------------------------------
# Fix 3: Per-reaction T= and energy= are rejected
# ---------------------------------------------------------------------------


class TestPerReactionOverridesRejected:
    def test_per_reaction_temperature_rejected(self):
        with pytest.raises(DSLError, match="Per-reaction T="):
            parse_dsl_to_mechanism(
                "reaction: A -> B; k=1; T=350", initials={}
            )

    def test_per_reaction_energy_unit_rejected(self):
        with pytest.raises(DSLError, match="Per-reaction energy="):
            parse_dsl_to_mechanism(
                "energy=kJ/mol\nreaction: A -> B; dG_act=10; energy=kcal/mol",
                initials={},
            )

    def test_per_equilibrium_temperature_rejected(self):
        with pytest.raises(DSLError, match="Per-reaction T="):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=1; kr=0.5; T=350", initials={}
            )

    def test_per_equilibrium_energy_rejected(self):
        with pytest.raises(DSLError, match="Per-reaction energy="):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=1; kr=0.5; energy=kJ/mol",
                initials={},
            )

    def test_bare_arrow_temperature_rejected(self):
        with pytest.raises(DSLError, match="Per-reaction T="):
            parse_dsl_to_mechanism("A -> B; k=1; T=350", initials={})


# ---------------------------------------------------------------------------
# Fix 4: Unknown reaction/equilibrium parameters rejected
# ---------------------------------------------------------------------------


class TestUnknownParametersRejected:
    def test_unknown_reaction_param_rejected(self):
        with pytest.raises(DSLError, match="Unknown reaction parameter"):
            parse_dsl_to_mechanism("A -> B; k=1; foo=5", initials={})

    def test_unknown_equilibrium_param_rejected(self):
        with pytest.raises(DSLError, match="Unknown equilibrium parameter"):
            parse_dsl_to_mechanism(
                "equilibrium: A <=> B; kf=1; kr=1; bar=2", initials={}
            )

    def test_k_fast_on_reaction_rejected(self):
        """k_fast is internally computed, not a user-facing param."""
        with pytest.raises(DSLError, match="Unknown reaction parameter"):
            parse_dsl_to_mechanism("A -> B; k=1; k_fast=1e6", initials={})

    def test_valid_reaction_params_accepted(self):
        """All known reaction params should still work."""
        result = parse_dsl("A -> B; k=1.0")
        assert result.ir is not None

    def test_valid_equilibrium_params_accepted(self):
        """All known equilibrium params should still work."""
        result = parse_dsl("equilibrium: A <=> B; kf=1; kr=0.5")
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Fix 5: Dead p0/p° alias removed
# ---------------------------------------------------------------------------


class TestP0AliasRemoved:
    def test_p0_not_in_key_aliases(self):
        assert "p0" not in _KEY_ALIASES

    def test_p_degree_not_in_key_aliases(self):
        assert "p°" not in _KEY_ALIASES


# ---------------------------------------------------------------------------
# Fix 6: Duplicate state names raise DSLError (not ValueError)
# ---------------------------------------------------------------------------


class TestDuplicateStateWrapped:
    def test_duplicate_state_raises_dsl_error(self):
        dsl = """\
state: A, kind=GS, energy=0
state: A, kind=GS, energy=0
"""
        with pytest.raises(DSLError, match="already exists"):
            parse_dsl(dsl)

    def test_duplicate_state_has_line_number(self):
        dsl = """\
state: A, kind=GS, energy=0
state: A, kind=GS, energy=5
"""
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2


# ---------------------------------------------------------------------------
# Fix 7: Malformed stoichiometry (empty terms) rejected
# ---------------------------------------------------------------------------


class TestMalformedStoichiometry:
    def test_double_plus_stoichiometry_rejected(self):
        with pytest.raises(DSLError, match="[Mm]alformed stoichiometry"):
            parse_dsl_to_mechanism("A ++ B -> C; k=1", initials={})

    def test_trailing_plus_rejected(self):
        with pytest.raises(DSLError, match="[Mm]alformed stoichiometry"):
            parse_dsl_to_mechanism("A + -> C; k=1", initials={})

    def test_leading_plus_rejected(self):
        with pytest.raises(DSLError, match="[Mm]alformed stoichiometry"):
            parse_dsl_to_mechanism("+ A -> C; k=1", initials={})

    def test_source_term_still_works(self):
        """Empty LHS (source/sink) must still parse: -> B; k=1."""
        result = parse_dsl("-> B; k=1")
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Fix 8: J/mol accepted as global energy unit
# ---------------------------------------------------------------------------


class TestJPerMolAccepted:
    def test_j_per_mol_global_energy_accepted(self):
        dsl = "energy=J/mol\nA -> B; k=1"
        result = parse_dsl(dsl)
        assert result.ir.energy_unit == "J/mol"

    def test_j_per_mol_dg_act_correct(self):
        """1000 J/mol with energy=J/mol should match 1 kJ/mol with energy=kJ/mol."""
        dsl_j = "energy=J/mol\nA -> B; dG_act=1000"
        dsl_kj = "energy=kJ/mol\nA -> B; dG_act=1"
        ir_j = parse_dsl(dsl_j).ir
        ir_kj = parse_dsl(dsl_kj).ir
        step_j = ir_j.steps[0]
        step_kj = ir_kj.steps[0]
        assert abs(step_j.kf - step_kj.kf) < 1e-6 * abs(step_kj.kf)


# ---------------------------------------------------------------------------
# Fix 9: Case-insensitive energy unit matching
# ---------------------------------------------------------------------------


class TestCaseInsensitiveEnergyUnit:
    def test_lowercase_energy_unit_accepted(self):
        result = parse_dsl("energy=kj/mol\nA -> B; k=1")
        assert result.ir.energy_unit == "kJ/mol"

    def test_uppercase_energy_unit_accepted(self):
        result = parse_dsl("energy=KJ/MOL\nA -> B; k=1")
        assert result.ir.energy_unit == "kJ/mol"

    def test_mixed_case_kcal_accepted(self):
        result = parse_dsl("energy=Kcal/Mol\nA -> B; k=1")
        assert result.ir.energy_unit == "kcal/mol"

    def test_lowercase_j_mol_accepted(self):
        result = parse_dsl("energy=j/mol\nA -> B; k=1")
        assert result.ir.energy_unit == "J/mol"

    def test_invalid_unit_still_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=eV\nA -> B; k=1")

    @pytest.mark.parametrize("value", ["kj", "kcal", "J", "j", " KJ ", "Kcal"])
    def test_short_form_unit_still_rejected(self, value: str):
        with pytest.raises(DSLError):
            parse_dsl(f"energy={value}\nA -> B; k=1")


# ---------------------------------------------------------------------------
# Non-finite directives rejected (NaN/inf in T, C0, κ, initials)
# ---------------------------------------------------------------------------


class TestNonFiniteDirectivesRejected:
    def test_t_nan_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("T=nan\nA -> B; k=1")

    def test_t_inf_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("T=inf\nA -> B; k=1")

    def test_c0_nan_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("C0=nan\nA -> B; k=1")

    def test_kappa_global_inf_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("κ=inf\nA -> B; k=1")

    def test_initial_nan_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("[A]=nan\nA -> B; k=1")

    def test_initial_inf_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("[A]=inf\nA -> B; k=1")


# ---------------------------------------------------------------------------
# K validation on reversible reaction: lines
# ---------------------------------------------------------------------------


class TestReactionKValidation:
    def test_reaction_K_zero_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A <-> B; k=1; K=0")

    def test_reaction_K_negative_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A <-> B; k=1; K=-1")

    def test_reaction_K_positive_accepted(self):
        result = parse_dsl("reaction: A <-> B; k=1; K=2")
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Arrhenius A validation
# ---------------------------------------------------------------------------


class TestArrheniusAValidation:
    def test_arrhenius_A_negative_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; A=-1; Ea=50")

    def test_arrhenius_A_zero_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; A=0; Ea=50")

    def test_arrhenius_A_positive_accepted(self):
        result = parse_dsl("reaction: A -> B; A=1e10; Ea=50")
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Per-step kappa validation
# ---------------------------------------------------------------------------


class TestPerStepKappaValidation:
    def test_per_step_kappa_zero_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; dG_act=50; κ=0")

    def test_per_step_kappa_negative_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; dG_act=50; κ=-1")

    def test_per_step_kappa_positive_accepted(self):
        result = parse_dsl("reaction: A -> B; dG_act=50; κ=0.5")
        assert result.ir is not None


# ---------------------------------------------------------------------------
# State degeneracy validation
# ---------------------------------------------------------------------------


class TestDegeneracyValidation:
    def test_state_degeneracy_zero_rejected(self):
        dsl = "state: A, degeneracy=0\nA -> B; k=1"
        with pytest.raises(DSLError):
            parse_dsl(dsl)

    def test_state_degeneracy_negative_rejected(self):
        dsl = "state: A, degeneracy=-1\nA -> B; k=1"
        with pytest.raises(DSLError):
            parse_dsl(dsl)

    def test_state_degeneracy_positive_accepted(self):
        dsl = "state: A, degeneracy=2\nA -> B; k=1"
        result = parse_dsl(dsl)
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Overflow guard for energy-to-rate math
# ---------------------------------------------------------------------------


class TestOverflowGuard:
    def test_overflow_dG_eq_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nequilibrium: A <=> B; dG_eq=-1000000")

    def test_overflow_dG_act_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nreaction: A -> B; dG_act=-1000000")

    def test_overflow_Ea_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nreaction: A -> B; A=1e10; Ea=-1000000")


# ---------------------------------------------------------------------------
# Per-step kappa: only validated when used in Eyring rate computation
# ---------------------------------------------------------------------------


class TestPerStepKappaPlacement:
    def test_explicit_rate_with_kappa_zero_accepted(self):
        result = parse_dsl("reaction: A -> B; k=1; κ=0")
        assert result.ir is not None

    def test_arrhenius_with_kappa_zero_accepted(self):
        result = parse_dsl("reaction: A -> B; A=1e10; Ea=50; κ=0")
        assert result.ir is not None

    def test_eyring_kappa_zero_still_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; dG_act=50; κ=0")

    def test_eyring_kappa_negative_still_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("reaction: A -> B; dG_act=50; κ=-1")


# ---------------------------------------------------------------------------
# K underflow to zero from large positive dG_eq
# ---------------------------------------------------------------------------


class TestKUnderflow:
    def test_K_underflow_from_large_positive_dG_eq_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nreaction: A <-> B; dG_act=50; dG_eq=1000000")

    def test_K_underflow_equilibrium_path_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nequilibrium: A <=> B; dG_eq=1000000")


# ---------------------------------------------------------------------------
# K validated whenever present, not just when used as divisor
# ---------------------------------------------------------------------------


class TestKAlwaysValidated:
    def test_K_zero_with_explicit_kr_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nreaction: A <-> B; dG_act=10; kr=1; K=0")

    def test_K_negative_with_explicit_kr_rejected(self):
        with pytest.raises(DSLError):
            parse_dsl("energy=kJ/mol\nreaction: A <-> B; dG_act=10; kr=1; K=-1")


# ---------------------------------------------------------------------------
# Edge KeyError / ValueError must be wrapped as DSLError
# ---------------------------------------------------------------------------


class TestEdgeErrorWrapping:
    def test_edge_unknown_state_raises_dsl_error(self):
        """edge: A,B with only state A defined must raise DSLError, not KeyError."""
        dsl = "state: A, kind=GS, energy=0\nedge: A,B"
        with pytest.raises(DSLError):
            parse_dsl(dsl)

    def test_edge_self_loop_raises_dsl_error(self):
        """edge: A,A (self-loop) must raise DSLError, not ValueError."""
        dsl = "state: A, kind=GS, energy=0\nedge: A,A"
        with pytest.raises(DSLError):
            parse_dsl(dsl)


# ---------------------------------------------------------------------------
# Stoichiometry coefficient overflow must raise DSLError
# ---------------------------------------------------------------------------


class TestStoichiometryCoefficientOverflow:
    def test_huge_coefficient_raises_dsl_error(self):
        """Extremely long digit string as coefficient must raise DSLError, not produce inf."""
        huge = "9" * 400
        with pytest.raises(DSLError):
            parse_dsl(f"{huge}A -> B ; k=1")


# ---------------------------------------------------------------------------
# Directive errors must include line_number
# ---------------------------------------------------------------------------


class TestDirectiveErrorLineContext:
    def test_unrecognized_line_has_line_number(self):
        dsl = "A -> B; k=1\ngarbage_line_here"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2

    def test_temperature_directive_error_has_line_number(self):
        dsl = "A -> B; k=1\nT=abc"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2

    def test_energy_directive_error_has_line_number(self):
        dsl = "A -> B; k=1\nenergy=xyz"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2

    def test_c0_directive_error_has_line_number(self):
        dsl = "A -> B; k=1\nC0=abc"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2

    def test_kappa_directive_error_has_line_number(self):
        dsl = "A -> B; k=1\nκ=abc"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 2


# ---------------------------------------------------------------------------
# Keq / K_eq aliases for equilibrium constant K
# ---------------------------------------------------------------------------


class TestKeqAliases:
    def test_duplicate_identical_keq_alias_on_equilibrium_rejected_with_line_context(self):
        dsl = "equilibrium: A <-> B ; kf=1 ; Keq=3 ; Keq=5"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "Keq" in str(exc_info.value)

    def test_duplicate_identical_k_on_equilibrium_rejected_with_line_context(self):
        dsl = "equilibrium: A <-> B ; kf=1 ; K=3 ; K=5"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "K" in str(exc_info.value)

    def test_duplicate_identical_k_eq_alias_on_equilibrium_rejected_with_line_context(self):
        dsl = "equilibrium: A <-> B ; kf=1 ; K_eq=3 ; K_eq=5"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "K_eq" in str(exc_info.value)

    def test_duplicate_identical_k_on_reaction_rejected_with_line_context(self):
        dsl = "reaction: A -> B ; k=3 ; k=5"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "k" in str(exc_info.value)

    def test_duplicate_identical_kf_on_reversible_reaction_rejected_with_line_context(self):
        dsl = "reaction: A <-> B ; kf=3 ; kf=5 ; K=2"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "kf" in str(exc_info.value)

    def test_duplicate_keq_and_k_on_equilibrium_rejected_with_line_context(self):
        dsl = "equilibrium: A <-> B ; kf=6 ; Keq=3 ; K=5"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "Keq" in str(exc_info.value)
        assert "K" in str(exc_info.value)

    def test_duplicate_keq_and_k_eq_on_reaction_rejected_with_line_context(self):
        dsl = "reaction: A <-> B ; Keq=3 ; K_eq=5 ; kf=1"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "Keq" in str(exc_info.value)
        assert "K_eq" in str(exc_info.value)

    def test_duplicate_k_and_keq_on_bare_equilibrium_rejected_with_line_context(self):
        dsl = "A <-> B ; K=3 ; keq=5 ; kf=1"
        with pytest.raises(DSLError) as exc_info:
            parse_dsl(dsl)
        assert exc_info.value.line_number == 1
        assert "K" in str(exc_info.value)
        assert "keq" in str(exc_info.value)

    def test_keq_alias_on_reaction(self):
        ref = parse_dsl("A <-> B ; kf=10 ; K=2.0")
        result = parse_dsl("A <-> B ; kf=10 ; Keq=2.0")
        assert result.ir is not None
        assert abs(result.ir.steps[0].kr - ref.ir.steps[0].kr) < 1e-12

    def test_k_eq_alias_on_reaction(self):
        ref = parse_dsl("A <-> B ; kf=10 ; K=2.0")
        result = parse_dsl("A <-> B ; kf=10 ; K_eq=2.0")
        assert result.ir is not None
        assert abs(result.ir.steps[0].kr - ref.ir.steps[0].kr) < 1e-12

    def test_keq_alias_on_equilibrium(self):
        ref = parse_dsl("equilibrium: A <-> B ; K=2.0 ; kf=10")
        result = parse_dsl("equilibrium: A <-> B ; Keq=2.0 ; kf=10")
        assert result.ir is not None
        assert abs(result.ir.steps[0].kr - ref.ir.steps[0].kr) < 1e-12

    def test_k_eq_alias_on_equilibrium(self):
        ref = parse_dsl("equilibrium: A <-> B ; K=2.0 ; kf=10")
        result = parse_dsl("equilibrium: A <-> B ; K_eq=2.0 ; kf=10")
        assert result.ir is not None
        assert abs(result.ir.steps[0].kr - ref.ir.steps[0].kr) < 1e-12

    def test_single_occurrence_duplicate_guard_sanity_cases_still_parse(self):
        cases = (
            "equilibrium: A <-> B ; kf=10 ; Keq=2.0",
            "equilibrium: A <-> B ; kf=10 ; K=2.0",
            "equilibrium: A <-> B ; kf=10 ; K_eq=2.0",
            "reaction: A -> B ; k=3",
            "reaction: A <-> B ; kf=3 ; K=2",
        )
        for dsl in cases:
            result = parse_dsl(dsl)
            assert result.ir is not None
            assert result.ir.steps

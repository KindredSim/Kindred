from __future__ import annotations

import kindred.core.simulator.dsl_text_update as dsl_text_update
import pytest

from kindred.core.simulator.dsl_text_update import (
    analyze_parameter_updates_to_dsl_text,
    analyze_step_parameter_update,
    apply_parameter_updates_to_dsl_text,
    authoritative_parameter_change_name_aware,
    authoritative_parameter_values_match,
    format_authoritative_parameter_value,
)


def test_apply_parameter_updates_to_dsl_text_reports_canonical_updater_errors():
    source = "reaction: A -> B; k=1.0"

    def _boom(_name: str, _value: float, _text: str) -> str:
        raise RuntimeError("metadata drift")

    updated_text, missing, update_errors = apply_parameter_updates_to_dsl_text(
        source,
        {"k1": 2.0},
        canonical_updater=_boom,
    )

    assert updated_text == source
    assert missing == []
    assert update_errors == [
        {
            "name": "k1",
            "exc_type": "RuntimeError",
            "message": "metadata drift",
        }
    ]


def test_apply_parameter_updates_to_dsl_text_treats_lookuperror_as_missing():
    source = "reaction: A -> B; k=1.0"

    def _missing(_name: str, _value: float, _text: str) -> str:
        raise LookupError("step not found")

    updated_text, missing, update_errors = apply_parameter_updates_to_dsl_text(
        source,
        {"k1": 2.0},
        canonical_updater=_missing,
    )

    assert updated_text == source
    assert missing == ["k1"]
    assert update_errors == []


def test_apply_parameter_updates_to_dsl_text_uses_authoritative_parameter_precision():
    source = "alpha = 1.0\nreaction: A -> B; k=1.0\n"

    updated_text, missing, update_errors = apply_parameter_updates_to_dsl_text(
        source,
        {"alpha": 1000000.1234567},
    )

    assert updated_text == "alpha = 1000000.1234567\nreaction: A -> B; k=1.0\n"
    assert missing == []
    assert update_errors == []


def test_authoritative_parameter_values_match_tracks_committed_precision_boundary():
    assert authoritative_parameter_values_match(0.2, 0.20000000000000004) is True
    assert authoritative_parameter_values_match(1000000.1234567, 1000000.1234568) is False
    assert format_authoritative_parameter_value(1000000.1234567) == "1000000.1234567"


def test_authoritative_parameter_values_match_treats_signed_zero_as_equal():
    assert authoritative_parameter_values_match(0.0, -0.0) is True
    assert authoritative_parameter_values_match(-0.0, 0.0) is True


def test_authoritative_parameter_formatting_normalizes_signed_zero():
    updated_text, missing, update_errors = apply_parameter_updates_to_dsl_text(
        "alpha = 1.0\nreaction: A -> B; k=1.0\n",
        {"alpha": -0.0},
    )

    assert format_authoritative_parameter_value(-0.0) == "0"
    assert updated_text == "alpha = 0\nreaction: A -> B; k=1.0\n"
    assert missing == []
    assert update_errors == []


def test_authoritative_parameter_change_name_aware_keeps_scalar_signed_zero_as_noop():
    assert authoritative_parameter_change_name_aware(
        "alpha",
        0.0,
        -0.0,
        source_text="alpha = 0\nreaction: A -> B; k=1.0\n",
    ) is False


def test_authoritative_parameter_change_name_aware_respects_step_writer_floor_for_signed_zero():
    def _flooring_updater(name: str, value: float, text: str) -> str:
        effective = 1e-12 if abs(float(value)) < 1e-12 else abs(float(value))
        return str(text).replace("K=0", f"{name[:-1]}={effective:.15g}", 1)

    assert authoritative_parameter_change_name_aware(
        "Keq1",
        0.0,
        -0.0,
        source_text="equilibrium: A <-> B ; kf=1, K=0",
        canonical_updater=_flooring_updater,
    ) is True


def test_analyze_step_parameter_update_reports_missing_step_target():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=1, kr=2",
        "kf2",
        3.5,
        authoritative_current_value=None,
    )

    assert outcome.parameter_name == "kf2"
    assert outcome.found_target is False
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value is None
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kf=1, kr=2"
    assert outcome.warning_reason == "missing_target"


def test_analyze_step_parameter_update_reports_derived_equilibrium_rate_as_unwritable():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kr=2, K=3",
        "kf1",
        9.0,
        authoritative_current_value=6.0,
    )

    assert outcome.parameter_name == "kf1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 6.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kr=2, K=3"
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_uses_current_text_owner_when_metadata_is_stale():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kr=2, K=3",
        "Keq1",
        6.0,
        authoritative_current_value=3.0,
        step_metadata={
            "kf1": {"editable": True},
            "kr1": {"editable": False, "derived": True},
        },
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 6.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.warning_reason is None
    assert "Keq=6" in outcome.updated_text
    assert "kr=2" in outcome.updated_text
    assert "kf=" not in outcome.updated_text


def test_analyze_step_parameter_update_ignores_stale_constraint_metadata_when_current_text_is_editable():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=6, K=3",
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
        step_metadata={
            "Keq1": {
                "editable": False,
                "derived": True,
                "constraint_reason": "algebra",
            }
        },
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 8.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kf=6, Keq=8"
    assert outcome.warning_reason is None


def test_analyze_step_parameter_update_blocks_explicit_k_target_for_current_text_constraint():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = 4",
        "Keq1",
        8.0,
        authoritative_current_value=4.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 4.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = 4"
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_blocks_plain_k_target_for_current_text_constraint():
    outcome = analyze_step_parameter_update(
        "reaction: A -> B ; k=3\n\n# Algebra\nparam k1 = 4",
        "k1",
        8.0,
        authoritative_current_value=4.0,
    )

    assert outcome.parameter_name == "k1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 4.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "reaction: A -> B ; k=3\n\n# Algebra\nparam k1 = 4"
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_blocks_explicit_k_target_for_current_text_scalar_backed_constraint():
    outcome = analyze_step_parameter_update(
        "alpha = 2\nequilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = alpha",
        "Keq1",
        8.0,
        authoritative_current_value=2.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 2.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "alpha = 2\nequilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = alpha"
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_blocks_K_edit_when_explicit_kr_token_would_be_rewritten_and_is_constrained():
    source_text = "equilibrium: A <-> B ; kf=6, kr=2, K=3\n\n# Algebra\nparam kr1 = 2"

    outcome = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 3.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == source_text
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_blocks_kf_edit_when_explicit_kr_token_would_be_rewritten_and_is_constrained():
    source_text = "equilibrium: A <-> B ; kf=6, kr=2, K=3\n\n# Algebra\nparam kr1 = 2"

    outcome = analyze_step_parameter_update(
        source_text,
        "kf1",
        9.0,
        authoritative_current_value=6.0,
    )

    assert outcome.parameter_name == "kf1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 6.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == source_text
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_blocks_kr_edit_when_explicit_K_token_would_be_rewritten_and_is_constrained():
    source_text = "equilibrium: A <-> B ; kf=6, kr=2, K=3\n\n# Algebra\nparam Keq1 = 3"

    outcome = analyze_step_parameter_update(
        source_text,
        "kr1",
        4.0,
        authoritative_current_value=2.0,
    )

    assert outcome.parameter_name == "kr1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 2.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == source_text
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_allows_kf_edit_when_constrained_explicit_K_is_preserved():
    source_text = "equilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam Keq1 = 5"

    outcome = analyze_step_parameter_update(
        source_text,
        "kf1",
        9.0,
        authoritative_current_value=6.0,
    )

    assert outcome.parameter_name == "kf1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 9.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kf=9, Keq=3\n\n# Algebra\nparam Keq1 = 5"
    assert outcome.warning_reason is None


def test_analyze_step_parameter_update_blocks_K_edit_when_constrained_explicit_kf_semantics_change_after_reload():
    source_text = "equilibrium: A <-> B ; kr=2, K=3\n\n# Algebra\nparam kf1 = 6"

    outcome = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 3.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == source_text
    assert outcome.warning_reason == "target_unwritable"


def test_analyze_step_parameter_update_allows_K_edit_when_constrained_explicit_kf_is_semantically_preserved():
    source_text = "equilibrium: A <-> B ; kf=6, K=3\n\n# Algebra\nparam kf1 = 6"

    outcome = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 8.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "equilibrium: A <-> B ; kf=6, Keq=8\n\n# Algebra\nparam kf1 = 6"
    assert outcome.warning_reason is None


def test_analyze_step_parameter_update_allows_same_step_constrained_K_edit_when_other_step_algebra_fails():
    source_text = "\n".join(
        [
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param kf1 = 6",
            "param Keq2 = sin",
        ]
    )

    outcome = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 8.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "\n".join(
        [
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param kf1 = 6",
            "param Keq2 = sin",
        ]
    )
    assert outcome.warning_reason is None


def test_analyze_step_parameter_update_allows_same_step_constrained_K_edit_when_other_step_uses_nonfinite_scalar():
    source_text = "\n".join(
        [
            "a = nan",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param kf1 = 6",
            "param Keq2 = a",
        ]
    )

    outcome = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 8.0
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False
    assert outcome.updated_text == "\n".join(
        [
            "a = nan",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param kf1 = 6",
            "param Keq2 = a",
        ]
    )
    assert outcome.warning_reason is None


def test_build_step_constraint_reasons_from_text_keeps_unrelated_constraint_when_scalar_name_matches_observable():
    from kindred.core.simulator.step_constraint_authority import build_step_constraint_reasons_from_text

    reasons = build_step_constraint_reasons_from_text(
        "\n".join(
            [
                "alpha = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "let alpha = [A]",
                "param Keq1 = 5",
            ]
        )
    )

    assert reasons["Keq1"] == "algebra"


def test_analyze_step_parameter_update_keeps_explicit_k_block_when_scalar_name_matches_observable():
    outcome = analyze_step_parameter_update(
        "\n".join(
            [
                "alpha = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "let alpha = [A]",
                "param Keq1 = 5",
            ]
        ),
        "Keq1",
        8.0,
        authoritative_current_value=5.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 5.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.warning_reason == "target_unwritable"


def test_build_step_constraint_reasons_from_text_keeps_unrelated_constraint_when_unused_builtin_shadow_scalar_input_present():
    from kindred.core.simulator.step_constraint_authority import build_step_constraint_reasons_from_text

    reasons = build_step_constraint_reasons_from_text(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = 5",
            ]
        )
    )

    assert reasons["Keq1"] == "algebra"


def test_analyze_step_parameter_update_keeps_explicit_k_block_when_unused_builtin_shadow_scalar_input_present():
    outcome = analyze_step_parameter_update(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = 5",
            ]
        ),
        "Keq1",
        8.0,
        authoritative_current_value=5.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 5.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.warning_reason == "target_unwritable"


def test_build_current_text_step_analysis_context_records_constraint_analysis_failure():
    context = dsl_text_update.build_current_text_step_analysis_context(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = sin",
            ]
        )
    )

    assert context.step_constraint_reasons == {}
    assert context.constraint_analysis_error is None
    assert context.step_constraint_analysis.step_analysis_errors[1].stage == "build_parameter_algebra_evaluation_model"
    assert "sin" in context.step_constraint_analysis.step_analysis_errors[1].message


def test_analyze_step_parameter_update_blocks_when_current_text_constraint_analysis_fails():
    outcome = analyze_step_parameter_update(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "",
                "# Algebra",
                "param Keq1 = sin",
            ]
        ),
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )

    assert outcome.parameter_name == "Keq1"
    assert outcome.found_target is True
    assert outcome.writable is False
    assert outcome.effective_authoritative_written_value == 3.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is False
    assert outcome.canonicalization_only_change is False
    assert outcome.warning_reason == "constraint_analysis_failed"


def test_build_current_text_step_analysis_context_scopes_constraint_analysis_failure_to_affected_step():
    context = dsl_text_update.build_current_text_step_analysis_context(
        "\n".join(
            [
                "sin = 2",
                "equilibrium: A <-> B ; kf=6, K=3",
                "equilibrium: B <-> C ; kf=4, K=5",
                "",
                "# Algebra",
                "param Keq2 = sin",
            ]
        )
    )

    assert context.constraint_analysis_error is None
    assert context.step_constraint_reasons == {}
    assert 1 not in context.step_constraint_analysis.step_analysis_errors
    assert context.step_constraint_analysis.step_analysis_errors[2].stage == "build_parameter_algebra_evaluation_model"
    assert "sin" in context.step_constraint_analysis.step_analysis_errors[2].message


def test_analyze_step_parameter_update_scopes_current_text_constraint_analysis_failure_per_step():
    source_text = "\n".join(
        [
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = sin",
        ]
    )

    unaffected = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )
    affected = analyze_step_parameter_update(
        source_text,
        "kf2",
        9.0,
        authoritative_current_value=4.0,
    )

    assert unaffected.parameter_name == "Keq1"
    assert unaffected.found_target is True
    assert unaffected.writable is True
    assert unaffected.warning_reason is None
    assert unaffected.updated_text == "\n".join(
        [
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = sin",
        ]
    )
    assert affected.parameter_name == "kf2"
    assert affected.found_target is True
    assert affected.writable is False
    assert affected.warning_reason == "constraint_analysis_failed"


def test_build_current_text_step_analysis_context_scopes_nonfinite_scalar_failure_to_affected_step():
    context = dsl_text_update.build_current_text_step_analysis_context(
        "\n".join(
            [
                "a = nan",
                "equilibrium: A <-> B ; kf=6, K=3",
                "equilibrium: B <-> C ; kf=4, K=5",
                "",
                "# Algebra",
                "param Keq2 = a",
            ]
        )
    )

    assert context.constraint_analysis_error is None
    assert context.step_constraint_reasons == {}
    assert 1 not in context.step_constraint_analysis.step_analysis_errors
    assert context.step_constraint_analysis.step_analysis_errors[2].stage == "build_parameter_algebra_evaluation_model"
    assert "a" in context.step_constraint_analysis.step_analysis_errors[2].message
    assert "non-finite" in context.step_constraint_analysis.step_analysis_errors[2].message.lower()


def test_analyze_step_parameter_update_scopes_nonfinite_current_text_scalar_failure_per_step():
    source_text = "\n".join(
        [
            "a = nan",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = a",
        ]
    )

    unaffected = analyze_step_parameter_update(
        source_text,
        "Keq1",
        8.0,
        authoritative_current_value=3.0,
    )
    affected = analyze_step_parameter_update(
        source_text,
        "kf2",
        9.0,
        authoritative_current_value=4.0,
    )

    assert unaffected.parameter_name == "Keq1"
    assert unaffected.found_target is True
    assert unaffected.writable is True
    assert unaffected.warning_reason is None
    assert unaffected.updated_text == "\n".join(
        [
            "a = nan",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = a",
        ]
    )
    assert affected.parameter_name == "kf2"
    assert affected.found_target is True
    assert affected.writable is False
    assert affected.warning_reason == "constraint_analysis_failed"


def test_analyze_step_parameter_update_distinguishes_canonicalization_only_step_rewrite():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; k=1, kr=2",
        "kf1",
        1.0,
        authoritative_current_value=1.0,
        step_constraint_context={"wegscheider_cyclicity_enabled": False},
    )

    assert outcome.parameter_name == "kf1"
    assert outcome.found_target is True
    assert outcome.writable is True
    assert outcome.effective_authoritative_written_value == 1.0
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is True
    assert outcome.updated_text == "equilibrium: A <-> B ; kr=2, kf=1"
    assert outcome.warning_reason is None


def test_analyze_step_parameter_update_rewrites_keq_target_to_canonical_k_token():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=6 ; Keq=3",
        "Keq1",
        5.0,
        authoritative_current_value=3.0,
    )

    assert outcome.updated_text == "equilibrium: A <-> B ; kf=6, Keq=5"
    assert outcome.semantic_value_change is True
    assert outcome.canonicalization_only_change is False


def test_analyze_step_parameter_update_canonicalizes_keq_spelling_when_kf_is_rewritten():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=6 ; Keq=3",
        "kf1",
        10.0,
        authoritative_current_value=6.0,
    )

    assert outcome.updated_text == "equilibrium: A <-> B ; kf=10, Keq=3"
    assert outcome.semantic_value_change is True
    assert outcome.canonicalization_only_change is False


def test_analyze_step_parameter_update_marks_keq_spelling_only_rewrite_as_canonicalization_only():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; kf=1 ; Keq=3",
        "Keq1",
        3.0,
        authoritative_current_value=3.0,
    )

    assert outcome.updated_text == "equilibrium: A <-> B ; kf=1, Keq=3"
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is True


def test_analyze_step_parameter_update_marks_k_eq_spelling_only_rewrite_as_canonicalization_only():
    outcome = analyze_step_parameter_update(
        "equilibrium: A <-> B ; K_eq=3 ; kf=1",
        "Keq1",
        3.0,
        authoritative_current_value=3.0,
    )

    assert outcome.updated_text == "equilibrium: A <-> B ; Keq=3, kf=1"
    assert outcome.semantic_value_change is False
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is True


def test_analyze_step_parameter_update_rejects_identical_duplicate_keq_aliases():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "equilibrium: A <-> B ; kf=1 ; Keq=3 ; Keq=5",
            "kf1",
            2.0,
            authoritative_current_value=1.0,
        )


def test_analyze_step_parameter_update_rejects_case_only_duplicate_keq_aliases():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "equilibrium: A <-> B ; kf=1 ; Keq=3 ; keq=5",
            "Keq1",
            4.0,
            authoritative_current_value=3.0,
        )


def test_analyze_step_parameter_update_rejects_duplicate_reaction_k_tokens():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "reaction: A -> B ; k=3 ; k=5",
            "k1",
            4.0,
            authoritative_current_value=5.0,
        )


def test_analyze_step_parameter_update_rejects_duplicate_reversible_kf_tokens():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "reaction: A <-> B ; kf=3 ; kf=5 ; K=2",
            "kf1",
            6.0,
            authoritative_current_value=5.0,
        )


def test_analyze_step_parameter_update_rejects_duplicate_equilibrium_dg_eq_aliases():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "equilibrium: A <-> B ; kf=1 ; dG_eq=-1 ; DG_EQ=-2",
            "kf1",
            2.0,
            authoritative_current_value=1.0,
        )


def test_analyze_step_parameter_update_rejects_duplicate_reaction_ea_aliases():
    with pytest.raises(ValueError, match="Duplicate parameter"):
        analyze_step_parameter_update(
            "reaction: A -> B ; k=1 ; Ea=5 ; ea=6",
            "k1",
            2.0,
            authoritative_current_value=1.0,
        )


def test_analyze_parameter_updates_to_dsl_text_reports_step_floor_as_real_semantic_change():
    analysis = analyze_parameter_updates_to_dsl_text(
        "equilibrium: A <-> B ; kf=1, K=0",
        {"Keq1": -0.0},
        authoritative_values={"Keq1": 0.0},
        step_constraint_context={"wegscheider_cyclicity_enabled": False},
    )

    assert analysis.updated_text == "equilibrium: A <-> B ; kf=1, Keq=1e-12"
    assert analysis.missing == ()
    assert analysis.update_errors == ()
    assert len(analysis.step_outcomes) == 1
    outcome = analysis.step_outcomes[0]
    assert outcome.parameter_name == "Keq1"
    assert outcome.effective_authoritative_written_value == 1e-12
    assert outcome.semantic_value_change is True
    assert outcome.would_change_text is True
    assert outcome.canonicalization_only_change is False


def test_analyze_parameter_updates_to_dsl_text_reports_nonfinite_scalar_input_as_update_error():
    analysis = analyze_parameter_updates_to_dsl_text(
        "alpha = 1.0\nreaction: A -> B; k=0.2",
        {"alpha": float("nan")},
        authoritative_values={"alpha": 1.0},
    )

    assert analysis.updated_text == "alpha = 1.0\nreaction: A -> B; k=0.2"
    assert analysis.missing == ()
    assert analysis.update_errors == (
        {
            "name": "alpha",
            "exc_type": "ValueError",
            "message": "Fitted value is non-finite.",
        },
    )
    assert analysis.step_outcomes == ()


def test_analyze_parameter_updates_to_dsl_text_reuses_shared_step_analysis_context_for_unchanged_source_text(
    monkeypatch,
):
    build_context = dsl_text_update.build_current_text_step_analysis_context
    calls: list[str] = []

    def _counted_build_context(source_text: str, *, step_constraint_context=None):
        calls.append(str(source_text))
        return build_context(source_text, step_constraint_context=step_constraint_context)

    monkeypatch.setattr(dsl_text_update, "build_current_text_step_analysis_context", _counted_build_context)

    analysis = analyze_parameter_updates_to_dsl_text(
        "equilibrium: A <-> B ; kr=2, K=3",
        {"kf1": 9.0, "Keq1": 6.0},
        authoritative_values={"kf1": 6.0, "Keq1": 3.0},
    )

    assert calls == ["equilibrium: A <-> B ; kr=2, K=3"]
    assert analysis.updated_text == "equilibrium: A <-> B ; kr=2, Keq=6"
    assert analysis.missing == ()
    assert analysis.update_errors == ()
    assert len(analysis.step_outcomes) == 2
    assert analysis.step_outcomes[0].parameter_name == "kf1"
    assert analysis.step_outcomes[0].writable is False
    assert analysis.step_outcomes[0].updated_text == "equilibrium: A <-> B ; kr=2, K=3"
    assert analysis.step_outcomes[1].parameter_name == "Keq1"
    assert analysis.step_outcomes[1].writable is True


def test_analyze_parameter_updates_to_dsl_text_best_effort_applies_unrelated_step_and_scalar_when_other_step_analysis_fails():
    source_text = "\n".join(
        [
            "alpha = 1.0",
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = sin",
        ]
    )

    analysis = analyze_parameter_updates_to_dsl_text(
        source_text,
        {"alpha": 2.0, "Keq1": 8.0, "Keq2": 9.0},
        authoritative_values={"alpha": 1.0, "Keq1": 3.0, "Keq2": 5.0},
    )

    assert analysis.updated_text == "\n".join(
        [
            "alpha = 2",
            "sin = 2",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = sin",
        ]
    )
    assert analysis.missing == ()
    assert analysis.update_errors == ()
    assert [outcome.parameter_name for outcome in analysis.step_outcomes] == ["Keq1", "Keq2"]
    assert analysis.step_outcomes[0].writable is True
    assert analysis.step_outcomes[0].warning_reason is None
    assert analysis.step_outcomes[1].writable is False
    assert analysis.step_outcomes[1].warning_reason == "constraint_analysis_failed"


def test_analyze_parameter_updates_to_dsl_text_best_effort_applies_unrelated_step_and_scalar_when_other_step_uses_nonfinite_scalar_input():
    source_text = "\n".join(
        [
            "alpha = 1.0",
            "a = nan",
            "equilibrium: A <-> B ; kf=6, K=3",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = a",
        ]
    )

    analysis = analyze_parameter_updates_to_dsl_text(
        source_text,
        {"alpha": 2.0, "Keq1": 8.0, "Keq2": 9.0},
        authoritative_values={"alpha": 1.0, "Keq1": 3.0, "Keq2": 5.0},
    )

    assert analysis.updated_text == "\n".join(
        [
            "alpha = 2",
            "a = nan",
            "equilibrium: A <-> B ; kf=6, Keq=8",
            "equilibrium: B <-> C ; kf=4, K=5",
            "",
            "# Algebra",
            "param Keq2 = a",
        ]
    )
    assert analysis.missing == ()
    assert analysis.update_errors == ()
    assert [outcome.parameter_name for outcome in analysis.step_outcomes] == ["Keq1", "Keq2"]
    assert analysis.step_outcomes[0].writable is True
    assert analysis.step_outcomes[0].warning_reason is None
    assert analysis.step_outcomes[1].writable is False
    assert analysis.step_outcomes[1].warning_reason == "constraint_analysis_failed"

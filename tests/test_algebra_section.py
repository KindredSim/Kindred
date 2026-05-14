from __future__ import annotations

from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text
from kindred.core.simulator.algebra_section import (
    extract_algebra_section_text,
    upsert_lines_into_algebra_section,
)
from kindred.core.simulator.parameter_algebra_spec import classify_parameter_algebra_declaration
from kindred.core.simulator.step_constraint_authority import build_step_constraint_reasons_from_text
import pytest

pytestmark = pytest.mark.unit



def test_extract_algebra_section_text_collects_interleaved_algebra_lines_only():
    text = "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "let total = [A] + [B]",
            "B -> C ; k=2.0",
            "let conversion = 1 - [A]/[A]_0",
            "param scale = 2.0",
        ]
    )

    assert extract_algebra_section_text(text) == "\n".join(
        [
            "let total = [A] + [B]",
            "let conversion = 1 - [A]/[A]_0",
            "param scale = 2.0",
        ]
    )


def test_extract_algebra_section_text_ignores_header_comment_and_matches_no_header_text():
    with_header = "\n".join(
        [
            "# Algebra",
            "let total = [A] + [B]",
            "reaction: A -> B ; k=1.0",
            "param scale = 2.0",
        ]
    )
    without_header = "\n".join(
        [
            "let total = [A] + [B]",
            "reaction: A -> B ; k=1.0",
            "param scale = 2.0",
        ]
    )

    assert extract_algebra_section_text(with_header) == extract_algebra_section_text(without_header)


def test_extract_algebra_section_text_ignores_header_prefix_comment():
    text = "\n".join(
        [
            "# Algebraic observables",
            "reaction: A -> B ; k=1.0",
            "let total = [A] + [B]",
        ]
    )

    assert extract_algebra_section_text(text) == "let total = [A] + [B]"


def test_upsert_lines_into_algebra_section_appends_without_creating_header():
    text = "reaction: A -> B ; k=1.0\n"

    updated = upsert_lines_into_algebra_section(text, ["param scale = 2.0"])

    assert "# Algebra" not in updated
    assert updated == "reaction: A -> B ; k=1.0\nparam scale = 2.0\n"


def test_upsert_lines_into_algebra_section_inserts_after_last_existing_algebra_line():
    text = "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "param scale = 2.0",
            "reaction: B -> C ; k=3.0",
            "let total = [A] + [B] + [C]",
            "reaction: C -> D ; k=4.0",
        ]
    )

    updated = upsert_lines_into_algebra_section(text, ["param bias = 0.5"])

    assert updated == "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "param scale = 2.0",
            "reaction: B -> C ; k=3.0",
            "let total = [A] + [B] + [C]",
            "param bias = 0.5",
            "reaction: C -> D ; k=4.0",
            "",
        ]
    )


def test_interleaved_observable_discovery_reads_extracted_algebra_text():
    text = "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "let signal = [A] + [B]",
            "B -> C ; k=2.0",
        ]
    )

    observables = extract_observables_from_algebra_text(extract_algebra_section_text(text))

    assert observables["signal"] == "[A] + [B]"


def test_extract_algebra_section_text_preserves_unsupported_bare_assignment_for_policy_rejection():
    text = "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "signal = [A] + [B]",
            "param scale = 2.0",
        ]
    )

    extracted = extract_algebra_section_text(text)

    assert extracted.splitlines() == [
        "signal = [A] + [B]",
        "param scale = 2.0",
    ]
    with pytest.raises(ValueError, match="Bare algebra assignment"):
        extract_observables_from_algebra_text(extracted)


def test_step_constraint_reasons_detect_interleaved_constraint_without_header():
    text = "\n".join(
        [
            "reaction: A -> B ; k=1.0",
            "param k1 = 2.0",
            "reaction: B -> C ; k=3.0",
        ]
    )

    reasons = build_step_constraint_reasons_from_text(text)

    assert reasons["k1"] == "algebra"


def test_unsupported_bare_assignment_classifier_rejects_prefixed_dsl_assignments():
    assert classify_parameter_algebra_declaration("reaction = nope").kind == "non_algebra"
    assert classify_parameter_algebra_declaration("equilibrium = nope").kind == "non_algebra"
    assert classify_parameter_algebra_declaration("state = nope").kind == "non_algebra"


def test_unsupported_bare_assignment_classifier_rejects_brace_block_rhs():
    assert classify_parameter_algebra_declaration("config = {").kind == "non_algebra"
    assert classify_parameter_algebra_declaration("result = { [A] = 1.0 }").kind == "non_algebra"


def test_unsupported_bare_assignment_classifier_detects_non_brace_rhs():
    assert classify_parameter_algebra_declaration("config = 1.0").kind == "unsupported_bare_assignment"
    assert classify_parameter_algebra_declaration("conversion = 1 - [A]/[A]_0").kind == "unsupported_bare_assignment"

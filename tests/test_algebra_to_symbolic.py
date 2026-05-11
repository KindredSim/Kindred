from __future__ import annotations

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.parameter_algebra_spec import (
    ParameterAssignment,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism


pytestmark = pytest.mark.unit


BASE_DSL = "\n".join(
    [
        "equilibrium: A <-> B ; kf=1 ; K=2",
        "equilibrium: B <-> C ; kf=1 ; K=3",
        "equilibrium: C <-> A ; kf=1 ; K=4",
        "param scale = 2",
        "param Keq3 = 1 / (Keq1 * Keq2)",
        "init: A=1, B=0, C=0",
    ]
)


def _spec(text: str = BASE_DSL):
    mechanism = parse_dsl_to_mechanism(text, initials={})
    return parse_parameter_algebra_spec_from_dsl_text(
        text,
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )


def _assignment(expr_src: str, *, name: str = "Keq3") -> ParameterAssignment:
    return ParameterAssignment(
        name=name,
        expr_src=expr_src,
        line_number=1,
        line_content=f"param {name} = {expr_src}",
    )


def test_supported_arithmetic_translates_to_exact_sympy_expression():
    from kindred.core.symbolic.parameter_expression import translate_parameter_expression

    translated = translate_parameter_expression(
        _assignment("1 / (Keq1 * Keq2)"),
        spec=_spec(),
    )

    assert translated.canonical_identifiers == ("Keq1", "Keq2")
    assert translated.normalized_source == "1/(Keq1*Keq2)"
    assert str(translated.expression) == "1/(Keq1*Keq2)"
    assert translated.fingerprint


def test_scientific_notation_numeric_literals_translate_without_identifier_rewrite():
    from kindred.core.symbolic.parameter_expression import translate_parameter_expression

    translated = translate_parameter_expression(
        _assignment("1e0 / (Keq1 * Keq2)"),
        spec=_spec(),
    )

    assert translated.canonical_identifiers == ("Keq1", "Keq2")
    assert translated.normalized_source == "1/(Keq1*Keq2)"


def test_case_insensitive_aliases_translate_to_canonical_names():
    from kindred.core.symbolic.parameter_expression import translate_parameter_expression

    translated = translate_parameter_expression(
        _assignment("1 / (keq1 * KEQ2)"),
        spec=_spec(),
    )

    assert translated.canonical_identifiers == ("Keq1", "Keq2")
    assert translated.normalized_source == "1/(Keq1*Keq2)"


@pytest.mark.parametrize(
    "expr_src",
    [
        "sqrt(Keq1)",
        "[A]",
        "[A]_0",
        "[A](T0)",
        "T",
        "T0",
        "Keq1 if Keq2 else 1",
        "Keq1 > Keq2",
        "unknown_name",
    ],
)
def test_unsupported_symbolic_expressions_are_rejected(expr_src: str):
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.parameter_expression import translate_parameter_expression

    with pytest.raises(UnsupportedSymbolicExpressionError):
        translate_parameter_expression(_assignment(expr_src), spec=_spec())


def test_exact_proof_accepts_identity_and_rejects_probe_match_expression():
    from kindred.core.symbolic.proof import prove_product_identity

    spec = _spec()
    accepted = prove_product_identity(
        target_factors={"Keq1": 1, "Keq2": 1, "Keq3": 1},
        candidate=_assignment("1 / (Keq1 * Keq2)"),
        spec=spec,
    )
    rejected = prove_product_identity(
        target_factors={"Keq1": 1, "Keq2": 1, "Keq3": 1},
        candidate=_assignment("1 / (Keq1 * Keq2) + (Keq1 - 2) * 0.001"),
        spec=spec,
    )

    assert accepted.proven is True
    assert accepted.fingerprint
    assert rejected.proven is False
    assert rejected.reason == "not_identity"


def test_exact_proof_expands_transitive_parameter_assignments():
    from kindred.core.symbolic.proof import prove_product_identity

    text = "\n".join(
        [
            "equilibrium: A <-> B ; kf=1 ; K=2",
            "equilibrium: B <-> C ; kf=1 ; K=3",
            "equilibrium: C <-> A ; kf=1 ; K=4",
            "param inv = 1e0 / (Keq1 * Keq2)",
            "param Keq3 = inv",
            "init: A=1, B=0, C=0",
        ]
    )
    spec = _spec(text)
    accepted = prove_product_identity(
        target_factors={"Keq1": 1, "Keq2": 1, "Keq3": 1},
        candidate=_assignment("inv", name="Keq3"),
        spec=spec,
    )

    assert accepted.proven is True
    assert accepted.fingerprint


def test_unsupported_transitive_assignment_fingerprint_changes_with_dependency_source():
    from kindred.core.symbolic.proof import prove_product_identity

    first_spec = _spec(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=4",
                "param inv = sqrt(Keq1)",
                "param Keq3 = inv",
                "init: A=1, B=0, C=0",
            ]
        )
    )
    changed_spec = _spec(
        "\n".join(
            [
                "equilibrium: A <-> B ; kf=1 ; K=2",
                "equilibrium: B <-> C ; kf=1 ; K=3",
                "equilibrium: C <-> A ; kf=1 ; K=4",
                "param inv = T",
                "param Keq3 = inv",
                "init: A=1, B=0, C=0",
            ]
        )
    )

    first = prove_product_identity(
        target_factors={"Keq1": 1, "Keq2": 1, "Keq3": 1},
        candidate=_assignment("inv", name="Keq3"),
        spec=first_spec,
    )
    changed = prove_product_identity(
        target_factors={"Keq1": 1, "Keq2": 1, "Keq3": 1},
        candidate=_assignment("inv", name="Keq3"),
        spec=changed_spec,
    )

    assert first.proven is False
    assert changed.proven is False
    assert first.reason == "unsupported"
    assert changed.reason == "unsupported"
    assert first.fingerprint != changed.fingerprint

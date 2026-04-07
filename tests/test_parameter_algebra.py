from dataclasses import replace

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_to_mechanism,
    evaluate_parameter_algebra,
    mechanism_parameter_namespace,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.parameter_namespace import build_flat_compat_namespace
from kindred.core.simulator.parameter_units import rate_constant_unit


def _base_mech(dsl_text: str):
    return parse_dsl_to_mechanism(dsl_text, initials={})


def _compat_namespace(names: set[str]):
    return build_flat_compat_namespace(names)


def test_parameter_algebra_recomputes_on_base_change():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: B -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param k1 = 4*k2",
            "",
        ]
    )
    mech = _base_mech(dsl)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.reactions[0].rate) == pytest.approx(4.0 * float(mech.reactions[1].rate))

    # Update base k2 and re-apply; k1 should recompute.
    mech.reactions[1] = replace(mech.reactions[1], rate=2.0)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.reactions[0].rate) == pytest.approx(8.0)


def test_parameter_algebra_cycle_detection():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: B -> C; k=2.0",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param k1 = k2",
            "param k2 = k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    with pytest.raises(DSLError) as exc:
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    msg = str(exc.value).lower()
    assert "cycle" in msg
    assert "k1" in msg and "k2" in msg


def test_parameter_algebra_rejects_concentration_reference():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: B -> C; k=2.0",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param k1 = [A]",
            "",
        ]
    )
    mech = _base_mech(dsl)
    with pytest.raises(DSLError) as exc:
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert "[a]" in str(exc.value).lower() or "concentration" in str(exc.value).lower()


def test_ambiguous_let_to_parameter_is_error():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: B -> C; k=2.0",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "let k1 = 4*k2",
            "",
        ]
    )
    mech = _base_mech(dsl)
    with pytest.raises(DSLError) as exc:
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    msg = str(exc.value)
    assert "param k1" in msg


def test_scalar_base_param_can_drive_constraints():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: A -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param a = 4",
            "param k2 = a*k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)

    # a is a base solver parameter (stored), k2 is derived.
    scalar_params = (mech.metadata or {}).get("scalar_params") or {}
    assert float(scalar_params["a"]) == pytest.approx(4.0)

    assert float(mech.reactions[1].rate) == pytest.approx(4.0 * float(mech.reactions[0].rate))


def test_changing_scalar_base_param_updates_derived_mechanism_param():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=2.0",
            "reaction: A -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param a = 4",
            "param k2 = a*k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.reactions[1].rate) == pytest.approx(8.0)

    # Simulate a slider/fitter update event by changing stored scalar param.
    mech.metadata.setdefault("scalar_params", {})["a"] = 5.0
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.reactions[1].rate) == pytest.approx(10.0)


def test_param_expression_cannot_reference_let_observable():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: A -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "let a = 4",
            "param k2 = a*k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    with pytest.raises(DSLError) as exc:
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    msg = str(exc.value)
    assert "observable" in msg.lower()
    assert "param a" in msg.lower()


def test_cycle_detection_across_named_params():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: A -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param a = b",
            "param b = a",
            "param k2 = a*k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    with pytest.raises(DSLError) as exc:
        apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert "cycle" in str(exc.value).lower()


def test_solver_param_editability_metadata_for_constraints():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: A -> C; k=0.5",
            "init: A=1.0, B=0.0, C=0.0",
            "",
            "# Algebra",
            "param a = 4",
            "param k2 = a*k1",
            "",
        ]
    )
    mech = _base_mech(dsl)
    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    info = (mech.metadata or {}).get("scalar_param_info") or {}
    assert info["a"]["editable"] is True
    constrained = (mech.metadata or {}).get("constrained_params") or {}
    assert "k2" in constrained


def test_unused_builtin_shadow_scalar_input_does_not_poison_parameter_algebra_evaluation():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param Keq1 = 5",
            ]
        ),
        mechanism_namespace=_compat_namespace({"kf1", "kr1", "Keq1"}),
        scalar_input_names={"sin"},
    )

    derived = evaluate_parameter_algebra(
        spec,
        base_values={"kf1": 6.0, "Keq1": 3.0, "sin": 2.0},
    )

    assert derived["Keq1"] == pytest.approx(5.0)


def test_referenced_builtin_shadow_scalar_input_is_rejected():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param Keq1 = sin",
            ]
        ),
        mechanism_namespace=_compat_namespace({"kf1", "kr1", "Keq1"}),
        scalar_input_names={"sin"},
    )

    with pytest.raises(DSLError, match="sin"):
        evaluate_parameter_algebra(
            spec,
            base_values={"kf1": 6.0, "Keq1": 3.0, "sin": 2.0},
        )


def test_referenced_nonfinite_scalar_input_is_rejected_with_assignment_context():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param Keq2 = a",
            ]
        ),
        mechanism_namespace=_compat_namespace({"kf1", "kr1", "Keq1", "kf2", "kr2", "Keq2"}),
        scalar_input_names={"a"},
    )

    with pytest.raises(DSLError, match="Non-finite") as exc:
        evaluate_parameter_algebra(
            spec,
            base_values={"kf1": 6.0, "Keq1": 3.0, "kf2": 4.0, "Keq2": 5.0, "a": float("nan")},
        )

    assert exc.value.line_number == 2
    assert exc.value.line_content == "param Keq2 = a"


@pytest.mark.parametrize(
    ("line", "mechanism_param_names", "expected_name", "base_values", "expected_value"),
    [
        ("param K1 = 5", {"k1"}, "k1", {"k1": 1.0}, 5.0),
        ("param K1 = 5", {"kf1", "kr1"}, "kf1", {"kf1": 2.0, "kr1": 0.5}, 5.0),
        ("param k1 = 5", {"kf1", "kr1"}, "kf1", {"kf1": 2.0, "kr1": 0.5}, 5.0),
        ("param KF2 = 5", {"kf2", "kr2"}, "kf2", {"kf2": 2.0, "kr2": 0.5}, 5.0),
        ("param KR2 = 5", {"kf2", "kr2"}, "kr2", {"kf2": 2.0, "kr2": 0.5}, 5.0),
        ("param KEQ3 = 5", {"kf3", "kr3", "Keq3"}, "Keq3", {"kf3": 8.0, "Keq3": 4.0}, 5.0),
        ("param keq3 = 5", {"kf3", "kr3", "Keq3"}, "Keq3", {"kf3": 8.0, "Keq3": 4.0}, 5.0),
    ],
)
def test_param_targets_resolve_case_insensitively_against_mechanism_namespace(
    line,
    mechanism_param_names,
    expected_name,
    base_values,
    expected_value,
):
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(["# Algebra", line]),
        mechanism_namespace=_compat_namespace(mechanism_param_names),
    )

    assert [assignment.name for assignment in spec.param_statements] == [expected_name]

    derived = evaluate_parameter_algebra(spec, base_values=dict(base_values))

    assert derived[expected_name] == pytest.approx(expected_value)


def test_param_k_target_rejects_equilibrium_constant_shorthand_with_keq_guidance():
    with pytest.raises(DSLError, match="Keq1") as exc:
        parse_parameter_algebra_spec_from_dsl_text(
            "\n".join(
                [
                    "# Algebra",
                    "param K1 = 5",
                ]
            ),
            mechanism_namespace=_compat_namespace({"kf1", "kr1", "Keq1"}),
        )

    assert exc.value.line_number == 2
    assert exc.value.line_content == "param K1 = 5"
    assert "K1" in str(exc.value)


@pytest.mark.parametrize(
    ("line", "mechanism_param_names"),
    [
        ("let K1 = 5", {"k1"}),
        ("K1 = 5", {"k1"}),
    ],
)
def test_k_like_non_param_assignments_reject_resolved_mechanism_targets(line, mechanism_param_names):
    with pytest.raises(DSLError, match="use 'param K1 ="):
        parse_parameter_algebra_spec_from_dsl_text(
            "\n".join(
                [
                    "# Algebra",
                    line,
                ]
            ),
            mechanism_namespace=_compat_namespace(mechanism_param_names),
        )


def test_let_k_like_name_without_mechanism_match_remains_observable():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "let K1 = 5",
            ]
        ),
        mechanism_namespace=_compat_namespace({"k2"}),
    )

    assert spec.param_statements == []
    assert spec.observable_names == {"K1"}


@pytest.mark.parametrize(
    ("line", "mechanism_param_names", "base_values", "expected"),
    [
        ("param a = 2*K1", {"k1"}, {"k1": 3.0}, 6.0),
        ("param a = K1 + K2", {"k1", "kf2", "kr2"}, {"k1": 3.0, "kf2": 4.0}, 7.0),
    ],
)
def test_rhs_mechanism_identifiers_resolve_case_insensitively(
    line,
    mechanism_param_names,
    base_values,
    expected,
):
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                line,
            ]
        ),
        mechanism_namespace=_compat_namespace(mechanism_param_names),
    )

    derived = evaluate_parameter_algebra(spec, base_values=dict(base_values))

    assert derived["a"] == pytest.approx(expected)


def test_rhs_exact_case_scalar_input_takes_priority_over_mechanism_canonicalization():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param k2 = K1",
            ]
        ),
        mechanism_namespace=_compat_namespace({"k1", "k2"}),
        scalar_input_names={"K1"},
    )

    derived = evaluate_parameter_algebra(
        spec,
        base_values={"k1": 1.0, "k2": 0.0, "K1": 99.0},
    )

    assert derived["k2"] == pytest.approx(99.0)


def test_rhs_mechanism_canonicalization_applies_when_exact_case_scalar_input_is_absent():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param k2 = K1",
            ]
        ),
        mechanism_namespace=_compat_namespace({"k1", "k2"}),
    )

    derived = evaluate_parameter_algebra(
        spec,
        base_values={"k1": 1.0, "k2": 0.0},
    )

    assert derived["k2"] == pytest.approx(1.0)


def test_rhs_k_identifier_rejects_equilibrium_constant_shorthand_with_raw_token_guidance():
    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param a = K1",
            ]
        ),
        mechanism_namespace=_compat_namespace({"kf1", "kr1", "Keq1"}),
    )

    with pytest.raises(DSLError, match="Keq1") as exc:
        evaluate_parameter_algebra(
            spec,
            base_values={"kf1": 6.0, "kr1": 2.0, "Keq1": 3.0},
        )

    assert exc.value.line_number == 2
    assert exc.value.line_content == "param a = K1"
    assert "K1" in str(exc.value)


def test_mechanism_parameter_namespace_requires_authoritative_step_index_map():
    class _MechanismWithoutStepMap:
        metadata = {}

    with pytest.raises(ValueError, match="step_index_map"):
        mechanism_parameter_namespace(_MechanismWithoutStepMap())


def test_rate_constant_unit_formatting():
    assert rate_constant_unit(1) == "1/s"
    assert rate_constant_unit(2) == "1/(M s)"
    assert rate_constant_unit(3) == "1/(M^2 s)"
    assert rate_constant_unit(0) == "M/s"

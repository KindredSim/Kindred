from dataclasses import replace

import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_algebra import (
    apply_parameter_algebra_to_mechanism,
    evaluate_parameter_algebra,
    parse_parameter_algebra_spec_from_dsl_text,
)
from kindred.core.simulator.parameter_units import rate_constant_unit


def _base_mech(dsl_text: str):
    return parse_dsl_to_mechanism(dsl_text, initials={})


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
        mechanism_param_names={"kf1", "kr1", "Keq1"},
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
        mechanism_param_names={"kf1", "kr1", "Keq1"},
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
        mechanism_param_names={"kf1", "kr1", "Keq1", "kf2", "kr2", "Keq2"},
        scalar_input_names={"a"},
    )

    with pytest.raises(DSLError, match="Non-finite") as exc:
        evaluate_parameter_algebra(
            spec,
            base_values={"kf1": 6.0, "Keq1": 3.0, "kf2": 4.0, "Keq2": 5.0, "a": float("nan")},
        )

    assert exc.value.line_number == 2
    assert exc.value.line_content == "param Keq2 = a"


def test_param_k_alias_canonicalizes_to_equilibrium_parameter_namespace():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=6.0; K=3.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )
    mech = _base_mech(dsl)

    apply_parameter_algebra_to_mechanism(dsl, mechanism=mech, require_mutable=False)
    assert float(mech.equilibria[0].Keq) == pytest.approx(5.0)

    spec = parse_parameter_algebra_spec_from_dsl_text(
        "\n".join(
            [
                "# Algebra",
                "param K1 = 5",
            ]
        ),
        mechanism_param_names={"kf1", "kr1", "Keq1"},
    )

    assert [assignment.name for assignment in spec.param_statements] == ["Keq1"]

    derived = evaluate_parameter_algebra(
        spec,
        base_values={"kf1": 6.0, "Keq1": 3.0},
    )

    assert derived["Keq1"] == pytest.approx(5.0)


@pytest.mark.parametrize("line", ["let K1 = 5", "K1 = 5"])
def test_k_alias_non_param_assignments_fail_clearly_for_mechanism_namespace(line):
    with pytest.raises(DSLError, match="rate/equilibrium parameter"):
        parse_parameter_algebra_spec_from_dsl_text(
            "\n".join(
                [
                    "# Algebra",
                    line,
                ]
            ),
            mechanism_param_names={"kf1", "kr1", "Keq1"},
        )


def test_rate_constant_unit_formatting():
    assert rate_constant_unit(1) == "1/s"
    assert rate_constant_unit(2) == "1/(M s)"
    assert rate_constant_unit(3) == "1/(M^2 s)"
    assert rate_constant_unit(0) == "M/s"

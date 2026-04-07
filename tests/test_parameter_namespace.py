from __future__ import annotations

import pytest

from kindred.core.simulator.dsl import _parse_dsl_ir, parse_dsl_to_mechanism
from kindred.core.simulator.parameter_namespace import (
    build_flat_compat_namespace,
    build_namespace_from_ir_steps,
    build_namespace_from_mechanism,
)


@pytest.mark.parametrize(
    ("dsl", "expected"),
    [
        ("reaction: A -> B; k=1\n", {"k1"}),
        ("equilibrium: A <-> B; kf=6; kr=2\n", {"kf1", "kr1"}),
        ("reaction: A <-> B; kf=6; Keq=3\n", {"kf1", "kr1", "Keq1"}),
    ],
)
def test_ir_and_mechanism_namespace_builders_match_for_canonical_names(dsl, expected):
    ir = _parse_dsl_ir(dsl)
    mechanism = parse_dsl_to_mechanism(dsl, initials={})

    ir_namespace = build_namespace_from_ir_steps(ir.steps)
    mechanism_namespace = build_namespace_from_mechanism(mechanism)

    assert ir_namespace.flat_names() == expected
    assert mechanism_namespace.flat_names() == expected
    assert ir_namespace.flat_names() == mechanism_namespace.flat_names()


def test_ir_and_mechanism_builders_omit_keq_without_explicit_keq():
    dsl = "equilibrium: A <-> B; kf=6; kr=2\n"
    ir = _parse_dsl_ir(dsl)
    mechanism = parse_dsl_to_mechanism(dsl, initials={})

    assert build_namespace_from_ir_steps(ir.steps).flat_names() == {"kf1", "kr1"}
    assert build_namespace_from_mechanism(mechanism).flat_names() == {"kf1", "kr1"}


def test_ir_and_mechanism_builders_include_keq_for_explicit_keq():
    dsl = "reaction: A <-> B; kf=6; Keq=3\n"
    ir = _parse_dsl_ir(dsl)
    mechanism = parse_dsl_to_mechanism(dsl, initials={})

    assert build_namespace_from_ir_steps(ir.steps).flat_names() == {"kf1", "kr1", "Keq1"}
    assert build_namespace_from_mechanism(mechanism).flat_names() == {"kf1", "kr1", "Keq1"}


def test_flat_compat_namespace_rejects_noncanonical_inputs():
    with pytest.raises(ValueError, match="already-canonical"):
        build_flat_compat_namespace({"K1"})

    with pytest.raises(ValueError, match="already-canonical"):
        build_flat_compat_namespace({"k01"})


def test_ir_namespace_builder_rejects_malformed_steps():
    with pytest.raises(ValueError, match="missing required namespace metadata"):
        build_namespace_from_ir_steps([object()])


def test_mechanism_namespace_builder_rejects_duplicate_step_indices():
    class _MechanismWithDuplicateStepIndex:
        metadata = {
            "step_index_map": [
                {"step_index": 1, "kind": "reaction"},
                {"step_index": 1, "kind": "reaction"},
            ]
        }

    with pytest.raises(ValueError, match="unique step indices"):
        build_namespace_from_mechanism(_MechanismWithDuplicateStepIndex())

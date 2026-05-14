import pytest

from kindred.core.simulator.dsl import _parse_dsl_ir, extract_parameter_names_from_dsl, parse_dsl_to_mechanism
from kindred.core.simulator.dsl_parameter_scan import _scan_mechanism_param_names
from kindred.core.simulator.errors import DSLError
from kindred.core.simulator.parameter_namespace import build_namespace_from_canonical_names
from kindred.core.simulator.parameter_algebra import mechanism_parameter_names
from kindred.core.simulator.parameter_algebra import parse_parameter_algebra_spec_from_dsl_text

pytestmark = pytest.mark.unit

def test_scan_public_path_accepts_indexed_k_direct_spelling_for_irreversible_step():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )

    assert "k1" in extract_parameter_names_from_dsl(dsl)


def test_scan_public_path_rejects_indexed_k_for_equilibrium_step():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; Keq=3.0; kf=6.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )

    with pytest.raises(DSLError, match="not a valid indexed parameter identifier"):
        extract_parameter_names_from_dsl(dsl)


def test_scan_public_path_canonicalizes_observable_rhs_direct_spelling_for_irreversible_step():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Algebra",
            "let signal = K1",
        ]
    )

    names = extract_parameter_names_from_dsl(dsl)

    assert "k1" in names
    assert "signal" in names


def test_scan_public_path_rejects_observable_rhs_indexed_k_for_equilibrium_step():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; Keq=3.0; kf=6.0",
            "# Algebra",
            "let signal = K1",
        ]
    )

    with pytest.raises(DSLError, match="not a valid indexed parameter identifier"):
        extract_parameter_names_from_dsl(dsl)


def test_scan_public_path_preserves_longer_observable_rhs_names_as_ordinary():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; Keq=3.0; kf=6.0",
            "# Algebra",
            "let signal = K1_test",
        ]
    )

    names = extract_parameter_names_from_dsl(dsl)

    assert "signal" in names


def test_scan_namespace_matches_global_step_indexing():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: C <-> D; kf=2.0; kr=0.5",
            "equilibrium: E <-> F; Keq=4.0; kf=8.0",
        ]
    )

    ir = _parse_dsl_ir(dsl)

    assert _scan_mechanism_param_names(ir) == {"k1", "kf2", "kr2", "Keq2", "kf3", "kr3", "Keq3"}


def test_scan_namespace_includes_keq_for_every_reversible_step():
    ir = _parse_dsl_ir("equilibrium: A <-> B; kf=6; kr=2\n")

    assert _scan_mechanism_param_names(ir) == {"kf1", "kr1", "Keq1"}


def test_scan_namespace_includes_keq_for_reversible_reaction_with_explicit_keq():
    ir = _parse_dsl_ir("reaction: A <-> B; kf=6; Keq=3\n")

    assert _scan_mechanism_param_names(ir) == {"kf1", "kr1", "Keq1"}


@pytest.mark.parametrize(
    ("dsl", "expected"),
    [
        ("reaction: A -> B; k=1\n", {"k1"}),
        ("reaction: A <-> B; kf=6; kr=2\n", {"kf1", "kr1", "Keq1"}),
        ("equilibrium: A <-> B; kf=6; kr=2\n", {"kf1", "kr1", "Keq1"}),
        ("equilibrium: A <-> B; kf=6; K=3\n", {"kf1", "kr1", "Keq1"}),
    ],
)
def test_scan_namespace_matches_executable_namespace_for_audit_cases(dsl, expected):
    ir = _parse_dsl_ir(dsl)
    mech = parse_dsl_to_mechanism(dsl, initials={})

    assert _scan_mechanism_param_names(ir) == expected
    assert mechanism_parameter_names(mech) == expected


def test_scan_private_validation_reversible_step_rejects_unresolved_indexed_k():
    with pytest.raises(DSLError, match="not a valid indexed parameter identifier"):
        parse_parameter_algebra_spec_from_dsl_text(
            "\n".join(
                [
                    "param K2 = 5",
                ]
            ),
            mechanism_namespace=build_namespace_from_canonical_names({"k1", "kf2", "kr2", "kf3", "kr3", "Keq3"}),
        )

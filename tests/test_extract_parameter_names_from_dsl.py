import pytest

from kindred.core.simulator.dsl import extract_parameter_names_from_dsl
from kindred.core.simulator.errors import DSLError


def test_extract_parameter_names_from_dsl_enforces_param_algebra_ambiguity_guard():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Algebra",
            "let k1 = 2.0",
        ]
    )
    with pytest.raises(DSLError):
        extract_parameter_names_from_dsl(dsl)


def test_extract_parameter_names_from_dsl_extracts_algebra_param_and_observable_names():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: B -> C; k=2.0",
            "param k1 = 4*k2",
            "let obs = 3.0",
            "x = 1.0",
        ]
    )
    names = extract_parameter_names_from_dsl(dsl)
    assert "k" in names
    assert "k1" in names
    assert "obs" in names
    assert "x" in names


def test_extract_parameter_names_from_dsl_rejects_duplicate_equilibrium_aliases():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=1.0; Keq=2.0; K_eq=3.0",
        ]
    )

    with pytest.raises(DSLError, match="Duplicate parameter"):
        extract_parameter_names_from_dsl(dsl)


def test_extract_parameter_names_from_dsl_accepts_k_alias_on_irreversible_step():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )

    names = extract_parameter_names_from_dsl(dsl)

    assert "k1" in names


def test_extract_parameter_names_from_dsl_rejects_k_alias_on_equilibrium_step():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; Keq=3.0; kf=6.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )

    with pytest.raises(DSLError, match="Keq1"):
        extract_parameter_names_from_dsl(dsl)


def test_extract_parameter_names_from_dsl_accepts_k_alias_on_equilibrium_without_explicit_keq():
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=6.0; kr=2.0",
            "# Algebra",
            "param K1 = 5",
        ]
    )

    names = extract_parameter_names_from_dsl(dsl)

    assert "kf1" in names


def test_extract_parameter_names_from_dsl_accepts_keq_name_for_reversible_reaction_with_explicit_keq():
    dsl = "\n".join(
        [
            "reaction: A <-> B; kf=6.0; Keq=3.0",
            "# Algebra",
            "param Keq1 = 5",
        ]
    )

    names = extract_parameter_names_from_dsl(dsl)

    assert "Keq1" in names

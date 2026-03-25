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
            "# Algebra",
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


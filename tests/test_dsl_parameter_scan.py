import pytest

from kindred.core.simulator.dsl_parameter_scan import _parameter_family


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("Keq", "Keq"),
        ("Keq1", "Keq"),
        ("Keq2", "Keq"),
        ("Keq10", "Keq"),
    ],
)
def test_parameter_family_recognizes_keq_suffixes(name, family):
    assert _parameter_family(name) == family

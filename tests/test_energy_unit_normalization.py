from __future__ import annotations

import pytest

from kindred.core.simulator.common import normalize_energy_unit


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("J/mol", "J/mol"),
        ("kJ/mol", "kJ/mol"),
        ("kcal/mol", "kcal/mol"),
    ],
)
def test_normalize_energy_unit_accepts_canonical_units(value: str, expected: str) -> None:
    assert normalize_energy_unit(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("j/mol", "J/mol"),
        ("kj/mol", "kJ/mol"),
        ("KJ/MOL", "kJ/mol"),
        ("KCAL/MOL", "kcal/mol"),
        ("Kcal/Mol", "kcal/mol"),
        ("j/MOL", "J/mol"),
    ],
)
def test_normalize_energy_unit_accepts_case_variations(value: str, expected: str) -> None:
    assert normalize_energy_unit(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("j", "J/mol"),
        ("kj", "kJ/mol"),
        ("kcal", "kcal/mol"),
    ],
)
def test_normalize_energy_unit_accepts_short_forms(value: str, expected: str) -> None:
    assert normalize_energy_unit(value) == expected


@pytest.mark.parametrize("value", ["hartree", "eh", "Hartree", "EH"])
def test_normalize_energy_unit_accepts_hartree_when_allowed(value: str) -> None:
    assert normalize_energy_unit(value, allow_hartree=True) == "hartree"


def test_normalize_energy_unit_rejects_hartree_without_default_when_disallowed() -> None:
    with pytest.raises(ValueError):
        normalize_energy_unit("hartree", allow_hartree=False)


def test_normalize_energy_unit_returns_default_for_hartree_when_disallowed() -> None:
    assert normalize_energy_unit("hartree", default="kJ/mol", allow_hartree=False) == "kJ/mol"


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        ("garbage", "kJ/mol", "kJ/mol"),
        (None, "kJ/mol", "kJ/mol"),
        ("", "kJ/mol", "kJ/mol"),
    ],
)
def test_normalize_energy_unit_returns_default_for_invalid_inputs(
    value: object,
    default: str,
    expected: str,
) -> None:
    assert normalize_energy_unit(value, default=default) == expected


@pytest.mark.parametrize("value", ["garbage", "", 42])
def test_normalize_energy_unit_raises_without_default(value: object) -> None:
    with pytest.raises(ValueError):
        normalize_energy_unit(value)


def test_normalize_energy_unit_strips_whitespace() -> None:
    assert normalize_energy_unit("  kJ/mol  ") == "kJ/mol"

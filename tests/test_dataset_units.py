from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit



def test_public_unit_display_tuples_use_ascii_micro_forms() -> None:
    from kindred.core.datasets.units import CONCENTRATION_UNIT_DISPLAY, TIME_UNIT_DISPLAY

    assert TIME_UNIT_DISPLAY == ("fs", "ps", "ns", "us", "ms", "s", "min", "h")
    assert CONCENTRATION_UNIT_DISPLAY == ("fM", "pM", "nM", "uM", "mM", "M")


def test_parse_time_unit_supports_all_declared_units() -> None:
    from kindred.core.datasets.units import parse_time_unit

    assert parse_time_unit("fs") == 1e-15
    assert parse_time_unit("ps") == 1e-12
    assert parse_time_unit("ns") == 1e-9
    assert parse_time_unit("µs") == 1e-6
    assert parse_time_unit("us") == 1e-6
    assert parse_time_unit("μs") == 1e-6
    assert parse_time_unit("ms") == 1e-3
    assert parse_time_unit("s") == 1.0
    assert parse_time_unit("min") == 60.0
    assert parse_time_unit("h") == 3600.0


@pytest.mark.parametrize("value", ["min", "MIN", "Min"])
def test_parse_time_unit_allows_case_insensitive_word_forms(value: str) -> None:
    from kindred.core.datasets.units import parse_time_unit

    assert parse_time_unit(value) == 60.0


@pytest.mark.parametrize("value", ["MS", "Ms"])
def test_parse_time_unit_rejects_case_mismatched_si_symbols(value: str) -> None:
    from kindred.core.datasets.units import parse_time_unit

    with pytest.raises(ValueError, match="Supported time units"):
        parse_time_unit(value)


def test_parse_time_unit_strips_whitespace_and_accepts_both_micro_unicode_forms() -> None:
    from kindred.core.datasets.units import parse_time_unit

    assert parse_time_unit("  µs  ") == 1e-6
    assert parse_time_unit("  μs  ") == 1e-6


@pytest.mark.parametrize("value", ["fortnight", ""])
def test_parse_time_unit_rejects_unknown_or_empty_strings(value: str) -> None:
    from kindred.core.datasets.units import parse_time_unit

    with pytest.raises(ValueError, match="Supported time units"):
        parse_time_unit(value)


def test_parse_concentration_unit_supports_all_declared_units() -> None:
    from kindred.core.datasets.units import parse_concentration_unit

    assert parse_concentration_unit("fM") == 1e-15
    assert parse_concentration_unit("pM") == 1e-12
    assert parse_concentration_unit("nM") == 1e-9
    assert parse_concentration_unit("µM") == 1e-6
    assert parse_concentration_unit("uM") == 1e-6
    assert parse_concentration_unit("μM") == 1e-6
    assert parse_concentration_unit("mM") == 1e-3
    assert parse_concentration_unit("M") == 1.0


def test_parse_concentration_unit_strips_whitespace_and_accepts_both_micro_unicode_forms() -> None:
    from kindred.core.datasets.units import parse_concentration_unit

    assert parse_concentration_unit("  µM  ") == 1e-6
    assert parse_concentration_unit("  μM  ") == 1e-6


@pytest.mark.parametrize("value", ["MM", "unknown", ""])
def test_parse_concentration_unit_rejects_invalid_or_unsupported_inputs(value: str) -> None:
    from kindred.core.datasets.units import parse_concentration_unit

    with pytest.raises(ValueError, match="Supported concentration units"):
        parse_concentration_unit(value)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (["s", "µM", "mM"], True),
        (["time", "species_A", "species_B"], False),
        (["s", "µM", "not_a_unit"], True),
        (["", "", ""], False),
        (["s"], True),
        (["min", "uM", "nM"], True),
    ],
)
def test_looks_like_unit_row_detects_likely_unit_rows(row: list[str], expected: bool) -> None:
    from kindred.core.datasets.units import looks_like_unit_row

    assert looks_like_unit_row(row) is expected


def test_parse_unit_returns_time_category_and_factor() -> None:
    from kindred.core.datasets.units import parse_unit

    assert parse_unit("ms") == ("time", 1e-3)


def test_parse_unit_returns_concentration_category_and_factor() -> None:
    from kindred.core.datasets.units import parse_unit

    assert parse_unit("uM") == ("concentration", 1e-6)


def test_parse_unit_rejects_unknown_units() -> None:
    from kindred.core.datasets.units import parse_unit

    with pytest.raises(ValueError, match="Supported time units"):
        parse_unit("kg")


def test_default_unit_assumptions_returns_canonical_si_defaults() -> None:
    from kindred.core.datasets.units import default_unit_assumptions

    assert default_unit_assumptions() == {
        "time": ("s", 1.0),
        "concentration": ("M", 1.0),
    }

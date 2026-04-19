import pytest

from kindred.gui.ui_helpers import safe_float_parse

pytestmark = [pytest.mark.unit]


def test_safe_float_parse_rejects_nan_and_inf():
    assert safe_float_parse("nan", 1.25) == 1.25
    assert safe_float_parse("inf", 1.25) == 1.25
    assert safe_float_parse("-inf", 1.25) == 1.25


def test_safe_float_parse_parses_finite_and_rejects_bad_values():
    assert safe_float_parse("1.5", 9.0) == 1.5
    assert safe_float_parse("1e-12", 9.0) == 1e-12
    assert safe_float_parse("not-a-number", 9.0) == 9.0
    assert safe_float_parse(None, 9.0) == 9.0  # type: ignore[arg-type]

from kindred.core.simulator.wegscheider import complex_key


def test_complex_key_does_not_strip_integer_trailing_zeros():
    assert complex_key({"A": 2}) != complex_key({"A": 20})
    assert complex_key({"A": 100}) == "100A"


def test_complex_key_formats_fractional_coefficients_compactly():
    assert complex_key({"A": 1.5}) == "1.5A"

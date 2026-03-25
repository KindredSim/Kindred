import math

import pytest


def test_sanitize_nonneg_finite_coerces_invalid_to_zero():
    from kindred.gui.species_sliders_logic import sanitize_nonneg_finite

    assert sanitize_nonneg_finite(float("nan")) == 0.0
    assert sanitize_nonneg_finite(float("inf")) == 0.0
    assert sanitize_nonneg_finite(-1.0) == 0.0
    assert sanitize_nonneg_finite(0.0) == 0.0
    assert sanitize_nonneg_finite(1.25) == pytest.approx(1.25)


def test_try_nonneg_finite_flags_invalid_inputs():
    from kindred.gui.species_sliders_logic import try_nonneg_finite

    v, ok = try_nonneg_finite("not-a-number")
    assert v == 0.0 and ok is False

    v, ok = try_nonneg_finite(float("nan"))
    assert v == 0.0 and ok is False

    v, ok = try_nonneg_finite(-1.0)
    assert v == 0.0 and ok is False

    v, ok = try_nonneg_finite(1.25)
    assert v == pytest.approx(1.25) and ok is True


def test_compute_row_max_ignores_negatives_and_nan():
    from kindred.gui.species_sliders_logic import compute_row_max

    assert compute_row_max([float("nan"), -2.0, 0.0]) == 0.0
    assert compute_row_max([-2.0, 3.0, float("nan"), 1.0]) == pytest.approx(3.0)


def test_compute_slider_max_option_c():
    from kindred.gui.species_sliders_logic import compute_slider_max_option_c

    # row_max = 0, v = 0 => at least 1.0
    assert compute_slider_max_option_c(v=0.0, row_max=0.0) == pytest.approx(1.0)

    # row_max drives 2*row_max
    assert compute_slider_max_option_c(v=0.1, row_max=2.0) == pytest.approx(4.0)

    # 5*v dominates when v is large relative to row_max
    assert compute_slider_max_option_c(v=3.0, row_max=1.0) == pytest.approx(15.0)

    # invalid inputs are treated as 0
    assert math.isfinite(compute_slider_max_option_c(v=float("nan"), row_max=2.0))
    assert compute_slider_max_option_c(v=float("nan"), row_max=2.0) == pytest.approx(4.0)

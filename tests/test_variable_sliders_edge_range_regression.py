from __future__ import annotations

import math

import pytest

from kindred.gui.widgets.variable_sliders import VariableSliders


pytestmark = [pytest.mark.gui]


def _make_sliders(qtbot):
    widget = VariableSliders()
    qtbot.addWidget(widget)
    widget.set_variables({"k1": 1.0})
    assert widget.has_variable("k1")
    return widget


def test_slider_mapping_endpoints_are_valid(qtbot):
    sliders = _make_sliders(qtbot)
    slider = sliders._sliders["k1"]

    sliders._on_slider_changed("k1", slider.minimum())
    min_value = float(sliders.get_variables()["k1"])
    assert math.isfinite(min_value)
    assert 1e-12 <= min_value <= 1e12
    assert min_value == pytest.approx(1e-12)

    sliders._on_slider_changed("k1", slider.maximum())
    max_value = float(sliders.get_variables()["k1"])
    assert math.isfinite(max_value)
    assert 1e-12 <= max_value <= 1e12
    assert max_value == pytest.approx(1e12)


def test_on_slider_changed_at_edges_does_not_throw_and_preserves_bounds(qtbot):
    sliders = _make_sliders(qtbot)
    slider = sliders._sliders["k1"]
    initial_range = tuple(sliders._slider_ranges["k1"])
    min_pos = int(slider.minimum())
    max_pos = int(slider.maximum())

    # Mirror the slider->commit round trip path used by the GUI.
    sliders.variableChanged.connect(lambda name, value: sliders.update_variable(name, value))

    expected = [
        (min_pos, 1e-12),
        (max_pos, 1e12),
        (min_pos, 1e-12),
        (max_pos, 1e12),
    ]
    for pos, expected_value in expected:
        sliders._on_slider_changed("k1", pos)
        current = float(sliders.get_variables()["k1"])
        assert math.isfinite(current)
        assert 1e-12 <= current <= 1e12
        assert current == pytest.approx(expected_value)
        assert tuple(sliders._slider_ranges["k1"]) == initial_range

    assert sliders._value_to_slider_pos("k1", 1e-12) == min_pos
    assert sliders._value_to_slider_pos("k1", 1e12) == max_pos


def test_variable_slider_label_is_bounded_and_preserves_full_tooltip(qtbot):
    widget = VariableSliders()
    qtbot.addWidget(widget)
    long_name = "rate_" + ("very_long_" * 18)
    long_label = "metadata " + ("label " * 24)

    widget.set_variables(
        {long_name: 1.0},
        metadata={long_name: {"label": long_label, "unit": "M/s", "scale": "linear"}},
    )

    label = widget._labels[long_name]
    assert label.maximumWidth() <= 260
    assert label.toolTip() == f"{long_name} ({long_label}) [M/s]"
    assert label.text() != label.toolTip()

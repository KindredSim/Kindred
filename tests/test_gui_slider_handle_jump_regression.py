from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from tests.worker_stubs import make_contained_simulation_worker_stub


pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _slider_handle_center(slider: QtWidgets.QSlider) -> QtCore.QPoint:
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider,
        option,
        QtWidgets.QStyle.SC_SliderHandle,
        slider,
    )
    return handle.center()


def test_slider_drag_does_not_recenter_other_ranges_or_break_ctc(main_window, qtbot, monkeypatch):
    """
    Regression (handle jump / snap):
    Derived-parameter refresh during slider gestures must not recenter slider ranges or
    move slider handles while slider-triggered simulations still update stats/CTC.
    """
    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                # Extreme but slider-representable values: derived kr = kf/K = 1e-24.
                "equilibrium: A <-> B ; kf=1e-12, K=1e12",
                "reaction: B -> C ; k=0.1",
                "init: A=1, B=0, C=0",
            ]
        )
        + "\n"
    )
    main_window._extract_and_populate_variables()

    # Speed: slider-run tests should not spend time preparing/binding.
    monkeypatch.setattr(main_window, "_prepare_slider_runtime", lambda *a, **k: None)

    def _payload(worker) -> dict:
        t = np.linspace(0.0, 1.0, 4)
        Y = np.vstack([np.linspace(1.0, 0.4, t.size), np.linspace(0.0, 0.6, t.size)])
        return {
            "t": t,
            "Y": Y,
            "species_names": ["A", "B"],
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": dict(worker._solver_config),
        }

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.ContainedSimulationWorker",
        make_contained_simulation_worker_stub(payload_factory=_payload),
    )

    sliders = main_window._mechanism_editor._variable_sliders
    assert sliders.has_variable("Keq1")
    assert sliders.has_variable("kf1")
    assert sliders.has_variable("kr1")

    k_slider = sliders._sliders["Keq1"]
    kr_slider = sliders._sliders["kr1"]

    kr_range_before = sliders._slider_ranges["kr1"]
    kr_pos_before = kr_slider.value()
    ctc_before = dict(getattr(main_window, "_last_simulation_ctc", {}) or {})
    preview = main_window._preview_session

    qtbot.addWidget(main_window)
    main_window.show()

    # Begin a realistic press/release gesture so the same drag/commit pipeline runs.
    press_pos = _slider_handle_center(k_slider)
    qtbot.mousePress(k_slider, QtCore.Qt.LeftButton, pos=press_pos)
    qtbot.waitUntil(lambda: preview.slider_drag_active(), timeout=1000)

    # Change the value to trigger derived updates + a debounced slider-run.
    k_slider.setValue(max(k_slider.minimum(), k_slider.value() - 50))
    QtWidgets.QApplication.processEvents()

    # Derived parameter refresh must not recenter ranges / move handles during the gesture.
    assert sliders._slider_ranges["kr1"] == kr_range_before
    assert kr_slider.value() == kr_pos_before

    qtbot.mouseRelease(k_slider, QtCore.Qt.LeftButton, pos=press_pos)

    # Qt can deliver queued `valueChanged` events after release; ensure the derived refresh
    # path still does not mutate slider ranges/positions once the drag handler has run.
    main_window._on_variable_changed("Keq1", float(sliders.get_variables()["Keq1"]))
    QtWidgets.QApplication.processEvents()
    assert sliders._slider_ranges["kr1"] == kr_range_before
    assert kr_slider.value() == kr_pos_before

    # Ensure the slider-triggered simulation completion path still runs and updates CTC.
    qtbot.waitUntil(lambda: bool(getattr(main_window, "_last_simulation_ctc", {}) or {}), timeout=2500)
    ctc_after = dict(getattr(main_window, "_last_simulation_ctc", {}) or {})
    assert ctc_after != ctc_before

from __future__ import annotations

from typing import Dict, Tuple

import pytest

pytestmark = [pytest.mark.gui]


def test_main_window_public_variable_slider_routing_round_trips_state(main_window) -> None:
    metadata = {"k1": {"label": "k1", "unit": "1/s"}}

    main_window.set_variable_sliders({"k1": 2.0}, metadata=metadata)

    assert main_window.variable_slider_values() == {"k1": pytest.approx(2.0)}

    main_window.clear_variable_sliders()

    assert main_window.variable_slider_values() == {}


def test_main_window_public_parameter_summary_routes_to_main_plot(main_window, monkeypatch) -> None:
    captured: dict[str, Tuple[float, str]] = {}

    def _capture(parameters: Dict[str, Tuple[float, str]]) -> None:
        captured.clear()
        captured.update(parameters)

    monkeypatch.setattr(main_window.main_plot(), "update_parameters", _capture)

    main_window.update_main_plot_parameter_summary({"k1": (2.0, "1/s")})

    assert captured == {"k1": (pytest.approx(2.0), "1/s")}


def test_main_window_variable_runtime_component_owns_preview_runtime_state(main_window) -> None:
    runtime = main_window._variable_runtime

    runtime._slider_runtime = object()
    runtime.set_slider_runtime_dirty(False)
    main_window.set_variable_metadata({"k1": {"type": "reaction"}})

    main_window._invalidate_slider_runtime()

    assert "_slider_runtime" not in main_window.__dict__
    assert "_slider_runtime_dirty" not in main_window.__dict__
    assert "_suppress_slider_runtime_invalidation" not in main_window.__dict__
    assert not hasattr(main_window.simulation_controller.run_state, "variable_metadata")
    assert runtime._slider_runtime is None
    assert runtime.slider_runtime_dirty() is True
    assert main_window.variable_metadata() == {"k1": {"type": "reaction"}}


def test_main_window_no_longer_exposes_dead_public_runtime_forwarders(main_window) -> None:
    for name in (
        "prepare_slider_runtime",
        "apply_slider_overrides_to_bindings",
        "set_slider_runtime_dirty",
    ):
        assert name not in type(main_window).__dict__, f"Dead runtime forwarder {name} should be removed from MainWindow."

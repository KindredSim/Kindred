from __future__ import annotations

from dataclasses import dataclass

import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.solver_settings import SolverSettingsDialog


pytestmark = pytest.mark.gui


@dataclass
class _FakeCacheResult:
    ok: bool
    operation: str
    message: str = ""
    stats: dict[str, dict[str, int]] | None = None
    cache_state_changed: bool = False


class _FakeCachePort:
    def __init__(self) -> None:
        self.stats_result = _FakeCacheResult(
            ok=True,
            operation="stats",
            stats={
                "result": {"used": 2, "cap": 10, "bytes": 1_048_576},
                "preview": {"used": 1, "cap": 3, "bytes": 524_288},
            },
        )
        self.set_caps_result = _FakeCacheResult(ok=True, operation="set_caps")
        self.purge_result_result = _FakeCacheResult(ok=True, operation="purge_result_cache")
        self.purge_preview_result = _FakeCacheResult(ok=True, operation="purge_preview_cache")
        self.purge_all_result = _FakeCacheResult(ok=True, operation="purge_all_caches")
        self.calls: list[tuple[str, object]] = []

    def set_simulation_cache_caps(self, *, result_cap: int, preview_cap: int, persist: bool = True):
        self.calls.append(("set_caps", (int(result_cap), int(preview_cap), bool(persist))))
        return self.set_caps_result

    def simulation_cache_stats(self):
        self.calls.append(("stats", None))
        return self.stats_result

    def purge_simulation_result_cache(self):
        self.calls.append(("purge_result", None))
        return self.purge_result_result

    def purge_simulation_preview_cache(self):
        self.calls.append(("purge_preview", None))
        return self.purge_preview_result

    def purge_simulation_all_caches(self):
        self.calls.append(("purge_all", None))
        return self.purge_all_result


def test_solver_settings_refresh_surfaces_cache_status_failure(qtbot):
    port = _FakeCachePort()
    dialog = SolverSettingsDialog(cache_port=port)
    qtbot.addWidget(dialog)

    assert dialog._label_result_cache_status.text() == "Result cache: 2/10, 1.0 MB"
    assert dialog._label_preview_cache_status.text() == "Preview cache: 1/3, 0.5 MB"
    assert dialog._label_cache_status_error.text() == ""

    port.stats_result = _FakeCacheResult(
        ok=False,
        operation="stats",
        message="Failed to read simulation cache status: cache backend unavailable",
    )
    dialog._refresh_cache_status()

    assert dialog._label_result_cache_status.text() == "Result cache: unavailable"
    assert dialog._label_preview_cache_status.text() == "Preview cache: unavailable"
    assert "cache backend unavailable" in dialog._label_cache_status_error.text()


def test_solver_settings_apply_failure_surfaces_cache_error(qtbot):
    port = _FakeCachePort()
    dialog = SolverSettingsDialog(cache_port=port)
    qtbot.addWidget(dialog)

    port.set_caps_result = _FakeCacheResult(
        ok=False,
        operation="set_caps",
        message="Failed to apply cache caps: cache backend unavailable",
    )

    expected_result_cap = int(dialog._spin_result_cache_cap.value()) + 1
    expected_preview_cap = int(dialog._spin_preview_cache_cap.value())
    dialog._spin_result_cache_cap.setValue(expected_result_cap)

    assert ("set_caps", (expected_result_cap, expected_preview_cap, True)) in port.calls
    assert dialog._label_result_cache_status.text() == "Result cache: unavailable"
    assert dialog._label_preview_cache_status.text() == "Preview cache: unavailable"
    assert "Failed to apply cache caps" in dialog._label_cache_status_error.text()


def test_solver_settings_purge_failure_surfaces_cache_error(qtbot, monkeypatch):
    port = _FakeCachePort()
    dialog = SolverSettingsDialog(cache_port=port)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    port.purge_result_result = _FakeCacheResult(
        ok=False,
        operation="purge_result_cache",
        message="Failed to clear simulation result cache: cache backend unavailable",
    )

    dialog._confirm_and_purge(which="result")

    assert ("purge_result", None) in port.calls
    assert dialog._label_result_cache_status.text() == "Result cache: unavailable"
    assert dialog._label_preview_cache_status.text() == "Preview cache: unavailable"
    assert "Failed to clear simulation result cache" in dialog._label_cache_status_error.text()


def test_solver_settings_refresh_success_clears_cache_error(qtbot):
    port = _FakeCachePort()
    dialog = SolverSettingsDialog(cache_port=port)
    qtbot.addWidget(dialog)

    port.stats_result = _FakeCacheResult(
        ok=False,
        operation="stats",
        message="Failed to read simulation cache status: cache backend unavailable",
    )
    dialog._refresh_cache_status()
    assert dialog._label_cache_status_error.text() != ""

    port.stats_result = _FakeCacheResult(
        ok=True,
        operation="stats",
        stats={
            "result": {"used": 4, "cap": 8, "bytes": 2_097_152},
            "preview": {"used": 2, "cap": 5, "bytes": 1_048_576},
        },
    )
    dialog._refresh_cache_status()

    assert dialog._label_result_cache_status.text() == "Result cache: 4/8, 2.0 MB"
    assert dialog._label_preview_cache_status.text() == "Preview cache: 2/5, 1.0 MB"
    assert dialog._label_cache_status_error.text() == ""


def test_solver_settings_round_trip_slider_preview_controls(qtbot):
    dialog = SolverSettingsDialog()
    qtbot.addWidget(dialog)

    dialog.set_settings(
        {
            "slider_preview_solver": "BDF",
            "slider_preview_points": 350,
            "parameter_preview_debounce_ms": 35,
            "equilibrium_preview_debounce_ms": 90,
        }
    )

    settings = dialog.get_settings()

    assert settings["slider_preview_solver"] == "BDF"
    assert settings["slider_preview_points"] == 350
    assert settings["parameter_preview_debounce_ms"] == 35
    assert settings["equilibrium_preview_debounce_ms"] == 90

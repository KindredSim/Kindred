from __future__ import annotations

import builtins
import importlib
import logging
import sys

import pytest


LEGACY_LOG_STRINGS = (
    "PyQtGraph not available - plotting features will be limited",
    "PyQtGraph not available - GridPlotView requires pyqtgraph",
)

PLOTTING_MODULES_WITH_PREVIOUS_IMPORT_SIDE_EFFECTS = (
    "kindred.gui.plot_config",
    "kindred.gui.widgets.grid_plot_view",
    "kindred.gui.widgets.pyqtgraph_plot_panel",
)


def _block_pyqtgraph_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "pyqtgraph" or str(name).startswith("pyqtgraph."):
            raise ImportError("blocked pyqtgraph import for side-effect test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_plotting_modules_do_not_log_legacy_pyqtgraph_missing_messages_at_import_time(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    caplog.set_level(logging.DEBUG)
    _block_pyqtgraph_imports(monkeypatch)

    for module_name in PLOTTING_MODULES_WITH_PREVIOUS_IMPORT_SIDE_EFFECTS:
        _fresh_import(module_name)

    captured_text = caplog.text
    for legacy in LEGACY_LOG_STRINGS:
        assert legacy not in captured_text


def test_pyqtgraph_availability_helper_reports_true_when_importable():
    try:
        import pyqtgraph  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Expected pyqtgraph to be importable in this test environment: {exc!r}")

    plot_config = _fresh_import("kindred.gui.plot_config")
    assert plot_config.is_pyqtgraph_available() is True

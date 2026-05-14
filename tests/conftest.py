import gc
import logging
import os
import sys
from contextlib import suppress

# Ensure Qt always runs headless before importing PySide6/pytest-qt fixtures
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import multiprocessing
except NotImplementedError:
    multiprocessing = None

import pytest

logger = logging.getLogger(__name__)

def _disable_qdarktheme() -> None:
    # Neuter qdarktheme before any test or fixture constructs a MainWindow.
    # setup_theme() does expensive Qt stylesheet processing on every call, and
    # setup_theme("auto") starts an OS-sync thread that hangs in WSL/offscreen.
    try:
        import qdarktheme  # type: ignore[import-not-found]
    except Exception:
        return
    qdarktheme.setup_theme = lambda *a, **kw: None

# Configure multiprocessing to use "spawn" on Linux to avoid fork-related warnings
# with PySide6/Qt. This only affects the test environment.
if sys.platform.startswith("linux") and multiprocessing is not None:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set; ignore.
        pass


@pytest.fixture(autouse=True, scope="session")
def _neutralize_pyqtgraph_connect_cleanup():
    import pyqtgraph as pg

    if not hasattr(pg, "_connectCleanup") or not callable(pg._connectCleanup):
        pytest.fail("pyqtgraph._connectCleanup missing or not callable", pytrace=False)

    monkeypatch = pytest.MonkeyPatch()
    # Test-harness-only: exitCleanup=False was probed and found insufficient.
    # Neutralize the aboutToQuit hookup before first GraphicsView construction,
    # and do not move this private patch into product code.
    monkeypatch.setattr(pg, "_connectCleanup", lambda: None)
    try:
        yield
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="session")
def qt_app(_neutralize_pyqtgraph_connect_cleanup):
    """Ensure a QApplication instance exists for GUI-driven tests."""
    _disable_qdarktheme()
    from PySide6 import QtCore, QtWidgets

    QtCore.QStandardPaths.setTestModeEnabled(True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture(autouse=True)
def _drain_qt_between_tests():
    yield
    try:
        from PySide6 import QtCore, QtWidgets
    except Exception:
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for _ in range(5):
        app.processEvents()
    with suppress(RuntimeError, TypeError):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(5):
        app.processEvents()


def _clear_test_qsettings() -> None:
    from PySide6 import QtCore

    # Ensure per-test isolation: clear QSettings so a prior test cannot persist UI
    # state (e.g., theme) into the next MainWindow construction.
    try:
        settings = QtCore.QSettings("Kindred", "KindredGUI")
        settings.clear()
        settings.sync()
    except Exception as exc:
        logger.debug("Failed to clear QSettings for test isolation: %s", exc, exc_info=True)


def _patch_main_window_test_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    request: pytest.FixtureRequest | None = None,
) -> None:
    from PySide6 import QtWidgets
    from kindred.gui.main_window import MainWindow

    def _fake_templates_dir(_self):
        target = tmp_path / "templates"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    real_runtime_readiness = bool(
        request is not None
        and request.node.get_closest_marker("real_runtime_readiness") is not None
    )
    if not real_runtime_readiness:
        from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot

        class _ReadyTestRuntimeOwner:
            is_ready = True
            is_running = True
            owner_epoch = 1

            @property
            def simulation_plan_payload(self):
                return {}

            def solve(self, *_args, **_kwargs):
                return {}

            def close(self, *, kill: bool = False) -> None:
                _ = kill

        ready_owner = _ReadyTestRuntimeOwner()
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController.ensure_interactive_simulation_runtimes_available",
            lambda self, *, wait=False: None,
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController.interactive_simulation_runtime_ready",
            lambda self, *, fast_mode: True,
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController.interactive_simulation_runtimes_ready",
            lambda self: True,
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController._ready_contained_simulation_owner_for_plan",
            lambda self, **_kwargs: ready_owner,
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController._acquire_ready_contained_simulation_owner_for_plan",
            lambda self, **_kwargs: ready_owner,
        )
        monkeypatch.setattr(
            "kindred.core.simulation_runtime_readiness.SimulationRuntimeApplication.acquire_ready_owner",
            lambda self, **_kwargs: ready_owner,
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController._interactive_simulation_runtime_snapshot",
            lambda self, *, fast_mode: RuntimeReadinessSnapshot(
                mode="preview" if bool(fast_mode) else "ordinary",
                status="ready",
                ready=True,
                generation=1,
                required=True,
                controls_ready=True,
                polling=False,
            ),
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController._selected_run_runtime_snapshot",
            lambda self: RuntimeReadinessSnapshot(
                mode="ordinary",
                status="ready",
                ready=True,
                generation=1,
                required=True,
                controls_ready=True,
                polling=False,
            ),
        )
        monkeypatch.setattr(
            "kindred.gui.controllers.simulation_controller.SimulationController.ensure_parallel_batch_runtime_ready",
            lambda self, *, wait=False: None,
        )

    def _quiet_dialog(*_args, **_kwargs):
        return QtWidgets.QMessageBox.StandardButton.Ok

    for attr in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, attr, _quiet_dialog)

    monkeypatch.setattr(
        "kindred.gui.tutorial_manager.launch_tutorial",
        lambda *args, **kwargs: None,
    )

def _cleanup_main_window(window) -> None:
    from PySide6 import QtCore, QtWidgets

    try:
        window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)
    except Exception as exc:
        logger.debug("Failed to shutdown batch lane pool during test cleanup: %s", exc, exc_info=True)
    try:
        window.simulation_controller.release_current_simulation_worker()
    except Exception as exc:
        logger.debug("Failed to release simulation worker during test cleanup: %s", exc, exc_info=True)

    # Stop preview-session timers before closing to prevent stale timer
    # callbacks from firing during teardown.
    try:
        window._preview_session.reset_preview_state()
    except Exception as exc:
        logger.debug("Failed to reset preview state during test cleanup: %s", exc, exc_info=True)

    try:
        window.close()
    except Exception as exc:
        logger.debug("Failed to close MainWindow during test cleanup: %s", exc, exc_info=True)

    app = QtWidgets.QApplication.instance()
    assert app is not None
    for _ in range(30):
        app.processEvents()
    try:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    except Exception as exc:
        logger.debug("Failed to send posted deferred-delete events: %s", exc, exc_info=True)
    for _ in range(15):
        app.processEvents()

    # Close any stray dialogs/top-level widgets created by a test.
    stray_widgets = []
    for widget in list(app.topLevelWidgets()):
        if widget is window:
            continue
        if not (getattr(widget, "isVisible", lambda: False)() or isinstance(widget, QtWidgets.QDialog)):
            continue
        try:
            widget.close()
        except Exception as exc:
            logger.debug("Failed to close stray widget %r during cleanup: %s", widget, exc, exc_info=True)
        try:
            widget.deleteLater()
            stray_widgets.append(widget)
        except Exception as exc:
            logger.debug("Failed to deleteLater stray widget %r during cleanup: %s", widget, exc, exc_info=True)

    for _ in range(30):
        app.processEvents()
    try:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    except Exception as exc:
        logger.debug("Failed to send deferred-delete events for window: %s", exc, exc_info=True)
    for widget in stray_widgets:
        try:
            QtCore.QCoreApplication.sendPostedEvents(widget, QtCore.QEvent.DeferredDelete)
        except Exception as exc:
            logger.debug(
                "Failed to send posted deferred-delete events for stray widget %r: %s",
                widget,
                exc,
                exc_info=True,
            )
    for _ in range(15):
        app.processEvents()

    try:
        window.deleteLater()
    except Exception as exc:
        logger.debug("Failed to deleteLater MainWindow during test cleanup: %s", exc, exc_info=True)

    for _ in range(15):
        app.processEvents()
    try:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    except Exception as exc:
        logger.debug("Failed to send deferred-delete events for MainWindow: %s", exc, exc_info=True)
    for _ in range(10):
        app.processEvents()

    # Break Python-side reference cycles so the C++ side can be collected.
    del window
    gc.collect()
    for _ in range(5):
        app.processEvents()

    assert QtWidgets.QApplication.activeModalWidget() is None
    assert QtWidgets.QApplication.activePopupWidget() is None
    extra_visible = [
        w
        for w in app.topLevelWidgets()
        if getattr(w, "isVisible", lambda: False)()
    ]
    assert not extra_visible, f"Visible top-level widgets leaked: {extra_visible!r}"

@pytest.fixture
def main_window(qt_app, monkeypatch, tmp_path, request):
    """
    Provide a MainWindow with dialogs routed to temporary locations.

    Heavy-weight dialogs are stubbed to keep GUI tests deterministic and
    avoid filesystem side effects.
    """
    from kindred.gui.main_window import MainWindow

    _ = qt_app
    _clear_test_qsettings()
    _patch_main_window_test_environment(monkeypatch, tmp_path, request=request)
    window = MainWindow()
    # Generic GUI tests stub startup runtime warming; keep controls usable unless
    # the test explicitly reinstates the real readiness path.
    real_runtime_readiness = (
        request is not None
        and request.node.get_closest_marker("real_runtime_readiness") is not None
    )
    if not real_runtime_readiness:
        try:
            window._set_runtime_backed_controls_ready(True)
        except Exception as exc:
            logger.debug("Failed to mark stubbed test runtime ready: %s", exc, exc_info=True)
    try:
        yield window
    finally:
        _cleanup_main_window(window)

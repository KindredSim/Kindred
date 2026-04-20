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
_retired_main_window_host = None
_retired_main_window = None

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


@pytest.fixture(scope="session")
def qt_app():
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

    def _quiet_dialog(*_args, **_kwargs):
        return QtWidgets.QMessageBox.StandardButton.Ok

    for attr in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, attr, _quiet_dialog)

    monkeypatch.setattr(
        "kindred.gui.tutorial_manager.launch_tutorial",
        lambda *args, **kwargs: None,
    )


def _get_retired_main_window_host():
    from PySide6 import QtWidgets

    global _retired_main_window_host
    if _retired_main_window_host is None:
        host = QtWidgets.QWidget()
        host.setObjectName("_retired_main_window_host")
        host.hide()
        _retired_main_window_host = host
    return _retired_main_window_host


def _qt_object_is_alive(obj) -> bool:
    try:
        from shiboken6 import isValid
    except Exception:
        return obj is not None

    if obj is None:
        return False
    try:
        return bool(isValid(obj))
    except Exception:
        return False


def _close_tracked_qt_widgets(item) -> None:
    widgets = getattr(item, "qt_widgets", None)
    if not widgets:
        return

    for widget_ref, before_close_func in item.qt_widgets:
        widget = widget_ref()
        if widget is None:
            continue
        if before_close_func is not None:
            before_close_func(widget)
        widget.close()
        if getattr(widget, "_kindred_skip_qtbot_delete", False):
            continue
        widget.deleteLater()
    del item.qt_widgets


def _retire_main_window(window) -> None:
    from PySide6 import QtCore, QtWidgets

    global _retired_main_window
    app = QtWidgets.QApplication.instance()
    assert app is not None

    if _retired_main_window is not None:
        if _qt_object_is_alive(_retired_main_window):
            try:
                _retired_main_window.deleteLater()
            except Exception as exc:
                logger.debug("Failed to delete previously retired MainWindow: %s", exc, exc_info=True)
            for _ in range(30):
                app.processEvents()
            try:
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            except Exception as exc:
                logger.debug("Failed to drain deferred deletes for retired MainWindow: %s", exc, exc_info=True)
            for _ in range(15):
                app.processEvents()
        _retired_main_window = None

    if not _qt_object_is_alive(window):
        return

    for attr in (
        "_update_parameter_table_from_sliders",
        "_restore_maximized_state_if_needed",
        "_recover_restored_floating_docks",
    ):
        try:
            setattr(window, attr, lambda *args, **kwargs: None)
        except Exception as exc:
            logger.debug("Failed to neutralize retired MainWindow callback %s: %s", attr, exc, exc_info=True)

    for timer in window.findChildren(QtCore.QTimer):
        try:
            timer.stop()
        except Exception as exc:
            logger.debug("Failed to stop retired MainWindow timer %r: %s", timer, exc, exc_info=True)

    try:
        central_widget = window.takeCentralWidget()
    except Exception as exc:
        central_widget = None
        logger.debug("Failed to detach central widget from retired MainWindow: %s", exc, exc_info=True)
    if central_widget is not None:
        try:
            central_widget.hide()
            central_widget.setParent(None)
            central_widget.deleteLater()
        except Exception as exc:
            logger.debug("Failed to delete retired MainWindow central widget: %s", exc, exc_info=True)

    for dock_widget in list(window.findChildren(QtWidgets.QDockWidget)):
        try:
            window.removeDockWidget(dock_widget)
        except Exception:
            pass
        try:
            dock_widget.hide()
            dock_widget.deleteLater()
        except Exception as exc:
            logger.debug("Failed to delete retired MainWindow dock widget %r: %s", dock_widget, exc, exc_info=True)

    for child_type in (QtWidgets.QMenuBar, QtWidgets.QStatusBar):
        for child in list(window.findChildren(child_type)):
            if child.parent() is not window:
                continue
            try:
                child.hide()
                child.setParent(None)
                child.deleteLater()
            except Exception as exc:
                logger.debug("Failed to detach retired MainWindow child %r: %s", child, exc, exc_info=True)

    for _ in range(30):
        app.processEvents()
    try:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    except Exception as exc:
        logger.debug("Failed to drain deferred deletes while retiring MainWindow shell: %s", exc, exc_info=True)
    for _ in range(15):
        app.processEvents()

    window.setParent(_get_retired_main_window_host())
    window.hide()
    _retired_main_window = window


def _cleanup_main_window(window) -> None:
    from PySide6 import QtCore, QtWidgets

    try:
        window.simulation_controller.shutdown_batch_executor(force_terminate=True)
    except Exception as exc:
        logger.debug("Failed to shutdown batch executor during test cleanup: %s", exc, exc_info=True)
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

    # Keep deferred-delete draining for stray widgets, but do not destroy the
    # shared MainWindow itself here. Qt emits a deterministic timer warning on
    # stderr when the shared test window is destroyed during test teardown.
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

    # Retire the just-closed window under a hidden host so it no longer
    # accumulates as a top-level widget between tests. The previously retired
    # window is destroyed on the next cleanup pass, after it has spent a full
    # test cycle quiescent.
    if _qt_object_is_alive(window):
        try:
            _retire_main_window(window)
        except Exception as exc:
            raise AssertionError("Failed to retire MainWindow under hidden test host") from exc

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


@pytest.fixture(autouse=True, scope="session")
def _patch_qtbot_widget_teardown():
    try:
        import pytestqt.plugin
        import pytestqt.qtbot
    except Exception:
        yield
        return

    original_plugin_close_widgets = pytestqt.plugin._close_widgets
    original_qtbot_close_widgets = pytestqt.qtbot._close_widgets
    pytestqt.plugin._close_widgets = _close_tracked_qt_widgets
    pytestqt.qtbot._close_widgets = _close_tracked_qt_widgets
    try:
        yield
    finally:
        pytestqt.plugin._close_widgets = original_plugin_close_widgets
        pytestqt.qtbot._close_widgets = original_qtbot_close_widgets

@pytest.fixture
def main_window(qt_app, monkeypatch, tmp_path):
    """
    Provide a MainWindow with dialogs routed to temporary locations.

    Heavy-weight dialogs are stubbed to keep GUI tests deterministic and
    avoid filesystem side effects.
    """
    from kindred.gui.main_window import MainWindow

    _ = qt_app
    _clear_test_qsettings()
    _patch_main_window_test_environment(monkeypatch, tmp_path)
    window = MainWindow()
    window._kindred_skip_qtbot_delete = True
    try:
        yield window
    finally:
        _cleanup_main_window(window)

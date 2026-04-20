from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

import tests.conftest as test_conftest
from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]

QT_TIMER_WARNING = "QObject::startTimer: Timers can only be used with threads started with QThread"


def _drain_qt(app: QtWidgets.QApplication, obj=None) -> None:
    for _ in range(20):
        app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(obj, QtCore.QEvent.DeferredDelete)
    for _ in range(20):
        app.processEvents()
    gc.collect()
    for _ in range(5):
        app.processEvents()


def _count_top_level_main_windows(app: QtWidgets.QApplication) -> int:
    return sum(1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))


def _count_visible_top_level_main_windows(app: QtWidgets.QApplication) -> int:
    return sum(
        1
        for widget in app.topLevelWidgets()
        if isinstance(widget, MainWindow) and widget.isVisible()
    )


def _count_non_main_window_top_levels(app: QtWidgets.QApplication) -> int:
    return sum(1 for widget in app.topLevelWidgets() if not isinstance(widget, MainWindow))


def _tracked_widgets(request: pytest.FixtureRequest) -> list[QtWidgets.QWidget]:
    tracked = getattr(request.node, "qt_widgets", None)
    if tracked is None:
        return []
    return [
        widget
        for widget_ref, _before_close in tracked
        for widget in [widget_ref()]
        if widget is not None
    ]


@pytest.fixture
def fixture_owned_main_window_teardown_probe(qt_app, tmp_path):
    app = qt_app
    monkeypatch = pytest.MonkeyPatch()
    test_conftest._clear_test_qsettings()
    test_conftest._patch_main_window_test_environment(monkeypatch, tmp_path)
    window = MainWindow()
    delete_counts = {"qtbot_or_fixture": 0}
    cleanup_calls = {"shutdown": 0, "release": 0, "reset_preview": 0}

    original_delete_later = window.deleteLater
    original_shutdown = window.simulation_controller.shutdown_batch_executor
    original_release = window.simulation_controller.release_current_simulation_worker
    original_reset_preview = window._preview_session.reset_preview_state

    def wrapped_delete_later(*args, **kwargs):
        delete_counts["qtbot_or_fixture"] += 1
        return original_delete_later(*args, **kwargs)

    def wrapped_shutdown(*args, **kwargs):
        cleanup_calls["shutdown"] += 1
        return original_shutdown(*args, **kwargs)

    def wrapped_release(*args, **kwargs):
        cleanup_calls["release"] += 1
        return original_release(*args, **kwargs)

    def wrapped_reset_preview(*args, **kwargs):
        cleanup_calls["reset_preview"] += 1
        return original_reset_preview(*args, **kwargs)

    window.deleteLater = wrapped_delete_later
    window.simulation_controller.shutdown_batch_executor = wrapped_shutdown
    window.simulation_controller.release_current_simulation_worker = wrapped_release
    window._preview_session.reset_preview_state = wrapped_reset_preview

    try:
        yield window, delete_counts, cleanup_calls
        assert isValid(window)
        assert delete_counts["qtbot_or_fixture"] == 1
        test_conftest._cleanup_main_window(window)
        assert cleanup_calls == {"shutdown": 1, "release": 1, "reset_preview": 1}
        _drain_qt(app)
        assert not isValid(window)
    finally:
        monkeypatch.undo()


def test_shared_main_window_fixtures_are_not_available(request) -> None:
    for fixture_name in ("shared_main_window", "reset_shared_main_window", "main_window_init_counter"):
        with pytest.raises(pytest.FixtureLookupError):
            request.getfixturevalue(fixture_name)


def test_main_window_cleanup_keeps_qt_stderr_clean() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cleanup_script = textwrap.dedent(
        """
        import os
        from tempfile import TemporaryDirectory
        from pathlib import Path

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.chdir(%r)

        import pyqtgraph as pg
        import pytest
        from PySide6 import QtCore, QtWidgets

        if not hasattr(pg, "_connectCleanup") or not callable(pg._connectCleanup):
            raise SystemExit("pyqtgraph._connectCleanup missing or not callable")
        pg._connectCleanup = lambda: None

        import tests.conftest as test_conftest
        from kindred.gui.main_window import MainWindow

        app = QtWidgets.QApplication.instance()
        if app is None:
            test_conftest._disable_qdarktheme()
            QtCore.QStandardPaths.setTestModeEnabled(True)
            app = QtWidgets.QApplication([])

        monkeypatch = pytest.MonkeyPatch()
        with TemporaryDirectory(prefix="kindred-main-window-cleanup-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            try:
                test_conftest._clear_test_qsettings()
                test_conftest._patch_main_window_test_environment(monkeypatch, tmp_root)
                window = MainWindow()
                test_conftest._cleanup_main_window(window)
            finally:
                monkeypatch.undo()
        """
        % str(repo_root)
    )
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, "-c", cleanup_script],
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert QT_TIMER_WARNING not in result.stderr, result.stderr


def test_pyqtgraph_patch_applies_to_qtbot_only_pytest_subprocess() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_grid_plot_view_viewport_not_opengl.py",
        ],
        capture_output=True,
        check=False,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert QT_TIMER_WARNING not in result.stderr, result.stderr


def test_main_window_cleanup_destroys_windows_without_accumulation(qt_app) -> None:
    app = qt_app
    baseline_main_windows = _count_top_level_main_windows(app)
    baseline_visible_main_windows = _count_visible_top_level_main_windows(app)
    baseline_other_top_levels = _count_non_main_window_top_levels(app)
    helper_pool_delta = None

    for cycle in range(3):
        monkeypatch = pytest.MonkeyPatch()
        with TemporaryDirectory(prefix="kindred-main-window-cleanup-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            try:
                test_conftest._clear_test_qsettings()
                test_conftest._patch_main_window_test_environment(monkeypatch, tmp_root)
                window = MainWindow()
                test_conftest._cleanup_main_window(window)
                _drain_qt(app)
                assert not isValid(window), f"MainWindow wrapper survived cleanup on cycle {cycle + 1}"
                assert _count_top_level_main_windows(app) == baseline_main_windows
                assert _count_visible_top_level_main_windows(app) == baseline_visible_main_windows
                current_helper_delta = _count_non_main_window_top_levels(app) - baseline_other_top_levels
                if helper_pool_delta is None:
                    helper_pool_delta = current_helper_delta
                else:
                    assert current_helper_delta == helper_pool_delta, (
                        f"Hidden helper-widget pool grew on cycle {cycle + 1}: "
                        f"expected delta {helper_pool_delta}, got {current_helper_delta}"
                    )
            finally:
                monkeypatch.undo()


def test_qtbot_tracks_main_window_and_preserves_before_close_func(qtbot, main_window, request) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    baseline_main_windows = _count_top_level_main_windows(app) - 1
    before_close_calls: list[QtWidgets.QWidget] = []
    call_counts = {"close": 0, "deleteLater": 0}

    original_close = main_window.close
    original_delete_later = main_window.deleteLater

    def wrapped_close(*args, **kwargs):
        call_counts["close"] += 1
        return original_close(*args, **kwargs)

    def wrapped_delete_later(*args, **kwargs):
        call_counts["deleteLater"] += 1
        return original_delete_later(*args, **kwargs)

    main_window.close = wrapped_close
    main_window.deleteLater = wrapped_delete_later

    def before_close(widget: QtWidgets.QWidget) -> None:
        before_close_calls.append(widget)

    def verify_qtbot_teardown() -> None:
        assert before_close_calls == [main_window]
        assert call_counts["close"] == 1
        assert call_counts["deleteLater"] == 1
        assert not hasattr(request.node, "qt_widgets")
        _drain_qt(app)
        assert not isValid(main_window)
        assert _count_top_level_main_windows(app) == baseline_main_windows

    request.addfinalizer(verify_qtbot_teardown)

    qtbot.addWidget(main_window, before_close_func=before_close)

    tracked_widgets = _tracked_widgets(request)
    assert main_window in tracked_widgets
    assert not hasattr(test_conftest, "_QtBotTrackedWidgets")
    assert not hasattr(test_conftest, "_exclude_fixture_owned_main_window_from_qtbot_tracking")


def test_fixture_owned_main_window_cleans_up_before_qtbot_delete_later(
    qtbot,
    fixture_owned_main_window_teardown_probe,
    request,
) -> None:
    window, delete_counts, cleanup_calls = fixture_owned_main_window_teardown_probe
    before_close_calls: list[QtWidgets.QWidget] = []

    def before_close(widget: QtWidgets.QWidget) -> None:
        before_close_calls.append(widget)

    def verify_tracking() -> None:
        assert before_close_calls == [window]
        assert not hasattr(request.node, "qt_widgets")

    request.addfinalizer(verify_tracking)

    qtbot.addWidget(window, before_close_func=before_close)

    tracked_widgets = _tracked_widgets(request)
    assert window in tracked_widgets
    assert delete_counts["qtbot_or_fixture"] == 0
    assert cleanup_calls == {"shutdown": 0, "release": 0, "reset_preview": 0}


def test_fixture_cleanup_tolerates_qtbot_destroyed_main_window(qtbot, qt_app, request) -> None:
    app = qt_app
    baseline_main_windows = _count_top_level_main_windows(app)
    baseline_visible_main_windows = _count_visible_top_level_main_windows(app)
    before_close_calls: list[QtWidgets.QWidget] = []
    call_counts = {"close": 0, "deleteLater": 0}

    monkeypatch = pytest.MonkeyPatch()
    tmp_dir = TemporaryDirectory(prefix="kindred-main-window-double-owner-")
    tmp_root = Path(tmp_dir.name)
    test_conftest._clear_test_qsettings()
    test_conftest._patch_main_window_test_environment(monkeypatch, tmp_root)
    window = MainWindow()

    original_close = window.close
    original_delete_later = window.deleteLater

    def wrapped_close(*args, **kwargs):
        call_counts["close"] += 1
        return original_close(*args, **kwargs)

    def wrapped_delete_later(*args, **kwargs):
        call_counts["deleteLater"] += 1
        return original_delete_later(*args, **kwargs)

    window.close = wrapped_close
    window.deleteLater = wrapped_delete_later

    def before_close(widget: QtWidgets.QWidget) -> None:
        before_close_calls.append(widget)

    def verify_qtbot_then_fixture_cleanup() -> None:
        try:
            assert before_close_calls == [window]
            assert call_counts["close"] == 1
            assert call_counts["deleteLater"] == 1
            assert not hasattr(request.node, "qt_widgets")
            _drain_qt(app)
            assert not isValid(window)
            test_conftest._cleanup_main_window(window)
            _drain_qt(app)
            assert _count_top_level_main_windows(app) == baseline_main_windows
            assert _count_visible_top_level_main_windows(app) == baseline_visible_main_windows
        finally:
            monkeypatch.undo()
            tmp_dir.cleanup()

    request.addfinalizer(verify_qtbot_then_fixture_cleanup)

    qtbot.addWidget(window, before_close_func=before_close)

    tracked_widgets = _tracked_widgets(request)
    assert window in tracked_widgets


def test_qtbot_ordinary_widget_teardown_remains_stock(qtbot, request) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    before_close_calls: list[QtWidgets.QWidget] = []
    call_counts = {"close": 0, "deleteLater": 0}
    destroyed = {"emitted": False}

    class TrackedDialog(QtWidgets.QDialog):
        def __init__(self) -> None:
            super().__init__()
            self.close_event_count = 0

        def closeEvent(self, event) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    dialog = TrackedDialog()
    dialog.destroyed.connect(lambda *_args: destroyed.__setitem__("emitted", True))

    original_close = dialog.close
    original_delete_later = dialog.deleteLater

    def wrapped_close(*args, **kwargs):
        call_counts["close"] += 1
        return original_close(*args, **kwargs)

    def wrapped_delete_later(*args, **kwargs):
        call_counts["deleteLater"] += 1
        return original_delete_later(*args, **kwargs)

    dialog.close = wrapped_close
    dialog.deleteLater = wrapped_delete_later

    def before_close(widget: QtWidgets.QWidget) -> None:
        before_close_calls.append(widget)

    def verify_qtbot_teardown() -> None:
        assert before_close_calls == [dialog]
        assert call_counts["close"] == 1
        assert call_counts["deleteLater"] == 1
        assert dialog.close_event_count == 1
        assert not hasattr(request.node, "qt_widgets")
        _drain_qt(app)
        assert not isValid(dialog)
        assert destroyed["emitted"] is True

    request.addfinalizer(verify_qtbot_teardown)

    qtbot.addWidget(dialog, before_close_func=before_close)

    tracked_widgets = _tracked_widgets(request)
    assert dialog in tracked_widgets

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PySide6 import QtWidgets

import tests.conftest as test_conftest
from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]

QT_TIMER_WARNING = "QObject::startTimer: Timers can only be used with threads started with QThread"


def test_shared_main_window_fixtures_are_not_available(request) -> None:
    for fixture_name in ("shared_main_window", "reset_shared_main_window", "main_window_init_counter"):
        with pytest.raises(pytest.FixtureLookupError):
            request.getfixturevalue(fixture_name)


def _count_top_level_main_windows(app: QtWidgets.QApplication) -> int:
    return sum(1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))


def _retired_main_window_count() -> int:
    host = test_conftest._get_retired_main_window_host()
    return sum(1 for child in host.children() if isinstance(child, MainWindow))


def _retired_main_window_dock_count() -> int:
    retired_window = getattr(test_conftest, "_retired_main_window", None)
    if retired_window is None:
        return 0
    return len(retired_window.findChildren(QtWidgets.QDockWidget))


def _retired_main_window_shell_shape() -> tuple[bool, int, int]:
    retired_window = getattr(test_conftest, "_retired_main_window", None)
    if retired_window is None:
        return (True, 0, 0)
    central_widget = retired_window.centralWidget()
    menu_bars = [
        child
        for child in retired_window.findChildren(QtWidgets.QMenuBar)
        if child.parent() is retired_window
    ]
    status_bars = [
        child
        for child in retired_window.findChildren(QtWidgets.QStatusBar)
        if child.parent() is retired_window
    ]
    return (central_widget is None, len(menu_bars), len(status_bars))


def test_main_window_cleanup_keeps_qt_stderr_clean() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cleanup_script = textwrap.dedent(
        """
        import os
        from tempfile import TemporaryDirectory
        from pathlib import Path

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.chdir(%r)

        import pytest
        from PySide6 import QtCore, QtWidgets

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


def test_main_window_cleanup_does_not_accumulate_top_level_windows(qt_app) -> None:
    app = qt_app
    baseline_count = _count_top_level_main_windows(app)

    with TemporaryDirectory(prefix="kindred-main-window-cleanup-") as tmp_dir:
        monkeypatch = pytest.MonkeyPatch()
        tmp_root = Path(tmp_dir)
        try:
            test_conftest._clear_test_qsettings()
            test_conftest._patch_main_window_test_environment(monkeypatch, tmp_root)

            def run_cycle() -> None:
                window = MainWindow()
                test_conftest._cleanup_main_window(window)

            run_cycle()
            gc.collect()
            for _ in range(10):
                app.processEvents()
            assert _count_top_level_main_windows(app) == baseline_count
            assert _retired_main_window_count() <= 1
            assert _retired_main_window_dock_count() == 0
            assert _retired_main_window_shell_shape() == (True, 0, 0)

            run_cycle()
            gc.collect()
            for _ in range(10):
                app.processEvents()
            assert _count_top_level_main_windows(app) == baseline_count
            assert _retired_main_window_count() <= 1
            assert _retired_main_window_dock_count() == 0
            assert _retired_main_window_shell_shape() == (True, 0, 0)
        finally:
            monkeypatch.undo()


def test_main_window_fixture_cleanup_coexists_with_qtbot(qtbot, main_window) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    baseline_count = _count_top_level_main_windows(app)

    qtbot.addWidget(main_window)
    gc.collect()
    for _ in range(10):
        app.processEvents()

    assert _count_top_level_main_windows(app) >= baseline_count

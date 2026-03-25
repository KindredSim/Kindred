from __future__ import annotations

import builtins
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.unit]


def test_authoritative_launcher_orchestrates_startup_helpers_in_order(monkeypatch) -> None:
    import kindred.__main__ as entry

    events: list[str] = []
    app_state = {"created": False}

    class _FakeApplication:
        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy) -> None:
            events.append(f"dpi:{policy}")

        def __init__(self, argv) -> None:
            _ = argv
            app_state["created"] = True
            events.append("qapp")

        def setStyle(self, style: str) -> None:
            events.append(f"style:{style}")

        def setWindowIcon(self, _icon) -> None:
            events.append("icon")

        def exec(self) -> int:
            events.append("exec")
            return 23

    class _FakeCoreApplication:
        @staticmethod
        def setOrganizationName(name: str) -> None:
            events.append(f"org:{name}")

        @staticmethod
        def setApplicationName(name: str) -> None:
            events.append(f"app:{name}")

    class _FakeQtCore:
        class Qt:
            class HighDpiScaleFactorRoundingPolicy:
                PassThrough = "pass-through"

        QCoreApplication = _FakeCoreApplication

    class _FakeWindow:
        def show(self) -> None:
            events.append("show")

    class _FakeIcon:
        def isNull(self) -> bool:
            return False

    def _cleanup() -> None:
        events.append("cleanup")

    monkeypatch.setattr(entry, "setup_logging", lambda level=None: events.append(f"logging:{level}"))
    monkeypatch.setattr(entry.sys, "argv", ["kindred"])
    monkeypatch.setattr(
        entry.gui_startup,
        "ensure_qt_modules",
        lambda: (_FakeQtCore, SimpleNamespace(QApplication=_FakeApplication), object()),
    )
    monkeypatch.setattr(entry.gui_startup, "startup_debug_enabled", lambda: False)
    monkeypatch.setattr(
        entry.gui_startup,
        "enable_startup_diagnostics",
        lambda *, enabled: _cleanup,
    )
    monkeypatch.setattr(
        entry.gui_startup,
        "apply_pre_qapplication_startup",
        lambda *, startup_debug: events.append(f"pre:{startup_debug}:{app_state['created']}") or True,
    )
    monkeypatch.setattr(
        entry.gui_startup,
        "apply_post_qapplication_startup",
        lambda *, startup_debug: events.append(f"post:{startup_debug}:{app_state['created']}"),
    )
    monkeypatch.setattr(
        entry.gui_startup,
        "construct_main_window",
        lambda *, window_factory, startup_debug: events.append(
            f"construct:{startup_debug}:{app_state['created']}"
        )
        or window_factory(),
    )
    monkeypatch.setattr(entry.gui_startup, "load_app_icon", lambda q_icon_type: _FakeIcon())
    monkeypatch.setitem(sys.modules, "kindred.gui.main_window", SimpleNamespace(MainWindow=_FakeWindow))

    rc = entry.main()

    assert rc == 23
    assert events.index("logging:None") < events.index("pre:False:False") < events.index("qapp")
    assert events.index("qapp") < events.index("post:False:True") < events.index("construct:False:True")
    assert events.index("construct:False:True") < events.index("show") < events.index("exec")
    assert events[-1] == "cleanup"


def test_construct_main_window_uses_stderr_filter_when_enabled(monkeypatch) -> None:
    import kindred.gui.startup as startup

    events: list[str] = []

    class _FakeFilter:
        def __init__(self, filter_text: str) -> None:
            events.append(f"filter:{filter_text}")

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            _ = (exc_type, exc_val, exc_tb)
            events.append("exit")
            return False

    monkeypatch.setattr(startup, "should_redirect_stderr", lambda *, startup_debug: True)
    monkeypatch.setattr(startup, "StartupStderrFilter", _FakeFilter)

    result = startup.construct_main_window(
        window_factory=lambda: events.append("window") or object(),
        startup_debug=False,
    )

    assert result is not None
    assert events == [
        "filter:QObject::connect(QStyleHints, QStyleHints)",
        "enter",
        "window",
        "exit",
    ]


def test_authoritative_launcher_applies_pre_startup_before_importing_main_window(monkeypatch) -> None:
    import kindred.__main__ as entry

    events: list[str] = []
    real_import = builtins.__import__

    class _FakeApplication:
        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(_policy) -> None:
            return None

        def __init__(self, argv) -> None:
            _ = argv

        def setStyle(self, _style: str) -> None:
            return None

        def setWindowIcon(self, _icon) -> None:
            return None

        def exec(self) -> int:
            return 0

    class _FakeCoreApplication:
        @staticmethod
        def setOrganizationName(_name: str) -> None:
            return None

        @staticmethod
        def setApplicationName(_name: str) -> None:
            return None

    class _FakeQtCore:
        class Qt:
            class HighDpiScaleFactorRoundingPolicy:
                PassThrough = "pass-through"

        QCoreApplication = _FakeCoreApplication

    class _FakeWindow:
        def show(self) -> None:
            return None

    class _FakeIcon:
        def isNull(self) -> bool:
            return True

    fake_main_window_module = ModuleType("kindred.gui.main_window")
    fake_main_window_module.MainWindow = _FakeWindow

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kindred.gui.main_window":
            events.append("import_main_window")
            return fake_main_window_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(entry, "setup_logging", lambda level=None: None)
    monkeypatch.setattr(entry.sys, "argv", ["kindred"])
    monkeypatch.delitem(sys.modules, "kindred.gui.main_window", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(
        entry.gui_startup,
        "ensure_qt_modules",
        lambda: (_FakeQtCore, SimpleNamespace(QApplication=_FakeApplication), object()),
    )
    monkeypatch.setattr(entry.gui_startup, "startup_debug_enabled", lambda: False)
    monkeypatch.setattr(entry.gui_startup, "enable_startup_diagnostics", lambda *, enabled: lambda: None)
    monkeypatch.setattr(
        entry.gui_startup,
        "apply_pre_qapplication_startup",
        lambda *, startup_debug: events.append("pre"),
    )
    monkeypatch.setattr(entry.gui_startup, "log_plot_backend_startup", lambda *, startup_debug: None)
    monkeypatch.setattr(entry.gui_startup, "apply_post_qapplication_startup", lambda *, startup_debug: None)
    monkeypatch.setattr(
        entry.gui_startup,
        "construct_main_window",
        lambda *, window_factory, startup_debug: window_factory(),
    )
    monkeypatch.setattr(entry.gui_startup, "load_app_icon", lambda q_icon_type: _FakeIcon())

    assert entry.main() == 0
    assert events == ["pre", "import_main_window"]

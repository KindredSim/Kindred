from __future__ import annotations

import faulthandler
import hashlib
import logging
import os
import signal
import sys
import threading
from typing import Any, Callable, Mapping, Optional

from kindred.io.resources import get_resource_path

logger = logging.getLogger(__name__)

_STARTUP_BEST_EFFORT_FAILURES: set[str] = set()

QtCore: Any = None
QtWidgets: Any = None
QIcon: Any = None


def record_startup_best_effort_failure(key: str) -> None:
    _STARTUP_BEST_EFFORT_FAILURES.add(str(key))


def _import_qt():
    try:
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtGui import QIcon
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySide6 is required to launch the Kindred GUI. Install Kindred or install the pinned runtime dependencies from requirements.txt."
        ) from exc
    return QtCore, QtWidgets, QIcon


def ensure_qt_modules():
    qt_core = globals().get("QtCore")
    qt_widgets = globals().get("QtWidgets")
    qt_icon = globals().get("QIcon")
    if qt_core is None or qt_widgets is None or qt_icon is None:
        qt_core, qt_widgets, qt_icon = _import_qt()
        globals()["QtCore"] = qt_core
        globals()["QtWidgets"] = qt_widgets
        globals()["QIcon"] = qt_icon
    return qt_core, qt_widgets, qt_icon


def ensure_entropy_available() -> None:
    """
    Ensure `os.urandom` is available in entropy-starved environments.

    Some sandboxed runners do not provide `/dev/urandom` (or equivalent). Python
    startup can still succeed with `PYTHONHASHSEED=0`, but downstream imports
    (notably NumPy) may call `os.urandom` and fail. This shim patches `os.urandom`
    only when it is unavailable, using a deterministic non-cryptographic fallback.
    """
    try:
        os.urandom(1)
    except NotImplementedError as exc:
        record_startup_best_effort_failure("entropy.os_urandom_not_implemented")
        logger.warning(
            "System lacks entropy; using deterministic os.urandom fallback (non-cryptographic): %s",
            exc,
            exc_info=True,
        )
    else:
        return

    lock = threading.Lock()
    counter = {"n": 0}

    def _deterministic_urandom(n: int) -> bytes:
        try:
            size = int(n)
        except Exception:
            size = 0
        if size <= 0:
            return b""

        out = bytearray()
        while len(out) < size:
            with lock:
                counter["n"] += 1
                token = counter["n"]
            out.extend(
                hashlib.blake2b(
                    f"kindred-deterministic-urandom:{token}".encode("utf-8"),
                    digest_size=32,
                ).digest()
            )
        return bytes(out[:size])

    os.urandom = _deterministic_urandom  # type: ignore[assignment]
    try:
        import random as _random

        _random._urandom = _deterministic_urandom  # type: ignore[attr-defined]
    except Exception as exc:
        record_startup_best_effort_failure("entropy.patch_random_urandom")
        logger.debug("Failed to patch random._urandom fallback: %s", exc, exc_info=True)


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def startup_debug_enabled() -> bool:
    """Return True when verbose startup diagnostics are explicitly enabled."""
    return _env_truthy("KINDRED_DEBUG_STARTUP")


def detect_wsl_environment(
    *,
    environ: Optional[Mapping[str, str]] = None,
    proc_version_path: str = "/proc/version",
) -> bool:
    """Return True when running under Windows Subsystem for Linux."""
    env = os.environ if environ is None else environ
    if str(env.get("WSL_INTEROP", "")).strip():
        return True
    if str(env.get("WSL_DISTRO_NAME", "")).strip():
        return True
    try:
        with open(proc_version_path, "r", encoding="utf-8", errors="ignore") as handle:
            version_text = handle.read().lower()
    except OSError:
        version_text = ""
    return "microsoft" in version_text


def _software_opengl_attribute():
    qt_core, _qt_widgets, _qt_icon = ensure_qt_modules()
    app_attr = getattr(qt_core.Qt, "ApplicationAttribute", None)
    if app_attr is not None and hasattr(app_attr, "AA_UseSoftwareOpenGL"):
        return app_attr.AA_UseSoftwareOpenGL
    return getattr(qt_core.Qt, "AA_UseSoftwareOpenGL")


def apply_qt_startup_workarounds(
    *,
    environ: Optional[Mapping[str, str]] = None,
    set_qt_attribute: Optional[Callable[[object, bool], None]] = None,
) -> bool:
    """
    Apply startup workarounds that must run before QApplication construction.

    KINDRED_QT_OPENGL controls behavior:
    - software: always force software OpenGL
    - default : never force software OpenGL
    - auto    : force software OpenGL only on WSL (default)
    """
    env = os.environ if environ is None else environ
    raw_mode = str(env.get("KINDRED_QT_OPENGL", "auto")).strip().lower()
    mode = raw_mode if raw_mode in {"auto", "software", "default"} else "auto"
    is_wsl = detect_wsl_environment(environ=env)

    apply_software = False
    reason = ""
    if mode == "software":
        apply_software = True
        reason = "user override"
    elif mode == "auto" and is_wsl:
        apply_software = True
        reason = "WSL detected"

    if not apply_software:
        return False

    if hasattr(env, "setdefault"):
        env.setdefault("QT_OPENGL", "software")
    qt_core, _qt_widgets, _qt_icon = ensure_qt_modules()
    setter = set_qt_attribute or qt_core.QCoreApplication.setAttribute
    setter(_software_opengl_attribute(), True)
    logger.warning("Applying Qt startup workaround: software OpenGL (reason: %s)", reason)
    return True


def apply_pre_qapplication_startup(*, startup_debug: bool) -> bool:
    ensure_entropy_available()
    workaround_applied = apply_qt_startup_workarounds()
    if startup_debug:
        logger.warning("Startup phase: Qt startup workarounds applied=%s", bool(workaround_applied))
    return workaround_applied


def should_redirect_stderr(*, startup_debug: bool) -> bool:
    """
    Return whether startup should wrap MainWindow init in stderr fd redirection.

    Disabled by default because fd-level redirection can hide startup failures and
    can block on some systems. Opt-in via KINDRED_ENABLE_STARTUP_STDERR_FILTER=1.
    """
    if bool(startup_debug):
        return False
    return _env_truthy("KINDRED_ENABLE_STARTUP_STDERR_FILTER")


class StartupStderrFilter:
    """
    Filter specific Qt warnings from stderr using fd-level redirection.

    This is opt-in only (see should_redirect_stderr) to avoid obscuring startup
    failures in normal runs.
    """

    def __init__(self, filter_text: str):
        self.filter_text = str(filter_text)
        self._original_stderr_fd: int | None = None
        self._temp_file = None
        self._best_effort_failures: set[str] = set()

    def __enter__(self):
        import tempfile

        self._original_stderr_fd = os.dup(2)
        self._temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".stderr")
        os.dup2(self._temp_file.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        temp_file = self._temp_file
        original_fd = self._original_stderr_fd

        if temp_file is None or original_fd is None:
            return False

        try:
            try:
                temp_file.flush()
            except Exception as exc:
                self._best_effort_failures.add("stderr_filter.flush")
                logger.debug("StartupStderrFilter: failed to flush temp stderr: %s", exc, exc_info=True)
            os.dup2(original_fd, 2)
            os.close(original_fd)

            try:
                temp_file.seek(0)
                for line in temp_file:
                    if self.filter_text not in line:
                        sys.stderr.write(line)
            except Exception as exc:
                self._best_effort_failures.add("stderr_filter.replay")
                logger.debug("StartupStderrFilter: failed to replay stderr: %s", exc, exc_info=True)
        finally:
            tmp_name = getattr(temp_file, "name", None)
            try:
                temp_file.close()
            except Exception as exc:
                self._best_effort_failures.add("stderr_filter.close")
                logger.debug("StartupStderrFilter: failed to close temp stderr: %s", exc, exc_info=True)
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except Exception as exc:
                    self._best_effort_failures.add("stderr_filter.unlink")
                    logger.debug(
                        "StartupStderrFilter: failed to remove temp stderr %s: %s",
                        tmp_name,
                        exc,
                        exc_info=True,
                    )
        return False


def enable_startup_diagnostics(*, enabled: bool):
    """
    Enable optional faulthandler-based startup diagnostics.

    Returns a cleanup callable.
    """
    if not bool(enabled):
        return lambda: None

    logger.warning("KINDRED_DEBUG_STARTUP=1 active: enabling startup diagnostics")

    try:
        faulthandler.enable(all_threads=True)
    except Exception as exc:
        record_startup_best_effort_failure("startup_diagnostics.enable")
        logger.debug("Startup diagnostics: faulthandler.enable failed: %s", exc, exc_info=True)

    try:
        faulthandler.dump_traceback_later(10, repeat=True)
    except Exception as exc:
        record_startup_best_effort_failure("startup_diagnostics.dump_traceback_later")
        logger.debug("Startup diagnostics: dump_traceback_later failed: %s", exc, exc_info=True)

    sigusr2_registered = False
    sigusr2 = getattr(signal, "SIGUSR2", None)
    if sigusr2 is not None:
        try:
            faulthandler.register(sigusr2, all_threads=True)
            sigusr2_registered = True
            logger.warning("Startup diagnostics: send SIGUSR2 to dump Python stacks")
        except Exception:
            sigusr2_registered = False

    def _cleanup() -> None:
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception as exc:
            record_startup_best_effort_failure("startup_diagnostics.cancel_dump_traceback_later")
            logger.debug("Startup diagnostics: cancel_dump_traceback_later failed: %s", exc, exc_info=True)
        if sigusr2_registered and sigusr2 is not None:
            try:
                faulthandler.unregister(sigusr2)
            except Exception as exc:
                record_startup_best_effort_failure("startup_diagnostics.unregister_sigusr2")
                logger.debug("Startup diagnostics: unregister SIGUSR2 failed: %s", exc, exc_info=True)

    return _cleanup


def log_plot_backend_startup(*, startup_debug: bool) -> None:
    from kindred.gui.plot_config import log_backend_info

    log_backend_info()
    if startup_debug:
        logger.warning("Startup phase: backend info logged")


def apply_post_qapplication_startup(*, startup_debug: bool) -> None:
    from kindred.gui.plot_config import (
        fix_pyqtgraph_csv_exporter_encoding,
        fix_pyqtgraph_stylehints_warning,
    )

    fix_pyqtgraph_stylehints_warning()
    if startup_debug:
        logger.warning("Startup phase: style-hints patch applied")

    fix_pyqtgraph_csv_exporter_encoding()


def load_app_icon(q_icon_type: Any | None = None):
    """Return the application icon if available, otherwise None."""
    q_icon_cls = q_icon_type
    if q_icon_cls is None:
        _qt_core, _qt_widgets, q_icon_cls = ensure_qt_modules()
    try:
        return q_icon_cls(str(get_resource_path("assets/kindred.ico")))
    except FileNotFoundError:
        return None


def construct_main_window(
    *,
    window_factory: Callable[[], Any],
    startup_debug: bool,
    filter_text: str = "QObject::connect(QStyleHints, QStyleHints)",
):
    use_filter = should_redirect_stderr(startup_debug=startup_debug)
    if startup_debug:
        logger.warning("Startup phase: creating MainWindow (stderr filter=%s)", use_filter)
    if use_filter:
        try:
            with StartupStderrFilter(filter_text):
                return window_factory()
        except (OSError, PermissionError):
            return window_factory()
    return window_factory()

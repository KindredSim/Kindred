from __future__ import annotations

import faulthandler
import logging
import os
import signal
from typing import Callable, Mapping

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def startup_debug_enabled() -> bool:
    """Return True when verbose startup diagnostics are explicitly enabled."""
    return _env_truthy("KINDRED_DEBUG_STARTUP")


def detect_wsl_environment(
    *,
    environ: Mapping[str, str] | None = None,
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


def _software_opengl_attribute(qt_core: object):
    return qt_core.Qt.ApplicationAttribute.AA_UseSoftwareOpenGL


def apply_qt_startup_workarounds(
    *,
    qt_core: object,
    environ: Mapping[str, str] | None = None,
    set_qt_attribute: Callable[[object, bool], None] | None = None,
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
    setter = set_qt_attribute or qt_core.QCoreApplication.setAttribute
    setter(_software_opengl_attribute(qt_core), True)
    logger.warning("Applying Qt startup workaround: software OpenGL (reason: %s)", reason)
    return True


def apply_pre_qapplication_startup(*, qt_core: object, startup_debug: bool) -> bool:
    workaround_applied = apply_qt_startup_workarounds(qt_core=qt_core)
    if startup_debug:
        logger.warning("Startup phase: Qt startup workarounds applied=%s", bool(workaround_applied))
    return workaround_applied


def enable_startup_diagnostics(*, enabled: bool) -> Callable[[], None]:
    """Enable optional faulthandler diagnostics and return their cleanup."""
    if not enabled:
        return lambda: None

    logger.warning("KINDRED_DEBUG_STARTUP=1 active: enabling startup diagnostics")
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(10, repeat=True)

    sigusr2 = getattr(signal, "SIGUSR2", None)
    if sigusr2 is not None:
        faulthandler.register(sigusr2, all_threads=True)
        logger.warning("Startup diagnostics: send SIGUSR2 to dump Python stacks")

    def _cleanup() -> None:
        faulthandler.cancel_dump_traceback_later()
        if sigusr2 is not None:
            faulthandler.unregister(sigusr2)

    return _cleanup


def log_plot_backend_startup(*, startup_debug: bool) -> None:
    from kindred.gui.plot_config import log_backend_info

    log_backend_info()
    if startup_debug:
        logger.warning("Startup phase: backend info logged")


def apply_post_qapplication_startup(*, startup_debug: bool) -> None:
    from kindred.gui.plot_config import fix_pyqtgraph_stylehints_warning

    fix_pyqtgraph_stylehints_warning()
    if startup_debug:
        logger.warning("Startup phase: style-hints patch applied")


def load_app_icon(q_icon_type: object):
    """Return the application icon if available, otherwise None."""
    from kindred.io.resources import get_resource_path

    try:
        return q_icon_type(str(get_resource_path("assets/kindred.ico")))
    except FileNotFoundError:
        return None

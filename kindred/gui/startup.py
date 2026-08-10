from __future__ import annotations

import logging
import os
from typing import Callable, Mapping

from kindred.gui.plot_config import require_pyqtgraph

logger = logging.getLogger(__name__)


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
    if "KINDRED_QT_OPENGL" in env:
        raw_mode = str(env["KINDRED_QT_OPENGL"])
        mode = raw_mode.strip().lower()
        if mode not in {"auto", "software", "default"}:
            raise ValueError(
                f"Invalid KINDRED_QT_OPENGL value {raw_mode!r}; "
                "expected one of: auto, software, default."
            )
    else:
        mode = "auto"
    require_pyqtgraph(context="Kindred GUI")
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


def load_app_icon(q_icon_type: object):
    """Return the application icon if available, otherwise None."""
    from kindred.io.resources import get_resource_path

    try:
        return q_icon_type(str(get_resource_path("assets/kindred.ico")))
    except FileNotFoundError:
        return None

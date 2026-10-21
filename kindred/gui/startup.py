from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


logger = logging.getLogger(__name__)

# Every Kindred process runs BLAS and OpenMP on one thread. Each extra OpenBLAS
# thread commits a buffer on Windows, and Kindred's small solves gain nothing.
SCIENTIFIC_THREAD_LIMITS = {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def detect_wsl_environment() -> bool:
    """Return True when running under Windows Subsystem for Linux."""
    if str(os.environ.get("WSL_INTEROP", "")).strip():
        return True
    if str(os.environ.get("WSL_DISTRO_NAME", "")).strip():
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as handle:
            version_text = handle.read().lower()
    except OSError:
        version_text = ""
    return "microsoft" in version_text


def _software_opengl_attribute(qt_core: object):
    return qt_core.Qt.ApplicationAttribute.AA_UseSoftwareOpenGL


def apply_qt_startup_workarounds(*, qt_core: object) -> bool:
    """
    Apply startup workarounds that must run before QApplication construction.

    KINDRED_QT_OPENGL controls behavior:
    - software: always force software OpenGL
    - default : never force software OpenGL
    - auto    : force software OpenGL only on WSL (default)
    """
    env = os.environ
    if "KINDRED_QT_OPENGL" in env:
        raw_mode = str(env["KINDRED_QT_OPENGL"])
        mode = raw_mode.strip().lower()
        if mode not in {"auto", "software", "default"}:
            raise ValueError(
                f"Invalid KINDRED_QT_OPENGL value {raw_mode!r}. "
                "Expected one of: auto, software, default."
            )
    else:
        mode = "auto"
    is_wsl = detect_wsl_environment()

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

    env.setdefault("QT_OPENGL", "software")
    qt_core.QCoreApplication.setAttribute(_software_opengl_attribute(qt_core), True)
    logger.warning("Applying Qt startup workaround: software OpenGL (reason: %s)", reason)
    return True


def physical_core_count() -> int:
    """Return the machine's physical processor cores; Kindred asks once, at startup."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        relation_processor_core = 0
        size = ctypes.c_ulong(0)
        # The first call only reports the buffer size the records need.
        kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(
            relation_processor_core, buffer, ctypes.byref(size)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # One variable-length record per core; each gives its own size after
        # its relationship.
        cores = offset = 0
        while offset < size.value:
            offset += int.from_bytes(buffer.raw[offset + 4:offset + 8], "little")
            cores += 1
        return cores
    if sys.platform.startswith("linux"):
        topology = Path("/sys/devices/system/cpu")
        return len({
            ((path / "physical_package_id").read_text().strip(), (path / "core_id").read_text().strip())
            for path in topology.glob("cpu[0-9]*/topology")
        })
    raise RuntimeError(f"Counting processor cores is not supported on {sys.platform}.")


def load_app_icon(q_icon_type: object):
    """Return the application icon; a missing file gives a null icon."""
    from kindred.io.resources import get_resource_path

    return q_icon_type(str(get_resource_path("assets/kindred.ico")))

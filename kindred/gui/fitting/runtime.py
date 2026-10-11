"""Installed Windows fitting runtime (stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import sysconfig


def scientific_environment() -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if key.upper() not in {"PYTHONHOME", "PYTHONPATH", "PYTHON_GIL"}
    }
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        env[name] = "1"
    return env


def fitting_process_command() -> tuple[list[str], dict[str, str], Path]:
    if sys.platform != "win32":
        raise RuntimeError("Fitting requires the installed native Windows scientific runtime.")
    # Keep the complete interpreter two levels below the GUI prefix so CPython
    # cannot mistake the GUI's parent pyvenv.cfg for its own environment.
    directory = Path(sys.prefix).resolve() / "kindred-fitting-runtime" / "python"
    executable = directory / "python.exe"
    if not executable.is_file():
        raise FileNotFoundError(
            f"Missing fitting runtime: {executable}. Provision the dedicated Windows "
            "Python 3.15t runtime with matching Kindred code and scientific dependencies."
        )
    return (
        [str(executable), "-E", "-s", "-m", "kindred.gui.fitting.process"],
        scientific_environment(),
        directory,
    )


def check_scientific_runtime() -> None:
    """Admit the actual interpreter and imported native dependency state."""
    if sys.platform != "win32" or sys.version_info[:2] != (3, 15):
        raise RuntimeError("Fitting requires native Windows Python 3.15t.")
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("The fitting interpreter is not a free-threaded build.")
    if sys._is_gil_enabled():
        raise RuntimeError("The fitting runtime enabled the GIL.")
    forbidden = [
        name for name in sys.modules
        if name.lower().startswith(("pyside", "shiboken", "pyqtgraph"))
        or name == "kindred.gui.fitting.worker"
    ]
    if forbidden:
        raise RuntimeError(f"GUI modules entered the fitting runtime: {forbidden}.")

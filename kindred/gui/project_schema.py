"""Single source of truth for fresh-project state.

Every project-scoped key and its factory-fresh default lives here.
The QSETTINGS_KEY_MAP bridges dual-persisted keys between PROJECT_DEFAULTS
names and their QSettings paths, enabling three-tier precedence:
factory defaults < user preferences (QSettings) < document overrides (.kin).
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from kindred.core.runtime_defaults import (
    LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT,
    MAX_PARALLEL_BATCH_WORKERS_DEFAULT,
    MAX_PARALLEL_WORKERS_CEILING,
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME
from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER

if TYPE_CHECKING:
    from PySide6.QtCore import QSettings

__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_DEFAULTS",
    "QSETTINGS_KEY_MAP",
    "FITTING_DEFAULTS_KEYS",
    "get_default_project_payload",
    "get_user_preference_payload",
]

PROJECT_SCHEMA_VERSION: int = 4

PROJECT_DEFAULTS: dict[str, object] = {
    "mechanism": "",
    "notes": "",
    "state_network": "",
    "batch_initial_conditions": {},
    "solver": DEFAULT_SOLVER_NAME,
    "rtol": 1e-6,
    "atol": 1e-12,
    "use_sparse_jacobian": USE_SPARSE_JACOBIAN_DEFAULT,
    "wegscheider_cyclicity_enabled": WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    "max_parallel_batch_workers": MAX_PARALLEL_BATCH_WORKERS_DEFAULT,
    "limit_blas_threads_per_worker": LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT,
    "temperature_K": 298.15,
    "simulation_time": "10.0",
    "num_points": 100,
    "fitting_method": "trf",
    "fitting_max_nfev": 1000,
    "fitting_ftol": 1e-10,
    "fitting_xtol": 1e-10,
    "fitting_parallel_enabled": False,
    "fitting_use_seed": True,
    "fitting_seed": 42,
    "fitting_solver": FITTING_DEFAULT_SOLVER,
    "fitting_rtol": 1e-6,
    "fitting_atol": 1e-12,
}


QSETTINGS_KEY_MAP: dict[str, str] = {
    "solver": "simulation/solver",
    "rtol": "simulation/rtol",
    "atol": "simulation/atol",
    "use_sparse_jacobian": "simulation/use_sparse_jacobian",
    "wegscheider_cyclicity_enabled": "simulation/wegscheider_cyclicity_enabled",
    "max_parallel_batch_workers": "simulation/max_parallel_batch_workers",
    "limit_blas_threads_per_worker": "simulation/limit_blas_threads_per_worker",
    "temperature_K": "simulation/temperature",
    "simulation_time": "simulation/time",
    "num_points": "simulation/points",
    "fitting_method": "fitting/method",
    "fitting_max_nfev": "fitting/max_nfev",
    "fitting_ftol": "fitting/ftol",
    "fitting_xtol": "fitting/xtol",
    "fitting_parallel_enabled": "fitting/parallel_enabled",
    "fitting_use_seed": "fitting/use_seed",
    "fitting_seed": "fitting/seed",
    "fitting_solver": "fitting/solver",
    "fitting_rtol": "fitting/rtol",
    "fitting_atol": "fitting/atol",
}
"""Maps each dual-persisted PROJECT_DEFAULTS key to its QSettings path."""

FITTING_DEFAULTS_KEYS: tuple[str, ...] = tuple(
    k for k in QSETTINGS_KEY_MAP if k.startswith("fitting_")
)


def get_default_project_payload() -> dict[str, object]:
    """Return a fresh mutable copy of the canonical empty-project payload.

    Callers must use this function instead of PROJECT_DEFAULTS directly
    to avoid mutating the shared module-level dict.
    """
    return copy.deepcopy(PROJECT_DEFAULTS)


def get_user_preference_payload(settings: QSettings) -> dict[str, object]:
    """Build a complete payload using QSettings values for dual-persisted keys.

    For each key in QSETTINGS_KEY_MAP the QSettings value is read with
    proper type coercion; keys absent from QSettings fall back to
    PROJECT_DEFAULTS.  Project-only keys always use factory defaults.
    """
    result = get_default_project_payload()

    # Read each dual-persisted key from QSettings with type-safe coercion.
    for key, qs_key in QSETTINGS_KEY_MAP.items():
        default = PROJECT_DEFAULTS[key]
        try:
            if isinstance(default, bool):
                result[key] = settings.value(qs_key, default, type=bool)
            elif isinstance(default, float):
                result[key] = settings.value(qs_key, default, type=float)
            elif isinstance(default, int):
                raw = settings.value(qs_key, default, type=int)
                if key == "max_parallel_batch_workers":
                    raw = min(
                        int(MAX_PARALLEL_WORKERS_CEILING),
                        max(1, raw),
                    )
                result[key] = raw
            elif key == "simulation_time":
                raw = settings.value(qs_key, default)
                if isinstance(raw, (int, float)):
                    result[key] = f"{float(raw):g}"
                else:
                    result[key] = str(raw) if raw else str(default)
            else:
                raw = settings.value(qs_key, default)
                result[key] = str(raw) if raw else str(default)
        except (TypeError, ValueError):
            result[key] = default

    return result

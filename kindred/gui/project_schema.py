"""Single source of truth for fresh-project state.

Every project-scoped key and its factory-fresh default lives here.
The QSETTINGS_KEY_MAP bridges dual-persisted keys between PROJECT_DEFAULTS
names and their QSettings paths, enabling three-tier precedence:
factory defaults < user preferences (QSettings) < document overrides (.kin).
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

if TYPE_CHECKING:
    from PySide6.QtCore import QSettings

__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_DEFAULTS",
    "QSETTINGS_KEY_MAP",
    "get_default_project_payload",
    "get_user_preference_payload",
]

PROJECT_SCHEMA_VERSION: int = 3

PROJECT_DEFAULTS: dict[str, object] = {
    "mechanism": "",
    "notes": "",
    "state_network": "",
    "batch_initial_conditions": {},
    "solver": DEFAULT_SOLVER_NAME,
    "rtol": 1e-6,
    "atol": 1e-12,
    "use_sparse_jacobian": False,
    "wegscheider_cyclicity_enabled": False,
    "max_parallel_batch_workers": 12,
    "limit_blas_threads_per_worker": True,
    "temperature_K": 298.15,
    "simulation_time": "10.0",
    "num_points": 100,
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
}
"""Maps each dual-persisted PROJECT_DEFAULTS key to its QSettings path."""


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
                    raw = max(1, raw)
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

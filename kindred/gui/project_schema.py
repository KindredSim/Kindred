"""Single source of truth for fresh-project state.

Every project-scoped key and its factory-fresh default lives here.
The QSETTINGS_KEY_MAP bridges dual-persisted keys between PROJECT_DEFAULTS
names and their QSettings paths, enabling three-tier precedence:
factory defaults < user preferences (QSettings) < document overrides (.kin).
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.runtime_defaults import (
    BATCH_RUNTIME_LANE_BUDGET_DEFAULT,
    LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT,
    MAX_PARALLEL_BATCH_WORKERS_DEFAULT,
    MAX_PARALLEL_WORKERS_CEILING,
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.exceptions import SolverError
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name
from kindred.gui.fitting.constants import (
    FITTING_DEFAULT_SOLVER,
    FITTING_MAX_NFEV_RANGE,
    FITTING_METHODS,
    FITTING_SCIENTIFIC_VALUE_MAX,
    FITTING_SEED_RANGE,
    FITTING_SOLVERS,
)

if TYPE_CHECKING:
    from PySide6.QtCore import QSettings

__all__ = [
    "PROJECT_DEFAULTS",
    "QSETTINGS_KEY_MAP",
    "FITTING_DEFAULTS_KEYS",
    "get_default_project_payload",
    "get_user_preference_payload",
    "sanitize_simulation_time_preference",
    "validate_project_payload",
]

PROJECT_DEFAULTS: dict[str, object] = {
    "mechanism_source": {
        "reactions_text": "",
        "state_network_dsl": "",
    },
    "notes": "",
    "batch_initial_conditions": {},
    "solver": DEFAULT_SOLVER_NAME,
    "rtol": 1e-6,
    "atol": 1e-12,
    "use_sparse_jacobian": USE_SPARSE_JACOBIAN_DEFAULT,
    "wegscheider_cyclicity_enabled": WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
    "max_parallel_batch_workers": MAX_PARALLEL_BATCH_WORKERS_DEFAULT,
    "batch_runtime_lane_budget": BATCH_RUNTIME_LANE_BUDGET_DEFAULT,
    "limit_blas_threads_per_worker": LIMIT_BLAS_THREADS_PER_WORKER_DEFAULT,
    "temperature_K": 298.15,
    "simulation_time": "10.0",
    "num_points": 100,
    "fitting_method": "trf",
    "fitting_max_nfev": 1000,
    "fitting_ftol": 1e-10,
    "fitting_xtol": 1e-10,
    "fitting_use_seed": True,
    "fitting_seed": 42,
    "fitting_solver": FITTING_DEFAULT_SOLVER,
    "fitting_rtol": 1e-6,
    "fitting_atol": 1e-12,
}

SIMULATION_NUM_POINTS_RANGE = (10, 100_000)
SIMULATION_TEMPERATURE_K_RANGE = (0.1, 10_000.0)
SIMULATION_TIME_RANGE = (0.0, FITTING_SCIENTIFIC_VALUE_MAX)


QSETTINGS_KEY_MAP: dict[str, str] = {
    "solver": "simulation/solver",
    "rtol": "simulation/rtol",
    "atol": "simulation/atol",
    "use_sparse_jacobian": "simulation/use_sparse_jacobian",
    "wegscheider_cyclicity_enabled": "simulation/wegscheider_cyclicity_enabled",
    "max_parallel_batch_workers": "simulation/max_parallel_batch_workers",
    "batch_runtime_lane_budget": "simulation/batch_runtime_lane_budget",
    "limit_blas_threads_per_worker": "simulation/limit_blas_threads_per_worker",
    "temperature_K": "simulation/temperature",
    "simulation_time": "simulation/time",
    "num_points": "simulation/points",
    "fitting_method": "fitting/method",
    "fitting_max_nfev": "fitting/max_nfev",
    "fitting_ftol": "fitting/ftol",
    "fitting_xtol": "fitting/xtol",
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


_PROJECT_STRING_FIELDS = frozenset(
    {
        "version",
        "notes",
        "solver",
        "simulation_time",
        "fitting_method",
        "fitting_solver",
    }
)
_PROJECT_OPTIONAL_STRING_FIELDS = frozenset()
_PROJECT_BOOL_FIELDS = frozenset(
    {
        "use_sparse_jacobian",
        "wegscheider_cyclicity_enabled",
        "limit_blas_threads_per_worker",
        "fitting_use_seed",
    }
)
_PROJECT_INT_FIELDS = frozenset(
    {
        "num_points",
        "max_parallel_batch_workers",
        "batch_runtime_lane_budget",
        "fitting_max_nfev",
        "fitting_seed",
    }
)
_PROJECT_INT_RANGES = {
    "num_points": SIMULATION_NUM_POINTS_RANGE,
    "max_parallel_batch_workers": (1, int(MAX_PARALLEL_WORKERS_CEILING)),
    "batch_runtime_lane_budget": (1, int(MAX_PARALLEL_WORKERS_CEILING)),
    "fitting_max_nfev": FITTING_MAX_NFEV_RANGE,
    "fitting_seed": FITTING_SEED_RANGE,
}
_PROJECT_POSITIVE_NUMBER_MAXIMUMS = {
    "rtol": FITTING_SCIENTIFIC_VALUE_MAX,
    "atol": FITTING_SCIENTIFIC_VALUE_MAX,
    "temperature_K": SIMULATION_TEMPERATURE_K_RANGE[1],
    "fitting_ftol": FITTING_SCIENTIFIC_VALUE_MAX,
    "fitting_xtol": FITTING_SCIENTIFIC_VALUE_MAX,
    "fitting_rtol": FITTING_SCIENTIFIC_VALUE_MAX,
    "fitting_atol": FITTING_SCIENTIFIC_VALUE_MAX,
}
_PROJECT_POSITIVE_NUMBER_MINIMUMS = {
    "temperature_K": SIMULATION_TEMPERATURE_K_RANGE[0],
}
_PROJECT_NUMBER_FIELDS = frozenset(_PROJECT_POSITIVE_NUMBER_MAXIMUMS)
_PROJECT_MAPPING_FIELDS = frozenset({"batch_initial_conditions"})
_PROJECT_REQUIRED_FIELDS = (
    frozenset(PROJECT_DEFAULTS)
    | _PROJECT_STRING_FIELDS
    | _PROJECT_OPTIONAL_STRING_FIELDS
    | _PROJECT_BOOL_FIELDS
    | _PROJECT_INT_FIELDS
    | _PROJECT_NUMBER_FIELDS
    | _PROJECT_MAPPING_FIELDS
)
_PROJECT_FIELDS = _PROJECT_REQUIRED_FIELDS


def _require_str_field(data: Mapping[str, object], field: str) -> None:
    if field in data and not isinstance(data[field], str):
        raise TypeError(f"project payload field {field!r} must be a str.")


def _require_optional_str_field(data: Mapping[str, object], field: str) -> None:
    if field in data and data[field] is not None and not isinstance(data[field], str):
        raise TypeError(f"project payload field {field!r} must be a str or None.")


def _require_bool_field(data: Mapping[str, object], field: str) -> None:
    if field in data and type(data[field]) is not bool:
        raise TypeError(f"project payload field {field!r} must be a bool.")


def _require_int_field(data: Mapping[str, object], field: str) -> None:
    if field in data and (type(data[field]) is bool or not isinstance(data[field], int)):
        raise TypeError(f"project payload field {field!r} must be an int.")
    if field not in data:
        return
    limits = _PROJECT_INT_RANGES.get(field)
    if limits is None:
        return
    minimum, maximum = limits
    value = int(data[field])
    if value < int(minimum) or value > int(maximum):
        raise ValueError(
            f"project payload field {field!r} must be between {int(minimum)} and {int(maximum)}."
        )


def _require_number_field(data: Mapping[str, object], field: str) -> None:
    if field in data and (type(data[field]) is bool or not isinstance(data[field], Real)):
        raise TypeError(f"project payload field {field!r} must be numeric.")
    if field not in data:
        return
    value = float(data[field])
    maximum = _PROJECT_POSITIVE_NUMBER_MAXIMUMS[field]
    minimum = _PROJECT_POSITIVE_NUMBER_MINIMUMS.get(field, 0.0)
    if not math.isfinite(value) or value <= 0.0 or value < minimum or value > maximum:
        raise ValueError(
            f"project payload field {field!r} must be positive, finite, and within the current project range."
        )


def _require_mapping_field(data: Mapping[str, object], field: str) -> None:
    if field in data and not isinstance(data[field], Mapping):
        raise TypeError(f"project payload field {field!r} must be a mapping.")


def _require_simulation_time(data: Mapping[str, object]) -> None:
    raw = str(data["simulation_time"]).strip()
    if not raw:
        raise ValueError("project payload field 'simulation_time' must be positive and finite.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("project payload field 'simulation_time' must be positive and finite.") from exc
    minimum, maximum = SIMULATION_TIME_RANGE
    if not math.isfinite(value) or value <= minimum or value > maximum:
        raise ValueError("project payload field 'simulation_time' must be positive and finite.")


def _require_choice_field(
    data: Mapping[str, object],
    field: str,
    choices: tuple[str, ...],
    *,
    normalize_solver: bool = False,
) -> None:
    value = str(data[field]).strip()
    if normalize_solver:
        try:
            normalized = normalize_solver_name(value)
        except SolverError as exc:
            if value.strip().upper() == "LSODA":
                raise ValueError(
                    f"project payload field {field!r} uses unsupported saved solver 'LSODA'; "
                    f"supported solvers are {choices!r}."
                ) from exc
            raise ValueError(
                f"project payload field {field!r} uses unsupported saved solver {value!r}; "
                f"supported solvers are {choices!r}."
            ) from exc
        if str(normalized) not in choices:
            raise ValueError(f"project payload field {field!r} must be one of {choices!r}.")
        return
    if value.lower() not in {choice.lower() for choice in choices}:
        raise ValueError(f"project payload field {field!r} must be one of {choices!r}.")


def validate_project_payload(data: Mapping[str, object]) -> MechanismAuthoringSource:
    """Validate current project payload shape before any UI/session mutation."""
    if not isinstance(data, Mapping):
        raise TypeError("project payload must be a mapping.")
    unknown_fields = sorted(str(field) for field in data.keys() if field not in _PROJECT_FIELDS)
    if unknown_fields:
        fields_text = ", ".join(repr(field) for field in unknown_fields)
        raise ValueError(f"project payload has unknown field(s): {fields_text}.")
    missing_fields = sorted(str(field) for field in _PROJECT_REQUIRED_FIELDS if field not in data)
    if missing_fields:
        fields_text = ", ".join(repr(field) for field in missing_fields)
        raise ValueError(f"project payload is missing required field(s): {fields_text}.")
    if "mechanism_source" not in data:
        raise ValueError("project payload is missing required field 'mechanism_source'.")

    source = MechanismAuthoringSource.from_payload(data["mechanism_source"])

    for field in sorted(_PROJECT_STRING_FIELDS):
        _require_str_field(data, field)
    for field in sorted(_PROJECT_OPTIONAL_STRING_FIELDS):
        _require_optional_str_field(data, field)
    for field in sorted(_PROJECT_BOOL_FIELDS):
        _require_bool_field(data, field)
    for field in sorted(_PROJECT_INT_FIELDS):
        _require_int_field(data, field)
    for field in sorted(_PROJECT_NUMBER_FIELDS):
        _require_number_field(data, field)
    for field in sorted(_PROJECT_MAPPING_FIELDS):
        _require_mapping_field(data, field)
    _require_simulation_time(data)
    _require_choice_field(data, "fitting_method", FITTING_METHODS)
    _require_choice_field(data, "fitting_solver", FITTING_SOLVERS, normalize_solver=True)
    _require_choice_field(data, "solver", ("Radau", "BDF"), normalize_solver=True)

    return source


def get_default_project_payload() -> dict[str, object]:
    """Return a fresh mutable copy of the canonical empty-project payload.

    Callers must use this function instead of PROJECT_DEFAULTS directly
    to avoid mutating the shared module-level dict.
    """
    return copy.deepcopy(PROJECT_DEFAULTS)


def sanitize_simulation_time_preference(raw: object, default: object | None = None) -> str:
    fallback = PROJECT_DEFAULTS["simulation_time"] if default is None else default
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return str(fallback)
    minimum, maximum = SIMULATION_TIME_RANGE
    if not math.isfinite(value) or value <= float(minimum) or value > float(maximum):
        return str(fallback)
    return f"{float(value):g}"


def get_user_preference_payload(settings: QSettings) -> dict[str, object]:
    """Build a complete payload using QSettings values for dual-persisted keys.

    For each key in QSETTINGS_KEY_MAP the QSettings value is read with
    proper type coercion; keys absent from QSettings fall back to
    PROJECT_DEFAULTS.  Project-only keys always use factory defaults.
    """
    result = get_default_project_payload()

    def _finite_positive_setting(raw: object, default: object, key: str) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        if not math.isfinite(value) or value <= 0.0:
            return float(default)
        minimum = _PROJECT_POSITIVE_NUMBER_MINIMUMS.get(key)
        if minimum is not None and value < float(minimum):
            return float(default)
        maximum = _PROJECT_POSITIVE_NUMBER_MAXIMUMS.get(key)
        if maximum is not None and value > float(maximum):
            return float(default)
        return float(value)

    def _int_setting(raw: object, default: object, key: str) -> int:
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return int(default)
        if key in {"max_parallel_batch_workers", "batch_runtime_lane_budget"}:
            return min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, value),
            )
        limits = _PROJECT_INT_RANGES.get(key)
        if limits is None:
            return int(value)
        minimum, maximum = limits
        if value < int(minimum) or value > int(maximum):
            return int(default)
        return int(value)

    def _solver_setting(raw: object, default: object) -> str:
        try:
            return normalize_solver_name(str(raw or default))
        except Exception:
            return normalize_solver_name(str(default))

    def _choice_setting(raw: object, default: object, choices: tuple[str, ...]) -> str:
        value = str(raw or default).strip()
        by_lower = {str(choice).lower(): str(choice) for choice in choices}
        return by_lower.get(value.lower(), str(default))

    # Read each dual-persisted key from QSettings with type-safe coercion.
    for key, qs_key in QSETTINGS_KEY_MAP.items():
        default = PROJECT_DEFAULTS[key]
        try:
            if isinstance(default, bool):
                result[key] = settings.value(qs_key, default, type=bool)
            elif isinstance(default, float):
                result[key] = _finite_positive_setting(
                    settings.value(qs_key, default, type=float),
                    default,
                    key,
                )
            elif isinstance(default, int):
                result[key] = _int_setting(settings.value(qs_key, default, type=int), default, key)
            elif key == "simulation_time":
                result[key] = sanitize_simulation_time_preference(
                    settings.value(qs_key, default),
                    default,
                )
            elif key == "fitting_method":
                result[key] = _choice_setting(settings.value(qs_key, default), default, FITTING_METHODS)
            elif key in {"solver", "fitting_solver"}:
                result[key] = _solver_setting(settings.value(qs_key, default), default)
            else:
                raw = settings.value(qs_key, default)
                result[key] = str(raw) if raw else str(default)
        except (TypeError, ValueError):
            result[key] = default

    return result

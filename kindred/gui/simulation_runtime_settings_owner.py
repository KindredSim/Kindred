from __future__ import annotations

import math
from typing import Callable, Optional

from kindred.core.runtime_defaults import (
    USE_SPARSE_JACOBIAN_DEFAULT,
    WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
)
from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name
from kindred.gui.fitting.constants import FITTING_SCIENTIFIC_VALUE_MAX

RUNTIME_SETTING_UNSET = object()


def _valid_solver_tolerance(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0 or numeric > FITTING_SCIENTIFIC_VALUE_MAX:
        return None
    return float(numeric)


def resolve_solver_tolerance(
    value: object,
    *,
    field_name: str,
    current_value: object,
    default_value: object,
) -> float:
    parsed = _valid_solver_tolerance(value)
    if parsed is not None:
        return parsed
    current = _valid_solver_tolerance(current_value)
    if current is not None:
        return current
    default = _valid_solver_tolerance(default_value)
    if default is None:
        raise ValueError(f"default solver tolerance {field_name!r} is invalid")
    return default


class SimulationRuntimeSettingsOwner:
    """Owner for current persistent simulation runtime settings and UI context readers."""

    def __init__(
        self,
        *,
        current_solver: object = DEFAULT_SOLVER_NAME,
        current_rtol: object = 1e-6,
        current_atol: object = 1e-12,
        use_sparse_jacobian: object = USE_SPARSE_JACOBIAN_DEFAULT,
        wegscheider_cyclicity_enabled: object = WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT,
        temperature_getter: Callable[[], float],
        num_points_getter: Callable[[], int],
        sim_time_text_getter: Callable[[], str],
        parse_sim_time_seconds: Callable[[], float],
        dsl_global_temperature_getter: Callable[[str], Optional[float]],
    ) -> None:
        self._current_solver = str(DEFAULT_SOLVER_NAME)
        self._current_rtol = 1e-6
        self._current_atol = 1e-12
        self._use_sparse_jacobian = bool(USE_SPARSE_JACOBIAN_DEFAULT)
        self._wegscheider_cyclicity_enabled = bool(WEGSCHEIDER_CYCLICITY_ENABLED_DEFAULT)
        self._temperature_getter = temperature_getter
        self._num_points_getter = num_points_getter
        self._sim_time_text_getter = sim_time_text_getter
        self._parse_sim_time_seconds = parse_sim_time_seconds
        self._dsl_global_temperature_getter = dsl_global_temperature_getter
        self.apply_runtime_settings(
            solver=current_solver,
            rtol=current_rtol,
            atol=current_atol,
            use_sparse_jacobian=use_sparse_jacobian,
            wegscheider_cyclicity_enabled=wegscheider_cyclicity_enabled,
        )

    def apply_runtime_settings(
        self,
        *,
        solver: object = RUNTIME_SETTING_UNSET,
        rtol: object = RUNTIME_SETTING_UNSET,
        atol: object = RUNTIME_SETTING_UNSET,
        use_sparse_jacobian: object = RUNTIME_SETTING_UNSET,
        wegscheider_cyclicity_enabled: object = RUNTIME_SETTING_UNSET,
    ) -> None:
        if solver is not RUNTIME_SETTING_UNSET:
            solver_label = str(solver or DEFAULT_SOLVER_NAME).strip() or str(DEFAULT_SOLVER_NAME)
            self._current_solver = str(normalize_solver_name(solver_label))
        if rtol is not RUNTIME_SETTING_UNSET:
            self._current_rtol = resolve_solver_tolerance(
                rtol,
                field_name="rtol",
                current_value=self._current_rtol,
                default_value=1e-6,
            )
        if atol is not RUNTIME_SETTING_UNSET:
            self._current_atol = resolve_solver_tolerance(
                atol,
                field_name="atol",
                current_value=self._current_atol,
                default_value=1e-12,
            )
        if use_sparse_jacobian is not RUNTIME_SETTING_UNSET:
            self._use_sparse_jacobian = bool(use_sparse_jacobian)
        if wegscheider_cyclicity_enabled is not RUNTIME_SETTING_UNSET:
            self._wegscheider_cyclicity_enabled = bool(wegscheider_cyclicity_enabled)

    def current_runtime_settings(self) -> dict[str, object]:
        return {
            "solver": self.current_solver_name(),
            "rtol": self.current_rtol(),
            "atol": self.current_atol(),
            "use_sparse_jacobian": self.use_sparse_jacobian(),
            "wegscheider_cyclicity_enabled": self.wegscheider_cyclicity_enabled(),
        }

    def current_solver_name(self) -> str:
        return str(self._current_solver)

    def current_rtol(self) -> float:
        return float(self._current_rtol)

    def current_atol(self) -> float:
        return float(self._current_atol)

    def temperature_spinbox_value(self) -> float:
        return float(self._temperature_getter())

    def num_points_spinbox_value(self) -> int:
        return int(self._num_points_getter())

    def sim_time_spinbox_text(self) -> str:
        return str(self._sim_time_text_getter())

    def parse_sim_time_seconds(self) -> float:
        return float(self._parse_sim_time_seconds())

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]:
        value = self._dsl_global_temperature_getter(str(dsl_text))
        return float(value) if value is not None else None

    def use_sparse_jacobian(self) -> bool:
        return bool(self._use_sparse_jacobian)

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_cyclicity_enabled)

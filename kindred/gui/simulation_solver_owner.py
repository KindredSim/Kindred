from __future__ import annotations

from typing import Callable, Optional


class SimulationSolverOwner:
    """Thin Qt adapter for simulation solver controls and startup defaults."""

    def __init__(
        self,
        *,
        initial_solver_getter: Callable[[], Optional[str]],
        initial_rtol_getter: Callable[[], Optional[float]],
        initial_atol_getter: Callable[[], Optional[float]],
        temperature_getter: Callable[[], float],
        num_points_getter: Callable[[], int],
        sim_time_text_getter: Callable[[], str],
        parse_sim_time_seconds: Callable[[], float],
        dsl_global_temperature_getter: Callable[[str], Optional[float]],
        sparse_jacobian_getter: Callable[[], bool],
        wegscheider_cyclicity_getter: Callable[[], bool],
    ) -> None:
        self._initial_solver_getter = initial_solver_getter
        self._initial_rtol_getter = initial_rtol_getter
        self._initial_atol_getter = initial_atol_getter
        self._temperature_getter = temperature_getter
        self._num_points_getter = num_points_getter
        self._sim_time_text_getter = sim_time_text_getter
        self._parse_sim_time_seconds = parse_sim_time_seconds
        self._dsl_global_temperature_getter = dsl_global_temperature_getter
        self._sparse_jacobian_getter = sparse_jacobian_getter
        self._wegscheider_cyclicity_getter = wegscheider_cyclicity_getter

    def initial_solver_name(self) -> Optional[str]:
        solver = self._initial_solver_getter()
        return str(solver) if solver is not None else None

    def initial_rtol(self) -> Optional[float]:
        value = self._initial_rtol_getter()
        return float(value) if value is not None else None

    def initial_atol(self) -> Optional[float]:
        value = self._initial_atol_getter()
        return float(value) if value is not None else None

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
        return bool(self._sparse_jacobian_getter())

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_cyclicity_getter())

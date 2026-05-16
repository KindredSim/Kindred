# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Tuple

import numpy as np

from kindred.core.intervention_schedule import InterventionSchedule
from kindred.core.temperature import TemperatureScheduleProtocol

from .jacobian import JacobianConfig

DEFAULT_SOLVER_NAME = "BDF"


class ODERhsNoTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray: ...


class ODERhsWithTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


ODERhs = ODERhsNoTemp | ODERhsWithTemp

Rhs2 = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
    t_span: Tuple[float, float]
    y0: np.ndarray
    solver: str = DEFAULT_SOLVER_NAME
    rtol: float = 1e-6
    atol: float = 1e-12
    max_step: Optional[float] = None
    first_step: Optional[float] = None
    t_eval: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, float | int]] = None
    rosenbrock_jacobian: JacobianConfig = JacobianConfig()
    jacobian_func: Optional[Callable[[float, np.ndarray], np.ndarray]] = None
    jac_sparsity: Optional[Any] = None
    events: Optional[Iterable[Callable[[float, np.ndarray], float]]] = None
    event_terminal: Optional[Iterable[bool]] = None
    positivity: Optional[str] = None
    pos_indices: Optional[Iterable[int]] = None

    progress_callback: Optional[Callable[[float, float, float], None]] = None
    temperature_schedule: TemperatureScheduleProtocol | None = None
    intervention_schedule: InterventionSchedule | Mapping[str, Any] | None = None
    species_names: Optional[Tuple[str, ...]] = None
    symbolic_wegscheider_identity: Optional[Mapping[str, Any]] = None
    symbolic_jacobian_status: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class SimulationOutput:
    t: np.ndarray
    Y: np.ndarray
    provenance: dict[str, object]
    fallback_occurred: bool = False
    fallback_message: Optional[str] = None

"""Own current fitting base-evaluator state for the fitting window."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Optional

import numpy as np

from kindred.core.api.simulation import SimulationBuilder
from kindred.core.fitting_evaluation import SerialFittingEvaluator, coerce_fitting_series_evaluator
from kindred.core.simulation_preparation import (
    PreparedSimulationMetadata,
    coerce_prepared_simulation_metadata,
)


@dataclass(frozen=True)
class FittingEvaluatorComponents:
    base_evaluator: Any
    evaluator_factory: Optional[Callable[[], Any]]
    prepared_simulation: Optional[PreparedSimulationMetadata]
    readiness_required: bool


class FittingEvaluatorStateOwner:
    """Owns mutable fitting base-evaluator identity and builder refresh decisions."""

    def __init__(
        self,
        *,
        base_evaluator: Any = None,
        simulation_builder: Optional[SimulationBuilder] = None,
    ) -> None:
        self._base_evaluator = base_evaluator
        self._simulation_builder = simulation_builder

    def current_base_evaluator(self) -> Any:
        return self._base_evaluator

    def set_base_evaluator(self, evaluator: Any) -> Any:
        self._base_evaluator = evaluator
        return self._base_evaluator

    def set_simulation_builder(self, simulation_builder: Optional[SimulationBuilder]) -> None:
        self._simulation_builder = simulation_builder

    def prepared_simulation_meta(self, evaluator: Any = None) -> Optional[PreparedSimulationMetadata]:
        target = self._base_evaluator if evaluator is None else evaluator
        return self.prepared_simulation_meta_for(target)

    def components_for_runtime_identity(
        self,
        *,
        mechanism_text: str,
        config: Mapping[str, Any],
        requested_solver: str,
        requested_rtol: float,
        requested_atol: float,
        runtime_settings: Mapping[str, Any],
        param_names_for_readiness: Callable[..., list[str]],
    ) -> Optional[FittingEvaluatorComponents]:
        base_evaluator = self._base_evaluator
        prepared_simulation = self.prepared_simulation_meta(base_evaluator)
        needs_builder = base_evaluator is None
        prepared_matches_request = False
        if prepared_simulation is not None:
            prepared_matches_request = (
                self._prepared_simulation_matches_mechanism(prepared_simulation, mechanism_text)
                and self._prepared_solver_normalized(prepared_simulation) == str(requested_solver)
                and self._prepared_simulation_matches_runtime_settings(prepared_simulation, runtime_settings)
                and self._prepared_tolerances_match(
                    prepared_simulation,
                    requested_rtol=float(requested_rtol),
                    requested_atol=float(requested_atol),
                )
            )
            needs_builder = needs_builder or not prepared_matches_request
        elif base_evaluator is not None:
            needs_builder = True
        if not needs_builder:
            readiness_required = type(coerce_fitting_series_evaluator(base_evaluator)) is SerialFittingEvaluator
            return FittingEvaluatorComponents(base_evaluator, None, prepared_simulation, readiness_required)
        if not callable(self._simulation_builder):
            if base_evaluator is not None:
                try:
                    if type(coerce_fitting_series_evaluator(base_evaluator)) is not SerialFittingEvaluator:
                        return FittingEvaluatorComponents(base_evaluator, None, prepared_simulation, False)
                except Exception:
                    return None
            return None
        param_names = param_names_for_readiness(
            config=dict(config or {}),
            prepared_simulation=prepared_simulation if prepared_matches_request else None,
            mechanism_text=mechanism_text,
        )
        runtime_snapshot = dict(runtime_settings or {})

        def _build_deferred_fit_evaluator():
            return self._simulation_builder(
                str(mechanism_text or ""),
                list(param_names),
                solver=str(requested_solver),
                rtol=float(requested_rtol),
                atol=float(requested_atol),
                temperature_K=float(runtime_snapshot["temperature_K"]),
                use_sparse_jacobian=bool(runtime_snapshot["use_sparse_jacobian"]),
                wegscheider_cyclicity_enabled=bool(runtime_snapshot["wegscheider_cyclicity_enabled"]),
            )

        return FittingEvaluatorComponents(base_evaluator, _build_deferred_fit_evaluator, None, True)

    def build_and_set(
        self,
        *,
        mechanism_text: str,
        param_names: list[str],
        requested_solver: str,
        requested_rtol: float,
        requested_atol: float,
        runtime_settings: Mapping[str, Any],
    ) -> tuple[Any, Optional[PreparedSimulationMetadata]]:
        if not callable(self._simulation_builder):
            raise RuntimeError("Simulation builder unavailable.")
        base_evaluator = self._simulation_builder(
            str(mechanism_text or ""),
            list(param_names),
            solver=str(requested_solver),
            rtol=float(requested_rtol),
            atol=float(requested_atol),
            temperature_K=float(runtime_settings["temperature_K"]),
            use_sparse_jacobian=bool(runtime_settings["use_sparse_jacobian"]),
            wegscheider_cyclicity_enabled=bool(runtime_settings["wegscheider_cyclicity_enabled"]),
        )
        self._base_evaluator = base_evaluator
        return base_evaluator, self.prepared_simulation_meta(base_evaluator)

    @staticmethod
    def prepared_simulation_meta_for(evaluator: Any) -> Optional[PreparedSimulationMetadata]:
        if evaluator is None:
            return None
        try:
            prepared = getattr(evaluator, "prepared_metadata", None)
        except Exception:
            prepared = None
        meta = coerce_prepared_simulation_metadata(prepared)
        if meta is not None:
            return meta
        try:
            prepared = getattr(evaluator, "_kindred_prepared_simulation_meta", None)
        except Exception:
            return None
        return coerce_prepared_simulation_metadata(prepared)

    @staticmethod
    def _prepared_solver_normalized(prepared_simulation: Optional[PreparedSimulationMetadata]) -> str:
        if prepared_simulation is None:
            return ""
        return str(prepared_simulation.solver_normalized).strip()

    @staticmethod
    def _mechanism_text_sha256(mechanism_text: str) -> str:
        return hashlib.sha256(str(mechanism_text or "").encode("utf-8")).hexdigest()

    @classmethod
    def _prepared_simulation_matches_mechanism(
        cls,
        prepared_simulation: Optional[PreparedSimulationMetadata],
        mechanism_text: str,
    ) -> bool:
        if prepared_simulation is None:
            return False
        expected_hash = str(getattr(prepared_simulation, "mechanism_text_sha256", "") or "")
        if not expected_hash:
            return False
        try:
            expected_len = int(getattr(prepared_simulation, "mechanism_text_len"))
        except Exception:
            return False
        text = str(mechanism_text or "")
        return expected_hash == cls._mechanism_text_sha256(text) and expected_len == len(text)

    @staticmethod
    def _prepared_simulation_matches_runtime_settings(
        prepared_simulation: Optional[PreparedSimulationMetadata],
        runtime_settings: Mapping[str, Any],
    ) -> bool:
        if prepared_simulation is None:
            return False
        try:
            prepared_temperature = float(prepared_simulation.temperature_K)
            requested_temperature = float(runtime_settings["temperature_K"])
        except Exception:
            return False
        return (
            np.isfinite(prepared_temperature)
            and np.isfinite(requested_temperature)
            and math.isclose(prepared_temperature, requested_temperature, rel_tol=1e-9, abs_tol=1e-12)
            and bool(prepared_simulation.use_sparse_jacobian) == bool(runtime_settings["use_sparse_jacobian"])
            and bool(prepared_simulation.wegscheider_cyclicity_enabled)
            == bool(runtime_settings["wegscheider_cyclicity_enabled"])
        )

    @staticmethod
    def _prepared_tolerances_match(
        prepared_simulation: PreparedSimulationMetadata,
        *,
        requested_rtol: float,
        requested_atol: float,
    ) -> bool:
        try:
            prepared_rtol = float(prepared_simulation.rtol)
            prepared_atol = float(prepared_simulation.atol)
        except Exception:
            return False
        return (
            np.isfinite(prepared_rtol)
            and np.isfinite(prepared_atol)
            and math.isclose(float(prepared_rtol), float(requested_rtol), rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(float(prepared_atol), float(requested_atol), rel_tol=1e-9, abs_tol=1e-12)
        )

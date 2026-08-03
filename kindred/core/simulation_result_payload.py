from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from kindred.core.result_repository import SimulationResultValue


SimulationSuccessPayload = Dict[str, Any]

_MISSING = object()
_SUCCESS_MESSAGE = "Simulation completed successfully"


def _build_simulation_success_payload_base(
    *,
    result: SimulationResultValue,
    base_species_count: int | None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
) -> SimulationSuccessPayload:
    if not isinstance(result, SimulationResultValue):
        raise TypeError("result must be an admitted SimulationResultValue.")
    species_names = list(result.series)
    y = np.vstack([result.series[name] for name in species_names])
    provenance_payload: Any = dict(result.provenance.solver)

    payload: SimulationSuccessPayload = {
        "t": result.t,
        "Y": y,
        "segment_ids": result.segment_ids,
        "segment_keys": tuple(result.segment_keys),
        "discontinuity_ids": result.discontinuity_ids,
        "species_names": species_names,
        "algebra_scalars": dict(result.scalars),
        "algebra_errors": [
            dict(error) for error in result.provenance.algebra_errors
        ],
        "warnings": list(warnings or []),
        "solver": str(solver),
        "nfev": getattr(result, "nfev", None),
        "success": True,
        "message": _SUCCESS_MESSAGE,
        "mechanism_text": str(mechanism_text or ""),
        "solver_config": dict(solver_config or {}),
        "provenance": provenance_payload,
    }
    payload["intervention_execution"] = (
        result.intervention_execution.provenance_projection()
    )
    if base_species_count is not None:
        payload["base_species_count"] = max(0, int(base_species_count))
    return payload


def build_simulation_success_payload(
    *,
    result: SimulationResultValue,
    base_species_count: int | None = None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    mechanism: Any = _MISSING,
) -> SimulationSuccessPayload:
    payload = _build_simulation_success_payload_base(
        result=result,
        base_species_count=base_species_count,
        warnings=warnings,
        solver=solver,
        mechanism_text=mechanism_text,
        solver_config=solver_config,
    )
    if mechanism is not _MISSING:
        payload["mechanism"] = mechanism
    return payload


def build_secondary_simulation_success_payload(
    *,
    result: SimulationResultValue,
    base_species_count: int | None = None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
) -> SimulationSuccessPayload:
    return _build_simulation_success_payload_base(
        result=result,
        base_species_count=base_species_count,
        warnings=warnings,
        solver=solver,
        mechanism_text=mechanism_text,
        solver_config=solver_config,
    )

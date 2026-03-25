from __future__ import annotations

from typing import Any, Dict, Mapping


SimulationSuccessPayload = Dict[str, Any]

_MISSING = object()
_SUCCESS_MESSAGE = "Simulation completed successfully"


def _build_simulation_success_payload_base(
    *,
    result: Any,
    y: Any,
    species_names: list[str],
    base_species_count: int | None,
    algebra_scalars: Mapping[str, Any] | None,
    algebra_errors: list[dict[str, Any]] | None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    extra_fields: Mapping[str, Any] | None = None,
) -> SimulationSuccessPayload:
    provenance = getattr(result, "provenance", {})
    if isinstance(provenance, Mapping):
        provenance_payload: Any = dict(provenance)
    elif provenance is None:
        provenance_payload = {}
    else:
        provenance_payload = provenance

    payload: SimulationSuccessPayload = {
        "t": result.t,
        "Y": y,
        "species_names": list(species_names),
        "algebra_scalars": dict(algebra_scalars or {}),
        "algebra_errors": list(algebra_errors or []),
        "warnings": list(warnings or []),
        "solver": str(solver),
        "nfev": getattr(result, "nfev", None),
        "success": True,
        "message": _SUCCESS_MESSAGE,
        "mechanism_text": str(mechanism_text or ""),
        "solver_config": dict(solver_config or {}),
        "provenance": provenance_payload,
        "fallback_occurred": bool(getattr(result, "fallback_occurred", False)),
        "fallback_message": getattr(result, "fallback_message", None),
    }
    if base_species_count is not None:
        payload["base_species_count"] = max(0, int(base_species_count))
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


def build_simulation_success_payload(
    *,
    result: Any,
    y: Any,
    species_names: list[str],
    base_species_count: int | None = None,
    algebra_scalars: Mapping[str, Any] | None,
    algebra_errors: list[dict[str, Any]] | None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    mechanism: Any = _MISSING,
    extra_fields: Mapping[str, Any] | None = None,
) -> SimulationSuccessPayload:
    payload = _build_simulation_success_payload_base(
        result=result,
        y=y,
        species_names=species_names,
        base_species_count=base_species_count,
        algebra_scalars=algebra_scalars,
        algebra_errors=algebra_errors,
        warnings=warnings,
        solver=solver,
        mechanism_text=mechanism_text,
        solver_config=solver_config,
        extra_fields=extra_fields,
    )
    if mechanism is not _MISSING:
        payload["mechanism"] = mechanism
    return payload


def build_secondary_simulation_success_payload(
    *,
    result: Any,
    y: Any,
    species_names: list[str],
    base_species_count: int | None = None,
    algebra_scalars: Mapping[str, Any] | None,
    algebra_errors: list[dict[str, Any]] | None,
    warnings: list[dict[str, Any]] | None,
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    extra_fields: Mapping[str, Any] | None = None,
) -> SimulationSuccessPayload:
    return _build_simulation_success_payload_base(
        result=result,
        y=y,
        species_names=species_names,
        base_species_count=base_species_count,
        algebra_scalars=algebra_scalars,
        algebra_errors=algebra_errors,
        warnings=warnings,
        solver=solver,
        mechanism_text=mechanism_text,
        solver_config=solver_config,
        extra_fields=extra_fields,
    )

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from kindred.core import simulation_algebra_policy
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_preparation import metadata_view_for_mechanism
from kindred.core.simulation_result_payload import (
    build_secondary_simulation_success_payload,
    build_simulation_success_payload,
)


@dataclass(frozen=True)
class FinalizedSimulationResult:
    y: np.ndarray
    species_names: list[str]
    base_species_count: int
    algebra_scalars: dict[str, float]
    algebra_errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", self.y.copy())
        object.__setattr__(self, "species_names", list(self.species_names))
        object.__setattr__(self, "algebra_scalars", deepcopy(dict(self.algebra_scalars)))
        object.__setattr__(self, "algebra_errors", [deepcopy(dict(error)) for error in self.algebra_errors])
        object.__setattr__(self, "warnings", [deepcopy(dict(warning)) for warning in self.warnings])


def _initials_for_algebra(
    mechanism: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
) -> dict[str, float]:
    if isinstance(initials_for_algebra, Mapping):
        return {str(name): float(value) for name, value in dict(initials_for_algebra).items()}
    species = getattr(mechanism, "species", {}) or {}
    return {
        str(name): float(getattr(species[str(name)], "initial_conc"))
        for name in species_names
        if str(name) in species
    }


def evaluate_simulation_result_algebra(
    *,
    mechanism: object,
    result: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    simulation_plan: SimulationPlan | Mapping[str, Any] | None,
) -> FinalizedSimulationResult:
    base_species_names = [str(name) for name in species_names]
    result_y = np.asarray(getattr(result, "Y"), dtype=float)
    algebra_scalars: dict[str, float] = {}
    algebra_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    algebra_text = metadata_view_for_mechanism(mechanism).algebra_text
    if not algebra_text:
        return FinalizedSimulationResult(
            y=result_y,
            species_names=base_species_names,
            base_species_count=len(base_species_names),
            algebra_scalars=algebra_scalars,
            algebra_errors=algebra_errors,
            warnings=warnings,
        )

    species_series = {
        str(species): result_y[index, :]
        for index, species in enumerate(base_species_names)
    }
    policy = simulation_algebra_policy.algebra_policy_from_simulation_plan(
        simulation_plan,
        default=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    evaluation = simulation_algebra_policy.evaluate_simulation_algebra(
        policy,
        mechanism,
        t=np.asarray(getattr(result, "t"), dtype=float).reshape(-1),
        species_series=species_series,
        initials=_initials_for_algebra(mechanism, base_species_names, initials_for_algebra),
    )
    algebra_scalars = dict(evaluation.scalars)
    algebra_errors = list(evaluation.errors)
    if evaluation.warning is not None:
        warnings.append(dict(evaluation.warning))
    if not evaluation.series:
        return FinalizedSimulationResult(
            y=result_y,
            species_names=base_species_names,
            base_species_count=len(base_species_names),
            algebra_scalars=algebra_scalars,
            algebra_errors=algebra_errors,
            warnings=warnings,
        )

    algebra_names = [str(name) for name in evaluation.series.keys()]
    algebra_matrix = np.vstack([np.asarray(evaluation.series[name], dtype=float) for name in algebra_names])
    return FinalizedSimulationResult(
        y=np.vstack([result_y, algebra_matrix]),
        species_names=base_species_names + algebra_names,
        base_species_count=len(base_species_names),
        algebra_scalars=algebra_scalars,
        algebra_errors=algebra_errors,
        warnings=warnings,
    )


def build_finalized_simulation_result_payload(
    *,
    mechanism: object,
    result: object,
    species_names: list[str],
    initials_for_algebra: Mapping[str, Any] | None,
    simulation_plan: SimulationPlan | Mapping[str, Any] | None,
    preparation_warnings: list[str] | tuple[str, ...],
    solver: str,
    mechanism_text: str,
    solver_config: Mapping[str, Any] | None,
    include_mechanism: bool,
) -> dict[str, Any]:
    finalized = evaluate_simulation_result_algebra(
        mechanism=mechanism,
        result=result,
        species_names=species_names,
        initials_for_algebra=initials_for_algebra,
        simulation_plan=simulation_plan,
    )
    warnings = [
        build_simulation_failure(
            "preparation_warning",
            str(message),
            details={"stage": "prepare_run_context"},
        )
        for message in list(preparation_warnings or [])
    ]
    warnings.extend(finalized.warnings)

    builder = build_simulation_success_payload if include_mechanism else build_secondary_simulation_success_payload
    kwargs: dict[str, Any] = {
        "result": result,
        "y": finalized.y,
        "species_names": finalized.species_names,
        "base_species_count": finalized.base_species_count,
        "algebra_scalars": finalized.algebra_scalars,
        "algebra_errors": finalized.algebra_errors,
        "warnings": warnings,
        "solver": str(solver),
        "mechanism_text": str(mechanism_text or ""),
        "solver_config": dict(solver_config or {}),
    }
    if include_mechanism:
        kwargs["mechanism"] = mechanism
    return builder(**kwargs)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.unit


class _Species:
    initial_conc = 1.0


class _Mechanism:
    metadata = {"algebra_text": "obs = [A]"}
    species = {"A": _Species()}


def test_result_finalization_preserves_algebra_series_scalars_errors_and_warnings(monkeypatch):
    from kindred.core.simulation_algebra_policy import SimulationAlgebraEvaluation
    from kindred.core.simulation_result_finalization import (
        build_finalized_simulation_result_payload,
    )

    def _evaluate(*_args, **_kwargs):
        return SimulationAlgebraEvaluation(
            series={"obs": np.asarray([2.0, 3.0], dtype=float)},
            scalars={"scale": 2.0},
            errors=[{"kind": "algebra_error", "message": "bad observable"}],
            warning={"kind": "algebra_warning", "message": "partial algebra"},
        )

    monkeypatch.setattr("kindred.core.simulation_algebra_policy.evaluate_simulation_algebra", _evaluate)

    result = SimpleNamespace(
        t=np.asarray([0.0, 1.0], dtype=float),
        Y=np.asarray([[1.0, 0.5]], dtype=float),
        provenance={},
    )

    payload = build_finalized_simulation_result_payload(
        mechanism=_Mechanism(),
        result=result,
        species_names=["A"],
        initials_for_algebra={"A": 1.0},
        simulation_plan=None,
        preparation_warnings=["normalized solver"],
        solver="BDF",
        mechanism_text="reaction: A -> A; k=0",
        solver_config={"solver": "BDF"},
        include_mechanism=False,
    )

    assert payload["species_names"] == ["A", "obs"]
    assert payload["base_species_count"] == 1
    assert np.asarray(payload["Y"]).shape == (2, 2)
    assert payload["algebra_scalars"] == {"scale": 2.0}
    assert payload["algebra_errors"] == [{"kind": "algebra_error", "message": "bad observable"}]
    assert [warning["kind"] for warning in payload["warnings"]] == [
        "preparation_warning",
        "algebra_warning",
    ]


def test_finalized_simulation_result_defensively_copies_constructor_inputs():
    from kindred.core.simulation_result_finalization import FinalizedSimulationResult

    class _SpeciesName(str):
        pass

    y = np.asarray([[1.0, 2.0]], dtype=float)
    species_name = _SpeciesName("A")
    species_names = [species_name]
    algebra_scalars = {"scale": 2.0, "nested": {"inner": 1.0}}
    algebra_errors = [
        {
            "kind": "algebra_error",
            "message": "bad observable",
            "context": {"stage": "evaluate"},
        }
    ]
    warnings = [
        {
            "kind": "algebra_warning",
            "message": "partial algebra",
            "details": {"stage": "prepare"},
        }
    ]

    finalized = FinalizedSimulationResult(
        y=y,
        species_names=species_names,
        base_species_count=1,
        algebra_scalars=algebra_scalars,
        algebra_errors=algebra_errors,
        warnings=warnings,
    )

    y[0, 0] = 99.0
    species_names.append("B")
    algebra_scalars["scale"] = 99.0
    algebra_scalars["nested"]["inner"] = 99.0
    algebra_errors[0]["message"] = "mutated"
    algebra_errors[0]["context"]["stage"] = "mutated"
    warnings[0]["message"] = "mutated"
    warnings[0]["details"]["stage"] = "mutated"

    assert np.array_equal(finalized.y, np.asarray([[1.0, 2.0]], dtype=float))
    assert finalized.species_names == ["A"]
    assert isinstance(finalized.species_names[0], _SpeciesName)
    assert finalized.algebra_scalars == {"scale": 2.0, "nested": {"inner": 1.0}}
    assert finalized.algebra_errors == [
        {
            "kind": "algebra_error",
            "message": "bad observable",
            "context": {"stage": "evaluate"},
        }
    ]
    assert finalized.warnings == [
        {
            "kind": "algebra_warning",
            "message": "partial algebra",
            "details": {"stage": "prepare"},
        }
    ]

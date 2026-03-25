from __future__ import annotations

import importlib.resources

import pytest


@pytest.mark.unit
def test_fitting_objective_uses_prepared_pipeline() -> None:
    source = importlib.resources.files("kindred.core").joinpath("fitting_objective.py").read_text(encoding="utf-8")

    assert "prepare_fitting_objective_context" in source
    assert "def build_prepared_fitting_objective(" in source
    assert "solve_policy_factory" in source
    assert "parameter_algebra_policy_factory" in source

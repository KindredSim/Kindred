from __future__ import annotations

import importlib.resources
import pytest

pytestmark = pytest.mark.unit



def test_parameter_algebra_module_uses_spec_and_eval_submodules() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("parameter_algebra.py")
        .read_text(encoding="utf-8")
    )

    assert "from kindred.core.simulator.parameter_algebra_spec import (" in source
    assert "from kindred.core.simulator.parameter_algebra_eval import (" in source
    assert "@dataclass(frozen=True)\nclass ParameterAssignment" not in source
    assert "def parse_parameter_algebra_spec_from_dsl_text(" not in source
    assert "def evaluate_parameter_algebra(" not in source

    simulator_package = importlib.resources.files("kindred.core.simulator")
    assert simulator_package.joinpath("parameter_algebra_spec.py").is_file()
    assert simulator_package.joinpath("parameter_algebra_eval.py").is_file()

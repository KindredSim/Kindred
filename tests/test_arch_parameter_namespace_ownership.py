from __future__ import annotations

import importlib.resources
import pytest

pytestmark = pytest.mark.unit



def test_parameter_algebra_spec_and_eval_do_not_own_mechanism_name_resolution() -> None:
    simulator_package = importlib.resources.files("kindred.core.simulator")
    spec_source = simulator_package.joinpath("parameter_algebra_spec.py").read_text(encoding="utf-8")
    eval_source = simulator_package.joinpath("parameter_algebra_eval.py").read_text(encoding="utf-8")

    assert "def _build_mechanism_param_lookup" not in spec_source
    assert "def _resolve_mechanism_param_name" not in spec_source
    assert ".resolve(" in spec_source

    assert "_build_mechanism_param_lookup" not in eval_source
    assert "_resolve_mechanism_param_name" not in eval_source
    assert ".resolve(" in eval_source


def test_dsl_parameter_scan_delegates_namespace_policy_to_shared_owner() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl_parameter_scan.py")
        .read_text(encoding="utf-8")
    )

    assert "build_namespace_from_ir_steps" in source
    assert "is_equilibrium_step = bool(" not in source
    assert 'mechanism_param_names.add(f"Keq{step_index}")' not in source


def test_dsl_build_delegates_shared_step_policy_to_parameter_namespace_owner() -> None:
    source = (
        importlib.resources.files("kindred.core.simulator")
        .joinpath("dsl_build.py")
        .read_text(encoding="utf-8")
    )

    assert "_namespace_policy_from_step" in source
    assert "is_equilibrium or (reversible and kr_attr is not None)" not in source
    assert 'bool(getattr(step, "Keq_input", None) is not None)' not in source


def test_parameter_algebra_and_preparation_do_not_keep_namespace_escape_hatches() -> None:
    simulator_package = importlib.resources.files("kindred.core.simulator")
    parameter_algebra_source = simulator_package.joinpath("parameter_algebra.py").read_text(encoding="utf-8")
    simulation_preparation_source = importlib.resources.files("kindred.core").joinpath(
        "simulation_preparation.py"
    ).read_text(encoding="utf-8")

    assert "Legacy fallback: per-type ordinals." not in parameter_algebra_source
    assert 'units[f"k{i}"]' not in parameter_algebra_source
    assert 'units[f"kf{i}"]' not in parameter_algebra_source
    assert 'units[f"kr{i}"]' not in parameter_algebra_source
    assert 'units[f"Keq{i}"]' not in parameter_algebra_source

    assert "Prepared parameter-algebra binding prepass failed" not in simulation_preparation_source
    assert "mech_bind_names = list(param_names)" not in simulation_preparation_source

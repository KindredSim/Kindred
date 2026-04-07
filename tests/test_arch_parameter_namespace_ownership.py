from __future__ import annotations

import importlib.resources


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

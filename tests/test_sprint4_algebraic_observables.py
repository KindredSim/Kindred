from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit



def test_detect_unknown_scalar_identifiers_uses_algebra_ast():
    from kindred.core.algebra.observable_introspection import detect_unknown_scalar_identifiers

    unknown = detect_unknown_scalar_identifiers(
        "scale * ([A] + [B]) + R + sin(1) + k1",
        observable_name="total",
        known_identifiers={"k1", "already_known"},
        mechanism_species={"A", "B"},
    )
    assert unknown == {"scale"}


def test_detect_unknown_scalar_identifiers_flags_bare_species_names():
    from kindred.core.algebra.observable_introspection import analyze_observable_expression

    analysis = analyze_observable_expression("A + [A] + foo")
    assert analysis.identifiers == {"A", "foo"}
    assert analysis.species_refs == {"A"}


def test_extract_observables_from_algebra_text_includes_let_declarations():
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    algebra_text = "\n".join(
        [
            "param a = 2.0",
            "let total_PBMP = [PBMPBPIN] + [PBMP]",
            "let selectivity = [PBMP] / max([PBMP] + [pinBOH], 1e-18)",
            "# comment",
        ]
    )
    obs = extract_observables_from_algebra_text(algebra_text)
    assert obs["total_PBMP"] == "[PBMPBPIN] + [PBMP]"
    assert "max(" in obs["selectivity"]
    assert "a" not in obs


@pytest.mark.parametrize("name", ["signal", "time", "reaction", "energy"])
def test_extract_observables_from_algebra_text_rejects_bare_assignment(name):
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    with pytest.raises(ValueError, match="Use 'let name = expr' or 'param name = expr'"):
        extract_observables_from_algebra_text(f"{name} = [A]")


@pytest.mark.parametrize("name", ["signal", "time", "reaction", "energy"])
def test_compile_algebra_observables_rejects_bare_assignment(name):
    from kindred.core.algebra.simulation_series import compile_algebra_observables

    with pytest.raises(ValueError, match="Use 'let name = expr' or 'param name = expr'"):
        compile_algebra_observables(f"{name} = [A]")


def test_extract_observables_from_algebra_text_rejects_protected_indexed_identifier():
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    with pytest.raises(ValueError, match="protected indexed parameter identifier"):
        extract_observables_from_algebra_text("let K1 = [A]")


def test_compile_algebra_observables_rejects_protected_indexed_identifier():
    from kindred.core.algebra.simulation_series import compile_algebra_observables

    with pytest.raises(ValueError, match="protected indexed parameter identifier"):
        compile_algebra_observables("let K1 = [A]")


def test_extract_observables_from_algebra_text_rejects_unresolved_protected_rhs():
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    with pytest.raises(ValueError, match="K1.*protected indexed parameter identifier"):
        extract_observables_from_algebra_text("let signal = K1")


def test_compile_algebra_observables_rejects_unresolved_protected_rhs():
    from kindred.core.algebra.simulation_series import compile_algebra_observables

    with pytest.raises(ValueError, match="K1.*protected indexed parameter identifier"):
        compile_algebra_observables("let signal = K1")


def test_compiled_algebra_observable_canonicalizes_indexed_k_direct_spelling_for_evaluation():
    from kindred.core.algebra.simulation_series import (
        compile_algebra_observables,
        evaluate_compiled_algebra_series_for_simulation,
    )
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=2.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )
    compiled = compile_algebra_observables(
        "let signal = K1 * [A]",
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )

    series, _scalars = evaluate_compiled_algebra_series_for_simulation(
        mechanism,
        compiled,
        t=np.asarray([0.0, 1.0]),
        species_series={"A": np.asarray([1.0, 0.25]), "B": np.asarray([0.0, 0.75])},
        initials={"A": 1.0, "B": 0.0},
    )

    np.testing.assert_allclose(series["signal"], [2.0, 0.5])
    assert "K1" not in compiled.processed_text
    assert "k1" in compiled.processed_text


def test_compiled_algebra_observable_canonicalized_source_preserves_grouping():
    from kindred.core.algebra.parser import parse_algebra
    from kindred.core.algebra.simulation_series import compile_algebra_observables
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    def _expr(source: str):
        return parse_algebra(f"# Algebra\nlet signal = {source}\n").lines[0].expr

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=2.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )
    compiled = compile_algebra_observables(
        "let signal = (K1 ^ 2) ^ 3\nlet gate = K1 && (scale || [A])",
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )

    lines = compiled.processed_text.splitlines()
    assert _expr(lines[1].split("=", 1)[1].strip()) == _expr("(k1 ^ 2) ^ 3")
    assert _expr(lines[2].split("=", 1)[1].strip()) == _expr("k1 && (scale || [A])")


def test_compiled_algebra_observable_rejects_indexed_k_for_reversible_only_step():
    from kindred.core.algebra.simulation_series import compile_algebra_observables
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "equilibrium: A <-> B; kf=1.0; kr=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )

    with pytest.raises(ValueError, match="K1.*not a valid indexed parameter identifier"):
        compile_algebra_observables(
            "let signal = K1",
            mechanism_namespace=build_namespace_from_mechanism(mechanism),
        )


@pytest.mark.parametrize("name", ["k9", "KF1", "KEQ1", "keq1"])
def test_compile_algebra_observables_rejects_unresolved_protected_rhs_identifiers(name: str):
    from kindred.core.algebra.simulation_series import compile_algebra_observables

    with pytest.raises(ValueError, match=rf"{name}.*protected indexed parameter identifier"):
        compile_algebra_observables(f"let signal = {name}")


def test_detect_unknown_scalar_identifiers_rejects_protected_indexed_missing_scalar():
    from kindred.core.algebra.observable_introspection import detect_unknown_scalar_identifiers

    with pytest.raises(ValueError, match="K1.*protected indexed parameter identifier"):
        detect_unknown_scalar_identifiers(
            "K1 * [A]",
            observable_name="signal",
            known_identifiers=set(),
            mechanism_species={"A"},
        )


def test_detect_unknown_scalar_identifiers_canonicalizes_resolved_indexed_spelling_before_missing_scalar_detection():
    from kindred.core.algebra.observable_introspection import detect_unknown_scalar_identifiers
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )

    unknown = detect_unknown_scalar_identifiers(
        "K1 * [A] + scale",
        observable_name="signal",
        known_identifiers={"k1"},
        mechanism_species={"A"},
        mechanism_namespace=build_namespace_from_mechanism(mechanism),
    )

    assert unknown == {"scale"}


@pytest.mark.parametrize("name", ["k", "kf", "kr", "K", "Keq"])
def test_extract_observables_from_algebra_text_rejects_bare_step_local_keys(name):
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    with pytest.raises(ValueError, match="step-local DSL key"):
        extract_observables_from_algebra_text(f"let {name} = [A]")


@pytest.mark.parametrize("name", ["k", "kf", "kr", "K", "Keq"])
def test_compile_algebra_observables_rejects_bare_step_local_keys(name):
    from kindred.core.algebra.simulation_series import compile_algebra_observables

    with pytest.raises(ValueError, match="step-local DSL key"):
        compile_algebra_observables(f"let {name} = [A]")


def test_global_fit_targets_algebra_observable_series_shared_scalar(monkeypatch):
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    def _fake_solve_constant(request):
        from kindred.core.simulator.solvers import SimulationOutput

        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        grid_n = int((request.grid or {}).get("N") or 6)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
        yy = np.tile(y0.reshape(-1, 1), (1, tt.size))
        return SimulationOutput(t=tt, Y=yy, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param a = 2.0",
            "let signal = a * [A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    t_exp = np.linspace(0.0, 1.0, 6)
    y_exp = np.full_like(t_exp, 2.0)
    datasets = [{"id": "ds1", "t": t_exp, "y": y_exp, "species": "signal"}]

    result = fit_global(prepared, datasets, {"a": 2.0}, method="lm", max_nfev=2)
    assert result.completion.status == "ok"
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))


def test_global_fit_targets_algebra_observable_series_per_dataset_scalar(monkeypatch):
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    def _fake_solve_constant(request):
        from kindred.core.simulator.solvers import SimulationOutput

        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        grid_n = int((request.grid or {}).get("N") or 6)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
        yy = np.tile(y0.reshape(-1, 1), (1, tt.size))
        return SimulationOutput(t=tt, Y=yy, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param a = 1.0",
            "let signal = a * [A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    t_exp = np.linspace(0.0, 1.0, 6)
    datasets = [
        {"id": "ds1", "t": t_exp, "y": np.full_like(t_exp, 2.0), "species": "signal"},
        {"id": "ds2", "t": t_exp, "y": np.full_like(t_exp, 3.0), "species": "signal"},
    ]
    dataset_variable_params = {
        "ds1": {"a": {"initial": 2.0, "min": -np.inf, "max": np.inf}},
        "ds2": {"a": {"initial": 3.0, "min": -np.inf, "max": np.inf}},
    }

    result = fit_global(
        prepared,
        datasets,
        {},
        dataset_variable_params=dataset_variable_params,
        method="lm",
        max_nfev=2,
    )
    assert result.completion.status == "ok"
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))

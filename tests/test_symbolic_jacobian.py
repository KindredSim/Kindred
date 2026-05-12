from __future__ import annotations

from dataclasses import replace
import types

import numpy as np
import pytest

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.rate_binding import RateBinding
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_prepared_simulation_func,
    coerce_prepared_simulation_metadata,
    prepare_bound_mechanism,
    prepare_fitting_objective_context,
    prepared_simulation_run_for_execution_request,
    prepare_simulation_worker_run,
)
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import solve_ode
from kindred.io.resources import get_preset_mechanism


pytestmark = pytest.mark.unit


SUPPORTED_DSL = "\n".join(
    [
        "reaction: A + B -> C; k=0.7",
        "reaction: C -> A; k=0.2",
        "equilibrium: B <-> C; kf=1.1; kr=0.4",
        "init: A=1.2, B=0.9, C=0.1",
    ]
)


def test_symbolic_jacobian_matches_finite_difference_rhs_for_supported_subset():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    mechanism = parse_dsl_to_mechanism(SUPPORTED_DSL, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    artifact = build_symbolic_jacobian_artifact(mechanism)

    y = np.asarray([1.2, 0.9, 0.1], dtype=float)
    symbolic_matrix = np.asarray(artifact.jacobian_func(0.0, y), dtype=float)
    fd_matrix = np.zeros_like(symbolic_matrix)
    eps_base = 1e-7
    for col in range(y.size):
        eps = eps_base * max(1.0, abs(float(y[col])))
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[col] += eps
        y_minus[col] -= eps
        fd_matrix[:, col] = (rhs(0.0, y_plus) - rhs(0.0, y_minus)) / (2.0 * eps)

    np.testing.assert_allclose(symbolic_matrix, fd_matrix, rtol=1e-6, atol=1e-8)


def test_symbolic_jacobian_structure_binds_distinct_immutable_parameter_snapshots():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=2.0, B=0.0",
            ]
        ),
        initials={},
    )

    structure = build_symbolic_jacobian_structure(mechanism)
    first = structure.bind({"k1": 1.0})
    second = structure.bind({"k1": 3.0})
    first_payload = first.identity.to_payload()
    second_payload = second.identity.to_payload()

    assert first_payload["structure_fingerprint"] == second_payload["structure_fingerprint"]
    assert first_payload["evaluation_snapshot_fingerprint"] != second_payload["evaluation_snapshot_fingerprint"]
    assert first_payload["parameter_symbols"] == ["k1"]
    np.testing.assert_allclose(first.jacobian_func(0.0, np.asarray([2.0, 0.0])), [[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(second.jacobian_func(0.0, np.asarray([2.0, 0.0])), [[-3.0, 0.0], [3.0, 0.0]])


def test_symbolic_jacobian_structure_does_not_carry_default_parameter_values():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=2.0, B=0.0",
            ]
        ),
        initials={},
    )

    structure = build_symbolic_jacobian_structure(mechanism)

    assert not hasattr(structure, "default_parameter_values")
    with pytest.raises(Exception, match="Missing symbolic parameter value"):
        structure.bind()


def test_symbolic_jacobian_uses_global_step_parameter_names_for_mixed_mechanisms():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_structure

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "equilibrium: B <-> C; kf=2.0; kr=0.5",
                "init: A=1.0, B=0.0, C=0.0",
            ]
        ),
        initials={},
    )

    structure = build_symbolic_jacobian_structure(mechanism)
    artifact = structure.bind({"k1": 1.0, "kf2": 4.0, "kr2": 0.25})
    identity = artifact.identity.to_payload()

    assert identity["parameter_symbols"] == ["k1", "kf2", "kr2"]
    assert artifact.evaluation_snapshot == (("k1", 1.0), ("kf2", 4.0), ("kr2", 0.25))
    np.testing.assert_allclose(
        artifact.jacobian_func(0.0, np.asarray([1.0, 2.0, 3.0])),
        [
            [-1.0, 0.0, 0.0],
            [1.0, -4.0, 0.25],
            [0.0, 4.0, -0.25],
        ],
    )


def test_symbolic_jacobian_rate_binding_snapshot_identity_changes_after_mutation():
    from kindred.core.mechanism import Mechanism
    from kindred.core.rate_binding import RateBinding
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    binding = RateBinding("k1", 1.0)
    mechanism = Mechanism()
    mechanism.add_species("A", 2.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_reaction(reactants={"A": 1.0}, products={"B": 1.0}, rate=binding)

    first = build_symbolic_jacobian_artifact(mechanism)
    binding.set(3.0)
    second = build_symbolic_jacobian_artifact(mechanism)
    first_payload = first.identity.to_payload()
    second_payload = second.identity.to_payload()

    assert first_payload["structure_fingerprint"] == second_payload["structure_fingerprint"]
    assert first_payload["evaluation_snapshot_fingerprint"] != second_payload["evaluation_snapshot_fingerprint"]
    np.testing.assert_allclose(first.jacobian_func(0.0, np.asarray([2.0, 0.0])), [[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(second.jacobian_func(0.0, np.asarray([2.0, 0.0])), [[-3.0, 0.0], [3.0, 0.0]])


def test_symbolic_jacobian_matches_finite_difference_for_same_side_catalyst():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A + E -> B + E; k=2.0",
                "init: A=3.0, E=5.0, B=0.0",
            ]
        ),
        initials={},
    )
    rhs = build_ode_rhs_from_mechanism(mechanism)
    artifact = build_symbolic_jacobian_artifact(mechanism)

    species_index = {name: idx for idx, name in enumerate(mechanism.species_names())}
    y = np.zeros(len(species_index), dtype=float)
    y[species_index["A"]] = 3.0
    y[species_index["E"]] = 5.0
    symbolic_matrix = np.asarray(artifact.jacobian_func(0.0, y), dtype=float)
    fd_matrix = np.zeros_like(symbolic_matrix)
    eps_base = 1e-7
    for col in range(y.size):
        eps = eps_base * max(1.0, abs(float(y[col])))
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[col] += eps
        y_minus[col] -= eps
        fd_matrix[:, col] = (rhs(0.0, y_plus) - rhs(0.0, y_minus)) / (2.0 * eps)

    np.testing.assert_allclose(symbolic_matrix, fd_matrix, rtol=1e-6, atol=1e-8)
    assert symbolic_matrix[species_index["A"], species_index["E"]] == pytest.approx(-6.0)
    assert symbolic_matrix[species_index["B"], species_index["E"]] == pytest.approx(6.0)
    assert symbolic_matrix[species_index["E"], species_index["E"]] == pytest.approx(0.0)


def test_bdf_preparation_uses_generated_symbolic_jacobian_callable():
    prepared = prepare_simulation_worker_run(
        mechanism_text=SUPPORTED_DSL,
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert callable(prepared.request.jacobian_func)
    assert identity is not None
    assert identity["kind"] == "jacobian"


def test_scheduled_temperature_disables_generated_symbolic_jacobian_truthfully():
    prepared = prepare_simulation_worker_run(
        mechanism_text=SUPPORTED_DSL + "\ntemp_step: t=[0,0.5,1.0], T=[298,310]",
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )

    assert prepared.temperature_schedule is not None
    assert prepared.request.jacobian_func is None
    assert prepared.request.jac_sparsity is None
    assert any("scheduled-temperature" in warning for warning in prepared.warnings)


def test_scheduled_temperature_with_dynamic_bindings_omits_sparsity_hint():
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text=SUPPORTED_DSL + "\ntemp_step: t=[0,0.5,1.0], T=[298,310]",
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 4},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"k1": 0.8},
        )
    )

    assert prepared.temperature_schedule is not None
    assert prepared.request.jacobian_func is None
    assert prepared.request.jac_sparsity is None
    assert any("scheduled-temperature" in warning for warning in prepared.warnings)


@pytest.mark.parametrize("dg_expr", ["-5", "0"])
def test_warm_structure_cache_does_not_reuse_symbolic_jacobian_for_unsupported_dg_equilibrium(dg_expr: str):
    from kindred.core.simulation_preparation import clear_symbolic_jacobian_structure_cache

    supported_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; Keq=4.0",
            "init: A=1.0, B=0.0",
        ]
    )
    unsupported_text = "\n".join(
        [
            f"equilibrium: A <-> B; kf=2.0; dG_eq={dg_expr}",
            "init: A=1.0, B=0.0",
        ]
    )
    solver_config = {
        "solver": "BDF",
        "grid": {"N": 5},
        "use_sparse_jacobian": True,
        "wegscheider_cyclicity_enabled": False,
    }

    clear_symbolic_jacobian_structure_cache()
    cold_unsupported = prepare_simulation_worker_run(
        mechanism_text=unsupported_text,
        initials={},
        t_span=(0.0, 1.0),
        solver_config=solver_config,
    )

    assert cold_unsupported.request.jacobian_func is None
    assert cold_unsupported.request.jac_sparsity is None
    assert any("Symbolic Jacobian unsupported" in warning for warning in cold_unsupported.warnings)

    clear_symbolic_jacobian_structure_cache()
    supported = prepare_simulation_worker_run(
        mechanism_text=supported_text,
        initials={},
        t_span=(0.0, 1.0),
        solver_config=solver_config,
    )
    warm_unsupported = prepare_simulation_worker_run(
        mechanism_text=unsupported_text,
        initials={},
        t_span=(0.0, 1.0),
        solver_config=solver_config,
    )

    assert callable(supported.request.jacobian_func)
    assert warm_unsupported.request.jacobian_func is None
    assert warm_unsupported.request.jac_sparsity is None
    assert any("Symbolic Jacobian unsupported" in warning for warning in warm_unsupported.warnings)


def test_bind_failure_records_unsupported_status_not_preflight_supported(monkeypatch):
    import kindred.core.simulation_preparation as simulation_preparation
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError

    def fail_bind(**_kwargs):
        raise UnsupportedSymbolicExpressionError("snapshot binding failed")

    monkeypatch.setattr(
        simulation_preparation,
        "_bind_symbolic_jacobian_for_current_mechanism",
        fail_bind,
    )

    prepared = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        initials={},
        t_span=(0.0, 0.1),
        solver_config={"solver": "BDF", "use_sparse_jacobian": True, "grid": {"N": 5}},
    )

    assert prepared.request.jacobian_func is None
    assert prepared.request.jac_sparsity is None
    assert prepared.request.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "binding-failed",
        "reason": "snapshot binding failed",
    }

    result = solve_ode(prepared.request)

    assert result.provenance["symbolic_jacobian"] is False
    assert result.provenance["symbolic_jacobian_status"] == prepared.request.symbolic_jacobian_status


def test_prepared_reuse_refreshes_symbolic_status_after_rebuild_bind_failure(monkeypatch):
    import kindred.core.simulation_preparation as simulation_preparation
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError

    solver_config = {"solver": "BDF", "use_sparse_jacobian": True, "grid": {"N": 5}}
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 0.1),
            solver_config=solver_config,
            mechanism_text="reaction: A -> B; k=1.0",
            parameter_overrides={"k1": 1.0},
        )
    )

    assert prepared.request.jacobian_func is not None
    assert prepared.request.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "supported",
        "code": "supported",
        "reason": "Symbolic Jacobian supported.",
    }

    def fail_bind(**_kwargs):
        raise UnsupportedSymbolicExpressionError("reuse binding failed")

    monkeypatch.setattr(
        simulation_preparation,
        "_bind_symbolic_jacobian_for_current_mechanism",
        fail_bind,
    )

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 0.1),
            solver_config=solver_config,
            mechanism_text="reaction: A -> B; k=1.0",
            parameter_overrides={"k1": 2.0},
        ),
    )

    assert changed.request.jacobian_func is None
    assert changed.request.jac_sparsity is None
    assert changed.request.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "binding-failed",
        "reason": "reuse binding failed",
    }


def test_fitting_unsupported_symbolic_jacobian_runs_numerically_after_supported_cache_warm():
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.simulation_preparation import clear_symbolic_jacobian_structure_cache

    supported_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; Keq=4.0",
            "init: A=1.0, B=0.0",
        ]
    )
    unsupported_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; dG_eq=-5",
            "init: A=1.0, B=0.0",
        ]
    )
    solver_config = {
        "solver": "BDF",
        "grid": {"N": 5},
        "use_sparse_jacobian": True,
        "wegscheider_cyclicity_enabled": False,
    }

    clear_symbolic_jacobian_structure_cache()
    warm_supported = prepare_simulation_worker_run(
        mechanism_text=supported_text,
        initials={},
        t_span=(0.0, 1.0),
        solver_config=solver_config,
    )
    context = prepare_fitting_execution_context(
        mechanism_text=unsupported_text,
        param_names=["kf1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
    )

    assert callable(warm_supported.request.jacobian_func)

    evaluator = SerialFittingEvaluator(context)
    result = evaluator({"kf1": 2.0})

    assert np.asarray(result.t, dtype=float).shape == (3,)
    assert set(result.species) == {"A", "B"}
    assert evaluator.prepared_metadata.symbolic_jacobian_identity is None


@pytest.mark.parametrize("solver", ["BDF", "Radau"])
def test_implicit_solver_receives_generated_symbolic_jacobian_callable(monkeypatch, solver):
    from kindred.core.simulator import solvers

    prepared = prepare_simulation_worker_run(
        mechanism_text=SUPPORTED_DSL,
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": solver,
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    received = {}

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        received["jac"] = kwargs.get("jac")
        kwargs["jac"](t_span[0], np.asarray(y0, dtype=float))
        t_eval = np.asarray(kwargs["t_eval"], dtype=float)
        return types.SimpleNamespace(
            success=True,
            message="ok",
            t=t_eval,
            y=np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t_eval.size, axis=1),
            t_events=[],
        )

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    result = solvers.solve_ode(prepared.request)

    assert received["jac"] is prepared.request.jacobian_func
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["symbolic_jacobian_identity"]["kind"] == "jacobian"


def test_active_intervention_interval_disables_symbolic_jacobian_through_segment_logic(monkeypatch):
    from kindred.core.simulator import solvers

    prepared = prepare_simulation_worker_run(
        mechanism_text=SUPPORTED_DSL,
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 3},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    assert prepared.request.jacobian_func is not None

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        assert "jac" not in kwargs
        fun(t_span[0], np.asarray(y0, dtype=float))
        t_eval = np.asarray(kwargs["t_eval"], dtype=float)
        return types.SimpleNamespace(
            success=True,
            message="ok",
            t=t_eval,
            y=np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t_eval.size, axis=1),
            t_events=[],
        )

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    result = solvers.solve_ode(
        replace(
            prepared.request,
            intervention_schedule={
                "intervals": [{"kind": "source", "species": "A", "start": 0.0, "end": 1.0, "rate": 0.1}]
            },
        )
    )

    assert result.provenance["has_intervention_schedule"] is True
    assert result.provenance["intervention_symbolic_jacobian_disabled"] is True
    assert result.provenance["symbolic_jacobian"] is False
    assert "symbolic_jacobian_identity" not in result.provenance
    assert result.provenance["symbolic_jacobian_status"] == {
        "kind": "jacobian",
        "state": "disabled",
        "code": "active-intervention-interval",
        "reason": "Symbolic Jacobian disabled for active intervention interval segments.",
    }


def test_mixed_intervention_segments_record_partial_symbolic_jacobian_disable(monkeypatch):
    from kindred.core.simulator import solvers

    prepared = prepare_simulation_worker_run(
        mechanism_text=SUPPORTED_DSL,
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    symbolic_jacobian = prepared.request.jacobian_func
    jacobian_receipts = []

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        jacobian_receipts.append(kwargs.get("jac") is symbolic_jacobian)
        t_eval = np.asarray(kwargs["t_eval"], dtype=float)
        return types.SimpleNamespace(
            success=True,
            message="ok",
            t=t_eval,
            y=np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t_eval.size, axis=1),
            t_events=[],
        )

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    result = solvers.solve_ode(
        replace(
            prepared.request,
            intervention_schedule={
                "intervals": [{"kind": "source", "species": "A", "start": 0.5, "end": 1.0, "rate": 0.1}]
            },
        )
    )

    assert jacobian_receipts == [True, False]
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["intervention_symbolic_jacobian_disabled"] is True
    assert result.provenance["intervention_symbolic_jacobian_partially_disabled"] is True
    assert result.provenance["intervention_segment_symbolic_jacobians"] == [True, False]
    assert result.provenance["symbolic_jacobian_identity"]["kind"] == "jacobian"
    assert result.provenance["symbolic_jacobian_status"] == {
        "kind": "jacobian",
        "state": "partially_disabled",
        "code": "active-intervention-interval",
        "reason": "Symbolic Jacobian disabled for active intervention interval segments.",
    }


def test_constant_temperature_arrhenius_uses_generated_symbolic_jacobian():
    prepared = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "energy=kJ/mol",
                "reaction: A -> B ; A=1e3 ; Ea=50",
                "init: A=1, B=0",
            ]
        ),
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["kind"] == "jacobian"
    assert prepared.warnings == []


def test_mixed_intervention_segments_keep_arrhenius_symbolic_jacobian_provenance(monkeypatch):
    from kindred.core.simulator import solvers

    prepared = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "energy=kJ/mol",
                "reaction: A -> B ; A=1e3 ; Ea=50",
                "init: A=1, B=0",
            ]
        ),
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    )
    symbolic_jacobian = prepared.request.jacobian_func
    symbolic_identity = getattr(symbolic_jacobian, "_kindred_symbolic_jacobian_identity", None)
    assert symbolic_jacobian is not None
    assert symbolic_identity is not None
    jacobian_receipts = []

    def fake_solve_ivp(*, fun, t_span, y0, **kwargs):
        jacobian_receipts.append(kwargs.get("jac") is symbolic_jacobian)
        t_eval = np.asarray(kwargs["t_eval"], dtype=float)
        return types.SimpleNamespace(
            success=True,
            message="ok",
            t=t_eval,
            y=np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t_eval.size, axis=1),
            t_events=[],
        )

    monkeypatch.setattr(solvers, "_solve_ivp", fake_solve_ivp)

    result = solvers.solve_ode(
        replace(
            prepared.request,
            intervention_schedule={
                "intervals": [{"kind": "source", "species": "A", "start": 0.5, "end": 1.0, "rate": 0.1}]
            },
        )
    )

    assert jacobian_receipts == [True, False]
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["intervention_symbolic_jacobian_disabled"] is True
    assert result.provenance["intervention_segment_symbolic_jacobians"] == [True, False]
    assert result.provenance["symbolic_jacobian_identity"] == symbolic_identity


def test_prepared_metadata_records_symbolic_jacobian_identity_after_first_use():
    simulation_func = build_prepared_simulation_func(
        mechanism_text=SUPPORTED_DSL,
        param_names=[],
        t_end=1.0,
        num_points=4,
        solver="BDF",
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
    )

    simulation_func({})
    metadata = coerce_prepared_simulation_metadata(
        getattr(simulation_func, "_kindred_prepared_simulation_meta")
    )

    assert metadata is not None
    assert metadata.symbolic_jacobian_identity is not None
    assert metadata.symbolic_jacobian_identity["kind"] == "jacobian"


def test_fitting_objective_context_uses_generated_symbolic_jacobian_callable():
    prepared = prepare_fitting_objective_context(
        mechanism_text=SUPPORTED_DSL,
        param_names=[],
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        target_species="C",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["kind"] == "jacobian"


def test_fitting_objective_context_uses_symbolic_jacobian_for_arrhenius_rates():
    prepared = prepare_fitting_objective_context(
        mechanism_text="\n".join(
            [
                "energy=kJ/mol",
                "reaction: A -> B ; A=1e3 ; Ea=50",
                "init: A=1, B=0",
            ]
        ),
        param_names=[],
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        target_species="B",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["kind"] == "jacobian"
    assert prepared.warnings == []


def test_fitting_objective_context_uses_snapshot_jacobian_for_supported_mutable_rate_bindings():
    prepared = prepare_fitting_objective_context(
        mechanism_text=SUPPORTED_DSL,
        param_names=["k1"],
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        target_species="C",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["kind"] == "jacobian"
    assert "k1" in identity["parameter_symbols"]
    assert identity["evaluation_snapshot_fingerprint"]
    assert prepared.request.jac_sparsity is None
    assert not any("dynamic rate bindings" in warning for warning in prepared.warnings)


def test_dynamic_rate_binding_sparse_preparation_generates_symbolic_after_concrete_override():
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text="\n".join(
                [
                    "reaction: A -> B; k=1.0",
                    "init: A=1.0, B=0.0",
                ]
            ),
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 5},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"k1": 2.0},
        )
    )

    result = solve_ode(prepared.request)

    assert callable(prepared.request.jacobian_func)
    assert prepared.request.jac_sparsity is None
    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    assert identity is not None
    assert identity["parameter_symbols"] == ["k1"]
    assert identity["evaluation_snapshot_fingerprint"]
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["jacobian_sparsity_hint"] is False
    assert result.provenance["symbolic_jacobian_identity"]["evaluation_snapshot_fingerprint"] == identity["evaluation_snapshot_fingerprint"]
    assert not any("dynamic rate bindings" in warning for warning in prepared.warnings)


def test_m1_sparse_dynamic_preview_uses_symbolic_jacobian_after_concrete_override():
    m1_without_algebra = "\n".join(
        line
        for line in get_preset_mechanism("M1").splitlines()
        if not line.strip().startswith("let ") and "Algebra" not in line
    )
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text=m1_without_algebra,
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 20},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"k1": 2.0},
        )
    )

    result = solve_ode(prepared.request)

    assert callable(prepared.request.jacobian_func)
    assert prepared.request.jac_sparsity is None
    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    assert identity is not None
    assert identity["evaluation_snapshot_fingerprint"]
    assert np.asarray(result.Y).shape == (2, 20)
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["jacobian_sparsity_hint"] is False
    assert not any("dynamic rate bindings" in warning for warning in prepared.warnings)


def test_m9_sparse_dynamic_preview_uses_symbolic_jacobian_after_concrete_override():
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text=get_preset_mechanism("M9"),
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 20},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"kf1": 2.0},
        )
    )

    result = solve_ode(prepared.request)

    assert callable(prepared.request.jacobian_func)
    assert prepared.request.jac_sparsity is None
    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    assert identity is not None
    assert identity["evaluation_snapshot_fingerprint"]
    assert np.asarray(result.Y).shape == (8, 20)
    assert result.provenance["symbolic_jacobian"] is True
    assert result.provenance["jacobian_sparsity_hint"] is False
    assert not any("dynamic rate bindings" in warning for warning in prepared.warnings)


def test_fitting_objective_context_disables_jacobian_for_mutable_keq_input_binding():
    prepared = prepare_fitting_objective_context(
        mechanism_text="\n".join(
            [
                "equilibrium: A <-> B; kf=1.0; K=2.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        param_names=["Keq1"],
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        target_species="B",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
    )

    assert prepared.request.jacobian_func is None
    assert prepared.request.jac_sparsity is None
    assert any("Keq input" in warning for warning in prepared.warnings)
    assert prepared.request.symbolic_jacobian_status == {
        "kind": "jacobian",
        "state": "unsupported",
        "code": "unsupported-keq-input",
        "reason": "Symbolic Jacobian does not support dynamic or non-finite Keq input for equilibrium 1.",
    }


def test_fitting_candidate_values_use_symbolic_snapshot_identity_when_supported():
    prepared = prepare_fitting_objective_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        param_names=["k1"],
        t_exp=np.asarray([0.0, 0.5, 1.0], dtype=float),
        target_species="B",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
    )

    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["parameter_symbols"] == ["k1"]
    assert identity["evaluation_snapshot_fingerprint"]
    assert prepared.request.jac_sparsity is None


def test_batch_per_set_parameter_snapshots_do_not_share_symbolic_identity():
    first = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text="\n".join(
                [
                    "reaction: A -> B; k=1.0",
                    "init: A=1.0, B=0.0",
                ]
            ),
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 5},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"k1": 1.5},
        )
    )
    second = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text="\n".join(
                [
                    "reaction: A -> B; k=1.0",
                    "init: A=1.0, B=0.0",
                ]
            ),
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 5},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            parameter_overrides={"k1": 2.5},
        )
    )

    first_identity = getattr(first.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    second_identity = getattr(second.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert first_identity is not None
    assert second_identity is not None
    assert first_identity["structure_fingerprint"] == second_identity["structure_fingerprint"]
    assert first_identity["evaluation_snapshot_fingerprint"] != second_identity["evaluation_snapshot_fingerprint"]


def test_execution_request_prepared_payload_rebuilds_symbolic_jacobian_for_batch_path():
    bound = prepare_bound_mechanism(
        mechanism_text=SUPPORTED_DSL,
        param_names=[],
        wegscheider_cyclicity_enabled=False,
    )
    execution_request = SimulationExecutionRequest(
        prepared_payload=bound.as_serializable_execution_payload(),
        initials={},
        t_span=(0.0, 1.0),
        solver_config={
            "solver": "BDF",
            "grid": {"N": 4},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
        mechanism_text=SUPPORTED_DSL,
    )

    prepared = prepare_simulation_worker_run(execution_request=execution_request)
    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert identity is not None
    assert identity["kind"] == "jacobian"


def test_execution_request_prepared_payload_drops_non_symbolic_jacobian_callable():
    def fake_jac(_t, y):
        return np.eye(len(y), dtype=float)

    bound = prepare_bound_mechanism(
        mechanism_text=SUPPORTED_DSL,
        param_names=[],
        wegscheider_cyclicity_enabled=False,
    )
    payload = dict(bound.as_serializable_execution_payload())
    payload["jacobian_func"] = fake_jac

    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=payload,
            initials={},
            t_span=(0.0, 1.0),
            solver_config={
                "solver": "BDF",
                "grid": {"N": 4},
                "use_sparse_jacobian": True,
                "wegscheider_cyclicity_enabled": False,
            },
            mechanism_text="",
        )
    )

    assert prepared.request.jacobian_func is not fake_jac
    identity = getattr(prepared.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    assert identity is not None
    assert any("non-symbolic Jacobian callable" in warning for warning in prepared.warnings)


def test_symbolic_jacobian_artifact_identity_changes_with_source_rates():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    first = build_symbolic_jacobian_artifact(parse_dsl_to_mechanism(SUPPORTED_DSL, initials={}))
    changed = build_symbolic_jacobian_artifact(
        parse_dsl_to_mechanism(SUPPORTED_DSL.replace("k=0.7", "k=0.8"), initials={})
    )

    assert first.identity.fingerprint != changed.identity.fingerprint


def test_symbolic_jacobian_artifact_identity_ignores_initial_concentration_values():
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    first = build_symbolic_jacobian_artifact(
        parse_dsl_to_mechanism("reaction: A -> B; k=1", initials={"A": 1.0, "B": 0.0})
    )
    changed_initials = build_symbolic_jacobian_artifact(
        parse_dsl_to_mechanism(
            "reaction: A -> B; k=1",
            initials={"A": 3.0, "B": 2.0},
        )
    )

    assert first.identity.to_payload() == changed_initials.identity.to_payload()


def test_symbolic_jacobian_rejects_unsupported_dynamic_rate_binding():
    from kindred.core.mechanism import Mechanism
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_reaction(reactants={"A": 1.0}, products={"B": 1.0}, rate=lambda: 1.0)

    with pytest.raises(UnsupportedSymbolicExpressionError):
        build_symbolic_jacobian_artifact(mechanism)


def test_symbolic_support_preflight_rejects_dynamic_rate_before_structure_fingerprint():
    from kindred.core.mechanism import Mechanism
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import symbolic_jacobian_structure_fingerprint_for_mechanism

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_reaction(reactants={"A": 1.0}, products={"B": 1.0}, rate=lambda: 1.0)

    with pytest.raises(UnsupportedSymbolicExpressionError, match="dynamic"):
        symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)


def test_symbolic_support_preflight_rejects_nonfinite_rate_binding_before_structure_fingerprint():
    from kindred.core.mechanism import Mechanism
    from kindred.core.rate_binding import RateBinding
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import symbolic_jacobian_structure_fingerprint_for_mechanism

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_reaction(reactants={"A": 1.0}, products={"B": 1.0}, rate=RateBinding("k1", float("nan")))

    with pytest.raises(UnsupportedSymbolicExpressionError, match="non-finite"):
        symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)


@pytest.mark.parametrize(
    "metadata",
    [
        {"forward_model": {"type": "Arrhenius"}},
        {"reverse_model": {"type": "Arrhenius"}},
    ],
)
def test_symbolic_support_preflight_rejects_equilibrium_model_metadata_before_structure_fingerprint(metadata):
    from kindred.core.mechanism import Mechanism
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import symbolic_jacobian_structure_fingerprint_for_mechanism

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=1.0,
        kr=0.5,
        metadata=metadata,
    )

    with pytest.raises(UnsupportedSymbolicExpressionError, match="Temperature-dependent equilibrium"):
        symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)


@pytest.mark.parametrize(
    "keq_input",
    [
        RateBinding("Keq1", 2.0),
        lambda: 2.0,
    ],
)
def test_symbolic_support_preflight_rejects_mutable_keq_input_before_structure_fingerprint(keq_input):
    from kindred.core.mechanism import Mechanism
    from kindred.core.mechanism_metadata import EquilibriumMetadataKeys
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import symbolic_jacobian_structure_fingerprint_for_mechanism

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kf=1.0,
        kr=0.5,
        Keq=2.0,
        metadata={EquilibriumMetadataKeys.KEQ_INPUT: keq_input},
    )

    with pytest.raises(UnsupportedSymbolicExpressionError, match="Keq input"):
        symbolic_jacobian_structure_fingerprint_for_mechanism(mechanism)


def test_symbolic_jacobian_rejects_nonpositive_direct_mechanism_keq_derivation():
    from kindred.core.mechanism import Mechanism
    from kindred.core.symbolic.errors import UnsupportedSymbolicExpressionError
    from kindred.core.symbolic.jacobian import build_symbolic_jacobian_artifact

    mechanism = Mechanism()
    mechanism.add_species("A", 1.0)
    mechanism.add_species("B", 0.0)
    mechanism.add_equilibrium(
        stoich_forward={"A": 1.0},
        stoich_back={"B": 1.0},
        kr=1.0,
        Keq=0.0,
    )

    with pytest.raises(UnsupportedSymbolicExpressionError, match="Keq"):
        build_symbolic_jacobian_artifact(mechanism)

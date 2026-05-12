from __future__ import annotations

import pytest
import numpy as np

from kindred.core.cache import fingerprint_simulation_request
from kindred.core.simulation_preparation import SimulationExecutionRequest, prepare_simulation_worker_run
from kindred.core.simulation_identity import SimulationIdentity

pytestmark = pytest.mark.unit


def _identity(schema_id: str = "schema-a") -> SimulationIdentity:
    return SimulationIdentity.build(
        schema_id=schema_id,
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
        },
        t_end=10.0,
        intervention_schedule_fingerprint="schedule-a",
    )


def test_simulation_identity_does_not_record_planned_symbolic_jacobian_identity_for_stiff_solver():
    identity = _identity()
    payload = identity.to_payload()

    assert "symbolic_jacobian_identity" not in payload


def test_simulation_identity_symbolic_jacobian_identity_affects_cache_and_runtime_keys():
    first = _identity(schema_id="schema-a")
    second = _identity(schema_id="schema-b")

    assert first.cache_key() != second.cache_key()
    assert first.prepared_runtime_key() != second.prepared_runtime_key()


def test_simulation_identity_can_use_actual_symbolic_jacobian_artifact_identity():
    actual_identity = {
        "kind": "jacobian",
        "backend_name": "sympy",
        "backend_version": "1.14.0",
        "profile_version": "test-profile",
        "source_fingerprint": "same-source",
        "artifact_fingerprint": "actual-artifact",
        "fingerprint": "actual-fingerprint",
    }

    planned = _identity(schema_id="schema-a")
    actual = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
        },
        t_end=10.0,
        intervention_schedule_fingerprint="schedule-a",
        symbolic_jacobian_identity=actual_identity,
    )

    assert actual.to_payload()["symbolic_jacobian_identity"] == actual_identity
    assert actual.cache_key() != planned.cache_key()
    assert actual.prepared_runtime_key() != planned.prepared_runtime_key()


def test_simulation_identity_prepared_runtime_key_ignores_jacobian_evaluation_snapshot():
    first = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
        },
        t_end=10.0,
        symbolic_jacobian_identity={
            "kind": "jacobian",
            "backend_name": "sympy",
            "backend_version": "1.14.0",
            "profile_version": "test-profile",
            "source_fingerprint": "same-structure",
            "structure_fingerprint": "same-structure",
            "artifact_fingerprint": "same-artifact",
            "evaluation_snapshot_fingerprint": "first-snapshot",
            "parameter_symbols": ["k1"],
            "fingerprint": "first-full-identity",
        },
    )
    second = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-b",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
        },
        t_end=10.0,
        symbolic_jacobian_identity={
            "kind": "jacobian",
            "backend_name": "sympy",
            "backend_version": "1.14.0",
            "profile_version": "test-profile",
            "source_fingerprint": "same-structure",
            "structure_fingerprint": "same-structure",
            "artifact_fingerprint": "same-artifact",
            "evaluation_snapshot_fingerprint": "second-snapshot",
            "parameter_symbols": ["k1"],
            "fingerprint": "second-full-identity",
        },
    )

    assert first.cache_key() != second.cache_key()
    assert first.prepared_runtime_key() == second.prepared_runtime_key()


def test_simulation_identity_wegscheider_identity_affects_cache_and_runtime_keys():
    first = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
            "wegscheider_cyclicity_enabled": True,
        },
        t_end=10.0,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "source_fingerprint": "first-source",
            "artifact_fingerprint": "first-artifact",
            "fingerprint": "first-full",
        },
    )
    second = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
            "wegscheider_cyclicity_enabled": True,
        },
        t_end=10.0,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "source_fingerprint": "second-source",
            "artifact_fingerprint": "second-artifact",
            "fingerprint": "second-full",
        },
    )

    assert first.cache_key() != second.cache_key()
    assert first.prepared_runtime_key() != second.prepared_runtime_key()


def test_simulation_identity_prepared_runtime_key_ignores_wegscheider_proof_identity():
    first = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
            "wegscheider_cyclicity_enabled": True,
        },
        t_end=10.0,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "source_fingerprint": "same-source",
            "artifact_fingerprint": "first-artifact",
            "fingerprint": "first-full",
        },
    )
    second = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-b",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "BDF",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
            "wegscheider_cyclicity_enabled": True,
        },
        t_end=10.0,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "source_fingerprint": "same-source",
            "artifact_fingerprint": "second-artifact",
            "fingerprint": "second-full",
        },
    )

    assert first.cache_key() != second.cache_key()
    assert first.prepared_runtime_key() == second.prepared_runtime_key()


def test_simulation_identity_omits_symbolic_jacobian_identity_for_non_jacobian_solver():
    identity = SimulationIdentity.build(
        schema_id="schema-a",
        param_fingerprint="params-a",
        canonical_initials_fingerprint="initials-a",
        solver_config={
            "solver": "RK45",
            "grid": {"N": 10},
            "use_sparse_jacobian": True,
            "temperature_K": 298.15,
        },
        t_end=10.0,
    )

    assert "symbolic_jacobian_identity" not in identity.to_payload()


def test_simulation_request_fingerprint_includes_symbolic_snapshot_identity():
    request_kwargs = {
        "prepared_payload": None,
        "mechanism_text": "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        "initials": {},
        "t_span": (0.0, 1.0),
        "solver_config": {
            "solver": "BDF",
            "grid": {"N": 5},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    }
    first = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            **request_kwargs,
            parameter_overrides={"k1": 1.5},
        )
    )
    second = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            **request_kwargs,
            parameter_overrides={"k1": 2.5},
        )
    )
    first_identity = getattr(first.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    second_identity = getattr(second.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert first_identity["structure_fingerprint"] == second_identity["structure_fingerprint"]
    assert first_identity["evaluation_snapshot_fingerprint"] != second_identity["evaluation_snapshot_fingerprint"]
    assert fingerprint_simulation_request(first.request) != fingerprint_simulation_request(second.request)


def test_planned_symbolic_snapshot_identity_matches_actual_after_scalar_algebra_override():
    from kindred.core.simulation_preparation import (
        clear_symbolic_jacobian_structure_cache,
        symbolic_jacobian_identity_for_execution_text,
        symbolic_jacobian_structure_cache_stats,
    )

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
            "param scale = 1.0",
            "param k1 = scale",
        ]
    )
    solver_config = {
        "solver": "BDF",
        "grid": {"N": 5},
        "use_sparse_jacobian": True,
        "wegscheider_cyclicity_enabled": False,
    }

    clear_symbolic_jacobian_structure_cache()
    planned = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=mechanism_text,
        solver_config=solver_config,
        parameter_overrides={"scale": 2.0},
    )
    actual = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            mechanism_text=mechanism_text,
            initials={},
            t_span=(0.0, 1.0),
            solver_config=solver_config,
            parameter_overrides={"scale": 2.0},
        )
    )
    actual_identity = getattr(actual.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)

    assert planned["structure_fingerprint"] == actual_identity["structure_fingerprint"]
    assert planned["evaluation_snapshot_fingerprint"] == actual_identity["evaluation_snapshot_fingerprint"]

    changed_planned = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=mechanism_text,
        solver_config=solver_config,
        parameter_overrides={"scale": 3.0},
    )
    stats = symbolic_jacobian_structure_cache_stats()
    assert stats.entries == 1
    assert stats.misses == 1
    assert stats.hits >= 1
    assert planned["structure_fingerprint"] == changed_planned["structure_fingerprint"]
    assert planned["evaluation_snapshot_fingerprint"] != changed_planned["evaluation_snapshot_fingerprint"]


def test_symbolic_structure_cache_reuses_across_base_rate_value_changes():
    from kindred.core.simulation_preparation import (
        clear_symbolic_jacobian_structure_cache,
        symbolic_jacobian_identity_for_execution_text,
        symbolic_jacobian_structure_cache_stats,
    )

    first_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
        ]
    )
    second_text = "\n".join(
        [
            "reaction: A -> B; k=3.0",
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
    first = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=first_text,
        solver_config=solver_config,
    )
    second = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=second_text,
        solver_config=solver_config,
    )
    stats = symbolic_jacobian_structure_cache_stats()

    assert first["structure_fingerprint"] == second["structure_fingerprint"]
    assert first["evaluation_snapshot_fingerprint"] != second["evaluation_snapshot_fingerprint"]
    assert stats.entries == 1
    assert stats.misses == 1
    assert stats.hits == 1


def test_symbolic_structure_cache_reuses_across_constant_temperature_snapshots():
    from kindred.core.simulation_preparation import (
        clear_symbolic_jacobian_structure_cache,
        symbolic_jacobian_identity_for_execution_text,
        symbolic_jacobian_structure_cache_stats,
    )

    mechanism_text = "\n".join(
        [
            "energy=kJ/mol",
            "reaction: A -> B ; A=1e3 ; Ea=50",
            "init: A=1.0, B=0.0",
        ]
    )
    first_solver_config = {
        "solver": "BDF",
        "grid": {"N": 5},
        "use_sparse_jacobian": True,
        "wegscheider_cyclicity_enabled": False,
        "temperature_K": 298.15,
    }
    second_solver_config = dict(first_solver_config, temperature_K=310.0)

    clear_symbolic_jacobian_structure_cache()
    first = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=mechanism_text,
        solver_config=first_solver_config,
    )
    second = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=mechanism_text,
        solver_config=second_solver_config,
    )
    stats = symbolic_jacobian_structure_cache_stats()

    assert first["structure_fingerprint"] == second["structure_fingerprint"]
    assert first["evaluation_snapshot_fingerprint"] != second["evaluation_snapshot_fingerprint"]
    assert stats.entries == 1
    assert stats.misses == 1
    assert stats.hits == 1


def test_symbolic_structure_cache_does_not_poison_supported_run_after_scheduled_temperature_miss():
    from kindred.core.simulation_preparation import (
        clear_symbolic_jacobian_structure_cache,
        symbolic_jacobian_identity_for_execution_text,
        symbolic_jacobian_structure_cache_stats,
    )

    supported_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
        ]
    )
    scheduled_text = "\n".join(
        [
            supported_text,
            "temp_step: t=[0, 1, 2], T=[298, 310]",
        ]
    )
    solver_config = {
        "solver": "BDF",
        "grid": {"N": 5},
        "use_sparse_jacobian": True,
        "wegscheider_cyclicity_enabled": False,
    }

    clear_symbolic_jacobian_structure_cache()
    assert symbolic_jacobian_identity_for_execution_text(
        mechanism_text=scheduled_text,
        solver_config=solver_config,
    ) is None
    supported = symbolic_jacobian_identity_for_execution_text(
        mechanism_text=supported_text,
        solver_config=solver_config,
    )
    stats = symbolic_jacobian_structure_cache_stats()

    assert supported["kind"] == "jacobian"
    assert stats.entries == 1
    assert stats.hits == 0


def test_prepared_run_uses_shared_symbolic_structure_cache_for_actual_artifact():
    from kindred.core.simulation_preparation import (
        clear_symbolic_jacobian_structure_cache,
        symbolic_jacobian_structure_cache_stats,
    )

    first_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
        ]
    )
    second_text = "\n".join(
        [
            "reaction: A -> B; k=3.0",
            "init: A=1.0, B=0.0",
        ]
    )
    request_kwargs = {
        "prepared_payload": None,
        "initials": {},
        "t_span": (0.0, 1.0),
        "solver_config": {
            "solver": "BDF",
            "grid": {"N": 5},
            "use_sparse_jacobian": True,
            "wegscheider_cyclicity_enabled": False,
        },
    }

    clear_symbolic_jacobian_structure_cache()
    first = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            **request_kwargs,
            mechanism_text=first_text,
        )
    )
    second = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            **request_kwargs,
            mechanism_text=second_text,
        )
    )
    first_identity = getattr(first.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    second_identity = getattr(second.request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
    stats = symbolic_jacobian_structure_cache_stats()

    assert first_identity["structure_fingerprint"] == second_identity["structure_fingerprint"]
    assert first_identity["evaluation_snapshot_fingerprint"] != second_identity["evaluation_snapshot_fingerprint"]
    assert stats.entries == 1
    assert stats.misses == 1
    assert stats.hits == 1


def test_symbolic_structure_cache_does_not_store_none_results():
    from kindred.core.symbolic.structure_cache import (
        SymbolicJacobianStructureCache,
        SymbolicJacobianStructureCacheKey,
    )

    cache = SymbolicJacobianStructureCache(max_entries=4)
    key = SymbolicJacobianStructureCacheKey(
        structure_fingerprint="same-structure",
        solver="BDF",
        wegscheider_cyclicity_enabled=False,
        backend_name="sympy",
        backend_version="test",
        profile_version="test",
    )
    calls = 0

    def _unsupported_builder():
        nonlocal calls
        calls += 1
        return None

    assert cache.get_or_build(key, _unsupported_builder) is None
    assert cache.get_or_build(key, _unsupported_builder) is None
    stats = cache.stats()

    assert stats.entries == 0
    assert stats.misses == 2
    assert stats.hits == 0
    assert calls == 2


def test_simulation_request_fingerprint_includes_symbolic_wegscheider_identity():
    from kindred.core.simulator.solvers import SimulationRequest

    base_request = {
        "rhs": lambda _t, y: -y,
        "t_span": (0.0, 1.0),
        "y0": np.array([1.0]),
        "solver": "BDF",
        "grid": {"N": 5},
    }
    first = SimulationRequest(
        **base_request,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "fingerprint": "first",
        },
    )
    second = SimulationRequest(
        **base_request,
        symbolic_wegscheider_identity={
            "kind": "wegscheider_cyclicity",
            "fingerprint": "second",
        },
    )

    assert fingerprint_simulation_request(first) != fingerprint_simulation_request(second)

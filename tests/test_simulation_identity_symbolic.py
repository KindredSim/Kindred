from __future__ import annotations

import pytest

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
            "fingerprint": "first",
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
            "fingerprint": "second",
        },
    )

    assert first.cache_key() != second.cache_key()
    assert first.prepared_runtime_key() != second.prepared_runtime_key()


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

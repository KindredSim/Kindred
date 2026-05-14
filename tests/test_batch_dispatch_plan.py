from __future__ import annotations

from types import SimpleNamespace

import pytest

from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.gui.controllers.batch_dispatch_plan import (
    BatchSetDispatchInput,
    ParallelBatchTaskInput,
    SerialBatchDispatchInput,
    build_batch_set_dispatch_plan,
    build_parallel_batch_task_plan,
    build_serial_batch_dispatch_plan,
)


pytestmark = pytest.mark.unit


def _request_payload(*, prepared_payload=None, initials=None):
    return {
        "prepared_payload": prepared_payload,
        "initials": dict(initials or {"A": 1.0}),
        "t_span": (0.0, 10.0),
        "solver_config": {"solver": "BDF"},
        "mechanism_text": "reaction: A -> B; k=1",
        "simulation_identity": {"schema_id": "schema", "param_fingerprint": "fingerprint"},
    }


def test_explicit_batch_dispatch_plan_drops_prepared_payload_and_uses_batch_algebra_policy():
    gui_plan = SimulationPlan.from_execution_request(
        _request_payload(prepared_payload={"version": 1, "prepared_for": "id1"}, initials={"A": 3.0}),
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "explicit-cache"},
        metadata={"set_id": "id1", "set_name": "set1"},
    ).to_payload()

    plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id="id1",
            set_name="set1",
            fast_mode=False,
            t_end=10.0,
            solver_config={"solver": "BDF"},
            cache_key="explicit-cache",
            scope_identity={"scope": "selected"},
            queue_ids=("id1",),
            initials={"A": 1.0},
            mechanism_text="reaction: A -> B; k=1",
            simulation_identity={"schema_id": "schema", "param_fingerprint": "fingerprint"},
            plan_payload=gui_plan,
            preview_batch_cache_token="",
        )
    )

    assert plan.plan_payload is not None
    submitted_plan = SimulationPlan.from_payload(plan.plan_payload)
    assert submitted_plan.algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT
    request = submitted_plan.to_execution_request().to_payload()
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 3.0}
    assert plan.simulation_identity == {"schema_id": "schema", "param_fingerprint": "fingerprint"}


def test_preview_batch_dispatch_plan_builds_plan_from_set_inputs_and_preview_token():
    plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id="id2",
            set_name="set2",
            fast_mode=True,
            t_end=5.0,
            solver_config={"solver": "BDF"},
            cache_key="preview-cache",
            scope_identity={"scope": "selected"},
            queue_ids=("id1", "id2"),
            initials={"A": 4.0},
            mechanism_text="reaction: A -> B; k=4",
            simulation_identity={"schema_id": "schema-b", "param_fingerprint": "fingerprint-b"},
            plan_payload=None,
            preview_batch_cache_token="preview-token",
        )
    )

    assert plan.plan_payload is not None
    submitted_plan = SimulationPlan.from_payload(plan.plan_payload)
    assert submitted_plan.execution_mode == "preview"
    assert submitted_plan.algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT
    assert submitted_plan.cache_identity_payload == {
        "cache_key": "preview-cache",
        "simulation_identity": {"schema_id": "schema-b", "param_fingerprint": "fingerprint-b"},
        "preview_batch_cache_token": "preview-token",
    }
    assert submitted_plan.cache_scope_payload == {
        "scope_identity": {"scope": "selected"},
        "queue_ids": ["id1", "id2"],
    }
    assert submitted_plan.metadata == {"set_id": "id2", "set_name": "set2", "fast_mode": True}
    request = submitted_plan.to_execution_request().to_payload()
    assert request["prepared_payload"] is None
    assert request["initials"] == {"A": 4.0}
    assert request["mechanism_text"] == "reaction: A -> B; k=4"


def test_batch_dispatch_plan_does_not_scan_raw_text_for_schedule_authority():
    plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id="id1",
            set_name="set1",
            fast_mode=False,
            t_end=5.0,
            solver_config={"solver": "BDF"},
            cache_key="cache",
            scope_identity={"scope": "selected"},
            queue_ids=("id1",),
            initials={"A": 1.0},
            mechanism_text="\n".join(
                [
                    "reaction: A -> B; k=1",
                    "intervention: op=set; species=A; time=0; value=2",
                ]
            ),
            simulation_identity={"schema_id": "schema"},
            plan_payload=None,
            preview_batch_cache_token="",
        )
    )

    assert plan.execution_request is not None
    assert plan.execution_request.get("intervention_schedule") is None
    assert "intervention:" in str(plan.execution_request.get("mechanism_text") or "")


def test_batch_dispatch_plan_preserves_explicit_intervention_schedule_carrier():
    schedule_payload = {
        "instant_events": [{"op": "set", "species": "A", "time": 0.0, "value": 2.0}]
    }

    plan = build_batch_set_dispatch_plan(
        BatchSetDispatchInput(
            set_id="id1",
            set_name="set1",
            fast_mode=False,
            t_end=5.0,
            solver_config={"solver": "BDF"},
            cache_key="cache",
            scope_identity={"scope": "selected"},
            queue_ids=("id1",),
            initials={"A": 1.0},
            mechanism_text="reaction: A -> B; k=1",
            simulation_identity={"schema_id": "schema"},
            plan_payload=None,
            preview_batch_cache_token="",
            intervention_schedule=schedule_payload,
        )
    )

    assert plan.execution_request is not None
    assert plan.execution_request["intervention_schedule"]["instant_events"] == schedule_payload["instant_events"]


def _payload_without_plan_by_set():
    return SimpleNamespace(
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=5.0,
        cache_key="cache",
        fast_mode=True,
        prepared_by_set_id={},
        simulation_plan_by_set_id={},
        mechanism_text_by_set_id={"id1": "reaction: A -> B; k=2"},
        mechanism_signature_by_set_id={"id1": "sig"},
        simulation_identity_by_set_id={"id1": {"schema_id": "schema"}},
        scope_identity={"scope": "selected"},
        preview_batch_cache_token_by_set_id={"id1": "preview-token"},
        run_id=7,
        request_id=11,
    )


def test_serial_dispatch_requires_per_set_simulation_plan():
    with pytest.raises(ValueError, match="Missing simulation plan payload"):
        build_serial_batch_dispatch_plan(
            SerialBatchDispatchInput(
                payload=_payload_without_plan_by_set(),
                queue_ids=("id1",),
                set_id="id1",
                set_name="set1",
                initials={"A": 1.0},
                slider_overrides={"k1": 2.0},
            )
        )


def test_parallel_dispatch_requires_per_set_simulation_plan():
    with pytest.raises(ValueError, match="Missing simulation plan payload"):
        build_parallel_batch_task_plan(
            ParallelBatchTaskInput(
                payload=_payload_without_plan_by_set(),
                set_id="id1",
                set_name="set1",
                queue_ids=("id1",),
                initials={"A": 1.0},
                include_mechanism_in_result_payload=True,
            )
        )

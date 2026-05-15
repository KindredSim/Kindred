from __future__ import annotations

from pathlib import Path

import pytest

from kindred.gui.controllers.batch_run_context_owner import BatchRunContextOwner, BatchRunStartRequest
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_completion_policy import (
    CacheAuthorityState,
    CompletionPolicyContext,
    SimulationCompletionPolicy,
    cache_truth_generation_value,
    next_cache_truth_generation,
)
from tests.batch_context_test_helpers import seed_batch_context


pytestmark = pytest.mark.unit


def _batch_run_start_request(**overrides):
    values = {
        "request_id": 7,
        "run_id": 9,
        "runtime_input_epoch": 3,
        "runtime_input_global_epoch": 4,
        "runtime_input_set_epoch_by_set_id": {"id1": 5},
        "fast_mode": False,
        "reuse_parallel_lane_pool": True,
        "parallel": True,
        "effective_workers": 2,
        "retain_prepared_payloads_in_context": False,
        "prepared_payload": {"shared": {"value": 1}},
        "prepared_payload_by_set_id": {"id1": {"prepared": True}},
        "primary_simulation_plan": {"execution_mode": "explicit", "metadata": {"set_id": "id1"}},
        "simulation_plan_by_set_id": {"id1": {"execution_mode": "explicit", "metadata": {"set_id": "id1"}}},
        "cache_key": "cache-1",
        "scope_identity": {"scope": "selected"},
        "full_dsl": "reaction: A -> B; k=1",
        "mechanism_text_by_set_id": {"id1": "reaction: A -> B; k=1"},
        "mechanism_signature": "sig-primary",
        "mechanism_signature_by_set_id": {"id1": "sig-id1"},
        "simulation_identity_by_set_id": {"id1": {"fingerprint": "before"}},
        "solver_config": {"solver": "BDF"},
        "t_end": 10.0,
        "rows": [0],
        "queue_ids": ["id1"],
        "queue_names": ["set1"],
        "pending_workspace_reset_set_ids": ["id1"],
        "pending_dirty_reset_generation_by_set_id": {"id1": 2},
        "primary_set_id": "id1",
        "pending_init_seed": {"set1": {"A": 1.25}},
        "pending_init_rewrite": "rewrite",
        "pending_init_applied": True,
        "explicit_cache_preview_token": None,
        "explicit_cache_preview_scope_set_ids": ("id1",),
        "explicit_cache_valid_set_ids": ("id1",),
        "explicit_cache_invalidated_set_ids": (),
        "preview_scope_set_ids": None,
        "preview_owner_epoch": 6,
        "preview_batch_cache_token_by_set_id": {"id1": "preview-token"},
    }
    values.update(overrides)
    return BatchRunStartRequest(**values)


def test_inactive_context_has_no_active_batch_state():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=False, parallel=True, queue_ids=["id1"])

    assert owner.active_batch_state() is None


def test_completion_publication_policy_context_reconciles_stale_same_run_cache_truth():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        cache_key="ck",
        queue_ids=["bad", "ok"],
        explicit_cache_valid_set_ids=("bad", "ok"),
    )
    callback_context = owner.callback_context_snapshot()
    captured_policy_context = owner.completion_policy_context(callback_context)
    assert captured_policy_context is not None
    owner.record_scoped_failure(
        set_id="bad",
        failure={"kind": "simulation_error", "message": "bad failed"},
    )

    context = owner.completion_publication_policy_context(
        callback_context=callback_context,
        policy_context=captured_policy_context,
    )

    assert context is not None
    assert context.explicit_cache_valid_set_ids == ("ok",)
    assert context.explicit_cache_invalidated_set_ids == ("bad",)


def test_completion_publication_policy_context_preserves_callback_owned_cache_truth():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        cache_key="ck",
        queue_ids=["id1", "id2"],
        completed_set_ids=["id1"],
        explicit_cache_preview_token="new-token",
        explicit_cache_preview_scope_set_ids=("id2",),
        explicit_cache_valid_set_ids=("id2",),
        explicit_cache_invalidated_set_ids=("id1",),
    )
    base_callback_context = owner.callback_context_snapshot()
    cache_truth_context = owner.completion_policy_context(base_callback_context)
    assert cache_truth_context is not None
    callback_context = owner.callback_context_with_cache_truth(
        base_callback_context,
        cache_truth_context,
    )
    captured_policy_context = owner.completion_policy_context(callback_context)
    assert captured_policy_context is not None
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        cache_key="ck",
        queue_ids=["id1", "id2"],
        completed_set_ids=["id1"],
        explicit_cache_preview_token="old-token",
        explicit_cache_preview_scope_set_ids=("id1",),
        explicit_cache_valid_set_ids=("id1",),
        explicit_cache_invalidated_set_ids=("id1",),
    )

    context = owner.completion_publication_policy_context(
        callback_context=callback_context,
        policy_context=captured_policy_context,
    )

    assert context is not None
    assert context.explicit_cache_preview_token == "new-token"
    assert context.explicit_cache_preview_scope_set_ids == ("id2",)
    assert context.explicit_cache_valid_set_ids == ("id2",)
    assert context.explicit_cache_invalidated_set_ids == ("id1",)


def test_completion_publication_policy_context_keeps_stale_callback_context_out_of_current_truth():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=1,
        request_id=2,
        cache_key="ck",
        queue_ids=["old"],
        explicit_cache_valid_set_ids=("old",),
    )
    callback_context = owner.callback_context_snapshot()
    captured_policy_context = owner.completion_policy_context(callback_context)
    assert captured_policy_context is not None
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=99,
        request_id=100,
        cache_key="other",
        queue_ids=["new"],
        explicit_cache_valid_set_ids=("new",),
        explicit_cache_invalidated_set_ids=("old",),
    )

    context = owner.completion_publication_policy_context(
        callback_context=callback_context,
        policy_context=captured_policy_context,
    )

    assert context is captured_policy_context
    assert context.explicit_cache_valid_set_ids == ("old",)
    assert context.explicit_cache_invalidated_set_ids is None


def test_seeded_context_copies_mutable_input_without_raw_context_reads():
    owner = BatchRunContextOwner()
    queue_ids = ["id1"]
    seed_batch_context(owner, active=True, queue_ids=queue_ids)

    queue_ids.append("id2")

    state = owner.active_batch_state()
    assert state is not None
    assert state.active is True
    assert state.queue_ids == ("id1",)


def test_active_parallel_state_exposes_batch_runtime_inputs_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, fast_mode=False, runtime_waiting=False, run_id=7, request_id=11, rows=[0, 2], queue_ids=["id1", "id2"], queue_names=["Set 1", "Set 2"], pos=1, effective_workers=2)

    state = owner.active_batch_state()

    assert state is not None
    assert state.active is True
    assert state.parallel is True
    assert state.fast_mode is False
    assert state.runtime_waiting is False
    assert state.run_id == 7
    assert state.request_id == 11
    assert state.rows == (0, 2)
    assert state.queue_ids == ("id1", "id2")
    assert state.queue_names == ("Set 1", "Set 2")
    assert state.pos == 1
    assert state.effective_workers == 2


def test_runtime_waiting_state_is_distinct_from_active_running_state():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=False, parallel=True, fast_mode=True, runtime_waiting=True, run_id=7, request_id=11, rows=[0], queue_ids=["id1"], queue_names=["Set 1"])

    state = owner.active_batch_state()

    assert state is not None
    assert state.active is False
    assert state.runtime_waiting is True
    assert state.parallel is True
    assert state.fast_mode is True


def test_active_fast_preview_scope_exposes_queue_or_preview_scope_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, fast_mode=True, parallel=True, queue_ids=[], preview_scope_set_ids=["id2", "id1", "id2"])

    assert owner.active_fast_preview_scope_set_ids() == ("id2", "id1")

    seed_batch_context(owner, active=True, fast_mode=True, parallel=True, queue_ids=["id3"], preview_scope_set_ids=["id2"])

    assert owner.active_fast_preview_scope_set_ids() == ("id3",)

    seed_batch_context(owner, active=True, fast_mode=False, queue_ids=["id1"])

    assert owner.active_fast_preview_scope_set_ids() is None


def test_active_parallel_error_dispatch_context_exposes_failure_metadata_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, fast_mode=True, run_id=7, request_id=11, preview_owner_epoch=13, cache_key="cache-1")

    dispatch = owner.active_parallel_error_dispatch_context()

    assert dispatch is not None
    assert dispatch.run_id == 7
    assert dispatch.request_id == 11
    assert dispatch.fast_mode is True
    assert dispatch.owner_epoch == 13
    assert dispatch.cache_key == "cache-1"
    assert dispatch.callback_context.run_id == 7
    assert dispatch.callback_context.request_id == 11
    assert dispatch.callback_context.cache_key == "cache-1"
    assert dispatch.simulation_identity == {}

    seed_batch_context(owner, active=True, parallel=False, run_id=7)

    assert owner.active_parallel_error_dispatch_context() is None


def test_parallel_start_payload_exposes_execution_inputs_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, fast_mode=False, run_id=7, request_id=11, rows=[0, 2], queue_ids=["id1", "id2"], queue_names=["Set 1", "Set 2"], effective_workers=2, keep_lane_pool_alive=True, preview_owner_epoch=13, active_timeout_s=22.5, cache_key="cache-1", full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=5.0, simulation_plan_by_set_id={"id1": {"metadata": {"set_id": "id1"}}}, mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"}, simulation_identity_by_set_id={"id1": {"fingerprint": "fp-1"}}, scope_identity={"scope": "selected"}, preview_batch_cache_token_by_set_id={"id1": "preview-token"}, pending_init_seed={"Set 1": {"A": 1.0}}, pending_init_applied=False)

    payload = owner.parallel_start_payload()

    assert payload is not None
    assert payload.run_id == 7
    assert payload.request_id == 11
    assert payload.rows == (0, 2)
    assert payload.queue_ids == ("id1", "id2")
    assert payload.queue_names == ("Set 1", "Set 2")
    assert payload.fast_mode is False
    assert payload.effective_workers == 2
    assert payload.keep_lane_pool_alive is True
    assert payload.preview_owner_epoch == 13
    assert payload.active_timeout_s == 22.5
    assert payload.cache_key == "cache-1"
    assert payload.full_dsl == "reaction: A -> B; k=1"
    assert payload.solver_config == {"solver": "BDF"}
    assert payload.t_end == 5.0
    assert payload.simulation_plan_by_set_id == {"id1": {"metadata": {"set_id": "id1"}}}
    assert payload.mechanism_text_by_set_id == {"id1": "reaction: A -> B; k=1"}
    assert payload.simulation_identity_by_set_id == {"id1": {"fingerprint": "fp-1"}}
    assert payload.scope_identity == {"scope": "selected"}
    assert payload.preview_batch_cache_token_by_set_id == {"id1": "preview-token"}
    assert payload.pending_init_seed == {"Set 1": {"A": 1.0}}
    assert payload.pending_init_applied is False


def test_callback_context_snapshot_excludes_heavy_execution_payload_maps():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        fast_mode=False,
        run_id=7,
        request_id=11,
        cache_key="cache-1",
        queue_ids=["id1", "id2"],
        queue_names=["Set 1", "Set 2"],
        completed_set_ids=["id1"],
        failed_set_ids=["id2"],
        stale_runtime_input_set_ids=["id2"],
        runtime_input_epoch=3,
        runtime_input_global_epoch=4,
        runtime_input_set_epoch_by_set_id={"id1": 5, "id2": 6},
        pending_workspace_reset_set_ids=["id1"],
        pending_dirty_reset_generation_by_set_id={"id1": 2},
        primary_set_id="id1",
        keep_lane_pool_alive=True,
        preview_owner_epoch=13,
        simulation_plan_by_set_id={"id1": {"metadata": {"set_id": "id1"}}},
        prepared_by_set_id={"id1": {"prepared": True}},
        mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"},
        mechanism_signature_by_set_id={"id1": "sig-id1"},
        simulation_identity_by_set_id={"id1": {"fingerprint": "fp-1"}},
        preview_batch_cache_token_by_set_id={"id1": "preview-token"},
    )

    callback_context = owner.callback_context_snapshot()

    assert callback_context.run_id == 7
    assert callback_context.request_id == 11
    assert callback_context.cache_key == "cache-1"
    assert callback_context.queue_ids == ("id1", "id2")
    assert callback_context.completed_set_ids == ("id1",)
    assert callback_context.failed_set_ids == ("id2",)
    assert callback_context.stale_runtime_input_set_ids == ("id2",)
    assert callback_context.runtime_input_set_epoch_by_set_id == {"id1": 5, "id2": 6}
    assert callback_context.keep_lane_pool_alive is True

    heavy_names = {
        "simulation_plan_by_set_id",
        "prepared_by_set_id",
        "mechanism_text_by_set_id",
        "mechanism_signature_by_set_id",
        "simulation_identity_by_set_id",
        "preview_batch_cache_token_by_set_id",
    }
    assert heavy_names.isdisjoint(callback_context.__dataclass_fields__)
    assert heavy_names.isdisjoint(getattr(callback_context, "__dict__", {}))


def test_callback_identity_reuses_shared_callback_context_without_deepcopying():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=True,
        run_id=7,
        request_id=11,
        cache_key="cache-1",
        queue_ids=["id1"],
        queue_names=["Set 1"],
    )
    callback_context = owner.callback_context_snapshot()

    identity = SimulationCallbackIdentity.capture(
        run_id=7,
        fast_mode=False,
        request_id=11,
        owner_epoch=13,
        batch_set="Set 1",
        batch_set_id="id1",
        cache_key="cache-1",
        callback_context=callback_context,
        simulation_identity={},
    )

    assert identity.callback_context is callback_context
    assert not hasattr(identity, "context_snapshot")
    assert not hasattr(identity, "policy_context")
    assert identity.batch_set == "Set 1"
    assert identity.batch_set_id == "id1"


def test_serial_next_payload_exposes_current_set_and_execution_inputs_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, fast_mode=True, pos=1, rows=[0, 2], queue_ids=["id1", "id2"], queue_names=["Set 1", "Set 2"], request_id=11, cache_key="cache-1", preview_owner_epoch=13, full_dsl="reaction: A -> B; k=1", solver_config={"solver": "BDF"}, t_end=5.0, simulation_plan={"metadata": {"set_id": "fallback"}}, simulation_plan_by_set_id={"id2": {"metadata": {"set_id": "id2"}}}, mechanism_text_by_set_id={"id2": "reaction: A -> C; k=2"}, mechanism_signature_by_set_id={"id2": "sig-id2"}, simulation_identity_by_set_id={"id2": {"fingerprint": "fp-2"}}, prepared={"prepared": "fallback"}, prepared_by_set_id={"id2": {"prepared": "id2"}}, scope_identity={"scope": "selected"}, preview_batch_cache_token_by_set_id={"id2": "preview-token"}, pending_init_seed={"Set 2": {"A": 1.0}}, pending_init_applied=False)

    payload = owner.serial_next_payload()

    assert payload is not None
    assert payload.pos == 1
    assert payload.total == 2
    assert payload.row == 2
    assert payload.set_id == "id2"
    assert payload.set_name == "Set 2"
    assert payload.queue_ids == ("id1", "id2")
    assert payload.fast_mode is True
    assert payload.request_id == 11
    assert payload.cache_key == "cache-1"
    assert payload.preview_owner_epoch == 13
    assert payload.full_dsl == "reaction: A -> B; k=1"
    assert payload.solver_config == {"solver": "BDF"}
    assert payload.t_end == 5.0
    assert payload.simulation_plan == {"metadata": {"set_id": "fallback"}}
    assert payload.simulation_plan_by_set_id == {"id2": {"metadata": {"set_id": "id2"}}}
    assert payload.mechanism_text_by_set_id == {"id2": "reaction: A -> C; k=2"}
    assert payload.mechanism_signature_by_set_id == {"id2": "sig-id2"}
    assert payload.simulation_identity_by_set_id == {"id2": {"fingerprint": "fp-2"}}
    assert payload.prepared == {"prepared": "fallback"}
    assert payload.prepared_by_set_id == {"id2": {"prepared": "id2"}}
    assert payload.scope_identity == {"scope": "selected"}
    assert payload.preview_batch_cache_token_by_set_id == {"id2": "preview-token"}
    assert payload.pending_init_seed == {"Set 2": {"A": 1.0}}
    assert payload.pending_init_applied is False

    seed_batch_context(owner, active=True, parallel=True, pos=0, queue_ids=["id1"])

    assert owner.serial_next_payload() is None


def test_completion_state_exposes_cursor_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, fast_mode=False, keep_lane_pool_alive=False, pos=1, total=3, queue_ids=["id1", "id2", "id3"], queue_names=["Set 1", "Set 2", "Set 3"], completed_set_ids=["id1"])

    state = owner.completion_state()

    assert state is not None
    assert state.active is True
    assert state.parallel is False
    assert state.fast_mode is False
    assert state.keep_lane_pool_alive is False
    assert state.pos == 1
    assert state.total == 3
    assert state.queue_ids == ("id1", "id2", "id3")
    assert state.queue_names == ("Set 1", "Set 2", "Set 3")
    assert state.completed_set_ids == ("id1",)
    assert state.completed_count == 1


def test_completion_state_returns_none_for_deactivated_context():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=False, parallel=False, queue_ids=["id1"], completed_set_ids=[])

    assert owner.completion_state() is None


def test_completion_cleanup_and_flush_contexts_expose_metadata_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=False, fast_mode=False, parallel=True, keep_lane_pool_alive=True, cache_key="cache-1", request_id=11, run_id=7)

    cleanup = owner.completion_cleanup_state()
    flush = owner.completion_flush_context()

    assert cleanup.fast_mode is False
    assert cleanup.parallel is True
    assert cleanup.keep_lane_pool_alive is True
    assert flush.cache_key == "cache-1"
    assert flush.request_id == 11
    assert flush.run_id == 7


def test_scoped_failure_cache_state_exposes_cache_validity_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, cache_key="cache-1", explicit_cache_valid_set_ids=("id1", "id3"), explicit_cache_invalidated_set_ids=("id2",), failed_set_ids=("id2",))

    state = owner.scoped_failure_cache_state()

    assert state.cache_key == "cache-1"
    assert state.explicit_cache_valid_set_ids == ("id1", "id3")
    assert state.explicit_cache_invalidated_set_ids == ("id2",)
    assert state.failed_count == 1


def test_pending_dirty_reset_state_exposes_owner_normalized_reset_truth():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, pending_workspace_reset_set_ids=["id1", "", "id2", "id1"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2, "": 3})

    state = owner.pending_dirty_reset_state()

    assert state.set_ids == ("id1", "id2")
    assert state.generation_by_set_id == {"id1": 1, "id2": 2}
    assert state.empty is False


def test_batch_payload_queries_expose_plan_and_preview_token_without_raw_context_reads():
    owner = BatchRunContextOwner()
    plan_payload = {"metadata": {"set_id": "id1"}, "execution_mode": "explicit"}
    seed_batch_context(owner, prepared={"fallback": True}, prepared_by_set_id={"id1": {"prepared": True}}, simulation_plan_by_set_id={"id1": plan_payload}, mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"}, solver_config={"solver": "BDF"}, preview_batch_cache_token_by_set_id={"id1": "preview-token"})

    state = owner.execution_payload_state()
    payload = owner.simulation_plan_payload_for_set("id1")
    token = owner.preview_batch_cache_token_for_set("id1")

    assert state.prepared == {"fallback": True}
    assert state.prepared_by_set_id == {"id1": {"prepared": True}}
    assert state.simulation_plan_by_set_id == {"id1": plan_payload}
    assert state.mechanism_text_by_set_id == {"id1": "reaction: A -> B; k=1"}
    assert state.solver_config == {"solver": "BDF"}
    assert payload == plan_payload
    assert payload is not plan_payload
    assert token == "preview-token"
    assert owner.simulation_plan_payload_for_set("missing") == {}
    assert owner.preview_batch_cache_token_for_set("missing") == ""


def test_deactivate_if_active_owns_context_activity_transition():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, queue_ids=["old"])

    inactive = owner.deactivate_if_active({
        "active": True,
        "queue_ids": ["id1"],
        "pos": "2",
    })

    assert inactive["active"] is False
    assert inactive["queue_ids"] == ["id1"]
    assert owner.completion_policy_context().active is False
    assert owner.completion_policy_context().queue_ids == ("id1",)

    snapshot = owner.deactivate_if_active({"active": False, "queue_ids": ["id2"]})

    assert snapshot == {"active": False, "queue_ids": ["id2"]}
    assert owner.completion_policy_context().queue_ids == ("id1",)


def test_deactivate_if_active_ignores_stale_identity_context():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, run_id=5, request_id=7, cache_key="current")

    inactive = owner.deactivate_if_active({
        "active": True,
        "run_id": 5,
        "request_id": 7,
        "cache_key": "stale",
    })

    assert inactive["active"] is True
    assert inactive["cache_key"] == "current"


def test_deactivate_if_active_ignores_stale_same_identity_progress_context():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        run_id=5,
        request_id=7,
        cache_key="ck",
        queue_ids=["id1", "id2"],
        completed_set_ids=["id1"],
        runtime_input_global_epoch=2,
    )

    inactive = owner.deactivate_if_active({
        "active": True,
        "run_id": 5,
        "request_id": 7,
        "cache_key": "ck",
        "queue_ids": ["id1", "id2"],
        "completed_set_ids": [],
        "runtime_input_global_epoch": 1,
    })

    assert inactive["active"] is True
    assert inactive["completed_set_ids"] == ["id1"]
    assert inactive["runtime_input_global_epoch"] == 2


def test_context_matches_current_run_identity_ignores_progress_but_rejects_cache_turnover():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        run_id=5,
        request_id=7,
        cache_key="ck",
        completed_set_ids=["id1"],
        runtime_input_global_epoch=2,
    )

    assert owner.context_matches_current_run_identity({
        "run_id": 5,
        "request_id": 7,
        "cache_key": "ck",
        "completed_set_ids": [],
        "runtime_input_global_epoch": 1,
    })
    assert not owner.context_matches_current_run_identity({
        "run_id": 5,
        "request_id": 7,
        "cache_key": "other",
    })


def test_simulation_controller_tests_do_not_read_pending_reset_raw_context_fields():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current().get("pending_workspace_reset_set_ids")' not in source
    assert 'current()["pending_workspace_reset_set_ids"]' not in source
    assert 'current().get("pending_dirty_reset_generation_by_set_id")' not in source
    assert 'current()["pending_dirty_reset_generation_by_set_id"]' not in source


def test_simulation_controller_dirty_reset_finalization_uses_owner_pending_reset_state():
    import inspect

    from kindred.gui.controllers.simulation_controller import SimulationController

    source = inspect.getsource(SimulationController._finalize_explicit_batch_dirty_reset)

    assert 'ctx.get("pending_workspace_reset_set_ids")' not in source
    assert 'ctx.get("pending_dirty_reset_generation_by_set_id")' not in source


def test_simulation_controller_batch_activity_checks_use_owner_queries():
    import inspect

    from kindred.gui.controllers.simulation_controller import SimulationController

    source = inspect.getsource(SimulationController)

    assert 'ctx.get("active")' not in source
    assert 'ctx["active"]' not in source


def test_simulation_controller_uses_owner_for_batch_payload_map_queries():
    import inspect

    from kindred.gui.controllers.simulation_controller import SimulationController

    source = inspect.getsource(SimulationController)

    assert "_simulation_plan_for_set_from_context" not in source
    assert "_preview_batch_cache_token_for_cached_result" not in source
    assert 'context.get("simulation_plan_by_set_id")' not in source
    assert 'context.get("preview_batch_cache_token_by_set_id")' not in source


def test_simulation_controller_tests_do_not_read_completion_failure_raw_context_fields():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current()["failed_set_ids"]' not in source
    assert 'current()["failed_set_errors"]' not in source
    assert 'current()["completed_set_ids"]' not in source
    assert 'current()["explicit_cache_valid_set_ids"]' not in source
    assert 'current()["explicit_cache_invalidated_set_ids"]' not in source
    assert 'current()["pending_init_applied"]' not in source


def test_simulation_controller_tests_do_not_read_batch_activity_raw_context_fields():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current().get("active")' not in source
    assert 'current()["active"]' not in source
    assert 'current()["pos"]' not in source


def test_simulation_controller_tests_do_not_read_cache_key_raw_context_field():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current()["cache_key"]' not in source


def test_simulation_controller_tests_do_not_read_preview_token_raw_context_field():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current()["preview_batch_cache_token_by_set_id"]' not in source


def test_simulation_controller_tests_do_not_read_execution_payload_raw_context_fields():
    from pathlib import Path

    source = Path("tests/test_simulation_controller.py").read_text(encoding="utf-8")

    assert 'current()["prepared_by_set_id"]' not in source
    assert 'current().get("prepared_by_set_id"' not in source
    assert 'current()["simulation_plan_by_set_id"]' not in source
    assert 'current().get("simulation_plan_by_set_id"' not in source
    assert 'current()["prepared"]' not in source
    assert 'current().get("prepared")' not in source
    assert 'current()["mechanism_text_by_set_id"]' not in source
    assert 'current()["solver_config"]' not in source


def test_active_runtime_input_staleness_uses_owned_context_and_supplied_current_epochs():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, runtime_input_global_epoch=4, runtime_input_epoch=8, runtime_input_set_epoch_by_set_id={"id1": 2})

    assert owner.active_runtime_input_stale_for_set(
        batch_set_id="id1",
        current_global_epoch=4,
        current_set_epoch=2,
        current_epoch=8,
    ) is False
    assert owner.active_runtime_input_stale_for_set(
        batch_set_id="id1",
        current_global_epoch=4,
        current_set_epoch=3,
        current_epoch=8,
    ) is True
    assert owner.active_runtime_input_stale_for_set(
        batch_set_id="id1",
        current_global_epoch=5,
        current_set_epoch=2,
        current_epoch=8,
    ) is True

    seed_batch_context(owner, active=False, runtime_input_global_epoch=4)

    assert owner.active_runtime_input_stale_for_set(
        batch_set_id="id1",
        current_global_epoch=5,
        current_set_epoch=2,
        current_epoch=8,
    ) is False


def test_consume_stale_serial_queue_prefix_uses_owned_context_and_current_epoch_map():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, rows=[0, 1, 2], queue_ids=["id1", "id2", "id3"], pos=0, runtime_input_global_epoch=4, runtime_input_epoch=8, runtime_input_set_epoch_by_set_id={"id1": 1, "id2": 2, "id3": 3}, pending_workspace_reset_set_ids=["id1", "id2", "id3"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 1, "id3": 1})

    transition = owner.consume_stale_serial_queue_prefix_for_current_epochs(
        current_global_epoch=4,
        current_set_epoch_by_set_id={"id1": 9, "id2": 8, "id3": 3},
        current_epoch=8,
    )

    assert transition.completed_count == 2
    assert transition.batch_done is False
    assert transition.context["pos"] == 2
    assert transition.context["completed_set_ids"] == ["id1", "id2"]
    assert transition.context["stale_runtime_input_set_ids"] == ["id1", "id2"]
    assert transition.context["pending_workspace_reset_set_ids"] == ["id3"]
    assert transition.context["pending_dirty_reset_generation_by_set_id"] == {"id3": 1}


def test_completion_metadata_queries_expose_cache_primary_and_identity_without_raw_dict_reads():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, fast_mode=False, total=3, cache_key="cache-1", primary_set_id="id2", simulation_identity_by_set_id={
            "id2": {"fingerprint": "fp-2"},
        })

    assert owner.explicit_batch_coalescing_for_completion(slider_triggered=False) is True
    assert owner.explicit_batch_coalescing_for_completion(slider_triggered=True) is False
    assert owner.completion_cache_key() == "cache-1"
    assert owner.primary_set_id() == "id2"
    assert owner.simulation_identity_for_set("id2") == {"fingerprint": "fp-2"}
    assert owner.simulation_identity_for_set("missing") == {}
    assert owner.active_fast_mode() is False


def test_start_run_owns_batch_context_start_policy_and_defensive_copies_payloads():
    owner = BatchRunContextOwner()
    prepared_payload = {"shared": {"value": 1}}
    plan_by_set_id = {"id1": {"execution_mode": "explicit", "metadata": {"set_id": "id1"}}}
    simulation_identity_by_set_id = {"id1": {"fingerprint": "before"}}

    context = owner.start_run(
        _batch_run_start_request(
            prepared_payload=prepared_payload,
            prepared_payload_by_set_id={"id1": {"prepared": True}},
            simulation_plan_by_set_id=plan_by_set_id,
            simulation_identity_by_set_id=simulation_identity_by_set_id,
        )
    )

    prepared_payload["shared"]["value"] = 99
    plan_by_set_id["id1"]["metadata"]["set_id"] = "mutated"
    simulation_identity_by_set_id["id1"]["fingerprint"] = "after"

    assert context["active"] is True
    assert context["parallel"] is True
    assert context["keep_lane_pool_alive"] is True
    assert context["prepared"] is None
    assert context["prepared_by_set_id"] == {}
    assert context["simulation_plan_by_set_id"]["id1"]["metadata"]["set_id"] == "id1"
    assert context["simulation_identity_by_set_id"]["id1"]["fingerprint"] == "before"
    assert "execution_request" not in context
    assert "execution_request_by_set_id" not in context
    execution_state = owner.execution_payload_state()
    assert execution_state.prepared is None
    assert execution_state.prepared_by_set_id == {}
    assert execution_state.simulation_plan_by_set_id["id1"]["metadata"]["set_id"] == "id1"
    assert owner.simulation_identity_for_set("id1")["fingerprint"] == "before"


def test_start_run_keeps_serial_explicit_prepared_payloads_when_context_owns_serial_queue():
    owner = BatchRunContextOwner()

    context = owner.start_run(
        _batch_run_start_request(
            request_id=11,
            run_id=None,
            runtime_input_epoch=1,
            runtime_input_global_epoch=1,
            runtime_input_set_epoch_by_set_id={"id1": 1},
            fast_mode=False,
            reuse_parallel_lane_pool=False,
            parallel=False,
            effective_workers=1,
            retain_prepared_payloads_in_context=True,
            prepared_payload={"prepared": "primary"},
            prepared_payload_by_set_id={"id1": {"prepared": "id1"}},
            primary_simulation_plan={"execution_mode": "explicit", "metadata": {"set_id": "id1"}},
            simulation_plan_by_set_id={"id1": {"execution_mode": "explicit", "metadata": {"set_id": "id1"}}},
            cache_key="cache-serial",
            scope_identity={"scope": "selected"},
            full_dsl="reaction: A -> B; k=1",
            mechanism_text_by_set_id={"id1": "reaction: A -> B; k=1"},
            mechanism_signature="sig-primary",
            mechanism_signature_by_set_id={"id1": "sig-id1"},
            simulation_identity_by_set_id={"id1": {"fingerprint": "id1"}},
            solver_config={"solver": "BDF"},
            t_end=10.0,
            rows=[0],
            queue_ids=["id1"],
            queue_names=["set1"],
            pending_workspace_reset_set_ids=[],
            pending_dirty_reset_generation_by_set_id={},
            primary_set_id="id1",
            pending_init_seed={},
            pending_init_rewrite=None,
            pending_init_applied=False,
            explicit_cache_preview_token=None,
            explicit_cache_preview_scope_set_ids=None,
            explicit_cache_valid_set_ids=("id1",),
            explicit_cache_invalidated_set_ids=(),
            preview_scope_set_ids=None,
            preview_owner_epoch=None,
            preview_batch_cache_token_by_set_id={},
        )
    )

    assert context["active"] is True
    assert context["parallel"] is False
    assert context["keep_lane_pool_alive"] is False
    assert context["prepared"] == {"prepared": "primary"}
    assert context["prepared_by_set_id"] == {"id1": {"prepared": "id1"}}
    assert context["simulation_plan"] == {
        "execution_mode": "explicit",
        "metadata": {"set_id": "id1"},
    }


def test_record_parallel_success_owns_completion_progress_and_deactivates_at_total():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, queue_ids=["id1", "id2"], completed_set_ids=[])

    first = owner.record_parallel_success(set_id="id1", total=2)

    assert first.completed_count == 1
    assert first.batch_done is False
    assert first.context["active"] is True
    assert first.context["completed_set_ids"] == ["id1"]

    second = owner.record_parallel_success(set_id="id2", total=2)

    assert second.completed_count == 2
    assert second.batch_done is True
    assert second.context["active"] is False
    assert second.context["completed_set_ids"] == ["id1", "id2"]


def test_record_serial_success_owns_queue_position_and_completion_transition():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, queue_ids=["id1", "id2"], pos=0, total=2)

    first = owner.record_serial_success(set_id="id1")

    assert first.completed_count == 1
    assert first.batch_done is False
    assert first.context["active"] is True
    assert first.context["pos"] == 1

    second = owner.record_serial_success(set_id="id2")

    assert second.completed_count == 2
    assert second.batch_done is True
    assert second.context["active"] is False
    assert second.context["pos"] == 2


def test_record_scoped_failure_owns_failed_set_and_cache_validity_state():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, queue_ids=["id1", "id2", "id3"], completed_set_ids=["id1"], failed_set_ids=[], failed_set_errors={}, pending_workspace_reset_set_ids=["id1", "id2", "id3"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2, "id3": 3}, explicit_cache_valid_set_ids=("id1", "id2", "id3"), explicit_cache_invalidated_set_ids=())

    failure = {"kind": "simulation_error", "message": "bad set"}
    outcome = owner.record_scoped_failure(set_id="id2", failure=failure)

    assert outcome.completed_count == 2
    assert outcome.context["completed_set_ids"] == ["id1", "id2"]
    assert outcome.context["failed_set_ids"] == ["id2"]
    assert outcome.context["failed_set_errors"] == {"id2": failure}
    assert outcome.context["pending_workspace_reset_set_ids"] == ["id1", "id3"]
    assert outcome.context["pending_dirty_reset_generation_by_set_id"] == {"id1": 1, "id3": 3}
    assert outcome.context["explicit_cache_valid_set_ids"] == ("id1", "id3")
    assert outcome.context["explicit_cache_invalidated_set_ids"] == ("id2",)


def test_completion_summary_separates_truthful_success_stale_and_failed_sets():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, fast_mode=False, completed_set_ids=["id1", "id2", "id3"], stale_runtime_input_set_ids=["id2"], failed_set_ids=["id3"], failed_set_errors={"id3": {"kind": "simulation_error", "message": "bad"}})

    summary = owner.completion_summary()

    assert summary.fast_mode is False
    assert summary.has_truthful_success is True
    assert summary.failed_set_ids == ("id3",)
    assert summary.failed_errors == {"id3": {"kind": "simulation_error", "message": "bad"}}


def test_runtime_input_staleness_is_decided_from_context_epochs():
    owner = BatchRunContextOwner()
    context = {
        "runtime_input_global_epoch": 2,
        "runtime_input_set_epoch_by_set_id": {"id1": 5},
        "runtime_input_epoch": 8,
    }

    assert owner.runtime_input_stale_for_set(
        context,
        batch_set_id="id1",
        current_global_epoch=3,
        current_set_epoch=5,
        current_epoch=8,
    ) is True
    assert owner.runtime_input_stale_for_set(
        context,
        batch_set_id="id1",
        current_global_epoch=2,
        current_set_epoch=6,
        current_epoch=8,
    ) is True
    assert owner.runtime_input_stale_for_set(
        {"runtime_input_epoch": 7},
        batch_set_id="id1",
        current_global_epoch=0,
        current_set_epoch=0,
        current_epoch=8,
    ) is True
    assert owner.runtime_input_stale_for_set(
        context,
        batch_set_id="id1",
        current_global_epoch=2,
        current_set_epoch=5,
        current_epoch=8,
    ) is False


def test_mark_stale_runtime_input_set_consumed_updates_queue_and_dirty_reset_state():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, completed_set_ids=["id1"], stale_runtime_input_set_ids=[], pending_workspace_reset_set_ids=["id1", "id2"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2}, pos=0)

    context = owner.mark_stale_runtime_input_set_consumed(set_id="id2", next_pos=2)

    assert context["completed_set_ids"] == ["id1", "id2"]
    assert context["stale_runtime_input_set_ids"] == ["id2"]
    assert context["pending_workspace_reset_set_ids"] == ["id1"]
    assert context["pending_dirty_reset_generation_by_set_id"] == {"id1": 1}
    assert context["pos"] == 2


def test_record_active_serial_runtime_input_superseded_owns_cursor_advance():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, completed_set_ids=[], stale_runtime_input_set_ids=[], pending_workspace_reset_set_ids=["id1", "id2"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2}, queue_ids=["id1", "id2"], pos=0)

    context = owner.record_active_serial_runtime_input_superseded(active_set_id="id1")

    assert context["pos"] == 1
    assert context["completed_set_ids"] == ["id1"]
    assert context["stale_runtime_input_set_ids"] == ["id1"]
    assert context["pending_workspace_reset_set_ids"] == ["id2"]
    assert context["pending_dirty_reset_generation_by_set_id"] == {"id2": 2}


def test_consume_stale_serial_queue_prefix_owns_repeated_cursor_consumption():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=False, completed_set_ids=[], stale_runtime_input_set_ids=[], pending_workspace_reset_set_ids=["id1", "id2", "id3"], pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 2, "id3": 3}, queue_ids=["id1", "id2", "id3"], rows=[0, 1, 2], pos=0)

    transition = owner.consume_stale_serial_queue_prefix(
        is_stale_set=lambda set_id: set_id in {"id1", "id2"},
    )

    assert transition.completed_count == 2
    assert transition.batch_done is False
    assert transition.context["active"] is True
    assert transition.context["pos"] == 2
    assert transition.context["completed_set_ids"] == ["id1", "id2"]
    assert transition.context["stale_runtime_input_set_ids"] == ["id1", "id2"]
    assert transition.context["pending_workspace_reset_set_ids"] == ["id3"]
    assert transition.context["pending_dirty_reset_generation_by_set_id"] == {"id3": 3}


def test_record_parallel_stale_callback_consumed_owns_completion_deactivation():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, queue_ids=["id1", "id2"], total=2, completed_set_ids=["id1"], stale_runtime_input_set_ids=[])

    transition = owner.record_parallel_stale_callback_consumed(set_id="id2")

    assert transition.completed_count == 2
    assert transition.batch_done is True
    assert transition.context["active"] is False
    assert transition.context["completed_set_ids"] == ["id1", "id2"]
    assert transition.context["stale_runtime_input_set_ids"] == ["id2"]


def test_record_parallel_stale_callback_consumed_ignores_inactive_or_serial_context():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=False, parallel=True, queue_ids=["id1"], completed_set_ids=[])

    assert owner.record_parallel_stale_callback_consumed_if_active(set_id="id1") is None
    state = owner.completion_state()
    assert state is None

    seed_batch_context(owner, active=True, parallel=False, queue_ids=["id1"], completed_set_ids=[])

    assert owner.record_parallel_stale_callback_consumed_if_active(set_id="id1") is None
    state = owner.completion_state()
    assert state is not None
    assert state.completed_set_ids == ()


def test_runtime_waiting_transition_deactivates_and_then_restores_context_flag():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active=True, parallel=True, runtime_waiting=False)

    waiting = owner.mark_runtime_waiting(required_lanes=3)

    assert waiting["active"] is False
    assert waiting["runtime_waiting"] is True
    assert waiting["runtime_waiting_required_lanes"] == 3

    ready = owner.clear_runtime_waiting()

    assert ready["active"] is False
    assert "runtime_waiting" not in ready
    assert "runtime_waiting_required_lanes" not in ready


def test_completion_policy_context_conversion_is_owned_by_batch_context_owner():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, active="false", fast_mode="false", parallel="false", keep_lane_pool_alive="false", request_id=float("inf"), run_id=float("-inf"), total="bad-total", pos="bad-pos", queue_ids="id1", queue_names="set1", pending_init_applied="false", pending_dirty_reset_generation_by_set_id={"id1": "bad-generation"})

    ctx = owner.completion_policy_context()

    assert ctx is not None
    assert ctx.active is False
    assert ctx.fast_mode is False
    assert ctx.parallel is False
    assert ctx.keep_lane_pool_alive is False
    assert ctx.request_id is None
    assert ctx.run_id is None
    assert ctx.total == 0
    assert ctx.pos == 0
    assert ctx.queue_ids == ("id1",)
    assert ctx.queue_names == ("set1",)
    assert ctx.pending_init_applied is False
    assert ctx.pending_dirty_reset_generation_by_set_id == {}


def test_completion_policy_context_serialization_preserves_unowned_context_fields():
    owner = BatchRunContextOwner()
    seed_batch_context(owner, cache_key="existing-cache", active=True, queue_ids=["old"])
    context = CompletionPolicyContext(
        active=False,
        request_id=7,
        run_id=9,
        fast_mode=False,
        parallel=True,
        keep_lane_pool_alive=True,
        queue_ids=("id1", "id2"),
        queue_names=("set1", "set2"),
        total=2,
        pos=1,
        primary_set_id="id1",
        completed_set_ids=("id1",),
        pending_workspace_reset_set_ids=("id2",),
        pending_dirty_reset_generation_by_set_id={"id2": 3},
        pending_init_seed={"set2": {"A": 1.5}},
        pending_init_rewrite="rewrite",
        pending_init_applied=True,
        explicit_cache_preview_token="preview-token",
        explicit_cache_preview_scope_set_ids=("id1",),
        explicit_cache_valid_set_ids=("id1",),
        explicit_cache_invalidated_set_ids=("id2",),
        preview_scope_set_ids=("id1", "id2"),
        preview_owner_epoch=4,
    )

    serialized = owner.serialize_completion_policy_context(context)

    assert serialized["cache_key"] == "existing-cache"
    assert serialized["active"] is False
    assert serialized["queue_ids"] == ["id1", "id2"]
    assert serialized["pending_init_seed"] == {"set2": {"A": 1.5}}
    assert serialized["preview_owner_epoch"] == 4
    policy_context = owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is False
    assert policy_context.queue_ids == ("id1", "id2")
    assert policy_context.pending_init_seed == {"set2": {"A": 1.5}}
    assert policy_context.preview_owner_epoch == 4


def test_completion_policy_serialization_from_callback_context_preserves_serial_execution_payload():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=9,
        request_id=7,
        cache_key="ck",
        pos=0,
        total=2,
        rows=[4, 8],
        queue_ids=["id1", "id2"],
        queue_names=["set1", "set2"],
        full_dsl="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        t_end=12.5,
        simulation_plan_by_set_id={"id2": {"metadata": {"set_id": "id2"}}},
        mechanism_text_by_set_id={"id2": "reaction: A -> C; k=2"},
        mechanism_signature_by_set_id={"id2": "sig-id2"},
        simulation_identity_by_set_id={"id2": {"fingerprint": "fp-2"}},
        prepared_by_set_id={"id2": {"prepared": "id2"}},
        scope_identity={"scope": "selected"},
        pending_init_seed={},
        pending_init_applied=True,
    )
    callback_context = owner.callback_context_snapshot()
    policy_context = owner.completion_policy_context(callback_context)
    assert policy_context is not None

    owner.serialize_completion_policy_context(policy_context, base_context=callback_context)
    owner.record_serial_success(set_id="id1")
    payload = owner.serial_next_payload()

    assert payload is not None
    assert payload.pos == 1
    assert payload.row == 8
    assert payload.set_id == "id2"
    assert payload.full_dsl == "reaction: A -> B; k=1"
    assert payload.solver_config == {"solver": "BDF"}
    assert payload.t_end == 12.5
    assert payload.simulation_plan_by_set_id == {"id2": {"metadata": {"set_id": "id2"}}}
    assert payload.mechanism_text_by_set_id == {"id2": "reaction: A -> C; k=2"}
    assert payload.mechanism_signature_by_set_id == {"id2": "sig-id2"}
    assert payload.simulation_identity_by_set_id == {"id2": {"fingerprint": "fp-2"}}
    assert payload.prepared_by_set_id == {"id2": {"prepared": "id2"}}
    assert payload.scope_identity == {"scope": "selected"}


def test_completion_policy_context_serialization_does_not_overwrite_newer_same_identity_progress():
    owner = BatchRunContextOwner()
    seed_batch_context(
        owner,
        active=True,
        run_id=9,
        request_id=7,
        cache_key="ck",
        queue_ids=["id1", "id2"],
        completed_set_ids=["id1"],
        runtime_input_global_epoch=2,
    )
    stale_base = {
        "active": True,
        "run_id": 9,
        "request_id": 7,
        "cache_key": "ck",
        "queue_ids": ["id1", "id2"],
        "queue_names": ["set1", "set2"],
        "completed_set_ids": [],
        "runtime_input_global_epoch": 1,
    }
    context = CompletionPolicyContext(
        active=True,
        request_id=7,
        run_id=9,
        fast_mode=False,
        parallel=True,
        keep_lane_pool_alive=False,
        queue_ids=("id1", "id2"),
        queue_names=("set1", "set2"),
        total=2,
        pos=0,
        primary_set_id="id1",
        completed_set_ids=(),
    )

    serialized = owner.serialize_completion_policy_context(context, base_context=stale_base)

    assert serialized["completed_set_ids"] == []
    assert serialized["runtime_input_global_epoch"] == 1
    current_policy = owner.completion_policy_context()
    assert current_policy is not None
    assert current_policy.completed_set_ids == ("id1",)
    assert owner.callback_context_snapshot().runtime_input_global_epoch == 2


def test_success_cache_truth_update_advances_generation_for_callback_reconciliation():
    policy = SimulationCompletionPolicy()
    context = CompletionPolicyContext(
        active=True,
        request_id=7,
        run_id=9,
        fast_mode=False,
        parallel=True,
        keep_lane_pool_alive=False,
        queue_ids=("id1", "id2"),
        explicit_cache_valid_set_ids=("id1",),
        explicit_cache_invalidated_set_ids=("id2",),
        explicit_cache_truth_generation=0,
    )
    cache_state = CacheAuthorityState(
        active_cache_key="cache-key",
        active_cache_preview_token="token-b",
        active_cache_preview_scope_set_ids=("id1", "id2"),
        active_cache_valid_set_ids=("id1", "id2"),
        active_cache_invalidated_set_ids=(),
    )

    updated = policy.build_context_update_from_cache_truth(
        context=context,
        cache_state=cache_state,
        cache_key="cache-key",
    )

    assert updated.explicit_cache_valid_set_ids == ("id1", "id2")
    assert updated.explicit_cache_invalidated_set_ids == ()
    assert updated.explicit_cache_truth_generation == 1


def test_cache_truth_generation_uses_shared_policy_helper():
    assert cache_truth_generation_value(None) == 0
    assert cache_truth_generation_value("bad") == 0
    assert next_cache_truth_generation(None) == 1
    assert next_cache_truth_generation("4") == 5

    owner_source = Path("kindred/gui/controllers/batch_run_context_owner.py").read_text(encoding="utf-8")
    policy_source = Path("kindred/gui/controllers/simulation_completion_policy.py").read_text(encoding="utf-8")

    assert "def _cache_truth_generation_value" not in owner_source
    assert "cache_truth_generation_value(" in owner_source
    assert "next_cache_truth_generation(" in owner_source
    assert "def cache_truth_generation_value" in policy_source
    assert "def next_cache_truth_generation" in policy_source

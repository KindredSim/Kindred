from __future__ import annotations

from kindred.gui.controllers.simulation_completion_policy import (
    CacheAuthorityState,
    CompletionPolicyContext,
    DirtySetState,
    PendingReplayDirective,
    PendingReplayState,
    RunActivitySnapshot,
    SimulationCompletionPolicy,
    pending_initial_seed_for_set,
)


def _context(**overrides) -> CompletionPolicyContext:
    base = CompletionPolicyContext(
        active=True,
        request_id=4,
        run_id=7,
        fast_mode=True,
        parallel=False,
        keep_executor_alive=False,
        queue_ids=("id1",),
        queue_names=("set1",),
        total=1,
        pos=0,
        primary_set_id="id1",
        completed_set_ids=(),
        pending_workspace_reset_set_ids=(),
        pending_dirty_reset_generation_by_set_id={},
        pending_init_seed={},
        pending_init_rewrite=None,
        pending_init_applied="false",
        explicit_cache_preview_token=None,
        explicit_cache_preview_scope_set_ids=None,
        explicit_cache_valid_set_ids=None,
        explicit_cache_invalidated_set_ids=None,
        preview_scope_set_ids=("id1",),
    )
    return base if not overrides else base.evolve(**overrides)


def test_resolve_superseded_fast_completion_allows_display_then_handoff_for_current_owner() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        parallel=False,
        queue_ids=("id1", "id2"),
        queue_names=("set1", "set2"),
        total=2,
        completed_set_ids=("id1",),
    )
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=None,
        worker_request_id=None,
        discarded_slider_preview_generation_id=None,
    )
    pending_replay = PendingReplayState(active=False, request_id=9, target_set_ids=())

    decision = policy.resolve_superseded_fast_completion(
        activity=activity,
        context=context,
        request_id=4,
        pending_replay=pending_replay,
        shutdown_requested=False,
    )

    assert decision.display_current_preview is True
    assert decision.schedule_pending_preview_run is True
    assert decision.defer_context_deactivation_until_after_display is True
    assert decision.state_patch.pending_replay == PendingReplayDirective.arm_existing(target_set_ids=())


def test_build_explicit_cache_reconciliation_preserves_narrowed_subset_and_preview_scope() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        parallel=True,
        explicit_cache_preview_token="preview-token",
        explicit_cache_preview_scope_set_ids=("id1",),
        explicit_cache_valid_set_ids=("id1",),
        explicit_cache_invalidated_set_ids=("id2",),
        queue_ids=("id1", "id2"),
        queue_names=("set1", "set2"),
        total=2,
    )
    cache_state = CacheAuthorityState(
        active_cache_key="fresh-cache",
        active_cache_preview_token="preview-token",
        active_cache_preview_scope_set_ids=("id1",),
        active_cache_valid_set_ids=("id1",),
        active_cache_invalidated_set_ids=("id2",),
    )

    decision = policy.build_explicit_cache_reconciliation(
        context=context,
        cache_state=cache_state,
        cache_key="fresh-cache",
    )

    assert decision.clear_active_selection_state is False
    assert decision.active_cache_valid_set_ids == ("id1",)
    assert decision.active_cache_invalidated_set_ids == ("id2",)
    assert decision.active_cache_preview_scope_set_ids == ("id1",)
    assert decision.redraw_valid_set_ids == ("id1",)
    assert decision.has_redraw_subset is True


def test_resolve_explicit_dirty_reset_uses_generation_match_only() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        pending_workspace_reset_set_ids=("id1", "id2"),
        pending_dirty_reset_generation_by_set_id={"id1": 1, "id2": 3},
    )
    dirty_state_by_set_id = {
        "id1": DirtySetState(is_dirty=True, generation=1),
        "id2": DirtySetState(is_dirty=True, generation=4),
    }

    decision = policy.resolve_explicit_dirty_reset(
        context=context,
        dirty_state_by_set_id=dirty_state_by_set_id,
    )

    assert decision.eligible_reset_set_ids == ("id1",)
    assert decision.state_patch.context.pending_workspace_reset_set_ids == ()
    assert decision.state_patch.context.pending_dirty_reset_generation_by_set_id == {}


def test_resolve_pending_replay_after_canonical_reset_preserves_non_targeted_ids() -> None:
    policy = SimulationCompletionPolicy()
    pending_replay = PendingReplayState(active=True, request_id=7, target_set_ids=("id1", "id2"))

    directive = policy.resolve_pending_replay_after_canonical_reset(
        pending_replay=pending_replay,
        reset_set_ids=("id1",),
    )

    assert directive == PendingReplayDirective.preserve(target_set_ids=("id2",), clear_plot_updates=True)


def test_resolve_pending_init_failure_clears_applied_flag_when_invalidation_is_required() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        pending_init_applied=True,
        pending_init_rewrite="reaction: A -> B; k=1",
    )

    decision = policy.resolve_pending_init_failure(context)

    assert decision.should_invalidate_preserved_results is True
    assert decision.state_patch.context.pending_init_applied is False


def test_resolve_pending_init_completion_extracts_seed_for_primary_set() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        primary_set_id="id1",
        pending_init_seed={"set1": {"A": 1.0, "B": 0.5}},
        pending_init_rewrite="reaction: A -> B; k=1",
        pending_init_applied=False,
    )

    decision = policy.resolve_pending_init_completion(
        context=context,
        batch_set="set1",
        is_preview=False,
        is_primary=True,
    )

    assert decision.should_attempt_apply is True
    assert decision.seed_for_ui == {"A": 1.0, "B": 0.5}
    assert decision.rewrite == "reaction: A -> B; k=1"


def test_pending_initial_seed_for_set_ignores_non_mapping_and_non_target_shapes() -> None:
    assert pending_initial_seed_for_set(None, set_name="set1") == {}
    assert pending_initial_seed_for_set({"A": 1.0}, set_name="set1") == {}
    assert pending_initial_seed_for_set({"set2": {"A": 1.0}}, set_name="set1") == {}


def test_pending_initial_seed_for_set_reads_nested_target_seed_only() -> None:
    seed = {
        "set1": {"A": 1.0},
        "set2": {"B": 2.0},
    }

    assert pending_initial_seed_for_set(seed, set_name="set2") == {"B": 2.0}


def test_has_active_explicit_simulation_falls_back_to_worker_and_run_state() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=5,
        simulation_running=True,
        slider_simulation_active=False,
        worker_running=True,
        worker_fast_mode=False,
        worker_request_id=5,
    )

    assert policy.has_active_explicit_simulation(activity=activity, context=None) is True
    assert policy.has_active_fast_preview_in_flight(activity=activity, context=None) is False


def test_completion_policy_context_normalizes_infinite_and_string_inputs() -> None:
    context = CompletionPolicyContext(
        active="false",
        request_id=float("inf"),
        run_id=float("-inf"),
        fast_mode="false",
        parallel="false",
        keep_executor_alive="false",
        queue_ids="id1",
        queue_names="set1",
        total="bad-total",
        pos="bad-pos",
        primary_set_id=None,
        completed_set_ids="id1",
        pending_workspace_reset_set_ids="id1",
        pending_dirty_reset_generation_by_set_id={"id1": "bad-generation"},
        pending_init_seed={},
        pending_init_rewrite=None,
        pending_init_applied=False,
        explicit_cache_preview_token=None,
        explicit_cache_preview_scope_set_ids=None,
        explicit_cache_valid_set_ids=None,
        explicit_cache_invalidated_set_ids=None,
        preview_scope_set_ids=None,
    )

    assert context.active is False
    assert context.fast_mode is False
    assert context.parallel is False
    assert context.keep_executor_alive is False
    assert context.request_id is None
    assert context.run_id is None
    assert context.queue_ids == ("id1",)
    assert context.queue_names == ("set1",)
    assert context.completed_set_ids == ("id1",)
    assert context.pending_workspace_reset_set_ids == ("id1",)
    assert context.pending_dirty_reset_generation_by_set_id == {}
    assert context.pending_init_applied is False


def test_completion_policy_context_preserves_duplicate_queue_names() -> None:
    context = _context(
        queue_ids=("id1", "id2"),
        queue_names=("set1", "set1"),
    )

    assert context.queue_names == ("set1", "set1")


def test_preview_request_can_display_rejects_discarded_stale_preview() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=None,
        worker_request_id=None,
        discarded_slider_preview_generation_id=9,
    )

    assert policy.preview_request_can_display(activity=activity, context=_context(), request_id=4) is False


def test_stale_fast_request_ownership_fails_closed_for_missing_context_or_worker_request_id() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(request_id=float("inf"))
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=False,
        worker_running=True,
        worker_fast_mode=True,
        worker_request_id=float("inf"),
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=RunActivitySnapshot(
            latest_request_id=9,
            simulation_running=True,
            slider_simulation_active=True,
            worker_running=False,
            worker_fast_mode=None,
            worker_request_id=None,
        ),
        context=context,
        request_id=4,
    ) is False
    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=4,
    ) is False


def test_stale_fast_request_ownership_uses_stopped_fast_worker_request_id_for_mismatch() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=True,
        worker_request_id=9,
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=4,
    ) is False


def test_stale_fast_request_ownership_uses_stopped_fast_worker_request_id_for_match() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=True,
        worker_request_id=9,
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=9,
    ) is True


def test_stale_fast_request_ownership_does_not_revive_cleared_preview_from_stopped_worker() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=False,
        worker_running=False,
        worker_fast_mode=True,
        worker_request_id=9,
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=9,
    ) is False


def test_stale_fast_request_ownership_rejects_stopped_worker_when_newer_preview_intent_exists() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=10,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=True,
        worker_request_id=9,
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=9,
    ) is False


def test_stale_fast_request_ownership_rejects_stopped_nonfast_worker_even_when_slider_active() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=10,
        simulation_running=True,
        slider_simulation_active=True,
        worker_running=False,
        worker_fast_mode=False,
        worker_request_id=9,
    )

    assert policy.stale_fast_request_still_owns_current_state(
        activity=activity,
        context=None,
        request_id=4,
    ) is False


def test_build_run_start_cache_decision_separates_explicit_and_preview_scope() -> None:
    policy = SimulationCompletionPolicy()

    preview_decision = policy.build_run_start_cache_decision(
        fast_mode=True,
        queue_ids=("id1", "id2"),
    )
    explicit_decision = policy.build_run_start_cache_decision(
        fast_mode=False,
        queue_ids=("id1", "id2"),
    )

    assert preview_decision.preview_scope_set_ids == ("id1", "id2")
    assert preview_decision.explicit_cache_valid_set_ids is None
    assert explicit_decision.preview_scope_set_ids is None
    assert explicit_decision.explicit_cache_valid_set_ids == ("id1", "id2")


def test_capture_dirty_reset_tracking_keeps_only_dirty_sets_with_generations() -> None:
    policy = SimulationCompletionPolicy()

    decision = policy.capture_dirty_reset_tracking(
        fast_mode=False,
        queue_ids=("id1", "id2", "id3"),
        dirty_state_by_set_id={
            "id1": DirtySetState(is_dirty=True, generation=2),
            "id2": DirtySetState(is_dirty=True, generation=None),
            "id3": DirtySetState(is_dirty=False, generation=7),
        },
    )

    assert decision.pending_workspace_reset_set_ids == ("id1",)
    assert decision.pending_dirty_reset_generation_by_set_id == {"id1": 2}


def test_resolve_superseded_fast_error_clears_pending_replay_when_nothing_should_replay() -> None:
    policy = SimulationCompletionPolicy()
    activity = RunActivitySnapshot(
        latest_request_id=9,
        simulation_running=True,
        slider_simulation_active=False,
        worker_running=False,
        worker_fast_mode=None,
        worker_request_id=None,
    )
    context = _context(active=True, request_id=4, fast_mode=True)
    pending_replay = PendingReplayState(active=False, request_id=None, target_set_ids=())

    decision = policy.resolve_superseded_fast_error(
        activity=activity,
        context=context,
        request_id=4,
        pending_replay=pending_replay,
    )

    assert decision.schedule_pending_preview_run is False
    assert decision.state_patch.pending_replay == PendingReplayDirective.clear(clear_plot_updates=False)


def test_build_context_update_from_cache_truth_clears_explicit_subset_on_cache_key_mismatch() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        explicit_cache_preview_token="preview-token",
        explicit_cache_preview_scope_set_ids=("id1",),
        explicit_cache_valid_set_ids=("id1",),
        explicit_cache_invalidated_set_ids=("id2",),
    )
    cache_state = CacheAuthorityState(
        active_cache_key="newer-cache",
        active_cache_preview_token="preview-token",
        active_cache_preview_scope_set_ids=("id1",),
        active_cache_valid_set_ids=("id1",),
        active_cache_invalidated_set_ids=("id2",),
    )

    updated = policy.build_context_update_from_cache_truth(
        context=context,
        cache_state=cache_state,
        cache_key="older-cache",
    )

    assert updated.explicit_cache_preview_token is None
    assert updated.explicit_cache_preview_scope_set_ids == ()
    assert updated.explicit_cache_valid_set_ids == ()
    assert updated.explicit_cache_invalidated_set_ids == ()


def test_resolve_preflight_abort_pending_replay_returns_none_for_preview_runs() -> None:
    policy = SimulationCompletionPolicy()
    pending_replay = PendingReplayState(active=True, request_id=7, target_set_ids=("id1",))

    directive = policy.resolve_preflight_abort_pending_replay(
        pending_replay=pending_replay,
        explicit_run=False,
    )

    assert directive is None


def test_resolve_explicit_error_pending_replay_queues_only_for_explicit_runs() -> None:
    policy = SimulationCompletionPolicy()
    pending_replay = PendingReplayState(active=True, request_id=7, target_set_ids=("id1",))

    assert policy.resolve_explicit_error_pending_replay(
        fast_mode=False,
        pending_replay=pending_replay,
    ) == PendingReplayDirective.queue_fresh(target_set_ids=("id1",))
    assert policy.resolve_explicit_error_pending_replay(
        fast_mode=True,
        pending_replay=pending_replay,
    ) == PendingReplayDirective.clear(clear_plot_updates=False)


def test_pending_init_apply_result_and_guard_follow_applied_state() -> None:
    policy = SimulationCompletionPolicy()
    context = _context(
        fast_mode=False,
        pending_init_applied=False,
        pending_init_rewrite="reaction: A -> B; k=1",
    )

    updated = policy.note_pending_init_apply_result(context=context, applied=True)

    assert updated.pending_init_applied is True
    assert policy.should_arm_pending_init_guard(
        context=updated,
        is_preview=False,
        is_primary=True,
    ) == "reaction: A -> B; k=1"


def test_resolve_pending_replay_after_canonical_reset_clears_when_all_targets_are_reset() -> None:
    policy = SimulationCompletionPolicy()
    pending_replay = PendingReplayState(active=True, request_id=7, target_set_ids=("id1",))

    directive = policy.resolve_pending_replay_after_canonical_reset(
        pending_replay=pending_replay,
        reset_set_ids=("id1",),
    )

    assert directive == PendingReplayDirective.clear(clear_plot_updates=True)

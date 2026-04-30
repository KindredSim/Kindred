from kindred.core.mechanism_runtime_transition import (
    AuthoritativeMechanismSnapshot,
    MechanismRuntimeTransitionService,
)


def _snapshot(reactions: str, state_network: str = "") -> AuthoritativeMechanismSnapshot:
    return AuthoritativeMechanismSnapshot.from_texts(
        reactions_text=reactions,
        state_network_text=state_network,
    )


def test_pending_init_rewrite_suppresses_matching_transition_until_next_real_edit():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0")
    )
    service.arm_pending_init_result_guard(
        rewrite="reaction: A -> B;   k=1.0",
        state_network_text="",
    )

    pending_init = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="pending_init_migration",
    )

    assert pending_init.runtime_invalidation_required is False
    assert pending_init.active_work_supersede_required is False
    assert pending_init.pending_init_preservation is True
    assert pending_init.epoch == 0

    next_edit = service.apply_authoritative_transition(
        _snapshot("reaction: A -> C; k=2.0"),
        source="manual_commit",
    )

    assert next_edit.runtime_invalidation_required is True
    assert next_edit.active_work_supersede_required is True
    assert next_edit.pending_init_preservation is False
    assert next_edit.epoch == 1


def test_deferred_readiness_schedule_is_owned_by_transition_epoch():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0")
    )

    outcome = service.apply_authoritative_transition(
        _snapshot("reaction: A -> C; k=2.0"),
        source="project_apply",
        schedule_runtime_refresh=False,
    )

    assert outcome.runtime_invalidation_required is True
    assert outcome.active_work_supersede_required is True
    assert outcome.readiness_schedule_required is False
    assert outcome.readiness_schedule_deferred is True
    assert service.consume_pending_readiness_epoch() == outcome.epoch
    assert service.consume_pending_readiness_epoch() is None


def test_force_transition_invalidates_even_when_snapshot_matches():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0")
    )

    outcome = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="programmatic_load",
        force_runtime_invalidation=True,
    )

    assert outcome.runtime_invalidation_required is True
    assert outcome.active_work_supersede_required is True
    assert outcome.epoch == 1


def test_suppressed_signal_does_not_advance_authoritative_identity_before_caller_transition():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0")
    )

    suppressed = service.apply_authoritative_transition(
        _snapshot("reaction: A -> C; k=2.0"),
        source="suppressed_widget_signal",
        input_suppressed=True,
    )
    caller_owned = service.apply_authoritative_transition(
        _snapshot("reaction: A -> C; k=2.0"),
        source="authoritative_editor_rewrite",
    )

    assert suppressed.runtime_invalidation_required is False
    assert suppressed.epoch == 0
    assert caller_owned.runtime_invalidation_required is True
    assert caller_owned.active_work_supersede_required is True
    assert caller_owned.epoch == 1


def test_canonical_batch_initial_change_advances_runtime_input_epoch_without_text_change():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0")
    )

    unchanged = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="baseline",
        canonical_batch_initials_by_set_id={"set-a": "initials:A=1"},
    )
    changed = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="species_commit",
        canonical_batch_initials_by_set_id={"set-a": "initials:A=2"},
        affected_set_ids=("set-a",),
    )

    assert unchanged.runtime_invalidation_required is False
    assert unchanged.active_work_supersede_required is False
    assert changed.runtime_invalidation_required is False
    assert changed.active_work_supersede_required is True
    assert changed.display_cache_invalidation_allowed is True
    assert changed.readiness_schedule_required is False
    assert changed.affected_set_ids == ("set-a",)
    assert changed.epoch == 1


def test_forced_transition_without_initials_baseline_does_not_retain_stale_canonical_identity():
    service = MechanismRuntimeTransitionService(
        initial_snapshot=_snapshot("reaction: A -> B; k=1.0"),
        initial_canonical_batch_initials_by_set_id={"set-a": "initials:A=1"},
    )

    first_commit = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="species_commit",
        canonical_batch_initials_by_set_id={"set-a": "initials:A=2"},
        affected_set_ids=("set-a",),
    )
    programmatic_load = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="programmatic_load",
        force_runtime_invalidation=True,
    )
    reused_id_commit = service.apply_authoritative_transition(
        _snapshot("reaction: A -> B; k=1.0"),
        source="species_commit_after_load",
        canonical_batch_initials_by_set_id={"set-a": "initials:A=2"},
        affected_set_ids=("set-a",),
    )

    assert first_commit.runtime_input_invalidation_required is True
    assert programmatic_load.runtime_invalidation_required is True
    assert reused_id_commit.runtime_input_invalidation_required is True
    assert reused_id_commit.active_work_supersede_required is True
    assert reused_id_commit.affected_set_ids == ("set-a",)

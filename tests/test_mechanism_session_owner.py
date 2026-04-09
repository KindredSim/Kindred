import pytest

from kindred.gui.mechanism_session_owner import MechanismSessionOwner, ValidationResult


pytestmark = pytest.mark.unit


VALID_REACTIONS = "A -> B ; k=1"
UPDATED_REACTIONS = "A -> B ; k=2"
INVALID_REACTIONS = "invalid garbage"
REACTIONS_WITH_NAMED_INITIAL_SET = "reaction: A -> B; k=1.0\n\nSet B = {\n[A] = 1.0\n}"
VALID_STATE_NETWORK = "\n".join(
    [
        "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
        "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
        "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
        "edge: A,TS1",
        "edge: TS1,B",
    ]
)


def test_construction_defaults_to_empty_locked_invalid_owner() -> None:
    owner = MechanismSessionOwner()

    assert owner.canonical_reactions_text == ""
    assert owner.canonical_state_network_dsl == ""
    assert owner.canonical_full_dsl == ""
    assert owner.draft_reactions_text == ""
    assert owner.draft_state_network_dsl == ""
    assert owner.draft_full_dsl == ""
    assert owner.edit_session_active is False
    assert owner.is_ready_for_explicit_run() is False
    assert owner.is_ready_for_preview() is False


def test_authoritative_update_sets_canonical_and_enables_explicit_run() -> None:
    owner = MechanismSessionOwner()

    owner.apply_authoritative_update(VALID_REACTIONS, "")

    assert owner.canonical_reactions_text == VALID_REACTIONS
    assert owner.canonical_state_network_dsl == ""
    assert owner.canonical_full_dsl == VALID_REACTIONS
    assert owner.draft_reactions_text == ""
    assert owner.edit_session_active is False
    assert owner.is_ready_for_explicit_run() is True


def test_authoritative_update_with_invalid_dsl_keeps_invalid_canonical() -> None:
    owner = MechanismSessionOwner()

    owner.apply_authoritative_update(INVALID_REACTIONS, "")

    assert owner.canonical_reactions_text == INVALID_REACTIONS
    assert owner.is_ready_for_explicit_run() is False


def test_edit_session_lifecycle_promotes_valid_draft_to_canonical() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")

    owner.begin_edit_session()

    assert owner.edit_session_active is True
    assert owner.draft_reactions_text == VALID_REACTIONS
    assert owner.draft_state_network_dsl == ""

    owner.update_draft_reactions(UPDATED_REACTIONS)

    assert owner.draft_reactions_text == UPDATED_REACTIONS
    assert owner.canonical_reactions_text == VALID_REACTIONS

    committed = owner.commit_edit_session()

    assert committed is True
    assert owner.edit_session_active is False
    assert owner.canonical_reactions_text == UPDATED_REACTIONS
    assert owner.draft_reactions_text == ""
    assert owner.draft_state_network_dsl == ""


def test_failed_commit_preserves_canonical_and_leaves_session_active() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(INVALID_REACTIONS)

    committed = owner.commit_edit_session()

    assert committed is False
    assert owner.edit_session_active is True
    assert owner.canonical_reactions_text == VALID_REACTIONS
    assert owner.draft_reactions_text == INVALID_REACTIONS


def test_cancel_edit_session_discards_draft_and_restores_locked_state() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(UPDATED_REACTIONS)

    owner.cancel_edit_session()

    assert owner.edit_session_active is False
    assert owner.canonical_reactions_text == VALID_REACTIONS
    assert owner.draft_reactions_text == ""
    assert owner.draft_state_network_dsl == ""


def test_authoritative_update_during_edit_session_replaces_canonical_and_clears_draft() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(UPDATED_REACTIONS)

    owner.apply_authoritative_update("A -> B ; k=3", "")

    assert owner.edit_session_active is False
    assert owner.canonical_reactions_text == "A -> B ; k=3"
    assert owner.draft_reactions_text == ""
    assert owner.draft_state_network_dsl == ""


def test_update_draft_reactions_raises_while_session_inactive() -> None:
    owner = MechanismSessionOwner()

    with pytest.raises(RuntimeError):
        owner.update_draft_reactions("x")


def test_update_draft_state_network_raises_while_session_inactive() -> None:
    owner = MechanismSessionOwner()

    with pytest.raises(RuntimeError):
        owner.update_draft_state_network(VALID_STATE_NETWORK)


def test_validation_results_return_structured_counts_for_canonical_and_draft() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")

    canonical_result = owner.validate_canonical()

    assert canonical_result == ValidationResult(
        valid=True,
        error_message="",
        species_count=2,
        reaction_count=1,
        equilibria_count=0,
    )

    owner.begin_edit_session()
    owner.update_draft_reactions("")
    owner.update_draft_state_network(VALID_STATE_NETWORK)

    draft_result = owner.validate_draft()

    assert draft_result == ValidationResult(
        valid=True,
        error_message="",
        species_count=2,
        reaction_count=0,
        equilibria_count=1,
    )


def test_empty_reactions_and_valid_state_network_are_ready_for_explicit_run() -> None:
    owner = MechanismSessionOwner()

    owner.apply_authoritative_update("", VALID_STATE_NETWORK)

    assert owner.is_ready_for_explicit_run() is True
    assert owner.canonical_full_dsl == "\n\n# State Network\n" + VALID_STATE_NETWORK


def test_preview_readiness_uses_draft_while_session_active() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()

    owner.update_draft_reactions(UPDATED_REACTIONS)
    assert owner.is_ready_for_preview() is True

    owner.update_draft_reactions(INVALID_REACTIONS)
    assert owner.is_ready_for_preview() is False


def test_preview_readiness_matches_explicit_run_when_session_inactive() -> None:
    owner = MechanismSessionOwner()

    assert owner.is_ready_for_preview() is False

    owner.apply_authoritative_update(VALID_REACTIONS, "")

    assert owner.is_ready_for_preview() is True
    assert owner.is_ready_for_preview() is owner.is_ready_for_explicit_run()


def test_explicit_run_source_raises_when_canonical_not_ready() -> None:
    owner = MechanismSessionOwner()

    with pytest.raises(RuntimeError):
        owner.explicit_run_source()


def test_explicit_run_source_uses_canonical_source_during_active_edit_session() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(INVALID_REACTIONS)

    assert owner.is_ready_for_explicit_run() is True
    assert owner.explicit_run_source() == VALID_REACTIONS


def test_preview_source_uses_draft_when_session_active() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(UPDATED_REACTIONS)
    owner.update_draft_state_network(VALID_STATE_NETWORK)

    assert owner.draft_full_dsl == UPDATED_REACTIONS + "\n\n# State Network\n" + VALID_STATE_NETWORK
    assert owner.preview_source() == UPDATED_REACTIONS + "\n\n# State Network\n" + VALID_STATE_NETWORK


def test_preview_source_raises_when_active_draft_is_not_ready() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(INVALID_REACTIONS)

    with pytest.raises(RuntimeError):
        owner.preview_source()


def test_preview_source_uses_canonical_source_when_session_is_inactive() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, VALID_STATE_NETWORK)

    assert owner.preview_source() == VALID_REACTIONS + "\n\n# State Network\n" + VALID_STATE_NETWORK


def test_named_initial_set_normalization_keeps_validation_ready() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(REACTIONS_WITH_NAMED_INITIAL_SET, "")

    result = owner.validate_canonical()

    assert result.valid is True
    assert result.species_count == 2
    assert result.reaction_count == 1
    assert owner.is_ready_for_explicit_run() is True
    assert owner.explicit_run_source() == REACTIONS_WITH_NAMED_INITIAL_SET


def test_preview_source_preserves_named_initial_set_blocks() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(REACTIONS_WITH_NAMED_INITIAL_SET)

    assert owner.preview_source() == REACTIONS_WITH_NAMED_INITIAL_SET


def test_sources_return_owned_text_with_state_network_dsl() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(REACTIONS_WITH_NAMED_INITIAL_SET, VALID_STATE_NETWORK)

    expected_source = (
        REACTIONS_WITH_NAMED_INITIAL_SET
        + "\n\n# State Network\n"
        + VALID_STATE_NETWORK
    )

    assert owner.explicit_run_source() == expected_source

    owner.begin_edit_session()

    assert owner.preview_source() == expected_source


def test_commit_requires_topology_validator_for_state_network_canonicalization() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: False)
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions("")
    owner.update_draft_state_network(VALID_STATE_NETWORK)

    committed = owner.commit_edit_session()

    assert committed is False
    assert owner.edit_session_active is True
    assert owner.canonical_reactions_text == VALID_REACTIONS


def test_explicit_run_readiness_requires_topology_validator_for_state_network() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: False)
    owner.apply_authoritative_update("", VALID_STATE_NETWORK)

    assert owner.validate_canonical().valid is True
    assert owner.is_ready_for_explicit_run() is False


def test_preview_readiness_ignores_topology_validator_while_editing_draft() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: False)
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions("")
    owner.update_draft_state_network(VALID_STATE_NETWORK)

    assert owner.validate_draft().valid is True
    assert owner.is_ready_for_preview() is True


def test_preview_readiness_uses_topology_validator_when_session_is_inactive() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: False)
    owner.apply_authoritative_update("", VALID_STATE_NETWORK)

    assert owner.is_ready_for_preview() is False


def test_explicit_and_preview_sources_respect_topology_validator_when_session_is_inactive() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: False)
    owner.apply_authoritative_update("", VALID_STATE_NETWORK)

    with pytest.raises(RuntimeError):
        owner.explicit_run_source()

    with pytest.raises(RuntimeError):
        owner.preview_source()


def test_begin_commit_and_cancel_raise_when_session_state_is_invalid() -> None:
    owner = MechanismSessionOwner()

    with pytest.raises(RuntimeError):
        owner.commit_edit_session()

    with pytest.raises(RuntimeError):
        owner.cancel_edit_session()

    owner.begin_edit_session()

    with pytest.raises(RuntimeError):
        owner.begin_edit_session()


def test_validate_draft_returns_empty_invalid_result_when_no_session_is_active() -> None:
    owner = MechanismSessionOwner()

    result = owner.validate_draft()

    assert result.valid is False
    assert result.error_message
    assert result.species_count == 0
    assert result.reaction_count == 0
    assert result.equilibria_count == 0


def test_text_mutators_reject_non_string_inputs() -> None:
    owner = MechanismSessionOwner()

    with pytest.raises(TypeError):
        owner.apply_authoritative_update(None, "")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        owner.apply_authoritative_update(VALID_REACTIONS, None)  # type: ignore[arg-type]

    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()

    with pytest.raises(TypeError):
        owner.update_draft_reactions(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        owner.update_draft_state_network(None)  # type: ignore[arg-type]


def test_authoritative_update_is_atomic_when_argument_validation_fails() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions(UPDATED_REACTIONS)

    with pytest.raises(TypeError):
        owner.apply_authoritative_update("A -> C ; k=3", None)  # type: ignore[arg-type]

    assert owner.canonical_reactions_text == VALID_REACTIONS
    assert owner.canonical_state_network_dsl == ""
    assert owner.edit_session_active is True
    assert owner.draft_reactions_text == UPDATED_REACTIONS


def test_topology_validator_true_allows_commit_and_explicit_readiness() -> None:
    owner = MechanismSessionOwner(topology_validator=lambda: True)
    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()
    owner.update_draft_reactions("")
    owner.update_draft_state_network(VALID_STATE_NETWORK)

    assert owner.commit_edit_session() is True
    assert owner.is_ready_for_explicit_run() is True


def test_validation_cache_invalidates_after_repeated_draft_and_canonical_updates() -> None:
    owner = MechanismSessionOwner()
    owner.apply_authoritative_update(VALID_REACTIONS, "")

    assert owner.validate_canonical().valid is True

    owner.apply_authoritative_update(INVALID_REACTIONS, "")

    assert owner.validate_canonical().valid is False

    owner.apply_authoritative_update(VALID_REACTIONS, "")
    owner.begin_edit_session()

    assert owner.validate_draft().valid is True

    owner.update_draft_reactions(INVALID_REACTIONS)

    assert owner.validate_draft().valid is False

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.unit]


def test_display_transition_outcome_must_be_inspected_explicitly() -> None:
    from kindred.gui.ports import (
        DisplayStatus,
        DisplayTransitionCause,
        DisplayTransitionOutcome,
        DisplayTransitionOutcomeKind,
    )

    outcome = DisplayTransitionOutcome(
        kind=DisplayTransitionOutcomeKind.DENIED,
        active_transaction=None,
        previous_transaction=None,
        display_status=DisplayStatus.DISPLAY_DENIED,
        cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
    )

    with pytest.raises(TypeError):
        bool(outcome)


@pytest.mark.parametrize(
    ("display_status", "base_text"),
    [
        ("DISPLAYED_COMPLETED_RUN", "Displayed completed run"),
        ("DISPLAYED_FRESH_PREVIEW", "Displayed fresh preview"),
        ("DISPLAYED_DIRECT_RESULT", "Displayed simulation result"),
        ("DISPLAYED_WORKSPACE_PREVIEW", "Displayed workspace preview"),
        ("DISPLAYED_RESOLVED_RESULT", "Displayed resolved result"),
        ("DISPLAYED_CACHED_RESULT", "Displayed cached result"),
    ],
)
def test_published_display_statuses_summarize_partial_request_outcome(
    display_status: str,
    base_text: str,
) -> None:
    from kindred.gui.controllers.results_display_projections import display_transition_status_text
    from kindred.gui.ports import (
        DisplayStatus,
        DisplayTransitionOutcome,
        DisplayTransitionOutcomeKind,
    )

    outcome = DisplayTransitionOutcome(
        kind=DisplayTransitionOutcomeKind.PUBLISHED,
        active_transaction=None,
        previous_transaction=None,
        display_status=getattr(DisplayStatus, display_status),
        requested_show_set_ids=("displayed", "missing"),
        display_set_ids=("displayed",),
        attempted_display_set_ids=("displayed",),
        affected_set_ids=("displayed",),
        unresolved_intent_set_ids=("missing",),
        missing_intent_set_ids=("missing",),
    )

    assert display_transition_status_text(outcome) == (
        f"{base_text}; 1 result needs a run."
    )


def test_display_transition_status_text_distinguishes_request_outcome_categories() -> None:
    from kindred.gui.controllers.results_display_projections import display_transition_status_text
    from kindred.gui.ports import (
        DisplayStatus,
        DisplayTransitionOutcome,
        DisplayTransitionOutcomeKind,
    )

    outcome = DisplayTransitionOutcome(
        kind=DisplayTransitionOutcomeKind.PUBLISHED,
        active_transaction=None,
        previous_transaction=None,
        display_status=DisplayStatus.DISPLAYED_CACHED_RESULT,
        requested_show_set_ids=("displayed", "failed", "missing", "semantic", "unresolved"),
        display_set_ids=("displayed",),
        failed_intent_set_ids=("failed",),
        missing_intent_set_ids=("missing",),
        semantic_unavailable_set_ids=("semantic",),
        unresolved_intent_set_ids=("missing", "semantic", "unresolved"),
        requested_labels_by_set_id={
            "failed": "Failed Set",
            "missing": "Missing Set",
            "semantic": "Semantic Set",
            "unresolved": "Unresolved Set",
        },
    )

    assert display_transition_status_text(outcome) == (
        "Displayed cached result; Failed Set failed, Missing Set needs a run, "
        "Semantic Set has no displayable result, and Unresolved Set is unavailable."
    )


def test_active_display_transaction_keeps_request_outcome_on_transition() -> None:
    from kindred.gui.ports import (
        ActiveDisplayKind,
        ActiveDisplayTransaction,
        DisplayStatus,
        DisplayTransitionOutcome,
        DisplayTransitionOutcomeKind,
    )

    active = ActiveDisplayTransaction(
        transaction_id="display-only",
        kind=ActiveDisplayKind.CACHED_RESULT,
        display_set_ids=("displayed",),
        primary_display_set_id="displayed",
        sets={},
        status=DisplayStatus.DISPLAYED_CACHED_RESULT,
    )
    outcome = DisplayTransitionOutcome(
        kind=DisplayTransitionOutcomeKind.PUBLISHED,
        active_transaction=active,
        previous_transaction=None,
        display_status=DisplayStatus.DISPLAYED_CACHED_RESULT,
        requested_show_set_ids=("displayed", "missing"),
        display_set_ids=("displayed",),
        unresolved_intent_set_ids=("missing",),
        missing_intent_set_ids=("missing",),
    )

    assert active.display_set_ids == ("displayed",)
    assert outcome.requested_show_set_ids == ("displayed", "missing")
    assert outcome.missing_intent_set_ids == ("missing",)
    with pytest.raises(TypeError):
        ActiveDisplayTransaction(
            transaction_id="bad-request-outcome",
            kind=ActiveDisplayKind.CACHED_RESULT,
            display_set_ids=("displayed",),
            primary_display_set_id="displayed",
            sets={},
            status=DisplayStatus.DISPLAYED_CACHED_RESULT,
            requested_show_set_ids=("displayed", "missing"),
        )

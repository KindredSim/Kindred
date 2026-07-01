from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from kindred.gui.ports import ReactionsDraftApplyResult


@dataclass(frozen=True, slots=True)
class ReactionsDraftResolution:
    status: Literal["clean", "discarded", "applied", "cancelled", "failed"]
    error: str = ""
    apply_result: ReactionsDraftApplyResult | None = None

    @property
    def may_continue(self) -> bool:
        return self.status in {"clean", "discarded", "applied"}


class ReactionsDraftResolutionOwner:
    """Owns dirty Reactions draft resolution policy for canonical changes."""

    def __init__(
        self,
        *,
        draft_dirty: Callable[[], bool],
        prompt_draft_decision: Callable[..., str],
        discard_draft: Callable[[], None],
        apply_dirty_draft_with_transition: Callable[[], ReactionsDraftApplyResult],
        apply_dirty_draft_without_transition: Callable[[], ReactionsDraftApplyResult],
        draft_error_text: Callable[[], str],
    ) -> None:
        self._draft_dirty = draft_dirty
        self._prompt_draft_decision = prompt_draft_decision
        self._discard_draft = discard_draft
        self._apply_dirty_draft_with_transition = apply_dirty_draft_with_transition
        self._apply_dirty_draft_without_transition = apply_dirty_draft_without_transition
        self._draft_error_text = draft_error_text

    def resolve_for_authoritative_change(
        self,
        action_label: str,
        *,
        continue_label: str = "Continue",
        apply_label: str | None = None,
        discard_label: str | None = None,
        informative_text: str | None = None,
    ) -> ReactionsDraftResolution:
        return self._resolve(
            action_label,
            continue_label=continue_label,
            apply_label=apply_label,
            discard_label=discard_label,
            informative_text=informative_text,
            apply_dirty_draft=self._apply_dirty_draft_with_transition,
        )

    def resolve_for_authoritative_rewrite(
        self,
        action_label: str,
        *,
        continue_label: str = "Continue",
    ) -> ReactionsDraftResolution:
        return self._resolve(
            action_label,
            continue_label=continue_label,
            apply_dirty_draft=self._apply_dirty_draft_without_transition,
        )

    def resolve_for_authoritative_source_load(
        self,
        action_label: str,
        *,
        continue_label: str = "Load",
    ) -> ReactionsDraftResolution:
        return self._resolve(
            action_label,
            continue_label=continue_label,
            apply_dirty_draft=self._apply_dirty_draft_without_transition,
        )

    def resolve_for_stale_command(
        self,
        action_label: str,
        *,
        apply_label: str,
        discard_label: str,
        informative_text: str,
    ) -> ReactionsDraftResolution:
        return self._resolve(
            action_label,
            continue_label="Continue",
            apply_label=apply_label,
            discard_label=discard_label,
            informative_text=informative_text,
            apply_dirty_draft=self._apply_dirty_draft_with_transition,
        )

    def _resolve(
        self,
        action_label: str,
        *,
        continue_label: str,
        apply_dirty_draft: Callable[[], ReactionsDraftApplyResult],
        apply_label: str | None = None,
        discard_label: str | None = None,
        informative_text: str | None = None,
    ) -> ReactionsDraftResolution:
        if not self._draft_dirty():
            return ReactionsDraftResolution("clean")

        decision = str(
            self._prompt_draft_decision(
                str(action_label),
                continue_label=str(continue_label or "Continue"),
                apply_label=apply_label,
                discard_label=discard_label,
                informative_text=informative_text,
            )
            or "cancel"
        ).strip().lower()
        if decision == "cancel":
            return ReactionsDraftResolution("cancelled")
        if decision == "discard":
            self._discard_draft()
            return ReactionsDraftResolution("discarded")
        if decision == "apply":
            result = apply_dirty_draft()
            if bool(result):
                return ReactionsDraftResolution(
                    "applied",
                    apply_result=result,
                )
            return ReactionsDraftResolution(
                "failed",
                error=str(self._draft_error_text() or ""),
                apply_result=result,
            )
        return ReactionsDraftResolution("cancelled")

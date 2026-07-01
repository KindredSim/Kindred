from __future__ import annotations

from typing import Callable

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.gui.reactions_draft_resolution_owner import ReactionsDraftResolutionOwner
from kindred.gui.ports import (
    AuthoritativeRewriteOutcome,
    AuthoritativeRewriteStatus,
    ReactionsDraftApplyResult,
)


class AuthoritativeRewriteOwner:
    """Coordinates authoritative mechanism-source rewrites as one operation."""

    def __init__(
        self,
        *,
        draft_resolution_owner: ReactionsDraftResolutionOwner,
        canonical_source: Callable[[], MechanismAuthoringSource],
        write_source: Callable[..., bool],
        publish_transition: Callable[..., object],
    ) -> None:
        self._draft_resolution_owner = draft_resolution_owner
        self._canonical_source = canonical_source
        self._write_source = write_source
        self._publish_transition = publish_transition

    def rewrite_mechanism_source(
        self,
        *,
        action_label: str,
        description: str,
        build_source: Callable[[MechanismAuthoringSource], MechanismAuthoringSource | None],
        record_undo: bool = True,
        transition_source: str = "authoritative_editor_rewrite",
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> AuthoritativeRewriteOutcome:
        if not callable(build_source):
            raise TypeError("build_source must be callable.")

        transition_source_s = str(transition_source or "authoritative_editor_rewrite")
        initial_source = self._canonical_source()
        draft_applied = False
        draft_apply_result: ReactionsDraftApplyResult | None = None
        draft_resolution_completed = False
        try:
            resolution = self._draft_resolution_owner.resolve_for_authoritative_rewrite(
                str(action_label or description or "Changing the mechanism source"),
                continue_label="Continue",
            )
            draft_resolution_completed = True
            if not resolution.may_continue:
                if resolution.status == "failed":
                    return AuthoritativeRewriteOutcome(
                        AuthoritativeRewriteStatus.FAILED,
                        error=str(resolution.error or ""),
                    )
                return AuthoritativeRewriteOutcome(AuthoritativeRewriteStatus.CANCELLED)
            if resolution.status == "applied":
                draft_apply_result = resolution.apply_result
                draft_applied = True

            base_source = self._canonical_source()
            new_source = build_source(base_source)
            if new_source is None:
                return self._finish_without_rewrite(
                    draft_applied=draft_applied,
                    draft_apply_result=draft_apply_result,
                    transition_source=transition_source_s,
                    before_publish=before_publish,
                    discard_dirty_preview=discard_dirty_preview,
                )
            if not isinstance(new_source, MechanismAuthoringSource):
                raise TypeError("build_source must return a MechanismAuthoringSource.")

            rewrite_changed = bool(
                self._write_source(
                    new_source,
                    description=str(description or action_label or "Rewrite mechanism source"),
                    record_undo=bool(record_undo),
                )
            )
        except Exception:
            if not draft_resolution_completed:
                raise
            source_updated = bool(
                draft_applied or self._canonical_source_differs_from(initial_source)
            )
            if source_updated:
                self._publish_for_source_update(
                    source_updated=True,
                    transition_source=transition_source_s,
                    draft_apply_result=draft_apply_result,
                    discard_dirty_preview=discard_dirty_preview,
                )
            raise

        source_updated = bool(rewrite_changed or draft_applied)
        self._publish_for_source_update(
            source_updated=source_updated,
            transition_source=transition_source_s,
            draft_apply_result=draft_apply_result,
            before_publish=before_publish,
            discard_dirty_preview=discard_dirty_preview,
        )
        if rewrite_changed:
            status = AuthoritativeRewriteStatus.REWRITE_APPLIED
        elif draft_applied:
            status = AuthoritativeRewriteStatus.NO_OP_AFTER_CANONICAL_CHANGE
        else:
            status = AuthoritativeRewriteStatus.CLEAN_NO_OP
        return AuthoritativeRewriteOutcome(
            status=status,
            dirty_draft_applied=bool(draft_applied),
            source_updated=source_updated,
        )

    def rewrite_reactions_text(
        self,
        *,
        action_label: str,
        description: str,
        rewrite_text: Callable[[str], str | None],
        record_undo: bool = True,
        transition_source: str = "authoritative_reactions_rewrite",
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> AuthoritativeRewriteOutcome:
        if not callable(rewrite_text):
            raise TypeError("rewrite_text must be callable.")

        def _build_source(source: MechanismAuthoringSource) -> MechanismAuthoringSource | None:
            next_reactions = rewrite_text(str(source.reactions_text))
            if next_reactions is None:
                return None
            return source.with_reactions_text(str(next_reactions))

        return self.rewrite_mechanism_source(
            action_label=str(action_label or description or "Changing Reactions"),
            description=str(description or action_label or "Rewrite Reactions"),
            build_source=_build_source,
            record_undo=bool(record_undo),
            transition_source=str(transition_source or "authoritative_reactions_rewrite"),
            before_publish=before_publish,
            discard_dirty_preview=discard_dirty_preview,
        )

    def _finish_without_rewrite(
        self,
        *,
        draft_applied: bool,
        draft_apply_result: ReactionsDraftApplyResult | None,
        transition_source: str,
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> AuthoritativeRewriteOutcome:
        self._publish_for_source_update(
            source_updated=bool(draft_applied),
            transition_source=str(transition_source or "authoritative_editor_rewrite"),
            draft_apply_result=draft_apply_result,
            before_publish=before_publish,
            discard_dirty_preview=discard_dirty_preview,
        )
        return AuthoritativeRewriteOutcome(
            status=(
                AuthoritativeRewriteStatus.DIRTY_DRAFT_APPLIED
                if bool(draft_applied)
                else AuthoritativeRewriteStatus.CLEAN_NO_OP
            ),
            dirty_draft_applied=bool(draft_applied),
            source_updated=bool(draft_applied),
        )

    def _publish_for_source_update(
        self,
        *,
        source_updated: bool,
        transition_source: str,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> None:
        if not bool(source_updated):
            return
        if callable(before_publish):
            try:
                before_publish()
            finally:
                self._publish_transition(
                    str(transition_source or "authoritative_editor_rewrite"),
                    draft_apply_result=draft_apply_result,
                    discard_dirty_preview=bool(discard_dirty_preview),
                )
            return
        self._publish_transition(
            str(transition_source or "authoritative_editor_rewrite"),
            draft_apply_result=draft_apply_result,
            discard_dirty_preview=bool(discard_dirty_preview),
        )

    def _canonical_source_differs_from(self, initial_source: MechanismAuthoringSource) -> bool:
        try:
            return self._canonical_source() != initial_source
        except Exception:
            return False

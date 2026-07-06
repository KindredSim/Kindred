from __future__ import annotations

from typing import Callable

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.gui.ports import (
    AuthoritativeRewriteOutcome,
    AuthoritativeRewriteStatus,
    ReactionsDraftApplyResult,
)
from kindred.gui.reactions_draft_resolution_owner import ReactionsDraftResolutionOwner
from kindred.gui.undo_commands import SetMechanismSourceCommand


class MechanismSourceTransactionOwner:
    """Owns GUI mechanism-source transactions and editor synchronization."""

    def __init__(
        self,
        *,
        mechanism_session_owner_getter: Callable[[], object | None],
        mechanism_editor_getter: Callable[[], object | None],
        undo_stack_getter: Callable[[], object],
        draft_resolution_owner: ReactionsDraftResolutionOwner,
        refresh_reactions_draft_state: Callable[[], None],
        get_authoritative_suppress: Callable[[], bool],
        set_authoritative_suppress: Callable[[bool], None],
        get_programmatic_text_suppress: Callable[[], bool],
        set_programmatic_text_suppress: Callable[[bool], None],
        publish_transition: Callable[..., object],
        apply_transition: Callable[..., object],
        publish_aborted_source_load_attempt: Callable[..., object | None],
        execute_source_load_transaction: Callable[..., object],
        guard_slider_transaction_invalidation: Callable[..., bool],
        prepare_source_load_reconciliation: Callable[..., object],
        apply_source_load_reconciliation_after_acceptance: Callable[[object], object | None],
        refresh_after_source_load: Callable[[], None],
        source_load_plan_materializes_initial_conditions: Callable[[object], bool],
        set_status_text: Callable[[str], None],
    ) -> None:
        self._mechanism_session_owner_getter = mechanism_session_owner_getter
        self._mechanism_editor_getter = mechanism_editor_getter
        self._undo_stack_getter = undo_stack_getter
        self._draft_resolution_owner = draft_resolution_owner
        self._refresh_reactions_draft_state = refresh_reactions_draft_state
        self._get_authoritative_suppress = get_authoritative_suppress
        self._set_authoritative_suppress = set_authoritative_suppress
        self._get_programmatic_text_suppress = get_programmatic_text_suppress
        self._set_programmatic_text_suppress = set_programmatic_text_suppress
        self._publish_transition = publish_transition
        self._apply_transition = apply_transition
        self._publish_aborted_source_load_attempt = publish_aborted_source_load_attempt
        self._execute_source_load_transaction = execute_source_load_transaction
        self._guard_slider_transaction_invalidation = guard_slider_transaction_invalidation
        self._prepare_source_load_reconciliation = prepare_source_load_reconciliation
        self._apply_source_load_reconciliation_after_acceptance = (
            apply_source_load_reconciliation_after_acceptance
        )
        self._refresh_after_source_load = refresh_after_source_load
        self._source_load_plan_materializes_initial_conditions = (
            source_load_plan_materializes_initial_conditions
        )
        self._set_status_text = set_status_text

    @property
    def canonical_source(self) -> MechanismAuthoringSource:
        return self._session_owner().canonical_source

    @property
    def canonical_reactions_text(self) -> str:
        return str(self._session_owner().canonical_reactions_text)

    @property
    def canonical_state_network_dsl(self) -> str:
        return str(self._session_owner().canonical_state_network_dsl or "")

    @property
    def canonical_full_dsl(self) -> str:
        return str(self._session_owner().canonical_full_dsl)

    def source_from_widgets(self) -> MechanismAuthoringSource:
        reactions_widget, state_editor = self._editor_source_widgets()
        return MechanismAuthoringSource.from_parts(
            reactions_text=str(reactions_widget.toPlainText()),
            state_network_dsl=str(state_editor.get_state_network_dsl() or ""),
        )

    def sync_from_widgets(self, *, authoritative: bool) -> None:
        if not bool(authoritative):
            raise RuntimeError("Reactions drafts are editor text until Apply Changes.")
        self._session_owner().apply_authoritative_source(self.source_from_widgets())

    def sync_state_network_from_widget(self) -> None:
        owner = self._session_owner()
        _reactions_widget, state_editor = self._editor_source_widgets()
        owner.apply_authoritative_source(
            MechanismAuthoringSource.from_parts(
                reactions_text=str(owner.canonical_reactions_text),
                state_network_dsl=str(state_editor.get_state_network_dsl() or ""),
            )
        )

    def sync_after_authoritative_widget_write(self, *, dispatch_consumers: bool) -> None:
        self.sync_from_widgets(authoritative=True)
        self._refresh_reactions_draft_state()
        if bool(dispatch_consumers):
            self._apply_transition()

    def restore_widgets_from_canonical(self) -> None:
        reactions_widget, state_editor = self._editor_source_widgets()
        canonical_reactions = self.canonical_reactions_text
        if str(reactions_widget.toPlainText()) != canonical_reactions:
            reactions_widget.blockSignals(True)
            try:
                reactions_widget.setPlainText(canonical_reactions)
            finally:
                reactions_widget.blockSignals(False)

        canonical_state_network = self.canonical_state_network_dsl
        if str(state_editor.get_state_network_dsl() or "") != canonical_state_network:
            state_editor.blockSignals(True)
            try:
                if canonical_state_network:
                    state_editor.set_state_network_dsl(canonical_state_network)
                else:
                    state_editor.clear()
            finally:
                state_editor.blockSignals(False)

        self._refresh_reactions_draft_state()

    def set_reactions_widget_text_from_canonical(self) -> None:
        self.set_reactions_widget_text(self.canonical_reactions_text)

    def set_reactions_widget_text(self, text: str) -> None:
        reactions_widget, _state_editor = self._editor_source_widgets()
        if str(reactions_widget.toPlainText()) == str(text):
            return
        previous = bool(self._get_programmatic_text_suppress())
        self._set_programmatic_text_suppress(True)
        try:
            reactions_widget.setPlainText(str(text))
        finally:
            self._set_programmatic_text_suppress(previous)

    def set_authoritative_editor_source(
        self,
        source: MechanismAuthoringSource,
        *,
        description: str,
        record_undo: bool = True,
    ) -> bool:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        previous = bool(self._get_authoritative_suppress())
        self._set_authoritative_suppress(True)
        try:
            reactions_widget, state_editor = self._editor_source_widgets()
            current_source = self.source_from_widgets()
            reactions_changed = str(source.reactions_text) != str(current_source.reactions_text)
            state_changed = str(source.state_network_dsl) != str(current_source.state_network_dsl)
            canonical_changed = source != self.canonical_source
            if not reactions_changed and not state_changed and not canonical_changed:
                return False
            old_source = self.canonical_source
            if bool(record_undo):
                command = SetMechanismSourceCommand(
                    new_source=source,
                    old_source=old_source,
                    apply_source=self.apply_undo_stack_replacement,
                    description=str(description),
                )
                self._undo_stack_getter().push(command)
            else:
                self.write_complete_source_to_editor(source)
        finally:
            self._set_authoritative_suppress(previous)
        return True

    def write_complete_source_to_editor(self, source: MechanismAuthoringSource) -> bool:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be a MechanismAuthoringSource.")
        reactions_widget, state_editor = self._editor_source_widgets()
        current_source = self.source_from_widgets()
        reactions_changed = str(source.reactions_text) != str(current_source.reactions_text)
        state_changed = str(source.state_network_dsl) != str(current_source.state_network_dsl)
        canonical_changed = source != self.canonical_source
        if not reactions_changed and not state_changed and not canonical_changed:
            return False
        validate_state = getattr(state_editor, "validate_state_network_dsl_loadable", None)
        if callable(validate_state):
            validate_state(str(source.state_network_dsl))

        previous_reactions_block = reactions_widget.blockSignals(True)
        previous_state_block = state_editor.blockSignals(True)
        try:
            if reactions_changed:
                reactions_widget.setPlainText(str(source.reactions_text))
            if state_changed:
                state_network_dsl = str(source.state_network_dsl)
                if state_network_dsl:
                    state_editor.set_state_network_dsl(state_network_dsl)
                else:
                    state_editor.clear()
        finally:
            state_editor.blockSignals(bool(previous_state_block))
            reactions_widget.blockSignals(bool(previous_reactions_block))

        self._session_owner().apply_authoritative_source(source)
        self._refresh_reactions_draft_state()
        return True

    def apply_undo_stack_replacement(self, source: MechanismAuthoringSource) -> None:
        changed = self.write_complete_source_to_editor(source)
        if not bool(changed):
            return
        if bool(self._get_authoritative_suppress()):
            return
        self._apply_transition(transition_source="mechanism_source_undo_redo")

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
        initial_source = self.canonical_source
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

            new_source = build_source(self.canonical_source)
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
                self.set_authoritative_editor_source(
                    new_source,
                    description=str(description or action_label or "Rewrite mechanism source"),
                    record_undo=bool(record_undo),
                )
            )
        except Exception:
            if not draft_resolution_completed:
                raise
            source_updated = bool(draft_applied or self._canonical_source_differs_from(initial_source))
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

    def prepare_source_load_preflight(
        self,
        *,
        action_label: str,
        guard_action_text: str,
        transition_source: str,
        schedule_runtime_refresh: bool,
        canceled_status_text: str | None = None,
        continue_label: str = "Load",
    ) -> tuple[bool, ReactionsDraftApplyResult | None]:
        draft_resolution = self._draft_resolution_owner.resolve_for_authoritative_source_load(
            str(action_label),
            continue_label=str(continue_label or "Load"),
        )
        if not draft_resolution.may_continue:
            self._publish_aborted_source_load(
                transition_source=transition_source,
                schedule_runtime_refresh=schedule_runtime_refresh,
                draft_apply_result=getattr(draft_resolution, "apply_result", None),
                canceled_status_text=canceled_status_text,
            )
            return False, getattr(draft_resolution, "apply_result", None)
        draft_apply_result = getattr(draft_resolution, "apply_result", None)
        if not self._guard_slider_transaction_invalidation(action_text=str(guard_action_text)):
            self._publish_aborted_source_load(
                transition_source=transition_source,
                schedule_runtime_refresh=schedule_runtime_refresh,
                draft_apply_result=draft_apply_result,
                canceled_status_text=canceled_status_text,
            )
            return False, draft_apply_result
        return True, draft_apply_result

    def prepare_source_load_request(
        self,
        *,
        source: MechanismAuthoringSource,
        action_label: str,
        guard_action_text: str,
        prompt_overwrite: bool,
        transition_source: str,
        schedule_runtime_refresh: bool,
        canceled_status_text: str | None = None,
        continue_label: str = "Load",
        destination_store: object | None = None,
    ) -> tuple[MechanismAuthoringSource, object, ReactionsDraftApplyResult | None] | None:
        preflight_ok, draft_apply_result = self.prepare_source_load_preflight(
            action_label=str(action_label),
            guard_action_text=str(guard_action_text),
            transition_source=str(transition_source or "authoritative_source_load"),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            canceled_status_text=canceled_status_text,
            continue_label=str(continue_label or "Load"),
        )
        if not preflight_ok:
            return None
        source, reconciliation_plan = self._prepare_source_load_reconciliation(
            source,
            prompt_overwrite=bool(prompt_overwrite),
            destination_store=destination_store,
        )
        if reconciliation_plan is None:
            self._publish_aborted_source_load(
                transition_source=transition_source,
                schedule_runtime_refresh=schedule_runtime_refresh,
                draft_apply_result=draft_apply_result,
                canceled_status_text=canceled_status_text,
            )
            return None
        return source, reconciliation_plan, draft_apply_result

    def execute_source_load(
        self,
        *,
        source: MechanismAuthoringSource,
        reconciliation_plan: object,
        draft_apply_result: ReactionsDraftApplyResult | None,
        description: str,
        transition_source: str,
        schedule_runtime_refresh: bool,
        record_undo: bool = True,
        canonical_materialized: bool | None = None,
        full_snapshot_required: bool = False,
        before_materialize: Callable[[], None] | None = None,
        before_refresh: Callable[[], None] | None = None,
        rollback_before_abort_publish: Callable[[], None] | None = None,
    ) -> None:
        previous = bool(self._get_programmatic_text_suppress())
        self._set_programmatic_text_suppress(True)
        try:
            record_source_undo = bool(record_undo) and not self._source_load_plan_materializes_initial_conditions(
                reconciliation_plan
            )
            self._execute_source_load_transaction(
                transition_source=str(transition_source or "authoritative_source_load"),
                schedule_runtime_refresh=bool(schedule_runtime_refresh),
                draft_apply_result=draft_apply_result,
                commit_source=lambda: self.set_authoritative_editor_source(
                    source,
                    description=str(description),
                    record_undo=record_source_undo,
                ),
                materialize_initials=lambda: self._apply_source_load_reconciliation_after_acceptance(
                    reconciliation_plan
                ),
                refresh_after_commit=self._refresh_after_source_load,
                canonical_materialized=canonical_materialized,
                full_snapshot_required=bool(full_snapshot_required),
                before_materialize=before_materialize,
                before_refresh=before_refresh,
                rollback_before_abort_publish=rollback_before_abort_publish,
            )
        finally:
            self._set_programmatic_text_suppress(previous)

    def _publish_aborted_source_load(
        self,
        *,
        transition_source: str,
        schedule_runtime_refresh: bool,
        draft_apply_result: ReactionsDraftApplyResult | None,
        canceled_status_text: str | None,
    ) -> None:
        self._publish_aborted_source_load_attempt(
            transition_source=str(transition_source or "authoritative_source_load"),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            draft_apply_result=draft_apply_result,
        )
        if canceled_status_text is not None:
            self._set_status_text(str(canceled_status_text))

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
            return self.canonical_source != initial_source
        except Exception:
            return False

    def _session_owner(self) -> object:
        owner = self._mechanism_session_owner_getter()
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")
        return owner

    def _editor(self) -> object:
        editor = self._mechanism_editor_getter()
        if editor is None:
            raise RuntimeError("Mechanism editor is unavailable.")
        return editor

    def _editor_source_widgets(self) -> tuple[object, object]:
        editor = self._editor()
        reactions_widget = getattr(editor, "_reactions_text", None)
        state_editor = getattr(editor, "_state_network_editor", None)
        if reactions_widget is None or state_editor is None:
            raise RuntimeError("Mechanism editor widgets are unavailable.")
        for method_name in ("toPlainText", "setPlainText", "blockSignals"):
            if not hasattr(reactions_widget, method_name):
                raise RuntimeError("Mechanism Reactions editor is unavailable.")
        for method_name in ("get_state_network_dsl", "set_state_network_dsl", "clear", "blockSignals"):
            if not hasattr(state_editor, method_name):
                raise RuntimeError("State Network editor is unavailable.")
        return reactions_widget, state_editor

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Mapping, Sequence

from PySide6 import QtCore

from kindred.gui.ports import ReactionsDraftApplyResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SourceLoadOutcome:
    dirty_draft_source_applied: bool = False
    authoritative_source_committed: bool = False
    source_load_completed: bool = False
    affected_set_ids: tuple[str, ...] = ()
    publish_full_snapshot: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dirty_draft_source_applied",
            bool(self.dirty_draft_source_applied),
        )
        object.__setattr__(
            self,
            "authoritative_source_committed",
            bool(self.authoritative_source_committed),
        )
        object.__setattr__(
            self,
            "source_load_completed",
            bool(self.source_load_completed),
        )
        object.__setattr__(
            self,
            "affected_set_ids",
            tuple(
                dict.fromkeys(
                    str(set_id).strip()
                    for set_id in (self.affected_set_ids or ())
                    if str(set_id).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "publish_full_snapshot",
            bool(self.publish_full_snapshot),
        )


class CanonicalInitialsTransitionOwner:
    """Owns canonical-initials transition queueing and publication decisions."""

    def __init__(
        self,
        *,
        batch_set_ids_for_rows: Callable[[Sequence[int]], tuple[str, ...]],
        canonical_initials_identity_by_set_id: Callable[[], Mapping[str, object]],
        canonical_initials_identity_for_set_ids: Callable[
            [Sequence[str]],
            Mapping[str, object],
        ],
        apply_authoritative_transition: Callable[..., object],
        apply_canonical_initials_transition_effects: Callable[..., None],
        qt_parent: QtCore.QObject | None = None,
    ) -> None:
        self._batch_set_ids_for_rows = batch_set_ids_for_rows
        self._canonical_initials_identity_by_set_id = canonical_initials_identity_by_set_id
        self._canonical_initials_identity_for_set_ids = canonical_initials_identity_for_set_ids
        self._apply_authoritative_transition = apply_authoritative_transition
        self._apply_canonical_initials_transition_effects = (
            apply_canonical_initials_transition_effects
        )
        self._pending_set_ids: tuple[str, ...] = ()
        self._pending_transition_source = "batch_initials_table_edit"
        self._pending_discard_dirty_preview = True
        self._queued = False
        self._timer = QtCore.QTimer(qt_parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.flush_pending_transition)

    @property
    def queued(self) -> bool:
        return bool(self._queued)

    def queue(
        self,
        *,
        affected_set_ids: Sequence[str],
        transition_source: str,
        discard_dirty_preview: bool,
        debounce_ms: int,
    ) -> None:
        normalized = tuple(
            dict.fromkeys(
                str(set_id)
                for set_id in (affected_set_ids or ())
                if str(set_id)
            )
        )
        if not normalized:
            return
        pending = list(self._pending_set_ids)
        for set_id in normalized:
            if set_id not in pending:
                pending.append(set_id)
        self._pending_set_ids = tuple(pending)
        self._pending_transition_source = str(
            transition_source or "batch_initials_table_edit"
        )
        self._pending_discard_dirty_preview = bool(discard_dirty_preview)
        self._queued = True
        if QtCore.QCoreApplication.instance() is None:
            self.flush_pending_transition()
            return
        self._timer.start(max(0, int(debounce_ms)))

    def consume_pending_transition(self) -> tuple[tuple[str, ...], str, bool]:
        if self._timer.isActive():
            self._timer.stop()
        affected_set_ids = tuple(self._pending_set_ids)
        transition_source = str(
            self._pending_transition_source or "batch_initials_table_edit"
        )
        discard_dirty_preview = bool(self._pending_discard_dirty_preview)
        self._pending_set_ids = ()
        self._queued = False
        return affected_set_ids, transition_source, discard_dirty_preview

    def flush_pending_transition(self) -> object | None:
        (
            affected_set_ids,
            transition_source,
            discard_dirty_preview,
        ) = self.consume_pending_transition()
        if not affected_set_ids:
            return None
        return self._publish_set_ids(
            transition_source=str(transition_source or "batch_initials_table_edit"),
            affected_set_ids=affected_set_ids,
            discard_dirty_preview=bool(discard_dirty_preview),
        )

    def publish_pending_canonical_initials_for_run(
        self,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
    ) -> object | None:
        if not (
            self._draft_result_changed_source(draft_apply_result)
            or self.queued
        ):
            return None
        return self.publish_authoritative_transition(
            "authoritative_change",
            draft_apply_result=(
                draft_apply_result
                if self._draft_result_changed_source(draft_apply_result)
                else None
            ),
        )

    def publish_applied_draft_run_abort(
        self,
        transition_source: str,
        *,
        draft_apply_result: ReactionsDraftApplyResult | None,
    ) -> object | None:
        if not self._draft_result_changed_source(draft_apply_result):
            return None
        transition_source_s = str(
            transition_source or "run_preflight_applied_draft_abort"
        )
        affected_rows = tuple(getattr(draft_apply_result, "affected_rows", ()) or ())
        affected_set_ids = self._batch_set_ids_for_rows(affected_rows)
        if affected_set_ids:
            return self._publish_set_ids(
                transition_source=transition_source_s,
                affected_set_ids=affected_set_ids,
                discard_dirty_preview=True,
            )
        return self._publish_source_only_transition(
            transition_source=transition_source_s,
            discard_dirty_preview=True,
        )

    def publish_canonical_initials_transition(
        self,
        *,
        affected_set_ids: Sequence[str],
        transition_source: str,
        discard_dirty_preview: bool = True,
    ) -> object | None:
        transition_source_s = str(transition_source or "canonical_batch_initials_change")
        affected_set_ids_t = tuple(
            dict.fromkeys(
                str(set_id) for set_id in (affected_set_ids or ()) if str(set_id)
            )
        )
        discard_dirty_preview_b = bool(discard_dirty_preview)
        if not affected_set_ids_t:
            return None
        return self._publish_set_ids(
            transition_source=transition_source_s,
            affected_set_ids=affected_set_ids_t,
            discard_dirty_preview=discard_dirty_preview_b,
        )

    def publish_authoritative_source_load(
        self,
        *,
        transition_source: str,
        schedule_runtime_refresh: bool,
    ) -> object:
        transition_source_s = str(transition_source or "authoritative_source_load")
        discard_dirty_preview = True

        if self.queued:
            (
                _pending_set_ids,
                _pending_transition_source,
                pending_discard_dirty_preview,
            ) = self.consume_pending_transition()
            discard_dirty_preview = bool(pending_discard_dirty_preview)

        transition_kwargs = {
            "schedule_runtime_refresh": bool(schedule_runtime_refresh),
            "force_runtime_invalidation": True,
        }
        canonical_initials_identity = dict(
            self._canonical_initials_identity_by_set_id() or {}
        )
        full_set_ids = tuple(
            dict.fromkeys(
                str(set_id)
                for set_id in canonical_initials_identity.keys()
                if str(set_id)
            )
        )
        if not full_set_ids:
            outcome = self._apply_authoritative_transition(
                transition_source=transition_source_s,
                **transition_kwargs,
            )
            self._apply_canonical_initials_transition_effects(
                outcome,
                affected_set_ids=(),
                discard_dirty_preview=discard_dirty_preview,
            )
            return outcome
        outcome = self._apply_authoritative_transition(
            transition_source=transition_source_s,
            **transition_kwargs,
            canonical_batch_initials_by_set_id=canonical_initials_identity,
            affected_set_ids=full_set_ids,
            canonical_batch_initials_scope_is_partial=False,
        )
        self._apply_canonical_initials_transition_effects(
            outcome,
            affected_set_ids=full_set_ids,
            discard_dirty_preview=discard_dirty_preview,
        )
        return outcome

    def _build_source_load_outcome(
        self,
        *,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
        source_committed: bool = False,
        materialization_result: object | None = None,
        canonical_materialized: bool | None = None,
        full_snapshot_required: bool = False,
        source_load_completed: bool = False,
    ) -> _SourceLoadOutcome:
        materialized_rows = tuple(
            int(row)
            for row in (getattr(materialization_result, "affected_rows", ()) or ())
        )
        species_changed = bool(
            getattr(materialization_result, "species_changed", False)
        )
        materialized = (
            bool(canonical_materialized)
            if canonical_materialized is not None
            else bool(materialized_rows or species_changed)
        )
        draft_rows = (
            tuple(int(row) for row in (getattr(draft_apply_result, "affected_rows", ()) or ()))
            if self._draft_result_changed_source(draft_apply_result)
            else ()
        )
        affected_set_ids = self._batch_set_ids_for_rows(materialized_rows or draft_rows)
        publish_full_snapshot = bool(
            source_load_completed
            and
            (source_committed or materialized)
            and (full_snapshot_required or materialized)
        )
        return _SourceLoadOutcome(
            dirty_draft_source_applied=self._draft_result_changed_source(
                draft_apply_result
            ),
            authoritative_source_committed=bool(source_committed),
            source_load_completed=bool(source_load_completed),
            affected_set_ids=affected_set_ids,
            publish_full_snapshot=publish_full_snapshot,
        )

    def publish_aborted_source_load_attempt(
        self,
        *,
        transition_source: str,
        schedule_runtime_refresh: bool,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
    ) -> object | None:
        return self._publish_source_load_outcome(
            transition_source=str(transition_source or "authoritative_source_load"),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            outcome=self._build_source_load_outcome(
                draft_apply_result=draft_apply_result,
            ),
        )

    def execute_source_load_transaction(
        self,
        *,
        transition_source: str,
        schedule_runtime_refresh: bool,
        draft_apply_result: ReactionsDraftApplyResult | None,
        commit_source: Callable[[], bool],
        materialize_initials: Callable[[], object | None],
        refresh_after_commit: Callable[[], None],
        canonical_materialized: bool | None = None,
        full_snapshot_required: bool = False,
        before_materialize: Callable[[], None] | None = None,
        before_refresh: Callable[[], None] | None = None,
        rollback_before_abort_publish: Callable[[], None] | None = None,
    ) -> _SourceLoadOutcome:
        transition_source_s = str(transition_source or "authoritative_source_load")
        abort_outcome = self._build_source_load_outcome(
            draft_apply_result=draft_apply_result,
        )
        publication_attempted = False
        try:
            source_committed = bool(commit_source())
            if source_committed:
                abort_outcome = self._build_source_load_outcome(
                    draft_apply_result=draft_apply_result,
                    source_committed=True,
                )
            if callable(before_materialize):
                before_materialize()
            materialization_result = materialize_initials()
            abort_outcome = self._build_source_load_outcome(
                draft_apply_result=draft_apply_result,
                source_committed=source_committed,
                materialization_result=materialization_result,
                canonical_materialized=canonical_materialized,
            )
            source_load_outcome = self._build_source_load_outcome(
                draft_apply_result=draft_apply_result,
                source_committed=source_committed,
                materialization_result=materialization_result,
                canonical_materialized=canonical_materialized,
                full_snapshot_required=bool(full_snapshot_required),
                source_load_completed=True,
            )
            if callable(before_refresh):
                before_refresh()
            refresh_after_commit()
            abort_outcome = source_load_outcome
            publication_attempted = True
            self._publish_source_load_outcome(
                transition_source=transition_source_s,
                schedule_runtime_refresh=bool(schedule_runtime_refresh),
                outcome=source_load_outcome,
            )
            return source_load_outcome
        except Exception:
            if callable(rollback_before_abort_publish):
                try:
                    rollback_before_abort_publish()
                except Exception:
                    logger.exception(
                        "Failed to roll back authoritative source-load state before abort publication"
                    )
            if not publication_attempted:
                self._publish_source_load_outcome(
                    transition_source=transition_source_s,
                    schedule_runtime_refresh=bool(schedule_runtime_refresh),
                    outcome=abort_outcome,
                )
            raise

    def _publish_source_load_outcome(
        self,
        *,
        transition_source: str,
        schedule_runtime_refresh: bool,
        outcome: _SourceLoadOutcome,
    ) -> object | None:
        if not isinstance(outcome, _SourceLoadOutcome):
            raise TypeError(
                "_publish_source_load_outcome() requires _SourceLoadOutcome."
            )
        queued = self.queued
        if (
            bool(outcome.publish_full_snapshot)
            or (
                bool(outcome.source_load_completed)
                and bool(outcome.authoritative_source_committed)
                and bool(queued)
            )
        ):
            return self.publish_authoritative_source_load(
                transition_source=str(transition_source or "authoritative_source_load"),
                schedule_runtime_refresh=bool(schedule_runtime_refresh),
            )
        if bool(outcome.authoritative_source_committed):
            discard_dirty_preview = True
            if self.queued:
                (
                    _pending_set_ids,
                    _pending_transition_source,
                    pending_discard_dirty_preview,
                ) = self.consume_pending_transition()
                discard_dirty_preview = bool(pending_discard_dirty_preview)
            return self._publish_source_only_transition(
                transition_source=str(transition_source or "authoritative_source_load"),
                discard_dirty_preview=discard_dirty_preview,
                transition_kwargs={
                    "schedule_runtime_refresh": bool(schedule_runtime_refresh),
                    "force_runtime_invalidation": True,
                },
            )
        affected_set_ids = tuple(outcome.affected_set_ids)

        discard_dirty_preview = True
        if self.queued:
            (
                pending_set_ids,
                _pending_transition_source,
                pending_discard_dirty_preview,
            ) = self.consume_pending_transition()
            affected_set_ids = tuple(
                dict.fromkeys((*affected_set_ids, *pending_set_ids))
            )
            discard_dirty_preview = bool(pending_discard_dirty_preview)

        if (
            not bool(outcome.dirty_draft_source_applied)
            and not affected_set_ids
        ):
            return None

        if affected_set_ids:
            return self._publish_set_ids(
                transition_source=str(transition_source or "authoritative_source_load"),
                affected_set_ids=affected_set_ids,
                discard_dirty_preview=discard_dirty_preview,
            )
        return self._publish_source_only_transition(
            transition_source=str(transition_source or "authoritative_source_load"),
            discard_dirty_preview=discard_dirty_preview,
        )

    def publish_authoritative_transition(
        self,
        transition_source: str,
        *,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
        discard_dirty_preview: bool = True,
    ) -> object:
        has_draft_apply_result = self._draft_result_changed_source(draft_apply_result)
        transition_source_s = str(transition_source or "authoritative_change")
        affected_rows = (
            tuple(getattr(draft_apply_result, "affected_rows", ()) or ())
            if has_draft_apply_result
            else ()
        )
        affected_set_ids = self._batch_set_ids_for_rows(affected_rows)
        source_discard_dirty_preview = bool(discard_dirty_preview)

        if self.queued:
            (
                pending_set_ids,
                pending_transition_source,
                pending_discard_dirty_preview,
            ) = self.consume_pending_transition()
            pending_set_ids = tuple(
                dict.fromkeys(str(set_id) for set_id in pending_set_ids if str(set_id))
            )
            if pending_set_ids:
                if (
                    not has_draft_apply_result
                    and not affected_set_ids
                    and transition_source_s == "authoritative_change"
                ):
                    affected_set_ids = pending_set_ids
                    source_discard_dirty_preview = bool(pending_discard_dirty_preview)
                else:
                    self._publish_set_ids(
                        transition_source=str(
                            pending_transition_source
                            or "batch_initials_table_edit"
                        ),
                        affected_set_ids=pending_set_ids,
                        discard_dirty_preview=bool(pending_discard_dirty_preview),
                    )
            if (
                not has_draft_apply_result
                and transition_source_s == "authoritative_change"
                and str(pending_transition_source or "").strip()
            ):
                transition_source_s = str(pending_transition_source)

        if not affected_set_ids:
            return self._publish_source_only_transition(
                transition_source=transition_source_s,
                discard_dirty_preview=source_discard_dirty_preview,
            )
        return self._publish_set_ids(
            transition_source=transition_source_s,
            affected_set_ids=affected_set_ids,
            discard_dirty_preview=source_discard_dirty_preview,
        )

    def _publish_source_only_transition(
        self,
        *,
        transition_source: str,
        discard_dirty_preview: bool,
        transition_kwargs: Mapping[str, object] | None = None,
    ) -> object:
        extra_kwargs = dict(transition_kwargs or {})
        outcome = self._apply_authoritative_transition(
            transition_source=str(transition_source or "authoritative_change"),
            **extra_kwargs,
        )
        self._apply_canonical_initials_transition_effects(
            outcome,
            affected_set_ids=(),
            discard_dirty_preview=bool(discard_dirty_preview),
        )
        return outcome

    def _publish_set_ids(
        self,
        *,
        transition_source: str,
        affected_set_ids: Sequence[str],
        discard_dirty_preview: bool,
        transition_kwargs: Mapping[str, object] | None = None,
    ) -> object:
        extra_kwargs = dict(transition_kwargs or {})
        affected_set_ids_t = tuple(
            dict.fromkeys(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id))
        )
        if not affected_set_ids_t:
            return self._apply_authoritative_transition(
                transition_source=str(transition_source or "authoritative_change"),
                **extra_kwargs,
            )
        canonical_initials_identity = self._canonical_initials_identity_for_set_ids(
            affected_set_ids_t
        )
        outcome = self._apply_authoritative_transition(
            transition_source=str(transition_source or "authoritative_change"),
            **extra_kwargs,
            canonical_batch_initials_by_set_id=canonical_initials_identity,
            affected_set_ids=affected_set_ids_t,
            canonical_batch_initials_scope_is_partial=True,
        )
        self._apply_canonical_initials_transition_effects(
            outcome,
            affected_set_ids=affected_set_ids_t,
            discard_dirty_preview=discard_dirty_preview,
        )
        return outcome

    @staticmethod
    def _draft_result_changed_source(
        draft_apply_result: ReactionsDraftApplyResult | None,
    ) -> bool:
        return bool(
            draft_apply_result is not None
            and bool(getattr(draft_apply_result, "source_applied", False))
        )

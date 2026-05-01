"""Authoritative mechanism transition state for runtime lifecycle decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


def _normalize_text_for_identity(text: str) -> str:
    return "\n".join(" ".join(str(line).split()) for line in str(text or "").splitlines()).strip()


def _normalize_mapping_identity(values: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping):
        return ()
    entries: list[tuple[str, str]] = []
    for raw_key, raw_value in values.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        entries.append((key, str(raw_value or "")))
    return tuple(sorted(entries))


def _normalize_set_ids(set_ids: Sequence[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_set_id in set_ids or ():
        set_id = str(raw_set_id or "").strip()
        if not set_id or set_id in seen:
            continue
        seen.add(set_id)
        ordered.append(set_id)
    return tuple(ordered)


def _changed_set_ids(
    previous: tuple[tuple[str, str], ...] | None,
    current: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if previous is None:
        return ()
    old = dict(previous)
    new = dict(current)
    changed = [set_id for set_id in sorted(set(old) | set(new)) if old.get(set_id) != new.get(set_id)]
    return tuple(changed)


@dataclass(frozen=True)
class AuthoritativeMechanismSnapshot:
    reactions_text: str
    state_network_text: str
    normalized_reactions_text: str
    normalized_state_network_text: str

    @classmethod
    def from_texts(
        cls,
        *,
        reactions_text: str,
        state_network_text: str = "",
    ) -> "AuthoritativeMechanismSnapshot":
        reactions = str(reactions_text or "")
        state_network = str(state_network_text or "")
        return cls(
            reactions_text=reactions,
            state_network_text=state_network,
            normalized_reactions_text=_normalize_text_for_identity(reactions),
            normalized_state_network_text=_normalize_text_for_identity(state_network),
        )

    @property
    def normalized_identity(self) -> tuple[str, str]:
        return (self.normalized_reactions_text, self.normalized_state_network_text)

    def matches(self, other: object) -> bool:
        if not isinstance(other, AuthoritativeMechanismSnapshot):
            return False
        return self.normalized_identity == other.normalized_identity


@dataclass(frozen=True)
class MechanismTransitionOutcome:
    epoch: int
    source: str
    runtime_invalidation_required: bool
    runtime_input_invalidation_required: bool
    active_work_supersede_required: bool
    display_cache_invalidation_allowed: bool
    readiness_schedule_required: bool
    readiness_schedule_deferred: bool
    pending_init_preservation: bool
    affected_set_ids: tuple[str, ...] = ()

    @property
    def stale_result_epoch(self) -> int:
        return int(self.epoch)

    @property
    def cache_stale_scope_is_global(self) -> bool:
        """Whether scientific result-cache staleness is global for this transition."""
        return bool(self.runtime_invalidation_required)

    @property
    def cache_stale_set_ids(self) -> tuple[str, ...]:
        """Set IDs stale for scoped cache invalidation; empty when global or none."""
        if self.cache_stale_scope_is_global or not self.runtime_input_invalidation_required:
            return ()
        return _normalize_set_ids(self.affected_set_ids)

    @property
    def display_clear_scope_is_global(self) -> bool:
        """Whether visible display clearing is global for this transition."""
        return bool(self.runtime_invalidation_required)

    @property
    def display_clear_set_ids(self) -> tuple[str, ...]:
        """Set IDs whose visible display is untruthful; empty when global or none."""
        if self.display_clear_scope_is_global or not self.runtime_input_invalidation_required:
            return ()
        return _normalize_set_ids(self.affected_set_ids)

    @property
    def active_work_supersede_scope_is_global(self) -> bool:
        """Whether active-work publication rejection is global for this transition."""
        return bool(self.runtime_invalidation_required)

    @property
    def active_work_supersede_set_ids(self) -> tuple[str, ...]:
        """Set IDs whose in-flight work is stale; empty when global or none."""
        if self.active_work_supersede_scope_is_global or not self.runtime_input_invalidation_required:
            return ()
        return _normalize_set_ids(self.affected_set_ids)

    @property
    def dirty_preview_reset_scope_is_global(self) -> bool:
        """Whether dirty preview reset is global for this transition."""
        return bool(self.runtime_invalidation_required)

    @property
    def dirty_preview_reset_set_ids(self) -> tuple[str, ...]:
        """Set IDs whose dirty preview state is obsolete; empty when global or none."""
        if self.dirty_preview_reset_scope_is_global or not self.runtime_input_invalidation_required:
            return ()
        return _normalize_set_ids(self.affected_set_ids)


class MechanismRuntimeTransitionService:
    """Owns non-GUI state for authoritative mechanism/runtime transitions."""

    def __init__(
        self,
        *,
        initial_snapshot: Optional[AuthoritativeMechanismSnapshot] = None,
        initial_canonical_batch_initials_by_set_id: Mapping[str, object] | None = None,
    ) -> None:
        self._epoch = 0
        self._current_snapshot = initial_snapshot or AuthoritativeMechanismSnapshot.from_texts(
            reactions_text="",
            state_network_text="",
        )
        self._current_canonical_batch_initials_identity: tuple[tuple[str, str], ...] | None = (
            _normalize_mapping_identity(initial_canonical_batch_initials_by_set_id)
            if initial_canonical_batch_initials_by_set_id is not None
            else None
        )
        self._pending_init_snapshot: Optional[AuthoritativeMechanismSnapshot] = None
        self._pending_readiness_epoch: Optional[int] = None

    @property
    def current_epoch(self) -> int:
        return int(self._epoch)

    def reset_current_snapshot(
        self,
        snapshot: AuthoritativeMechanismSnapshot,
        *,
        canonical_batch_initials_by_set_id: Mapping[str, object] | None = None,
    ) -> None:
        self._current_snapshot = snapshot
        if canonical_batch_initials_by_set_id is not None:
            self._current_canonical_batch_initials_identity = _normalize_mapping_identity(
                canonical_batch_initials_by_set_id
            )
        self._pending_init_snapshot = None
        self._pending_readiness_epoch = None

    def arm_pending_init_result_guard(self, *, rewrite: str, state_network_text: str = "") -> None:
        self._pending_init_snapshot = AuthoritativeMechanismSnapshot.from_texts(
            reactions_text=str(rewrite or ""),
            state_network_text=str(state_network_text or ""),
        )

    def consume_pending_init_result_guard(self) -> Optional[AuthoritativeMechanismSnapshot]:
        snapshot = self._pending_init_snapshot
        self._pending_init_snapshot = None
        return snapshot

    def consume_pending_readiness_epoch(self) -> Optional[int]:
        epoch = self._pending_readiness_epoch
        self._pending_readiness_epoch = None
        return epoch

    def apply_authoritative_transition(
        self,
        snapshot: AuthoritativeMechanismSnapshot,
        *,
        source: str,
        force_runtime_invalidation: bool = False,
        edit_session_active: bool = False,
        input_suppressed: bool = False,
        slider_runtime_invalidation_suppressed: bool = False,
        schedule_runtime_refresh: bool = True,
        canonical_batch_initials_by_set_id: Mapping[str, object] | None = None,
        affected_set_ids: Sequence[str] = (),
    ) -> MechanismTransitionOutcome:
        source_s = str(source or "authoritative_change")
        pending_init_preservation = False
        runtime_invalidation_required = False
        canonical_identity_supplied = canonical_batch_initials_by_set_id is not None
        canonical_identity = (
            _normalize_mapping_identity(canonical_batch_initials_by_set_id)
            if canonical_identity_supplied
            else self._current_canonical_batch_initials_identity
        )
        canonical_changed_set_ids = (
            _changed_set_ids(self._current_canonical_batch_initials_identity, canonical_identity or ())
            if canonical_identity_supplied
            else ()
        )
        affected_set_ids_t = _normalize_set_ids(affected_set_ids) or canonical_changed_set_ids
        runtime_input_invalidation_required = bool(
            canonical_identity_supplied
            and (
                (
                    self._current_canonical_batch_initials_identity is not None
                    and canonical_identity != self._current_canonical_batch_initials_identity
                )
                or (
                    self._current_canonical_batch_initials_identity is None
                    and bool(affected_set_ids_t)
                )
            )
        )

        if bool(force_runtime_invalidation):
            self._pending_init_snapshot = None
            runtime_invalidation_required = True
        else:
            pending_snapshot = self._pending_init_snapshot
            if pending_snapshot is not None:
                pending_init_source = source_s in {
                    "pending_init_migration",
                    "authoritative_change",
                    "authoritative_editor_rewrite",
                }
                if snapshot.matches(pending_snapshot) and (
                    pending_init_source or not runtime_input_invalidation_required
                ):
                    pending_init_preservation = True
                    self._current_snapshot = snapshot
                    if canonical_identity_supplied:
                        self._current_canonical_batch_initials_identity = canonical_identity or ()
                    return MechanismTransitionOutcome(
                        epoch=int(self._epoch),
                        source=source_s,
                        runtime_invalidation_required=False,
                        runtime_input_invalidation_required=False,
                        active_work_supersede_required=False,
                        display_cache_invalidation_allowed=False,
                        readiness_schedule_required=False,
                        readiness_schedule_deferred=False,
                        pending_init_preservation=True,
                        affected_set_ids=(),
                    )
                self._pending_init_snapshot = None
            runtime_invalidation_required = not snapshot.matches(self._current_snapshot)
            edit_session_blocks_transition = bool(edit_session_active) and (
                runtime_invalidation_required or not runtime_input_invalidation_required
            )
            if (
                edit_session_blocks_transition
                or bool(input_suppressed)
                or bool(slider_runtime_invalidation_suppressed)
            ):
                return MechanismTransitionOutcome(
                    epoch=int(self._epoch),
                    source=source_s,
                    runtime_invalidation_required=False,
                    runtime_input_invalidation_required=False,
                    active_work_supersede_required=False,
                    display_cache_invalidation_allowed=False,
                    readiness_schedule_required=False,
                    readiness_schedule_deferred=False,
                    pending_init_preservation=False,
                    affected_set_ids=(),
                )

        if runtime_invalidation_required:
            self._epoch += 1
            self._current_snapshot = snapshot
            if canonical_identity_supplied:
                self._current_canonical_batch_initials_identity = canonical_identity or ()
            elif bool(force_runtime_invalidation):
                self._current_canonical_batch_initials_identity = None
            epoch = int(self._epoch)
            readiness_schedule_required = bool(schedule_runtime_refresh)
            readiness_schedule_deferred = not readiness_schedule_required
            if readiness_schedule_deferred:
                self._pending_readiness_epoch = epoch
            else:
                self._pending_readiness_epoch = None
            return MechanismTransitionOutcome(
                epoch=epoch,
                source=source_s,
                runtime_invalidation_required=True,
                runtime_input_invalidation_required=True,
                active_work_supersede_required=True,
                display_cache_invalidation_allowed=True,
                readiness_schedule_required=readiness_schedule_required,
                readiness_schedule_deferred=readiness_schedule_deferred,
                pending_init_preservation=pending_init_preservation,
                affected_set_ids=(),
            )

        if runtime_input_invalidation_required:
            self._epoch += 1
            self._current_snapshot = snapshot
            if canonical_identity_supplied:
                self._current_canonical_batch_initials_identity = canonical_identity or ()
            epoch = int(self._epoch)
            return MechanismTransitionOutcome(
                epoch=epoch,
                source=source_s,
                runtime_invalidation_required=False,
                runtime_input_invalidation_required=True,
                active_work_supersede_required=True,
                display_cache_invalidation_allowed=True,
                readiness_schedule_required=False,
                readiness_schedule_deferred=False,
                pending_init_preservation=pending_init_preservation,
                affected_set_ids=affected_set_ids_t,
            )

        self._current_snapshot = snapshot
        if canonical_identity_supplied:
            self._current_canonical_batch_initials_identity = canonical_identity or ()
        return MechanismTransitionOutcome(
            epoch=int(self._epoch),
            source=source_s,
            runtime_invalidation_required=False,
            runtime_input_invalidation_required=False,
            active_work_supersede_required=False,
            display_cache_invalidation_allowed=False,
            readiness_schedule_required=False,
            readiness_schedule_deferred=False,
            pending_init_preservation=pending_init_preservation,
            affected_set_ids=(),
        )

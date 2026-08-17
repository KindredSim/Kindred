"""Clock-driven, retargetable transitions for transient plot projections."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kindred.gui.interactive_preview import (
    InteractivePreviewBatchIdentity,
    InteractivePreviewFrame,
    InteractivePreviewFrameDisposition,
    InteractivePreviewFrameDispositionKind,
    InteractivePreviewPresentedReceipt,
    InteractivePreviewTerminalOutcome,
    InteractivePreviewTerminalState,
)
from kindred.gui.workspace_result_presenter import (
    InteractiveDisplayCurve,
    InteractiveDisplayProjection,
    WorkspaceResultPresenter,
    WorkspaceResultProjection,
)


_DISPLAY_EPSILON = 1e-9

__all__ = [
    "InteractiveDisplayController",
    "InteractiveDisplayLayer",
    "InteractiveDisplayState",
]


def _same_curve_topology(
    left: InteractiveDisplayCurve,
    right: InteractiveDisplayCurve,
) -> bool:
    """Return whether interpolation can preserve all topology boundaries."""

    left_segments = left.segment_ids
    right_segments = right.segment_ids
    if left_segments is None:
        return right_segments is None
    if right_segments is None:
        return False
    left_discontinuities = left.discontinuity_ids
    right_discontinuities = right.discontinuity_ids
    if left_discontinuities is None or right_discontinuities is None:
        return False
    return bool(
        np.array_equal(left_segments, right_segments)
        and left.segment_keys == right.segment_keys
        and np.array_equal(left_discontinuities, right_discontinuities)
        and left.intervention_connectors == right.intervention_connectors
    )


@dataclass(frozen=True, slots=True)
class InteractiveDisplayLayer:
    """One immutable plot roster and its physical cross-fade opacity."""

    projection: InteractiveDisplayProjection
    opacity: float

    def __post_init__(self) -> None:
        if not isinstance(self.projection, InteractiveDisplayProjection):
            raise TypeError("projection must be an InteractiveDisplayProjection.")
        opacity = float(self.opacity)
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValueError("display-layer opacity must be between zero and one.")
        object.__setattr__(self, "opacity", opacity)


@dataclass(frozen=True, slots=True)
class InteractiveDisplayState:
    """The complete visible transient roster at one display-clock instant."""

    layers: tuple[InteractiveDisplayLayer, ...]
    settled: bool
    masked_set_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if any(not isinstance(layer, InteractiveDisplayLayer) for layer in layers):
            raise TypeError("layers must contain InteractiveDisplayLayer values.")
        masked = frozenset(str(value).strip() for value in self.masked_set_ids)
        if any(not value for value in masked):
            raise ValueError("masked set IDs must be nonblank.")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "settled", bool(self.settled))
        object.__setattr__(self, "masked_set_ids", masked)


@dataclass(frozen=True, slots=True)
class _Transition:
    source: tuple[InteractiveDisplayLayer, ...]
    target: InteractiveDisplayProjection
    started_at: float
    interpolate: bool


@dataclass(slots=True)
class _BatchTargetState:
    """Physical and lifecycle settlement for one immutable batch target.

    A target has one physical disposition for the one frame it may emit.  The
    frame itself is retained separately when it is renderable so a sibling
    frame can be composed without making render-map emptiness the batch
    lifecycle signal.
    """

    frame: InteractivePreviewFrame | None = None
    disposition: InteractivePreviewFrameDispositionKind | None = None
    terminal: InteractivePreviewTerminalState | None = None


class InteractiveDisplayController:
    """Present newest transient targets at a bounded, fake-clock-testable cadence.

    The controller owns no Qt objects.  A physical surface supplies a one-shot
    scheduler and a render callback; tests can instead advance a fake clock and
    call :meth:`tick` directly.
    """

    def __init__(
        self,
        *,
        render: Callable[[InteractiveDisplayState], None],
        disposition: Callable[[InteractivePreviewFrameDisposition], None],
        request_tick: Callable[[float], None] | None = None,
        cancel_tick: Callable[[], None] | None = None,
        render_failed: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cadence_seconds: float = 1.0 / 30.0,
        transition_seconds: float = 0.12,
    ) -> None:
        if not callable(render):
            raise TypeError("render must be callable.")
        if not callable(disposition):
            raise TypeError("disposition must be callable.")
        if request_tick is not None and not callable(request_tick):
            raise TypeError("request_tick must be callable.")
        if cancel_tick is not None and not callable(cancel_tick):
            raise TypeError("cancel_tick must be callable.")
        if render_failed is not None and not callable(render_failed):
            raise TypeError("render_failed must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        cadence = float(cadence_seconds)
        duration = float(transition_seconds)
        if not np.isfinite(cadence) or cadence <= 0.0:
            raise ValueError("cadence_seconds must be positive and finite.")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("transition_seconds must be positive and finite.")
        self._render = render
        self._emit_disposition = disposition
        self._request_tick = request_tick
        self._cancel_tick = cancel_tick
        self._render_failed = render_failed
        self._clock = clock
        self._cadence = cadence
        self._duration = duration
        self._visible: tuple[InteractiveDisplayLayer, ...] | None = None
        self._transition: _Transition | None = None
        self._pending: InteractiveDisplayProjection | None = None
        self._last_presented_at: float | None = None
        self._tick_requested = False
        self._frozen = False
        self._retired = False
        self._batch: InteractivePreviewBatchIdentity | None = None
        self._issued_batch: InteractivePreviewBatchIdentity | None = None
        self._presented_receipt: InteractivePreviewPresentedReceipt | None = None
        self._next_generation = 0
        self._generation_floor: int | None = None
        self._workspace_projection: WorkspaceResultProjection | None = None
        self._frames_by_set: dict[str, InteractivePreviewFrame] = {}
        self._target_dispositions: dict[
            str,
            InteractivePreviewFrameDispositionKind,
        ] = {}
        self._unsettled_frames: dict[str, InteractivePreviewFrame] = {}
        # This is the sole transient batch protocol state.  It is bounded by
        # the immutable target roster and is discarded atomically on seal;
        # settled frame keys are never retained as per-frame tombstones.
        self._batch_targets: dict[str, _BatchTargetState] = {}
        self._sealed_batch: InteractivePreviewBatchIdentity | None = None
        self._sealed_target_dispositions: dict[
            str,
            InteractivePreviewFrameDispositionKind,
        ] = {}
        # Exact release fences the approximation's admission/cadence without
        # retiring the batch identity.  The exact worker still needs that
        # identity to settle the batch after its result is published.
        self._exact_release_fenced = False

    @property
    def has_visible_frame(self) -> bool:
        return bool(
            self._visible is not None
            and any(layer.projection.curves for layer in self._visible)
        )

    @property
    def has_current_batch(self) -> bool:
        """Whether a batch still owns transient physical settlement."""

        return self._batch is not None and not self._retired

    @property
    def has_renderable_frame(self) -> bool:
        """Whether current raw custody can project at least one curve."""

        projection = self._workspace_projection
        if self._batch is None or projection is None:
            return False
        return any(
            WorkspaceResultPresenter.compose_interactive_display(
                frame,
                projection=projection,
            ).curves
            for frame in self._frames_by_set.values()
        )

    @property
    def masking_intent(self) -> frozenset[str]:
        """Return the controller-owned completed-foreground mask roster."""

        visible = self._visible or ()
        return self._masked_set_ids(visible)

    def exact_release_batch(self) -> InteractivePreviewPresentedReceipt | None:
        """Atomically consume the receipt held by the physically visible batch.

        Allocation and worker emission do not establish physical custody.  A
        release is valid only after a nonempty frame has returned successfully
        from the physical renderer and issued this receipt.  Queued frames are
        retired as part of the same release transition.
        """

        receipt = self._presented_receipt
        if receipt is None:
            # A frame can be admitted into the cadence queue before its first
            # physical presentation.  Exact release must retire that queued
            # custody and fence the batch, while manufacturing no receipt.
            if self._batch is None:
                # Allocation publishes the worker-facing identity before the
                # first frame can establish controller custody.  Fence that
                # issued identity as well, otherwise its late callback could
                # open a fresh batch after release.
                issued = self._issued_batch
                if (
                    issued is None
                    or issued == self._sealed_batch
                    or self._exact_release_fenced
                ):
                    return None
                self._generation_floor = issued.generation
                self._sealed_batch = issued
                self._sealed_target_dispositions.clear()
                self._retired = True
                self._exact_release_fenced = True
                return None
            if (
                self._retired
                or self._exact_release_fenced
            ):
                return None
            for frame in tuple(self._unsettled_frames.values()):
                self._settle_frame(
                    frame,
                    InteractivePreviewFrameDispositionKind.REPLACED,
                    message=(
                        "Queued approximation was replaced by exact Slider "
                        "release before cadence presentation."
                    ),
                )
                self._frames_by_set.pop(frame.set_id, None)
                state = self._batch_targets.get(frame.set_id)
                if state is not None:
                    state.frame = None
            self._pending = None
            self._transition = None
            self._frozen = True
            self._cancel_scheduled_tick()
            # There is no physical custody to keep alive for a late exact
            # terminal: seal the complete queued batch now.  The immutable
            # sealed identity and generation floor continue to reject late
            # frames, while has_current_batch becomes false so the plot can
            # release its gesture-owned range lock on the next base render.
            self._seal_batch()
            return None
        if (
            self._batch is None
            or self._batch != self._issued_batch
            or receipt.batch != self._batch
            or self._retired
            or set(self._batch_targets) != set(self._batch.target_set_ids)
        ):
            return None
        self._presented_receipt = None
        # Anything still queued has not acquired renderer custody.  Settle it
        # as replaced, while retaining the physically presented roster until
        # the exact terminal outcome settles the released batch.
        for frame in tuple(self._unsettled_frames.values()):
            self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.REPLACED,
                message=(
                    "Queued approximation was transferred to exact Slider "
                    "release before cadence presentation."
                ),
            )
            self._frames_by_set.pop(frame.set_id, None)
            state = self._batch_targets.get(frame.set_id)
            if state is not None:
                state.frame = None
        # A roster target that has not emitted a frame has no physical object
        # to settle.  Seal that absence as an explicit empty disposition so a
        # successful exact terminal can close the released batch without
        # waiting for a late approximate callback.
        for set_id, state in self._batch_targets.items():
            if state.disposition is None and state.frame is None:
                state.disposition = (
                    InteractivePreviewFrameDispositionKind.ACCEPTED_EMPTY
                )
        self._pending = None
        self._transition = None
        self._frozen = True
        self._cancel_scheduled_tick()
        self._exact_release_fenced = True
        return receipt

    def allocate_batch_identity(
        self,
        *,
        tab_id: str,
        document_identity: tuple[int, str],
        target_set_ids: tuple[str, ...],
    ) -> InteractivePreviewBatchIdentity:
        """Mint the sole generation and batch identity for a new gesture."""

        tab = str(tab_id or "").strip()
        identity = tuple(document_identity)
        roster = tuple(str(value or "").strip() for value in target_set_ids)
        if (
            not tab
            or len(identity) != 2
            or type(identity[0]) is not int
            or identity[0] < 0
            or str(identity[1] or "").strip() != tab
            or not roster
            or any(not value for value in roster)
            or len(set(roster)) != len(roster)
        ):
            raise ValueError("interactive batch allocation identity is invalid.")
        self._next_generation += 1
        generation = self._next_generation
        batch_id = hashlib.sha256(
            repr((tab, generation, identity, roster)).encode("utf-8")
        ).hexdigest()
        batch = InteractivePreviewBatchIdentity(
            tab_id=tab,
            generation=generation,
            batch_id=batch_id,
            target_set_ids=roster,
        )
        if self._issued_batch is not None:
            self._settle_unresolved(
                InteractivePreviewFrameDispositionKind.REPLACED,
                message="A newer Interactive Preview replaced this queued frame.",
            )
            self._pending = None
            self._transition = None
            self._frozen = True
            self._cancel_scheduled_tick()
            self._seal_batch()
        self._issued_batch = batch
        self._presented_receipt = None
        return batch

    @property
    def cadence_seconds(self) -> float:
        return self._cadence

    def synchronize(self, projection: InteractiveDisplayProjection) -> None:
        """Install an already-visible completed frame without animation."""

        self._require_projection(projection)
        self._cancel_scheduled_tick()
        self._pending = None
        self._transition = None
        self._frozen = False
        self._retired = False
        previous_visible = self._visible
        self._visible = (InteractiveDisplayLayer(projection, 1.0),)
        now = float(self._clock())
        if not self._present(InteractiveDisplayState(
            self._visible,
            settled=True,
            masked_set_ids=self._masked_set_ids(self._visible),
        )):
            self._visible = previous_visible
            return
        self._last_presented_at = now

    def offer_frame(
        self,
        frame: InteractivePreviewFrame,
        *,
        projection: WorkspaceResultProjection,
    ) -> InteractivePreviewFrameDisposition | None:
        """Admit one emitted frame and settle it exactly once.

        Acceptance is reported only once its nonempty projection has reached
        the physical renderer.  Empty projections settle immediately as
        ``accepted-empty``; stale or structurally unrenderable frames settle
        negatively without entering display custody.
        """

        if not isinstance(frame, InteractivePreviewFrame):
            raise TypeError("frame must be an InteractivePreviewFrame.")
        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if frame.tab_id != projection.tab_id:
            return self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.REJECTED_UNRENDERABLE,
                message="Interactive frame belongs to a different plot tab.",
            )
        batch = self._batch_for_frame(frame)
        stale_kind = self._admission_rejection(batch)
        if stale_kind is not None:
            return self._settle_frame(frame, stale_kind)
        if self._batch is None or self._batch != batch:
            self._begin_batch(batch)
        state = self._batch_targets.get(frame.set_id)
        if state is None:
            return self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.REJECTED_STALE,
                message="Interactive target is outside the admitted roster.",
            )
        if state.disposition is not None or state.frame is not None:
            # Duplicate delivery is idempotent.  The first delivery already
            # owns the target's one physical disposition (or is still queued).
            return None
        if state.terminal is not None and state.terminal is not InteractivePreviewTerminalState.SUCCESS:
            return self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.REJECTED_STALE,
                message="Interactive target was already terminal.",
            )
        self._workspace_projection = projection
        try:
            target = WorkspaceResultPresenter.compose_interactive_display(
                frame,
                projection=projection,
            )
        except Exception as exc:
            settled = self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.REJECTED_UNRENDERABLE,
                message=f"Interactive frame projection failed: {exc}",
            )
            self._maybe_seal_batch()
            return settled
        state.frame = frame
        self._frames_by_set[frame.set_id] = frame
        self._unsettled_frames[frame.set_id] = frame
        if not target.curves:
            settled = self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.ACCEPTED_EMPTY,
            )
            self._frames_by_set.pop(frame.set_id, None)
            state.frame = None
            self._maybe_seal_batch()
            return settled
        if self._visible is None:
            baseline = InteractiveDisplayProjection(
                tab_id=projection.tab_id,
                set_id=target.set_id,
                generation=target.generation,
                x_label=projection.x_label,
                y_label=projection.y_label,
                curves=(),
                target_set_ids=batch.target_set_ids,
                batch_id=batch.batch_id,
            )
            self.synchronize(baseline)
            if self._batch != batch:
                # A renderer failure during baseline installation already
                # emitted the frame's negative disposition and sealed custody.
                return None
        self.offer(self._compose_current_projection(projection))
        return None

    def offer(self, projection: InteractiveDisplayProjection) -> None:
        """Retain only the newest approximate target for the next display tick."""

        accepted = self._queue(projection)
        if accepted:
            self._frozen = False

    def _queue(
        self,
        projection: InteractiveDisplayProjection,
    ) -> bool:
        self._require_projection(projection)
        batch = self._batch
        if self._exact_release_fenced:
            return False
        if (
            batch is None
            or projection.tab_id != batch.tab_id
            or projection.generation != batch.generation
            or projection.batch_id != batch.batch_id
            or projection.target_set_ids != batch.target_set_ids
        ):
            return False
        if self._visible is None:
            self.synchronize(projection)
            return True
        self._pending = projection
        self._schedule_next_tick()
        return True

    def tick(self) -> InteractiveDisplayState | None:
        """Advance from the injected clock; safe for QTimer or a fake clock."""

        self._tick_requested = False
        if self._frozen:
            return None
        now = float(self._clock())
        if not np.isfinite(now):
            raise ValueError("display clock must return a finite timestamp.")
        pending = self._pending
        due = self._cadence_due(now)
        if not due:
            # QTimer can wake before the requested deadline.  An early wake is
            # bookkeeping only: rendering the old frame here would move the
            # presentation origin forward and could postpone a pending frame
            # forever under repeated early wakeups.
            if self._pending is not None or self._transition is not None:
                self._schedule_next_tick()
            return None
        previous_visible = self._visible
        previous_transition = self._transition
        retry_projection = pending
        if retry_projection is None and previous_transition is not None:
            retry_projection = previous_transition.target
        if pending is not None:
            projection = pending
            self._pending = None
            # Retarget from the state physically visible at this clock instant.
            # A worker frame may arrive between display ticks; materializing the
            # current transition here keeps the retarget continuous without
            # exposing an intermediate frame as a separate display event.
            source = self._visible
            if source is None:
                source = (InteractiveDisplayLayer(projection, 1.0),)
            elif self._transition is not None:
                advanced = self._materialize(now)
                if advanced is not None:
                    source = advanced.layers
                    self._visible = source
            self._transition = _Transition(
                source=source,
                target=projection,
                started_at=now,
                interpolate=self._can_interpolate(source, projection),
            )
            current = self._materialize(now)
            if current is not None:
                self._visible = current.layers
        else:
            current = self._materialize(now)
            if current is not None:
                self._visible = current.layers

        if current is not None:
            current = InteractiveDisplayState(
                current.layers,
                settled=current.settled,
                masked_set_ids=self._masked_set_ids(current.layers),
            )
            if self._present(current):
                self._visible = current.layers
                self._last_presented_at = now
            else:
                # The panel's call-local physical journal restored the
                # previous roster.  Restore controller truth as well and
                # leave the attempted target queued for a later retry.
                self._visible = previous_visible
                self._transition = None
                if retry_projection is not None:
                    self._pending = retry_projection
                self._frozen = False
                self._schedule_next_tick()
        if self._pending is not None or self._transition is not None:
            self._schedule_next_tick()
        return current

    def stop(self) -> InteractiveDisplayState | None:
        """Freeze the stable frame while retiring cadence and currentness."""

        now = float(self._clock())
        self._settle_unresolved(
            InteractivePreviewFrameDispositionKind.RETIRED,
            message="Interactive frame was retired by Stop.",
        )
        self._pending = None
        self._transition = None
        self._frozen = True
        self._retired = True
        self._cancel_scheduled_tick()
        self._seal_batch()
        if self._visible is None:
            return None
        frozen = InteractiveDisplayState(
            self._visible,
            settled=True,
            masked_set_ids=self._masked_set_ids(self._visible),
        )
        self._present(frozen, settle_frames=False)
        self._last_presented_at = now
        return frozen

    def clear(self, *, reset_currentness: bool = False) -> None:
        """Retire controller custody and remove every approximate resource."""

        self._settle_unresolved(
            InteractivePreviewFrameDispositionKind.RETIRED,
            message="Interactive display custody was cleared.",
        )
        self._pending = None
        self._transition = None
        self._visible = None
        self._last_presented_at = None
        self._frozen = True
        self._retired = True
        self._seal_batch()
        self._cancel_scheduled_tick()
        self._present(InteractiveDisplayState((), settled=True), settle_frames=False)
        self._workspace_projection = None
        if reset_currentness:
            self._issued_batch = None
            self._next_generation = 0
            self._generation_floor = None
            self._sealed_batch = None
            self._sealed_target_dispositions.clear()
            self._frozen = False
            self._retired = False

    def apply_terminal_outcome(
        self,
        outcome: InteractivePreviewTerminalOutcome,
        *,
        projection: WorkspaceResultProjection,
    ) -> bool:
        """Settle one target's exact terminal fact against current custody."""

        if not isinstance(outcome, InteractivePreviewTerminalOutcome):
            raise TypeError(
                "outcome must be InteractivePreviewTerminalOutcome."
            )
        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        batch = self._batch
        if (
            batch is None
            or self._retired
            or outcome.batch != batch
            or projection.tab_id != batch.tab_id
        ):
            return False
        state = self._batch_targets.get(outcome.target_set_id)
        if state is None or state.terminal is not None:
            return False
        if not outcome.settles_display:
            return True
        target = outcome.target_set_id
        state.terminal = outcome.state
        if outcome.state is InteractivePreviewTerminalState.SUCCESS:
            self._maybe_seal_batch()
            return True
        if outcome.state is InteractivePreviewTerminalState.STOPPED:
            self.stop()
            return True
        frame = self._unsettled_frames.get(target)
        if frame is not None:
            self._settle_frame(
                frame,
                InteractivePreviewFrameDispositionKind.RETIRED,
                message=(outcome.message or "Interactive target settled before display."),
            )
        self._frames_by_set.pop(target, None)
        state.frame = None
        self._workspace_projection = projection
        self._reconcile_current_frames(projection)
        self._maybe_seal_batch()
        return True

    def prepare_workspace_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        """Clear old-semantics resources before exact/base publication.

        Successful exact targets transfer out of approximate custody here,
        before the panel is allowed to publish their exact projection.  Active
        sibling frames remain raw controller custody and are reprojected by
        :meth:`commit_workspace_projection` after the base render succeeds.
        A stopped overlay is retired rather than combined with new semantics.
        """

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        if self._batch is None and self._visible is None:
            self._workspace_projection = projection
            return
        if (self._frozen and not self._exact_release_fenced) or self._retired:
            self.retire(clear_visible=True)
            self._workspace_projection = projection
            return
        for set_id, state in tuple(self._batch_targets.items()):
            if state.terminal is not InteractivePreviewTerminalState.SUCCESS:
                continue
            frame = self._unsettled_frames.get(set_id)
            if frame is not None:
                self._settle_frame(
                    frame,
                    InteractivePreviewFrameDispositionKind.RETIRED,
                    message="Queued approximation retired before exact publication.",
                )
            self._frames_by_set.pop(set_id, None)
            state.frame = None
        self._pending = None
        self._transition = None
        self._visible = None
        self._cancel_scheduled_tick()
        self._present(InteractiveDisplayState((), settled=True), settle_frames=False)
        self._workspace_projection = projection
        self._maybe_seal_batch()

    def commit_workspace_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        """Reproject retained active frames only after the base scene exists."""

        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be a WorkspaceResultProjection.")
        self._workspace_projection = projection
        if self._batch is None or self._retired or not self._frames_by_set:
            return
        target = self._compose_current_projection(projection)
        self.synchronize(target)

    def retire(self, *, clear_visible: bool = False) -> bool:
        """Retire cadence/currentness, optionally removing the stable overlay."""

        changed = bool(
            self._batch is not None
            or self._pending is not None
            or self._transition is not None
            or self._visible is not None
        )
        self._settle_unresolved(
            InteractivePreviewFrameDispositionKind.RETIRED,
            message="Interactive frame was retired before presentation.",
        )
        self._pending = None
        self._transition = None
        self._frozen = True
        self._retired = True
        self._cancel_scheduled_tick()
        self._seal_batch()
        if clear_visible:
            self._visible = None
            self._last_presented_at = None
            self._present(
                InteractiveDisplayState((), settled=True),
                settle_frames=False,
            )
        elif self._visible is not None:
            self._present(
                InteractiveDisplayState(
                    self._visible,
                    settled=True,
                    masked_set_ids=self._masked_set_ids(self._visible),
                ),
                settle_frames=False,
            )
        return changed

    def retire_targets(self, target_set_ids: tuple[str, ...]) -> bool:
        """Retire removed targets without letting the panel own the roster."""

        batch = self._batch
        if batch is None or self._retired:
            return False
        targets = {
            str(value).strip()
            for value in target_set_ids
            if str(value).strip()
        }
        affected = tuple(
            set_id for set_id in batch.target_set_ids if set_id in targets
        )
        if not affected:
            return False
        for set_id in affected:
            state = self._batch_targets[set_id]
            frame = self._unsettled_frames.get(set_id)
            if frame is not None:
                self._settle_frame(
                    frame,
                    InteractivePreviewFrameDispositionKind.RETIRED,
                    message="Interactive target was removed before presentation.",
                )
            self._frames_by_set.pop(set_id, None)
            state.frame = None
            state.terminal = InteractivePreviewTerminalState.OBSOLETE
        projection = self._workspace_projection
        if projection is None or not self._frames_by_set:
            self._visible = None
            self._pending = None
            self._transition = None
            self._present(InteractiveDisplayState((), settled=True), settle_frames=False)
            self._maybe_seal_batch()
            return True
        self._reconcile_current_frames(projection)
        return True

    @staticmethod
    def reject_unrenderable(
        frame: InteractivePreviewFrame,
        *,
        message: str,
    ) -> InteractivePreviewFrameDisposition:
        """Build the negative no-surface disposition for MainWindow routing."""

        if not isinstance(frame, InteractivePreviewFrame):
            raise TypeError("frame must be an InteractivePreviewFrame.")
        return InteractivePreviewFrameDisposition(
            batch=InteractiveDisplayController._batch_for_frame(frame),
            target_set_id=frame.set_id,
            kind=(
                InteractivePreviewFrameDispositionKind.REJECTED_UNRENDERABLE
            ),
            message=str(message or "Interactive plot is unavailable."),
        )

    def _begin_batch(self, batch: InteractivePreviewBatchIdentity) -> None:
        previous = self._batch
        if previous is not None and previous != batch:
            self._settle_unresolved(
                InteractivePreviewFrameDispositionKind.REPLACED,
                message="A newer Interactive Preview replaced this queued frame.",
            )
        self._pending = None
        self._transition = None
        self._frames_by_set.clear()
        self._target_dispositions.clear()
        self._unsettled_frames.clear()
        self._batch_targets = {
            set_id: _BatchTargetState()
            for set_id in batch.target_set_ids
        }
        self._batch = batch
        self._presented_receipt = None
        self._generation_floor = batch.generation
        self._frozen = False
        self._retired = False
        self._exact_release_fenced = False

    def _seal_batch(self) -> None:
        batch = self._batch
        if batch is not None:
            self._generation_floor = batch.generation
            self._sealed_batch = batch
            self._sealed_target_dispositions = dict(self._target_dispositions)
        self._batch = None
        self._frames_by_set.clear()
        self._target_dispositions.clear()
        self._unsettled_frames.clear()
        self._batch_targets.clear()
        self._presented_receipt = None
        self._retired = True
        self._exact_release_fenced = False

    def _admission_rejection(
        self,
        batch: InteractivePreviewBatchIdentity,
    ) -> InteractivePreviewFrameDispositionKind | None:
        issued = self._issued_batch
        if issued is not None and batch != issued:
            return InteractivePreviewFrameDispositionKind.REJECTED_STALE
        current = self._batch
        if current is not None:
            if self._exact_release_fenced and batch == current:
                return InteractivePreviewFrameDispositionKind.REJECTED_STALE
            if batch == current:
                return None
            if batch.generation > current.generation:
                return None
            return InteractivePreviewFrameDispositionKind.REJECTED_STALE
        floor = self._generation_floor
        if floor is None or batch.generation > floor:
            return None
        return InteractivePreviewFrameDispositionKind.REJECTED_STALE

    def _compose_current_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> InteractiveDisplayProjection:
        batch = self._batch
        if batch is None:
            raise RuntimeError("interactive display batch has no current identity.")
        frames = tuple(
            self._frames_by_set[set_id]
            for set_id in batch.target_set_ids
            if set_id in self._frames_by_set
        )
        if not frames:
            raise RuntimeError("interactive display batch has no frame custody.")
        return WorkspaceResultPresenter.compose_interactive_display_batch(
            frames,
            projection=projection,
        )

    def _reconcile_current_frames(
        self,
        projection: WorkspaceResultProjection,
    ) -> None:
        if self._batch is None:
            return
        if self._exact_release_fenced:
            # Exact release owns the physical roster from this point until
            # terminal settlement.  Keep sibling raw custody for the exact
            # outcome, but do not re-render an approximate subset.
            self._pending = None
            self._transition = None
            self._maybe_seal_batch()
            return
        if not self._frames_by_set:
            self._visible = None
            self._present(InteractiveDisplayState((), settled=True), settle_frames=False)
            self._maybe_seal_batch()
            return
        target = self._compose_current_projection(projection)
        self._pending = None
        self._transition = None
        self._visible = (InteractiveDisplayLayer(target, 1.0),)
        self._present(
            InteractiveDisplayState(
                self._visible,
                settled=True,
                masked_set_ids=self._masked_set_ids(self._visible),
            )
        )

    def _maybe_seal_batch(self) -> None:
        """Seal only after every roster target has terminal physical truth.

        Render-map emptiness is intentionally irrelevant here.  A target with
        an empty or unrenderable frame remains represented by its compact
        target state, so an unresolved sibling cannot become stale merely
        because no curve was retained for the first target.
        """

        batch = self._batch
        if batch is None or set(self._batch_targets) != set(batch.target_set_ids):
            return
        for state in self._batch_targets.values():
            terminal = state.terminal
            if terminal is None:
                return
            if terminal is InteractivePreviewTerminalState.SUCCESS:
                if state.disposition is None or state.frame is not None:
                    return
            elif state.frame is not None:
                # Negative terminal outcomes may have no emitted frame.  If
                # one was emitted, its one physical disposition is settled by
                # the time its raw frame is removed above.
                return
        self._seal_batch()

    def _settle_unresolved(
        self,
        kind: InteractivePreviewFrameDispositionKind,
        *,
        message: str,
    ) -> None:
        for frame in tuple(self._unsettled_frames.values()):
            self._settle_frame(frame, kind, message=message)

    def _settle_frame(
        self,
        frame: InteractivePreviewFrame,
        kind: InteractivePreviewFrameDispositionKind,
        *,
        message: str = "",
    ) -> InteractivePreviewFrameDisposition | None:
        batch = self._batch_for_frame(frame)
        state = self._batch_targets.get(frame.set_id)
        if (
            state is not None
            and self._batch == batch
            and state.disposition is not None
        ):
            if (
                self._exact_release_fenced
                and state.disposition
                is InteractivePreviewFrameDispositionKind.ACCEPTED_EMPTY
                and state.frame is None
            ):
                # Release sealed the absence of an earlier frame.  A worker
                # callback arriving after that fence is a new physical frame,
                # not a duplicate disposition for the absent frame.
                state.disposition = None
            else:
                return None
        if batch == self._sealed_batch:
            if frame.set_id in self._sealed_target_dispositions:
                return None
            disposition = InteractivePreviewFrameDisposition(
                batch=batch,
                target_set_id=frame.set_id,
                kind=kind,
                message=message,
            )
            self._sealed_target_dispositions[frame.set_id] = kind
            self._emit_disposition(disposition)
            return disposition
        disposition = InteractivePreviewFrameDisposition(
            batch=batch,
            target_set_id=frame.set_id,
            kind=kind,
            message=message,
        )
        if state is not None and self._batch == batch:
            state.disposition = kind
            self._target_dispositions[frame.set_id] = kind
        self._unsettled_frames.pop(frame.set_id, None)
        self._emit_disposition(disposition)
        self._maybe_seal_batch()
        return disposition

    def _present(
        self,
        state: InteractiveDisplayState,
        *,
        settle_frames: bool = True,
    ) -> bool:
        if self._exact_release_fenced and any(
            layer.projection.curves for layer in state.layers
        ):
            # A fenced batch may still need its exact terminal outcomes, but
            # it no longer admits approximate physical presentation.
            return False
        try:
            self._render(state)
        except Exception as exc:
            detail = f"Interactive display render failed: {exc}"
            if self._render_failed is not None:
                try:
                    self._render_failed(detail)
                except Exception:
                    pass
            # The physical renderer owns the rollback journal.  Keep the
            # logical frame and transition custody intact so the next cadence
            # tick can retry the same batch without manufacturing an orphan or
            # consuming a disposition that never physically succeeded.
            return False
        current_sets = self._current_visible_set_ids(state.layers)
        batch = self._batch
        if (
            batch is not None
            and not self._exact_release_fenced
            and current_sets
            and self._presented_receipt is None
        ):
            self._presented_receipt = InteractivePreviewPresentedReceipt.issue(
                batch,
            )
        if settle_frames:
            for set_id in tuple(current_sets):
                frame = self._unsettled_frames.get(set_id)
                if frame is not None:
                    self._settle_frame(
                        frame,
                        InteractivePreviewFrameDispositionKind.ACCEPTED,
                    )
        return True

    def _current_visible_set_ids(
        self,
        layers: tuple[InteractiveDisplayLayer, ...],
    ) -> frozenset[str]:
        batch = self._batch
        if batch is None:
            return frozenset()
        return frozenset(
            str(curve.key[2])
            for layer in layers
            if (
                layer.projection.tab_id == batch.tab_id
                and layer.projection.generation == batch.generation
                and layer.projection.batch_id == batch.batch_id
                and layer.projection.target_set_ids == batch.target_set_ids
            )
            for curve in layer.projection.curves
            if (
                len(curve.key) >= 3
                and curve.key[0] == "simulation"
                and curve.key[1] == "foreground"
                and str(curve.key[2]) in batch.target_set_ids
                and layer.opacity * curve.opacity > _DISPLAY_EPSILON
            )
        )

    @staticmethod
    def _masked_set_ids(
        layers: tuple[InteractiveDisplayLayer, ...],
    ) -> frozenset[str]:
        return frozenset(
            str(curve.key[2])
            for layer in layers
            for curve in layer.projection.curves
            if (
                len(curve.key) >= 3
                and curve.key[0] == "simulation"
                and curve.key[1] == "foreground"
            )
        )

    @staticmethod
    def _batch_for_frame(
        frame: InteractivePreviewFrame,
    ) -> InteractivePreviewBatchIdentity:
        return InteractivePreviewBatchIdentity(
            tab_id=frame.tab_id,
            generation=frame.generation,
            batch_id=frame.batch_id,
            target_set_ids=frame.target_set_ids,
        )

    def _materialize(self, now: float) -> InteractiveDisplayState | None:
        transition = self._transition
        if transition is None:
            if self._visible is None:
                return None
            return InteractiveDisplayState(
                self._visible,
                settled=self._pending is None,
            )
        progress = min(
            1.0,
            max(0.0, (now - transition.started_at) / self._duration),
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        layers: tuple[InteractiveDisplayLayer, ...]
        if transition.interpolate and progress <= _DISPLAY_EPSILON:
            # The interpolation endpoint still owns the source identity.  Do
            # not relabel source-only geometry with the target batch before a
            # positive target contribution reaches the physical surface.
            layers = transition.source
        elif transition.interpolate:
            source = transition.source[0].projection
            projection = self._interpolate_projection(
                source,
                transition.target,
                eased,
            )
            layers = (InteractiveDisplayLayer(projection, 1.0),)
        else:
            layers = tuple(
                InteractiveDisplayLayer(layer.projection, layer.opacity * (1.0 - eased))
                for layer in transition.source
                if layer.opacity * (1.0 - eased) > _DISPLAY_EPSILON
            )
            if eased > _DISPLAY_EPSILON:
                layers = (*layers, InteractiveDisplayLayer(transition.target, eased))
            if not layers:
                layers = (InteractiveDisplayLayer(transition.target, 1.0),)
        settled = progress >= 1.0
        if settled:
            layers = (InteractiveDisplayLayer(transition.target, 1.0),)
            self._transition = None
        return InteractiveDisplayState(layers, settled=settled)

    def _cadence_due(self, now: float) -> bool:
        last = self._last_presented_at
        return last is None or now - last >= self._cadence - 1e-9

    def _schedule_next_tick(self) -> None:
        if self._tick_requested:
            return
        delay = self._cadence
        last = self._last_presented_at
        if last is not None:
            delay = max(0.0, self._cadence - (float(self._clock()) - last))
        self._tick_requested = True
        if self._request_tick is not None:
            self._request_tick(delay)

    def _cancel_scheduled_tick(self) -> None:
        self._tick_requested = False
        if self._cancel_tick is not None:
            self._cancel_tick()

    @staticmethod
    def _require_projection(projection: object) -> None:
        if not isinstance(projection, InteractiveDisplayProjection):
            raise TypeError("target must be an InteractiveDisplayProjection.")

    @staticmethod
    def _can_interpolate(
        source: tuple[InteractiveDisplayLayer, ...],
        target: InteractiveDisplayProjection,
    ) -> bool:
        if len(source) != 1 or source[0].opacity < 1.0 - 1e-9:
            return False
        source_curves = source[0].projection.curves
        return tuple(curve.key for curve in source_curves) == tuple(
            curve.key for curve in target.curves
        ) and all(
            left.markers == right.markers
            and _same_curve_topology(left, right)
            for left, right in zip(source_curves, target.curves, strict=True)
        )

    @classmethod
    def _interpolate_projection(
        cls,
        source: InteractiveDisplayProjection,
        target: InteractiveDisplayProjection,
        progress: float,
    ) -> InteractiveDisplayProjection:
        curves = tuple(
            cls._interpolate_curve(left, right, progress)
            for left, right in zip(source.curves, target.curves, strict=True)
        )
        return InteractiveDisplayProjection(
            tab_id=target.tab_id,
            set_id=target.set_id,
            generation=target.generation,
            x_label=target.x_label,
            y_label=target.y_label,
            curves=curves,
            target_set_ids=target.target_set_ids,
            batch_id=target.batch_id,
        )

    @staticmethod
    def _interpolate_curve(
        source: InteractiveDisplayCurve,
        target: InteractiveDisplayCurve,
        progress: float,
    ) -> InteractiveDisplayCurve:
        count = int(target.x.size)
        if source.x.size == count:
            source_x = source.x
            source_y = source.y
        else:
            source_axis = np.linspace(0.0, 1.0, int(source.x.size))
            target_axis = np.linspace(0.0, 1.0, count)
            source_x = np.interp(target_axis, source_axis, source.x)
            source_y = np.interp(target_axis, source_axis, source.y)
        amount = float(progress)
        return InteractiveDisplayCurve(
            key=target.key,
            label=target.label,
            x=(1.0 - amount) * source_x + amount * target.x,
            y=(1.0 - amount) * source_y + amount * target.y,
            color=target.color,
            dashed=target.dashed,
            markers=target.markers,
            point_size=target.point_size,
            opacity=target.opacity,
            segment_ids=target.segment_ids,
            segment_keys=target.segment_keys,
            discontinuity_ids=target.discontinuity_ids,
            intervention_connectors=target.intervention_connectors,
        )

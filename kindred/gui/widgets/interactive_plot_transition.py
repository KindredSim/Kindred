"""Cadence-limited transient rendering for Slider Preview results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import time

import numpy as np

from kindred.core.result_repository import SimulationResultValue
from kindred.gui.workspace_result_presenter import (
    InteractiveDisplayCurve,
    InteractiveDisplayProjection,
    ResultRenderInput,
    compose_interactive_display,
)


_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class InteractiveDisplayLayer:
    projection: InteractiveDisplayProjection
    opacity: float


@dataclass(frozen=True, slots=True)
class InteractiveDisplayState:
    layers: tuple[InteractiveDisplayLayer, ...]


@dataclass(frozen=True, slots=True)
class _Transition:
    source: tuple[InteractiveDisplayLayer, ...]
    target: InteractiveDisplayProjection
    started_at: float
    interpolate: bool


def _same_topology(
    left: InteractiveDisplayCurve,
    right: InteractiveDisplayCurve,
) -> bool:
    """Return whether two curves can share one physical interpolation item."""

    if left.segment_ids is None or right.segment_ids is None:
        return left.segment_ids is None and right.segment_ids is None
    return bool(
        np.array_equal(left.segment_ids, right.segment_ids)
        and left.segment_keys == right.segment_keys
        and np.array_equal(left.discontinuity_ids, right.discontinuity_ids)
        and left.intervention_connectors == right.intervention_connectors
    )


class InteractiveDisplayController:
    """Own the admitted interactive batch and transient physical state.

    The request roster and generation live here, at the boundary where a
    callback is admitted. Individual results carry no copied roster or
    competing generation fact; the result map is only the retained result
    payload for the one current batch.
    """

    def __init__(
        self,
        *,
        render: Callable[[InteractiveDisplayState], bool],
        request_tick: Callable[[float], None] | None = None,
        cancel_tick: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cadence_seconds: float = 1.0 / 30.0,
        transition_seconds: float = 0.12,
    ) -> None:
        if not callable(render):
            raise TypeError("render must be callable.")
        self._render = render
        self._request_tick = request_tick
        self._cancel_tick_callback = cancel_tick
        self._clock = clock
        self._cadence = float(cadence_seconds)
        self._duration = float(transition_seconds)
        if not np.isfinite(self._cadence) or self._cadence <= 0:
            raise ValueError("cadence_seconds must be positive and finite.")
        if not np.isfinite(self._duration) or self._duration <= 0:
            raise ValueError("transition_seconds must be positive and finite.")

        self._visible: tuple[InteractiveDisplayLayer, ...] | None = None
        self._transition: _Transition | None = None
        self._pending: InteractiveDisplayProjection | None = None
        self._results: dict[str, SimulationResultValue] = {}
        self._generation: int | None = None
        self._target_set_ids: tuple[str, ...] = ()
        self._render_input: ResultRenderInput | None = None
        self._last_presented_at: float | None = None
        self._tick_requested = False
        self._frozen = False
        self._rendered_frame = False

    @property
    def cadence_seconds(self) -> float:
        return self._cadence

    @property
    def has_visible_frame(self) -> bool:
        return bool(
            self._visible
            and any(layer.projection.curves for layer in self._visible)
        )

    @property
    def has_renderable_frame(self) -> bool:
        return self._rendered_frame

    @property
    def has_pending_batch(self) -> bool:
        return bool(self._results or self._pending or self._transition)

    def _accept_input(self, render_input: ResultRenderInput) -> None:
        self._render_input = render_input

    def prepare_result_input(self, render_input: ResultRenderInput) -> None:
        """Release transient custody before a static result scene changes.

        A static input is repository projection authority.  It must never
        inherit an admitted interactive batch, including one that is stopped
        or waiting for a cadence tick.  Keep this fence at the controller
        boundary so callers cannot accidentally render the static scene and
        then recompose the old transient results.
        """

        self.clear(reset_currentness=True)
        self._accept_input(render_input)
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()

    def commit_result_input(self, render_input: ResultRenderInput) -> None:
        """Retain the static input without reviving transient results.

        ``prepare_result_input`` has already retired the prior transient
        batch.  This commit only records the input used by subsequent Slider
        admissions; static publication itself remains the repository-backed
        plot scene.
        """

        self._accept_input(render_input)

    def offer_result(
        self,
        result: SimulationResultValue,
        *,
        set_id: str,
        generation: int,
        target_set_ids: tuple[str, ...],
        render_input: ResultRenderInput,
    ) -> bool:
        """Admit one result into the current request and queue its projection."""

        self._accept_input(render_input)
        set_id = str(set_id)
        generation = int(generation)
        if self._generation is not None and generation < self._generation:
            return False
        if self._frozen and self._generation is not None and generation <= self._generation:
            return False
        roster = tuple(dict.fromkeys(str(value) for value in target_set_ids))
        if set_id not in roster:
            return False

        if self._compose_result(result, set_id=set_id) is None:
            # A result with no foreground curves is not a publication and
            # cannot replace a previously displayed frame.
            return False

        if self._generation is None:
            self._generation = generation
            self._target_set_ids = roster
            self._frozen = False
        elif generation > self._generation:
            # Preserve retained insertion order. Removing a target from the
            # successor roster must never reintroduce its old physical curve.
            allowed = set(roster)
            self._results = {
                target: value
                for target, value in self._results.items()
                if target in allowed
            }
            self._generation = generation
            self._target_set_ids = roster
            self._frozen = False
        elif set_id not in self._target_set_ids:
            return False
        self._results[set_id] = result
        self._recompose()
        return True

    def offer(self, projection: InteractiveDisplayProjection) -> None:
        self._pending = projection
        self._schedule_tick()

    def tick(self) -> InteractiveDisplayState | None:
        self._tick_requested = False
        now = float(self._clock())
        if self._pending is not None and self._cadence_due(now):
            target = self._pending
            self._pending = None
            source = self._visible or ()
            self._transition = _Transition(
                source,
                target,
                now,
                self._can_interpolate(source, target),
            )
        state = self._materialize(now)
        if state is None:
            return None
        if self._present(state):
            self._visible = state.layers
            self._last_presented_at = now
        if self._transition is not None or self._pending is not None:
            self._schedule_tick()
        return state

    def stop(self) -> InteractiveDisplayState | None:
        """Freeze the last physically stable frame without retiring it."""

        self._frozen = True
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()
        if self._visible is None:
            return None
        state = InteractiveDisplayState(self._visible)
        self._present(state)
        return state

    def clear(self, *, reset_currentness: bool = False) -> None:
        """Release transient custody once, then forget the admitted batch."""

        had_physical_state = bool(
            self._visible or self._rendered_frame or self._pending or self._transition
        )
        if had_physical_state:
            if not self._present(InteractiveDisplayState(())):
                return
            self._visible = None
            self._rendered_frame = False
            self._last_presented_at = None
        self._pending = None
        self._transition = None
        self._results.clear()
        self._target_set_ids = ()
        self._frozen = True
        if reset_currentness:
            self._generation = None
        self._cancel_scheduled_tick()

    def retire_targets(
        self,
        target_set_ids: tuple[str, ...],
        *,
        generation: int | None = None,
    ) -> bool:
        """Retire targets only after their physical frame publishes successfully."""

        if generation is not None and self._generation != int(generation):
            return False
        targets = {str(value) for value in target_set_ids}
        if not targets:
            return False

        known = set(self._target_set_ids)
        known.update(self._results)
        for layer in self._visible or ():
            known.update(self._projection_target_ids(layer.projection))
        known.update(self._projection_target_ids(self._pending))
        if self._transition is not None:
            known.update(self._projection_target_ids(self._transition.target))
            for layer in self._transition.source:
                known.update(self._projection_target_ids(layer.projection))
        affected = targets.intersection(known)
        if not affected:
            # An already absent target owns no physical custody. Treat this
            # idempotent retirement as success so a static replacement is not
            # declined merely because no transient batch was admitted.
            return True

        visible = tuple(
            InteractiveDisplayLayer(filtered, layer.opacity)
            for layer in self._visible or ()
            if (
                filtered := self._filter_projection(layer.projection, affected)
            )
            is not None
        )
        # Publish the filtered state before retiring the admitted result facts.
        state = InteractiveDisplayState(visible)
        if not self._present(state):
            return False

        self._visible = visible or None
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()
        self._results = {
            set_id: result
            for set_id, result in self._results.items()
            if set_id not in affected
        }
        self._target_set_ids = tuple(
            set_id for set_id in self._target_set_ids if set_id not in affected
        )
        if self._generation is not None:
            # Local retirement fencing rejects late callbacks from the retired
            # request even when the computation owner has not started a
            # successor request yet.
            self._generation += 1
        self._frozen = not bool(self._results or self._visible)
        self._rendered_frame = bool(self._visible)
        if self._results:
            self._recompose()
        return True

    def _recompose(self) -> None:
        if not self._results:
            return
        projection = self._compose()
        if projection is not None:
            self.offer(projection)

    def _compose(self) -> InteractiveDisplayProjection | None:
        if self._render_input is None or not self._results:
            return None
        curves: list[InteractiveDisplayCurve] = []
        latest: InteractiveDisplayProjection | None = None
        # Dict insertion order is the admitted result order. Filtering a
        # successor roster therefore preserves unaffected physical results.
        for set_id, result in self._results.items():
            if set_id not in self._target_set_ids:
                continue
            projection = self._compose_result(result, set_id=set_id)
            if projection is None:
                continue
            latest = projection
            curves.extend(projection.curves)
        if latest is None or not curves:
            return None
        return replace(latest, curves=tuple(curves))

    def _compose_result(
        self,
        result: SimulationResultValue,
        *,
        set_id: str,
    ) -> InteractiveDisplayProjection | None:
        if self._render_input is None:
            return None
        return compose_interactive_display(
            result,
            tab_id=self._render_input.tab_id,
            set_id=set_id,
            render_input=self._render_input,
        )

    @staticmethod
    def _projection_target_ids(
        projection: InteractiveDisplayProjection | None,
    ) -> set[str]:
        if projection is None:
            return set()
        return {
            str(curve.key[2])
            for curve in projection.curves
            if len(curve.key) >= 3
        }

    @staticmethod
    def _filter_projection(
        projection: InteractiveDisplayProjection,
        removed: set[str],
    ) -> InteractiveDisplayProjection | None:
        curves = tuple(
            curve
            for curve in projection.curves
            if len(curve.key) < 3 or str(curve.key[2]) not in removed
        )
        if not curves:
            return None
        return replace(projection, curves=curves)

    def _present(self, state: InteractiveDisplayState) -> bool:
        published = bool(self._render(state))
        if published:
            self._rendered_frame = bool(
                state.layers and any(layer.projection.curves for layer in state.layers)
            )
        return published

    def _materialize(self, now: float) -> InteractiveDisplayState | None:
        if self._transition is None:
            if self._visible is None:
                return None
            return InteractiveDisplayState(self._visible)
        transition = self._transition
        progress = min(
            1.0,
            max(0.0, (now - transition.started_at) / self._duration),
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        if transition.interpolate:
            projection = self._interpolate_projection(
                transition.source[0].projection,
                transition.target,
                eased,
            )
            layers = (InteractiveDisplayLayer(projection, 1.0),)
        else:
            layers = tuple(
                InteractiveDisplayLayer(
                    item.projection,
                    item.opacity * (1.0 - eased),
                )
                for item in transition.source
                if item.opacity * (1.0 - eased) > _EPSILON
            )
            if eased > _EPSILON:
                layers = (
                    *layers,
                    InteractiveDisplayLayer(transition.target, eased),
                )
            if not layers:
                layers = (InteractiveDisplayLayer(transition.target, 1.0),)
        if progress >= 1.0:
            layers = (InteractiveDisplayLayer(transition.target, 1.0),)
            self._transition = None
        return InteractiveDisplayState(layers)

    def _schedule_tick(self) -> None:
        if self._tick_requested:
            return
        self._tick_requested = True
        if self._request_tick is not None:
            self._request_tick(self._cadence)

    def _cancel_scheduled_tick(self) -> None:
        self._tick_requested = False
        if self._cancel_tick_callback is not None:
            self._cancel_tick_callback()

    def _cadence_due(self, now: float) -> bool:
        return (
            self._last_presented_at is None
            or now - self._last_presented_at >= self._cadence - _EPSILON
        )

    @staticmethod
    def _can_interpolate(
        source: tuple[InteractiveDisplayLayer, ...],
        target: InteractiveDisplayProjection,
    ) -> bool:
        if len(source) != 1 or source[0].opacity < 1.0 - _EPSILON:
            return False
        source_curves = source[0].projection.curves
        target_curves = target.curves
        if tuple(curve.key for curve in source_curves) != tuple(
            curve.key for curve in target_curves
        ):
            return False
        return all(
            left.markers == right.markers
            and left.x.shape == right.x.shape
            and left.y.shape == right.y.shape
            and np.array_equal(left.x, right.x)
            and _same_topology(left, right)
            for left, right in zip(source_curves, target_curves, strict=True)
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
        return replace(target, curves=curves)

    @staticmethod
    def _interpolate_curve(
        left: InteractiveDisplayCurve,
        right: InteractiveDisplayCurve,
        progress: float,
    ) -> InteractiveDisplayCurve:
        return replace(right, y=left.y + (right.y - left.y) * progress)

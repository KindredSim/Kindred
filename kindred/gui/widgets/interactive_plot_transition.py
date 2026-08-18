"""Cadence-limited transient rendering for Slider Preview frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.gui.workspace_result_presenter import (
    InteractiveDisplayCurve,
    InteractiveDisplayProjection,
    ResultDataBlock,
    ResultRenderInput,
    compose_interactive_display,
    compose_interactive_display_batch,
)


_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class InteractiveDisplayLayer:
    projection: InteractiveDisplayProjection
    opacity: float


@dataclass(frozen=True, slots=True)
class InteractiveDisplayState:
    layers: tuple[InteractiveDisplayLayer, ...]
    settled: bool
    masked_set_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Transition:
    source: tuple[InteractiveDisplayLayer, ...]
    target: InteractiveDisplayProjection
    started_at: float
    interpolate: bool


def _same_topology(left: InteractiveDisplayCurve, right: InteractiveDisplayCurve) -> bool:
    if left.segment_ids is None or right.segment_ids is None:
        return left.segment_ids is None and right.segment_ids is None
    return bool(
        np.array_equal(left.segment_ids, right.segment_ids)
        and left.segment_keys == right.segment_keys
        and np.array_equal(left.discontinuity_ids, right.discontinuity_ids)
        and left.intervention_connectors == right.intervention_connectors
    )


class InteractiveDisplayController:
    """Own only physical cadence and transient plot state."""

    def __init__(self, *, render: Callable[[InteractiveDisplayState], None],
                 request_tick: Callable[[float], None] | None = None,
                 cancel_tick: Callable[[], None] | None = None,
                 render_failed: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 cadence_seconds: float = 1.0 / 30.0,
                 transition_seconds: float = 0.12) -> None:
        if not callable(render):
            raise TypeError("render must be callable.")
        self._render = render
        self._request_tick = request_tick
        self._cancel_tick_callback = cancel_tick
        self._render_failed = render_failed
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
        self._frames: dict[str, object] = {}
        self._generation: int | None = None
        self._target_set_ids: tuple[str, ...] = ()
        self._render_input: ResultRenderInput | None = None
        self._last_presented_at: float | None = None
        self._tick_requested = False
        self._frozen = False

    @property
    def cadence_seconds(self) -> float:
        return self._cadence

    @property
    def has_visible_frame(self) -> bool:
        return bool(self._visible and any(layer.projection.curves for layer in self._visible))

    @property
    def has_current_batch(self) -> bool:
        return bool(self._generation is not None and not self._frozen and (self._frames or self._visible))

    @property
    def has_renderable_frame(self) -> bool:
        return bool(self._frames and any(self._compose().curves for _ in (0,)))

    @property
    def masking_intent(self) -> frozenset[str]:
        return frozenset(
            str(curve.key[2])
            for layer in self._visible or ()
            for curve in layer.projection.curves
            if len(curve.key) >= 3 and curve.key[0:2] == ("simulation", "foreground")
        )

    def _accept_input(self, render_input: ResultRenderInput) -> None:
        if not isinstance(render_input, ResultRenderInput):
            raise TypeError("render_input must be a ResultRenderInput.")
        self._render_input = render_input

    def prepare_result_input(self, render_input: ResultRenderInput) -> None:
        self._accept_input(render_input)
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()

    def commit_result_input(self, render_input: ResultRenderInput) -> None:
        self._accept_input(render_input)
        if self._frames:
            self._recompose()

    def refresh_result_input(self, render_input: ResultRenderInput) -> None:
        self.commit_result_input(render_input)

    def offer_frame(self, frame: object, *, render_input: ResultRenderInput) -> bool:
        from kindred.gui.interactive_preview import InteractivePreviewFrame
        if not isinstance(frame, InteractivePreviewFrame):
            raise TypeError("frame must be an InteractivePreviewFrame.")
        self._accept_input(render_input)
        if frame.tab_id != render_input.tab_id:
            return False
        if self._generation is not None and frame.generation < self._generation:
            return False
        if self._generation != frame.generation:
            self._frames.clear()
            self._pending = None
            self._transition = None
            self._generation = frame.generation
            self._target_set_ids = frame.target_set_ids
            self._frozen = False
        self._frames[frame.set_id] = frame
        self._target_set_ids = frame.target_set_ids
        self._recompose()
        return True

    def offer(self, projection: InteractiveDisplayProjection) -> None:
        if not isinstance(projection, InteractiveDisplayProjection):
            raise TypeError("projection must be InteractiveDisplayProjection.")
        self._pending = projection
        self._schedule_tick()

    def synchronize(self, projection: InteractiveDisplayProjection) -> None:
        if not isinstance(projection, InteractiveDisplayProjection):
            raise TypeError("projection must be InteractiveDisplayProjection.")
        self._cancel_scheduled_tick()
        self._transition = None
        self._pending = None
        self._visible = (InteractiveDisplayLayer(projection, 1.0),)
        self._present(InteractiveDisplayState(self._visible, True, self.masking_intent))
        self._last_presented_at = float(self._clock())

    def tick(self) -> InteractiveDisplayState | None:
        self._tick_requested = False
        now = float(self._clock())
        if self._pending is not None and self._cadence_due(now):
            target = self._pending
            self._pending = None
            source = self._visible or ()
            self._transition = _Transition(source, target, now, self._can_interpolate(source, target))
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
        self._frozen = True
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()
        if self._visible is None:
            return None
        state = InteractiveDisplayState(self._visible, True, self.masking_intent)
        self._present(state)
        return state

    def clear(self, *, reset_currentness: bool = False) -> None:
        self._pending = None
        self._transition = None
        self._frames.clear()
        self._visible = None
        self._frozen = bool(reset_currentness)
        if reset_currentness:
            self._generation = None
            self._target_set_ids = ()
        self._cancel_scheduled_tick()
        self._present(InteractiveDisplayState((), True))

    def retire(self, *, clear_visible: bool = False) -> bool:
        changed = bool(self._frames or self._visible or self._pending or self._transition)
        self._pending = None
        self._transition = None
        self._frames.clear()
        self._frozen = True
        self._cancel_scheduled_tick()
        if clear_visible:
            self._visible = None
            self._present(InteractiveDisplayState((), True))
        return changed

    def retire_targets(self, target_set_ids: tuple[str, ...]) -> bool:
        targets = set(target_set_ids)
        affected = set(self._frames).intersection(targets)
        if not affected:
            return False
        for set_id in affected:
            self._frames.pop(set_id, None)
        if self._frames:
            self._recompose()
        else:
            self.retire(clear_visible=False)
        return True

    def _recompose(self) -> None:
        if not self._frames:
            return
        projection = compose_interactive_display_batch(tuple(self._frames.values()), render_input=self._render_input)
        self.offer(projection)

    def _compose(self) -> InteractiveDisplayProjection:
        if self._render_input is None or not self._frames:
            raise RuntimeError("interactive display has no current frames")
        return compose_interactive_display_batch(tuple(self._frames.values()), render_input=self._render_input)

    def _present(self, state: InteractiveDisplayState) -> bool:
        try:
            self._render(state)
        except Exception as exc:  # presentation failure stays panel-local
            if self._render_failed is not None:
                self._render_failed(str(exc))
            return False
        return True

    def _materialize(self, now: float) -> InteractiveDisplayState | None:
        if self._transition is None:
            if self._visible is None:
                return None
            return InteractiveDisplayState(self._visible, self._pending is None, self.masking_intent)
        transition = self._transition
        progress = min(1.0, max(0.0, (now - transition.started_at) / self._duration))
        eased = progress * progress * (3.0 - 2.0 * progress)
        if transition.interpolate:
            projection = self._interpolate_projection(transition.source[0].projection, transition.target, eased)
            layers = (InteractiveDisplayLayer(projection, 1.0),)
        else:
            layers = tuple(InteractiveDisplayLayer(item.projection, item.opacity * (1.0 - eased)) for item in transition.source if item.opacity * (1.0 - eased) > _EPSILON)
            layers = (*layers, InteractiveDisplayLayer(transition.target, eased)) if eased > _EPSILON else layers
            if not layers:
                layers = (InteractiveDisplayLayer(transition.target, 1.0),)
        if progress >= 1.0:
            layers = (InteractiveDisplayLayer(transition.target, 1.0),)
            self._transition = None
        return InteractiveDisplayState(layers, self._transition is None, self._masked(layers))

    def _reconcile_current_frames(self) -> None:
        if self._frames:
            self._recompose()

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
        return self._last_presented_at is None or now - self._last_presented_at >= self._cadence - _EPSILON

    @staticmethod
    def _masked(layers: tuple[InteractiveDisplayLayer, ...]) -> frozenset[str]:
        return frozenset(str(curve.key[2]) for layer in layers for curve in layer.projection.curves if len(curve.key) >= 3 and curve.key[0:2] == ("simulation", "foreground"))

    @staticmethod
    def _can_interpolate(source: tuple[InteractiveDisplayLayer, ...], target: InteractiveDisplayProjection) -> bool:
        if len(source) != 1 or source[0].opacity < 1.0 - _EPSILON:
            return False
        return tuple(curve.key for curve in source[0].projection.curves) == tuple(curve.key for curve in target.curves) and all(left.markers == right.markers and _same_topology(left, right) for left, right in zip(source[0].projection.curves, target.curves, strict=True))

    @classmethod
    def _interpolate_projection(cls, source: InteractiveDisplayProjection, target: InteractiveDisplayProjection, progress: float) -> InteractiveDisplayProjection:
        curves = tuple(cls._interpolate_curve(left, right, progress) for left, right in zip(source.curves, target.curves, strict=True))
        return InteractiveDisplayProjection(target.tab_id, target.set_id, target.generation, target.x_label, target.y_label, curves, target.target_set_ids)

    @staticmethod
    def _interpolate_curve(left: InteractiveDisplayCurve, right: InteractiveDisplayCurve, progress: float) -> InteractiveDisplayCurve:
        return InteractiveDisplayCurve(key=right.key, label=right.label, x=right.x, y=left.y + (right.y - left.y) * progress, color=right.color, dashed=right.dashed, markers=right.markers, point_size=right.point_size, opacity=right.opacity, segment_ids=right.segment_ids, segment_keys=right.segment_keys, discontinuity_ids=right.discontinuity_ids, intervention_connectors=right.intervention_connectors)

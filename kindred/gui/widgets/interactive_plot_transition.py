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

    def __init__(self, *, render: Callable[[InteractiveDisplayState], bool],
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
        self._results: dict[
            str,
            tuple[SimulationResultValue, int, tuple[str, ...]],
        ] = {}
        self._generation: int | None = None
        self._target_set_ids: tuple[str, ...] = ()
        self._render_input: ResultRenderInput | None = None
        self._last_presented_at: float | None = None
        self._tick_requested = False
        self._frozen = False
        self._rendered_frame = False
        self._published_generation: int | None = None

    @property
    def cadence_seconds(self) -> float:
        return self._cadence

    @property
    def has_visible_frame(self) -> bool:
        return bool(self._visible and any(layer.projection.curves for layer in self._visible))

    @property
    def has_current_batch(self) -> bool:
        return bool(
            self._rendered_frame
            and self._generation is not None
            and self._published_generation == self._generation
            and not self._frozen
        )

    @property
    def has_renderable_frame(self) -> bool:
        return self._rendered_frame

    @property
    def has_pending_batch(self) -> bool:
        return bool(self._results or self._pending or self._transition)

    @property
    def masking_intent(self) -> frozenset[str]:
        return frozenset(
            str(curve.key[2])
            for layer in self._visible or ()
            for curve in layer.projection.curves
            if len(curve.key) >= 3 and curve.key[0:2] == ("simulation", "foreground")
        )

    def _accept_input(self, render_input: ResultRenderInput) -> None:
        self._render_input = render_input

    def prepare_result_input(self, render_input: ResultRenderInput) -> None:
        self._accept_input(render_input)
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()

    def commit_result_input(self, render_input: ResultRenderInput) -> None:
        self._accept_input(render_input)
        if self._results:
            self._recompose()

    def refresh_result_input(self, render_input: ResultRenderInput) -> None:
        self.commit_result_input(render_input)

    def offer_result(
        self,
        result: SimulationResultValue,
        *,
        set_id: str,
        generation: int,
        target_set_ids: tuple[str, ...],
        render_input: ResultRenderInput,
    ) -> bool:
        self._accept_input(render_input)
        if self._generation is not None and generation < self._generation:
            return False
        result_targets = tuple(target_set_ids)
        result_target_set = set(result_targets)
        if self._generation is None:
            self._generation = generation
            self._target_set_ids = result_targets
            self._frozen = False
        elif self._generation != generation:
            previous_targets = set(self._target_set_ids)
            removed = previous_targets.difference(result_target_set)
            if removed:
                self._retire_projection_targets(removed)
            # A successor may reuse unaffected frames from the prior request.
            # Only targets absent from the successor roster are discarded.
            self._results = {
                set_id: value
                for set_id, value in self._results.items()
                if set_id in result_target_set
            }
            self._generation = generation
            self._target_set_ids = result_targets
            self._frozen = False
        elif set_id not in set(self._target_set_ids):
            # A same-generation callback cannot expand the admitted roster.
            return False
        self._results[set_id] = (result, generation, result_targets)
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
        self._results.clear()
        self._target_set_ids = ()
        self._visible = None
        self._frozen = True
        if reset_currentness:
            self._generation = None
        self._cancel_scheduled_tick()
        self._present(InteractiveDisplayState((), True))

    def retire_targets(
        self,
        target_set_ids: tuple[str, ...],
        *,
        generation: int | None = None,
    ) -> bool:
        """Retire targets only when the physical frame owns that generation."""

        if generation is not None and self._published_generation != int(generation):
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
            known.update(
                self._projection_target_ids(self._transition.target)
            )
            for layer in self._transition.source:
                known.update(self._projection_target_ids(layer.projection))
        affected = targets.intersection(known)
        if generation is not None:
            physically_visible = set()
            for layer in self._visible or ():
                physically_visible.update(
                    self._projection_target_ids(layer.projection)
                )
            affected.intersection_update(physically_visible)
        if not affected:
            return False
        # Retiring a target is a local currentness fence as well as a
        # projection removal. Any callback from the retired result generation
        # must be rejected even when the execution owner has not yet issued a
        # successor request.
        if self._generation is not None:
            self._generation += 1
        roster = tuple(self._target_set_ids)
        if not roster:
            for _result, _generation, result_targets in self._results.values():
                roster = tuple(result_targets)
                if roster:
                    break
        if not roster:
            projection = self._visible or self._pending
            if projection is not None:
                roster = tuple(projection.target_set_ids)
        self._target_set_ids = tuple(
            set_id for set_id in roster if set_id not in targets
        )
        for set_id in affected:
            self._results.pop(set_id, None)
        self._retire_projection_targets(affected)
        self._pending = None
        self._transition = None
        self._cancel_scheduled_tick()
        if self._visible is None:
            if self._results:
                self._recompose()
            else:
                self._frozen = True
                self._present(InteractiveDisplayState((), True))
            return True
        self._frozen = False
        self._present(
            InteractiveDisplayState(
                self._visible,
                True,
                self.masking_intent,
            )
        )
        if self._results:
            self._recompose()
        if generation is not None and self._rendered_frame:
            self._published_generation = int(generation)
        return True

    def _recompose(self) -> None:
        if not self._results:
            return
        projection = self._compose()
        self.offer(projection)

    def _compose(self) -> InteractiveDisplayProjection:
        if self._render_input is None or not self._results:
            raise RuntimeError("interactive display has no current results")
        results = tuple(
            (set_id, result, generation, target_set_ids)
            for set_id, (result, generation, target_set_ids) in self._results.items()
            if set_id in set(self._target_set_ids)
        )
        if not results:
            raise RuntimeError("interactive display has no current results")
        # Compose each retained result independently so a successor can carry
        # an unaffected result from the prior generation without transporting
        # or rewriting owner generation facts.  Accumulate only the curves
        # needed by the final physical projection.
        latest = compose_interactive_display(
            results[0][1],
            tab_id=self._render_input.tab_id,
            set_id=results[0][0],
            generation=results[0][2],
            target_set_ids=results[0][3],
            render_input=self._render_input,
        )
        curves: list[InteractiveDisplayCurve] = list(latest.curves)
        for set_id, result, generation, target_set_ids in results[1:]:
            latest = compose_interactive_display(
                result,
                tab_id=self._render_input.tab_id,
                set_id=set_id,
                generation=generation,
                target_set_ids=target_set_ids,
                render_input=self._render_input,
            )
            curves.extend(latest.curves)
        current_set_id = next(
            (set_id for set_id, _result, _generation, _targets in results if set_id in self._target_set_ids),
            results[0][0],
        )
        return replace(
            latest,
            set_id=current_set_id,
            generation=self._generation or latest.generation,
            curves=tuple(curves),
            target_set_ids=self._target_set_ids or latest.target_set_ids,
        )

    @staticmethod
    def _projection_target_ids(
        projection: InteractiveDisplayProjection | None,
    ) -> set[str]:
        if projection is None:
            return set()
        return {str(value) for value in projection.target_set_ids}

    @staticmethod
    def _filter_projection(
        projection: InteractiveDisplayProjection | None,
        removed: set[str],
    ) -> InteractiveDisplayProjection | None:
        if projection is None:
            return None
        target_set_ids = tuple(
            set_id
            for set_id in projection.target_set_ids
            if set_id not in removed
        )
        if not target_set_ids:
            return None
        curves = tuple(
            curve
            for curve in projection.curves
            if len(curve.key) < 3 or curve.key[2] not in removed
        )
        set_id = (
            projection.set_id
            if projection.set_id in target_set_ids
            else target_set_ids[0]
        )
        return replace(
            projection,
            set_id=set_id,
            curves=curves,
            target_set_ids=target_set_ids,
        )

    def _retire_projection_targets(self, removed: set[str]) -> None:
        if not removed:
            return
        if self._visible is not None:
            visible = tuple(
                InteractiveDisplayLayer(filtered, layer.opacity)
                for layer in self._visible
                if (
                    filtered := self._filter_projection(
                        layer.projection,
                        removed,
                    )
                ) is not None
            )
            self._visible = visible or None
        self._pending = self._filter_projection(self._pending, removed)
        if self._transition is not None:
            source = tuple(
                InteractiveDisplayLayer(filtered, layer.opacity)
                for layer in self._transition.source
                if (
                    filtered := self._filter_projection(
                        layer.projection,
                        removed,
                    )
                ) is not None
            )
            target = self._filter_projection(
                self._transition.target,
                removed,
            )
            if target is None:
                self._transition = None
            else:
                self._transition = _Transition(
                    source=source,
                    target=target,
                    started_at=self._transition.started_at,
                    interpolate=False,
                )

    def _present(self, state: InteractiveDisplayState) -> bool:
        try:
            published = bool(self._render(state))
        except Exception as exc:  # presentation failure stays panel-local
            if self._render_failed is not None:
                self._render_failed(str(exc))
            return False
        if published:
            has_curves = bool(
                state.layers
                and any(layer.projection.curves for layer in state.layers)
            )
            self._rendered_frame = has_curves
            self._published_generation = self._generation if has_curves else None
        return published

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

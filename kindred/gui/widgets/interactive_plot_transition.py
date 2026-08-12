"""Clock-driven, retargetable transitions for transient plot projections."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kindred.gui.workspace_result_presenter import (
    InteractiveDisplayCurve,
    InteractiveDisplayProjection,
)

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

    if left.segment_ids is None or right.segment_ids is None:
        return left.segment_ids is None and right.segment_ids is None
    return bool(
        np.array_equal(left.segment_ids, right.segment_ids)
        and left.segment_keys == right.segment_keys
        and np.array_equal(left.discontinuity_ids, right.discontinuity_ids)
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
    exact: bool

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("an interactive display state cannot be empty.")
        if any(not isinstance(layer, InteractiveDisplayLayer) for layer in layers):
            raise TypeError("layers must contain InteractiveDisplayLayer values.")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "settled", bool(self.settled))
        object.__setattr__(self, "exact", bool(self.exact))


@dataclass(frozen=True, slots=True)
class _Transition:
    source: tuple[InteractiveDisplayLayer, ...]
    target: InteractiveDisplayProjection
    started_at: float
    exact: bool
    interpolate: bool


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
        request_tick: Callable[[float], None] | None = None,
        cancel_tick: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cadence_seconds: float = 1.0 / 30.0,
        transition_seconds: float = 0.12,
    ) -> None:
        if not callable(render):
            raise TypeError("render must be callable.")
        if request_tick is not None and not callable(request_tick):
            raise TypeError("request_tick must be callable.")
        if cancel_tick is not None and not callable(cancel_tick):
            raise TypeError("cancel_tick must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        cadence = float(cadence_seconds)
        duration = float(transition_seconds)
        if not np.isfinite(cadence) or cadence <= 0.0:
            raise ValueError("cadence_seconds must be positive and finite.")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("transition_seconds must be positive and finite.")
        self._render = render
        self._request_tick = request_tick
        self._cancel_tick = cancel_tick
        self._clock = clock
        self._cadence = cadence
        self._duration = duration
        self._visible: tuple[InteractiveDisplayLayer, ...] | None = None
        self._transition: _Transition | None = None
        self._pending: tuple[InteractiveDisplayProjection, bool] | None = None
        self._last_presented_at: float | None = None
        self._tick_requested = False
        self._frozen = False
        self._accepted_generation: int | None = None
        self._accepted_batch_id: str | None = None

    @property
    def has_visible_frame(self) -> bool:
        return self._visible is not None

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
        self._accepted_generation = projection.generation
        self._accepted_batch_id = projection.batch_id
        self._visible = (InteractiveDisplayLayer(projection, 1.0),)
        now = float(self._clock())
        self._last_presented_at = now
        self._render(InteractiveDisplayState(self._visible, settled=True, exact=False))

    def offer(self, projection: InteractiveDisplayProjection) -> None:
        """Retain only the newest approximate target for the next display tick."""

        accepted = self._queue(projection, exact=False)
        if accepted:
            self._frozen = False

    def settle(self, projection: InteractiveDisplayProjection) -> None:
        """Retarget the currently visible frame toward one exact endpoint."""

        accepted = self._queue(projection, exact=True)
        if accepted:
            self._frozen = False

    def _queue(
        self,
        projection: InteractiveDisplayProjection,
        *,
        exact: bool,
    ) -> bool:
        self._require_projection(projection)
        if not self._accept_currentness(projection):
            return False
        if self._visible is None:
            self.synchronize(projection)
            if exact:
                visible = self._visible
                if visible is None:
                    raise RuntimeError("synchronized display has no visible frame.")
                self._render(
                    InteractiveDisplayState(visible, settled=True, exact=True)
                )
            return True
        self._pending = (projection, bool(exact))
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
        if pending is not None:
            projection, exact = pending
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
                exact=exact,
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
            self._render(current)
            self._last_presented_at = now
        if self._pending is not None or self._transition is not None:
            self._schedule_next_tick()
        return current

    def refresh(self) -> InteractiveDisplayState | None:
        """Repaint the current physical state without consuming a pending target.

        Completed-result projection changes can rebuild the canonical scene while
        an overlay transition is active.  Refreshing re-materializes the state
        against the same clock but deliberately does not move the cadence origin;
        a pending frame therefore cannot be postponed by an early refresh.
        """

        if self._visible is None:
            return None
        now = float(self._clock())
        if not np.isfinite(now):
            raise ValueError("display clock must return a finite timestamp.")
        current = self._materialize(now)
        if current is not None:
            self._visible = current.layers
            self._render(current)
        return current

    def stop(self) -> InteractiveDisplayState | None:
        """Cancel cadence/transition work and freeze the currently visible frame."""

        now = float(self._clock())
        self._pending = None
        self._transition = None
        self._frozen = True
        self._cancel_scheduled_tick()
        if self._visible is None:
            return None
        frozen = InteractiveDisplayState(
            self._visible,
            settled=True,
            exact=False,
        )
        self._render(frozen)
        self._last_presented_at = now
        return frozen

    def clear(self) -> None:
        """Remove controller custody; the physical caller decides scene cleanup."""

        self._pending = None
        self._transition = None
        self._visible = None
        self._last_presented_at = None
        self._frozen = False
        self._accepted_generation = None
        self._accepted_batch_id = None
        self._cancel_scheduled_tick()

    def _materialize(self, now: float) -> InteractiveDisplayState | None:
        transition = self._transition
        if transition is None:
            if self._visible is None:
                return None
            return InteractiveDisplayState(
                self._visible,
                settled=self._pending is None,
                exact=False,
            )
        progress = min(
            1.0,
            max(0.0, (now - transition.started_at) / self._duration),
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        layers: tuple[InteractiveDisplayLayer, ...]
        if transition.interpolate:
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
                if layer.opacity * (1.0 - eased) > 1e-9
            )
            if eased > 1e-9:
                layers = (*layers, InteractiveDisplayLayer(transition.target, eased))
            if not layers:
                layers = (InteractiveDisplayLayer(transition.target, 1.0),)
        settled = progress >= 1.0
        exact = bool(transition.exact and settled)
        if settled:
            layers = (InteractiveDisplayLayer(transition.target, 1.0),)
            self._transition = None
        return InteractiveDisplayState(layers, settled=settled, exact=exact)

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

    def _accept_currentness(self, projection: InteractiveDisplayProjection) -> bool:
        """Reject stale generations before they can become display custody."""

        generation = projection.generation
        accepted = self._accepted_generation
        if accepted is not None:
            if generation < accepted:
                return False
            if (
                generation == accepted
                and projection.batch_id != self._accepted_batch_id
            ):
                return False
        self._accepted_generation = generation
        self._accepted_batch_id = projection.batch_id
        return True

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

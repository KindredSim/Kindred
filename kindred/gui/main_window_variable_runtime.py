"""Tab-owned Slider drafts and their exact/interactive execution intents."""

from __future__ import annotations

from kindred.core.quantity_units import parse_unit

from dataclasses import dataclass, replace
import math
from typing import Iterable

from PySide6 import QtCore

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    MechanismTabDocument,
    canonical_initial_values,
    canonical_species_names,
    reconcile_tab_mechanism_source,
)
from kindred.core.simulation_preparation import SimulationExecutionRequest
from kindred.core.species import validate_concentration
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_source_with_parameter_updates,
)
from kindred.gui.controllers.simulation_job_owner import (
    PreviewCapture,
    PreviewTarget,
    build_explicit_target,
    build_execution_request,
)
from kindred.gui.interactive_preview import (
    InteractivePreviewRequest,
    InteractivePreviewTarget,
    SliderInteractionKind,
)
from kindred.gui.parameter_enumeration import (
    GuiParameterCoordinate,
    admit_gui_parameter_value,
    enumerate_parameter_coordinates,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableView,
    resolve_live_slider_target_set_ids,
)
from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed

__all__ = [
    "TabSliderBinding",
    "ActiveSliderRenderProjection",
    "ParameterSliderProjection",
    "build_parameter_slider_projection",
]


@dataclass(frozen=True, slots=True)
class ParameterSliderProjection:
    coordinates: tuple[GuiParameterCoordinate, ...] = ()

    def values_dict(self) -> dict[str, float]:
        return {item.name: item.value for item in self.coordinates}

    def coordinate(self, name: str) -> GuiParameterCoordinate | None:
        return next((item for item in self.coordinates if item.name == name), None)


@dataclass(frozen=True, slots=True)
class ActiveSliderRenderProjection:
    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    parameter_values: tuple[tuple[str, float], ...]
    targets: tuple[str, ...]
    species_values: tuple[tuple[str, float], ...]
    mixed_species: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PendingGesture:
    kind: str
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class _PreviewPreflight:
    """Admitted Slider Preview inputs, captured at a gesture or checked live."""

    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    targets: tuple[str, ...]


def build_parameter_slider_projection(
    prepared_dsl: PreparedDslRequestSet,
) -> ParameterSliderProjection:
    return ParameterSliderProjection(enumerate_parameter_coordinates(prepared_dsl))


class TabSliderBinding(QtCore.QObject):
    """Own one tab's Slider drafts; the job owner owns computation custody."""

    draftStateChanged = QtCore.Signal(str)
    sliderDraftStateChanged = QtCore.Signal(str, bool, bool)
    targetSetIdsChanged = QtCore.Signal(object)
    errorRaised = QtCore.Signal(str)

    def __init__(
        self,
        *,
        tab_id: str,
        registry,
        table: BatchInitialConditionsTableView,
        variable_sliders,
        species_sliders,
        slider_workspace_owner: MechanismEditorTabbed,
        simulation_jobs,
        preview_points: int,
        preview_delay_ms: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._tab_id, self._registry, self._table = tab_id, registry, table
        self._variable_sliders, self._species_sliders = (
            variable_sliders,
            species_sliders,
        )
        self._slider_workspace_owner, self._simulation_jobs = (
            slider_workspace_owner,
            simulation_jobs,
        )
        self._projection = None
        self._last_targets = ()
        self._exact_preview_capture: PreviewCapture | None = None
        self._rendering = False
        self._disposed = False
        self._interactive_stop_fenced = False
        # Stop on a Preview holds until the next slider gesture.
        self._preview_stopped = False
        self._defer_rejected_slider_restore = False
        self._deferred_restore_pending = False
        self.set_preview_settings(points=preview_points, delay_ms=preview_delay_ms)
        self._throttle_timer = QtCore.QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._on_throttle_timeout)
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        variable_sliders.interactionRequested.connect(self._on_parameter_interaction)
        variable_sliders.editorValidityChanged.connect(self.publish_slider_draft_state)
        species_sliders.interactionRequested.connect(self._on_species_interaction)
        self._disposable_connections = (
            (variable_sliders.interactionRequested, self._on_parameter_interaction),
            (variable_sliders.editorValidityChanged, self.publish_slider_draft_state),
            (species_sliders.interactionRequested, self._on_species_interaction),
        )

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def active_preview_set_ids(self) -> tuple[str, ...]:
        return tuple(self._simulation_jobs.preview_set_ids(self._tab_id))

    def _stop_preview_timers(self) -> None:
        self._throttle_timer.stop()

    def _clear_gesture_state(self, *, stop_preview_timer: bool = True) -> None:
        if stop_preview_timer:
            self._stop_preview_timers()
        self._active_drag = self._gesture_capture = self._latest_admissible_pending = (
            None
        )

    def _cancel_deferred_restore(self) -> None:
        self._deferred_restore_pending = self._defer_rejected_slider_restore = False

    def _queue_rejected_slider_restore(self) -> None:
        if not self._deferred_restore_pending:
            self._deferred_restore_pending = True
            QtCore.QTimer.singleShot(0, self._restore_rejected_slider_action_later)

    @QtCore.Slot()
    def _restore_rejected_slider_action_later(self) -> None:
        if self._deferred_restore_pending:
            self._deferred_restore_pending = False
            if not self._disposed:
                self.render()

    def set_preview_settings(self, *, points: int, delay_ms: int) -> None:
        self._preview_points = points
        self._preview_delay_ms = delay_ms

    @QtCore.Slot(str, float, object)
    def _on_parameter_interaction(
        self, name: str, value: float, interaction: SliderInteractionKind
    ) -> None:
        self._handle_interaction("parameter", name, value, interaction)

    @QtCore.Slot(str, float, object)
    def _on_species_interaction(
        self, name: str, value: float, interaction: SliderInteractionKind
    ) -> None:
        self._handle_interaction("species", name, value, interaction)

    def _handle_interaction(
        self, kind: str, name: str, value: float, interaction: SliderInteractionKind
    ) -> None:
        identity = (str(kind), str(name))
        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            self._interactive_stop_fenced = False
            if self._active_drag != identity:
                (
                    self._active_drag,
                    self._gesture_capture,
                    self._latest_admissible_pending,
                ) = identity, None, None
            self._record_raw_gesture(*identity, float(value))
            return
        if (
            interaction is SliderInteractionKind.EXACT_DRAG_RELEASE
            and self._interactive_stop_fenced
        ):
            return
        capture = self._gesture_capture if self._active_drag == identity else None
        self._active_drag = self._latest_admissible_pending = None
        self._accept_and_submit(
            kind,
            name,
            float(value),
            interaction=interaction,
            defer_rejected_restore=interaction
            in {
                SliderInteractionKind.EXACT_DRAG_RELEASE,
                SliderInteractionKind.EXACT_DIRECT_CHANGE,
            },
            gesture_capture=capture,
        )
        self._clear_gesture_state()

    def _record_raw_gesture(self, kind: str, name: str, value: float) -> None:
        if not math.isfinite(value) or (kind == "species" and value < 0):
            return
        if self._gesture_capture is None:
            prepared = self._preflight_preview_action(self.resolve_target_set_ids())
            if prepared is None:
                return
            self._gesture_capture = _PreviewPreflight(
                prepared.tab,
                prepared.projection,
                prepared.targets,
            )
        pending = _PendingGesture(
            str(kind),
            str(name),
            float(value),
        )
        if pending == self._latest_admissible_pending:
            return
        self._latest_admissible_pending = pending
        if not self._throttle_timer.isActive():
            # Updating an active timer's interval would postpone dispatch.
            self._throttle_timer.start(self._preview_delay_ms)

    def _accept_and_submit(
        self,
        kind: str,
        name: str,
        value: float,
        *,
        interaction: SliderInteractionKind,
        defer_rejected_restore: bool,
        gesture_capture: _PreviewPreflight | None,
    ) -> bool:
        self._preview_stopped = False
        self._cancel_deferred_restore()
        self._defer_rejected_slider_restore = defer_rejected_restore
        try:
            accepted = (
                self.stage_parameter(
                    name,
                    value,
                    _interaction=interaction,
                    _gesture_capture=gesture_capture,
                )
                if kind == "parameter"
                else self.stage_species(
                    name,
                    value,
                    _interaction=interaction,
                    _gesture_capture=gesture_capture,
                )
            )
        finally:
            self._defer_rejected_slider_restore = False
        if accepted and interaction is SliderInteractionKind.EXACT_TYPED_COMMIT:
            sliders = (
                self._variable_sliders if kind == "parameter" else self._species_sliders
            )
            sliders.remember_editor_state(name)
        if not accepted and defer_rejected_restore:
            self._queue_rejected_slider_restore()
        return bool(accepted)

    def resolve_target_set_ids(self) -> tuple[str, ...]:
        tab = self._registry.tab(self._tab_id)
        return resolve_live_slider_target_set_ids(
            table=self._table,
            pinned_set_ids=self._registry.presentation(
                self._tab_id
            ).pinned_slider_set_ids,
            existing_set_ids=(item.set_id for item in tab.initial_condition_sets),
        )

    def has_slider_draft(self) -> bool:
        runtime = self._registry.tab_runtime(self._tab_id)
        return bool(
            runtime.slider_parameter_draft or runtime.slider_species_drafts_by_set_id
        )

    def publish_slider_draft_state(self) -> None:
        dirty = self.has_slider_draft()
        self.sliderDraftStateChanged.emit(
            self._tab_id, dirty,
            dirty or self._variable_sliders.has_rejected_editor_text(),
        )

    def prepare_render(self) -> ActiveSliderRenderProjection:
        tab, runtime = (
            self._registry.tab(self._tab_id),
            self._registry.tab_runtime(self._tab_id),
        )
        projection = (
            ParameterSliderProjection()
            if tab.prepared_dsl is None
            else build_parameter_slider_projection(tab.prepared_dsl)
        )
        if self._active_drag and self._active_drag[0] == "parameter":
            name = self._active_drag[1]
            old = None if self._projection is None else self._projection.coordinate(name)
            new = projection.coordinate(name)
            if old is None or new is None or parse_unit(old.unit) != parse_unit(new.unit):
                self._active_drag = None
        parameters = projection.values_dict()
        parameters.update(runtime.slider_parameter_draft)
        if self._active_drag and self._active_drag[0] == "parameter":
            live = self._variable_sliders.get_variables().get(self._active_drag[1])
            if live is not None:
                parameters[self._active_drag[1]] = float(live)
        targets = self.resolve_target_set_ids()
        species, mixed = self._project_species_targets(tab, targets)
        if self._active_drag and self._active_drag[0] == "species":
            live = self._species_sliders.get_current_values().get(self._active_drag[1])
            if live is not None:
                species[self._active_drag[1]] = float(live)
        return ActiveSliderRenderProjection(
            tab,
            projection,
            tuple(parameters.items()),
            targets,
            tuple(species.items()),
            mixed,
        )

    def apply_render_projection(
        self, prepared: ActiveSliderRenderProjection, *, preserve_editor_drafts: bool = True
    ) -> None:
        self._rendering = True
        vb, sb = (
            QtCore.QSignalBlocker(self._variable_sliders),
            QtCore.QSignalBlocker(self._species_sliders),
        )
        try:
            self._variable_sliders.render_values(
                dict(prepared.parameter_values),
                coordinates=prepared.projection.coordinates,
                preserve_editor_drafts=preserve_editor_drafts,
            )
            self._species_sliders.render_values(
                dict(prepared.species_values),
                target_set_ids=prepared.targets,
                mixed_species=prepared.mixed_species,
                preserve_editor_drafts=preserve_editor_drafts,
            )
        finally:
            del sb
            del vb
            self._rendering = False
        self._projection = prepared.projection
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def settle_render_projection(self, prepared: ActiveSliderRenderProjection) -> None:
        self._reconcile_target_change(prepared.tab, prepared.targets)
        capture = self._exact_preview_capture
        if capture is not None:
            captured_targets = tuple(target.set_id for target in capture.targets)
            if (
                captured_targets == prepared.targets
                and self._simulation_jobs.owns_preview_capture(capture)
            ):
                return
            self._exact_preview_capture = None
        runtime = self._registry.tab_runtime(prepared.tab.tab_id)
        if not self._preview_stopped and (
            runtime.slider_parameter_draft or runtime.slider_species_drafts_by_set_id
        ):
            self._begin_preview_cycle(
                prepared.tab,
                prepared.targets,
                interaction=SliderInteractionKind.EXACT_TYPED_COMMIT,
            )

    def render(self, *, preserve_editor_drafts: bool = True) -> None:
        prepared = self.prepare_render()
        self.apply_render_projection(prepared, preserve_editor_drafts=preserve_editor_drafts)
        self.settle_render_projection(prepared)
        self.publish_slider_draft_state()

    def clear_projection(self) -> None:
        vb, sb = (
            QtCore.QSignalBlocker(self._variable_sliders),
            QtCore.QSignalBlocker(self._species_sliders),
        )
        try:
            self._variable_sliders.clear()
            self._species_sliders.clear()
        finally:
            del sb
            del vb
        self._projection = None
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def _reconcile_target_change(
        self, tab: MechanismTabDocument, targets: tuple[str, ...]
    ) -> bool:
        previous, current = self._last_targets, tuple(targets)
        self._last_targets = current
        if previous != current:
            self.targetSetIdsChanged.emit(current)
        runtime = self._registry.tab_runtime(tab.tab_id)
        for set_id in set(previous) - set(current):
            runtime.slider_species_drafts_by_set_id.pop(set_id, None)
        removed = tuple(item for item in previous if item not in set(current))
        if removed:
            self._simulation_jobs.reconcile_canonical_targets(tab.tab_id, removed)
        return previous != current

    def _project_species_targets(
        self, tab: MechanismTabDocument, targets: tuple[str, ...]
    ) -> tuple[dict[str, float], frozenset[str]]:
        names = tuple(canonical_species_names(tab))
        if not targets:
            return {name: 0.0 for name in names}, frozenset()
        runtime = self._registry.tab_runtime(tab.tab_id)
        rows = []
        for set_id in targets:
            values = dict(canonical_initial_values(tab, set_id))
            values.update(runtime.slider_species_drafts_by_set_id.get(set_id, {}))
            rows.append(values)
        projected, mixed = {}, set()
        for name in names:
            first = float(rows[0].get(name, 0.0))
            projected[name] = first
            if any(float(row.get(name, 0.0)) != first for row in rows[1:]):
                mixed.add(name)
        return projected, frozenset(mixed)

    def _preflight_preview_action(
        self, targets: tuple[str, ...]
    ) -> _PreviewPreflight | None:
        tab, runtime = (
            self._registry.tab(self._tab_id),
            self._registry.tab_runtime(self._tab_id),
        )
        if runtime.reactions_draft is not None or not targets:
            return None
        if not isinstance(tab.prepared_dsl, PreparedDslRequestSet):
            return None
        return _PreviewPreflight(tab, build_parameter_slider_projection(tab.prepared_dsl), tuple(targets))

    def _captured_gesture_transaction(
        self, capture: _PreviewPreflight, targets: tuple[str, ...]
    ) -> _PreviewPreflight | None:
        if (
            capture.tab.tab_id != self._tab_id
            or capture.targets != tuple(targets)
            or self._registry.tab(self._tab_id) is not capture.tab
        ):
            return None
        return capture

    def stage_parameter(
        self,
        name: str,
        value: float,
        *,
        _interaction: SliderInteractionKind = SliderInteractionKind.EXACT_TYPED_COMMIT,
        _gesture_capture: _PreviewPreflight | None = None,
    ) -> bool:
        if self._rendering:
            return False
        targets = self.resolve_target_set_ids()
        prepared = (
            self._preflight_preview_action(targets)
            if _gesture_capture is None
            else self._captured_gesture_transaction(_gesture_capture, targets)
        )
        if prepared is None:
            return self._restore_rejected_slider_action()
        coordinate = prepared.projection.coordinate(str(name))
        if coordinate is None:
            return self._reject_slider_action(f"Parameter {name!r} is not editable.")
        try:
            value_f = admit_gui_parameter_value(
                value,
                role=coordinate.role,
                unit=coordinate.unit,
                allow_exact_zero=coordinate.allow_exact_zero,
                field=f"Parameter {name!r}",
            )
        except (TypeError, ValueError) as exc:
            if _interaction is SliderInteractionKind.EXACT_TYPED_COMMIT:
                accepted_value = self._registry.tab_runtime(
                    self._tab_id
                ).slider_parameter_draft.get(
                    name, prepared.projection.values_dict()[name]
                )
                self._variable_sliders.reject_editor_value(name, accepted_value)
                self.errorRaised.emit(str(exc))
                return False
            return self._reject_slider_action(str(exc))
        runtime = self._registry.tab_runtime(self._tab_id)
        before = dict(runtime.slider_parameter_draft)
        if value_f == prepared.projection.values_dict().get(str(name)):
            runtime.slider_parameter_draft.pop(str(name), None)
        else:
            runtime.slider_parameter_draft[str(name)] = value_f
        draft_unchanged = before == runtime.slider_parameter_draft
        if draft_unchanged:
            if _interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
                self._emit_draft_state()
                return True
            # Exact releases and direct commits still need a fresh exact
            # capture, even when the value was already staged by the drag.
        self._exact_preview_capture = None
        if not self._begin_preview_cycle(
            prepared.tab, targets, interaction=_interaction
        ):
            runtime.slider_parameter_draft.clear()
            runtime.slider_parameter_draft.update(before)
            self._emit_draft_state()
            return self._restore_rejected_slider_action()
        self._emit_draft_state()
        return True

    def stage_species(
        self,
        species: str,
        value: float,
        *,
        _interaction: SliderInteractionKind = SliderInteractionKind.EXACT_TYPED_COMMIT,
        _gesture_capture: _PreviewPreflight | None = None,
    ) -> bool:
        if self._rendering:
            return False
        try:
            value_f = validate_concentration(
                value, field=f"Initial value for {species!r}"
            )
        except (TypeError, ValueError) as exc:
            return self._reject_slider_action(str(exc))
        targets = self.resolve_target_set_ids()
        prepared = (
            self._preflight_preview_action(targets)
            if _gesture_capture is None
            else self._captured_gesture_transaction(_gesture_capture, targets)
        )
        if prepared is None or str(species) not in canonical_species_names(
            prepared.tab
        ):
            return self._reject_slider_action(
                f"Species {species!r} is absent or unavailable."
            )
        runtime = self._registry.tab_runtime(self._tab_id)
        drafts = runtime.slider_species_drafts_by_set_id
        before = {item: dict(drafts.get(item, {})) for item in targets}
        for set_id in targets:
            canonical = float(
                prepared.tab.initial_condition_set(set_id)
                .values_dict()
                .get(str(species), 0.0)
            )
            if value_f == canonical:
                item = drafts.get(set_id)
                if item is not None:
                    item.pop(str(species), None)
                    (drafts.pop(set_id, None) if not item else None)
            else:
                drafts.setdefault(set_id, {})[str(species)] = value_f
        draft_unchanged = before == {
            item: dict(drafts.get(item, {})) for item in targets
        }
        if draft_unchanged:
            if _interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
                self._emit_draft_state()
                return True
            # Exact releases and direct commits still need a fresh exact
            # capture, even when the value was already staged by the drag.
        self._exact_preview_capture = None
        if not self._begin_preview_cycle(
            prepared.tab, targets, interaction=_interaction
        ):
            for set_id, snapshot in before.items():
                if snapshot:
                    drafts[set_id] = snapshot
                else:
                    drafts.pop(set_id, None)
            self._emit_draft_state()
            return self._restore_rejected_slider_action()
        self._emit_draft_state()
        return True

    def _construct_preview_requests(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> tuple[tuple[str, str, SimulationExecutionRequest], ...]:
        runtime = self._registry.tab_runtime(tab.tab_id)
        output = []
        for set_id in targets:
            settings = tab.resolved_execution_settings(set_id)
            initials = dict(canonical_initial_values(tab, set_id))
            initials.update(runtime.slider_species_drafts_by_set_id.get(set_id, {}))
            request = build_execution_request(
                prepared_dsl=tab.prepared_dsl,
                initial_values=initials.items(),
                execution_settings=settings,
                scientific_settings=tab.scientific_settings,
                point_count=self._preview_points,
                parameter_values=runtime.slider_parameter_draft or None,
            )
            output.append((set_id, tab.initial_condition_set(set_id).name, request))
        return tuple(output)

    def capture_exact_preview(
        self,
        targets: Iterable[str],
        *,
        tab: MechanismTabDocument | None = None,
    ) -> PreviewCapture:
        base = self._registry.tab(self._tab_id) if tab is None else tab
        roster = tuple(str(item) for item in targets)
        built = self._construct_preview_requests(base, roster)
        explicit = tuple(
            build_explicit_target(
                prepared_dsl=base.prepared_dsl,
                tab_id=base.tab_id,
                set_id=set_id,
                set_name=set_name,
                initial_values=canonical_initial_values(base, set_id),
                execution_settings=base.resolved_execution_settings(set_id),
                scientific_settings=base.scientific_settings,
            )
            for set_id, set_name, _ in built
        )
        capture = PreviewCapture(
            base.tab_id,
            tuple(
                PreviewTarget(base.tab_id, set_id, set_name, request, canonical)
                for (set_id, set_name, request), canonical in zip(
                    built, explicit, strict=True
                )
            ),
        )
        return capture

    def _canonical_preview_requests_match(
        self,
        capture: PreviewCapture,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> bool:
        """Require the settled Preview's canonical requests to remain current."""

        try:
            current_requests = tuple(
                build_explicit_target(
                    prepared_dsl=tab.prepared_dsl,
                    tab_id=tab.tab_id,
                    set_id=set_id,
                    set_name=tab.initial_condition_set(set_id).name,
                    initial_values=canonical_initial_values(tab, set_id),
                    execution_settings=tab.resolved_execution_settings(set_id),
                    scientific_settings=tab.scientific_settings,
                ).execution_request
                for set_id in targets
            )
        except (TypeError, ValueError, KeyError):
            return False
        return all(
            (target.canonical_run_target.execution_request == current_request)
            for target, current_request in zip(
                capture.targets,
                current_requests,
                strict=True,
            )
        )

    def capture_interactive_preview(
        self,
        targets: Iterable[str],
        *,
        tab: MechanismTabDocument | None = None,
    ) -> InteractivePreviewRequest:
        base = self._registry.tab(self._tab_id) if tab is None else tab
        built = self._construct_preview_requests(
            base, tuple(str(item) for item in targets)
        )
        return InteractivePreviewRequest(
            base.tab_id,
            tuple(
                InteractivePreviewTarget(set_id, set_name, request)
                for set_id, set_name, request in built
            ),
        )

    def continue_interactive_preview(self, request: InteractivePreviewRequest) -> bool:
        self._stop_preview_timers()
        started = self._simulation_jobs.request_interactive_preview(request)
        return bool(started or self._simulation_jobs.interactive_preview_busy)

    def _begin_preview_cycle(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
        *,
        interaction: SliderInteractionKind = SliderInteractionKind.EXACT_TYPED_COMMIT,
    ) -> bool:
        runtime = self._registry.tab_runtime(tab.tab_id)
        has_draft = bool(
            runtime.slider_parameter_draft
            or any(
                runtime.slider_species_drafts_by_set_id.get(item) for item in targets
            )
        )
        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            if not has_draft:
                self._simulation_jobs.reconcile_canonical_targets(
                    tab.tab_id,
                    targets,
                )
                return True
            return self.continue_interactive_preview(self.capture_interactive_preview(targets, tab=tab))
        capture = self.capture_exact_preview(targets, tab=tab)
        self._stop_preview_timers()
        started = self._simulation_jobs.request_preview(capture)
        accepted = bool(
            started
            or self._simulation_jobs.pending_preview_tab_id == capture.tab_id
        )
        self._exact_preview_capture = capture if accepted else None
        return accepted

    @QtCore.Slot()
    def _on_throttle_timeout(self) -> None:
        pending = self._latest_admissible_pending
        if pending is not None:
            self._latest_admissible_pending = None
            self._accept_and_submit(
                pending.kind,
                pending.name,
                pending.value,
                interaction=SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE,
                defer_rejected_restore=False,
                gesture_capture=self._gesture_capture,
            )

    def discard_slider_draft(self, *, render: bool = True) -> bool:
        self._preview_stopped = False
        runtime = self._registry.tab_runtime(self._tab_id)
        changed = bool(
            runtime.slider_parameter_draft or runtime.slider_species_drafts_by_set_id
        )
        self.cancel_pending_preview_intent()
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        if render:
            self.render(preserve_editor_drafts=False)
        self._emit_draft_state()
        return changed

    def cancel_pending_preview_intent(
        self,
        set_ids: tuple[str, ...] | None = None,
    ) -> None:
        self._cancel_deferred_restore()
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        self._simulation_jobs.retire_preview(self._tab_id, set_ids)
        self._exact_preview_capture = None

    def reset_project_session(self) -> None:
        """Clear retained gestures and editors for the replacement project."""

        self._cancel_deferred_restore()
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        self._simulation_jobs.retire_preview(self._tab_id)
        self._exact_preview_capture = None
        self.clear_projection()
        self._last_targets = ()

    def fence_stop(self) -> bool:
        changed = bool(
            self._throttle_timer.isActive()
            or self._active_drag
            or self._latest_admissible_pending
        )
        self._interactive_stop_fenced = True
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        return changed

    def settle_stop(self, *, hold: bool) -> bool:
        changed = self._simulation_jobs.retire_preview(
            self._tab_id, self.resolve_target_set_ids()
        )
        self._exact_preview_capture = None
        if hold:
            self._preview_stopped = True
        return bool(changed)

    def _build_slider_candidate(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ):
        runtime = self._registry.tab_runtime(self._tab_id)
        candidate_document = self._registry.document
        candidate_tab = tab
        if runtime.slider_parameter_draft:
            updates = tuple(
                DslParameterCoordinateUpdate(name, value)
                for name, value in sorted(runtime.slider_parameter_draft.items())
            )
            derived_text = derive_source_with_parameter_updates(
                tab.mechanism_source.reactions_text,
                tab.prepared_dsl,
                updates,
            )
            visible_source = MechanismAuthoringSource.from_parts(
                reactions_text=derived_text,
            )
            reconciliation = reconcile_tab_mechanism_source(
                candidate_document,
                self._tab_id,
                visible_source,
            )
            candidate_document = reconciliation.proposed_document
            candidate_tab = candidate_document.tab(self._tab_id)

        for set_id in targets:
            draft = runtime.slider_species_drafts_by_set_id.get(set_id)
            if not draft:
                continue
            initial_set = candidate_tab.initial_condition_set(set_id)
            values = dict(canonical_initial_values(candidate_tab, set_id))
            values.update(draft)
            candidate_tab = candidate_tab.replace_initial_condition_set(
                initial_set.with_values(values)
            )
        return replace(
            candidate_document,
            tabs=tuple(
                candidate_tab if item.tab_id == self._tab_id else item
                for item in candidate_document.tabs
            ),
        )

    def apply(self) -> bool:
        runtime = self._registry.tab_runtime(self._tab_id)
        targets = self.resolve_target_set_ids()
        if not targets:
            return self._reject("Slider Apply requires at least one target set.")
        if not runtime.slider_parameter_draft and not any(
            runtime.slider_species_drafts_by_set_id.get(item) for item in targets
        ):
            self.publish_slider_draft_state()
            return False
        if self._simulation_jobs.busy:
            return self._reject(
                "Slider Apply requires the exact Preview to settle first."
            )
        capture = self._exact_preview_capture
        if (
            capture is None
            or tuple(target.set_id for target in capture.targets) != targets
        ):
            return self._reject(
                "Slider Apply requires one settled exact Preview for the current draft and target roster."
            )
        current_tab = self._registry.tab(self._tab_id)
        if not self._canonical_preview_requests_match(capture, current_tab, targets):
            return self._reject(
                "Slider Apply requires one settled exact Preview for the current tab science and target roster."
            )
        pairs = self._simulation_jobs.capture_settled_preview(
            self._tab_id,
            targets,
            expected_requests=tuple(
                (target.set_id, target.execution_request)
                for target in capture.targets
            ),
        )
        if pairs is None:
            return self._reject(
                "Slider Apply requires one settled exact Preview for the current draft and target roster."
        )
        candidate = self._build_slider_candidate(current_tab, targets)
        # The settled Preview request is the admitted authority for Apply.
        # Rebuilding an equivalent request from the candidate document would
        # lose the strict object identity checked above and could publish a
        # request other than the one that produced these results.
        canonical_pairs = tuple(pairs)
        # Capture the session token only after the same-tab request, source,
        # and target-roster guards have passed.  Edits confined to unrelated
        # tabs therefore do not invalidate this tab's settled preview.
        self._registry.replace_document(
            candidate,
            expected_token=self._registry.capture_content_token(),
        )
        self._simulation_jobs.promote_preview_to_run(
            self._tab_id,
            canonical_pairs,
        )
        runtime.slider_parameter_draft.clear()
        for set_id in targets:
            runtime.slider_species_drafts_by_set_id.pop(set_id, None)
            runtime.initial_text_by_set_id.pop(set_id, None)
        self._exact_preview_capture = None
        self._emit_draft_state()
        return True

    def reset(self) -> bool:
        self._preview_stopped = False
        runtime = self._registry.tab_runtime(self._tab_id)
        targets = self.resolve_target_set_ids()
        changed = bool(
            runtime.slider_parameter_draft
            or self._variable_sliders.has_rejected_editor_text()
            or any(
                runtime.slider_species_drafts_by_set_id.get(item) for item in targets
            )
        )
        self.cancel_pending_preview_intent(targets)
        runtime.slider_parameter_draft.clear()
        for item in targets:
            runtime.slider_species_drafts_by_set_id.pop(item, None)
        if changed:
            self.render(preserve_editor_drafts=False)
        self._emit_draft_state()
        return changed

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for signal, slot in self._disposable_connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._stop_preview_timers()
        self._clear_gesture_state(stop_preview_timer=False)
        self._cancel_deferred_restore()
        self._exact_preview_capture = None

    def _emit_draft_state(self) -> None:
        self.draftStateChanged.emit(self._tab_id)
        self.publish_slider_draft_state()

    def _restore_rejected_slider_action(self) -> bool:
        if self._defer_rejected_slider_restore:
            self._queue_rejected_slider_restore()
        elif self._active_drag is None:
            self.render()
        return False

    def _reject_slider_action(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return self._restore_rejected_slider_action()

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

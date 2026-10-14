"""Tab-owned Slider drafts and their exact/interactive execution intents."""

from __future__ import annotations

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

__all__ = ["TabSliderBinding"]


@dataclass(frozen=True, slots=True)
class ParameterSliderProjection:
    coordinates: tuple[GuiParameterCoordinate, ...] = ()

    def values_dict(self) -> dict[str, float]:
        return {item.name: item.value for item in self.coordinates}

    def coordinate(self, name: str) -> GuiParameterCoordinate | None:
        return next((item for item in self.coordinates if item.name == name), None)


@dataclass(frozen=True, slots=True)
class _PendingGesture:
    kind: str
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class _PreviewPreflight:
    """Admitted Slider Preview inputs for one staged value."""

    tab: MechanismTabDocument
    projection: ParameterSliderProjection


def build_parameter_slider_projection(
    prepared_dsl: PreparedDslRequestSet,
) -> ParameterSliderProjection:
    return ParameterSliderProjection(enumerate_parameter_coordinates(prepared_dsl))


class TabSliderBinding(QtCore.QObject):
    """Own one tab's Slider drafts; the job owner owns computation custody."""

    draftStateChanged = QtCore.Signal(str)
    targetSetIdsChanged = QtCore.Signal(object)
    errorRaised = QtCore.Signal(str)
    dragHeldChanged = QtCore.Signal()

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
        # The rendered parameter coordinates and the accepted mechanism they
        # were built from.
        self._projection = None
        self._projection_dsl = None
        self._last_targets = ()
        self._exact_preview_capture: PreviewCapture | None = None
        self._disposed = False
        # Stop on a Preview holds until the next slider gesture.
        self._preview_stopped = False
        self.set_preview_settings(points=preview_points, delay_ms=preview_delay_ms)
        self._throttle_timer = QtCore.QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._on_throttle_timeout)
        # The slider being dragged; Stop clears it, so its release starts nothing.
        self._active_drag = None
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

    @property
    def drag_held(self) -> bool:
        return self._active_drag is not None

    def _clear_gesture_state(self) -> None:
        self._stop_preview_timers()
        held = self._active_drag is not None
        self._active_drag = self._latest_admissible_pending = None
        if held:
            self.dragHeldChanged.emit()

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
            if self._active_drag != identity:
                self._active_drag, self._latest_admissible_pending = identity, None
                self.dragHeldChanged.emit()
            self._record_raw_gesture(*identity, float(value))
            return
        if (
            interaction is SliderInteractionKind.EXACT_DRAG_RELEASE
            and self._active_drag != identity
        ):
            return
        self._clear_gesture_state()
        self._accept_and_submit(kind, name, float(value), interaction=interaction)

    def _record_raw_gesture(self, kind: str, name: str, value: float) -> None:
        if not math.isfinite(value) or (kind == "species" and value < 0):
            return
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
    ) -> None:
        stage = self.stage_parameter if kind == "parameter" else self.stage_species
        accepted = stage(name, value, interaction=interaction)
        if accepted:
            # Only an accepted change ends a Stop's hold.
            self._preview_stopped = False
        if accepted and interaction is SliderInteractionKind.EXACT_TYPED_COMMIT:
            sliders = (
                self._variable_sliders if kind == "parameter" else self._species_sliders
            )
            sliders.remember_editor_state(name)
        if not accepted and interaction is not SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            # The widget already shows the refused value; show the accepted one.
            self.render()

    def resolve_target_set_ids(self) -> tuple[str, ...]:
        tab = self._registry.tab(self._tab_id)
        return resolve_live_slider_target_set_ids(
            table=self._table,
            pinned_set_ids=self._registry.presentation(
                self._tab_id
            ).pinned_slider_set_ids,
            existing_set_ids=(item.set_id for item in tab.initial_condition_sets),
        )

    def publish_slider_draft_state(self) -> None:
        dirty = self._registry.tab_runtime(self._tab_id).has_slider_draft
        self._slider_workspace_owner.set_slider_draft_actions_enabled(
            dirty, dirty or self._variable_sliders.has_rejected_editor_text(),
        )

    def render(self, *, preserve_editor_drafts: bool = True) -> None:
        tab, runtime = (
            self._registry.tab(self._tab_id),
            self._registry.tab_runtime(self._tab_id),
        )
        projection = (
            ParameterSliderProjection()
            if tab.prepared_dsl is None
            else build_parameter_slider_projection(tab.prepared_dsl)
        )
        if self._active_drag and self._active_drag[0] == "parameter" and projection != self._projection:
            # New coordinates rebuild the sliders, which ends the press.
            self._clear_gesture_state()
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
        vb, sb = (
            QtCore.QSignalBlocker(self._variable_sliders),
            QtCore.QSignalBlocker(self._species_sliders),
        )
        try:
            self._variable_sliders.render_values(
                parameters,
                coordinates=projection.coordinates,
                preserve_editor_drafts=preserve_editor_drafts,
            )
            self._species_sliders.render_values(
                species,
                target_set_ids=targets,
                mixed_species=mixed,
                preserve_editor_drafts=preserve_editor_drafts,
            )
        finally:
            del sb
            del vb
        self._projection, self._projection_dsl = projection, tab.prepared_dsl
        self._slider_workspace_owner.render_slider_workspace_presentation()
        self._settle_rendered_targets(tab, targets)
        self.publish_slider_draft_state()

    def _settle_rendered_targets(self, tab: MechanismTabDocument, targets: tuple[str, ...]) -> None:
        self._reconcile_target_change(tab, targets)
        capture = self._exact_preview_capture
        if capture is not None:
            captured_targets = tuple(target.set_id for target in capture.targets)
            if (
                captured_targets == targets
                and self._simulation_jobs.owns_preview_capture(capture)
            ):
                return
            self._exact_preview_capture = None
        if (
            not self._preview_stopped
            and self._registry.tab_runtime(tab.tab_id).has_slider_draft
        ):
            self._begin_preview_cycle(
                tab,
                targets,
                interaction=SliderInteractionKind.EXACT_TYPED_COMMIT,
            )

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
        self._projection = self._projection_dsl = None
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def _reconcile_target_change(
        self, tab: MechanismTabDocument, targets: tuple[str, ...]
    ) -> None:
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
        # Previews build from the accepted science; a Reactions draft does
        # not block them.
        tab = self._registry.tab(self._tab_id)
        if not targets or tab.prepared_dsl is None:
            return None
        projection = (
            self._projection if self._projection_dsl is tab.prepared_dsl
            else build_parameter_slider_projection(tab.prepared_dsl)
        )
        return _PreviewPreflight(tab, projection)

    def stage_parameter(
        self,
        name: str,
        value: float,
        *,
        interaction: SliderInteractionKind,
    ) -> bool:
        targets = self.resolve_target_set_ids()
        prepared = self._preflight_preview_action(targets)
        if prepared is None:
            return False
        coordinate = prepared.projection.coordinate(str(name))
        if coordinate is None:
            return self._reject(f"Parameter {name!r} is not editable.")
        try:
            value_f = admit_gui_parameter_value(
                value,
                role=coordinate.role,
                unit=coordinate.unit,
                allow_exact_zero=coordinate.allow_exact_zero,
                field=f"Parameter {name!r}",
            )
        except ValueError as exc:
            if interaction is SliderInteractionKind.EXACT_TYPED_COMMIT:
                accepted_value = self._registry.tab_runtime(
                    self._tab_id
                ).slider_parameter_draft.get(
                    name, prepared.projection.values_dict()[name]
                )
                self._variable_sliders.reject_editor_value(name, accepted_value)
            return self._reject(str(exc))
        runtime = self._registry.tab_runtime(self._tab_id)
        before = dict(runtime.slider_parameter_draft)
        if value_f == prepared.projection.values_dict().get(str(name)):
            runtime.slider_parameter_draft.pop(str(name), None)
        else:
            runtime.slider_parameter_draft[str(name)] = value_f
        draft_unchanged = before == runtime.slider_parameter_draft
        if draft_unchanged:
            if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
                self._emit_draft_state()
                return True
            # Exact releases and direct commits still need a fresh exact
            # capture, even when the value was already staged by the drag.
        self._exact_preview_capture = None
        if not self._begin_preview_cycle(
            prepared.tab, targets, interaction=interaction
        ):
            runtime.slider_parameter_draft.clear()
            runtime.slider_parameter_draft.update(before)
            self._emit_draft_state()
            return False
        self._emit_draft_state()
        return True

    def stage_species(
        self,
        species: str,
        value: float,
        *,
        interaction: SliderInteractionKind,
    ) -> bool:
        try:
            value_f = validate_concentration(
                value, field=f"Initial value for {species!r}"
            )
        except ValueError as exc:
            return self._reject(str(exc))
        targets = self.resolve_target_set_ids()
        prepared = self._preflight_preview_action(targets)
        if prepared is None or str(species) not in canonical_species_names(
            prepared.tab
        ):
            return self._reject(
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
            if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
                self._emit_draft_state()
                return True
            # Exact releases and direct commits still need a fresh exact
            # capture, even when the value was already staged by the drag.
        self._exact_preview_capture = None
        if not self._begin_preview_cycle(
            prepared.tab, targets, interaction=interaction
        ):
            for set_id, snapshot in before.items():
                if snapshot:
                    drafts[set_id] = snapshot
                else:
                    drafts.pop(set_id, None)
            self._emit_draft_state()
            return False
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
        tab: MechanismTabDocument,
    ) -> PreviewCapture:
        roster = tuple(str(item) for item in targets)
        built = self._construct_preview_requests(tab, roster)
        explicit = tuple(
            build_explicit_target(
                prepared_dsl=tab.prepared_dsl,
                set_id=set_id,
                set_name=set_name,
                initial_values=canonical_initial_values(tab, set_id),
                execution_settings=tab.resolved_execution_settings(set_id),
                scientific_settings=tab.scientific_settings,
            )
            for set_id, set_name, _ in built
        )
        capture = PreviewCapture(
            tab.tab_id,
            tuple(
                PreviewTarget(set_id, set_name, request, canonical)
                for (set_id, set_name, request), canonical in zip(
                    built, explicit, strict=True
                )
            ),
        )
        return capture

    def capture_interactive_preview(
        self,
        targets: Iterable[str],
        *,
        tab: MechanismTabDocument,
    ) -> InteractivePreviewRequest:
        built = self._construct_preview_requests(
            tab, tuple(str(item) for item in targets)
        )
        return InteractivePreviewRequest(
            tab.tab_id,
            tuple(
                InteractivePreviewTarget(set_id, set_name, request)
                for set_id, set_name, request in built
            ),
        )

    def _begin_preview_cycle(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
        *,
        interaction: SliderInteractionKind,
    ) -> bool:
        self._stop_preview_timers()
        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            return self._simulation_jobs.request_interactive_preview(
                self.capture_interactive_preview(targets, tab=tab)
            )
        if not self._registry.tab_runtime(tab.tab_id).has_slider_draft:
            # Released, clicked or typed back at the accepted values.
            self._simulation_jobs.reconcile_canonical_targets(tab.tab_id, targets)
            return True
        capture = self.capture_exact_preview(targets, tab=tab)
        accepted = self._simulation_jobs.request_preview(capture)
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
            )

    def discard_slider_draft(self) -> None:
        """Reset or Discard the tab's Slider drafts and editor text, showing canonical values."""

        self._preview_stopped = False
        runtime = self._registry.tab_runtime(self._tab_id)
        self.cancel_pending_preview_intent()
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        # Replacing the editors' text also clears rejected parameter text,
        # which alone enables Reset.
        self.render(preserve_editor_drafts=False)
        self._emit_draft_state()

    def cancel_pending_preview_intent(self) -> None:
        self._clear_gesture_state()
        self._simulation_jobs.retire_preview(self._tab_id)
        self._exact_preview_capture = None

    def reset_project_session(self) -> None:
        """Clear retained gestures and editors for the replacement project."""

        self.cancel_pending_preview_intent()
        self.clear_projection()
        self._last_targets = ()

    def fence_stop(self) -> None:
        self._clear_gesture_state()

    def settle_stop(self, *, hold: bool) -> None:
        self._simulation_jobs.retire_preview(
            self._tab_id, self.resolve_target_set_ids()
        )
        self._exact_preview_capture = None
        if hold:
            self._preview_stopped = True

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
        if not runtime.has_slider_draft:
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
        # A document change affecting these sets clears their Preview results,
        # so a settled capture is also current for the tab's science.
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
        candidate = self._build_slider_candidate(self._registry.tab(self._tab_id), targets)
        # The settled Preview request is the admitted authority for Apply.
        # Rebuilding an equivalent request from the candidate document would
        # lose the strict object identity checked above and could publish a
        # request other than the one that produced these results.
        canonical_pairs = tuple(pairs)
        # Capture the session token only after the same-tab request and
        # target-roster guards have passed.  Edits confined to unrelated
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

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for signal, slot in self._disposable_connections:
            signal.disconnect(slot)
        self._clear_gesture_state()
        self._exact_preview_capture = None

    def _emit_draft_state(self) -> None:
        self.draftStateChanged.emit(self._tab_id)
        self.publish_slider_draft_state()

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

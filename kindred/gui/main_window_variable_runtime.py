"""Tab-bound parameter/species Slider drafts and exact Preview intent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable

from PySide6 import QtCore

from kindred.core.project_document import (
    MechanismSourceReconciliation,
    MechanismTabDocument,
    canonical_initial_values,
    canonical_species_names,
)
from kindred.core.project_session import ProjectContentToken
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_simulation_execution_request,
)
from kindred.core.species import validate_concentration
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_source_with_parameter_updates,
)
from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport
from kindred.gui.parameter_enumeration import (
    GuiParameterCoordinate,
    admit_gui_parameter_value,
    enumerate_parameter_coordinates,
)
from kindred.gui.interactive_preview import (
    InteractivePreviewRequest,
    InteractivePreviewSampling,
    InteractivePreviewTarget,
    SliderInteractionKind,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
    resolve_live_slider_target_set_ids,
)
from kindred.gui.workflow_contracts import (
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionRequest,
)
from kindred.gui.controllers.simulation_job_owner import PreviewTarget, build_preview_target
from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed

logger = logging.getLogger(__name__)

__all__ = [
    "TabSliderBinding",
    "ActiveSliderRenderProjection",
    "ParameterSliderProjection",
    "build_parameter_slider_projection",
]


@dataclass(frozen=True, slots=True)
class ParameterSliderProjection:
    values: tuple[tuple[str, float], ...]
    metadata: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    coordinates: tuple[GuiParameterCoordinate, ...] = ()

    def values_dict(self) -> dict[str, float]:
        return dict(self.values)

    def metadata_dict(self) -> dict[str, dict[str, object]]:
        return {name: dict(items) for name, items in self.metadata}

@dataclass(frozen=True, slots=True)
class ActiveSliderRenderProjection:
    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    parameter_values: tuple[tuple[str, float], ...]
    targets: tuple[str, ...]
    target_display_names: tuple[str, ...]
    species_values: tuple[tuple[str, float], ...]
    mixed_species: frozenset[str]
    preserve_visibility: bool


@dataclass(frozen=True, slots=True)
class _PreviewCapture:
    tab_id: str
    content_token: ProjectContentToken
    prepared_dsl: PreparedDslRequestSet
    projection: ParameterSliderProjection
    targets: tuple[tuple[str, str, SimulationExecutionRequest], ...]
    debounce_kind: str


@dataclass(frozen=True, slots=True)
class _PendingGesture:
    kind: str
    name: str
    value: float
    debounce_kind: str


@dataclass(frozen=True, slots=True)
class _GestureCapture:
    """Gesture-invariant accepted facts reusable at slider cadence."""

    document_identity: tuple[int, str]
    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SuspendedPreviewDisposition:
    capture: _PreviewCapture | None
    pending_targets: tuple[PreviewTarget, ...] | None


@dataclass(frozen=True, slots=True)
class _PreviewPreflight:
    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    targets: tuple[str, ...]


def build_parameter_slider_projection(
    prepared_dsl: PreparedDslRequestSet,
    *,
    wegscheider_cyclicity_enabled: bool,
) -> ParameterSliderProjection:
    """Project one accepted typed DSL authority into stable Slider values."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    coordinates = tuple(
        coordinate
        for coordinate in enumerate_parameter_coordinates(
            prepared_dsl,
            wegscheider_cyclicity_enabled=bool(
                wegscheider_cyclicity_enabled
            ),
        )
        if coordinate.writable and not coordinate.derived
    )
    normalized_values = tuple(
        (coordinate.name, float(coordinate.value))
        for coordinate in coordinates
    )
    normalized_metadata = tuple(
        (
            coordinate.name,
            tuple({
                "type": coordinate.family,
                "family": coordinate.family,
                "role": coordinate.role,
                "label": coordinate.display_name,
                "unit": coordinate.unit,
                "editable": coordinate.writable,
                "derived": coordinate.derived,
                "min": coordinate.minimum,
                "max": coordinate.maximum,
                "scale": (
                    "log" if coordinate.log10_allowed else "linear"
                ),
            }.items()),
        )
        for coordinate in coordinates
    )
    return ParameterSliderProjection(
        normalized_values,
        normalized_metadata,
        coordinates,
    )


class TabSliderBinding(QtCore.QObject):
    """Own one tab's Slider drafts; SimulationJobOwner owns Preview execution."""

    draftStateChanged = QtCore.Signal(str, bool)
    sliderDraftStateChanged = QtCore.Signal(str, bool)
    targetSetIdsChanged = QtCore.Signal(object)
    previewDetailedBalanceCorrectionRequired = QtCore.Signal(object)
    previewDetailedBalanceCommitRequested = QtCore.Signal(object)
    previewDetailedBalanceBlockCleared = QtCore.Signal(str)
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
        interactive_preview_owner,
        preview_preflight: Callable[[tuple[str, ...]], object] | None = None,
        parameter_preview_debounce_ms: int = 80,
        equilibrium_preview_debounce_ms: int = 150,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        if not callable(getattr(registry, "tab", None)):
            raise TypeError("registry must provide tab(tab_id).")
        if not isinstance(slider_workspace_owner, MechanismEditorTabbed):
            raise TypeError(
                "slider_workspace_owner must be MechanismEditorTabbed."
            )
        self._tab_id = target
        self._registry = registry
        self._table = table
        self._variable_sliders = variable_sliders
        self._species_sliders = species_sliders
        self._slider_workspace_owner = slider_workspace_owner
        self._simulation_jobs = simulation_jobs
        self._interactive_preview_owner = interactive_preview_owner
        self._preview_preflight = preview_preflight
        self._projection: ParameterSliderProjection | None = None
        self._tab_snapshot: MechanismTabDocument | None = None
        self._last_targets: tuple[str, ...] = ()
        self._preview_capture: _PreviewCapture | None = None
        self._suspended_preview_disposition: (
            _SuspendedPreviewDisposition | None
        ) = None
        self._pending_detailed_balance_request: (
            DetailedBalanceCorrectionRequest | None
        ) = None
        self._rendering = False
        self._parameter_preview_debounce_ms = int(
            parameter_preview_debounce_ms
        )
        self._equilibrium_preview_debounce_ms = int(
            equilibrium_preview_debounce_ms
        )
        # Species Preview timing is fixed and has no Config preference.
        self._species_preview_debounce_ms = 80
        self._throttle_timer = self._make_preview_timer(
            self._parameter_preview_debounce_ms
        )
        self._active_drag: tuple[str, str] | None = None
        self._gesture_capture: _GestureCapture | None = None
        self._latest_admissible_pending: _PendingGesture | None = None
        self._gesture_admitted = False
        self._last_admitted_sample: tuple[str, str, float] | None = None
        self._latest_preview_set_ids: set[str] = set()
        self._interactive_stop_fenced = False
        self._stop_exact_set_ids: tuple[str, ...] = ()
        self._defer_rejected_slider_restore = False
        self._deferred_restore_pending = False
        self._disposed = False
        model = table.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            raise TypeError("table must use BatchInitialConditionsTableModel.")
        selection_model = table.selectionModel()
        variable_sliders.interactionRequested.connect(
            self._on_parameter_interaction
        )
        species_sliders.interactionRequested.connect(
            self._on_species_interaction
        )
        selection_model.currentChanged.connect(self._on_live_current_changed)
        model.snapshotRendered.connect(self._on_table_snapshot_rendered)
        self._disposable_connections = (
            (
                variable_sliders.interactionRequested,
                self._on_parameter_interaction,
            ),
            (
                species_sliders.interactionRequested,
                self._on_species_interaction,
            ),
            (selection_model.currentChanged, self._on_live_current_changed),
            (model.snapshotRendered, self._on_table_snapshot_rendered),
            (self._throttle_timer.timeout, self._on_throttle_timeout),
        )

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def active_preview_set_ids(self) -> tuple[str, ...]:
        """Return the set roster currently admitted as Slider Preview."""

        return tuple(sorted(self._latest_preview_set_ids))

    def _make_preview_timer(self, interval_ms: int) -> QtCore.QTimer:
        interval = int(interval_ms)
        if not 0 <= interval <= 1000:
            raise ValueError("Slider debounce must be between 0 and 1000 ms.")
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(interval)
        timer.timeout.connect(self._on_throttle_timeout)
        return timer

    def set_preview_debounce_intervals(
        self,
        *,
        parameter_ms: int,
        equilibrium_K_ms: int,
    ) -> None:
        """Apply Config-owned values without creating a species preference."""

        parameter = int(parameter_ms)
        equilibrium = int(equilibrium_K_ms)
        if not 0 <= parameter <= 1000 or not 0 <= equilibrium <= 1000:
            raise ValueError("Slider debounce must be between 0 and 1000 ms.")
        self._parameter_preview_debounce_ms = parameter
        self._equilibrium_preview_debounce_ms = equilibrium

    def _stop_preview_timers(self) -> None:
        self._throttle_timer.stop()
        self._sync_slider_transient_custody()

    def _sync_slider_transient_custody(self) -> None:
        """Project local gesture/debounce roles into the sole Stop owner."""

        suspended = self._suspended_preview_disposition
        active = bool(
            not self._disposed
            and (
                self._active_drag is not None
                or self._throttle_timer.isActive()
                or self._latest_admissible_pending is not None
                or self._preview_capture is not None
                or (
                    suspended is not None
                    and (
                        suspended.capture is not None
                        or suspended.pending_targets is not None
                    )
                )
            )
        )
        # Gesture/debounce custody is local to this binding.  Stop derives it
        # from this binding and the InteractivePreviewOwner; the direct
        # computation owner never mirrors transient Slider state.
        del active

    def _clear_gesture_state(
        self,
        *,
        stop_preview_timer: bool = True,
    ) -> None:
        if stop_preview_timer:
            self._stop_preview_timers()
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        self._gesture_admitted = False
        self._last_admitted_sample = None
        self._sync_slider_transient_custody()

    def _start_throttle_if_inactive(self, debounce_kind: str) -> None:
        timer = self._preview_timer(debounce_kind)
        if not timer.isActive():
            timer.start()
        self._sync_slider_transient_custody()

    def _cancel_deferred_restore(self) -> None:
        self._deferred_restore_pending = False
        self._defer_rejected_slider_restore = False

    def _queue_rejected_slider_restore(self) -> None:
        if self._deferred_restore_pending:
            return
        self._deferred_restore_pending = True
        QtCore.QTimer.singleShot(0, self._restore_rejected_slider_action_later)

    @QtCore.Slot()
    def _restore_rejected_slider_action_later(self) -> None:
        if not self._deferred_restore_pending:
            return
        self._deferred_restore_pending = False
        if self._disposed or self._suspended_preview_disposition is not None:
            return
        self.render(preserve_visibility=True)

    def _start_gesture(self, kind: str, name: str) -> None:
        """Record raw widget intent without changing publication authority."""

        # A press is only raw widget intent.  It must not discard an already
        # admitted exact Preview capture/intent or stop its submission timer.
        # The first accepted numeric sample performs the exact admission fence.
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        self._gesture_admitted = False
        self._last_admitted_sample = None
        self._active_drag = (str(kind), str(name))
        self._sync_slider_transient_custody()

    @QtCore.Slot(str, float, object)
    def _on_parameter_interaction(
        self,
        name: str,
        value: float,
        interaction: object,
    ) -> None:
        self._handle_interaction("parameter", name, value, interaction)

    @QtCore.Slot(str, float, object)
    def _on_species_interaction(
        self,
        species: str,
        value: float,
        interaction: object,
    ) -> None:
        self._handle_interaction("species", species, value, interaction)

    def _handle_interaction(
        self,
        kind: str,
        name: str,
        value: float,
        interaction: object,
    ) -> None:
        if not isinstance(interaction, SliderInteractionKind):
            raise TypeError("Slider events require SliderInteractionKind.")
        identity = (str(kind), str(name))
        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            self._interactive_stop_fenced = False
            if self._active_drag != identity:
                self._start_gesture(*identity)
            self._record_raw_gesture(*identity, float(value))
            return

        if (
            interaction is SliderInteractionKind.EXACT_DRAG_RELEASE
            and self._interactive_stop_fenced
        ):
            # The widget may deliver the release that closes the gesture after
            # Stop has already fenced its binding.  It is not a new user action
            # and must not recreate exact Preview or display custody.
            return
        capture = self._gesture_capture if self._active_drag == identity else None
        # Exact release admits the current values directly.  The transient
        # display remains plot-local and is not a scientific prerequisite.
        self._active_drag = None
        self._latest_admissible_pending = None
        accepted = self._accept_and_submit(
            kind,
            name,
            float(value),
            interaction=interaction,
            defer_rejected_restore=(
                interaction is SliderInteractionKind.EXACT_DRAG_RELEASE
            ),
            gesture_capture=capture,
        )
        self._clear_gesture_state(
            stop_preview_timer=(accepted or self._preview_capture is None)
        )

    def _record_raw_gesture(self, kind: str, name: str, value: float) -> None:
        value_f = float(value)
        kind_s = str(kind)
        name_s = str(name)
        debounce_kind = self._gesture_debounce_kind(kind_s, name_s)
        capture = self._gesture_capture
        if capture is None:
            targets = self.resolve_target_set_ids()
            prepared = self._preflight_preview_action(targets)
            if prepared is None:
                return
            if isinstance(prepared, _PreviewPreflight):
                tab, projection, targets = (
                    prepared.tab,
                    prepared.projection,
                    prepared.targets,
                )
            else:
                tab, projection = prepared
            capture = _GestureCapture(
                document_identity=self._registry.document_identity(
                    self._tab_id
                ),
                tab=tab,
                projection=projection,
                targets=targets,
            )
            self._gesture_capture = capture
        if not self._gesture_admitted:
            accepted = self._accept_and_submit(
                kind,
                name,
                value_f,
                interaction=SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE,
                gesture_capture=capture,
            )
            if accepted:
                self._gesture_admitted = True
                self._last_admitted_sample = (kind_s, name_s, value_f)
            return
        if not self._is_physical_gesture_value_admissible(kind_s, value_f):
            return
        pending = _PendingGesture(
            kind=kind_s,
            name=name_s,
            value=value_f,
            debounce_kind=debounce_kind,
        )
        previous = self._latest_admissible_pending
        if previous == pending:
            return
        if (
            previous is None
            and self._last_admitted_sample == (kind_s, name_s, value_f)
            and self._current_preview_generations_are_current(kind_s)
        ):
            return
        self._latest_admissible_pending = pending
        self._arm_gesture_throttle(debounce_kind)

    def _arm_gesture_throttle(self, debounce_kind: str) -> None:
        self._start_throttle_if_inactive(str(debounce_kind))

    def _gesture_debounce_kind(self, kind: str, name: str) -> str:
        if str(kind) == "species":
            return "species"
        projection = self._projection
        if projection is None:
            tab = self._registry.tab(self._tab_id)
            prepared_dsl = tab.prepared_dsl
            if isinstance(prepared_dsl, PreparedDslRequestSet):
                try:
                    projection = build_parameter_slider_projection(
                        prepared_dsl,
                        wegscheider_cyclicity_enabled=bool(
                            tab.scientific_settings.wegscheider_policy
                        ),
                    )
                except (TypeError, ValueError):
                    projection = None
        metadata = (
            projection.metadata_dict().get(str(name), {})
            if projection is not None
            else {}
        )
        return (
            "equilibrium"
            if str(metadata.get("role") or "") == "Keq"
            else "parameter"
        )

    @staticmethod
    def _is_physical_gesture_value_admissible(
        kind: str,
        value: float,
    ) -> bool:
        if not math.isfinite(float(value)):
            return False
        return str(kind) != "species" or float(value) >= 0.0

    def _current_preview_generations_are_current(self, kind: str) -> bool:
        targets = self.resolve_target_set_ids()
        return bool(targets) and set(targets) == self._latest_preview_set_ids

    def _accept_and_submit(
        self,
        kind: str,
        name: str,
        value: float,
        *,
        interaction: SliderInteractionKind,
        defer_rejected_restore: bool = False,
        gesture_capture: _GestureCapture | None = None,
    ) -> bool:
        self._cancel_deferred_restore()
        previous_capture = self._preview_capture
        self._defer_rejected_slider_restore = bool(defer_rejected_restore)
        try:
            accepted = (
                self.stage_parameter(
                    name,
                    value,
                    _interaction=interaction,
                    _gesture_capture=gesture_capture,
                )
                if str(kind) == "parameter"
                else self.stage_species(
                    name,
                    value,
                    _interaction=interaction,
                    _gesture_capture=gesture_capture,
                )
            )
        finally:
            self._defer_rejected_slider_restore = False
        if accepted and self._preview_capture is not previous_capture:
            self._submit_preview()
        elif defer_rejected_restore:
            self._queue_rejected_slider_restore()
        return bool(accepted)

    def _replace_preview_capture(
        self,
        successor: _PreviewCapture | None,
        *,
        message: str,
    ) -> None:
        """Replace the one local exact capture with the newest capture."""

        self._preview_capture = successor
        self._sync_slider_transient_custody()

    def resolve_target_set_ids(self) -> tuple[str, ...]:
        tab = self._registry.tab(self._tab_id)
        return resolve_live_slider_target_set_ids(
            table=self._table,
            pinned_set_ids=self._registry.presentation(
                self._tab_id
            ).pinned_slider_set_ids,
            existing_set_ids=(item.set_id for item in tab.initial_condition_sets),
        )

    def prepare_render(
        self,
        *,
        preserve_visibility: bool = True,
    ) -> ActiveSliderRenderProjection:
        """Prepare one complete parameter/species view without widget mutation."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        projection = (
            ParameterSliderProjection((), ())
            if tab.prepared_dsl is None
            else build_parameter_slider_projection(
                tab.prepared_dsl,
                wegscheider_cyclicity_enabled=(
                    tab.scientific_settings.wegscheider_policy
                ),
            )
        )
        parameter_values = projection.values_dict()
        parameter_values.update(runtime.slider_parameter_draft)
        if self._active_drag is not None:
            drag_kind, drag_name = self._active_drag
            if drag_kind == "parameter":
                live_value = self._variable_sliders.get_variables().get(
                    drag_name
                )
                if live_value is not None:
                    parameter_values[drag_name] = float(live_value)
        targets = self.resolve_target_set_ids()
        species_values, mixed_species = self._project_species_targets(
            tab,
            targets,
        )
        if self._active_drag is not None:
            drag_kind, drag_name = self._active_drag
            if drag_kind == "species":
                live_value = self._species_sliders.get_current_values().get(
                    drag_name
                )
                if live_value is not None:
                    species_values[drag_name] = float(live_value)
        return ActiveSliderRenderProjection(
            tab=tab,
            projection=projection,
            parameter_values=tuple(parameter_values.items()),
            targets=targets,
            target_display_names=tuple(
                tab.initial_condition_set(set_id).name
                for set_id in targets
            ),
            species_values=tuple(species_values.items()),
            mixed_species=mixed_species,
            preserve_visibility=bool(preserve_visibility),
        )

    def apply_render_projection(
        self,
        prepared: ActiveSliderRenderProjection,
    ) -> None:
        """Apply one fully prepared coherent parameter/species projection."""

        if not isinstance(prepared, ActiveSliderRenderProjection):
            raise TypeError(
                "prepared must be ActiveSliderRenderProjection."
            )
        self._rendering = True
        variable_blocker = QtCore.QSignalBlocker(self._variable_sliders)
        species_blocker = QtCore.QSignalBlocker(self._species_sliders)
        try:
            self._variable_sliders.render_values(
                dict(prepared.parameter_values),
                metadata=prepared.projection.metadata_dict(),
                preserve_visibility=prepared.preserve_visibility,
                visibility_scope_signature=tuple(
                    (
                        coordinate.name,
                        coordinate.family,
                        coordinate.role,
                        "log" if coordinate.log10_allowed else "linear",
                    )
                    for coordinate in prepared.projection.coordinates
                ),
            )
            self._species_sliders.render_values(
                dict(prepared.species_values),
                target_set_ids=prepared.targets,
                target_display_names=prepared.target_display_names,
                mixed_species=prepared.mixed_species,
                preserve_visibility=prepared.preserve_visibility,
                visibility_scope_signature=tuple(
                    name for name, _value in prepared.species_values
                ),
            )
        finally:
            del species_blocker
            del variable_blocker
            self._rendering = False
        self._projection = prepared.projection
        self._tab_snapshot = prepared.tab
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def settle_render_projection(
        self,
        prepared: ActiveSliderRenderProjection,
    ) -> None:
        """Publish target/Preview effects after the whole workspace applies."""

        if not isinstance(prepared, ActiveSliderRenderProjection):
            raise TypeError(
                "prepared must be ActiveSliderRenderProjection."
            )
        if self._suspended_preview_disposition is not None:
            return
        tab = prepared.tab
        runtime = self._registry.tab_runtime(tab.tab_id)
        targets = prepared.targets
        targets_changed = self._reconcile_target_change(tab, targets)
        if targets_changed and any(
            runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id.get(set_id)
            for set_id in targets
        ):
            prepared = self._preflight_preview_action(targets)
            if prepared is not None:
                current_tab = prepared.tab
                if current_tab.tab_id == tab.tab_id:
                    self._begin_preview_cycle(
                        (current_tab, prepared.projection),
                        prepared.targets,
                        debounce_kind=(
                            "parameter"
                            if runtime.slider_parameter_draft
                            else "species"
                        ),
                    )

    def render(self, *, preserve_visibility: bool = True) -> bool:
        try:
            prepared = self.prepare_render(
                preserve_visibility=preserve_visibility
            )
        except (TypeError, ValueError) as exc:
            self.errorRaised.emit(
                "Slider projection failed; stale controls were cleared while "
                f"draft and semantic state were retained: {exc}"
            )
            self.clear_projection()
            self.publish_slider_draft_state()
            return False
        try:
            self.apply_render_projection(prepared)
            self.settle_render_projection(prepared)
        except Exception as exc:
            self.errorRaised.emit(
                "Slider render failed; stale controls were cleared while "
                f"draft and semantic state were retained: {exc}"
            )
            self.clear_projection()
            self.publish_slider_draft_state()
            return False
        self.publish_slider_draft_state()
        return True

    def clear_projection(self) -> None:
        """Clear only physical Slider projection state."""
        variable_blocker = QtCore.QSignalBlocker(self._variable_sliders)
        species_blocker = QtCore.QSignalBlocker(self._species_sliders)
        try:
            self._variable_sliders.clear(preserve_visibility=True)
            self._species_sliders.clear(preserve_visibility=True)
        finally:
            del species_blocker
            del variable_blocker
        self._projection = None
        self._tab_snapshot = None
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def _reconcile_target_change(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> bool:
        existing = {item.set_id for item in tab.initial_condition_sets}
        previous = self._last_targets
        current = tuple(targets)
        self._last_targets = current
        if current != previous:
            self.targetSetIdsChanged.emit(current)
        removed = tuple(
            set_id
            for set_id in previous
            if set_id not in current and set_id in existing
        )
        added = tuple(set_id for set_id in current if set_id not in previous)
        runtime = self._registry.tab_runtime(tab.tab_id)
        dirty_removed = tuple(
            set_id
            for set_id in removed
            if runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id.get(set_id)
        )
        clean_removed = tuple(
            set_id for set_id in removed if set_id not in dirty_removed
        )
        dirty_added = tuple(
            set_id
            for set_id in added
            if runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id.get(set_id)
        )
        dirty_transition = tuple(dict.fromkeys((*dirty_removed, *dirty_added)))
        if dirty_transition:
            self._simulation_jobs.deauthorize_targets(
                tab.tab_id,
                dirty_transition,
            )
            for set_id in dirty_transition:
                self._latest_preview_set_ids.discard(set_id)
        if clean_removed:
            self._simulation_jobs.reconcile_canonical_targets(
                tab.tab_id,
                clean_removed,
            )
        return current != previous

    def dispose(self) -> None:
        """Disconnect and forget Slider-owned captures idempotently."""

        if self._disposed:
            return
        self._disposed = True
        for signal, slot in self._disposable_connections:
            try:
                signal.disconnect(slot)
            except Exception:
                logger.exception(
                    "Slider binding signal disconnection failed for tab %s",
                    self._tab_id,
                )
        self._interactive_preview_owner.retire(self._tab_id)
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        self.cancel_pending_preview_intent()
        self._projection = None
        self._tab_snapshot = None
        self._last_targets = ()
        self._latest_preview_set_ids.clear()
        self.clear_preview_detailed_balance_block()

    def evict_preview_targets(self, set_ids: Iterable[str]) -> None:
        """Forget Slider Preview keys and captures for deleted targets."""

        targets = {str(set_id) for set_id in set_ids}
        if not targets:
            return
        for set_id in targets:
            self._latest_preview_set_ids.discard(set_id)
        capture = self._preview_capture
        if capture is None:
            return
        surviving = tuple(
            target
            for target in capture.targets
            if target[0] not in targets
        )
        if surviving:
            self._replace_preview_capture(
                replace(capture, targets=surviving),
                message=(
                    "Exact Slider settlement was retired because one of its "
                    "captured targets was deleted."
                ),
            )
        else:
            self._replace_preview_capture(
                None,
                message=(
                    "Exact Slider settlement was retired because its target "
                    "was deleted."
                ),
            )
            self._stop_preview_timers()

    def _project_species_targets(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> tuple[dict[str, float], frozenset[str]]:
        species_names = tuple(canonical_species_names(tab))
        if not targets:
            return ({name: 0.0 for name in species_names}, frozenset())
        runtime = self._registry.tab_runtime(tab.tab_id)
        target_values: list[dict[str, float]] = []
        for set_id in targets:
            values = dict(canonical_initial_values(tab, set_id))
            values.update(
                runtime.slider_species_drafts_by_set_id.get(set_id, {})
            )
            target_values.append(values)
        projected: dict[str, float] = {}
        mixed: set[str] = set()
        for species in species_names:
            first = float(target_values[0].get(species, 0.0))
            if not math.isfinite(first) or first < 0.0:
                raise ValueError(
                    f"Initial concentration for {species!r} must be finite "
                    "and nonnegative."
                )
            projected[species] = first
            # Mixed-state display authority is a discrete projection choice.
            # Exact binary64 equality defines authored identity.
            if any(
                float(values.get(species, 0.0)) != first
                for values in target_values[1:]
            ):
                mixed.add(species)
        return projected, frozenset(mixed)

    @QtCore.Slot(QtCore.QModelIndex, QtCore.QModelIndex)
    def _on_live_current_changed(
        self,
        _current: QtCore.QModelIndex,
        _previous: QtCore.QModelIndex,
    ) -> None:
        if not self._rendering:
            self.render(preserve_visibility=True)

    @QtCore.Slot()
    def _on_table_snapshot_rendered(self) -> None:
        if not self._rendering:
            self.render(preserve_visibility=True)

    @QtCore.Slot(str, float)
    def stage_parameter(
        self,
        name: str,
        value: float,
        *,
        _interaction: SliderInteractionKind = (
            SliderInteractionKind.EXACT_TYPED_COMMIT
        ),
        _gesture_capture: _GestureCapture | None = None,
    ) -> bool:
        if (
            self._rendering
            or self._suspended_preview_disposition is not None
        ):
            return False
        targets = self.resolve_target_set_ids()
        if not targets:
            return self._reject_slider_action(
                "Parameter edits require at least one valid Slider target."
            )
        prepared = (
            self._preflight_preview_action(targets)
            if _gesture_capture is None
            else self._captured_gesture_transaction(
                _gesture_capture,
                targets,
            )
        )
        if prepared is None:
            return self._restore_rejected_slider_action()
        # Admission returns the exact post-admission target identity.
        if isinstance(prepared, _PreviewPreflight):
            tab, projection, targets = (
                prepared.tab,
                prepared.projection,
                prepared.targets,
            )
        else:
            tab, projection = prepared
        coordinate = next(
            (
                item
                for item in projection.coordinates
                if item.name == str(name)
            ),
            None,
        )
        if coordinate is None:
            return self._reject_slider_action(
                f"Parameter {name!r} is absent."
            )
        if not coordinate.writable or coordinate.derived:
            return self._reject_slider_action(
                f"Parameter {name!r} is derived and cannot be edited."
            )
        try:
            value_f = admit_gui_parameter_value(
                value,
                scientific_category=coordinate.scientific_category,
                allow_exact_zero=coordinate.allow_exact_zero,
                field=f"Parameter {name!r}",
            )
        except (TypeError, ValueError) as exc:
            return self._reject_slider_action(str(exc))
        runtime = self._registry.tab_runtime(tab.tab_id)
        canonical = projection.values_dict()
        previous = dict(runtime.slider_parameter_draft)
        was_parameter_dirty = bool(previous)
        if str(name) in canonical and value_f == canonical[str(name)]:
            runtime.slider_parameter_draft.pop(str(name), None)
        else:
            runtime.slider_parameter_draft[str(name)] = value_f
        if runtime.slider_parameter_draft == previous:
            return self._begin_preview_cycle(
                (tab, projection),
                targets,
                interaction=_interaction,

                debounce_kind=(
                    "equilibrium" if coordinate.role == "Keq"
                    else "parameter"
                ),
            )
        preview_admitted = self._begin_preview_cycle(
            (tab, projection),
            targets,
            interaction=_interaction,

            debounce_kind=(
                "equilibrium" if coordinate.role == "Keq" else "parameter"
            ),
        )
        if not preview_admitted:
            runtime.slider_parameter_draft.clear()
            runtime.slider_parameter_draft.update(previous)
            self._emit_draft_state()
            return self._restore_rejected_slider_action()
        target_ids = set(targets)
        unselected = tuple(
            item.set_id
            for item in tab.initial_condition_sets
            if item.set_id not in target_ids
        )
        if (
            not runtime.slider_parameter_draft
            and was_parameter_dirty
            and unselected
        ):
            clean_unselected = tuple(
                set_id
                for set_id in unselected
                if not runtime.slider_species_drafts_by_set_id.get(set_id)
            )
            dirty_unselected = tuple(
                set_id
                for set_id in unselected
                if runtime.slider_species_drafts_by_set_id.get(set_id)
            )
            if clean_unselected:
                self._reconcile_canonical_set_ids(tab, clean_unselected)
            if dirty_unselected:
                self._simulation_jobs.deauthorize_targets(
                    tab.tab_id,
                    dirty_unselected,
                )
        self._emit_draft_state()
        return True

    def _reconcile_canonical_set_ids(
        self,
        tab: MechanismTabDocument,
        set_ids: tuple[str, ...],
    ) -> None:
        self._simulation_jobs.reconcile_canonical_targets(
            tab.tab_id,
            tuple(dict.fromkeys(set_ids)),
        )

    @QtCore.Slot(str, float)
    def stage_species(
        self,
        species: str,
        value: float,
        *,
        _interaction: SliderInteractionKind = (
            SliderInteractionKind.EXACT_TYPED_COMMIT
        ),
        _gesture_capture: _GestureCapture | None = None,
    ) -> bool:
        if (
            self._rendering
            or self._suspended_preview_disposition is not None
        ):
            return False
        try:
            value_f = validate_concentration(
                value,
                field=f"Initial value for {species!r}",
            )
        except (TypeError, ValueError) as exc:
            return self._reject_slider_action(str(exc))
        targets = self.resolve_target_set_ids()
        if not targets:
            return self._restore_rejected_slider_action()
        prepared = (
            self._preflight_preview_action(targets)
            if _gesture_capture is None
            else self._captured_gesture_transaction(
                _gesture_capture,
                targets,
            )
        )
        if prepared is None:
            return self._restore_rejected_slider_action()
        if isinstance(prepared, _PreviewPreflight):
            tab, projection, targets = (
                prepared.tab,
                prepared.projection,
                prepared.targets,
            )
        else:
            tab, projection = prepared
        species_name = str(species)
        if species_name not in canonical_species_names(tab):
            return self._reject_slider_action(
                f"Species {species_name!r} is absent."
            )
        drafts = self._registry.tab_runtime(
            tab.tab_id
        ).slider_species_drafts_by_set_id
        before = {
            set_id: dict(drafts.get(set_id, {}))
            for set_id in targets
        }
        for set_id in targets:
            initial_set = tab.initial_condition_set(set_id)
            canonical = initial_set.values_dict().get(species_name, 0.0)
            if value_f == float(canonical):
                set_drafts = drafts.get(set_id)
                if set_drafts is not None:
                    set_drafts.pop(species_name, None)
                    if not set_drafts:
                        drafts.pop(set_id, None)
            else:
                drafts.setdefault(set_id, {})[species_name] = value_f
        after = {
            set_id: dict(drafts.get(set_id, {}))
            for set_id in targets
        }
        if after == before:
            return self._begin_preview_cycle(
                (tab, projection),
                targets,
                interaction=_interaction,

                debounce_kind="species",
            )
        preview_admitted = self._begin_preview_cycle(
            (tab, projection),
            targets,
            interaction=_interaction,

            debounce_kind="species",
        )
        if not preview_admitted:
            for set_id, snapshot in before.items():
                if snapshot:
                    drafts[set_id] = dict(snapshot)
                else:
                    drafts.pop(set_id, None)
            self._emit_draft_state()
            return self._restore_rejected_slider_action()
        self._emit_draft_state()
        return True

    def _captured_gesture_transaction(
        self,
        capture: _GestureCapture,
        targets: tuple[str, ...],
    ) -> tuple[MechanismTabDocument, ParameterSliderProjection] | None:
        """Reuse accepted facts only while their whole capture stays current."""

        if not isinstance(capture, _GestureCapture):
            raise TypeError("capture must be a _GestureCapture.")
        if (
            capture.tab.tab_id != self._tab_id
            or capture.targets != tuple(targets)
            or self._registry.document_identity(self._tab_id)
            != capture.document_identity
            or self._registry.tab(self._tab_id) != capture.tab
        ):
            self.errorRaised.emit(
                "Slider gesture stopped because its accepted project content "
                "or target roster changed."
            )
            return None
        if not self._preview_inputs_valid(capture.targets):
            return None
        return capture.tab, capture.projection

    def _preflight_preview_action(
        self,
        targets: tuple[str, ...],
    ) -> _PreviewPreflight | None:
        """Use one canonical source result before any Slider mutation."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        visible_prepared: PreparedDslRequestSet | None = tab.prepared_dsl
        callback_result: object = True
        callback = self._preview_preflight
        admitted_targets = tuple(targets)
        if callback is not None:
            callback_result = callback(targets)
            if isinstance(callback_result, MechanismSourceReconciliation):
                tab = self._registry.tab(self._tab_id)
                visible_prepared = callback_result.prepared_dsl
                if callback_result.materialized_set_ids:
                    admitted_targets = (
                        str(callback_result.materialized_set_ids[0]),
                    )
                else:
                    admitted_targets = self.resolve_target_set_ids()
                if callback_result.detailed_balance.requires_correction:
                    prospective_tab = callback_result.proposed_document.tab(
                        self._tab_id
                    )
                    request = self._preview_detailed_balance_request(
                        prospective_tab,
                        callback_result.detailed_balance.report,
                        prepared_dsl=visible_prepared,
                        reactions_text=(
                            prospective_tab.mechanism_source.reactions_text
                        ),
                    )
                    self._pending_detailed_balance_request = request
                    self.previewDetailedBalanceCorrectionRequired.emit(request)
                    return None
            elif not bool(callback_result):
                return None
        else:
            if runtime.reactions_draft is not None:
                self.errorRaised.emit(
                    "Preview requires visible Reactions to be accepted first."
                )
                return None
            if not self._preview_inputs_valid(targets):
                return None
        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        try:
            if visible_prepared is None:
                raise ValueError(
                    "Preview requires accepted Reactions with retained DSL analysis."
                )
            projection = build_parameter_slider_projection(
                visible_prepared,
                wegscheider_cyclicity_enabled=(
                    tab.scientific_settings.wegscheider_policy
                ),
            )
        except (TypeError, ValueError) as exc:
            self.errorRaised.emit(f"Preview parameter projection failed: {exc}")
            return None
        missing = set(runtime.slider_parameter_draft).difference(
            projection.values_dict()
        )
        if missing:
            self.errorRaised.emit(
                "Preview draft parameter(s) are absent from accepted "
                "Reactions: " + ", ".join(sorted(missing))
            )
            return None
        self._projection = projection
        self._tab_snapshot = tab
        self.clear_preview_detailed_balance_block()
        return _PreviewPreflight(tab, projection, admitted_targets)

    def _begin_preview_cycle(
        self,
        transaction: tuple[
            MechanismTabDocument,
            ParameterSliderProjection,
        ],
        targets: tuple[str, ...],
        *,
        interaction: SliderInteractionKind = (
            SliderInteractionKind.EXACT_TYPED_COMMIT
        ),
        debounce_kind: str,
    ) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        tab, projection = transaction
        runtime = self._registry.tab_runtime(tab.tab_id)
        dirty_targets = tuple(
            set_id
            for set_id in targets
            if runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id.get(set_id)
        )
        canonical_targets = tuple(
            set_id for set_id in targets if set_id not in dirty_targets
        )
        if (
            not dirty_targets
            and interaction is not SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
        ):
            # Release is an exact settlement even when motion returned to the
            # canonical value or matched the last approximate sample.
            dirty_targets = tuple(targets)
            canonical_targets = ()
        if not dirty_targets:
            if canonical_targets:
                self._reconcile_canonical_set_ids(tab, canonical_targets)
            self.cancel_pending_preview_intent()
            return True
        prepared_dsl = tab.prepared_dsl
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            self._reject(
                "Preview requires accepted Reactions with retained DSL analysis."
            )
            return False

        constructed: list[
            tuple[str, str, SimulationExecutionRequest]
        ] = []
        try:
            sampling = (
                InteractivePreviewSampling()
                if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
                else None
            )
            for set_id in dirty_targets:
                parameter_values, initial_values = self._preview_values(
                    tab,
                    projection,
                    set_id,
                )
                execution_settings = tab.resolved_execution_settings(set_id)
                if sampling is not None:
                    execution_settings = replace(
                        execution_settings,
                        integration_points=sampling.lightweight_integration_points(
                            execution_settings.integration_points
                        ),
                    )
                execution_request = build_simulation_execution_request(
                    prepared_dsl=prepared_dsl,
                    mechanism_source=tab.mechanism_source,
                    initial_values=initial_values,
                    execution_settings=execution_settings,
                    scientific_settings=tab.scientific_settings,
                    parameter_values=dict(parameter_values),
                )
                contract = execution_request.intervention_execution
                if contract is None:
                    raise ValueError(
                        "Preview intervention schedule did not bind to one "
                        "exact execution contract."
                    )
                constructed.append(
                    (
                        set_id,
                        tab.initial_condition_set(set_id).name,
                        execution_request,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview construction failed before admission "
                f"({type(exc).__name__}: {exc})."
            )
            return False

        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            # The first admitted motion sample fences every older exact Slider
            # capture/intent. Completed Run/reference truth remains untouched.
            if sampling is None:
                raise RuntimeError(
                    "Interactive Preview construction lost its sampling policy."
            )
            self._stop_preview_timers()
            self._replace_preview_capture(
                None,
                message=(
                    "Exact Slider settlement became obsolete when a newer "
                    "interactive gesture was admitted."
                ),
            )
            self._latest_preview_set_ids = set(dirty_targets)
            request = InteractivePreviewRequest(
                tab_id=tab.tab_id,
                targets=tuple(
                    InteractivePreviewTarget(
                        set_id=set_id,
                        set_name=set_name,
                        prepared_dsl=prepared_dsl,
                        execution_request=execution_request,
                    )
                    for set_id, set_name, execution_request in constructed
                ),
                sampling=sampling,
            )
            self._simulation_jobs.retire_preview(
                tab.tab_id,
                dirty_targets,
            )
            self._interactive_preview_owner.request(request)
            return True

        # Construction above is fallible. Only a complete exact batch reaches
        # the scheduler's single fence/admission transition.
        previous_capture = self._preview_capture
        previous_timer_active = self._throttle_timer.isActive()
        self._stop_preview_timers()
        try:
            self._interactive_preview_owner.retire(
                tab.tab_id,
                dirty_targets,
            )
            self._simulation_jobs.deauthorize_targets(
                tab.tab_id,
                tuple(set_id for set_id, _name, _request in constructed),
                preserve_reference=True,
            )
            admitted_targets = tuple(
                PreviewTarget(
                    tab_id=tab.tab_id,
                    set_id=set_id,
                    set_name=set_name,
                    prepared_dsl=prepared_dsl,
                    execution_request=execution_request,
                )
                for set_id, set_name, execution_request in constructed
            )
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview admission failed "
                f"({type(exc).__name__}: {exc})."
            )
            return False
        if not admitted_targets:
            if previous_capture is not None and previous_timer_active:
                self._start_throttle_if_inactive(
                    previous_capture.debounce_kind
                )
            return False
        if canonical_targets:
            self._reconcile_canonical_set_ids(tab, canonical_targets)
        if runtime.slider_parameter_draft:
            admitted_targets = set(dirty_targets)
            unselected = tuple(
                item.set_id
                for item in tab.initial_condition_sets
                if item.set_id not in admitted_targets
            )
            if unselected:
                # A global parameter Draft becomes publication authority only
                # after its complete Preview request batch is admitted.
                self._simulation_jobs.deauthorize_targets(
                    tab.tab_id,
                    unselected,
                )
        captured = tuple(
            (set_id, set_name, execution_request)
            for set_id, set_name, execution_request in constructed
        )
        self._latest_preview_set_ids = set(dirty_targets)
        self._replace_preview_capture(
            _PreviewCapture(
                tab_id=tab.tab_id,
                content_token=self._registry.capture_content_token(),
                prepared_dsl=prepared_dsl,
                projection=projection,
                targets=captured,
                debounce_kind=str(debounce_kind),
            ),
            message=(
                "Pending exact Slider settlement was replaced by a newer "
                "exact Preview capture."
            ),
        )
        self._start_throttle_if_inactive(debounce_kind)
        return True

    def _preview_timer(self, debounce_kind: str) -> QtCore.QTimer:
        intervals = {
            "parameter": self._parameter_preview_debounce_ms,
            "equilibrium": self._equilibrium_preview_debounce_ms,
            "species": self._species_preview_debounce_ms,
        }
        interval = intervals.get(str(debounce_kind))
        if interval is None:
            raise ValueError(
                f"Unknown Preview debounce kind {debounce_kind!r}."
            )
        self._throttle_timer.setInterval(interval)
        return self._throttle_timer

    @staticmethod
    def _preview_detailed_balance_request(
        tab: MechanismTabDocument,
        report: WegscheiderCyclicityReport,
        *,
        prepared_dsl: PreparedDslRequestSet,
        reactions_text: str,
    ) -> DetailedBalanceCorrectionRequest:
        source = str(reactions_text)
        unresolved = report.unresolved_cycles
        if not unresolved:
            raise ValueError("Preview detailed-balance admission is already resolved.")
        issue = "Unresolved detailed-balance cycle"
        if len(unresolved) != 1:
            issue += "s"
        issue += ": " + "; ".join(
            f"{cycle.cycle_id} ({', '.join(cycle.parameter_names)})"
            for cycle in unresolved
        )
        source_energy_unit = str(prepared_dsl.analysis.energy_unit)
        return DetailedBalanceCorrectionRequest(
            origin="Preview",
            tab_id=tab.tab_id,
            source=source,
            issue=issue,
            temperature_K=float(tab.scientific_settings.temperature),
            report=report,
            accepted_analysis=prepared_dsl.analysis,
            source_energy_unit=str(source_energy_unit),
        )

    def complete_preview_detailed_balance_correction(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> bool:
        """Accept one typed modal outcome without resubmitting Preview."""

        if not isinstance(outcome, DetailedBalanceCorrectionOutcome):
            raise TypeError("outcome must be DetailedBalanceCorrectionOutcome.")
        pending = self._pending_detailed_balance_request
        if pending is None:
            return False
        if (
            outcome.origin,
            outcome.tab_id,
            outcome.source,
            outcome.issue,
        ) != (
            pending.origin,
            pending.tab_id,
            pending.source,
            pending.issue,
        ):
            self.errorRaised.emit(
                "Preview detailed-balance outcome is no longer current."
            )
            return False
        if outcome.cancelled:
            # Cancellation aborts this correction attempt but deliberately
            # retains the blocked request so Fix can be invoked again.
            return True
        corrected = outcome.corrected_source
        if corrected is None:
            raise ValueError("Accepted correction has no corrected source.")
        current_visible_source = (
            self._registry.tab_runtime(self._tab_id).reactions_draft
            or self._registry.tab(self._tab_id).mechanism_source
        )
        if (
            self._tab_id != pending.tab_id
            or current_visible_source.reactions_text
            not in {pending.source, corrected}
        ):
            self.errorRaised.emit(
                "Preview detailed-balance source is no longer current."
            )
            return False
        self._pending_detailed_balance_request = None
        # Integration commits this exact typed outcome. No Preview request is
        # scheduled here or by the editor modal.
        self.previewDetailedBalanceCommitRequested.emit(outcome)
        return True

    def clear_preview_detailed_balance_block(self) -> bool:
        pending = self._pending_detailed_balance_request
        if pending is None:
            return False
        self._pending_detailed_balance_request = None
        self.previewDetailedBalanceBlockCleared.emit(pending.tab_id)
        return True

    def _preview_inputs_valid(self, targets: tuple[str, ...]) -> bool:
        runtime = self._registry.tab_runtime(self._tab_id)
        invalid_targets = tuple(
            set_id
            for set_id in targets
            if runtime.invalid_initial_text_by_set_id.get(set_id)
        )
        if invalid_targets:
            self.errorRaised.emit(
                "Preview blocked by invalid visible Initial Conditions in: "
                + ", ".join(invalid_targets)
            )
            return False
        return True

    @QtCore.Slot()
    def _on_throttle_timeout(self) -> None:
        pending = self._latest_admissible_pending
        if pending is not None:
            self._latest_admissible_pending = None
            accepted = self._accept_and_submit(
                pending.kind,
                pending.name,
                pending.value,
                interaction=SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE,
                gesture_capture=self._gesture_capture,
            )
            if accepted:
                self._gesture_admitted = True
                self._last_admitted_sample = (
                    pending.kind,
                    pending.name,
                    pending.value,
                )
            self._sync_slider_transient_custody()
            return
        self._submit_preview()

    @QtCore.Slot()
    def _submit_preview(self) -> None:
        self._throttle_timer.stop()
        if self._suspended_preview_disposition is not None:
            return
        capture = self._preview_capture
        self._preview_capture = None
        self._sync_slider_transient_custody()
        if capture is None:
            return
        targets = tuple(
            build_preview_target(
                prepared_dsl=capture.prepared_dsl,
                tab_id=capture.tab_id,
                set_id=set_id,
                set_name=set_name,
                execution_request=execution_request,
            )
            for set_id, set_name, execution_request in capture.targets
        )
        if targets:
            self._simulation_jobs.request_preview(
                capture.tab_id,
                targets,
                content_token=capture.content_token,
            )

    def has_slider_draft(self) -> bool:
        runtime = self._registry.tab_runtime(self._tab_id)
        return bool(
            runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        )

    def publish_slider_draft_state(self) -> None:
        """Publish this exact binding's current Slider Draft truth."""

        self.sliderDraftStateChanged.emit(
            self._tab_id,
            self.has_slider_draft(),
        )

    def suspend_preview_disposition(self) -> _SuspendedPreviewDisposition:
        """Fence one nested Run/Fit disposition without changing its draft."""

        if self._suspended_preview_disposition is not None:
            raise RuntimeError("A Slider Preview disposition is already active.")
        self._interactive_preview_owner.retire(self._tab_id)
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        transaction = _SuspendedPreviewDisposition(
            capture=self._preview_capture,
            pending_targets=None,
        )
        self._preview_capture = None
        self._suspended_preview_disposition = transaction
        self._sync_slider_transient_custody()
        return transaction

    def resume_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """Resume the unchanged exact capture only after a modal cancellation."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        self._suspended_preview_disposition = None
        self._preview_capture = transaction.capture
        self._sync_slider_transient_custody()
        if transaction.capture is not None:
            self._start_throttle_if_inactive(transaction.capture.debounce_kind)

    def discard_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """End one disposition fence without restoring its Preview capture."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        self._simulation_jobs.drop_pending_preview(self._tab_id)
        self._suspended_preview_disposition = None
        self._preview_capture = None
        self._sync_slider_transient_custody()

    def discard_slider_draft(self, *, render: bool = True) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        runtime = self._registry.tab_runtime(self._tab_id)
        parameter_dirty = bool(runtime.slider_parameter_draft)
        changed = bool(
            runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        )
        if not changed:
            self.publish_slider_draft_state()
            return False
        tab = self._tab_snapshot or self._registry.tab(self._tab_id)
        if tab is None:
            return self._reject(
                f"Cannot discard Slider draft for unavailable tab {self._tab_id!r}."
            )
        affected = (
            tuple(item.set_id for item in tab.initial_condition_sets)
            if parameter_dirty
            else tuple(runtime.slider_species_drafts_by_set_id)
        )
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self.cancel_pending_preview_intent()
        self._latest_preview_set_ids.clear()
        self.clear_preview_detailed_balance_block()
        self._reconcile_canonical_set_ids(tab, affected)
        if render:
            self.render()
        self._emit_draft_state()
        return changed

    def cancel_pending_preview_intent(self) -> None:
        """Fence every timer, transient request, and exact pending intent."""

        self._cancel_deferred_restore()
        self._clear_gesture_state()
        self._replace_preview_capture(
            None,
            message="Interactive Preview exact capture retired.",
        )
        self._interactive_preview_owner.retire(self._tab_id)
        self._simulation_jobs.drop_pending_preview(self._tab_id)
        self._latest_preview_set_ids.clear()
        self._sync_slider_transient_custody()

    def fence_stop(self) -> bool:
        """Fence local gesture/capture roles before physical display freeze."""

        self._interactive_stop_fenced = True
        changed = bool(
            self._throttle_timer.isActive()
            or self._active_drag is not None
            or self._latest_admissible_pending is not None
            or self._preview_capture is not None
            or self._simulation_jobs.pending_preview_tab_id == self._tab_id
        )
        self._stop_exact_set_ids = tuple(sorted(self._latest_preview_set_ids))
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        self._preview_capture = None
        self._sync_slider_transient_custody()
        changed = self.clear_preview_detailed_balance_block() or changed
        return changed

    def settle_stop(self) -> bool:
        """Ask the operation owner to settle roles after physical freeze."""

        exact_set_ids = self._stop_exact_set_ids
        self._stop_exact_set_ids = ()
        changed = self._interactive_preview_owner.retire(self._tab_id)
        self._simulation_jobs.drop_pending_preview(self._tab_id)
        self._latest_preview_set_ids.clear()
        self._sync_slider_transient_custody()
        return bool(changed)

    def apply(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        runtime = self._registry.tab_runtime(self._tab_id)
        parameter_updates = dict(runtime.slider_parameter_draft)
        species_updates = {
            set_id: dict(values)
            for set_id, values in runtime.slider_species_drafts_by_set_id.items()
            if values
        }
        if not parameter_updates and not species_updates:
            self.publish_slider_draft_state()
            return False
        preflight_targets = tuple(
            dict.fromkeys(
                (*self.resolve_target_set_ids(), *species_updates)
            )
        )
        prepared = self._preflight_preview_action(preflight_targets)
        if prepared is None:
            return False
        if isinstance(prepared, _PreviewPreflight):
            tab, _projection = prepared.tab, prepared.projection
        else:
            tab, _projection = prepared
        runtime = self._registry.tab_runtime(self._tab_id)
        parameter_updates = dict(runtime.slider_parameter_draft)
        species_updates = {
            set_id: dict(values)
            for set_id, values in runtime.slider_species_drafts_by_set_id.items()
            if values
        }

        source = tab.mechanism_source
        candidate_prepared_dsl = tab.prepared_dsl
        if parameter_updates:
            try:
                prepared_dsl = tab.prepared_dsl
                if not isinstance(prepared_dsl, PreparedDslRequestSet):
                    raise TypeError(
                        "Slider Apply requires retained DSL analysis."
                    )
                derived_source, candidate_prepared_dsl = (
                    derive_source_with_parameter_updates(
                        source.reactions_text,
                        prepared_dsl,
                        tuple(
                            DslParameterCoordinateUpdate(
                                name=str(name),
                                value=float(value),
                            )
                            for name, value in parameter_updates.items()
                        ),
                        wegscheider_cyclicity_enabled=bool(
                            tab.scientific_settings.wegscheider_policy
                        ),
                    )
                )
                source = source.with_reactions_text(derived_source)
            except (LookupError, TypeError, ValueError) as exc:
                return self._reject(f"Slider Apply failed ({exc}).")
        candidate_sets = []
        for initial_set in tab.initial_condition_sets:
            updates = species_updates.get(initial_set.set_id)
            if not updates:
                candidate_sets.append(initial_set)
                continue
            values = initial_set.values_dict()
            values.update(updates)
            candidate_sets.append(initial_set.with_values(values))
        candidate = replace(
            tab,
            mechanism_source=source,
            prepared_dsl=candidate_prepared_dsl,
        ).with_initial_condition_sets(tuple(candidate_sets))
        result = self._registry.replace_tab(
            self._tab_id,
            candidate,
            expected_token=self._registry.capture_content_token(),
        )
        if result is None:
            return False
        self.cancel_pending_preview_intent()
        self.clear_preview_detailed_balance_block()
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self._latest_preview_set_ids.clear()
        self._emit_draft_state()
        return True

    def reset(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        targets = self.resolve_target_set_ids()
        parameter_dirty = bool(runtime.slider_parameter_draft)
        if not parameter_dirty and not any(
            runtime.slider_species_drafts_by_set_id.get(set_id)
            for set_id in targets
        ):
            self.publish_slider_draft_state()
            return False
        changed = parameter_dirty
        runtime.slider_parameter_draft.clear()
        for set_id in targets:
            changed = bool(
                runtime.slider_species_drafts_by_set_id.pop(set_id, None)
            ) or changed
        self.cancel_pending_preview_intent()
        self._latest_preview_set_ids.clear()
        self.clear_preview_detailed_balance_block()
        reconciled_targets = (
            tuple(item.set_id for item in tab.initial_condition_sets)
            if parameter_dirty
            else targets
        )
        remaining_dirty = set(runtime.slider_species_drafts_by_set_id)
        canonical_targets = tuple(
            set_id
            for set_id in reconciled_targets
            if set_id not in remaining_dirty
        )
        still_dirty_targets = tuple(
            set_id
            for set_id in reconciled_targets
            if set_id in remaining_dirty
        )
        if canonical_targets:
            self._reconcile_canonical_set_ids(tab, canonical_targets)
        if still_dirty_targets:
            self._simulation_jobs.deauthorize_targets(
                tab.tab_id,
                still_dirty_targets,
            )
        self.render()
        self._emit_draft_state()
        return changed

    def _preview_values(
        self,
        tab: MechanismTabDocument,
        projection: ParameterSliderProjection,
        set_id: str,
    ) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
        runtime = self._registry.tab_runtime(tab.tab_id)
        parameter_values = projection.values_dict()
        parameter_values.update(runtime.slider_parameter_draft)
        initial_values = dict(canonical_initial_values(tab, set_id))
        initial_values.update(
            runtime.slider_species_drafts_by_set_id.get(set_id, {})
        )
        return (
            tuple(parameter_values.items()),
            tuple(initial_values.items()),
        )

    def _emit_draft_state(self) -> None:
        self.draftStateChanged.emit(
            self._tab_id,
            self._registry.tab_runtime(self._tab_id).has_dirty_drafts,
        )
        self.publish_slider_draft_state()

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

    def _restore_rejected_slider_action(self) -> bool:
        if self._defer_rejected_slider_restore:
            self._queue_rejected_slider_restore()
            return False
        if self._active_drag is not None:
            return False
        self.render(preserve_visibility=True)
        return False

    def _reject_slider_action(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return self._restore_rejected_slider_action()

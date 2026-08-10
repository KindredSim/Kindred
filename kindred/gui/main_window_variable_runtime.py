"""Tab-bound parameter/species Slider drafts and exact Preview intent."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable, Iterable

from PySide6 import QtCore

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    MechanismSourceReconciliation,
    MechanismTabDocument,
    canonical_initial_values,
    canonical_species_names,
)
from kindred.core.result_repository import (
    PreviewResultIdentity,
    PreviewResultKey,
    preview_result_identity_for_tab,
)
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_simulation_execution_request,
)
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_prepared_dsl_with_parameter_updates,
)
from kindred.core.simulator.wegscheider_symbolic import WegscheiderCyclicityReport
from kindred.gui.parameter_enumeration import (
    GuiParameterCoordinate,
    enumerate_parameter_coordinates,
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
from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed

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
    prepared_dsl: PreparedDslRequestSet
    projection: ParameterSliderProjection
    targets: tuple[
        tuple[PreviewResultKey, str, SimulationExecutionRequest], ...
    ]
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
    previewTransitionOriginCaptureRequested = QtCore.Signal(object, bool)
    previewEndpointPresentationRequested = QtCore.Signal(object, bool)
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
        preview_preflight: Callable[[tuple[str, ...]], object] | None = None,
        parameter_preview_debounce_ms: int = 80,
        equilibrium_preview_debounce_ms: int = 150,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        registry.tab(target)
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
        self._latest_preview_keys: dict[str, PreviewResultKey] = {}
        self._defer_rejected_slider_restore = False
        self._deferred_restore_pending = False
        self._disposed = False
        variable_sliders.variableChanged.connect(
            self._on_parameter_value_changed
        )
        variable_sliders.sliderDragStarted.connect(
            self._on_parameter_drag_started
        )
        variable_sliders.sliderDragFinished.connect(
            self._on_parameter_drag_finished
        )
        species_sliders.speciesEdited.connect(self._on_species_value_changed)
        species_sliders.speciesDragStarted.connect(
            self._on_species_drag_started
        )
        species_sliders.speciesDragFinished.connect(
            self._on_species_drag_finished
        )
        model = table.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            raise TypeError("table must use BatchInitialConditionsTableModel.")
        table.selectionModel().currentChanged.connect(self._on_live_current_changed)
        model.snapshotRendered.connect(self._on_table_snapshot_rendered)

    @property
    def tab_id(self) -> str:
        return self._tab_id

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

    def _clear_gesture_state(self) -> None:
        self._stop_preview_timers()
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        self._gesture_admitted = False
        self._last_admitted_sample = None

    def _start_throttle_if_inactive(self, debounce_kind: str) -> None:
        timer = self._preview_timer(debounce_kind)
        if not timer.isActive():
            timer.start()

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

    @QtCore.Slot(str)
    def _on_parameter_drag_started(self, name: str) -> None:
        self._start_gesture("parameter", str(name))

    @QtCore.Slot(str)
    def _on_species_drag_started(self, species: str) -> None:
        self._start_gesture("species", str(species))

    def _start_gesture(self, kind: str, name: str) -> None:
        """Capture accepted gesture-invariant facts exactly once."""

        # A press is only raw widget intent.  It must not discard an already
        # admitted exact Preview capture/intent or stop its submission timer.
        # The first accepted numeric sample performs the exact admission fence.
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        self._gesture_admitted = False
        self._last_admitted_sample = None
        self._active_drag = (str(kind), str(name))
        targets = self.resolve_target_set_ids()
        prepared = self._preflight_preview_action(targets)
        if prepared is None:
            return
        tab, projection = prepared
        self._gesture_capture = _GestureCapture(
            document_identity=self._registry.document_identity(self._tab_id),
            tab=tab,
            projection=projection,
            targets=targets,
        )

    @QtCore.Slot(str, float)
    def _on_parameter_value_changed(self, name: str, value: float) -> None:
        if self._active_drag == ("parameter", str(name)):
            self._record_raw_gesture("parameter", name, value)
            return
        self.stage_parameter(name, value)

    @QtCore.Slot(str, float)
    def _on_species_value_changed(self, species: str, value: float) -> None:
        if self._active_drag == ("species", str(species)):
            self._record_raw_gesture("species", species, value)
            return
        self.stage_species(species, value)

    @QtCore.Slot(str)
    def _on_parameter_drag_finished(self, name: str) -> None:
        if self._active_drag != ("parameter", str(name)):
            return
        value = self._variable_sliders.get_variables().get(str(name))
        capture = self._gesture_capture
        self._stop_preview_timers()
        self._active_drag = None
        self._latest_admissible_pending = None
        if value is not None and capture is not None:
            self._accept_and_submit(
                "parameter",
                name,
                float(value),
                defer_rejected_restore=True,
                gesture_capture=capture,
                animate_preview=True,
            )
        else:
            self._queue_rejected_slider_restore()
        self._clear_gesture_state()

    @QtCore.Slot(str)
    def _on_species_drag_finished(self, species: str) -> None:
        if self._active_drag != ("species", str(species)):
            return
        value = self._species_sliders.get_current_values().get(str(species))
        capture = self._gesture_capture
        self._stop_preview_timers()
        self._active_drag = None
        self._latest_admissible_pending = None
        if value is not None and capture is not None:
            self._accept_and_submit(
                "species",
                species,
                float(value),
                defer_rejected_restore=True,
                gesture_capture=capture,
                animate_preview=True,
            )
        else:
            self._queue_rejected_slider_restore()
        self._clear_gesture_state()

    def _record_raw_gesture(self, kind: str, name: str, value: float) -> None:
        value_f = float(value)
        kind_s = str(kind)
        name_s = str(name)
        debounce_kind = self._gesture_debounce_kind(kind_s, name_s)
        capture = self._gesture_capture
        if capture is None:
            return
        if not self._gesture_admitted:
            accepted = self._accept_and_submit(
                kind,
                name,
                value_f,
                gesture_capture=capture,
                animate_preview=True,
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
            and self._current_preview_keys_are_current(kind_s)
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

    def _current_preview_keys_are_current(self, kind: str) -> bool:
        targets = self.resolve_target_set_ids()
        if not targets:
            return False
        return all(
            (
                key := self._latest_preview_keys.get(set_id)
            ) is not None
            and self._simulation_jobs.preview_key_is_current(key)
            for set_id in targets
        )

    def _accept_and_submit(
        self,
        kind: str,
        name: str,
        value: float,
        *,
        defer_rejected_restore: bool = False,
        gesture_capture: _GestureCapture | None = None,
        animate_preview: bool = False,
    ) -> bool:
        self._cancel_deferred_restore()
        self._stop_preview_timers()
        self._defer_rejected_slider_restore = bool(defer_rejected_restore)
        try:
            accepted = (
                self.stage_parameter(
                    name,
                    value,
                    _gesture_capture=gesture_capture,
                    _animate_preview=animate_preview,
                )
                if str(kind) == "parameter"
                else self.stage_species(
                    name,
                    value,
                    _gesture_capture=gesture_capture,
                    _animate_preview=animate_preview,
                )
            )
        finally:
            self._defer_rejected_slider_restore = False
        if accepted:
            self._submit_preview()
        elif defer_rejected_restore:
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
                current_tab, _current_projection = prepared
                if current_tab.tab_id == tab.tab_id:
                    self._begin_preview_cycle(
                        prepared,
                        targets,
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
                self._latest_preview_keys.pop(set_id, None)
        if clean_removed:
            self._simulation_jobs.reconcile_canonical_targets(
                tab.tab_id,
                clean_removed,
            )
        return current != previous

    def dispose(self) -> None:
        """Forget Slider-owned captures after the Run owner purges a tab."""

        self._disposed = True
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        self.cancel_pending_preview_intent()
        self._projection = None
        self._tab_snapshot = None
        self._last_targets = ()
        self._latest_preview_keys.clear()
        self.clear_preview_detailed_balance_block()

    def _evict_preview_targets(self, set_ids: Iterable[str]) -> None:
        """Forget Slider Preview keys and captures for deleted targets."""

        targets = {str(set_id) for set_id in set_ids}
        if not targets:
            return
        for set_id in targets:
            self._latest_preview_keys.pop(set_id, None)
        capture = self._preview_capture
        if capture is None:
            return
        surviving = tuple(
            target
            for target in capture.targets
            if target[0].set_id not in targets
        )
        if surviving:
            self._preview_capture = replace(capture, targets=surviving)
        else:
            self._preview_capture = None
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
            # Numerically equivalent accepted values share one display value.
            if any(
                not math.isclose(
                    float(values.get(species, 0.0)),
                    first,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
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
        _gesture_capture: _GestureCapture | None = None,
        _animate_preview: bool = False,
    ) -> bool:
        if (
            self._rendering
            or self._suspended_preview_disposition is not None
        ):
            return False
        value_f = float(value)
        if not math.isfinite(value_f):
            return self._reject_slider_action(
                f"Parameter {name!r} must be finite."
            )
        targets = self.resolve_target_set_ids()
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
        tab, projection = prepared
        metadata = projection.metadata_dict()
        coordinate_metadata = metadata.get(str(name))
        if coordinate_metadata is None:
            return self._reject_slider_action(
                f"Parameter {name!r} is absent."
            )
        if coordinate_metadata.get("editable") is False:
            return self._reject_slider_action(
                f"Parameter {name!r} is derived and cannot be edited."
            )
        runtime = self._registry.tab_runtime(tab.tab_id)
        canonical = projection.values_dict()
        previous = dict(runtime.slider_parameter_draft)
        was_parameter_dirty = bool(previous)
        if str(name) in canonical and value_f == canonical[str(name)]:
            runtime.slider_parameter_draft.pop(str(name), None)
        else:
            runtime.slider_parameter_draft[str(name)] = value_f
        if runtime.slider_parameter_draft == previous:
            self._ensure_preview_capture_for_draft(
                tab,
                projection,
                targets,
                debounce_kind=(
                    "equilibrium" if str(coordinate_metadata.get("role") or "") == "Keq"
                    else "parameter"
                ),
                animate_preview=_animate_preview,
            )
            return True
        target_ids = set(targets)
        unselected = tuple(
            item.set_id
            for item in tab.initial_condition_sets
            if item.set_id not in target_ids
        )
        if (
            runtime.slider_parameter_draft
            and not was_parameter_dirty
            and unselected
        ):
            # A global parameter edit is scientific for every set, even when
            # Preview itself is scoped to exact Slider targets.
            self._simulation_jobs.deauthorize_targets(tab.tab_id, unselected)
        elif (
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
        if not targets:
            self.cancel_pending_preview_intent()
        else:
            role = str(coordinate_metadata.get("role") or "")
            self._begin_preview_cycle(
                prepared,
                targets,
                debounce_kind=(
                    "equilibrium" if role == "Keq" else "parameter"
                ),
                animate_preview=_animate_preview,
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
        _gesture_capture: _GestureCapture | None = None,
        _animate_preview: bool = False,
    ) -> bool:
        if (
            self._rendering
            or self._suspended_preview_disposition is not None
        ):
            return False
        value_f = float(value)
        if not math.isfinite(value_f) or value_f < 0.0:
            return self._reject_slider_action(
                f"Initial value for {species!r} must be finite and nonnegative."
            )
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
            if math.isclose(
                value_f,
                float(canonical),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
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
            self._ensure_preview_capture_for_draft(
                tab,
                projection,
                targets,
                debounce_kind="species",
                animate_preview=_animate_preview,
            )
            return True
        self._begin_preview_cycle(
            prepared,
            targets,
            debounce_kind="species",
            animate_preview=_animate_preview,
        )
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
    ) -> tuple[MechanismTabDocument, ParameterSliderProjection] | None:
        """Use one canonical source result before any Slider mutation."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        visible_source = runtime.reactions_draft or tab.mechanism_source
        visible_prepared: PreparedDslRequestSet | None = tab.prepared_dsl
        callback_result: object = True
        callback = self._preview_preflight
        if callback is not None:
            callback_result = callback(targets)
            if isinstance(callback_result, MechanismSourceReconciliation):
                tab = self._registry.tab(self._tab_id)
                visible_source = tab.mechanism_source
                visible_prepared = callback_result.prepared_dsl
                if callback_result.detailed_balance.requires_correction:
                    request = self._preview_detailed_balance_request(
                        tab,
                        callback_result.detailed_balance.report,
                        prepared_dsl=visible_prepared,
                        reactions_text=visible_source.reactions_text,
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
        return tab, projection

    def _begin_preview_cycle(
        self,
        transaction: tuple[
            MechanismTabDocument,
            ParameterSliderProjection,
        ],
        targets: tuple[str, ...],
        *,
        debounce_kind: str,
        animate_preview: bool = False,
    ) -> None:
        if self._suspended_preview_disposition is not None:
            return
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
        if canonical_targets:
            self._reconcile_canonical_set_ids(tab, canonical_targets)
        if not dirty_targets:
            self.cancel_pending_preview_intent()
            return
        prepared_dsl = tab.prepared_dsl
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            self._reject(
                "Preview requires accepted Reactions with retained DSL analysis."
            )
            return

        constructed: list[
            tuple[PreviewResultIdentity, str, SimulationExecutionRequest]
        ] = []
        try:
            for set_id in dirty_targets:
                parameter_values, initial_values = self._preview_values(
                    tab,
                    projection,
                    set_id,
                )
                execution_request = build_simulation_execution_request(
                    prepared_dsl=prepared_dsl,
                    mechanism_source=tab.mechanism_source,
                    initial_values=initial_values,
                    execution_settings=tab.resolved_execution_settings(set_id),
                    scientific_settings=tab.scientific_settings,
                    parameter_values=dict(parameter_values),
                )
                contract = execution_request.snapshot.intervention_execution
                if contract is None:
                    raise ValueError(
                        "Preview intervention schedule did not bind to one "
                        "exact execution contract."
                    )
                identity = preview_result_identity_for_tab(
                    tab,
                    set_id,
                    parameter_values=parameter_values,
                    initial_values=initial_values,
                    intervention_execution=contract,
                )
                constructed.append(
                    (
                        identity,
                        tab.initial_condition_set(set_id).name,
                        execution_request,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview construction failed before admission "
                f"({type(exc).__name__}: {exc})."
            )
            self.cancel_pending_preview_intent()
            return

        # Construction above is fallible. Only a complete exact batch reaches
        # the scheduler's single fence/admission transition.
        self._stop_preview_timers()
        self._preview_capture = None
        self.previewTransitionOriginCaptureRequested.emit(
            dirty_targets,
            bool(animate_preview),
        )
        keys = self._simulation_jobs.admit_preview_edits(
            identity for identity, _set_name, _request in constructed
        )
        if not keys:
            self.cancel_pending_preview_intent()
            return
        captured = tuple(
            (key, set_name, execution_request)
            for key, (_identity, set_name, execution_request) in zip(
                keys,
                constructed,
                strict=True,
            )
        )
        for key, _set_name, _execution_request in captured:
            self._latest_preview_keys[key.set_id] = key
        self._preview_capture = _PreviewCapture(
            tab_id=tab.tab_id,
            prepared_dsl=prepared_dsl,
            projection=projection,
            targets=captured,
            debounce_kind=str(debounce_kind),
        )
        self.previewEndpointPresentationRequested.emit(
            tuple(key for key, _set_name, _request in captured),
            bool(animate_preview),
        )
        self._start_throttle_if_inactive(debounce_kind)

    def _ensure_preview_capture_for_draft(
        self,
        tab: MechanismTabDocument,
        projection: ParameterSliderProjection,
        targets: tuple[str, ...],
        *,
        debounce_kind: str,
        animate_preview: bool = False,
    ) -> None:
        """Recapture an exact accepted draft when no current key remains.

        Pending raw motion intentionally leaves both the admitted exact key and
        accepted draft unchanged. Rebuild only when the owner no longer
        considers every affected exact key current.
        """

        runtime = self._registry.tab_runtime(tab.tab_id)
        dirty_targets = tuple(
            set_id
            for set_id in targets
            if runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id.get(set_id)
        )
        if not dirty_targets:
            return
        if all(
            (
                key := self._latest_preview_keys.get(set_id)
            ) is not None
            and self._simulation_jobs.preview_key_is_current(key)
            for set_id in dirty_targets
        ):
            return
        self._begin_preview_cycle(
            (tab, projection),
            targets,
            debounce_kind=debounce_kind,
            animate_preview=animate_preview,
        )

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
        identity = hashlib.sha256(
            (
                f"Preview\0{tab.tab_id}\0"
                f"{tab.scientific_settings.temperature:.17g}\0{source}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        return DetailedBalanceCorrectionRequest(
            request_id=identity,
            origin="Preview",
            tab_id=tab.tab_id,
            source=source,
            issue=issue,
            temperature_K=float(tab.scientific_settings.temperature),
            report=report,
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
            (
                outcome.request_id,
                outcome.origin,
                outcome.tab_id,
                outcome.source,
                outcome.issue,
            )
            != (
                pending.request_id,
                pending.origin,
                pending.tab_id,
                pending.source,
                pending.issue,
            )
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
            if self._simulation_jobs.is_executing:
                self._arm_gesture_throttle(pending.debounce_kind)
                return
            self._latest_admissible_pending = None
            accepted = self._accept_and_submit(
                pending.kind,
                pending.name,
                pending.value,
                gesture_capture=self._gesture_capture,
                animate_preview=True,
            )
            if accepted:
                self._gesture_admitted = True
                self._last_admitted_sample = (
                    pending.kind,
                    pending.name,
                    pending.value,
                )
            return
        self._submit_preview()

    @QtCore.Slot()
    def _submit_preview(self) -> None:
        from kindred.gui.controllers.simulation_job_owner import (
            PreviewIntent,
            build_preview_target,
        )

        self._throttle_timer.stop()
        if self._suspended_preview_disposition is not None:
            return
        capture = self._preview_capture
        self._preview_capture = None
        if capture is None:
            return
        targets = tuple(
            build_preview_target(
                prepared_dsl=capture.prepared_dsl,
                key=key,
                set_name=set_name,
                execution_request=execution_request,
            )
            for key, set_name, execution_request in capture.targets
            if self._simulation_jobs.preview_key_is_current(key)
        )
        if targets:
            self._simulation_jobs.request_preview(
                PreviewIntent(tab_id=capture.tab_id, targets=targets)
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
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        transaction = _SuspendedPreviewDisposition(
            capture=self._preview_capture,
        )
        self._preview_capture = None
        self._suspended_preview_disposition = transaction
        self._simulation_jobs.drop_pending_preview(self._tab_id)
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
        if transaction.capture is not None:
            self._start_throttle_if_inactive(transaction.capture.debounce_kind)

    def discard_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """End one disposition fence without restoring its Preview capture."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        self._suspended_preview_disposition = None
        self._preview_capture = None
        self._simulation_jobs.drop_pending_preview(self._tab_id)

    def discard_slider_draft(self) -> bool:
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
        self._latest_preview_keys.clear()
        self.cancel_pending_preview_intent()
        self.clear_preview_detailed_balance_block()
        self._reconcile_canonical_set_ids(tab, affected)
        self.render()
        self._emit_draft_state()
        return changed

    def cancel_pending_preview_intent(self) -> None:
        capture = self._preview_capture
        self._stop_preview_timers()
        if capture is not None:
            self._preview_capture = None
        self._simulation_jobs.drop_pending_preview(self._tab_id)

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
                candidate_prepared_dsl = (
                    derive_prepared_dsl_with_parameter_updates(
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
            except (LookupError, TypeError, ValueError) as exc:
                return self._reject(f"Slider Apply failed ({exc}).")
            source = MechanismAuthoringSource.from_parts(
                reactions_text=candidate_prepared_dsl.analysis.normalized_text
            )

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
        for tab_id, set_ids in result.delta.affected_targets_by_tab:
            if set_ids:
                self._simulation_jobs.deauthorize_targets(tab_id, set_ids)
        self.clear_preview_detailed_balance_block()
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self._latest_preview_keys.clear()
        self.render()
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
        self._latest_preview_keys.clear()
        for set_id in targets:
            changed = bool(
                runtime.slider_species_drafts_by_set_id.pop(set_id, None)
            ) or changed
        self.cancel_pending_preview_intent()
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

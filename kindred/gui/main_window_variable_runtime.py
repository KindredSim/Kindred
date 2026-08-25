"""Tab-bound parameter/species Slider drafts and exact Preview intent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Literal

from PySide6 import QtCore

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
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
    PreparedDslRequestSet,
)
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
from kindred.gui.controllers.simulation_job_owner import (
    PreviewCapture,
    PreviewTarget,
    build_explicit_target,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
    resolve_live_slider_target_set_ids,
)
from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed

logger = logging.getLogger(__name__)

__all__ = [
    "TabSliderBinding",
    "ActiveSliderRenderProjection",
    "ExactPreviewCapture",
]


SourceProvenance = Literal["canonical", "draft"]


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
    tab_id: str


@dataclass(frozen=True, slots=True)
class _PreviewPreflight:
    tab: MechanismTabDocument
    projection: ParameterSliderProjection
    targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactPreviewCapture:
    """Exact Slider Preview admission facts consumed by one Apply."""

    capture: PreviewCapture
    candidate_tab: MechanismTabDocument


def build_parameter_slider_projection(
    prepared_dsl: PreparedDslRequestSet,
) -> ParameterSliderProjection:
    """Project one accepted typed DSL authority into stable Slider values."""

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("prepared_dsl must be a PreparedDslRequestSet.")
    coordinates = tuple(
        coordinate
        for coordinate in enumerate_parameter_coordinates(
            prepared_dsl,
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
        exact_preview_reviewer: Callable[[ExactPreviewCapture], object],
        interactive_preview_reviewer: Callable[[InteractivePreviewRequest], object],
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
        if not callable(exact_preview_reviewer):
            raise TypeError("exact_preview_reviewer must be callable.")
        if not callable(interactive_preview_reviewer):
            raise TypeError("interactive_preview_reviewer must be callable.")
        self._exact_preview_reviewer = exact_preview_reviewer
        self._interactive_preview_reviewer = interactive_preview_reviewer
        self._projection: ParameterSliderProjection | None = None
        self._tab_snapshot: MechanismTabDocument | None = None
        self._last_targets: tuple[str, ...] = ()
        self._suspended_preview_disposition: (
            _SuspendedPreviewDisposition | None
        ) = None
        self._exact_preview_capture: ExactPreviewCapture | None = None
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
        self._interactive_stop_fenced = False
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
        """Return semantic exact Preview identities from the owner."""

        return tuple(
            self._simulation_jobs.preview_set_ids(self._tab_id)
        )

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

    def _start_gesture(self, kind: str, name: str) -> None:
        """Record raw widget intent without changing publication authority."""

        # A press is only raw widget intent.  The owner receives exact Preview
        # only after the release builds a complete request batch.
        self._active_drag = None
        self._gesture_capture = None
        self._latest_admissible_pending = None
        self._active_drag = (str(kind), str(name))

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
        self._accept_and_submit(
            kind,
            name,
            float(value),
            interaction=interaction,
            defer_rejected_restore=(
                interaction is SliderInteractionKind.EXACT_DRAG_RELEASE
            ),
            gesture_capture=capture,
        )
        self._clear_gesture_state(stop_preview_timer=True)

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
        self._latest_admissible_pending = pending
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
        previous = self._last_targets
        current = tuple(targets)
        self._last_targets = current
        if current != previous:
            self.targetSetIdsChanged.emit(current)
        removed = tuple(set_id for set_id in previous if set_id not in current)
        runtime = self._registry.tab_runtime(tab.tab_id)
        inherited_species: dict[str, float] = {}
        existing = {
            initial_set.set_id for initial_set in tab.initial_condition_sets
        }
        for set_id in previous:
            if set_id not in existing:
                continue
            draft = runtime.slider_species_drafts_by_set_id.get(set_id)
            if not draft:
                continue
            inherited_species.update(
                {
                    species: float(value)
                    for species, value in (
                        dict(canonical_initial_values(tab, set_id))
                        | dict(draft)
                    ).items()
                    if species in draft
                }
            )
            break
        for set_id in set(previous).difference(current):
            runtime.slider_species_drafts_by_set_id.pop(set_id, None)
        for set_id in current:
            if set_id in previous:
                continue
            canonical = dict(canonical_initial_values(tab, set_id))
            inherited = {
                species: float(value)
                for species, value in inherited_species.items()
                if float(value) != float(canonical.get(species, 0.0))
            }
            if inherited:
                runtime.slider_species_drafts_by_set_id[set_id] = inherited
            else:
                runtime.slider_species_drafts_by_set_id.pop(set_id, None)
        if removed:
            self._simulation_jobs.reconcile_canonical_targets(
                tab.tab_id,
                removed,
            )
        return current != previous

    def dispose(self, *, project_replacement_settled: bool = False) -> None:
        """Disconnect and forget Slider-owned captures idempotently.

        Project replacement calls this after the semantic owners and registry
        have already settled and the new document is installed.  That path
        only tears down the old physical binding; ordinary tab disposal still
        retires the binding's live Preview roles here.
        """

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
        if project_replacement_settled:
            self._cancel_deferred_restore()
            self._interactive_stop_fenced = True
            self._clear_gesture_state()
        else:
            self.cancel_pending_preview_intent()
        self._exact_preview_capture = None
        self._projection = None
        self._tab_snapshot = None
        self._last_targets = ()

    def evict_preview_targets(self, set_ids: Iterable[str]) -> None:
        """Retire transient Preview participation for deleted targets."""

        targets = {str(set_id) for set_id in set_ids}
        if not targets:
            return
        self._simulation_jobs.retire_preview(self._tab_id, targets)

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
            )
        preview_admitted = self._begin_preview_cycle(
            (tab, projection),
            targets,
            interaction=_interaction,
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
            if unselected:
                self._simulation_jobs.reconcile_canonical_targets(
                    tab.tab_id,
                    unselected,
                )
        self._emit_draft_state()
        return True

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
            )
        preview_admitted = self._begin_preview_cycle(
            (tab, projection),
            targets,
            interaction=_interaction,
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
        admitted_targets = tuple(targets)
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
        return _PreviewPreflight(tab, projection, admitted_targets)

    def _construct_preview_requests(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
        *,
        sampling: InteractivePreviewSampling | None = None,
    ) -> tuple[tuple[str, str, SimulationExecutionRequest], ...]:
        prepared_dsl = tab.prepared_dsl
        runtime = self._registry.tab_runtime(tab.tab_id)
        parameter_values = {
            str(name): float(value)
            for name, value in runtime.slider_parameter_draft.items()
        }
        constructed: list[tuple[str, str, SimulationExecutionRequest]] = []
        for set_id in targets:
            initial_values = dict(canonical_initial_values(tab, set_id))
            initial_values.update(
                {
                    str(name): float(value)
                    for name, value in runtime.slider_species_drafts_by_set_id.get(
                        set_id,
                        {},
                    ).items()
                }
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
                parameter_values=parameter_values or None,
            )
            constructed.append(
                (
                    set_id,
                    tab.initial_condition_set(set_id).name,
                    execution_request,
                )
            )
        return tuple(constructed)

    def _capture_preview_batch(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
        *,
        content_token: ProjectContentToken | None,
        source: MechanismAuthoringSource | None,
        source_provenance: SourceProvenance | None,
    ) -> tuple[
        ProjectContentToken,
        MechanismAuthoringSource,
        SourceProvenance,
        MechanismTabDocument,
        tuple[tuple[str, str], ...],
    ]:
        if tab.tab_id != self._tab_id:
            raise ValueError("Preview capture tab does not belong to this binding.")
        runtime = self._registry.tab_runtime(tab.tab_id)
        token = (
            self._registry.capture_content_token()
            if content_token is None
            else content_token
        )
        visible_source = (
            runtime.reactions_draft or tab.mechanism_source
            if source is None
            else source
        )
        provenance = (
            ("draft" if runtime.reactions_draft is not None else "canonical")
            if source_provenance is None
            else source_provenance
        )
        target_specs = tuple(
            (
                set_id,
                tab.initial_condition_set(set_id).name,
            )
            for set_id in targets
        )
        return token, visible_source, provenance, tab, target_specs

    def capture_exact_preview(
        self,
        targets: Iterable[str],
        *,
        tab: MechanismTabDocument | None = None,
        content_token: ProjectContentToken | None = None,
        source: MechanismAuthoringSource | None = None,
        source_provenance: SourceProvenance | None = None,
    ) -> ExactPreviewCapture | None:
        """Capture one exact Preview before any external review continuation."""

        if self._suspended_preview_disposition is not None:
            return None
        roster = tuple(str(value) for value in targets)
        base_tab = self._registry.tab(self._tab_id) if tab is None else tab
        try:
            token, visible_source, provenance, admitted_tab, target_specs = (
                self._capture_preview_batch(
                    base_tab,
                    roster,
                    content_token=content_token,
                    source=source,
                    source_provenance=source_provenance,
                )
            )
            constructed = self._construct_preview_requests(admitted_tab, roster)
            canonical_targets = tuple(
                build_explicit_target(
                    prepared_dsl=admitted_tab.prepared_dsl,
                    tab_id=admitted_tab.tab_id,
                    set_id=set_id,
                    set_name=set_name,
                    mechanism_source=admitted_tab.mechanism_source,
                    initial_values=canonical_initial_values(admitted_tab, set_id),
                    execution_settings=admitted_tab.resolved_execution_settings(
                        set_id
                    ),
                    scientific_settings=admitted_tab.scientific_settings,
                )
                for set_id, set_name, _execution_request in constructed
            )
            capture = PreviewCapture(
                tab_id=base_tab.tab_id,
                content_token=token,
                source=visible_source,
                source_provenance=provenance,
                targets=tuple(
                    PreviewTarget(
                        tab_id=base_tab.tab_id,
                        set_id=set_id,
                        set_name=set_name,
                        execution_request=execution_request,
                        canonical_run_target=canonical_target,
                    )
                    for (set_id, set_name), (_, _, execution_request), canonical_target in zip(
                        target_specs,
                        constructed,
                        canonical_targets,
                        strict=True,
                    )
                ),
            )
            return ExactPreviewCapture(capture=capture, candidate_tab=admitted_tab)
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview construction failed before admission "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    def capture_interactive_preview(
        self,
        targets: Iterable[str],
        *,
        sampling: InteractivePreviewSampling | None = None,
        tab: MechanismTabDocument | None = None,
    ) -> InteractivePreviewRequest | None:
        """Prepare one admitted Interactive Preview request."""

        if self._suspended_preview_disposition is not None:
            return None
        roster = tuple(str(value) for value in targets)
        base_tab = self._registry.tab(self._tab_id) if tab is None else tab
        try:
            sampling = sampling or InteractivePreviewSampling()
            constructed = self._construct_preview_requests(
                base_tab,
                roster,
                sampling=sampling,
            )
            return InteractivePreviewRequest(
                tab_id=base_tab.tab_id,
                targets=tuple(
                    InteractivePreviewTarget(
                        set_id=set_id,
                        set_name=set_name,
                        execution_request=execution_request,
                    )
                    for set_id, set_name, execution_request in constructed
                ),
                sampling=sampling,
            )
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview construction failed before admission "
                f"({type(exc).__name__}: {exc})."
            )
            return None

    @staticmethod
    def _with_confirmed_currentness(
        capture: PreviewCapture,
        *,
        content_token: ProjectContentToken | None,
        source: MechanismAuthoringSource | None,
        source_provenance: SourceProvenance | None,
    ) -> PreviewCapture:
        values = (content_token, source, source_provenance)
        if any(value is not None for value in values) and any(
            value is None for value in values
        ):
            raise ValueError(
                "confirmed token, source, and provenance must be supplied together."
            )
        if content_token is None:
            return capture
        return replace(
            capture,
            content_token=content_token,
            source=source,
            source_provenance=source_provenance,
        )

    def continue_exact_preview(
        self,
        capture: ExactPreviewCapture,
        *,
        candidate_tab: MechanismTabDocument | None = None,
        content_token: ProjectContentToken | None = None,
        source: MechanismAuthoringSource | None = None,
        source_provenance: SourceProvenance | None = None,
    ) -> bool:
        """Continue one capture with caller-confirmed currentness facts."""

        if not isinstance(capture, ExactPreviewCapture):
            raise TypeError("capture must be ExactPreviewCapture.")
        if content_token is not None and candidate_tab is None:
            raise ValueError(
                "confirmed Preview currentness requires the corrected candidate tab."
            )
        current = self._with_confirmed_currentness(
            capture.capture,
            content_token=content_token,
            source=source,
            source_provenance=source_provenance,
        )
        admitted = replace(capture, capture=current)
        if self._suspended_preview_disposition is not None:
            return False
        self._stop_preview_timers()
        try:
            started = self._simulation_jobs.request_preview(admitted.capture)
        except Exception as exc:  # noqa: BLE001
            self._reject(
                "Preview admission failed "
                f"({type(exc).__name__}: {exc})."
            )
            return False
        self._exact_preview_capture = admitted
        return bool(
            started
            or self._simulation_jobs.pending_preview_tab_id == admitted.capture.tab_id
        )

    def continue_interactive_preview(
        self,
        request: InteractivePreviewRequest,
    ) -> bool:
        """Start one already-prepared Interactive Preview request."""

        if self._suspended_preview_disposition is not None:
            return False
        self._stop_preview_timers()
        self._exact_preview_capture = None
        started = self._simulation_jobs.request_interactive_preview(request)
        return bool(started or self._simulation_jobs.interactive_preview_busy)

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
    ) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        tab, _projection = transaction
        runtime = self._registry.tab_runtime(tab.tab_id)
        roster = tuple(targets)
        has_draft = bool(
            runtime.slider_parameter_draft
            or any(
                runtime.slider_species_drafts_by_set_id.get(set_id)
                for set_id in roster
            )
        )
        if (
            not has_draft
            and interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE
        ):
            self.cancel_pending_preview_intent(
                roster,
                preserve_reference=True,
            )
            return True
        if interaction is SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE:
            request = self.capture_interactive_preview(
                roster,
                sampling=InteractivePreviewSampling(),
                tab=tab,
            )
            if request is None:
                return False
            self._interactive_preview_reviewer(request)
            return True
        capture = self.capture_exact_preview(roster, tab=tab)
        if capture is None:
            return False
        self._exact_preview_reviewer(capture)
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
            self._accept_and_submit(
                pending.kind,
                pending.name,
                pending.value,
                interaction=SliderInteractionKind.INTERACTIVE_DRAG_SAMPLE,
                gesture_capture=self._gesture_capture,
            )
            return

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
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        transaction = _SuspendedPreviewDisposition(self._tab_id)
        self._suspended_preview_disposition = transaction
        return transaction

    def resume_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """Resume the unchanged Slider draft after a modal cancellation."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        self._suspended_preview_disposition = None

    def discard_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """End one disposition fence without restoring its Preview intent."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        targets = self.resolve_target_set_ids()
        self._simulation_jobs.retire_preview(self._tab_id, targets)
        self._suspended_preview_disposition = None
        self._exact_preview_capture = None

    def discard_slider_draft(self, *, render: bool = True) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        runtime = self._registry.tab_runtime(self._tab_id)
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
        targets = self.resolve_target_set_ids()
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self.cancel_pending_preview_intent(targets)
        if render:
            self.render()
        self._emit_draft_state()
        return changed

    def cancel_pending_preview_intent(
        self,
        set_ids: tuple[str, ...] | None = None,
        *,
        preserve_reference: bool = False,
    ) -> None:
        """Fence every timer, transient request, and exact pending intent."""

        self._cancel_deferred_restore()
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        self._simulation_jobs.retire_preview(
            self._tab_id,
            set_ids,
            preserve_reference=preserve_reference,
        )
        self._exact_preview_capture = None

    def fence_stop(self) -> bool:
        """Fence local gesture/capture roles before physical display freeze."""

        self._interactive_stop_fenced = True
        changed = bool(
            self._throttle_timer.isActive()
            or self._active_drag is not None
            or self._latest_admissible_pending is not None
            or self._simulation_jobs.pending_preview_tab_id == self._tab_id
        )
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        return changed

    def settle_stop(self) -> bool:
        """Ask the operation owner to settle roles after physical freeze."""

        targets = self.resolve_target_set_ids()
        changed = self._simulation_jobs.retire_preview(self._tab_id, targets)
        self._exact_preview_capture = None
        return bool(changed)

    def apply(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._interactive_stop_fenced = True
        self._clear_gesture_state()
        runtime = self._registry.tab_runtime(self._tab_id)
        roster = self.resolve_target_set_ids()
        if not runtime.slider_parameter_draft and not any(
            runtime.slider_species_drafts_by_set_id.get(set_id)
            for set_id in roster
        ):
            self.publish_slider_draft_state()
            return False
        if runtime.reactions_draft is not None:
            return self._reject(
                "Slider Apply requires a settled exact Preview for the accepted Reactions."
            )
        capture = self._exact_preview_capture
        if capture is None or capture.candidate_tab.tab_id != self._tab_id:
            return self._reject(
                "Slider Apply requires one complete, settled exact Preview for the current draft and target roster."
            )
        if tuple(target.set_id for target in capture.capture.targets) != roster:
            return self._reject(
                "Slider Apply requires one complete, settled exact Preview for the current draft and target roster."
            )
        tab = self._registry.tab(self._tab_id)
        visible_source = runtime.reactions_draft or tab.mechanism_source
        visible_provenance = (
            "draft" if runtime.reactions_draft is not None else "canonical"
        )
        if (
            visible_source != capture.capture.source
            or visible_provenance != capture.capture.source_provenance
        ):
            return False
        if not roster or not self._preview_inputs_valid(roster):
            return False
        if self._simulation_jobs.interactive_preview_busy:
            return self._reject(
                "Slider Apply requires a settled exact Preview; interactive Preview is still active."
            )
        preview_pairs = self._simulation_jobs.capture_preview_pairs_for_apply(
            tab.tab_id,
            roster,
            expected_requests=tuple(
                (target.set_id, target.execution_request)
                for target in capture.capture.targets
            ),
        )
        if preview_pairs is None:
            return self._reject(
                "Slider Apply requires one complete, settled exact Preview for the current draft and target roster."
            )

        result = self._registry.replace_tab(
            self._tab_id,
            capture.candidate_tab,
            expected_token=capture.capture.content_token,
        )
        if result is None:
            return False
        self._simulation_jobs.publish_preview_pairs_as_run(
            tab.tab_id,
            preview_pairs,
        )
        live_runtime = self._registry.tab_runtime(self._tab_id)
        live_runtime.slider_parameter_draft.clear()
        for set_id in roster:
            live_runtime.slider_species_drafts_by_set_id.pop(set_id, None)
        self._exact_preview_capture = None
        return True

    def reset(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        self._cancel_deferred_restore()
        self._clear_gesture_state()
        runtime = self._registry.tab_runtime(self._tab_id)
        targets = self.resolve_target_set_ids()
        self.cancel_pending_preview_intent(targets)
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
        self.render()
        self._emit_draft_state()
        return changed

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

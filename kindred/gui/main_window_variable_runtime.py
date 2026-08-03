"""Active-tab parameter/species Slider drafts and exact Preview intent."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable

from PySide6 import QtCore

from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import MechanismTabDocument
from kindred.core.result_repository import (
    ExplicitResultKey,
    PreviewResultIdentity,
    PreviewResultKey,
    scientific_identity,
    standard_concentration_identity,
)
from kindred.core.simulator.dsl_parse import (
    DslParameterCoordinateUpdate,
    PreparedDslRequestSet,
    derive_prepared_dsl_with_parameter_updates,
    prepare_dsl_request_set,
)
from kindred.core.simulator.errors import DSLError
from kindred.core.units import UnitsModel
from kindred.core.simulator.wegscheider_symbolic import (
    UnresolvedWegscheiderCyclicityError,
    WegscheiderCyclicityReport,
)
from kindred.gui.parameter_enumeration import (
    GuiParameterCoordinate,
    enumerate_parameter_coordinates,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
    projected_analysis_identity,
    projected_initial_values,
    projected_species_names,
    resolve_live_slider_target_set_ids,
)
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionRequest,
    MechanismEditorTabbed,
)
from kindred.gui.workspace_tab_registry import ScientificDocumentDelta

__all__ = [
    "ActiveTabSliderBinding",
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

    def coordinate_dict(self) -> dict[str, GuiParameterCoordinate]:
        return {coordinate.name: coordinate for coordinate in self.coordinates}


@dataclass(frozen=True, slots=True)
class _ActiveSliderRenderProjection:
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
    targets: tuple[tuple[PreviewResultKey, str], ...]
    debounce_kind: str


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


class ActiveTabSliderBinding(QtCore.QObject):
    """Own transient Slider drafts; SimulationJobOwner owns Preview execution."""

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
        registry,
        table: BatchInitialConditionsTableView,
        variable_sliders,
        species_sliders,
        simulation_jobs,
        preview_preflight: Callable[[tuple[str, ...]], bool] | None = None,
        parameter_preview_debounce_ms: int = 80,
        equilibrium_preview_debounce_ms: int = 150,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._table = table
        self._variable_sliders = variable_sliders
        self._species_sliders = species_sliders
        slider_workspace_owner = variable_sliders.parentWidget()
        while (
            slider_workspace_owner is not None
            and not isinstance(slider_workspace_owner, MechanismEditorTabbed)
        ):
            slider_workspace_owner = slider_workspace_owner.parentWidget()
        if not isinstance(slider_workspace_owner, MechanismEditorTabbed):
            raise TypeError("Slider widgets must belong to MechanismEditorTabbed.")
        species_owner = species_sliders.parentWidget()
        while (
            species_owner is not None
            and not isinstance(species_owner, MechanismEditorTabbed)
        ):
            species_owner = species_owner.parentWidget()
        if species_owner is not slider_workspace_owner:
            raise TypeError("Slider widgets must share one MechanismEditorTabbed.")
        self._slider_workspace_owner = slider_workspace_owner
        self._simulation_jobs = simulation_jobs
        self._preview_preflight = preview_preflight
        self._projection_by_tab: dict[str, ParameterSliderProjection] = {}
        self._tab_snapshot_by_id: dict[str, MechanismTabDocument] = {}
        self._last_targets_by_tab: dict[str, tuple[str, ...]] = {}
        self._preview_capture: _PreviewCapture | None = None
        self._suspended_preview_disposition: (
            _SuspendedPreviewDisposition | None
        ) = None
        self._pending_detailed_balance_request: (
            DetailedBalanceCorrectionRequest | None
        ) = None
        self._rendering = False
        self._parameter_timer = self._make_preview_timer(
            parameter_preview_debounce_ms
        )
        self._equilibrium_timer = self._make_preview_timer(
            equilibrium_preview_debounce_ms
        )
        # Species Preview timing is fixed and has no Config preference.
        self._species_timer = self._make_preview_timer(80)
        variable_sliders.variableChanged.connect(self.stage_parameter)
        species_sliders.speciesEdited.connect(self.stage_species)
        model = table.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            raise TypeError("table must use BatchInitialConditionsTableModel.")
        table.selectionModel().currentChanged.connect(self._on_live_current_changed)
        model.snapshotRendered.connect(self._on_table_snapshot_rendered)

    def _make_preview_timer(self, interval_ms: int) -> QtCore.QTimer:
        interval = int(interval_ms)
        if not 0 <= interval <= 1000:
            raise ValueError("Slider debounce must be between 0 and 1000 ms.")
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(interval)
        timer.timeout.connect(self._submit_preview)
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
        self._parameter_timer.setInterval(parameter)
        self._equilibrium_timer.setInterval(equilibrium)

    def _stop_preview_timers(self) -> None:
        self._parameter_timer.stop()
        self._equilibrium_timer.stop()
        self._species_timer.stop()

    def resolve_target_set_ids(self) -> tuple[str, ...]:
        tab = self._registry.active_tab
        return resolve_live_slider_target_set_ids(
            table=self._table,
            pinned_set_ids=self._registry.active_presentation.pinned_slider_set_ids,
            existing_set_ids=(item.set_id for item in tab.initial_condition_sets),
        )

    def prepare_active_render(
        self,
        *,
        preserve_visibility: bool = True,
    ) -> _ActiveSliderRenderProjection:
        """Prepare one complete parameter/species view without widget mutation."""

        tab = self._registry.active_tab
        runtime = self._registry.active_runtime
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
        targets = self.resolve_target_set_ids()
        species_values, mixed_species = self._project_species_targets(
            tab,
            targets,
        )
        return _ActiveSliderRenderProjection(
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
        prepared: _ActiveSliderRenderProjection,
    ) -> None:
        """Apply one fully prepared coherent parameter/species projection."""

        if not isinstance(prepared, _ActiveSliderRenderProjection):
            raise TypeError(
                "prepared must be _ActiveSliderRenderProjection."
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
                    prepared.projection.metadata
                ),
            )
            self._species_sliders.render_values(
                dict(prepared.species_values),
                target_set_ids=prepared.targets,
                target_display_names=prepared.target_display_names,
                mixed_species=prepared.mixed_species,
                preserve_visibility=prepared.preserve_visibility,
                visibility_scope_signature=prepared.species_values,
            )
        finally:
            del species_blocker
            del variable_blocker
            self._rendering = False
        self._projection_by_tab[prepared.tab.tab_id] = prepared.projection
        self._tab_snapshot_by_id[prepared.tab.tab_id] = prepared.tab
        self._slider_workspace_owner.render_slider_workspace_presentation()

    def settle_render_projection(
        self,
        prepared: _ActiveSliderRenderProjection,
    ) -> None:
        """Publish target/Preview effects after the whole workspace applies."""

        if not isinstance(prepared, _ActiveSliderRenderProjection):
            raise TypeError(
                "prepared must be _ActiveSliderRenderProjection."
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

    def render_active(self, *, preserve_visibility: bool = True) -> bool:
        try:
            prepared = self.prepare_active_render(
                preserve_visibility=preserve_visibility
            )
        except (TypeError, ValueError) as exc:
            self.errorRaised.emit(
                "Slider projection failed; the previous coherent Slider "
                f"view was kept: {exc}"
            )
            return False
        self.apply_render_projection(prepared)
        self.settle_render_projection(prepared)
        return True

    def _reconcile_target_change(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> bool:
        existing = {item.set_id for item in tab.initial_condition_sets}
        previous = self._last_targets_by_tab.get(tab.tab_id, ())
        current = tuple(targets)
        self._last_targets_by_tab[tab.tab_id] = current
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
        if clean_removed:
            self._simulation_jobs.reconcile_canonical_targets(
                self._explicit_key(tab, set_id) for set_id in clean_removed
            )
        return current != previous

    def drop_tab_runtime(self, tab_id: str) -> None:
        """Forget Slider-owned captures after the Run owner purges a tab."""

        target = str(tab_id)
        self.cancel_pending_preview_intent(target)
        self._projection_by_tab.pop(target, None)
        self._tab_snapshot_by_id.pop(target, None)
        previous = self._last_targets_by_tab.pop(target, ())
        self.clear_preview_detailed_balance_block(target)
        if previous and target == self._registry.active_tab_id:
            self.targetSetIdsChanged.emit(())

    def reset_project_session_state(self) -> None:
        """Reset only transient Slider/Preview state after project replacement."""

        self._stop_preview_timers()
        self._preview_capture = None
        self._suspended_preview_disposition = None
        self._simulation_jobs.drop_pending_preview()
        for tab_id, _presentation in self._registry.presentation_items():
            runtime = self._registry.tab_runtime(tab_id)
            runtime.slider_parameter_draft.clear()
            runtime.slider_species_drafts_by_set_id.clear()
        self._projection_by_tab.clear()
        self._tab_snapshot_by_id.clear()
        self._last_targets_by_tab.clear()
        self._pending_detailed_balance_request = None
        self._rendering = False

        self._slider_workspace_owner.reset_slider_session_presentation()

    def _project_species_targets(
        self,
        tab: MechanismTabDocument,
        targets: tuple[str, ...],
    ) -> tuple[dict[str, float], frozenset[str]]:
        species_names = tuple(projected_species_names(tab))
        if not targets:
            return ({name: 0.0 for name in species_names}, frozenset())
        runtime = self._registry.tab_runtime(tab.tab_id)
        target_values: list[dict[str, float]] = []
        for set_id in targets:
            values = dict(projected_initial_values(tab, set_id))
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
            self.render_active(preserve_visibility=True)

    @QtCore.Slot()
    def _on_table_snapshot_rendered(self) -> None:
        if not self._rendering:
            self.render_active(preserve_visibility=True)

    @QtCore.Slot(str, float)
    def stage_parameter(self, name: str, value: float) -> bool:
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
        prepared = self._preflight_preview_action(targets)
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
            self.cancel_pending_preview_intent(tab.tab_id)
        else:
            role = str(coordinate_metadata.get("role") or "")
            self._begin_preview_cycle(
                prepared,
                targets,
                debounce_kind=(
                    "equilibrium" if role == "Keq" else "parameter"
                ),
            )
        self._emit_draft_state(tab.tab_id)
        return True

    def _reconcile_canonical_set_ids(
        self,
        tab: MechanismTabDocument,
        set_ids: tuple[str, ...],
    ) -> None:
        self._simulation_jobs.reconcile_canonical_targets(
            self._explicit_key(tab, set_id)
            for set_id in tuple(dict.fromkeys(set_ids))
        )

    @QtCore.Slot(str, float)
    def stage_species(self, species: str, value: float) -> bool:
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
        prepared = self._preflight_preview_action(targets)
        if prepared is None:
            return self._restore_rejected_slider_action()
        tab, projection = prepared
        species_name = str(species)
        if species_name not in projected_species_names(tab):
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
                canonical,
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
            return True
        self._begin_preview_cycle(
            prepared,
            targets,
            debounce_kind="species",
        )
        self._emit_draft_state(tab.tab_id)
        return True

    def _preflight_preview_action(
        self,
        targets: tuple[str, ...],
    ) -> tuple[MechanismTabDocument, ParameterSliderProjection] | None:
        """Settle visible science before any Slider draft/currentness mutation."""

        tab = self._registry.active_tab
        runtime = self._registry.active_runtime
        visible_source = runtime.reactions_draft or tab.mechanism_source
        visible_projection: ParameterSliderProjection | None = None
        visible_prepared: PreparedDslRequestSet | None = None
        try:
            visible_prepared = (
                tab.prepared_dsl
                if runtime.reactions_draft is None
                else prepare_dsl_request_set(
                    visible_source.reactions_text,
                    initials={},
                    units=UnitsModel(
                        temperature_K=tab.scientific_settings.temperature
                    ),
                )
            )
            if visible_prepared is None:
                raise ValueError("Preview requires nonempty Reactions.")
            visible_projection = build_parameter_slider_projection(
                visible_prepared,
                wegscheider_cyclicity_enabled=(
                    tab.scientific_settings.wegscheider_policy
                ),
            )
        except UnresolvedWegscheiderCyclicityError as exc:
            request = self._preview_detailed_balance_request(
                tab,
                exc,
                reactions_text=visible_source.reactions_text,
                source_energy_unit=(
                    str(visible_prepared.analysis.structure.get("energy_unit", "kJ/mol"))
                    if visible_prepared is not None
                    else "kJ/mol"
                ),
            )
            self._pending_detailed_balance_request = request
            self.previewDetailedBalanceCorrectionRequired.emit(request)
            return None
        except (DSLError, TypeError, ValueError):
            # The application settlement callback presents source diagnostics.
            visible_projection = None

        callback = self._preview_preflight
        if callback is None:
            if runtime.reactions_draft is not None:
                self.errorRaised.emit(
                    "Preview requires visible Reactions to be accepted first."
                )
                return None
            if not self._preview_inputs_valid(targets):
                return None
        elif not bool(callback(targets)):
            return None

        tab = self._registry.active_tab
        runtime = self._registry.active_runtime
        try:
            if tab.prepared_dsl is None:
                raise ValueError(
                    "Preview requires accepted Reactions with retained DSL analysis."
                )
            projection = (
                visible_projection
                if visible_prepared is tab.prepared_dsl
                and visible_projection is not None
                else build_parameter_slider_projection(
                    tab.prepared_dsl,
                    wegscheider_cyclicity_enabled=(
                        tab.scientific_settings.wegscheider_policy
                    ),
                )
            )
        except UnresolvedWegscheiderCyclicityError as exc:
            request = self._preview_detailed_balance_request(
                tab,
                exc,
                reactions_text=tab.mechanism_source.reactions_text,
                source_energy_unit=(
                    str(tab.prepared_dsl.analysis.structure.get("energy_unit", "kJ/mol"))
                    if tab.prepared_dsl is not None
                    else "kJ/mol"
                ),
            )
            self._pending_detailed_balance_request = request
            self.previewDetailedBalanceCorrectionRequired.emit(request)
            return None
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
        self._projection_by_tab[tab.tab_id] = projection
        self._tab_snapshot_by_id[tab.tab_id] = tab
        self.clear_preview_detailed_balance_block(tab.tab_id)
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
            self.cancel_pending_preview_intent(tab.tab_id)
            return
        prepared_dsl = tab.prepared_dsl
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            self._reject(
                "Preview requires accepted Reactions with retained DSL analysis."
            )
            return

        captured: list[tuple[PreviewResultKey, str]] = []
        self._stop_preview_timers()
        self._preview_capture = None
        for set_id in dirty_targets:
            key = self._simulation_jobs.begin_preview_edit(
                self._preview_identity(tab, projection, set_id)
            )
            if key is not None:
                captured.append((key, tab.initial_condition_set(set_id).name))
        if not captured:
            self.cancel_pending_preview_intent(tab.tab_id)
            return
        self._preview_capture = _PreviewCapture(
            tab_id=tab.tab_id,
            prepared_dsl=prepared_dsl,
            projection=projection,
            targets=tuple(captured),
            debounce_kind=str(debounce_kind),
        )
        self._preview_timer(debounce_kind).start()

    def _preview_timer(self, debounce_kind: str) -> QtCore.QTimer:
        timers = {
            "parameter": self._parameter_timer,
            "equilibrium": self._equilibrium_timer,
            "species": self._species_timer,
        }
        timer = timers.get(str(debounce_kind))
        if timer is None:
            raise ValueError(
                f"Unknown Preview debounce kind {debounce_kind!r}."
            )
        return timer

    @staticmethod
    def _preview_detailed_balance_request(
        tab: MechanismTabDocument,
        error: UnresolvedWegscheiderCyclicityError,
        *,
        reactions_text: str,
        source_energy_unit: str,
    ) -> DetailedBalanceCorrectionRequest:
        source = str(reactions_text)
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
            issue=str(error),
            temperature_K=float(tab.scientific_settings.temperature),
            report=WegscheiderCyclicityReport(cycles=tuple(error.cycles)),
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
            self._registry.active_runtime.reactions_draft
            or self._registry.active_tab.mechanism_source
        )
        if (
            self._registry.active_tab_id != pending.tab_id
            or current_visible_source.reactions_text
            not in {pending.source, corrected}
        ):
            self.errorRaised.emit(
                "Preview detailed-balance source is no longer current."
            )
            return False
        self._verify_preview_detailed_balance_source(
            corrected,
            temperature_K=pending.temperature_K,
        )
        self._pending_detailed_balance_request = None
        # Integration commits this exact typed outcome. No Preview request is
        # scheduled here or by the editor modal.
        self.previewDetailedBalanceCommitRequested.emit(outcome)
        return True

    def clear_preview_detailed_balance_block(
        self,
        tab_id: str | None = None,
    ) -> bool:
        pending = self._pending_detailed_balance_request
        if pending is None:
            return False
        target = None if tab_id is None else str(tab_id)
        if target is not None and pending.tab_id != target:
            return False
        self._pending_detailed_balance_request = None
        self.previewDetailedBalanceBlockCleared.emit(pending.tab_id)
        return True

    @staticmethod
    def _verify_preview_detailed_balance_source(
        reactions_text: str,
        *,
        temperature_K: float,
    ) -> None:
        from kindred.core.simulator.parameter_algebra import (
            apply_parameter_algebra_spec_to_mechanism,
        )
        from kindred.core.simulator.wegscheider_symbolic import (
            validate_wegscheider_cyclicity_resolved,
        )
        from kindred.core.units import UnitsModel

        source = str(reactions_text or "")
        prepared_dsl = prepare_dsl_request_set(
            source,
            initials={},
            units=UnitsModel(temperature_K=float(temperature_K)),
        )
        mechanism = prepared_dsl.mechanism_for_execution()
        apply_parameter_algebra_spec_to_mechanism(
            prepared_dsl.parameter_algebra_spec_for_execution(),
            mechanism=mechanism,
            require_mutable=False,
        )
        validate_wegscheider_cyclicity_resolved(mechanism)

    def _preview_inputs_valid(self, targets: tuple[str, ...]) -> bool:
        runtime = self._registry.active_runtime
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
    def _submit_preview(self) -> None:
        from kindred.gui.controllers.simulation_job_owner import (
            PreviewIntent,
            build_preview_target,
        )

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
            )
            for key, set_name in capture.targets
            if self._simulation_jobs.preview_key_is_current(key)
        )
        if targets:
            self._simulation_jobs.request_preview(
                PreviewIntent(tab_id=capture.tab_id, targets=targets)
            )

    def has_slider_draft(self, tab_id: str | None = None) -> bool:
        runtime = (
            self._registry.active_runtime
            if tab_id is None
            else self._registry.tab_runtime(tab_id)
        )
        return bool(
            runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        )

    def suspend_preview_disposition(self) -> _SuspendedPreviewDisposition:
        """Fence one nested Run/Fit disposition without changing its draft."""

        if self._suspended_preview_disposition is not None:
            raise RuntimeError("A Slider Preview disposition is already active.")
        self._stop_preview_timers()
        transaction = _SuspendedPreviewDisposition(
            capture=self._preview_capture,
        )
        self._preview_capture = None
        self._suspended_preview_disposition = transaction
        self._simulation_jobs.drop_pending_preview()
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
            self._preview_timer(
                transaction.capture.debounce_kind
            ).start()

    def discard_preview_disposition(
        self,
        transaction: _SuspendedPreviewDisposition,
    ) -> None:
        """End one disposition fence without restoring its Preview capture."""

        if transaction is not self._suspended_preview_disposition:
            raise ValueError("Slider Preview disposition is not current.")
        self._suspended_preview_disposition = None
        self._preview_capture = None
        self._simulation_jobs.drop_pending_preview()

    def discard_slider_draft(self, tab_id: str | None = None) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        target_tab_id = (
            self._registry.active_tab_id if tab_id is None else str(tab_id)
        )
        runtime = self._registry.tab_runtime(target_tab_id)
        parameter_dirty = bool(runtime.slider_parameter_draft)
        changed = bool(
            runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        )
        if not changed:
            return False
        tab = (
            self._registry.active_tab
            if target_tab_id == self._registry.active_tab_id
            else self._tab_snapshot_by_id.get(target_tab_id)
        )
        if tab is None:
            return self._reject(
                f"Cannot discard Slider draft for unavailable tab {target_tab_id!r}."
            )
        affected = (
            tuple(item.set_id for item in tab.initial_condition_sets)
            if parameter_dirty
            else tuple(runtime.slider_species_drafts_by_set_id)
        )
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self.cancel_pending_preview_intent(target_tab_id)
        self.clear_preview_detailed_balance_block(target_tab_id)
        self._reconcile_canonical_set_ids(tab, affected)
        if target_tab_id == self._registry.active_tab_id:
            self.render_active()
        self._emit_draft_state(target_tab_id)
        return changed

    def cancel_pending_preview_intent(self, tab_id: str | None = None) -> None:
        target = None if tab_id is None else str(tab_id)
        capture = self._preview_capture
        if capture is not None and (target is None or capture.tab_id == target):
            self._stop_preview_timers()
            self._preview_capture = None
        self._simulation_jobs.drop_pending_preview(target)

    def apply(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        runtime = self._registry.active_runtime
        parameter_updates = dict(runtime.slider_parameter_draft)
        species_updates = {
            set_id: dict(values)
            for set_id, values in runtime.slider_species_drafts_by_set_id.items()
            if values
        }
        if not parameter_updates and not species_updates:
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
        runtime = self._registry.active_runtime
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
        delta = self._registry.replace_active_tab(candidate)
        if delta is None:
            return False
        self.cancel_pending_preview_intent(tab.tab_id)
        self._consume_document_delta(delta)
        self.clear_preview_detailed_balance_block(tab.tab_id)
        runtime.slider_parameter_draft.clear()
        runtime.slider_species_drafts_by_set_id.clear()
        self.render_active()
        self._emit_draft_state(tab.tab_id)
        return True

    def reset(self) -> bool:
        if self._suspended_preview_disposition is not None:
            return False
        tab = self._registry.active_tab
        runtime = self._registry.active_runtime
        targets = self.resolve_target_set_ids()
        parameter_dirty = bool(runtime.slider_parameter_draft)
        if not parameter_dirty and not any(
            runtime.slider_species_drafts_by_set_id.get(set_id)
            for set_id in targets
        ):
            return False
        changed = parameter_dirty
        runtime.slider_parameter_draft.clear()
        for set_id in targets:
            changed = bool(
                runtime.slider_species_drafts_by_set_id.pop(set_id, None)
            ) or changed
        self.cancel_pending_preview_intent(tab.tab_id)
        self.clear_preview_detailed_balance_block(tab.tab_id)
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
        self.render_active()
        self._emit_draft_state(tab.tab_id)
        return changed

    def _preview_identity(
        self,
        tab: MechanismTabDocument,
        projection: ParameterSliderProjection,
        set_id: str,
    ) -> PreviewResultIdentity:
        runtime = self._registry.tab_runtime(tab.tab_id)
        parameter_values = projection.values_dict()
        parameter_values.update(runtime.slider_parameter_draft)
        initial_values = dict(projected_initial_values(tab, set_id))
        initial_values.update(
            runtime.slider_species_drafts_by_set_id.get(set_id, {})
        )
        return PreviewResultIdentity(
            tab_id=tab.tab_id,
            set_id=set_id,
            mechanism_source=tab.mechanism_source,
            analysis_identity=projected_analysis_identity(tab),
            standard_concentration_identity=standard_concentration_identity(
                tab.prepared_dsl
            ),
            parameter_values=tuple(parameter_values.items()),
            initial_values=tuple(initial_values.items()),
            execution_settings=tab.resolved_execution_settings(set_id),
            scientific_settings=tab.scientific_settings,
        )

    def _consume_document_delta(self, delta: ScientificDocumentDelta) -> None:
        if not isinstance(delta, ScientificDocumentDelta):
            raise TypeError("delta must be ScientificDocumentDelta.")
        for tab_id, set_ids in delta.affected_targets_by_tab:
            if set_ids:
                self._simulation_jobs.deauthorize_targets(tab_id, set_ids)

    @staticmethod
    def _explicit_key(
        tab: MechanismTabDocument,
        set_id: str,
    ) -> ExplicitResultKey:
        return ExplicitResultKey(
            tab_id=tab.tab_id,
            set_id=set_id,
            mechanism_source=tab.mechanism_source,
            analysis_identity=projected_analysis_identity(tab),
            standard_concentration_identity=standard_concentration_identity(
                tab.prepared_dsl
            ),
            scientific_identity=scientific_identity(tab.prepared_dsl),
            initial_values=projected_initial_values(tab, set_id),
            execution_settings=tab.resolved_execution_settings(set_id),
            scientific_settings=tab.scientific_settings,
        )

    def _emit_draft_state(self, tab_id: str) -> None:
        slider_dirty = self.has_slider_draft(tab_id)
        self.draftStateChanged.emit(
            str(tab_id),
            self._registry.tab_runtime(tab_id).has_dirty_drafts,
        )
        self.sliderDraftStateChanged.emit(str(tab_id), slider_dirty)

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

    def _restore_rejected_slider_action(self) -> bool:
        self.render_active(preserve_visibility=True)
        return False

    def _reject_slider_action(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return self._restore_rejected_slider_action()

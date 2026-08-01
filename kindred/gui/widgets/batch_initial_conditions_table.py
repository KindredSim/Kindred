"""Snapshot projection and exact-row intents for Initial Conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_document import (
    InitialConditionSet,
    MechanismTabDocument,
    SimulationExecutionSettings,
    TabScientificSettings,
    new_set_id,
    next_set_name,
)
from kindred.core.project_presentation_state import ProjectPresentationState
from kindred.core.result_repository import ExplicitResultKey
from kindred.gui.widgets.solver_settings import SolverSettingsDialog
from kindred.gui.workspace_tab_registry import ScientificDocumentDelta

__all__ = [
    "BatchInitialConditionsTableModel",
    "BatchInitialConditionsTableView",
    "InitialConditionsTableActions",
    "projected_initial_values",
    "projected_analysis_identity",
    "projected_species_names",
    "resolve_live_slider_target_set_ids",
]


_INVALID_BRUSH = QtGui.QBrush(QtGui.QColor(255, 210, 210))
_RUN_GEAR = "\N{GEAR}"


@dataclass(frozen=True, slots=True)
class _InitialConditionsRenderProjection:
    tab: MechanismTabDocument
    presentation: ProjectPresentationState
    invalid_raw_by_set: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    species_names: tuple[str, ...]
    selected_run_set_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProjectedRow:
    set_id: str
    name: str
    canonical_values: tuple[tuple[str, float], ...]
    invalid_raw_values: tuple[tuple[str, str], ...]
    custom_run_settings: bool
    plot_requested: bool
    slider_pinned: bool

    def display_value(self, species: str) -> str:
        invalid = dict(self.invalid_raw_values)
        if species in invalid:
            return invalid[species]
        return f"{dict(self.canonical_values).get(species, 0.0):.12g}"


class BatchInitialConditionsTableModel(QtCore.QAbstractTableModel):
    """Resettable view projection; canonical values live in ProjectDocument."""

    initialValueEditRequested = QtCore.Signal(str, str, str)
    setNameEditRequested = QtCore.Signal(str, str)
    plotMembershipChangeRequested = QtCore.Signal(str, bool)
    sliderPinChangeRequested = QtCore.Signal(str, bool)
    customRunSettingsToggleRequested = QtCore.Signal(str, bool)
    runSettingsEditRequested = QtCore.Signal(str)
    snapshotRendered = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: tuple[_ProjectedRow, ...] = ()
        self._species: tuple[str, ...] = ()
        self._binding_generation = 1
        self._snapshot_render_timer = QtCore.QTimer(self)
        self._snapshot_render_timer.setSingleShot(True)
        self._snapshot_render_timer.timeout.connect(self.snapshotRendered)

    @property
    def binding_generation(self) -> int:
        return self._binding_generation

    def render_snapshot(
        self,
        tab: MechanismTabDocument,
        presentation: ProjectPresentationState,
        *,
        invalid_raw_by_set: dict[str, dict[str, str]] | None = None,
        species_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("tab must be a MechanismTabDocument.")
        if not isinstance(presentation, ProjectPresentationState):
            raise TypeError("presentation must be ProjectPresentationState.")
        if species_names is None:
            species = projected_species_names(tab)
        else:
            species = tuple(
                dict.fromkeys(str(item) for item in species_names if str(item))
            )
        override_ids = {
            set_id for set_id, _settings in tab.run_overrides_by_set_id
        }
        invalid = invalid_raw_by_set or {}
        plot_ids = set(presentation.plot_set_ids)
        pinned_ids = set(presentation.pinned_slider_set_ids)
        rows = tuple(
            _ProjectedRow(
                set_id=initial_set.set_id,
                name=initial_set.name,
                canonical_values=initial_set.values,
                invalid_raw_values=tuple(
                    (str(species_name), str(raw))
                    for species_name, raw
                    in invalid.get(initial_set.set_id, {}).items()
                ),
                custom_run_settings=initial_set.set_id in override_ids,
                plot_requested=initial_set.set_id in plot_ids,
                slider_pinned=initial_set.set_id in pinned_ids,
            )
            for initial_set in tab.initial_condition_sets
        )
        self.beginResetModel()
        try:
            self._binding_generation += 1
            self._rows = rows
            self._species = species
        finally:
            self.endResetModel()
        self._snapshot_render_timer.start(0)

    def species_names(self) -> tuple[str, ...]:
        return self._species

    def set_ids(self) -> tuple[str, ...]:
        return tuple(row.set_id for row in self._rows)

    def set_id_for_row(self, row: int) -> str | None:
        row_i = int(row)
        if 0 <= row_i < len(self._rows):
            return self._rows[row_i].set_id
        return None

    def set_id_for_live_index(self, index: QtCore.QModelIndex) -> str | None:
        if (
            not index.isValid()
            or index.model() is not self
            or int(index.internalId()) != self._binding_generation
        ):
            return None
        return self.set_id_for_row(index.row())

    def index(
        self,
        row: int,
        column: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> QtCore.QModelIndex:  # noqa: N802
        if parent.isValid() or not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        return self.createIndex(int(row), int(column), self._binding_generation)

    def parent(self, _index: QtCore.QModelIndex) -> QtCore.QModelIndex:  # noqa: N802
        return QtCore.QModelIndex()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._species) + 5

    def slider_column(self) -> int:
        return len(self._species) + 1

    def plot_column(self) -> int:
        return len(self._species) + 2

    def custom_column(self) -> int:
        return len(self._species) + 3

    def run_gear_column(self) -> int:
        return len(self._species) + 4

    def is_numeric_column(self, column: int) -> bool:
        return 1 <= int(column) <= len(self._species)

    def species_for_column(self, column: int) -> str | None:
        return (
            self._species[int(column) - 1]
            if self.is_numeric_column(column)
            else None
        )

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):  # noqa: N802
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Vertical:
            return str(int(section) + 1)
        if section == 0:
            return "Set"
        if self.is_numeric_column(section):
            return f"{self._species[section - 1]} (M)"
        return {
            self.slider_column(): "Slider",
            self.plot_column(): "Plot",
            self.custom_column(): "Custom",
            self.run_gear_column(): "",
        }.get(int(section), "")

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        set_id = self.set_id_for_live_index(index)
        if set_id is None:
            return QtCore.Qt.ItemFlag.NoItemFlags
        row = self._rows[index.row()]
        column = int(index.column())
        base = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if column in {self.slider_column(), self.plot_column()}:
            return base | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if column == self.custom_column():
            return base | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if column == self.run_gear_column():
            return base if row.custom_run_settings else (
                QtCore.Qt.ItemFlag.ItemIsSelectable
            )
        if column <= len(self._species):
            return base | QtCore.Qt.ItemFlag.ItemIsEditable
        return base

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if self.set_id_for_live_index(index) is None:
            return None
        row = self._rows[index.row()]
        column = int(index.column())
        if role == QtCore.Qt.ItemDataRole.CheckStateRole:
            if column == self.slider_column():
                checked = row.slider_pinned
            elif column == self.plot_column():
                checked = row.plot_requested
            elif column == self.custom_column():
                checked = row.custom_run_settings
            else:
                return None
            return (
                QtCore.Qt.CheckState.Checked
                if checked
                else QtCore.Qt.CheckState.Unchecked
            )
        if role in {
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        }:
            if column == 0:
                return row.name
            if self.is_numeric_column(column):
                return row.display_value(self._species[column - 1])
            if column == self.run_gear_column() and row.custom_run_settings:
                return _RUN_GEAR
            return ""
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignCenter)
        if (
            role == QtCore.Qt.ItemDataRole.BackgroundRole
            and self.is_numeric_column(column)
            and self._species[column - 1] in dict(row.invalid_raw_values)
        ):
            return _INVALID_BRUSH
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if column == self.slider_column():
                return "Pin this exact set as a Slider target."
            if column == self.plot_column():
                return "Show this exact set in the shared plot."
            if column == self.custom_column():
                return (
                    "Use one complete four-field override for this set"
                    if row.custom_run_settings
                    else "Create one complete four-field override for this set"
                )
            if column == self.run_gear_column() and row.custom_run_settings:
                return "Edit this set's complete four-field override"
        return None

    def setData(
        self,
        index: QtCore.QModelIndex,
        value,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:  # noqa: N802
        set_id = self.set_id_for_live_index(index)
        if set_id is None:
            return False
        column = int(index.column())
        if role == QtCore.Qt.ItemDataRole.CheckStateRole:
            checked_value = getattr(
                QtCore.Qt.CheckState.Checked, "value", QtCore.Qt.CheckState.Checked
            )
            checked = int(getattr(value, "value", value)) == int(checked_value)
            if column == self.slider_column():
                self.sliderPinChangeRequested.emit(set_id, checked)
                return True
            if column == self.plot_column():
                self.plotMembershipChangeRequested.emit(set_id, checked)
                return True
            if column == self.custom_column():
                self.customRunSettingsToggleRequested.emit(set_id, checked)
                return True
            return False
        if role != QtCore.Qt.ItemDataRole.EditRole:
            return False
        if column == 0:
            self.setNameEditRequested.emit(set_id, str(value))
            return True
        species = self.species_for_column(column)
        if species is not None:
            self.initialValueEditRequested.emit(set_id, species, str(value))
            return True
        return False

    def request_run_settings_for_index(self, index: QtCore.QModelIndex) -> bool:
        set_id = self.set_id_for_live_index(index)
        if set_id is None or int(index.column()) != self.run_gear_column():
            return False
        row = self._rows[index.row()]
        if not row.custom_run_settings:
            return False
        self.runSettingsEditRequested.emit(set_id)
        return True


class BatchInitialConditionsTableView(QtWidgets.QTableView):
    pasteRequested = QtCore.Signal(int, int, str)
    pasteError = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.setAlternatingRowColors(True)

    def selected_set_ids(self) -> tuple[str, ...]:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return ()
        rows = sorted({index.row() for index in self.selectionModel().selectedRows()})
        return tuple(
            set_id
            for row in rows
            if (set_id := model.set_id_for_row(row)) is not None
        )

    def settle_active_editor(self) -> None:
        """Synchronously submit the active delegate before an action boundary."""
        editor = QtWidgets.QApplication.focusWidget()
        if editor is not None and editor is not self and self.isAncestorOf(editor):
            editor.clearFocus()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            model = self.model()
            index = self.currentIndex()
            if (
                isinstance(model, BatchInitialConditionsTableModel)
                and index.isValid()
                and model.is_numeric_column(index.column())
            ):
                clipboard = QtWidgets.QApplication.clipboard()
                self.pasteRequested.emit(
                    int(index.row()),
                    int(index.column()),
                    str(clipboard.text()),
                )
                event.accept()
                return
        super().keyPressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.indexAt(event.position().toPoint())
        model = self.model()
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and isinstance(model, BatchInitialConditionsTableModel)
            and model.request_run_settings_for_index(index)
        ):
            event.accept()
            return
        super().mouseReleaseEvent(event)


def resolve_live_slider_target_set_ids(
    *,
    table: BatchInitialConditionsTableView,
    pinned_set_ids: Iterable[str],
    existing_set_ids: Iterable[str],
) -> tuple[str, ...]:
    """Resolve exact pins, else the live model-bound current row, else none."""

    existing = {str(set_id) for set_id in existing_set_ids}
    pinned = tuple(
        set_id
        for set_id in dict.fromkeys(str(item) for item in pinned_set_ids)
        if set_id in existing
    )
    if pinned:
        return pinned
    model = table.model()
    if not isinstance(model, BatchInitialConditionsTableModel):
        return ()
    index = table.currentIndex()
    if not index.isValid() or index.model() is not model:
        return ()
    set_id = model.set_id_for_live_index(index)
    return (set_id,) if set_id in existing else ()


def projected_species_names(tab: MechanismTabDocument) -> tuple[str, ...]:
    """Return accepted-mechanism species without materializing absent zeros."""

    prepared_dsl = tab.prepared_dsl
    if prepared_dsl is None:
        return ()
    from kindred.core.simulator.dsl_parse import PreparedDslRequestSet

    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise TypeError("tab prepared_dsl must be PreparedDslRequestSet.")
    return tuple(prepared_dsl.analysis.species_names)


def projected_analysis_identity(tab: MechanismTabDocument) -> str:
    """Return the exact retained accepted analysis identity."""

    from kindred.core.simulator.dsl_parse import PreparedDslRequestSet

    prepared_dsl = tab.prepared_dsl
    if not isinstance(prepared_dsl, PreparedDslRequestSet):
        raise ValueError(
            f"Mechanism tab {tab.tab_id!r} has no accepted DSL analysis."
        )
    identity = prepared_dsl.analysis.identity
    if type(identity) is not str or not identity or identity != identity.strip():
        raise ValueError(
            f"Mechanism tab {tab.tab_id!r} has an invalid analysis identity."
        )
    return identity


def projected_initial_values(
    tab: MechanismTabDocument,
    set_id: str,
) -> tuple[tuple[str, float], ...]:
    """Project absent accepted species as exact zero without changing the set."""

    canonical = tab.initial_condition_set(set_id).values_dict()
    return tuple(
        (species, float(canonical.get(species, 0.0)))
        for species in projected_species_names(tab)
    )


class InitialConditionsTableActions(QtCore.QObject):
    """Direct active-tab commits plus transient invalid/selection ownership."""

    errorRaised = QtCore.Signal(str)
    activeSnapshotChanged = QtCore.Signal()
    invalidCellStateChanged = QtCore.Signal(str, str, str, bool)
    selectedRunSetIdsChanged = QtCore.Signal(object)

    def __init__(
        self,
        *,
        registry,
        model: BatchInitialConditionsTableModel,
        table: BatchInitialConditionsTableView,
        simulation_jobs,
        species_for_tab: Callable[[MechanismTabDocument], Sequence[str]] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._model = model
        self._table = table
        self._simulation_jobs = simulation_jobs
        self._species_for_tab = species_for_tab or projected_species_names
        self._rendering = False
        model.initialValueEditRequested.connect(self.accept_value_edit)
        model.setNameEditRequested.connect(self.accept_name_edit)
        model.sliderPinChangeRequested.connect(self.set_slider_pinned)
        model.customRunSettingsToggleRequested.connect(
            self.toggle_custom_run_settings
        )
        model.runSettingsEditRequested.connect(self.edit_custom_run_settings)
        table.pasteRequested.connect(self.accept_paste)
        table.selectionModel().selectionChanged.connect(self._capture_run_selection)
        table.selectionModel().currentChanged.connect(self._capture_current_set)

    def prepare_active_render(self) -> _InitialConditionsRenderProjection:
        """Prepare the complete table projection without touching widgets."""

        tab = self._registry.active_tab
        runtime = self._registry.active_runtime
        return _InitialConditionsRenderProjection(
            tab=tab,
            presentation=self._registry.active_presentation,
            invalid_raw_by_set=tuple(
                (
                    str(set_id),
                    tuple(
                        (str(species), str(raw))
                        for species, raw in values.items()
                    ),
                )
                for set_id, values
                in runtime.invalid_initial_text_by_set_id.items()
            ),
            species_names=tuple(self._species_for_tab(tab)),
            selected_run_set_ids=tuple(runtime.selected_run_set_ids),
        )

    def apply_render_projection(
        self,
        projection: _InitialConditionsRenderProjection,
    ) -> None:
        """Apply one fully prepared table projection."""

        if not isinstance(projection, _InitialConditionsRenderProjection):
            raise TypeError(
                "projection must be _InitialConditionsRenderProjection."
            )
        selection = self._table.selectionModel()
        selection_blocker = QtCore.QSignalBlocker(selection)
        self._rendering = True
        try:
            self._model.render_snapshot(
                projection.tab,
                projection.presentation,
                invalid_raw_by_set={
                    set_id: dict(values)
                    for set_id, values in projection.invalid_raw_by_set
                },
                species_names=projection.species_names,
            )
            selection.clearSelection()
            selected = set(projection.selected_run_set_ids)
            current_set_id = projection.presentation.current_set_id
            for row, set_id in enumerate(self._model.set_ids()):
                index = self._model.index(row, 0)
                if set_id in selected:
                    selection.select(
                        index,
                        QtCore.QItemSelectionModel.SelectionFlag.Select
                        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )
                if set_id == current_set_id:
                    selection.setCurrentIndex(
                        index,
                        QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
        finally:
            self._rendering = False
            del selection_blocker

    def render_active(self) -> None:
        self.apply_render_projection(self.prepare_active_render())

    def settle_active_editor(self) -> None:
        self._table.settle_active_editor()

    def selected_run_set_ids(self) -> tuple[str, ...]:
        return tuple(self._registry.active_runtime.selected_run_set_ids)

    def invalid_cells(
        self,
        set_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        runtime = self._registry.active_runtime
        allowed = None if set_ids is None else {str(item) for item in set_ids}
        return tuple(
            (set_id, species, raw)
            for set_id, values in runtime.invalid_initial_text_by_set_id.items()
            if allowed is None or set_id in allowed
            for species, raw in values.items()
        )

    def has_invalid_cells(self, set_ids: Iterable[str] | None = None) -> bool:
        return bool(self.invalid_cells(set_ids))

    def discard_invalid_cells(self, set_ids: Iterable[str] | None = None) -> bool:
        runtime = self._registry.active_runtime
        allowed = None if set_ids is None else {str(item) for item in set_ids}
        changed = False
        recovered_set_ids: list[str] = []
        for set_id in tuple(runtime.invalid_initial_text_by_set_id):
            if allowed is not None and set_id not in allowed:
                continue
            values = runtime.invalid_initial_text_by_set_id.pop(set_id)
            changed = changed or bool(values)
            if values:
                recovered_set_ids.append(set_id)
            for species in values:
                self.invalidCellStateChanged.emit(
                    self._registry.active_tab_id,
                    set_id,
                    species,
                    False,
                )
        if changed:
            self._reconcile_canonical_sets(
                self._registry.active_tab,
                recovered_set_ids,
            )
            self.render_active()
            self.activeSnapshotChanged.emit()
        return changed

    @QtCore.Slot(str, str, str)
    def accept_value_edit(self, set_id: str, species: str, raw_text: str) -> bool:
        tab = self._registry.active_tab
        initial_set = tab.initial_condition_set(set_id)
        try:
            value = float(raw_text)
        except (TypeError, ValueError):
            value = math.nan
        invalid = self._registry.active_runtime.invalid_initial_text_by_set_id
        was_invalid = species in invalid.get(set_id, {})
        if not math.isfinite(value) or value < 0.0:
            raw = str(raw_text)
            if invalid.get(set_id, {}).get(species) == raw:
                return False
            invalid.setdefault(set_id, {})[species] = str(raw_text)
            if not was_invalid:
                self._simulation_jobs.deauthorize_targets(
                    tab.tab_id,
                    (set_id,),
                )
            self.invalidCellStateChanged.emit(
                tab.tab_id, set_id, species, True
            )
            self.render_active()
            self.activeSnapshotChanged.emit()
            return False

        values = initial_set.values_dict()
        canonical = float(values.get(str(species), 0.0))
        changed = value != canonical
        delta: ScientificDocumentDelta | None = None
        if changed:
            values[str(species)] = value
            delta = self._registry.replace_active_initial_condition_set(
                initial_set.with_values(values),
            )
        invalid_cleared = self._clear_invalid_value(set_id, species)
        if was_invalid and not changed:
            self._reconcile_canonical_sets(tab, (set_id,))
        self.render_active()
        if delta is not None:
            self._consume_document_delta(delta)
        elif invalid_cleared:
            self.activeSnapshotChanged.emit()
        return changed

    @QtCore.Slot(int, int, str)
    def accept_paste(self, start_row: int, start_column: int, text: str) -> bool:
        rows = str(text).splitlines()
        if not rows:
            return False
        cells = [row.split("\t") for row in rows]
        width = max((len(row) for row in cells), default=0)
        if width <= 0 or any(len(row) != width for row in cells):
            return self._reject("Paste must be a rectangular table.")
        if (
            start_row < 0
            or start_column < 1
            or start_row + len(cells) > self._model.rowCount()
            or start_column + width - 1 > len(self._model.species_names())
        ):
            return self._reject(
                "Paste must fit entirely within existing concentration cells."
            )
        parsed: list[list[float]] = []
        try:
            for row in cells:
                values = [float(raw.strip()) for raw in row]
                if any(
                    not math.isfinite(value) or value < 0.0
                    for value in values
                ):
                    raise ValueError
                parsed.append(values)
        except (TypeError, ValueError):
            return self._reject(
                "Every pasted concentration must be finite and nonnegative."
            )

        tab = self._registry.active_tab
        invalid_before = {
            set_id: frozenset(values)
            for set_id, values
            in self._registry.active_runtime.invalid_initial_text_by_set_id.items()
        }
        candidate_sets = list(tab.initial_condition_sets)
        changed_ids: list[str] = []
        cleared_cells: list[tuple[str, str]] = []
        species_names = self._model.species_names()
        for row_offset, values_to_paste in enumerate(parsed):
            row_index = start_row + row_offset
            initial_set = candidate_sets[row_index]
            values = initial_set.values_dict()
            row_changed = False
            for column_offset, value in enumerate(values_to_paste):
                species = species_names[start_column + column_offset - 1]
                cleared_cells.append((initial_set.set_id, species))
                if float(values.get(species, 0.0)) != value:
                    values[species] = value
                    row_changed = True
            if row_changed:
                candidate_sets[row_index] = initial_set.with_values(values)
                changed_ids.append(initial_set.set_id)

        delta: ScientificDocumentDelta | None = None
        if changed_ids:
            delta = self._registry.replace_active_tab(
                tab.with_initial_condition_sets(tuple(candidate_sets)),
            )
        invalid_cleared = False
        for set_id, species in cleared_cells:
            invalid_cleared = (
                self._clear_invalid_value(set_id, species)
                or invalid_cleared
            )
        canonical_recovery_ids = tuple(
            dict.fromkeys(
                set_id
                for set_id, species in cleared_cells
                if species in invalid_before.get(set_id, ())
                and set_id not in changed_ids
            )
        )
        if canonical_recovery_ids:
            self._reconcile_canonical_sets(tab, canonical_recovery_ids)
        self.render_active()
        if delta is not None:
            self._consume_document_delta(delta)
        elif invalid_cleared:
            self.activeSnapshotChanged.emit()
        return bool(delta is not None or invalid_cleared)

    @QtCore.Slot(str, str)
    def accept_name_edit(self, set_id: str, name: str) -> bool:
        normalized = str(name).strip()
        tab = self._registry.active_tab
        if not normalized:
            return self._reject("Initial-condition set names must be nonblank.")
        if any(
            item.set_id != set_id
            and item.name.casefold() == normalized.casefold()
            for item in tab.initial_condition_sets
        ):
            return self._reject(
                f"An initial-condition set named {normalized!r} already exists."
            )
        initial_set = tab.initial_condition_set(set_id)
        delta = self._registry.replace_active_initial_condition_set(
            replace(initial_set, name=normalized),
        )
        self.render_active()
        return self._consume_document_delta(delta)

    def add_set(self) -> InitialConditionSet | None:
        tab = self._registry.active_tab
        created = InitialConditionSet(
            set_id=new_set_id(),
            name=next_set_name(item.name for item in tab.initial_condition_sets),
            values=tuple(
                (str(species), 0.0) for species in self._species_for_tab(tab)
            ),
        )
        delta = self._registry.mutate_active_tab(
            lambda current: current.add_initial_condition_set(created),
        )
        self.render_active()
        return created if self._consume_document_delta(delta) else None

    def move_selected(self, delta: int) -> tuple[str, ...]:
        self.settle_active_editor()
        selected = self.selected_run_set_ids()
        if not selected or int(delta) == 0:
            return selected
        tab = self._registry.active_tab
        ordered = list(tab.initial_condition_sets)
        active = {
            index for index, item in enumerate(ordered) if item.set_id in set(selected)
        }
        direction = -1 if int(delta) < 0 else 1
        for _ in range(abs(int(delta))):
            traversal = (
                sorted(active) if direction < 0 else sorted(active, reverse=True)
            )
            for row in traversal:
                destination = row + direction
                if (
                    destination < 0
                    or destination >= len(ordered)
                    or destination in active
                ):
                    continue
                ordered[row], ordered[destination] = ordered[destination], ordered[row]
                active.remove(row)
                active.add(destination)
        delta_result = self._registry.mutate_active_tab(
            lambda current: current.with_initial_condition_sets(tuple(ordered)),
        )
        self.render_active()
        self._consume_document_delta(delta_result)
        return tuple(ordered[row].set_id for row in sorted(active))

    def delete_selected(self) -> bool:
        self.settle_active_editor()
        targets = self.selected_run_set_ids()
        if not targets:
            return False
        delta = self._registry.remove_initial_condition_sets(targets)
        self.render_active()
        return self._consume_document_delta(delta)

    @QtCore.Slot(str, bool)
    def set_plot_requested(self, set_id: str, requested: bool) -> bool:
        state = self._registry.active_presentation
        ids = [item for item in state.plot_set_ids if item != set_id]
        if requested:
            ids.append(set_id)
        next_state = replace(state, plot_set_ids=tuple(ids))
        if next_state == state:
            return False
        self._registry.set_presentation(self._registry.active_tab_id, next_state)
        self.render_active()
        self.activeSnapshotChanged.emit()
        return True

    @QtCore.Slot(str, bool)
    def set_slider_pinned(self, set_id: str, pinned: bool) -> bool:
        state = self._registry.active_presentation
        ids = [item for item in state.pinned_slider_set_ids if item != set_id]
        if pinned:
            ids.append(set_id)
        next_state = replace(state, pinned_slider_set_ids=tuple(ids))
        if next_state == state:
            return False
        self._registry.set_presentation(self._registry.active_tab_id, next_state)
        self.render_active()
        self.activeSnapshotChanged.emit()
        return True

    def fence_inheriting_run_defaults_edit(self) -> None:
        """Draft start is not an accepted scientific change and never fences."""

    def set_workspace_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> bool:
        tab = self._registry.active_tab
        if settings == tab.run_defaults:
            return False
        delta = self._registry.set_active_run_defaults(settings)
        self.render_active()
        return self._consume_document_delta(delta)

    def set_tab_scientific_settings(
        self,
        settings: TabScientificSettings,
    ) -> bool:
        tab = self._registry.active_tab
        if settings == tab.scientific_settings:
            return False
        delta = self._registry.set_active_scientific_settings(settings)
        self.render_active()
        return self._consume_document_delta(delta)

    @QtCore.Slot(str, bool)
    def toggle_custom_run_settings(self, set_id: str, checked: bool) -> bool:
        tab = self._registry.active_tab
        overrides = dict(tab.run_overrides_by_set_id)
        has_override = set_id in overrides
        if checked:
            if has_override:
                return True
            return self._open_settings_dialog(set_id, tab.run_defaults)
        if not has_override:
            return True
        answer = QtWidgets.QMessageBox.question(
            self._table,
            "Use Tab Defaults",
            "Remove this set's complete custom Run settings and restore live "
            "inheritance from the tab defaults?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            self.render_active()
            return False
        return self.set_custom_run_settings(set_id, None)

    @QtCore.Slot(str)
    def edit_custom_run_settings(self, set_id: str) -> bool:
        tab = self._registry.active_tab
        override = dict(tab.run_overrides_by_set_id).get(set_id)
        if override is None:
            return False
        return self._open_settings_dialog(set_id, override)

    def _open_settings_dialog(
        self,
        set_id: str,
        seed: SimulationExecutionSettings,
    ) -> bool:
        tab = self._registry.active_tab
        initial_set = tab.initial_condition_set(set_id)
        dialog = SolverSettingsDialog(
            self._table,
            title=f"Custom Run Settings — {initial_set.name}",
        )
        dialog.set_execution_settings(seed)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self.render_active()
            return False
        return self.set_custom_run_settings(set_id, dialog.execution_settings())

    def set_custom_run_settings(
        self,
        set_id: str,
        settings: SimulationExecutionSettings | None,
    ) -> bool:
        tab = self._registry.active_tab
        current = dict(tab.run_overrides_by_set_id).get(set_id)
        if current == settings:
            return False
        delta = self._registry.set_active_run_override(set_id, settings)
        self.render_active()
        return self._consume_document_delta(delta)

    @QtCore.Slot()
    def _capture_run_selection(self, *_args) -> None:
        if self._rendering:
            return
        valid = set(self._model.set_ids())
        selected = tuple(
            set_id for set_id in self._table.selected_set_ids() if set_id in valid
        )
        self._registry.active_runtime.selected_run_set_ids = selected
        self.selectedRunSetIdsChanged.emit(selected)

    @QtCore.Slot()
    def _capture_current_set(self, *_args) -> None:
        if self._rendering:
            return
        set_id = self._model.set_id_for_live_index(self._table.currentIndex())
        if set_id is None:
            return
        state = self._registry.active_presentation
        if state.current_set_id != set_id:
            self._registry.set_presentation(
                self._registry.active_tab_id,
                replace(state, current_set_id=set_id),
            )
            self.activeSnapshotChanged.emit()

    def _clear_invalid_value(self, set_id: str, species: str) -> bool:
        invalid = self._registry.active_runtime.invalid_initial_text_by_set_id
        set_values = invalid.get(set_id)
        if set_values is None or species not in set_values:
            return False
        set_values.pop(species, None)
        if not set_values:
            invalid.pop(set_id, None)
        self.invalidCellStateChanged.emit(
            self._registry.active_tab_id,
            set_id,
            species,
            False,
        )
        return True

    def _consume_document_delta(
        self,
        delta: ScientificDocumentDelta | None,
    ) -> bool:
        if delta is None:
            return False
        if not isinstance(delta, ScientificDocumentDelta):
            raise TypeError("delta must be ScientificDocumentDelta.")
        for tab_id, set_ids in delta.affected_targets_by_tab:
            if set_ids:
                self._simulation_jobs.deauthorize_targets(tab_id, set_ids)
        self.activeSnapshotChanged.emit()
        return True

    def _reconcile_canonical_sets(
        self,
        tab: MechanismTabDocument,
        set_ids: Iterable[str],
    ) -> None:
        keys = tuple(
            ExplicitResultKey(
                tab_id=tab.tab_id,
                set_id=set_id,
                mechanism_source=tab.mechanism_source,
                analysis_identity=projected_analysis_identity(tab),
                initial_values=projected_initial_values(tab, set_id),
                execution_settings=tab.resolved_execution_settings(set_id),
                scientific_settings=tab.scientific_settings,
            )
            for set_id in dict.fromkeys(str(item) for item in set_ids)
        )
        self._simulation_jobs.reconcile_canonical_targets(keys)

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

"""Snapshot projection and exact-row intents for Initial Conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_document import (
    InitialConditionSet,
    MechanismTabDocument,
    SimulationExecutionSettings,
    canonical_species_names,
    new_set_id,
    next_set_name,
)
from kindred.core.project_presentation_state import ProjectPresentationState
from kindred.core.project_session import ProjectContentToken
from kindred.core.species import validate_concentration
from kindred.gui.widgets.solver_settings import SolverSettingsDialog

__all__ = [
    "BatchInitialConditionsTableModel",
    "BatchInitialConditionsTableView",
    "InitialConditionsTableActions",
    "resolve_live_slider_target_set_ids",
]


_INVALID_BRUSH = QtGui.QBrush(QtGui.QColor(255, 210, 210))
_RUN_GEAR = "\N{GEAR}"
_MAX_SET_COLUMN_WIDTH = 220
_MAX_SPECIES_COLUMN_WIDTH = 160
_SLIDER_COLUMN_WIDTH = 72
_PLOT_COLUMN_WIDTH = 56
_CUSTOM_COLUMN_WIDTH = 72
_RUN_GEAR_COLUMN_WIDTH = 36
_INVALID_EDITOR_STYLE = (
    "QLineEdit { background-color: rgb(255, 210, 210); "
    "color: rgb(80, 0, 0); }"
)


@dataclass(frozen=True, slots=True)
class _InitialConditionsRenderProjection:
    """Prepared incoming values for one admitted tab snapshot."""

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
    raw_values: tuple[tuple[str, str], ...]
    invalid_raw_values: tuple[tuple[str, str], ...]
    custom_run_settings: bool
    plot_requested: bool
    slider_pinned: bool

    def display_value(self, species: str) -> str:
        raw = dict(self.raw_values)
        if species in raw:
            return raw[species]
        # Preserve the exact accepted binary64 value in the editable
        # projection; formatting must never become a scientific write.
        return repr(float(dict(self.canonical_values).get(species, 0.0)))


class BatchInitialConditionsTableModel(QtCore.QAbstractTableModel):
    """Retained view projection; canonical values live in ProjectDocument."""

    initialValueEditRequested = QtCore.Signal(str, str, str)
    setNameEditRequested = QtCore.Signal(str, str)
    plotMembershipChangeRequested = QtCore.Signal(str, bool)
    sliderPinChangeRequested = QtCore.Signal(str, bool)
    customRunSettingsToggleRequested = QtCore.Signal(str, bool)
    runSettingsEditRequested = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: tuple[_ProjectedRow, ...] = ()
        self._species: tuple[str, ...] = ()
        self._binding_generation = 1

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
            species = canonical_species_names(tab)
        else:
            species = tuple(
                dict.fromkeys(str(item) for item in species_names if str(item))
            )
        admitted_sets = tab.initial_condition_sets
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
                raw_values=tuple(
                    (str(species_name), str(raw))
                    for species_name, raw
                    in invalid.get(initial_set.set_id, {}).items()
                ),
                invalid_raw_values=tuple(
                    (str(species_name), str(raw))
                    for species_name, raw
                    in invalid.get(initial_set.set_id, {}).items()
                    if not self._is_valid_concentration(raw, str(species_name))
                ),
                custom_run_settings=initial_set.set_id in override_ids,
                plot_requested=initial_set.set_id in plot_ids,
                slider_pinned=initial_set.set_id in pinned_ids,
            )
            for initial_set in admitted_sets
        )
        if rows == self._rows and species == self._species:
            return
        structure_changed = bool(
            species != self._species
            or tuple(row.set_id for row in rows)
            != tuple(row.set_id for row in self._rows)
        )
        if structure_changed:
            self.beginResetModel()
            try:
                self._binding_generation += 1
                self._rows = rows
                self._species = species
            finally:
                self.endResetModel()
        else:
            previous_rows = self._rows
            self._rows = rows
            self._emit_nonstructural_changes(previous_rows, rows)

    def _emit_nonstructural_changes(
        self,
        previous_rows: tuple[_ProjectedRow, ...],
        rows: tuple[_ProjectedRow, ...],
    ) -> None:
        """Notify only changed cells so an unrelated active editor survives."""

        for row_index, (previous, current) in enumerate(
            zip(previous_rows, rows, strict=True)
        ):
            changed_columns: set[int] = set()
            if previous.name != current.name:
                changed_columns.add(0)
            previous_invalid = dict(previous.invalid_raw_values)
            current_invalid = dict(current.invalid_raw_values)
            for species_index, species in enumerate(self._species, start=1):
                if (
                    previous.display_value(species)
                    != current.display_value(species)
                    or (species in previous_invalid)
                    != (species in current_invalid)
                ):
                    changed_columns.add(species_index)
            if previous.slider_pinned != current.slider_pinned:
                changed_columns.add(self.slider_column())
            if previous.plot_requested != current.plot_requested:
                changed_columns.add(self.plot_column())
            if previous.custom_run_settings != current.custom_run_settings:
                changed_columns.update(
                    (self.custom_column(), self.run_gear_column())
                )
            for column in sorted(changed_columns):
                index = self.index(row_index, column)
                self.dataChanged.emit(index, index, [])

    def species_names(self) -> tuple[str, ...]:
        return self._species

    def set_ids(self) -> tuple[str, ...]:
        return tuple(row.set_id for row in self._rows)

    def set_id_for_row(self, row: int) -> str | None:
        row_i = int(row)
        if 0 <= row_i < len(self._rows):
            return self._rows[row_i].set_id
        return None

    @staticmethod
    def _is_valid_concentration(raw: object, species: str) -> bool:
        try:
            validate_concentration(
                float(raw),
                field=f"initial concentration for {species}",
            )
        except (TypeError, ValueError, OverflowError):
            return False
        return True

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
                return "Show this exact set in the active mechanism plot."
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


class _InitialConditionsItemDelegate(QtWidgets.QStyledItemDelegate):
    """Keep rejected numeric edits available for correction."""

    def createEditor(  # noqa: N802 - Qt override
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtWidgets.QWidget:
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QtWidgets.QLineEdit):
            editor.textChanged.connect(
                lambda _text, target=editor: self._update_editor_style(target)
            )
            self._update_editor_style(editor)
        return editor

    def eventFilter(  # noqa: N802 - Qt override
        self,
        editor: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        if (
            isinstance(editor, QtWidgets.QLineEdit)
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
        ):
            index = self._index_for_editor(editor)
            model = index.model()
            if (
                index.isValid()
                and isinstance(model, BatchInitialConditionsTableModel)
                and model.is_numeric_column(index.column())
                and not self._is_valid_concentration(editor.text())
            ):
                self.commitData.emit(editor)
                self._update_editor_style(editor)
                event.accept()
                return True
        return super().eventFilter(editor, event)

    @staticmethod
    def _is_valid_concentration(text: str) -> bool:
        try:
            validate_concentration(
                float(text),
                field="initial concentration",
            )
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    def _index_for_editor(
        self,
        editor: QtWidgets.QLineEdit,
    ) -> QtCore.QModelIndex:
        view = self.parent()
        if not isinstance(view, BatchInitialConditionsTableView):
            return QtCore.QModelIndex()
        center = editor.mapTo(view.viewport(), editor.rect().center())
        return view.indexAt(center)

    def _update_editor_style(self, editor: QtWidgets.QLineEdit) -> None:
        index = self._index_for_editor(editor)
        editor.setStyleSheet(
            _INVALID_EDITOR_STYLE
            if index.isValid() and index.column() != 0
            and not self._is_valid_concentration(editor.text())
            else ""
        )


class BatchInitialConditionsTableView(QtWidgets.QTableView):
    pasteRequested = QtCore.Signal(int, int, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setItemDelegate(_InitialConditionsItemDelegate(self))
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
        self.horizontalHeader().setStretchLastSection(False)
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setAlternatingRowColors(True)

    def setModel(
        self, model: QtCore.QAbstractItemModel | None
    ) -> None:  # noqa: N802
        old_model = self.model()
        if isinstance(old_model, BatchInitialConditionsTableModel):
            try:
                old_model.modelReset.disconnect(self._apply_column_presentation)
            except (TypeError, RuntimeError):
                pass
        super().setModel(model)
        if isinstance(model, BatchInitialConditionsTableModel):
            model.modelReset.connect(self._apply_column_presentation)
        self._apply_column_presentation()

    def _apply_column_presentation(self) -> None:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return
        header = self.horizontalHeader()
        desired_order = [
            0,
            model.slider_column(),
            model.plot_column(),
            model.custom_column(),
            model.run_gear_column(),
            *range(1, len(model.species_names()) + 1),
        ]
        movable_before = header.sectionsMovable()
        header.setSectionsMovable(True)
        for visual_index, logical_index in enumerate(desired_order):
            current_visual = header.visualIndex(int(logical_index))
            if current_visual != visual_index:
                header.moveSection(current_visual, visual_index)
        header.setSectionsMovable(bool(movable_before))

        for column in range(model.columnCount()):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
        self._auto_fit_columns()
        for column, width in (
            (model.slider_column(), _SLIDER_COLUMN_WIDTH),
            (model.plot_column(), _PLOT_COLUMN_WIDTH),
            (model.custom_column(), _CUSTOM_COLUMN_WIDTH),
            (model.run_gear_column(), _RUN_GEAR_COLUMN_WIDTH),
        ):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Fixed,
            )
            self.setColumnWidth(column, width)

    def _auto_fit_columns(self) -> None:
        model = self.model()
        if not isinstance(model, BatchInitialConditionsTableModel):
            return
        header = self.horizontalHeader()
        header.setMaximumSectionSize(_MAX_SET_COLUMN_WIDTH)
        header_text = model.headerData(
            0,
            QtCore.Qt.Orientation.Horizontal,
            QtCore.Qt.ItemDataRole.DisplayRole,
        )
        header_minimum = header.fontMetrics().horizontalAdvance(
            f" {str(header_text or '')} "
        ) + 20
        for column in range(len(model.species_names()) + 1):
            self.resizeColumnToContents(column)
            maximum = (
                _MAX_SET_COLUMN_WIDTH
                if column == 0
                else _MAX_SPECIES_COLUMN_WIDTH
            )
            width = min(self.columnWidth(column), maximum)
            if column == 0:
                width = max(width, header_minimum)
            self.setColumnWidth(column, width)

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
            self.commitData(editor)
            self.closeEditor(
                editor,
                QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint,
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            model = self.model()
            if (isinstance(model, BatchInitialConditionsTableModel)
                    and model.request_run_settings_for_index(self.currentIndex())):
                event.accept()
                return
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
    """Resolve the current row followed by stable valid Slider pins.

    The current row remains the deterministic primary target even when pins
    exist.  Pinned rows are then appended in authored presentation order,
    with invalid and duplicate identities discarded at this boundary.
    """

    existing = {str(set_id) for set_id in existing_set_ids}
    current: tuple[str, ...] = ()
    model = table.model()
    if isinstance(model, BatchInitialConditionsTableModel):
        index = table.currentIndex()
        if index.isValid() and index.model() is model:
            set_id = model.set_id_for_live_index(index)
            if set_id in existing:
                current = (set_id,)
    pinned = tuple(
        set_id
        for set_id in dict.fromkeys(str(item) for item in pinned_set_ids)
        if set_id in existing
    )
    return tuple(dict.fromkeys((*current, *pinned)))

class InitialConditionsTableActions(QtCore.QObject):
    """Own IC intents for one immutable mechanism-tab identity."""

    errorRaised = QtCore.Signal(str)
    draftStateChanged = QtCore.Signal(str, bool)
    initialValuesChanged = QtCore.Signal(str)
    setNameChanged = QtCore.Signal(str)
    selectedRunSetIdsChanged = QtCore.Signal(object)

    def __init__(
        self,
        *,
        tab_id: str,
        registry,
        model: BatchInitialConditionsTableModel,
        table: BatchInitialConditionsTableView,
        species_for_tab: Callable[[MechanismTabDocument], Sequence[str]] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        if not callable(getattr(registry, "tab", None)):
            raise TypeError("registry must provide tab(tab_id).")
        self._tab_id = target
        self._registry = registry
        self._model = model
        self._table = table
        self._species_for_tab = species_for_tab or canonical_species_names
        self._rendering = False
        model.initialValueEditRequested.connect(self.accept_value_edit)
        model.setNameEditRequested.connect(self.accept_name_edit)
        model.customRunSettingsToggleRequested.connect(
            self.toggle_custom_run_settings
        )
        model.runSettingsEditRequested.connect(self.edit_custom_run_settings)
        table.pasteRequested.connect(self.accept_paste)
        table.selectionModel().selectionChanged.connect(self._capture_run_selection)
        table.selectionModel().currentChanged.connect(self._capture_current_set)

    @property
    def tab_id(self) -> str:
        return self._tab_id

    def prepare_render(self) -> _InitialConditionsRenderProjection:
        """Prepare the complete table projection without touching widgets."""

        tab = self._registry.tab(self._tab_id)
        runtime = self._registry.tab_runtime(self._tab_id)
        return _InitialConditionsRenderProjection(
            tab=tab,
            presentation=self._registry.presentation(self._tab_id),
            invalid_raw_by_set=tuple(
                (
                    str(set_id),
                    tuple(
                        (str(species), str(raw))
                        for species, raw in values.items()
                    ),
                )
                for set_id, values
                in runtime.initial_text_by_set_id.items()
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
            selected_ordered = tuple(projection.selected_run_set_ids)
            selected = set(selected_ordered)
            current_set_id = projection.presentation.current_set_id
            if tuple(self._table.selected_set_ids()) != selected_ordered:
                selection.clearSelection()
                for row, set_id in enumerate(self._model.set_ids()):
                    if set_id not in selected:
                        continue
                    selection.select(
                        self._model.index(row, 0),
                        QtCore.QItemSelectionModel.SelectionFlag.Select
                        | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )
            rendered_current = self._model.set_id_for_live_index(
                self._table.currentIndex()
            )
            if rendered_current != current_set_id:
                for row, set_id in enumerate(self._model.set_ids()):
                    if set_id != current_set_id:
                        continue
                    selection.setCurrentIndex(
                        self._model.index(row, 0),
                        QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
                    break
        finally:
            self._rendering = False
            del selection_blocker

    def render(self) -> None:
        self.apply_render_projection(self.prepare_render())

    def capture_active_editor(
        self,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Capture the focused numeric editor without committing its value."""

        if not isinstance(expected_token, ProjectContentToken):
            raise TypeError("expected_token must be ProjectContentToken.")
        if not self._registry.capture_content_token() == expected_token:
            return None
        editor = QtWidgets.QApplication.focusWidget()
        if not isinstance(editor, QtWidgets.QLineEdit):
            return expected_token
        if editor is self._table or not self._table.isAncestorOf(editor):
            return expected_token
        center = editor.mapTo(self._table.viewport(), editor.rect().center())
        index = self._table.indexAt(center)
        if not index.isValid() or index.model() is not self._model:
            return expected_token
        species = self._model.species_for_column(index.column())
        set_id = self._model.set_id_for_live_index(index)
        if species is None or set_id is None:
            return expected_token
        runtime = self._registry.tab_runtime(self._tab_id)
        runtime.initial_text_by_set_id.setdefault(set_id, {})[species] = editor.text()
        # closeEditor(NoHint) does not emit commitData.  Block the native
        # signals while focus leaves so editingFinished cannot publish a
        # second path during a modal disposition boundary.
        blockers = (
            QtCore.QSignalBlocker(editor),
            QtCore.QSignalBlocker(self._table),
            QtCore.QSignalBlocker(self._model),
        )
        try:
            self._table.closeEditor(
                editor,
                QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint,
            )
        finally:
            del blockers
        self.render()
        self.draftStateChanged.emit(self._tab_id, runtime.has_dirty_drafts)
        return expected_token

    def settle_active_editor(self) -> None:
        """Synchronously submit the active cell editor."""

        self._table.settle_active_editor()

    def selected_run_set_ids(self) -> tuple[str, ...]:
        return tuple(
            self._registry.tab_runtime(self._tab_id).selected_run_set_ids
        )

    def invalid_cells(
        self,
        set_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        runtime = self._registry.tab_runtime(self._tab_id)
        allowed = None if set_ids is None else {str(item) for item in set_ids}
        return tuple(
            (set_id, species, raw)
            for set_id, values in runtime.initial_text_by_set_id.items()
            if allowed is None or set_id in allowed
            for species, raw in values.items()
            if not self._valid_raw(raw, species)
        )

    def has_invalid_cells(self, set_ids: Iterable[str] | None = None) -> bool:
        return bool(self.invalid_cells(set_ids))

    @QtCore.Slot(str, str, str)
    def accept_value_edit(self, set_id: str, species: str, raw_text: str) -> bool:
        tab = self._registry.tab(self._tab_id)
        initial_set = tab.initial_condition_set(set_id)
        try:
            value = validate_concentration(
                float(raw_text),
                field=f"initial concentration for {species}",
            )
        except (TypeError, ValueError, OverflowError):
            value = None
        invalid = self._registry.tab_runtime(
            self._tab_id
        ).initial_text_by_set_id
        if value is None:
            if invalid.get(set_id, {}).get(species) == str(raw_text):
                return False
            invalid.setdefault(set_id, {})[species] = str(raw_text)
            self.render()
            self.draftStateChanged.emit(
                self._tab_id,
                self._registry.tab_runtime(self._tab_id).has_dirty_drafts,
            )
            return False

        values = dict(initial_set.values)
        canonical = float(values.get(str(species), 0.0))
        changed = value != canonical
        result: ProjectContentToken | None = None
        if changed:
            values[str(species)] = value
            candidate = initial_set.with_values(values)
            result = self._registry.replace_initial_condition_set(
                self._tab_id,
                candidate,
                expected_token=self._registry.capture_content_token(),
            )
            if result is None:
                return False
        invalid_cleared = self._clear_invalid_value(set_id, species)
        self.render()
        if changed:
            self.initialValuesChanged.emit(self._tab_id)
        if invalid_cleared:
            self.draftStateChanged.emit(
                self._tab_id,
                self._registry.tab_runtime(self._tab_id).has_dirty_drafts,
            )
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
                values = [
                    validate_concentration(
                        float(raw.strip()),
                        field="pasted initial concentration",
                    )
                    for raw in row
                ]
                parsed.append(values)
        except (TypeError, ValueError, OverflowError):
            return self._reject(
                "Every pasted concentration must be within the supported "
                "concentration domain."
            )

        tab = self._registry.tab(self._tab_id)
        candidate_sets = list(tab.initial_condition_sets)
        changed_ids: list[str] = []
        cleared_cells: list[tuple[str, str]] = []
        species_names = self._model.species_names()
        for row_offset, values_to_paste in enumerate(parsed):
            row_index = start_row + row_offset
            initial_set = candidate_sets[row_index]
            values = dict(initial_set.values)
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

        result: ProjectContentToken | None = None
        if changed_ids:
            result = self._registry.replace_tab(
                self._tab_id,
                tab.with_initial_condition_sets(tuple(candidate_sets)),
                expected_token=self._registry.capture_content_token(),
            )
            if result is None:
                return False
        invalid_cleared = False
        for set_id, species in cleared_cells:
            invalid_cleared = (
                self._clear_invalid_value(set_id, species)
                or invalid_cleared
            )
        self.render()
        if result is not None:
            self.initialValuesChanged.emit(self._tab_id)
        if invalid_cleared:
            self.draftStateChanged.emit(
                self._tab_id,
                self._registry.tab_runtime(self._tab_id).has_dirty_drafts,
            )
        return bool(result is not None or invalid_cleared)

    @QtCore.Slot(str, str)
    def accept_name_edit(self, set_id: str, name: str) -> bool:
        normalized = str(name).strip()
        tab = self._registry.tab(self._tab_id)
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
        changed = initial_set.name != normalized
        result = self._registry.replace_initial_condition_set(
            self._tab_id,
            replace(initial_set, name=normalized),
            expected_token=self._registry.capture_content_token(),
        )
        accepted = result is not None
        self.render()
        if accepted and changed:
            self.setNameChanged.emit(self._tab_id)
        return accepted

    def add_set(self) -> InitialConditionSet | None:
        tab = self._registry.tab(self._tab_id)
        created = InitialConditionSet(
            set_id=new_set_id(),
            name=next_set_name(item.name for item in tab.initial_condition_sets),
            values=tuple(
                (str(species), 0.0) for species in self._species_for_tab(tab)
            ),
        )
        result = self._registry.mutate_tab(
            self._tab_id,
            lambda current: current.add_initial_condition_set(created),
            expected_token=self._registry.capture_content_token(),
        )
        accepted = result is not None
        self.render()
        return created if accepted else None

    def move_selected(self, delta: int) -> tuple[str, ...]:
        self.settle_active_editor()
        selected = self.selected_run_set_ids()
        if not selected or int(delta) == 0:
            return selected
        tab = self._registry.tab(self._tab_id)
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
        self._registry.mutate_tab(
            self._tab_id,
            lambda current: current.with_initial_condition_sets(tuple(ordered)),
            expected_token=self._registry.capture_content_token(),
        )
        self.render()
        return tuple(ordered[row].set_id for row in sorted(active))

    def delete_selected(self) -> bool:
        self.settle_active_editor()
        targets = self.selected_run_set_ids()
        if not targets:
            return False
        result = self._registry.remove_initial_condition_sets(
            self._tab_id,
            targets,
            expected_token=self._registry.capture_content_token(),
        )
        accepted = result is not None
        self.render()
        return accepted

    @QtCore.Slot(str, bool)
    def set_plot_requested(self, set_id: str, requested: bool) -> bool:
        state = self._registry.presentation(self._tab_id)
        ids = [item for item in state.plot_set_ids if item != set_id]
        if requested:
            ids.append(set_id)
        next_state = replace(state, plot_set_ids=tuple(ids))
        if next_state == state:
            return False
        self._registry.set_presentation(
            self._tab_id,
            next_state,
        )
        self.render()
        return True

    @QtCore.Slot(object)
    def union_plot_membership(self, set_ids: Iterable[str]) -> bool:
        admitted = tuple(str(set_id) for set_id in set_ids)
        state = self._registry.presentation(self._tab_id)
        next_ids = tuple(dict.fromkeys((*state.plot_set_ids, *admitted)))
        if next_ids == state.plot_set_ids:
            return False
        self._registry.set_presentation(
            self._tab_id,
            replace(state, plot_set_ids=next_ids),
        )
        self.render()
        return True

    @QtCore.Slot(str, bool)
    def set_slider_pinned(self, set_id: str, pinned: bool) -> bool:
        state = self._registry.presentation(self._tab_id)
        ids = [item for item in state.pinned_slider_set_ids if item != set_id]
        if pinned:
            ids.append(set_id)
        next_state = replace(state, pinned_slider_set_ids=tuple(ids))
        if next_state == state:
            return False
        self._registry.set_presentation(
            self._tab_id,
            next_state,
        )
        self.render()
        return True

    def set_workspace_run_defaults(
        self,
        settings: SimulationExecutionSettings,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        result = self._registry.set_run_defaults(
            self._tab_id,
            settings,
            expected_token=expected_token,
        )
        self.render()
        return result

    @QtCore.Slot(str, bool)
    def toggle_custom_run_settings(self, set_id: str, checked: bool) -> bool:
        tab = self._registry.tab(self._tab_id)
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
            self.render()
            return False
        return self.set_custom_run_settings(set_id, None)

    @QtCore.Slot(str)
    def edit_custom_run_settings(self, set_id: str) -> bool:
        tab = self._registry.tab(self._tab_id)
        override = dict(tab.run_overrides_by_set_id).get(set_id)
        if override is None:
            return False
        return self._open_settings_dialog(set_id, override)

    def _open_settings_dialog(
        self,
        set_id: str,
        seed: SimulationExecutionSettings,
    ) -> bool:
        tab = self._registry.tab(self._tab_id)
        initial_set = tab.initial_condition_set(set_id)
        dialog = SolverSettingsDialog(
            self._table,
            title=f"Custom Run Settings — {initial_set.name}",
        )
        dialog.set_execution_settings(seed)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self.render()
            return False
        return self.set_custom_run_settings(set_id, dialog.execution_settings())

    def set_custom_run_settings(
        self,
        set_id: str,
        settings: SimulationExecutionSettings | None,
    ) -> bool:
        tab = self._registry.tab(self._tab_id)
        current = dict(tab.run_overrides_by_set_id).get(set_id)
        if current == settings:
            return False
        result = self._registry.set_run_override(
            self._tab_id,
            set_id,
            settings,
            expected_token=self._registry.capture_content_token(),
        )
        accepted = result is not None
        self.render()
        return accepted

    @QtCore.Slot()
    def _capture_run_selection(self, *_args) -> None:
        if self._rendering:
            return
        valid = set(self._model.set_ids())
        selected = tuple(
            set_id for set_id in self._table.selected_set_ids() if set_id in valid
        )
        runtime = self._registry.tab_runtime(self._tab_id)
        if tuple(runtime.selected_run_set_ids) == selected:
            return
        runtime.selected_run_set_ids = selected
        self.selectedRunSetIdsChanged.emit(selected)

    @QtCore.Slot()
    def _capture_current_set(self, *_args) -> None:
        if self._rendering:
            return
        set_id = self._model.set_id_for_live_index(self._table.currentIndex())
        if set_id is None:
            return
        state = self._registry.presentation(self._tab_id)
        if state.current_set_id != set_id:
            self._registry.set_presentation(
                self._tab_id,
                replace(state, current_set_id=set_id),
            )

    def _clear_invalid_value(self, set_id: str, species: str) -> bool:
        invalid = self._registry.tab_runtime(
            self._tab_id
        ).initial_text_by_set_id
        set_values = invalid.get(set_id)
        if set_values is None or species not in set_values:
            return False
        set_values.pop(species, None)
        if not set_values:
            invalid.pop(set_id, None)
        return True

    @staticmethod
    def _valid_raw(raw: object, species: str) -> bool:
        try:
            validate_concentration(
                float(raw),
                field=f"initial concentration for {species}",
            )
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    def _reject(self, message: str) -> bool:
        self.errorRaised.emit(str(message))
        return False

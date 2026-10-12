"""Shared observation mapping and initial-concentration controls."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.datasets.foundation import RawDatasetRecord
from kindred.core.datasets.observation_mapping import (
    InitialConcentrationSource,
    ObservationMappingError,
    admit_measured_start,
    first_data_row,
    initial_samples,
    normalize_observation_mapping,
    suggest_observation_mapping,
    supplies_concentration,
)
from kindred.core.species import concentration_from_text
from kindred.core.validation import six_digit_text
from kindred.gui.ui_helpers import (
    INVALID_EDITOR_STYLE,
    clamp_initial_size,
    data_unit_labels,
    enter_confirms_only,
    ignore_wheel,
)

_DATA = "data"
_MANUAL = "manual"
_SHOWN_VALUE_STYLE = "QLineEdit {{ background: transparent; border: none;{extra} }}"


@dataclass(frozen=True)
class SourceSettings:
    """One dataset's starting-concentration choices, copied between datasets."""

    row_source: InitialConcentrationSource | None
    # Species -> None for From data, or the typed text of a Manual choice.
    manual: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeciesStart:
    """How one species starts under the current choices."""

    species: str
    # None means no concentration is supplied, so the species starts at 0.
    value: float | None
    source: str
    note: str = ""
    problem: str = ""
    # Why From data is impossible for this species, if it is.
    blocked: str = ""


class ObservationMappingWidget(QtWidgets.QWidget):
    """One row per model target: its column and, for species, its starting value."""

    changed = QtCore.Signal()

    def __init__(
        self,
        record: RawDatasetRecord,
        model_targets: tuple[str, ...],
        *,
        current_mapping: tuple[tuple[str, str], ...] = (),
        mechanism_species: tuple[str, ...] = (),
        create_set: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._record = record
        self._targets = model_targets
        self._species = tuple(name for name in model_targets if name in mechanism_species)
        self._series = {item.raw_observation_id: item for item in record.observations}
        self._first_row = first_data_row(record.observations)
        self._create_set = create_set
        self._existing: tuple[str, dict[str, float]] | None = None
        # Each species' choice: None for From data, or the text typed for Manual.
        self._manual: dict[str, str | None] = {name: None for name in self._species}
        # Computed once per change of this table; every reader uses it.
        self._starts: tuple[SpeciesStart, ...] = ()
        self._updating = False

        # A saved mapping with mapped columns opens exactly as saved; a dataset
        # without a mapped column receives name suggestions.
        if not current_mapping:
            current_mapping = tuple(
                (item.raw_observation_id, item.model_target)
                for item in suggest_observation_mapping(
                    raw_observations=tuple(
                        (item.raw_observation_id, item.column_name) for item in record.observations
                    ),
                    model_targets=model_targets,
                )
            )
        column_by_target = {target: raw_id for raw_id, target in current_mapping}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        source_row = QtWidgets.QHBoxLayout()
        self._source_title = QtWidgets.QLabel("Initial concentrations from", self)
        source_row.addWidget(self._source_title)
        self._source_combo = QtWidgets.QComboBox(self)
        ignore_wheel(self._source_combo)
        self._source_combo.setAccessibleName("Initial concentrations from")
        self._source_combo.addItem("Choose…", None)
        self._source_combo.addItem("Recorded data at t = 0", InitialConcentrationSource.RECORDED_ZERO)
        self._source_combo.addItem("First data row", InitialConcentrationSource.FIRST_DATA_ROW)
        has_zero = any(0.0 in item.t for item in record.observations)
        self._source_combo.model().item(1).setEnabled(has_zero)
        if not has_zero:
            self._source_combo.model().item(1).setToolTip("The dataset has no recorded observation at t = 0.")
        if has_zero and any(
            column_by_target.get(name) in self._series
            and supplies_concentration(self._series[column_by_target[name]])
            and 0.0 in self._series[column_by_target[name]].t
            for name in self._species
        ):
            self._source_combo.setCurrentIndex(1)
        source_row.addWidget(self._source_combo, 1)
        self._source_note = QtWidgets.QLabel(self)
        source_row.addWidget(self._source_note)
        layout.addLayout(source_row)

        self._table = QtWidgets.QTableWidget(len(model_targets), 4, self)
        self._table.setHorizontalHeaderLabels(["Maps to", "Column", "Source", "Initial Concentration (M)"])
        self._table.setAccessibleName("Observation mapping")
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._table.verticalHeader().hide()
        self._table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self._table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 16)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        # Fixed, so switching between a Source dropdown and a set name keeps the width.
        sizing = QtWidgets.QComboBox()
        sizing.addItems(["From data", "Manual"])
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self._table.setColumnWidth(2, sizing.sizeHint().width() + 8)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self._table.setMinimumHeight(120)

        self._column_combos: dict[str, QtWidgets.QComboBox] = {}
        self._source_combos: dict[str, QtWidgets.QComboBox] = {}
        self._value_edits: dict[str, QtWidgets.QLineEdit] = {}
        for row, target in enumerate(model_targets):
            is_species = target in self._species
            name = QtWidgets.QTableWidgetItem(target if is_species else f"{target} (let)")
            if not is_species and self._species:
                name.setToolTip("A let quantity: it can be mapped to a column but has no initial concentration.")
            self._table.setItem(row, 0, name)
            combo = QtWidgets.QComboBox(self._table)
            ignore_wheel(combo)
            combo.setAccessibleName(f"Column for {target}")
            combo.addItem("—", None)
            for series in record.observations:
                combo.addItem(series.column_name, series.raw_observation_id)
                combo.setItemData(
                    combo.count() - 1, f"{series.column_name} · {data_unit_labels(series.unit)[0]}",
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
            selected = column_by_target.get(target)
            if selected in self._series:
                combo.setCurrentIndex(combo.findData(selected))
            combo.currentIndexChanged.connect(self._refresh)
            self._column_combos[target] = combo
            self._table.setCellWidget(row, 1, combo)
            if not is_species:
                for column in (2, 3):
                    blank = QtWidgets.QTableWidgetItem("")
                    blank.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                    self._table.setItem(row, column, blank)
                continue
            edit = QtWidgets.QLineEdit(self._table)
            edit.setAccessibleName(f"Initial concentration for {target}")
            edit.textEdited.connect(lambda text, species=target: self._on_manual_edited(species, text))
            self._value_edits[target] = edit
            self._table.setCellWidget(row, 3, edit)
        layout.addWidget(self._table, 1)

        self._unused = QtWidgets.QLabel(self)
        self._unused.setWordWrap(True)
        layout.addWidget(self._unused)
        self._feedback = QtWidgets.QLabel(self)
        self._feedback.setTextFormat(QtCore.Qt.PlainText)
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        self._source_combo.currentIndexChanged.connect(self._refresh)
        self._show_mode()

    # -- public surface ---------------------------------------------------------
    def confirmed_mapping(self) -> tuple[tuple[str, str], ...]:
        return normalize_observation_mapping(
            tuple((combo.currentData(), target) for target, combo in self._column_combos.items()
                  if combo.currentData() is not None),
            raw_observations=tuple((item.raw_observation_id, item.column_name) for item in self._record.observations),
            model_targets=self._targets,
        )

    def creation_inputs(
        self,
    ) -> tuple[InitialConcentrationSource | None, tuple[tuple[str, float | None], ...]]:
        """The row source (None when no species reads data) and every Manual species.

        An empty Manual value is None: it supplies no concentration.
        """
        starts = self.starts()
        problems = [item.problem for item in starts if item.problem]
        if problems:
            raise ObservationMappingError(problems[0])
        manual = tuple((item.species, item.value) for item in starts if item.source == _MANUAL)
        return (self._source_combo.currentData() if self._reads_data() else None), manual

    def problems(self) -> tuple[str, ...]:
        if not self._create_set:
            return ()
        return tuple(item.problem for item in self.starts() if item.problem)

    def show_problem(self, text: str) -> None:
        self._feedback.setText(text)
        self._feedback.setVisible(bool(text))

    def source_settings(self) -> SourceSettings:
        """The choices as shown, for copying to other datasets.

        Each species keeps its displayed Source. An untouched forced Manual (no
        readable column) is not a choice, so it never overrides another
        dataset's data. The row source is copied only while some species reads
        data.
        """
        return SourceSettings(
            self._source_combo.currentData() if self._reads_data() else None,
            {start.species: self._manual[start.species] for start in self.starts()
             if start.source == _DATA or self._manual[start.species] is not None},
        )

    def apply_source_settings(self, settings: SourceSettings) -> None:
        """Take another dataset's choices once; they stay editable here."""
        # A copied choice this dataset cannot use leaves its own choice.
        index = self._source_combo.findData(settings.row_source)
        if settings.row_source is not None and self._source_combo.model().item(index).isEnabled():
            with QtCore.QSignalBlocker(self._source_combo):
                self._source_combo.setCurrentIndex(index)
        unreadable = {start.species for start in self._compute_starts() if start.blocked}
        self._manual.update({name: text for name, text in settings.manual.items()
                             if name in self._manual and not (text is None and name in unreadable)})
        self._refresh()

    def set_create_set(self, enabled: bool) -> None:
        """Edit starting values for a new set, or show no starting values."""
        if self._create_set == enabled and self._existing is None:
            return
        self._create_set = enabled
        self._existing = None
        self._show_mode()

    def show_existing_set(self, name: str, values: dict[str, float]) -> None:
        """Show the values of the existing set this dataset maps to, read-only."""
        if not self._create_set and self._existing == (name, values):
            return
        self._create_set = False
        self._existing = (name, values)
        self._show_mode()

    def starts(self) -> tuple[SpeciesStart, ...]:
        """Each mechanism species' starting value under the current choices."""
        return self._starts

    def _compute_starts(self) -> tuple[SpeciesStart, ...]:
        row_source = self._source_combo.currentData()
        result = []
        for species in self._species:
            series = self._series.get(self._column_combos[species].currentData())
            reason = ""
            samples: tuple[tuple[int, float], ...] = ()
            if series is None:
                reason = "No column feeds this species."
            elif not supplies_concentration(series):
                reason = f"{series.column_name} is not a concentration ({data_unit_labels(series.unit)[0]})."
            elif row_source is not None:
                samples = initial_samples(series, source=row_source, first_row=self._first_row)
                if len(samples) > 1:
                    values = ", ".join(f"{six_digit_text(value)} (row {row})" for row, value in samples)
                    reason = f"Column {series.column_name} has {len(samples)} values at t = 0: {values}."
            if reason or self._manual[species] is not None:
                result.append(self._manual_start(species, reason, ambiguous=len(samples) > 1))
                continue
            if row_source is None:
                result.append(SpeciesStart(species, None, _DATA,
                                           problem="Choose a source for the initial concentrations."))
                continue
            if not samples:
                where = "t = 0" if row_source is InitialConcentrationSource.RECORDED_ZERO else f"row {self._first_row}"
                result.append(SpeciesStart(
                    species, None, _DATA, f"Column {series.column_name} has no value at {where}; starts at 0 M.",
                ))
                continue
            try:
                value = admit_measured_start(samples[0][1], species=species)
            except ObservationMappingError as exc:
                result.append(SpeciesStart(species, None, _DATA, str(exc), str(exc)))
                continue
            result.append(SpeciesStart(species, value, _DATA, f"From {series.column_name}, row {samples[0][0]}."))
        return tuple(result)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        # The themed palette arrives when the dialog is first shown.
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange):
            self._draw()

    # -- internals ----------------------------------------------------------------
    def _manual_start(self, species: str, reason: str, *, ambiguous: bool) -> SpeciesStart:
        text = (self._manual[species] or "").strip()
        if not text:
            if ambiguous:
                # Measured, but which value is the start is unclear: never a silent 0.
                return SpeciesStart(species, None, _MANUAL, reason,
                                    f"{species}: type a starting concentration. {reason}", reason)
            return SpeciesStart(species, None, _MANUAL, reason or "Not supplied; starts at 0 M.", blocked=reason)
        try:
            value = concentration_from_text(text, field=f"Initial concentration for {species}")
        except ValueError as exc:
            return SpeciesStart(species, None, _MANUAL, str(exc), str(exc), reason)
        return SpeciesStart(species, value, _MANUAL, reason, blocked=reason)

    def _readable_species(self) -> tuple[str, ...]:
        """Species mapped to a concentration column, which could read data."""
        return tuple(
            species for species in self._species
            if (series := self._series.get(self._column_combos[species].currentData())) is not None
            and supplies_concentration(series)
        )

    def _reads_data(self) -> bool:
        """Whether any species reads data, so a row source is required."""
        return any(start.source == _DATA for start in self.starts())

    def _show_mode(self) -> None:
        enabled = self._create_set
        for widget in (self._source_title, self._source_combo, self._source_note):
            widget.setVisible(enabled)
        shown = enabled or self._existing is not None
        self._table.setColumnHidden(2, not shown)
        self._table.setColumnHidden(3, not shown)
        # A mapped existing set shows its name where a new set chooses a Source.
        for row, target in enumerate(self._targets):
            if target not in self._species:
                continue
            if enabled and target not in self._source_combos:
                self._install_source_combo(row, target)
            elif not enabled and target in self._source_combos:
                del self._source_combos[target]
                self._table.removeCellWidget(row, 2)
        self._refresh()

    def _install_source_combo(self, row: int, target: str) -> None:
        source = QtWidgets.QComboBox(self._table)
        ignore_wheel(source)
        source.setAccessibleName(f"Source for {target}")
        source.addItem("From data", _DATA)
        source.addItem("Manual", _MANUAL)
        source.currentIndexChanged.connect(lambda _index, species=target: self._on_source_changed(species))
        self._source_combos[target] = source
        self._table.takeItem(row, 2)
        self._table.setCellWidget(row, 2, source)

    def _on_source_changed(self, species: str) -> None:
        if self._updating:
            return
        # Choosing From data discards a typed value; choosing Manual starts empty.
        self._manual[species] = None if self._source_combos[species].currentData() == _DATA else ""
        self._refresh()

    def _on_manual_edited(self, species: str, text: str) -> None:
        # Typing a value makes Manual this species' choice.
        self._manual[species] = text
        self._refresh(keep_text=species)

    def _refresh(self, *_args, keep_text: str = "") -> None:
        self._starts = self._compute_starts()
        self._draw(keep_text)
        self.changed.emit()

    def _draw(self, keep_text: str = "") -> None:
        self._updating = True
        try:
            self._refresh_columns()
            if self._create_set:
                self._refresh_starts(keep_text)
            else:
                self._refresh_existing()
        finally:
            self._updating = False

    def _refresh_columns(self) -> None:
        # A column feeds one target, so columns in use elsewhere are unavailable.
        used = {combo.currentData(): target for target, combo in self._column_combos.items()
                if combo.currentData() is not None}
        for target, combo in self._column_combos.items():
            model = combo.model()
            for index in range(1, combo.count()):
                owner = used.get(combo.itemData(index))
                model.item(index).setEnabled(owner is None or owner == target)
        unused = [item.column_name for item in self._record.observations if item.raw_observation_id not in used]
        self._unused.setText(f"Unused columns: {', '.join(unused)}" if unused else "")
        self._unused.setVisible(bool(unused))

    def _refresh_starts(self, keep_text: str) -> None:
        reads_data = self._reads_data()
        # Usable even while unused: choosing a row can make data readable again.
        self._source_combo.setEnabled(bool(self._readable_species()))
        self._source_note.setText("" if reads_data else "Not used (no row reads data)")
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        starts = self.starts()
        for start in starts:
            species = start.species
            source = self._source_combos[species]
            source.model().item(0).setEnabled(not start.blocked)
            source.model().item(0).setToolTip(start.blocked)
            with QtCore.QSignalBlocker(source):
                source.setCurrentIndex(source.findData(start.source))
            edit = self._value_edits[species]
            manual = start.source == _MANUAL
            edit.setReadOnly(not manual)
            if species != keep_text:
                if manual:
                    edit.setText(self._manual[species] or "")
                elif start.value is None:
                    edit.setText("" if start.problem else "0")
                else:
                    edit.setText(six_digit_text(start.value))
            if manual:
                edit.setPlaceholderText("required" if start.problem else "not supplied (0)")
            else:
                edit.setPlaceholderText("")
            if manual and start.problem and edit.text().strip():
                edit.setStyleSheet(INVALID_EDITOR_STYLE)
            elif not manual:
                # Values read from data are shown, not edited; not supplied is
                # italic in the theme grey, rgba keeping its transparency.
                extra = (
                    f" color: rgba({muted.red()}, {muted.green()}, {muted.blue()}, {muted.alpha()});"
                    " font-style: italic;"
                    if start.value is None else ""
                )
                edit.setStyleSheet(_SHOWN_VALUE_STYLE.format(extra=extra))
            else:
                edit.setStyleSheet("")
            edit.setToolTip(start.problem or start.note)
        problems = [item.problem for item in starts if item.problem]
        self.show_problem(problems[0] if problems else "")

    def _refresh_existing(self) -> None:
        self.show_problem("")
        if self._existing is None:
            return
        name, values = self._existing
        tooltip = f"Mapping keeps the values of “{name}”; edit them in Initial Conditions."
        for row, target in enumerate(self._targets):
            if target not in self._species:
                continue
            item = QtWidgets.QTableWidgetItem(name)
            item.setToolTip(tooltip)
            self._table.setItem(row, 2, item)
            edit = self._value_edits[target]
            edit.setReadOnly(True)
            edit.setText(six_digit_text(values.get(target, 0.0)))
            edit.setPlaceholderText("")
            edit.setToolTip(tooltip)
            edit.setStyleSheet(_SHOWN_VALUE_STYLE.format(extra=""))


class ObservationMappingDialog(QtWidgets.QDialog):
    """Confirm an observation mapping, with starting concentrations when creating a set."""

    def __init__(self, record, model_targets, *, current_mapping=(), mechanism_species=(), create_set=False, title, parent):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._create_set = create_set
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel("Selecting OK confirms the choices.", self)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.editor = ObservationMappingWidget(
            record, model_targets, current_mapping=current_mapping,
            mechanism_species=mechanism_species, create_set=create_set, parent=self,
        )
        layout.addWidget(self.editor, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept_mapping)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        clamp_initial_size(self, 720, 620 if create_set else 480)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override
        # Enter confirms a typed value; only OK itself confirms the dialog.
        if not enter_confirms_only(self, event):
            super().keyPressEvent(event)

    def _accept_mapping(self) -> None:
        try:
            self.editor.confirmed_mapping()
            if self._create_set:
                self.editor.creation_inputs()
        except ObservationMappingError as exc:
            self.editor.show_problem(str(exc))
            return
        self.accept()

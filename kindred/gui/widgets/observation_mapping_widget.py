"""Shared observation mapping and initial-concentration preview controls."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from kindred.core.datasets.foundation import RawDatasetRecord
from kindred.core.datasets.observation_mapping import (
    InitialConcentrationSource,
    ObservationMappingError,
    extract_initial_concentrations,
    normalize_observation_mapping,
    suggest_observation_mapping,
)
from kindred.core.quantity_units import MOLAR
from kindred.gui.ui_helpers import clamp_initial_size


class ObservationMappingWidget(QtWidgets.QWidget):
    """Edit mappings, optionally previewing an explicitly chosen IC source."""

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
        self._species = mechanism_species
        self._create_set = create_set
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        current = dict(current_mapping)
        suggestions = {
            item.raw_observation_id: item.model_target
            for item in suggest_observation_mapping(
                raw_observations=tuple((item.raw_observation_id, item.column_name) for item in record.observations),
                model_targets=model_targets,
            )
        }
        self._observation_combos: dict[str, QtWidgets.QComboBox] = {}
        for series in record.observations:
            combo = QtWidgets.QComboBox(self)
            combo.setAccessibleName(f"Model target for {series.column_name}")
            combo.addItem("Unmapped", None)
            for name in model_targets:
                combo.addItem(name, name)
            selected = current.get(series.raw_observation_id, suggestions.get(series.raw_observation_id))
            if selected in model_targets:
                combo.setCurrentIndex(combo.findData(selected))
            self._observation_combos[series.raw_observation_id] = combo
            form.addRow(series.column_name, combo)
        layout.addLayout(form)

        self._concentrations_group = QtWidgets.QGroupBox("Initial concentrations", self)
        concentrations_layout = QtWidgets.QVBoxLayout(self._concentrations_group)
        source_form = QtWidgets.QFormLayout()
        self._source_combo = QtWidgets.QComboBox(self._concentrations_group)
        self._source_combo.setAccessibleName("Initial concentrations from")
        self._source_combo.addItem("Choose…", None)
        self._source_combo.addItem("Recorded data at t = 0", InitialConcentrationSource.RECORDED_ZERO)
        self._source_combo.addItem("First data row", InitialConcentrationSource.FIRST_DATA_ROW)
        self._source_combo.model().item(1).setEnabled(any(0.0 in item.t for item in record.observations))
        if any(
            self._observation_combos[item.raw_observation_id].currentData() in mechanism_species
            and item.unit.compatible(MOLAR) and 0.0 in item.t
            for item in record.observations
        ):
            self._source_combo.setCurrentIndex(1)
        source_form.addRow("Initial concentrations from", self._source_combo)
        concentrations_layout.addLayout(source_form)
        self._source_label = QtWidgets.QLabel(self._concentrations_group)
        concentrations_layout.addWidget(self._source_label)
        self._preview = QtWidgets.QTableWidget(0, 3, self._concentrations_group)
        self._preview.setAccessibleName("Initial concentration preview")
        self._preview.setHorizontalHeaderLabels(["Species", "Initial concentration", "Source"])
        self._preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._preview.verticalHeader().hide()
        self._preview.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self._preview.horizontalHeader().setStretchLastSection(True)
        self._preview.setMaximumHeight(190)
        concentrations_layout.addWidget(self._preview)
        explanation = QtWidgets.QLabel(
            "Only mapped columns with concentration units supply initial concentrations. "
            "Species without a supplied concentration start at 0 M.", self._concentrations_group,
        )
        explanation.setWordWrap(True)
        concentrations_layout.addWidget(explanation)
        layout.addWidget(self._concentrations_group)
        self._feedback = QtWidgets.QLabel(self)
        self._feedback.setTextFormat(QtCore.Qt.PlainText)
        self._feedback.setWordWrap(True)
        layout.addWidget(self._feedback)
        self._source_combo.currentIndexChanged.connect(self._update_preview)
        for combo in self._observation_combos.values():
            combo.currentIndexChanged.connect(self._update_preview)
        self.set_create_set(create_set)

    def confirmed_mapping(self) -> tuple[tuple[str, str], ...]:
        return normalize_observation_mapping(
            tuple((raw_id, combo.currentData()) for raw_id, combo in self._observation_combos.items()
                  if combo.currentData() is not None),
            raw_observations=tuple((item.raw_observation_id, item.column_name) for item in self._record.observations),
            model_targets=self._targets,
        )

    def creation_source(self) -> InitialConcentrationSource:
        source = self._source_combo.currentData()
        if source is None:
            raise ObservationMappingError("Choose a source for the initial concentrations.")
        extract_initial_concentrations(
            observations=self._record.observations,
            mechanism_species=self._species,
            confirmed_mapping=self.confirmed_mapping(),
            source=source,
        )
        return source

    def set_create_set(self, enabled: bool) -> None:
        self._create_set = enabled
        self._concentrations_group.setVisible(enabled)
        self._update_preview()

    def _update_preview(self) -> None:
        self._feedback.clear()
        self._preview.setRowCount(0)
        self._source_label.clear()
        if not self._create_set:
            return
        source = self._source_combo.currentData()
        if source is None:
            self._feedback.setText("Choose a source for the initial concentrations.")
            return
        if source is InitialConcentrationSource.FIRST_DATA_ROW:
            row = min(row for item in self._record.observations for row in item.source_rows)
            self._source_label.setText(f"First data row — row {row}")
        else:
            self._source_label.setText("Recorded data at t = 0")
        try:
            values = extract_initial_concentrations(
                observations=self._record.observations,
                mechanism_species=self._species,
                confirmed_mapping=self.confirmed_mapping(),
                source=source,
            )
        except ObservationMappingError as exc:
            self._feedback.setText(str(exc))
            return
        self._preview.setRowCount(len(values))
        for row, (name, value) in enumerate(values):
            texts = (
                name, f"{0.0 if value is None else value:g} M",
                "No concentration supplied" if value is None else "From data",
            )
            for column, text in enumerate(texts):
                self._preview.setItem(row, column, QtWidgets.QTableWidgetItem(text))


class ObservationMappingDialog(QtWidgets.QDialog):
    """Confirm an observation mapping, with IC preview when creating a set."""

    def __init__(self, record, model_targets, *, current_mapping=(), mechanism_species=(), create_set=False, title, parent):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._create_set = create_set
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Review every suggested mapping. Selecting OK explicitly confirms the choices.", self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.editor = ObservationMappingWidget(
            record, model_targets, current_mapping=current_mapping,
            mechanism_species=mechanism_species, create_set=create_set, parent=self,
        )
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept_mapping)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        clamp_initial_size(self, 650, 650 if create_set else 520)

    def _accept_mapping(self) -> None:
        try:
            self.editor.confirmed_mapping()
            if self._create_set:
                self.editor.creation_source()
        except ObservationMappingError as exc:
            self.editor._feedback.setText(str(exc))
            return
        self.accept()

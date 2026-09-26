"""One reviewed, scrollable mapping surface for experimental datasets."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from kindred.core.dataset_mapping import (
    DatasetMappingBatchProposal,
    DatasetMappingChoice,
    DatasetMappingError,
    DatasetMappingProposal,
    OccupiedSetChoice,
    allocate_initial_condition_set_name,
)
from kindred.gui.ui_helpers import clamp_initial_size
from kindred.core.datasets.foundation import RawDatasetRecord
from kindred.core.project_document import MechanismTabDocument, canonical_species_names
from kindred.gui.widgets.observation_mapping_widget import ObservationMappingWidget

__all__ = ["DatasetMappingBatchDialog"]


@dataclass(slots=True)
class _DatasetMappingRow:
    record: RawDatasetRecord
    action: QtWidgets.QComboBox
    target: QtWidgets.QComboBox
    occupancy: QtWidgets.QComboBox
    proposed_name: QtWidgets.QLabel
    editor: ObservationMappingWidget


class DatasetMappingBatchDialog(QtWidgets.QDialog):
    """Review every dataset mapping choice before one atomic tab replacement."""

    def __init__(
        self,
        *,
        records: tuple[RawDatasetRecord, ...],
        tab: MechanismTabDocument,
        model_targets: tuple[str, ...],
        create_only: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if type(records) is not tuple or not records:
            raise ValueError("Batch mapping review requires datasets.")
        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("Batch mapping review requires one mechanism tab.")
        targets = tuple(str(item).strip() for item in model_targets)
        if not targets or any(not item for item in targets):
            raise ValueError("Batch mapping review requires model targets.")
        self._create_only = create_only
        self._records = records
        self._tab = tab
        self._model_targets = targets
        self._rows: list[_DatasetMappingRow] = []
        self.setWindowTitle("Review New Datasets" if create_only else "Review Imported Dataset Mapping")
        self.setModal(True)
        clamp_initial_size(self, 900, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Review mappings and starting concentrations for each new dataset. Each receives a new initial-condition set."
            if self._create_only else "Review every committed dataset below. Suggestions are never "
            "applied until you press OK; each observation may be explicitly "
            "mapped or left Unmapped.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 2, 2)
        for record in self._records:
            content_layout.addWidget(self._build_dataset_group(record, content))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        if self._create_only:
            buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(f"Create {len(self._records)} datasets")
            buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Back to editor")
        buttons.accepted.connect(self._accept_review)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_dataset_group(
        self,
        record: RawDatasetRecord,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(record.display_name, parent)
        group_layout = QtWidgets.QVBoxLayout(group)
        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("Action:", group), 0, 0)
        action = QtWidgets.QComboBox(group)
        action.addItem("Create Set", DatasetMappingChoice.CREATE_SET.value)
        if not self._create_only:
            action.addItem("Map Existing", DatasetMappingChoice.MAP_EXISTING.value)
            action.addItem("Leave Unmapped", DatasetMappingChoice.LEAVE_UNMAPPED.value)
        controls.addWidget(action, 0, 1)

        controls.addWidget(QtWidgets.QLabel("Target set:", group), 1, 0)
        target = QtWidgets.QComboBox(group)
        target.addItem("(none)", None)
        for item in self._tab.initial_condition_sets:
            target.addItem(item.name, item.set_id)
        controls.addWidget(target, 1, 1)

        controls.addWidget(QtWidgets.QLabel("Occupancy:", group), 2, 0)
        occupancy = QtWidgets.QComboBox(group)
        occupancy.addItem("(automatic / none)", None)
        occupancy.addItem("Move set to this dataset", OccupiedSetChoice.OVERRIDE.value)
        occupancy.addItem("Create a separate copy", OccupiedSetChoice.CREATE_COPY.value)
        occupancy.addItem("Cancel this occupied choice", OccupiedSetChoice.CANCEL.value)
        controls.addWidget(occupancy, 2, 1)

        controls.addWidget(QtWidgets.QLabel("Proposed set name:", group), 3, 0)
        proposed_name = QtWidgets.QLabel("(not applicable)", group)
        proposed_name.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(proposed_name, 3, 1)
        if self._create_only:
            for row in range(3):
                for column in range(2):
                    controls.itemAtPosition(row, column).widget().hide()
        controls.setColumnStretch(1, 1)
        group_layout.addLayout(controls)

        group_layout.addWidget(QtWidgets.QLabel("Observation target mappings (confirm to apply suggestions):", group))
        editor = ObservationMappingWidget(
            record, self._model_targets,
            mechanism_species=canonical_species_names(self._tab), create_set=True, parent=group,
        )
        group_layout.addWidget(editor)

        row = _DatasetMappingRow(
            record=record,
            action=action,
            target=target,
            occupancy=occupancy,
            proposed_name=proposed_name,
            editor=editor,
        )
        self._rows.append(row)
        action.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        target.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        occupancy.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        self._refresh_rows()
        return group

    def _refresh_rows(self) -> None:
        names = [item.name for item in self._tab.initial_condition_sets]
        projected_occupants = {
            link.set_id: dataset_id
            for dataset_id, link in self._tab.dataset_links
            if link.set_id is not None
        }
        for row in self._rows:
            # A row is a complete proposal member.  Re-project its prior
            # occupancy before applying its current action so changing a row
            # cannot leave stale occupancy behind for later rows.
            for set_id, occupant in tuple(projected_occupants.items()):
                if occupant == row.record.dataset_id:
                    del projected_occupants[set_id]

            action = DatasetMappingChoice(row.action.currentData())
            is_create = action is DatasetMappingChoice.CREATE_SET
            is_existing = action is DatasetMappingChoice.MAP_EXISTING
            row.editor.set_create_set(is_create)
            row.target.setEnabled(is_existing)
            row.occupancy.setEnabled(False)
            if is_create:
                self._set_occupancy_choice(row, None)
                proposed = allocate_initial_condition_set_name(
                    row.record.display_name,
                    existing_names=names,
                )
                row.proposed_name.setText(proposed)
                names.append(proposed)
                continue
            if not is_existing:
                self._set_occupancy_choice(row, None)
                row.proposed_name.setText("(not applicable)")
                continue
            target_id = row.target.currentData()
            target = next(
                (item for item in self._tab.initial_condition_sets if item.set_id == target_id),
                None,
            )
            if target is None:
                self._set_occupancy_choice(row, None)
                row.proposed_name.setText("(select a target)")
                continue
            occupant = projected_occupants.get(target.set_id)
            if occupant is None:
                # This target may have been free in the committed tab but
                # occupied by an earlier row in this same batch.
                self._set_occupancy_choice(row, None)
                row.proposed_name.setText("(existing set)")
                projected_occupants[target.set_id] = row.record.dataset_id
                continue

            row.occupancy.setEnabled(True)
            choice = row.occupancy.currentData()
            if choice not in {
                OccupiedSetChoice.OVERRIDE.value,
                OccupiedSetChoice.CREATE_COPY.value,
                OccupiedSetChoice.CANCEL.value,
            }:
                self._set_occupancy_choice(row, None)
                choice = None
            if choice == OccupiedSetChoice.OVERRIDE.value:
                for set_id, current_occupant in tuple(projected_occupants.items()):
                    if current_occupant == occupant:
                        del projected_occupants[set_id]
                projected_occupants[target.set_id] = row.record.dataset_id
                row.proposed_name.setText("(existing set; override earlier mapping)")
            elif choice == OccupiedSetChoice.CREATE_COPY.value:
                proposed = allocate_initial_condition_set_name(
                    target.name,
                    existing_names=names,
                )
                row.proposed_name.setText(proposed)
                names.append(proposed)
            elif choice == OccupiedSetChoice.CANCEL.value:
                row.proposed_name.setText("(mapping canceled)")
            else:
                row.proposed_name.setText("(existing set; choose occupancy disposition)")

    @staticmethod
    def _set_occupancy_choice(
        row: _DatasetMappingRow,
        choice: str | None,
    ) -> None:
        """Reset automatic occupancy without recursively re-projecting rows."""
        target_index = row.occupancy.findData(choice)
        if target_index < 0:
            raise DatasetMappingError("Mapping review occupancy choice is unavailable.")
        if row.occupancy.currentIndex() == target_index:
            return
        blocker = QtCore.QSignalBlocker(row.occupancy)
        try:
            row.occupancy.setCurrentIndex(target_index)
        finally:
            del blocker

    def _accept_review(self) -> None:
        try:
            self.proposal()
        except (DatasetMappingError, ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.accept()

    def proposal(self) -> DatasetMappingBatchProposal:
        # Ensure choices are folded against all earlier rows even when a
        # caller changes controls programmatically without emitting Qt events.
        self._refresh_rows()
        proposals: list[DatasetMappingProposal] = []
        names = [item.name for item in self._tab.initial_condition_sets]
        projected_occupants = {
            link.set_id: dataset_id
            for dataset_id, link in self._tab.dataset_links
            if link.set_id is not None
        }
        for row in self._rows:
            action = DatasetMappingChoice(row.action.currentData())
            target_id = row.target.currentData() if action is DatasetMappingChoice.MAP_EXISTING else None
            occupied = row.occupancy.currentData() if action is DatasetMappingChoice.MAP_EXISTING else None
            proposed_name: str | None = None
            if action is DatasetMappingChoice.CREATE_SET:
                proposed_name = allocate_initial_condition_set_name(
                    row.record.display_name,
                    existing_names=names,
                )
                names.append(proposed_name)
            elif action is DatasetMappingChoice.MAP_EXISTING and occupied == OccupiedSetChoice.CREATE_COPY.value:
                target = next((item for item in self._tab.initial_condition_sets if item.set_id == target_id), None)
                if target is None:
                    raise DatasetMappingError("Map Existing requires a target set.")
                proposed_name = allocate_initial_condition_set_name(
                    target.name,
                    existing_names=names,
                )
                names.append(proposed_name)
            if action is DatasetMappingChoice.MAP_EXISTING:
                target = next(
                    (item for item in self._tab.initial_condition_sets if item.set_id == target_id),
                    None,
                )
                if target is None:
                    raise DatasetMappingError("Map Existing requires a target set.")
                occupant = projected_occupants.get(target.set_id)
                if occupant is None and occupied is not None:
                    raise DatasetMappingError(
                        "An unoccupied target cannot include an occupancy choice."
                    )
                if occupant is not None and occupied is None:
                    raise DatasetMappingError(
                        "An occupied target requires Override, Create Copy, or Cancel."
                    )
                if occupied == OccupiedSetChoice.OVERRIDE.value:
                    for set_id, current_occupant in tuple(projected_occupants.items()):
                        if current_occupant == occupant:
                            del projected_occupants[set_id]
                    projected_occupants[target.set_id] = row.record.dataset_id
                elif occupied == OccupiedSetChoice.CREATE_COPY.value:
                    # The original target remains occupied; the copied set is
                    # not selectable by later rows in this review.
                    pass
                elif occupied != OccupiedSetChoice.CANCEL.value:
                    projected_occupants[target.set_id] = row.record.dataset_id
            mapping = row.editor.confirmed_mapping()
            proposals.append(
                DatasetMappingProposal(
                    dataset_id=row.record.dataset_id,
                    action=action,
                    target_set_id=target_id,
                    occupied_choice=occupied,
                    observation_mapping=mapping,
                    proposed_set_name=proposed_name,
                    initial_concentration_source=(
                        row.editor.creation_source() if action is DatasetMappingChoice.CREATE_SET else None
                    ),
                )
            )
        return DatasetMappingBatchProposal(tab_id=self._tab.tab_id, proposals=tuple(proposals))

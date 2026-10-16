"""Review every new dataset's mapping and initial concentrations in one dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.dataset_mapping import (
    DatasetMappingBatchProposal,
    DatasetMappingChoice,
    DatasetMappingProposal,
    OccupiedSetChoice,
    allocate_initial_condition_set_name,
    apply_dataset_mapping_batch,
)
from kindred.core.datasets.foundation import RawDatasetRecord
from kindred.core.project_document import (
    MechanismTabDocument,
    ProjectDocument,
    canonical_initial_values,
    canonical_species_names,
)
from kindred.core.validation import six_digit_text
from kindred.gui.theme_manager import semantic_color
from kindred.gui.ui_helpers import (
    clamp_initial_size,
    enter_confirms_only,
    ignore_wheel,
    set_bounded_label_text,
)
from kindred.gui.widgets.observation_mapping_widget import ObservationMappingWidget

__all__ = ["DatasetMappingBatchDialog"]

_COPY_TOOLTIP = (
    "Copies this dataset's initial-concentration source, each species' Source and typed "
    "values to every other dataset. Each dataset still reads its own data; column choices "
    "and Action are not copied. Every dataset can be changed afterwards."
)


@dataclass(slots=True)
class _DatasetMappingRow:
    record: RawDatasetRecord
    short_name: str
    action: QtWidgets.QComboBox
    target: QtWidgets.QComboBox
    target_label: QtWidgets.QWidget
    occupancy: QtWidgets.QComboBox
    occupancy_label: QtWidgets.QWidget
    result: QtWidgets.QLabel
    editor: ObservationMappingWidget
    # Projected outcome, rebuilt by every projection.
    status: str = ""
    problem: str = ""
    new_set_name: str | None = None
    copy_of: str | None = None
    existing_set_id: str | None = None


class DatasetMappingBatchDialog(QtWidgets.QDialog):
    """Review every dataset mapping choice before one atomic tab replacement."""

    def __init__(
        self,
        *,
        document: ProjectDocument,
        records: tuple[RawDatasetRecord, ...],
        tab: MechanismTabDocument,
        model_targets: tuple[str, ...],
        create_only: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._create_only = create_only
        self._document = document
        self._records = records
        self._tab = tab
        self._model_targets = model_targets
        self._species = canonical_species_names(tab)
        self._sets_by_id = {item.set_id: item for item in tab.initial_condition_sets}
        self._rows: list[_DatasetMappingRow] = []
        # Dataset index of each line of the Initial-Condition Sets table.
        self._set_lines: list[int] = []
        self._refreshing = False
        self.setWindowTitle("Review New Datasets" if create_only else "Review Imported Dataset Mapping")
        self.setModal(True)
        clamp_initial_size(self, 1080, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        workspace = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        workspace.setChildrenCollapsible(False)
        layout.addWidget(workspace, 1)
        upper = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, workspace)
        upper.setChildrenCollapsible(False)

        datasets = QtWidgets.QGroupBox("Datasets", upper)
        datasets_layout = QtWidgets.QVBoxLayout(datasets)
        self._list = QtWidgets.QTreeWidget(datasets)
        self._list.setColumnCount(2)
        self._list.setHeaderLabels(["Dataset", "Result"])
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        # Long names share the width and shorten in the middle; tooltips hold the full text.
        self._list.header().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._list.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        datasets_layout.addWidget(self._list, 1)
        self._copy_btn = QtWidgets.QPushButton("Copy Settings to Other Datasets", datasets)
        datasets_layout.addWidget(self._copy_btn)

        settings = QtWidgets.QGroupBox(upper)
        settings_layout = QtWidgets.QVBoxLayout(settings)
        self._settings_label = QtWidgets.QLabel(settings)
        bold = self._settings_label.font()
        bold.setBold(True)
        self._settings_label.setFont(bold)
        settings_layout.addWidget(self._settings_label)
        self._pages = QtWidgets.QStackedWidget(settings)
        settings_layout.addWidget(self._pages, 1)
        upper.setStretchFactor(0, 0)
        upper.setStretchFactor(1, 1)
        upper.setSizes([340, 720])

        preview = QtWidgets.QGroupBox(workspace)
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_title = QtWidgets.QLabel("Initial-Condition Sets", preview)
        preview_title.setFont(bold)
        preview_layout.addWidget(preview_title)
        self._sets = QtWidgets.QTableWidget(preview)
        self._sets.setAccessibleName("Initial-condition sets")
        self._sets.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._sets.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._sets.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._sets.setAlternatingRowColors(True)
        self._sets.horizontalHeader().setStretchLastSection(False)
        self._sets.setMinimumHeight(90)
        preview_layout.addWidget(self._sets, 1)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)
        workspace.setSizes([470, 220])

        self._readiness = QtWidgets.QLabel(self)
        self._readiness.setWordWrap(True)
        layout.addWidget(self._readiness)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        if self._create_only:
            buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(f"Create {len(self._records)} Datasets")
            buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Back to Editor")
        buttons.accepted.connect(self._accept_review)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Sheets of one workbook list by sheet name unless that is ambiguous.
        shorts = [record.display_name.rpartition("::")[2] or record.display_name for record in self._records]
        for record, short in zip(self._records, shorts, strict=True):
            unique = short if shorts.count(short) == 1 else record.display_name
            self._pages.addWidget(self._build_page(record, unique))
            item = QtWidgets.QTreeWidgetItem([unique, ""])
            item.setToolTip(0, record.display_name)
            self._list.addTopLevelItem(item)
        datasets.setVisible(len(self._records) > 1)
        self._list.currentItemChanged.connect(self._on_current_changed)
        self._sets.itemSelectionChanged.connect(self._on_set_line_selected)
        self._copy_btn.clicked.connect(self._copy_settings)
        self._refresh_rows()
        self._list.setCurrentItem(self._list.topLevelItem(0))

    def _build_page(self, record: RawDatasetRecord, short_name: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self._pages)
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        action = QtWidgets.QComboBox(page)
        ignore_wheel(action)
        action.addItem("Create Set", DatasetMappingChoice.CREATE_SET.value)
        if not self._create_only:
            action.addItem("Map Existing", DatasetMappingChoice.MAP_EXISTING.value)
            action.addItem("Leave Unmapped", DatasetMappingChoice.LEAVE_UNMAPPED.value)
        target = QtWidgets.QComboBox(page)
        ignore_wheel(target)
        # Long set names must not widen the unscrolled settings page.
        target.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        target.setMinimumContentsLength(16)
        target.addItem("Choose…", None)
        for item in self._tab.initial_condition_sets:
            target.addItem(item.name, item.set_id)
        occupancy = QtWidgets.QComboBox(page)
        ignore_wheel(occupancy)
        occupancy.addItem("Choose…", None)
        occupancy.addItem("Move set to this dataset", OccupiedSetChoice.OVERRIDE.value)
        occupancy.addItem("Create a separate copy", OccupiedSetChoice.CREATE_COPY.value)
        occupancy.addItem("Cancel this occupied choice", OccupiedSetChoice.CANCEL.value)
        result = QtWidgets.QLabel(page)
        result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        # A long result ends in "…" and shows in full as its tooltip.
        set_bounded_label_text(result, "", max_width=16777215)
        form.addRow("Action", action)
        form.addRow("Target set", target)
        form.addRow("Occupied set", occupancy)
        form.addRow("Result", result)
        if self._create_only:
            action.hide()
            form.labelForField(action).hide()
        page_layout.addLayout(form)
        editor = ObservationMappingWidget(
            record, self._model_targets,
            mechanism_species=self._species, create_set=True, parent=page,
        )
        page_layout.addWidget(editor, 1)
        row = _DatasetMappingRow(
            record=record,
            short_name=short_name,
            action=action,
            target=target,
            target_label=form.labelForField(target),
            occupancy=occupancy,
            occupancy_label=form.labelForField(occupancy),
            result=result,
            editor=editor,
        )
        self._rows.append(row)
        action.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        target.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        occupancy.currentIndexChanged.connect(lambda _index: self._refresh_rows())
        editor.changed.connect(self._on_editor_changed)
        return page

    # -- projection -----------------------------------------------------------------
    def _refresh_rows(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._project_rows()
        finally:
            self._refreshing = False
        self._render()

    def _project_rows(self) -> None:
        names = [item.name for item in self._tab.initial_condition_sets]
        projected_occupants = {
            link.set_id: dataset_id
            for dataset_id, link in self._tab.dataset_links
            if link.set_id is not None
        }
        row_by_dataset = {row.record.dataset_id: row for row in self._rows}
        for row in self._rows:
            # Re-project this row's prior occupancy so a change cannot leave
            # stale occupancy behind for later rows.
            for set_id, occupant in tuple(projected_occupants.items()):
                if occupant == row.record.dataset_id:
                    del projected_occupants[set_id]
            row.problem = ""
            row.new_set_name = None
            row.copy_of = None
            row.existing_set_id = None
            action = DatasetMappingChoice(row.action.currentData())
            is_existing = action is DatasetMappingChoice.MAP_EXISTING
            row.target.setVisible(is_existing)
            row.target_label.setVisible(is_existing)
            row.occupancy.setVisible(False)
            row.occupancy_label.setVisible(False)
            if action is DatasetMappingChoice.CREATE_SET:
                row.editor.set_create_set(True)
                self._set_occupancy_choice(row, None)
                proposed = allocate_initial_condition_set_name(row.record.display_name, existing_names=names)
                names.append(proposed)
                row.new_set_name = proposed
                row.status = "New set"
                row.result.setText(f"New set “{proposed}”")
                continue
            if not is_existing:
                row.editor.set_create_set(False)
                self._set_occupancy_choice(row, None)
                row.status = "Unmapped"
                row.result.setText("No initial-condition set")
                continue
            target = self._sets_by_id.get(row.target.currentData())
            if target is None:
                row.editor.set_create_set(False)
                self._set_occupancy_choice(row, None)
                row.status = "Choose a target set"
                row.problem = "Choose a target set."
                row.result.setText("Choose a target set")
                continue
            row.editor.show_existing_set(target.name, dict(canonical_initial_values(self._tab, target.set_id)))
            occupant = projected_occupants.get(target.set_id)
            if occupant is None:
                # The target may have been free in the tab but occupied by an
                # earlier row in this same batch.
                self._set_occupancy_choice(row, None)
                projected_occupants[target.set_id] = row.record.dataset_id
                row.existing_set_id = target.set_id
                row.status = f"Mapped: {target.name}"
                row.result.setText(f"Uses existing set “{target.name}”")
                continue
            occupant_name = (
                row_by_dataset[occupant].short_name if occupant in row_by_dataset
                else self._document.dataset(occupant).display_name
            )
            row.occupancy.setVisible(True)
            row.occupancy_label.setVisible(True)
            row.occupancy.setToolTip(f"“{target.name}” is already used by {occupant_name}.")
            choice = row.occupancy.currentData()
            if choice == OccupiedSetChoice.OVERRIDE.value:
                for set_id, current_occupant in tuple(projected_occupants.items()):
                    if current_occupant == occupant:
                        del projected_occupants[set_id]
                projected_occupants[target.set_id] = row.record.dataset_id
                row.existing_set_id = target.set_id
                row.status = f"Mapped: {target.name} (moved here)"
                row.result.setText(f"Moves “{target.name}” here; {occupant_name} becomes unmapped")
                if occupant in row_by_dataset:
                    # The earlier dataset loses the set when this review is applied.
                    earlier = row_by_dataset[occupant]
                    earlier.existing_set_id = None
                    earlier.editor.set_create_set(False)
                    earlier.status = f"Unmapped ({target.name} moved to {row.short_name})"
                    earlier.result.setText(earlier.status)
            elif choice == OccupiedSetChoice.CREATE_COPY.value:
                proposed = allocate_initial_condition_set_name(target.name, existing_names=names)
                names.append(proposed)
                row.new_set_name = proposed
                row.copy_of = target.set_id
                row.status = f"New set: copy of {target.name}"
                row.result.setText(f"New set “{proposed}”, a copy of “{target.name}”")
            elif choice == OccupiedSetChoice.CANCEL.value:
                row.editor.set_create_set(False)
                row.status = "Unmapped (canceled)"
                row.result.setText("No initial-condition set (canceled)")
            else:
                row.status = f"Choose what to do with {target.name}"
                row.problem = f"“{target.name}” is already used by {occupant_name}; choose what to do with it."
                row.result.setText(f"“{target.name}” is already used by {occupant_name}")

    @staticmethod
    def _set_occupancy_choice(row: _DatasetMappingRow, choice: str | None) -> None:
        """Reset automatic occupancy without recursively re-projecting rows."""
        target_index = row.occupancy.findData(choice)
        if row.occupancy.currentIndex() == target_index:
            return
        blocker = QtCore.QSignalBlocker(row.occupancy)
        try:
            row.occupancy.setCurrentIndex(target_index)
        finally:
            del blocker

    def _row_problem(self, row: _DatasetMappingRow) -> str:
        if row.problem:
            return row.problem
        if DatasetMappingChoice(row.action.currentData()) is DatasetMappingChoice.CREATE_SET:
            problems = row.editor.problems()
            if problems:
                return problems[0]
        return ""

    # -- rendering -------------------------------------------------------------------
    def _warning_color(self) -> QtGui.QColor:
        dark = self.palette().color(QtGui.QPalette.ColorRole.Text).lightness() > 128
        return semantic_color("warning", dark)

    def _render(self) -> None:
        self._readiness.clear()
        warning = self._warning_color()
        for index, row in enumerate(self._rows):
            item = self._list.topLevelItem(index)
            if item is None:
                continue
            problem = self._row_problem(row)
            item.setText(1, f"! {row.status}" if problem else row.status)
            item.setToolTip(1, problem or row.status)
            item.setForeground(1, warning if problem else self.palette().color(QtGui.QPalette.ColorRole.Text))
        self._render_sets()
        self._render_header()
        self._render_copy_button()

    def _render_copy_button(self) -> None:
        index = self._pages.currentIndex()
        creating = index >= 0 and (
            DatasetMappingChoice(self._rows[index].action.currentData()) is DatasetMappingChoice.CREATE_SET
        )
        self._copy_btn.setEnabled(creating)
        self._copy_btn.setToolTip(
            _COPY_TOOLTIP if creating
            else "Only a dataset set to Create Set has initial-concentration settings to copy."
        )

    def _render_header(self) -> None:
        index = self._pages.currentIndex()
        if index >= 0:
            set_bounded_label_text(self._settings_label, f"Mapping for: {self._rows[index].short_name}", max_width=700)

    def _render_sets(self) -> None:
        """One line per set the review creates or maps, with its starting values."""
        current = self._pages.currentIndex()
        lines = [
            (index, row) for index, row in enumerate(self._rows)
            if row.new_set_name is not None or row.existing_set_id is not None or row.problem
        ]
        species = self._species
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        warning = self._warning_color()
        blocker = QtCore.QSignalBlocker(self._sets)
        try:
            self._sets.clear()
            self._sets.clearSpans()
            self._sets.setColumnCount(len(species) + 1)
            self._sets.setRowCount(len(lines))
            self._sets.setHorizontalHeaderLabels(["Set", *(f"{name} (M)" for name in species)])
            self._set_lines = [index for index, _row in lines]
            messages: list[tuple[int, str]] = []
            for line, (_index, row) in enumerate(lines):
                problem = self._row_problem(row)
                if row.existing_set_id is not None:
                    label = f"{self._sets_by_id[row.existing_set_id].name} (existing)"
                else:
                    label = row.new_set_name or row.short_name
                name = QtWidgets.QTableWidgetItem(f"! {label}" if problem else label)
                name.setToolTip(f"{label}\nDataset: {row.record.display_name}")
                self._sets.setItem(line, 0, name)
                if problem:
                    name.setForeground(warning)
                    messages.append((line, problem))
                    continue
                kept = row.copy_of or row.existing_set_id
                if kept is not None:
                    note = "Copied from the existing set." if row.copy_of else "The existing set's values are kept."
                    starts = [(value, note) for _name, value in canonical_initial_values(self._tab, kept)]
                else:
                    starts = [(start.value, start.note) for start in row.editor.starts()]
                for column, (value, note) in enumerate(starts, 1):
                    cell = QtWidgets.QTableWidgetItem("0" if value is None else six_digit_text(value))
                    cell.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
                    if value is None:
                        # Not supplied: it starts at 0, unlike a measured 0.
                        cell.setForeground(muted)
                        italic = cell.font()
                        italic.setItalic(True)
                        cell.setFont(italic)
                        cell.setToolTip(note or "No concentration supplied; starts at 0 M.")
                    elif note:
                        cell.setToolTip(note)
                    self._sets.setItem(line, column, cell)
            self._sets.resizeColumnsToContents()
            # Problem text spans the value columns without widening them.
            for line, problem in messages:
                message = QtWidgets.QTableWidgetItem(problem)
                message.setForeground(warning)
                message.setToolTip(problem)
                self._sets.setItem(line, 1, message)
                if len(species) > 1:
                    self._sets.setSpan(line, 1, 1, len(species))
            if current in self._set_lines:
                self._sets.selectRow(self._set_lines.index(current))
            else:
                self._sets.clearSelection()
        finally:
            del blocker

    # -- interaction -----------------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override
        # Enter confirms a typed value; only OK itself applies the review.
        if not enter_confirms_only(self, event):
            super().keyPressEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) and self._rows:
            self._render()

    def _on_current_changed(self, current, _previous) -> None:
        if current is None:
            return
        self._pages.setCurrentIndex(self._list.indexOfTopLevelItem(current))
        self._render_header()
        self._render_sets()
        self._render_copy_button()

    def _on_set_line_selected(self) -> None:
        selected = self._sets.selectionModel().selectedRows()
        if not selected:
            return
        index = self._set_lines[selected[0].row()]
        if index != self._pages.currentIndex():
            self._list.setCurrentItem(self._list.topLevelItem(index))

    def _on_editor_changed(self) -> None:
        if not self._refreshing:
            self._render()

    def _copy_settings(self) -> None:
        template = self._rows[self._pages.currentIndex()]
        settings = template.editor.source_settings()
        self._refreshing = True
        try:
            for row in self._rows:
                if row is not template:
                    row.editor.apply_source_settings(settings)
        finally:
            self._refreshing = False
        self._render()

    def _accept_review(self) -> None:
        for index, row in enumerate(self._rows):
            problem = self._row_problem(row)
            if problem:
                self._list.setCurrentItem(self._list.topLevelItem(index))
                self._readiness.setText(f"{row.short_name}: {problem}")
                return
        # The core fold is the single owner of occupancy and naming rules.
        try:
            apply_dataset_mapping_batch(
                document=self._document,
                tab=self._tab,
                batch=self.proposal(),
                model_targets=self._model_targets,
            )
        except ValueError as exc:
            self._readiness.setText(str(exc))
            return
        self.accept()

    def proposal(self) -> DatasetMappingBatchProposal:
        """Project the current controls into one mapping batch proposal."""

        proposals: list[DatasetMappingProposal] = []
        for row in self._rows:
            action = DatasetMappingChoice(row.action.currentData())
            is_existing = action is DatasetMappingChoice.MAP_EXISTING
            source, manual = (
                row.editor.creation_inputs() if action is DatasetMappingChoice.CREATE_SET else (None, ())
            )
            proposals.append(
                DatasetMappingProposal(
                    dataset_id=row.record.dataset_id,
                    action=action,
                    target_set_id=row.target.currentData() if is_existing else None,
                    occupied_choice=row.occupancy.currentData() if is_existing else None,
                    observation_mapping=row.editor.confirmed_mapping(),
                    initial_concentration_source=source,
                    manual_initial_concentrations=manual,
                )
            )
        return DatasetMappingBatchProposal(tab_id=self._tab.tab_id, proposals=tuple(proposals))

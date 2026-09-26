# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.foundation import ImportedSource
from kindred.core.datasets.csv_import import read_csv_rows_from_bytes
from kindred.core.datasets.excel_import import (
    ExcelWorkbookPair,
)
from kindred.gui.import_workflow import (
    ImportConfig,
    ImportSettings,
    capture_import_source,
    resolve_sheet_plan,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog, ImportDialogResult
from kindred.gui.dataset_payload_snapshot import (
    ActiveDatasetLinkSnapshot,
    DataManagerSnapshot,
)
from kindred.core.datasets.foundation import (
    RawDatasetRecord,
    thaw_json_value,
)
from kindred.core.project_document import (
    ProjectDocument,
    canonical_observation_target_names,
)
from kindred.gui.ui_helpers import ControlRowsLayout
from kindred.gui.display_name_policy import (
    literal_tooltip,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DataManagerPanel",
]


class _DatasetSelector(QtWidgets.QComboBox):
    """Keep the closed selector compact and expand filenames in its popup."""

    def __init__(self):
        super().__init__()
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMaxVisibleItems(12)

    def showPopup(self):
        text_width = max((self.fontMetrics().horizontalAdvance(self.itemText(i))
                          for i in range(self.count())), default=0)
        self.view().setMinimumWidth(min(
            max(self.width(), text_width + 40),
            self.screen().availableGeometry().width() - 24,
        ))
        super().showPopup()


class _DatasetControls(QtWidgets.QScrollArea):
    """Allow the control form to shrink to one font-sized action row."""

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        hint.setHeight(
            self.fontMetrics().height() + 8 + 2 * self.frameWidth()
            + self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent)
        )
        return hint


class _DatasetActions(QtWidgets.QWidget):
    """Wrap the standard dataset buttons at the available dock width."""

    def __init__(self, buttons):
        super().__init__()
        self._rows = ControlRowsLayout(self)
        for button in buttons:
            self._rows.addWidget(button)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setFixedHeight(self._rows.heightForWidth(self.width()))


class _MappingField(QtWidgets.QWidget):
    """Keep each value and button together, including at enlarged font sizes."""

    def __init__(self, value: QtWidgets.QWidget, action: QtWidgets.QPushButton):
        super().__init__()
        self._value = value
        self._action = action
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        self._grid.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self._grid.addWidget(value, 0, 0)
        self._grid.addWidget(action, 0, 1)
        self._grid.setColumnStretch(0, 1)
        self._wide = True

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            max(self._value.minimumWidth(), self._action.sizeHint().width()),
            self._action.sizeHint().height(),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        wide = self.width() >= self._value.minimumWidth() + self._action.sizeHint().width() + 8
        if wide != self._wide:
            self._wide = wide
            self._grid.removeWidget(self._action)
            self._grid.addWidget(self._action, 0 if wide else 1, 1 if wide else 0, Qt.AlignRight)
            self.updateGeometry()


class _DatasetName(QtWidgets.QLabel):
    """Show a compact name while keeping its full text available."""

    def set_name(self, name: str) -> None:
        self._name = name
        self.setTextFormat(Qt.PlainText)
        self.setToolTip(literal_tooltip(name))
        self.setAccessibleName(name)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(
            getattr(self, "_name", ""), Qt.ElideRight, self.contentsRect().width()
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


def _source_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    return label


class _SourceDialog(QtWidgets.QDialog):
    """Present the dataset's origin and retained measurement units."""

    def __init__(self, record: RawDatasetRecord, source: ImportedSource | None, parent):
        super().__init__(parent)
        self.setWindowTitle("File Details" if source is not None else "Dataset Details")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        metadata = thaw_json_value(record.source_metadata)
        original = QtWidgets.QGroupBox("Original file" if source is not None else "Created in Kindred")
        form = QtWidgets.QFormLayout(original)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setVerticalSpacing(6)
        source_names = {}
        if source is not None:
            fields = [
                ("Filename", source.original_filename),
                ("Original filepath", source.original_filepath),
                ("File type", "Excel (XLSX)" if source.original_format == "xlsx" else "CSV"),
            ]
            if "sheet_name" in metadata:
                fields.append(("Worksheet", metadata["sheet_name"]))
            source_names = {column["assigned_name"]: column["authored_name"] for column in metadata["physical_columns"]}
            if metadata["time_column"] is not None:
                fields.extend([
                    ("Time column", source_names[metadata["time_column"]]),
                    ("Time unit", metadata["original_time_unit"]),
                ])
        else:
            fields = [("Dataset", record.display_name), ("Origin", "Created with the spreadsheet editor")]
        times = {time for series in record.observations for time in series.t}
        if times:
            fields.append(("Distinct recorded timestamps", str(len(times))))
        else:
            fields.append(("Time", "Not recorded"))
        for title, value in fields:
            form.addRow(title, _source_label(value))
        layout.addWidget(original)
        layout.addWidget(QtWidgets.QLabel("Measured columns"))
        table = QtWidgets.QTableWidget(len(record.observations), 4 if source is not None else 3)
        table.setAccessibleName("Measured columns, measurement counts and units")
        table.setHorizontalHeaderLabels(["Column in file", "Measurements", "Imported unit", "Converted to"] if source is not None else ["Column", "Measurements", "Scientific unit"])
        table.verticalHeader().hide()
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        units = metadata.get("original_concentration_units", {}) if metadata is not None else {}
        for row, series in enumerate(record.observations):
            converted = "Dimensionless" if series.unit.symbol == "1" else series.unit.symbol
            values = (series.column_name, str(len(series.y)), converted)
            if source is not None:
                imported = units[series.column_name]
                values = (source_names[series.column_name], str(len(series.y)),
                          "Dimensionless" if imported == "1" else imported, converted)
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(literal_tooltip(value))
                table.setItem(row, column, item)
        layout.addWidget(table, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(600, 600)


class _ICSetDialog(QtWidgets.QDialog):
    CREATE_FROM_DATA = 2

    def __init__(self, available_sets, current_set_id, parent):
        super().__init__(parent)
        self.setWindowTitle("Assign IC Set")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.sets = QtWidgets.QComboBox()
        self.sets.setAccessibleName("Existing IC set")
        for set_id, name in available_sets:
            self.sets.addItem(name, set_id)
        if current_set_id is not None:
            self.sets.setCurrentIndex(self.sets.findData(current_set_id))
        if not available_sets:
            self.sets.addItem("No IC sets available")
            self.sets.setEnabled(False)
        form.addRow("Existing IC set", self.sets)
        layout.addLayout(form)
        create = QtWidgets.QPushButton("Create IC set from data…")
        create.setAutoDefault(False)
        create.clicked.connect(lambda: self.done(self.CREATE_FROM_DATA))
        layout.addWidget(create, 0, Qt.AlignLeft)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        assign = buttons.addButton("Assign", QtWidgets.QDialogButtonBox.AcceptRole)
        assign.setEnabled(bool(available_sets) and self.sets.currentIndex() >= 0)
        self.sets.currentIndexChanged.connect(lambda index: assign.setEnabled(bool(available_sets) and index >= 0))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(440, self.sizeHint().height())


class DataManagerPanel(QtWidgets.QWidget):
    """
    Browse committed datasets and assign their initial sets and observations.

    The panel retains only rendered snapshots and import UI configuration.
    The application import owner retains workers and uncommitted output;
    ProjectSession callbacks own every canonical mutation.
    """

    datasetRenameRequested = QtCore.Signal(str, str)
    datasetDeleteRequested = QtCore.Signal(str)
    datasetExportOriginalRequested = QtCore.Signal(str)
    datasetMappingRequested = QtCore.Signal(str, str, object, object)
    datasetObservationMappingRequested = QtCore.Signal(str)
    datasetImportRequested = QtCore.Signal(str, object)
    datasetCreationRequested = QtCore.Signal(str, str)

    def __init__(
        self,
        *,
        origin_tab_id: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """
        Initialize data manager panel.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        if (
            type(origin_tab_id) is not str
            or not origin_tab_id
            or origin_tab_id != origin_tab_id.strip()
        ):
            raise ValueError(
                "DataManagerPanel requires one stable edge-normalized origin "
                "tab ID."
            )
        self._origin_tab_id = origin_tab_id

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.import_action = QtGui.QAction("Import Data…", self)
        self.import_action.setToolTip("Import experimental data from CSV or Excel files.")
        self.import_action.setObjectName("loadDataAction")
        self.import_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+L"))
        self.import_action.triggered.connect(self._load_dataset)
        self.addAction(self.import_action)
        self._import_btn = QtWidgets.QPushButton(self.import_action.text())
        self._empty_import_btn = QtWidgets.QPushButton(self.import_action.text())
        for button in (self._import_btn, self._empty_import_btn):
            button.setToolTip(self.import_action.toolTip())
            button.clicked.connect(self.import_action.trigger)
        self.import_action.changed.connect(self._sync_import_buttons)
        self.create_action = QtGui.QAction("Create from Data…", self)
        self.create_action.setToolTip("Open selected datasets in a temporary spreadsheet to create new experimental datasets.")
        self.create_action.triggered.connect(lambda: self.datasetCreationRequested.emit(self._origin_tab_id, self._selected_dataset_id()))
        self._create_btn = QtWidgets.QPushButton(self.create_action.text())
        self._create_btn.setToolTip(self.create_action.toolTip())
        self._create_btn.clicked.connect(self.create_action.trigger)
        self.create_action.changed.connect(lambda: self._create_btn.setEnabled(self.create_action.isEnabled()))
        self._creation_busy = False
        self._rename_btn = QtWidgets.QPushButton("Rename…")
        self._rename_btn.setToolTip("Change this dataset’s name in Kindred.")
        self._rename_btn.clicked.connect(self._rename_dataset)
        self._export_original_btn = QtWidgets.QPushButton("Export Original…")
        self._export_original_btn.setToolTip("Save a copy of the original imported file, with its original contents and units.")
        self._export_original_btn.clicked.connect(self._export_original)
        self._remove_btn = QtWidgets.QPushButton("Delete…")
        self._remove_btn.setToolTip("Remove this dataset from the project, including its mappings and plot overlays across all tabs.")
        self._remove_btn.clicked.connect(self._remove_dataset)
        self._source_btn = QtWidgets.QPushButton("File Details…")
        self._source_btn.setToolTip("View the original filename, import file path, columns, and units.")
        self._source_btn.clicked.connect(self._show_source)
        self._browser = QtWidgets.QWidget()
        browser_layout = QtWidgets.QVBoxLayout(self._browser)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.setSpacing(8)
        selector_row = QtWidgets.QFormLayout()
        selector_row.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        self._dataset_selector = _DatasetSelector()
        self._dataset_selector.setAccessibleName("Dataset")
        self._dataset_selector.currentIndexChanged.connect(self._on_dataset_selected)
        self._dataset_selector.setMinimumWidth(self.fontMetrics().horizontalAdvance("dataset"))
        selector_row.addRow("Dataset", _MappingField(self._dataset_selector, self._import_btn))
        browser_layout.addLayout(selector_row)
        browser_layout.addWidget(self._create_btn)
        self._actions = _DatasetActions([
            self._rename_btn, self._export_original_btn, self._source_btn, self._remove_btn,
        ])
        self._actions.layout().itemAt(3).setAlignment(Qt.AlignRight)
        browser_layout.addWidget(self._actions)
        self._browser.hide()
        layout.addWidget(self._browser)

        self._empty = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self._empty)
        empty_layout.addStretch()
        label = QtWidgets.QLabel("Import experimental data to get started.")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(label)
        empty_layout.addWidget(self._empty_import_btn, 0, Qt.AlignCenter)
        formats = QtWidgets.QLabel("CSV or Excel files")
        formats.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(formats)
        empty_layout.addStretch()
        layout.addWidget(self._empty)

        self._details_scroll = _DatasetControls()
        self._details_scroll.setWidgetResizable(True)
        self._details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._details_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        self._dataset_summary = _source_label("")
        self._dataset_summary.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self._tab_name = _DatasetName()
        self._tab_name.setMinimumWidth(self.fontMetrics().horizontalAdvance("Mechanism"))
        self._tab_name.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        for title, value in (("Tab", self._tab_name), ("Data", self._dataset_summary)):
            label = QtWidgets.QLabel(title)
            label_font = label.font()
            label_font.setBold(True)
            label.setFont(label_font)
            if title == "Tab":
                label.setToolTip("The mechanism tab these mappings apply to.")
            form.addRow(label, value)
        self._change_set_btn = QtWidgets.QPushButton("Change IC Set…")
        self._change_set_btn.setToolTip("Map this dataset to an existing initial-condition set, or create one from its values at t=0.")
        self._change_set_btn.clicked.connect(self._change_ic_set)
        self._set_value = _DatasetName()
        self._observation_mapping_btn = QtWidgets.QPushButton("Map Columns…")
        self._observation_mapping_btn.setToolTip("Choose which measured columns correspond to model species or observables in this tab.")
        self._observation_mapping_btn.clicked.connect(self._request_observation_mapping)
        self._observation_value = _source_label("")
        for title, value, action, tooltip in (
            ("Initial-condition set", self._set_value, self._change_set_btn,
             "The initial-condition set mapped to this dataset in the current tab."),
            ("Columns → Model", self._observation_value, self._observation_mapping_btn,
             "Number of measured columns with a confirmed mapping to the model in the current tab."),
        ):
            value.setMinimumWidth(self.fontMetrics().horizontalAdvance("8 of 8 mapped"))
            value.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
            label = QtWidgets.QLabel(title)
            label_font = label.font()
            label_font.setBold(True)
            label.setFont(label_font)
            label.setToolTip(tooltip)
            form.addRow(label, _MappingField(value, action))
        controls_layout.addLayout(form)
        self._mapping_status = _source_label("")
        controls_layout.addWidget(self._mapping_status)
        controls_layout.addStretch()
        self._details_scroll.setWidget(controls)
        self._details_scroll.hide()
        layout.addWidget(self._details_scroll, 1)
        self._snapshot = DataManagerSnapshot((), None, (), ())
        self._visible_records_by_id: Dict[str, RawDatasetRecord] = {}
        self._mechanism_name = ""
        self._mapping_available = False
        self._import_busy = False
        self._update_mapping_area()

    def set_creation_busy(self, busy: bool) -> None:
        self._creation_busy = busy
        self._update_mapping_area()

    def _sync_import_buttons(self) -> None:
        for button in (self._import_btn, self._empty_import_btn):
            button.setEnabled(self.import_action.isEnabled())

    def _show_source(self) -> None:
        record = self._visible_records_by_id.get(self._selected_dataset_id())
        if record is None:
            return
        selector = self._dataset_selector
        dialog = _SourceDialog(record, selector.currentData(Qt.UserRole + 2), self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _show_message(
        self,
        icon: QtWidgets.QMessageBox.Icon,
        title: str,
        message: object,
    ) -> None:
        """Show one literal message, including messages containing markup."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(str(title))
        box.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        box.setText(str(message))
        box.exec()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clear_projection(self) -> None:
        """Clear only rendered Data state and revoke all Data actions."""
        self._snapshot = DataManagerSnapshot((), None, (), ())
        self._visible_records_by_id.clear()
        self._dataset_selector.clear()
        self._browser.hide()
        self._details_scroll.hide()
        self._empty.show()
        self._dataset_summary.clear()
        self._mapping_available = False
        self._mechanism_name = ""
        self._update_mapping_area()

    def render_snapshot(
        self,
        snapshot: DataManagerSnapshot,
        document: ProjectDocument,
    ) -> None:
        """Render the catalog while retaining the selected dataset identity."""
        selected_id = self._selected_dataset_id()
        previous_row = self._dataset_selector.currentIndex()
        self._snapshot = snapshot
        tab = document.tab(snapshot.active_tab_id) if snapshot.active_tab_id is not None else None
        self._mechanism_name = tab.name if tab is not None else ""
        self._mapping_available = tab is not None and bool(canonical_observation_target_names(tab))
        self._visible_records_by_id = {record.dataset_id: record for record in snapshot.datasets}
        blocker = QtCore.QSignalBlocker(self._dataset_selector)
        self._dataset_selector.clear()
        selected_row = -1
        for row, record in enumerate(snapshot.datasets):
            self._dataset_selector.addItem(record.display_name, record.dataset_id)
            self._dataset_selector.setItemData(row, literal_tooltip(record.display_name), Qt.ToolTipRole)
            self._dataset_selector.setItemData(row, record.display_name, Qt.AccessibleTextRole)
            source = document.imported_source(record.source_id) if record.source_id is not None else None
            self._dataset_selector.setItemData(row, source, Qt.UserRole + 2)
            if record.dataset_id == selected_id:
                selected_row = row
        if snapshot.datasets:
            self._dataset_selector.setCurrentIndex(selected_row if selected_row >= 0 else min(max(0, previous_row), len(snapshot.datasets) - 1))
        del blocker
        self._empty.setVisible(not snapshot.datasets)
        self._browser.setVisible(bool(snapshot.datasets))
        self._details_scroll.setVisible(bool(snapshot.datasets))
        self._on_dataset_selected(self._dataset_selector.currentIndex())

    def _load_dataset(self):
        """Load dataset(s) using per-file import configuration."""
        if self._import_busy:
            return
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Dataset(s)",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)",
        )

        if not filenames:
            return
        sources: List[ImportedSource] = []
        rejected_sources: List[Tuple[str, str]] = []
        for filepath in filenames:
            try:
                sources.append(capture_import_source(filepath))
            except ValueError as exc:
                rejected_sources.append((str(filepath), str(exc)))

        if rejected_sources:
            rejection_details = "\n\n".join(
                f"{filename}:\n{diagnostic}"
                for filename, diagnostic in rejected_sources
            )
            self._show_message(
                QtWidgets.QMessageBox.Icon.Critical,
                "Unsupported Dataset Source",
                "The following selected files were rejected:\n\n"
                f"{rejection_details}",
            )

        if not sources:
            return

        configs: List[ImportConfig] = []
        index = 0
        while index < len(sources):
            source = sources[index]
            remaining = len(sources) - index - 1
            result = self._collect_import_config(source, remaining)
            if result.action == "cancel":
                return
            if result.action == "skip":
                index += 1
                continue
            if result.config is None:
                index += 1
                continue
            configs.append(result.config)
            source_settings = result.config.remaining_file_template
            if source_settings is not None:
                source_sheet_names = tuple(
                    plan.sheet_name for plan in result.config.plans
                    if plan.sheet_name is not None
                )
                for remaining_idx in range(index + 1, len(sources)):
                    remaining_source = sources[remaining_idx]
                    try:
                        remaining_config = self._build_remaining_file_config(
                            remaining_source,
                            source_settings,
                            source_sheet_names=source_sheet_names,
                        )
                    except (ValueError, OSError) as exc:
                        self._show_message(
                            QtWidgets.QMessageBox.Icon.Critical,
                            "Import Error",
                            f"Cannot apply settings to "
                            f"'{remaining_source.original_filename}':\n\n{exc}",
                        )
                        continue
                    configs.append(remaining_config)
                break
            index += 1

        if not configs:
            return

        expected_count = sum(len(config.plans) for config in configs)
        dataset_label = f"{expected_count} dataset{'s' if expected_count != 1 else ''}"
        logger.info("User initiated dataset import: %s", dataset_label)
        self.datasetImportRequested.emit(self._origin_tab_id, tuple(configs))

    def _collect_import_config(
        self,
        source: ImportedSource,
        remaining_count: int,
    ) -> ImportDialogResult:
        try:
            dialog = ImportConfigDialog(
                source,
                remaining_count=remaining_count,
                parent=self,
            )
        except Exception as exc:
            self._show_message(
                QtWidgets.QMessageBox.Icon.Critical,
                "Load Error",
                f"Failed to open import configuration for "
                f"'{source.original_filename}':\n\n{type(exc).__name__}: {exc}",
            )
            return ImportDialogResult(config=None, action="skip")
        try:
            dialog.exec()
            return dialog.get_result()
        finally:
            dialog.deleteLater()

    def _build_remaining_file_config(
        self,
        source: ImportedSource,
        source_settings: ImportSettings,
        *,
        source_sheet_names: Tuple[str, ...] = (),
    ) -> ImportConfig:
        """Resolve shared choices against each remaining source's first row."""
        plans = []
        if source.original_format == "csv":
            columns, rows = read_csv_rows_from_bytes(
                source.exact_source_bytes,
                source_label=source.original_filename,
            )
            try:
                first_row = next(iter(rows), None)
            finally:
                rows.close()
            plans.append(resolve_sheet_plan(None, columns, first_row.values if first_row else None, source_settings))
        else:
            with ExcelWorkbookPair(
                source.exact_source_bytes,
                source_label=source.original_filename,
            ) as workbooks:
                sheets = workbooks.sheet_names
                if not sheets:
                    raise ValueError("Excel file has no sheets")
                if source_sheet_names:
                    sheets = [name for name in sheets if name in source_sheet_names]
                    if not sheets:
                        raise ValueError("No matching sheets found")
                for sheet_name in sheets:
                    rows = workbooks.read_sheet_rows(sheet_name)
                    try:
                        first_row = next(iter(rows), None)
                    finally:
                        rows.close()
                    columns = tuple(first_row) if first_row is not None else ()
                    plans.append(resolve_sheet_plan(
                        sheet_name, columns, first_row, source_settings,
                    ))
        return ImportConfig(source=source, plans=tuple(plans))

    def set_import_busy(self, busy: bool) -> None:
        """Project application-global import custody onto every import entry."""
        self._import_busy = bool(busy)
        self.import_action.setEnabled(not self._import_busy)

    def _remove_dataset(self):
        """Emit one global-delete intent; confirmation belongs to integration."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetDeleteRequested.emit(dataset_id)

    def _export_original(self) -> None:
        """Request the integration-owned destination dialog for exact bytes."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetExportOriginalRequested.emit(dataset_id)

    def _rename_dataset(self) -> None:
        dataset_id = self._selected_dataset_id()
        record = self._visible_records_by_id.get(dataset_id)
        if record is None:
            return
        value, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Rename Dataset",
            "Dataset name:",
            text=str(record.display_name),
        )
        if accepted:
            self.datasetRenameRequested.emit(dataset_id, str(value))

    def _on_dataset_selected(self, index):
        record = self._visible_records_by_id.get(self._selected_dataset_id())
        self._dataset_selector.setToolTip(literal_tooltip(record.display_name) if record else "")
        if record is None:
            self._dataset_summary.clear()
        else:
            rows = {row for series in record.observations for row in series.source_rows}
            timing = "recorded time" if any(series.t for series in record.observations) else "no recorded time"
            column_label = "column" if len(record.observations) == 1 else "columns"
            self._dataset_summary.setText(
                f"{len(rows)} observations · {len(record.observations)} measured {column_label} · {timing}"
            )
        self._update_mapping_area()

    def _selected_dataset_id(self) -> str:
        return self._dataset_selector.currentData() or ""

    def _active_link(self, dataset_id: str) -> ActiveDatasetLinkSnapshot | None:
        return self._snapshot.link_by_dataset_id().get(str(dataset_id))

    def _update_mapping_area(self) -> None:
        dataset_id = self._selected_dataset_id()
        record = self._visible_records_by_id.get(dataset_id)
        enabled = record is not None and self._mapping_available
        for action in (self._rename_btn, self._remove_btn, self._export_original_btn, self._source_btn):
            action.setEnabled(record is not None)
        self.create_action.setEnabled(bool(self._visible_records_by_id) and self._mapping_available and not self._creation_busy)
        self._export_original_btn.setEnabled(record is not None and record.source_id is not None)
        self._source_btn.setText("File Details…" if record is None or record.source_id is not None else "Dataset Details…")
        self._source_btn.setToolTip("View dataset measurements, units, and origin.")
        self._change_set_btn.setEnabled(enabled)
        self._observation_mapping_btn.setEnabled(enabled)
        link = self._active_link(dataset_id)
        self._set_value.set_name(link.set_name or link.set_id if link and link.set_id else "No IC set mapped")
        mapped_ids = set(dict(link.observation_mapping)) if link else set()
        total = len(record.observations) if record else 0
        assigned = sum(series.raw_observation_id in mapped_ids for series in record.observations) if record else 0
        self._observation_value.setText(f"{assigned} of {total} mapped")
        self._tab_name.set_name(self._mechanism_name or "No active mechanism")
        self._mapping_status.setVisible(not self._mapping_available)
        if not self._mechanism_name:
            self._mapping_status.setText("No active mechanism. Datasets remain available; select a mechanism to assign data.")
        elif not self._mapping_available:
            self._mapping_status.setText("Apply a valid mechanism with observation targets to assign data.")
        else:
            self._mapping_status.clear()

    def _emit_mapping_action(
        self,
        action: str,
        *,
        set_id: str | None = None,
        occupied_choice: str | None = None,
    ) -> None:
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetMappingRequested.emit(
            dataset_id,
            str(action),
            set_id,
            occupied_choice,
        )

    def _change_ic_set(self) -> None:
        link = self._active_link(self._selected_dataset_id())
        dialog = _ICSetDialog(self._snapshot.available_sets, link.set_id if link else None, self)
        try:
            result = dialog.exec()
            set_id = dialog.sets.currentData()
            label = dialog.sets.currentText()
        finally:
            dialog.deleteLater()
        if result == _ICSetDialog.CREATE_FROM_DATA:
            self._emit_mapping_action("create_set")
            return
        if result != QtWidgets.QDialog.Accepted:
            return
        selected_dataset_id = self._selected_dataset_id()
        occupant = next(
            (
                link.dataset_id
                for link in self._snapshot.active_links
                if link.set_id == set_id and link.dataset_id != selected_dataset_id
            ),
            None,
        )
        occupied_choice = None
        if occupant is not None:
            occupant_name = next(
                record.display_name
                for record in self._snapshot.datasets
                if record.dataset_id == occupant
            )
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            box.setWindowTitle("Initial-condition set already in use")
            box.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            box.setText(
                f'"{label}" is already mapped to "{occupant_name}" in this '
                "mechanism tab. Each dataset must use its own initial-condition set."
            )
            box.setInformativeText(
                f'Moving the set unmaps "{occupant_name}".'
            )
            override_button = box.addButton(
                "Move set to this dataset",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            copy_button = box.addButton(
                "Create a separate copy",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            cancel_button = box.addButton(
                "Cancel",
                QtWidgets.QMessageBox.ButtonRole.RejectRole,
            )
            box.setDefaultButton(cancel_button)
            box.exec()
            clicked = box.clickedButton()
            if clicked is override_button:
                occupied_choice = "override"
            elif clicked is copy_button:
                occupied_choice = "create_copy"
            else:
                return
        self._emit_mapping_action(
            "map_existing",
            set_id=set_id,
            occupied_choice=occupied_choice,
        )

    def _request_observation_mapping(self) -> None:
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetObservationMappingRequested.emit(dataset_id)

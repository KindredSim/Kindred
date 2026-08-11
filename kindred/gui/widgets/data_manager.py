# kindred/gui/widgets/data_manager.py
"""Data manager panel for loading and managing experimental datasets."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from kindred.core.datasets.source_rules import (
    PhysicalColumn,
)
from kindred.core.datasets.csv_import import read_csv_rows_from_bytes
from kindred.core.datasets.excel_import import (
    ExcelWorkbookPair,
)
from kindred.gui.import_workflow import (
    ImportConfig,
    ImportSourceSnapshot,
    SheetImportIntent,
    UnitDetection,
    UserImportIntent,
    capture_import_source,
    detect_units_from_physical_row,
    rebuild_intent_for_target,
    resolve_import_plans,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog, ImportDialogResult
from kindred.gui.controllers.dataset_errors import (
    DatasetImportAdmissionError,
)
from kindred.gui.dataset_import_staging import (
    DatasetImportCancellation,
    DatasetImportFailure,
    DatasetImportProgress,
    DatasetImportRequest,
    DatasetImportStaging,
    DatasetImportWorkerAdmission,
    DatasetImportWorkerRequest,
)
from kindred.gui.dataset_payload_snapshot import (
    ActiveDatasetLinkSnapshot,
    DataManagerSnapshot,
)
from kindred.core.datasets.foundation import ImportedSource, RawDatasetRecord, thaw_json_value
from kindred.gui.display_name_policy import DATASET_LIST_LABEL_MAX_CHARS, compact_dataset_label

logger = logging.getLogger(__name__)

__all__ = [
    "DataManagerPanel",
    "DatasetDeleteIntent",
    "DatasetExportOriginalIntent",
    "DatasetFitInclusionIntent",
    "DatasetMappingIntent",
    "DatasetObservationMappingIntent",
    "DatasetPlotOverlayIntent",
    "DatasetRenameIntent",
]


@dataclass(frozen=True)
class DatasetRenameIntent:
    dataset_id: str
    display_name: str


@dataclass(frozen=True)
class DatasetDeleteIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetExportOriginalIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetMappingIntent:
    dataset_id: str
    action: str
    set_id: str | None = None
    occupied_choice: str | None = None


@dataclass(frozen=True)
class DatasetObservationMappingIntent:
    dataset_id: str


@dataclass(frozen=True)
class DatasetPlotOverlayIntent:
    dataset_id: str

    def __post_init__(self) -> None:
        if (
            type(self.dataset_id) is not str
            or not self.dataset_id
            or self.dataset_id != self.dataset_id.strip()
        ):
            raise ValueError(
                "Plot Overlay navigation requires one stable dataset ID."
            )


@dataclass(frozen=True)
class DatasetFitInclusionIntent:
    dataset_id: str
    include: bool


class DataManagerPanel(QtWidgets.QWidget):
    """
    Import panel for loading and previewing committed experimental datasets.

    Features:
    - Load CSV and Excel files with experimental data
    - Auto-detect time column (tries: time, time_s, t, Time, T, x)
    - Extract all numeric columns as species
    - Preview loaded datasets
    - Import configuration dialog with unit selection
    - Multiple dataset support

    The panel retains only rendered snapshots and import UI configuration.
    DatasetImportStaging owns workers and uncommitted output; ProjectSession
    callbacks own every canonical mutation.
    """

    datasetRenameRequested = QtCore.Signal(object)
    datasetDeleteRequested = QtCore.Signal(object)
    datasetExportOriginalRequested = QtCore.Signal(object)
    datasetMappingRequested = QtCore.Signal(object)
    datasetObservationMappingRequested = QtCore.Signal(object)
    datasetPlotOverlayRequested = QtCore.Signal(object)
    datasetFitInclusionRequested = QtCore.Signal(object)
    def __init__(
        self,
        staging: DatasetImportStaging,
        *,
        origin_tab_id: str,
        source_resolver: Callable[[str], ImportedSource],
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
        if not isinstance(staging, DatasetImportStaging):
            raise TypeError("DataManagerPanel requires the application DatasetImportStaging owner.")
        if (
            type(origin_tab_id) is not str
            or not origin_tab_id
            or origin_tab_id != origin_tab_id.strip()
        ):
            raise ValueError(
                "DataManagerPanel requires one stable edge-normalized origin "
                "tab ID."
            )
        if not callable(source_resolver):
            raise TypeError(
                "DataManagerPanel requires a callable dataset source resolver."
            )
        self._staging = staging
        self._origin_tab_id = origin_tab_id
        self._source_resolver = source_resolver

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        layout.addWidget(QtWidgets.QLabel("<b>Loaded Datasets</b>"))

        # Dataset list
        self._dataset_list = QtWidgets.QListWidget()
        self._dataset_list.setUniformItemSizes(True)
        self._dataset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        layout.addWidget(self._dataset_list)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._load_btn = QtWidgets.QPushButton("Load Data")
        self._load_btn.setObjectName("loadDataButton")
        self._load_btn.clicked.connect(self._load_dataset)
        self._rename_btn = QtWidgets.QPushButton("Rename")
        self._rename_btn.clicked.connect(self._rename_dataset)
        self._remove_btn = QtWidgets.QPushButton("Delete")
        self._remove_btn.clicked.connect(self._remove_dataset)
        self._export_original_btn = QtWidgets.QPushButton("Export Original…")
        self._export_original_btn.clicked.connect(self._export_original)
        btn_layout.addWidget(self._load_btn)
        btn_layout.addWidget(self._rename_btn)
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addWidget(self._export_original_btn)
        layout.addLayout(btn_layout)

        mapping_group = QtWidgets.QGroupBox("Active-tab mapping")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        self._mapping_status = QtWidgets.QLabel("Select a dataset")
        self._mapping_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._mapping_status.setWordWrap(True)
        mapping_layout.addWidget(self._mapping_status)
        mapping_actions = QtWidgets.QHBoxLayout()
        self._create_set_btn = QtWidgets.QPushButton("Create Set From t=0")
        self._create_set_btn.clicked.connect(
            lambda: self._emit_mapping_action("create_set_from_t0")
        )
        self._map_existing_btn = QtWidgets.QPushButton("Map Existing")
        self._map_existing_btn.clicked.connect(self._map_existing)
        self._leave_unmapped_btn = QtWidgets.QPushButton("Leave Unmapped")
        self._leave_unmapped_btn.clicked.connect(
            lambda: self._emit_mapping_action("leave_unmapped")
        )
        mapping_actions.addWidget(self._create_set_btn)
        mapping_actions.addWidget(self._map_existing_btn)
        mapping_actions.addWidget(self._leave_unmapped_btn)
        mapping_layout.addLayout(mapping_actions)
        self._observation_mapping_btn = QtWidgets.QPushButton("Observation Mapping…")
        self._observation_mapping_btn.clicked.connect(self._request_observation_mapping)
        self._plot_overlay_btn = QtWidgets.QPushButton("Plot Overlay…")
        self._plot_overlay_btn.clicked.connect(self._request_plot_overlay)
        observation_actions = QtWidgets.QHBoxLayout()
        observation_actions.addWidget(self._observation_mapping_btn)
        observation_actions.addWidget(self._plot_overlay_btn)
        mapping_layout.addLayout(observation_actions)
        self._fit_include = QtWidgets.QCheckBox("Include mapped dataset in Fit")
        self._fit_include.toggled.connect(self._on_fit_include_toggled)
        mapping_layout.addWidget(self._fit_include)
        layout.addWidget(mapping_group)

        # Preview area
        layout.addWidget(QtWidgets.QLabel("Preview:"))
        self._preview_label = QtWidgets.QLabel("No dataset selected")
        self._preview_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._preview_label.setContentsMargins(10, 10, 10, 10)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)
        self._preview_label.hide()

        # Rendered global-catalog records for list/preview only.
        self._snapshot = DataManagerSnapshot((), None, (), ())
        self._visible_records_by_id: Dict[str, RawDatasetRecord] = {}

        self._progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._initiating_import_request: DatasetImportRequest | None = None
        self._staging.busyChanged.connect(self._on_staging_busy_changed)
        self._staging.workerProgressed.connect(self._on_import_progress)
        self._staging.workerFailed.connect(self._on_import_failure)
        self._staging.workerCanceled.connect(self._on_import_cancellation)
        self._set_mapping_controls_enabled(False)

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
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()
        self._mapping_status.setText("Select a dataset")
        self._set_mapping_controls_enabled(False)
        self._rename_btn.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._export_original_btn.setEnabled(False)
        self._plot_overlay_btn.setEnabled(False)
        blocker = QtCore.QSignalBlocker(self._fit_include)
        try:
            self._fit_include.setChecked(False)
        finally:
            del blocker

    def trigger_load_dialog(self) -> None:
        """Programmatically invoke the load dialog (mirrors Load button)."""
        self._load_dataset()

    def render_snapshot(self, snapshot: DataManagerSnapshot) -> None:
        """Render one global-catalog/active-tab snapshot without retaining truth."""
        if not isinstance(snapshot, DataManagerSnapshot):
            raise TypeError("DataManagerPanel requires a DataManagerSnapshot.")
        selected_id = self._selected_dataset_id()
        self._snapshot = snapshot
        self._visible_records_by_id = {
            str(record.dataset_id): record for record in snapshot.datasets
        }
        self._dataset_list.clear()
        self._preview_label.setText("No dataset selected")
        self._preview_label.hide()
        selected_row = -1
        for row, record in enumerate(snapshot.datasets):
            compact = compact_dataset_label(record.display_name, max_chars=DATASET_LIST_LABEL_MAX_CHARS)
            item = QtWidgets.QListWidgetItem(compact.display)
            item.setToolTip(
                f"<span>{html.escape(compact.full, quote=True)}</span>"
            )
            item.setData(Qt.UserRole, str(record.dataset_id))
            item.setData(Qt.UserRole + 1, compact.full)
            self._dataset_list.addItem(item)
            if str(record.dataset_id) == selected_id:
                selected_row = row
        if selected_row >= 0:
            self._dataset_list.setCurrentRow(selected_row)
        self._load_btn.setEnabled(not self._staging.busy)
        self._update_mapping_area()

    def _load_dataset(self):
        """Load dataset(s) using per-file import configuration and background workers."""
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Dataset(s)",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)",
        )

        if not filenames:
            return
        try:
            sources = tuple(
                capture_import_source(filepath)
                for filepath in filenames
            )
        except ValueError as exc:
            self._show_message(
                QtWidgets.QMessageBox.Icon.Critical,
                "Unsupported Dataset Source",
                str(exc),
            )
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
            if result.config.file_intent.apply_to_remaining:
                source_intent = result.config.remaining_file_template
                if source_intent is None:
                    raise RuntimeError(
                        "apply_to_remaining is True but remaining_file_template is None"
                    )
                source_sheet_names = result.config.file_intent.sheet_names
                for remaining_idx in range(index + 1, len(sources)):
                    remaining_source = sources[remaining_idx]
                    try:
                        remaining_config = self._build_remaining_file_config(
                            remaining_source,
                            source_intent,
                            source_sheet_names=source_sheet_names,
                        )
                    except (ValueError, OSError, UnicodeDecodeError) as exc:
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
        roster = tuple(
            DatasetImportWorkerAdmission(
                source=config.source,
                admitted_plans=tuple(config.plans),
            )
            for config in configs
        )

        try:
            import_request: DatasetImportRequest = self._staging.begin(
                origin_tab_id=self._origin_tab_id,
                roster=roster,
            )
        except DatasetImportAdmissionError as exc:
            self._show_message(
                QtWidgets.QMessageBox.Icon.Warning,
                "Dataset Import",
                str(exc),
            )
            return
        self._initiating_import_request = import_request
        dataset_label = f"{expected_count} dataset{'s' if expected_count != 1 else ''}"
        logger.info("User initiated dataset import: %s", dataset_label)

        self._progress_dialog = QtWidgets.QProgressDialog(
            f"Loading {dataset_label}...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Loading Datasets")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.canceled.connect(self._on_load_canceled)
        self._progress_dialog.show()

        try:
            self._staging.start_import(import_request)
        except DatasetImportAdmissionError as exc:
            self._show_message(
                QtWidgets.QMessageBox.Icon.Critical,
                "Dataset Import",
                str(exc),
            )

    def _collect_import_config(
        self,
        source: ImportSourceSnapshot,
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
        dialog.exec()
        return dialog.get_result()

    def _build_remaining_file_config(
        self,
        source: ImportSourceSnapshot,
        source_intent: SheetImportIntent,
        *,
        source_sheet_names: Tuple[str, ...] = (),
    ) -> ImportConfig:
        """Build an ImportConfig for a remaining file using the source intent.

        Raises ValueError if the source intent is incompatible with the file.
        """
        file_type = source.file_type

        per_sheet_intents: Dict[Optional[str], SheetImportIntent] = {}
        per_sheet_detections: Dict[Optional[str], UnitDetection] = {}
        per_sheet_columns: Dict[Optional[str], List[PhysicalColumn]] = {}
        target_sheet_names: Tuple[str, ...] = ()

        # Scope detection to source-selected columns only
        selected_keys = set(source_intent.species_columns)
        if source_intent.time_column:
            selected_keys.add(source_intent.time_column)

        if file_type == "csv":
            columns_tuple, rows = read_csv_rows_from_bytes(
                source.exact_source_bytes,
                source_label=source.original_filename,
            )
            try:
                first_row = next(iter(rows), None)
            finally:
                rows.close()
            columns = list(columns_tuple)
            if first_row is not None:
                det = detect_units_from_physical_row(
                    first_row.values, list(selected_keys)
                )
            else:
                det = UnitDetection.empty()
            per_sheet_intents[None] = rebuild_intent_for_target(
                source_intent, det,
            )
            per_sheet_detections[None] = det
            per_sheet_columns[None] = columns
        else:
            with ExcelWorkbookPair(
                source.exact_source_bytes,
                source_label=source.original_filename,
            ) as workbooks:
                sheets = workbooks.sheet_names
                if not sheets:
                    raise ValueError("Excel file has no sheets")
                # Filter sheets by source's checked set
                if source_sheet_names:
                    source_set = set(source_sheet_names)
                    sheets = [s for s in sheets if s in source_set]
                    if not sheets:
                        raise ValueError("No matching sheets found")
                for sheet_name in sheets:
                    rows = workbooks.read_sheet_rows(sheet_name)
                    try:
                        first_row = next(iter(rows), None)
                    finally:
                        rows.close()
                    if first_row is None:
                        per_sheet_columns[sheet_name] = []
                        per_sheet_detections[sheet_name] = UnitDetection.empty()
                    else:
                        per_sheet_columns[sheet_name] = list(first_row.keys())
                        per_sheet_detections[sheet_name] = detect_units_from_physical_row(
                            dict(first_row), list(selected_keys)
                        )
                    per_sheet_intents[sheet_name] = rebuild_intent_for_target(
                        source_intent, per_sheet_detections[sheet_name],
                    )
                target_sheet_names = tuple(sheets)

        plans = resolve_import_plans(
            source,
            per_sheet_intents, per_sheet_detections, per_sheet_columns,
        )

        file_intent = UserImportIntent(
            sheet_names=target_sheet_names,
            apply_to_remaining=False,
        )
        return ImportConfig(
            source=source,
            file_intent=file_intent,
            plans=tuple(plans),
        )

    def _on_import_progress(self, event: DatasetImportProgress) -> None:
        """Render one staging-owned aggregate progress event."""
        if not isinstance(event, DatasetImportProgress):
            raise TypeError("Dataset import progress event has the wrong type.")
        if event.request is not self._initiating_import_request:
            return
        if self._progress_dialog:
            self._progress_dialog.setValue(event.percent)

    def _on_import_failure(self, event: DatasetImportFailure) -> None:
        """Present one staging-owned typed worker failure."""
        if not isinstance(event, DatasetImportFailure):
            raise TypeError("Dataset import failure event has the wrong type.")
        if not self._owns_worker_request(event.request):
            return
        self._show_message(
            QtWidgets.QMessageBox.Icon.Critical,
            "Load Error",
            f"Failed to load dataset:\n\n{event.message}",
        )

    def _on_import_cancellation(
        self,
        event: DatasetImportCancellation,
    ) -> None:
        """Observe one staging-owned typed worker cancellation."""
        if not isinstance(event, DatasetImportCancellation):
            raise TypeError("Dataset import cancellation event has the wrong type.")
        if not self._owns_worker_request(event.request):
            return
        logger.info("Dataset import canceled: %s", event.source_label)

    def _on_load_canceled(self):
        """Handle cancel request during dataset loading."""
        logger.info("Dataset import cancellation requested by user")
        self._staging.cancel_active(outcome="user_canceled")

    def _finalize_progress_dialog(self) -> None:
        """Close and clear the progress dialog if it exists."""
        if not self._progress_dialog:
            return
        try:
            self._progress_dialog.canceled.disconnect(self._on_load_canceled)
        except (TypeError, RuntimeError):
            pass
        self._progress_dialog.close()
        self._progress_dialog = None

    def _remove_dataset(self):
        """Emit one global-delete intent; confirmation belongs to integration."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetDeleteRequested.emit(DatasetDeleteIntent(dataset_id))

    def _export_original(self) -> None:
        """Request the integration-owned destination dialog for exact bytes."""
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetExportOriginalRequested.emit(
            DatasetExportOriginalIntent(dataset_id)
        )

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
            self.datasetRenameRequested.emit(
                DatasetRenameIntent(dataset_id=dataset_id, display_name=str(value))
            )

    def _on_dataset_selected(self, current, previous):
        """Update preview when dataset is selected."""
        if not current:
            self._preview_label.setText("No dataset selected")
            self._preview_label.hide()
            self._update_mapping_area()
            return

        dataset_id = str(current.data(Qt.UserRole) or "")
        record = self._visible_records_by_id.get(dataset_id)
        if record is None:
            self._preview_label.hide()
            self._update_mapping_area()
            return

        def escape(value: object) -> str:
            return html.escape(str(value), quote=True)

        name = escape(record.display_name)
        self._preview_label.show()
        unique_times = {
            time_value
            for series in record.observations
            for time_value in series.t
        }
        observation_labels = tuple(
            escape(series.column_name) for series in record.observations
        )

        thawed_metadata = thaw_json_value(record.source_metadata)
        metadata = dict(thawed_metadata) if isinstance(thawed_metadata, dict) else {}
        mapping_source = escape(metadata.get("mapping_source", "normalized"))
        source = self._source_resolver(record.source_id)
        source_filename = escape(source.original_filename)
        source_format = escape(source.original_format)
        time_column = escape(metadata.get("time_column", "unknown"))
        preview = f"<b>{name}</b><br>"
        preview += (
            f"Original source: {source_filename} "
            f"({source_format})<br>"
        )
        preview += f"{escape(len(unique_times))} time points<br>"
        preview += f"Time column: {time_column} ({mapping_source})<br>"
        preview += (
            f"{escape(len(observation_labels))} species: "
            f"{', '.join(observation_labels[:5])}"
        )
        if len(observation_labels) > 5:
            preview += f", ... ({escape(len(observation_labels)-5)} more)"

        self._preview_label.setText(preview)
        self._update_mapping_area()

    def _selected_dataset_id(self) -> str:
        current = self._dataset_list.currentItem()
        return "" if current is None else str(current.data(Qt.UserRole) or "")

    def _active_link(self, dataset_id: str) -> ActiveDatasetLinkSnapshot | None:
        return self._snapshot.link_by_dataset_id().get(str(dataset_id))

    def _set_mapping_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._create_set_btn,
            self._map_existing_btn,
            self._leave_unmapped_btn,
            self._observation_mapping_btn,
            self._fit_include,
        ):
            widget.setEnabled(bool(enabled))

    def _update_mapping_area(self) -> None:
        dataset_id = self._selected_dataset_id()
        record = self._visible_records_by_id.get(dataset_id)
        enabled = record is not None and self._snapshot.active_tab_id is not None
        self._rename_btn.setEnabled(record is not None)
        self._remove_btn.setEnabled(record is not None)
        self._export_original_btn.setEnabled(record is not None)
        self._plot_overlay_btn.setEnabled(record is not None)
        self._set_mapping_controls_enabled(enabled)
        link = self._active_link(dataset_id)
        self._leave_unmapped_btn.setEnabled(
            bool(enabled and (link is None or link.set_id is None))
        )
        self._fit_include.blockSignals(True)
        try:
            self._fit_include.setChecked(bool(link and link.include_in_fit))
            self._fit_include.setEnabled(bool(enabled and link and link.set_id))
        finally:
            self._fit_include.blockSignals(False)
        if record is None:
            self._mapping_status.setText("Select a dataset")
        elif self._snapshot.active_tab_id is None:
            self._mapping_status.setText("No active mechanism tab; raw dataset remains available.")
        elif link is None or link.set_id is None:
            self._mapping_status.setText("Unmapped in the active mechanism tab")
        else:
            mapped_name = link.set_name or link.set_id
            mapped_columns = len(link.observation_mapping)
            self._mapping_status.setText(
                f"Mapped to {mapped_name}; {mapped_columns} confirmed observation mapping(s)"
            )

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
            DatasetMappingIntent(
                dataset_id=dataset_id,
                action=str(action),
                set_id=set_id,
                occupied_choice=occupied_choice,
            )
        )

    def _map_existing(self) -> None:
        if not self._snapshot.available_sets:
            self._show_message(
                QtWidgets.QMessageBox.Icon.Information,
                "Map Existing",
                "The active mechanism tab has no available initial-condition set.",
            )
            return
        labels = [name for _set_id, name in self._snapshot.available_sets]
        label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Map Existing",
            "Initial-condition set:",
            labels,
            editable=False,
        )
        if not accepted:
            return
        set_id = next(
            set_id
            for set_id, name in self._snapshot.available_sets
            if name == str(label)
        )
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
                f'Moving the set unmaps "{occupant_name}" and turns off '
                "its Fit inclusion."
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
        self.datasetObservationMappingRequested.emit(
            DatasetObservationMappingIntent(dataset_id=dataset_id)
        )

    def _request_plot_overlay(self) -> None:
        dataset_id = self._selected_dataset_id()
        if not dataset_id:
            return
        self.datasetPlotOverlayRequested.emit(
            DatasetPlotOverlayIntent(dataset_id=dataset_id)
        )

    def _on_fit_include_toggled(self, checked: bool) -> None:
        dataset_id = self._selected_dataset_id()
        link = self._active_link(dataset_id)
        if (
            not dataset_id
            or link is None
            or link.set_id is None
        ):
            return
        self.datasetFitInclusionRequested.emit(
            DatasetFitInclusionIntent(dataset_id=dataset_id, include=bool(checked))
        )

    def _on_staging_busy_changed(self, busy: bool) -> None:
        self._load_btn.setEnabled(not bool(busy))
        if not busy:
            self._finalize_progress_dialog()
            self._initiating_import_request = None

    def _owns_worker_request(
        self,
        request: DatasetImportWorkerRequest,
    ) -> bool:
        initiating = self._initiating_import_request
        return bool(
            initiating is not None
            and request.import_request is initiating
        )
